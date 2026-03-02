import os
import httpx # Required for making async API calls to Groq, Currency, and Jina
import re    # NEW: Required for extracting URLs from user text
from typing import Annotated, TypedDict, Sequence, Optional
import operator
import json

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.future import select

from app.db.session import AsyncSessionLocal 
from app.models import ledger 
from deep_translator import GoogleTranslator 
from langchain_community.tools import DuckDuckGoSearchRun 
from bs4 import BeautifulSoup

# =============================================================================
# IMPORTS FOR PURCHASE EVALUATION
# =============================================================================
from app.models.schemas import PurchaseExtraction
from app.services.risk_analyzer import RiskAnalyzer

# =============================================================================
# 1. State Definition (Agent Memory)
# =============================================================================
class AgentState(TypedDict):
    """
    The central memory structure of our LangGraph pipeline.
    This dictionary flows through every node, accumulating data.
    """
    # Core Chat Data
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_id: int
    translated_text: Optional[str]          
    intent: Optional[str]                   
    
    # Financial Data from Database & Math Engine
    extracted_transactions: list[dict]      
    financial_result: Optional[float]       
    metrics: Optional[dict]                 
    
    # Specific Fields for Purchase Evaluation Logic
    purchase_data: Optional[dict]           
    purchase_risk: Optional[dict]           
    market_context: Optional[str]   

    # Vision and URL Contexts
    image_base64: Optional[str]
    vision_context: Optional[str]
    url_context: Optional[str]      # NEW: Stores the scraped Markdown text from links

# =============================================================================
# 2. LLM Initialization (Local text routing & generation)
# =============================================================================
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Fast, deterministic LLM used strictly for routing logic (JSON output)
# INCREASED num_ctx to 2048 so it can read web page texts without truncating memory!
llm_router = ChatOllama(
    base_url=OLLAMA_URL,
    model="llama3.2:1b", 
    temperature=0.0, 
    format="json",
    num_predict=50,      
    num_ctx=2048          
)

# Chat LLM used for generating the final human-readable response
llm_chat = ChatOllama(
    base_url=OLLAMA_URL,
    model="llama3.2:1b", 
    temperature=0.0, 
    num_ctx=1024     
)

# Structured LLM forced to output data matching the PurchaseExtraction Pydantic schema
structured_llm = llm_router.with_structured_output(PurchaseExtraction)

# =============================================================================
# 3. Nodes (Agentic Steps)
# =============================================================================

async def translate_user_input(state: AgentState) -> dict:
    """
    Node 0: Detects language and translates to English if necessary.
    """
    print("[NODE] Translating User Input...")
    messages = state.get("messages", [])
    if not messages:
        return {"translated_text": ""}
        
    original_text = messages[-1].content
    translator = GoogleTranslator(source='auto', target='en')
    
    try:
        translated_text = translator.translate(original_text)
        if translated_text.strip().lower() != original_text.strip().lower():
            print(f"🌍 [Translator] '{original_text}' -> '{translated_text}'")
            return {"translated_text": translated_text}
        else:
            print(f"✅ [Translator] Text is already English. Bypassing.")
            return {"translated_text": original_text}
    except Exception as e:
        print(f"⚠️ [Translator Warning] API failed: {e}. Falling back to original text.")
        return {"translated_text": original_text}

async def extract_url_content(state: AgentState) -> dict:
    """Node 0.5: Smart URL Scraper without forced currency."""
    import json 
    
    print("[NODE] Checking for URLs in input...")
    messages = state.get("messages", [])
    last_user_message = messages[-1].content if messages else ""

    urls = re.findall(r'(https?://[^\s]+)', last_user_message)
    if not urls:
        return {"url_context": "No URL provided."}

    target_url = urls[0]
    print(f"🔗 [URL] Found link: {target_url}. Fetching metadata...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    has_price = False

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(target_url, headers=headers)
            response.raise_for_status()
            html = response.text

        soup = BeautifulSoup(html, 'html.parser')
        
        og_title = soup.find("meta", property="og:title")
        og_desc = soup.find("meta", property="og:description")
        
        title_text = og_title["content"] if og_title else (soup.title.string if soup.title else "")
        desc_text = og_desc["content"] if og_desc else ""
        
        price_amount = soup.find("meta", property="product:price:amount")
        price_currency = soup.find("meta", property="product:price:currency")
        price_text = ""
        
        if price_amount:
            # Більше ніяких дефолтних UAH! Беремо тільки те, що є на сайті.
            currency = price_currency["content"] if price_currency else ""
            price_text = f"EXACT PRICE: {price_amount['content']} {currency}"
            has_price = True

        if not has_price:
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(script.string)
                    if isinstance(data, list):
                        for item in data:
                            if "offers" in item and "price" in item["offers"]:
                                price_text = f"EXACT PRICE: {item['offers']['price']} {item['offers'].get('priceCurrency', '')}"
                                has_price = True
                                break
                    elif isinstance(data, dict):
                        if "offers" in data and "price" in data["offers"]:
                            price_text = f"EXACT PRICE: {data['offers']['price']} {data['offers'].get('priceCurrency', '')}"
                            has_price = True
                except:
                    continue

        extracted_info = f"Product Title: {title_text}\nDescription: {desc_text[:500]}\n{price_text}"
        
        if title_text and has_price:
            print(f"📄 https://www.success.com/ Found Title and Price in HTML:\n{extracted_info}")
            return {"url_context": extracted_info}
            
    except Exception as e:
        print(f"⚠️ https://metastatus.com/ HTML parsing failed: {e}")

    print("🔄 Price not found in HTML headers. Falling back to Jina Reader...")
    try:
        jina_url = f"https://r.jina.ai/{target_url}"
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(jina_url)
            response.raise_for_status()
            content = response.text

        clean_content = content[:2000]
        print(f"📄 https://www.success.com/ Scraped via Jina Reader.")
        return {"url_context": clean_content}
        
    except Exception as e:
        print(f"⚠️ https://www.merriam-webster.com/dictionary/error Jina Reader also failed: {e}")
        return {"url_context": "Failed to scrape URL."}
    

async def analyze_intent(state: AgentState) -> dict:
    """
    Node 1: The AI Router. 
    """
    print("[NODE] Analyzing Intent with Ollama...")
    messages = state.get("messages", [])
    last_user_message = state.get("translated_text") or (messages[-1].content if messages else "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strictly logical financial routing AI. 
        Analyze the user's input and determine their intent.
        You MUST respond ONLY with a valid JSON object containing a single key "intent".
        
        The value for "intent" MUST be exactly one of these FOUR strings:
        - "get_balance" (asking about current money, transactions, balance)
        - "analyze_runway" (asking how long money will last, burn rate)
        - "evaluate_purchase" (asking if they can afford an item, want to buy something, or mentioning a purchase)
        - "general_chat" (greetings, irrelevant topics)
        
        EXAMPLES:
        User: "Can I afford a PS5 for 600 dollars?"
        AI: {{"intent": "evaluate_purchase"}}
        
        User: "I want to buy a new car for 20000"
        AI: {{"intent": "evaluate_purchase"}}
        
        User: "How much money do I have?"
        AI: {{"intent": "get_balance"}}
        
        User: "Hello bot!"
        AI: {{"intent": "general_chat"}}
        """),
        ("human", "{user_input}")
    ])
    
    try:
        response = await (prompt | llm_router).ainvoke({"user_input": last_user_message})
        parsed_json = json.loads(response.content)
        intent = parsed_json.get("intent", "general_chat")
        
        if intent not in ["get_balance", "analyze_runway", "evaluate_purchase", "general_chat"]:
            intent = "general_chat"
            
    except Exception as e:
        print(f"[ERROR] LLM Routing failed: {e}")
        intent = "general_chat"
        
    print(f"[ROUTER DETECTED INTENT]: '{intent}'")
    return {"intent": intent}

async def fetch_ledger_data(state: AgentState) -> dict:
    """Node 2: The Database Fetcher."""
    print("[NODE] Fetching Ledger Data from PostgreSQL...")
    user_id = state.get("user_id")
    
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ledger.Transaction).where(ledger.Transaction.user_id == user_id)
        )
        transactions = result.scalars().all()
        
        extracted = [
            {
                "amount": float(tx.amount),
                "currency": tx.currency,
                "tx_type": tx.tx_type.name if hasattr(tx.tx_type, 'name') else str(tx.tx_type),
                "description": tx.description,
                "timestamp": tx.timestamp.isoformat()
            }
            for tx in transactions
        ]
        
    print(f"[LEDGER] Retrieved {len(extracted)} transactions for User {user_id}")
    return {"extracted_transactions": extracted}

def run_math_engine(state: AgentState) -> dict:
    """Node 3: The Deterministic Math Engine."""
    print("[NODE] Running Deterministic Math...")
    transactions = state.get("extracted_transactions", [])
    
    total_spent = 0.0
    total_income = 0.0
    balance = 0.0
    
    for tx in transactions:
        amount = float(tx["amount"])
        if tx["tx_type"] == "DEPOSIT":
            balance += amount
            total_income += amount
        else: 
            balance -= amount
            total_spent += amount
            
    runway_months = 0.0
    if total_spent > 0:
        runway_months = balance / total_spent
        
    print(f"[MATH] Balance: ${balance:.2f} | Spent: ${total_spent:.2f} | Income: ${total_income:.2f}")
    
    return {
        "financial_result": balance,
        "metrics": {
            "burn_rate": total_spent,
            "monthly_income": total_income,
            "runway_months": runway_months
        }
    }

async def analyze_image(state: AgentState) -> dict:
    """Node 3.5: Vision Processing via Groq API."""
    image_data = state.get("image_base64")
    
    if not image_data:
        return {"vision_context": "No image provided."}

    print("[NODE] Analyzing Image with Groq Vision AI...")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[VISION ERROR] GROQ_API_KEY is missing from environment variables!")
        return {"vision_context": "Failed to analyze image: API key missing."}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.2-11b-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Look at this screenshot. Find the exact product name, its numerical price, and the currency symbol (like ₴, грн, UAH). IMPORTANT: Translate your final answer to English so the next system can understand it. Reply STRICTLY with the product name, price, and currency."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0
    }

    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
        vision_result = data["choices"][0]["message"]["content"]
        print(f"[VISION] Extracted from image (Groq): {vision_result}")
        return {"vision_context": vision_result}
        
    except Exception as e:
        print(f"[VISION ERROR] Groq API failed: {e}")
        return {"vision_context": "Failed to analyze image."}


async def extract_purchase_info(state: AgentState) -> dict:
    """
    Node 4a: Data Extractor.
    Now looks at both [IMAGE DATA] and https://www.merriam-webster.com/dictionary/data to find the purchase info.
    """
    print("[NODE] Extracting Purchase Information...")
    messages = state.get("messages", [])
    
    history_text = "\n".join([f"{'User' if m.type == 'human' else 'AI'}: {m.content}" for m in messages[-4:]])
    
    vision_info = state.get("vision_context", "No image provided.")
    url_info = state.get("url_context", "No URL provided.") # NEW: Retrieve URL text
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a STRICT data extraction algorithm. 
        Your ONLY job is to extract the EXACT item name, the EXACT price, and the CURRENCY from the conversation, image data, AND web page text.
        
        [IMAGE DATA]: {vision_info}
        [WEB PAGE TEXT]: {url_info}
        
        CRITICAL RULES:
        1. If a URL was provided, look closely at the [WEB PAGE TEXT] to find the product name and price. 
        2. IGNORE real-world market prices in your extraction. Use ONLY the data provided.
        3. Convert currency to a float number (e.g. if the text says 4645 UAH, extract 4645.0).
        4. If no item can be found, use "unknown item".
        5. Extract the EXACT price AND the currency (e.g., "USD", "UAH", "EUR").
        
        CONVERSATION HISTORY:
        {{history}}
        """),
        ("user", "Extract the target item, newest price, and currency.")
    ])
    
    try:
        chain = prompt | structured_llm
        result: PurchaseExtraction = await chain.ainvoke({"history": history_text})
        print(f"[EXTRACTOR] Found item: {result.item_name} for {result.item_price} {result.currency}")
        return {"purchase_data": result.model_dump()}
    except Exception as e:
        print(f"[EXTRACTOR ERROR] {e}")
        return {"purchase_data": {"item_name": "unknown item", "item_price": 0.0, "currency": "USD", "is_credit": False, "credit_months": 1}}
    
async def convert_currency(state: AgentState) -> dict:
    """Node 4.1: Live Currency Converter."""
    print("[NODE] Converting Currency to Base (USD)...")
    purchase = state.get("purchase_data", {})
    
    price = purchase.get("item_price", 0.0)
    currency = purchase.get("currency", "USD").upper()
    
    if currency == "USD" or price == 0.0:
        print("[CURRENCY] Already in USD or price is 0. Bypassing.")
        return {"purchase_data": purchase}

    try:
        url = f"https://open.er-api.com/v6/latest/{currency}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
        usd_rate = data.get("rates", {}).get("USD")
        
        if usd_rate:
            converted_price = round(price * usd_rate, 2)
            print(f"[CURRENCY] Converted {price} {currency} -> ${converted_price} USD (Rate: {usd_rate})")
            
            purchase["original_price"] = price
            purchase["original_currency"] = currency
            purchase["item_price"] = converted_price
            purchase["currency"] = "USD"
        else:
            print("[CURRENCY ERROR] USD rate not found in API response.")
            
    except Exception as e:
        print(f"[CURRENCY ERROR] API failed: {e}. Falling back to original price without conversion.")

    return {"purchase_data": purchase}

def fetch_market_price(state: AgentState) -> dict:
    """Node 4.5: Web Search."""
    print("[NODE] Fetching Market Price from Web...")
    item_name = state.get("purchase_data", {}).get("item_name")
    
    if not item_name or item_name == "unknown item":
        return {"market_context": "No market data available."}

    try:
        search = DuckDuckGoSearchRun()
        query = f"average price of {item_name} USD 2026"
        results = search.invoke(query)
        
        market_info = results[:1000] if results else "No clear pricing found online."
        print(f"[SEARCH] Found context: {market_info[:100]}...")
        
        return {"market_context": market_info}
    except Exception as e:
        print(f"[SEARCH ERROR] {e}")
        return {"market_context": "Could not fetch market data due to search error."}
    
def analyze_purchase_risk(state: AgentState) -> dict:
    """Node 4b: Financial Risk Engine."""
    print("[NODE] Analyzing Purchase Risk...")
    data = state.get("purchase_data", {})
    metrics = state.get("metrics", {})
    real_balance = state.get("financial_result", 0.0)
    
    real_income = metrics.get("monthly_income", 0.0)
    real_expenses = metrics.get("burn_rate", 0.0)
    
    analyzer = RiskAnalyzer(
        monthly_income=real_income, 
        monthly_expenses=real_expenses, 
        current_balance=real_balance
    )
    
    verdict = analyzer.assess_purchase_risk(
        item_name=data.get("item_name", "item"),
        item_price=data.get("item_price", 0.0),
        is_credit=data.get("is_credit", False),
        credit_months=data.get("credit_months", 1)
    )
    
    print(f"[RISK ANALYZER] Verdict: {'Risky' if verdict['is_risky'] else 'Safe'} - {verdict['reason']}")
    return {"purchase_risk": verdict}


async def generate_final_response(state: AgentState) -> dict:
    """Node 5: The Chat Responder."""
    print("[NODE] Generating Final Response...")
    messages = state.get("messages", [])
    last_user_message = state.get("translated_text") or (messages[-1].content if messages else "")
    
    transactions = state.get("extracted_transactions") or []
    balance = state.get("financial_result")
    
    if balance is None:
        balance = 0.0
        
    intent = state.get("intent", "general_chat")
    metrics = state.get("metrics") or {}
    
    burn_rate = metrics.get("burn_rate", 0.0)
    runway = metrics.get("runway_months", 0.0)
    
    if intent == "evaluate_purchase":
        purchase = state.get("purchase_data", {})
        risk = state.get("purchase_risk", {})
        market = state.get("market_context", "No data")
        
        orig_price = purchase.get('original_price', purchase.get('item_price'))
        orig_currency = purchase.get('original_currency', 'USD')
        usd_price = purchase.get('item_price')
        
        system_prompt = f"""You are a strict data formatter. Your ONLY job is to output the exact template below, filling in the bracketed info using the [DATA]. 
        
        [DATA]
        Item: {purchase.get('item_name')}
        Requested Price: {orig_price} {orig_currency}
        Converted Price: ${usd_price:.2f} USD
        Current Balance: ${balance:.2f} USD
        Math Conclusion: {'Balance is lower than price' if risk.get('is_risky') else 'Balance is sufficient'}
        Market Info: {market}

        OUTPUT TEMPLATE (Copy this exactly and fill it in, do not add any other text):
        Product: [Insert Item here]
        Price: [Insert Requested Price here] (Converted to [Insert Converted Price])
        Current Balance: [Insert Current Balance]
        Conclusion: [Insert Math Conclusion here]. Therefore, you [can/cannot] afford this item.
        Market Comparison: [Summarize Market Info in 1 sentence, or say 'No data available']
        """
        
        clean_messages = [
            ("system", system_prompt),
            ("human", "Format the data into the template.")
        ]
        

        response = await llm_chat.ainvoke(clean_messages)
        print("[SUCCESS] Final response generated (Clean Mode).")
        return {"messages": [response]}
        
    elif intent == "analyze_runway":
        system_prompt = f"""You are MonoMind, an elite financial AI assistant. 
        Use ONLY these deterministic facts provided by the math engine:
        - Current Balance: ${balance:.2f}
        - Total Spent (Burn Rate): ${burn_rate:.2f}
        - Estimated Financial Runway: {runway:.1f} months
        
        Explain these metrics clearly. Warn if the runway is short. Do not invent numbers.
        CRITICAL: Always respond in English, regardless of the language the user speaks."""
    else:
        tx_text = "\n".join([f"- {tx['tx_type']} | ${tx['amount']} | {tx['description']}" for tx in transactions])
        if not tx_text:
            tx_text = "No transactions found."
            
        system_prompt = f"""You are MonoMind, an elite financial AI assistant. 
        Answer using ONLY the deterministic data below. Do NOT make up numbers.
        Current Calculated Balance: ${balance:.2f}
        Transaction History:
        {tx_text}
        
        CRITICAL: Always respond in English, regardless of the language the user speaks."""
    

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_input}")
    ])
    
    response = await (prompt | llm_chat).ainvoke({"user_input": last_user_message})
    print("[SUCCESS] Final response generated.")
    return {"messages": [response]}

# =============================================================================
# 4. Edges (Routing Logic)
# =============================================================================
def route_based_on_intent(state: AgentState) -> str:
    """Routes the graph based on the initial intent detected by Node 1."""
    if state.get("intent") in ["get_balance", "analyze_runway", "evaluate_purchase"]:
        return "fetch_ledger_data"
    return "generate_final_response"

def route_after_math(state: AgentState) -> str:
    """Routes the graph after core math processing is complete."""
    if state.get("intent") == "evaluate_purchase":
        return "analyze_image"
    return "generate_final_response"

# =============================================================================
# 5. Graph Compilation
# =============================================================================
workflow = StateGraph(AgentState)

# Register all nodes
workflow.add_node("translate_input", translate_user_input)
workflow.add_node("extract_url_content", extract_url_content) # NEW NODE
workflow.add_node("analyze_intent", analyze_intent)
workflow.add_node("fetch_ledger_data", fetch_ledger_data)
workflow.add_node("run_math_engine", run_math_engine)
workflow.add_node("analyze_image", analyze_image)
workflow.add_node("extract_purchase_info", extract_purchase_info)
workflow.add_node("convert_currency", convert_currency) 
workflow.add_node("fetch_market_price", fetch_market_price)
workflow.add_node("analyze_purchase_risk", analyze_purchase_risk)
workflow.add_node("generate_final_response", generate_final_response)

# Flow definitions
workflow.set_entry_point("translate_input")
workflow.add_edge("translate_input", "extract_url_content")   # UPDATED ROUTE
workflow.add_edge("extract_url_content", "analyze_intent")    # UPDATED ROUTE

# Conditional routing based on user intent
workflow.add_conditional_edges(
    "analyze_intent",
    route_based_on_intent,
    {
        "fetch_ledger_data": "fetch_ledger_data",
        "generate_final_response": "generate_final_response"
    }
)

workflow.add_edge("fetch_ledger_data", "run_math_engine")

# Conditional routing after database & math calculations
workflow.add_conditional_edges(
    "run_math_engine",
    route_after_math,
    {
        "analyze_image": "analyze_image",
        "generate_final_response": "generate_final_response"
    }
)

# Linear flow for purchase evaluation
workflow.add_edge("analyze_image", "extract_purchase_info")
workflow.add_edge("extract_purchase_info", "convert_currency") 
workflow.add_edge("convert_currency", "fetch_market_price") 
workflow.add_edge("fetch_market_price", "analyze_purchase_risk")
workflow.add_edge("analyze_purchase_risk", "generate_final_response") 
workflow.add_edge("generate_final_response", END)

# Export the uncompiled workflow. 
app_workflow = workflow