import os
from typing import Annotated, TypedDict, Sequence, Optional
import operator
import json

from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.future import select

from app.db.session import AsyncSessionLocal # Importing the async session factory to fetch data from PostgreSQL
from app.models import ledger # Importing ledger to access Transaction and User models for DB operations
from deep_translator import GoogleTranslator # NEW: For language detection and translation 
from langchain_community.tools import DuckDuckGoSearchRun # NEW: For real-time price checking in purchase evaluation logic
# =============================================================================
# NEW IMPORTS FOR PURCHASE EVALUATION
# =============================================================================
from app.models.schemas import PurchaseExtraction
from app.services.risk_analyzer import RiskAnalyzer

# =============================================================================
# 1. State Definition (Agent memory)
# =============================================================================
class AgentState(TypedDict):
    """
    The central memory structure of our LangGraph pipeline.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_id: int
    translated_text: Optional[str]          
    intent: Optional[str]                   
    extracted_transactions: list[dict]      
    financial_result: Optional[float]       
    metrics: Optional[dict]                 
    
    # NEW FIELDS: For Purchase Evaluation logic
    purchase_data: Optional[dict]           # Extracted JSON from user's text (item_name, item_price)
    purchase_risk: Optional[dict]           # Deterministic math output (is_risky, reason, details)
    market_context: Optional[str]           # Real-time market data for the item 
# =============================================================================
# 2. LLM Initialization
# =============================================================================
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Fast, deterministic router
llm_router = ChatOllama(
    base_url=OLLAMA_URL,
    model="llama3.2:1b", 
    temperature=0.0, 
    format="json",
    num_predict=50,      
    num_ctx=512          
)

# More creative LLM for final response generation
llm_chat = ChatOllama(
    base_url=OLLAMA_URL,
    model="llama3.2:1b", 
    temperature=0.3, 
    num_ctx=512     
)

# Structured LLM specifically forced to output PurchaseExtraction schema
structured_llm = llm_router.with_structured_output(PurchaseExtraction)

# =============================================================================
# 3. Nodes (Agentic Steps)
# =============================================================================

async def translate_user_input(state: AgentState) -> dict:
    """Node 0: Detects language and translates to English if necessary."""
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

async def analyze_intent(state: AgentState) -> dict:
    """
    Node 1: The AI Router. 
    Now uses few-shot prompting for better accuracy and prevents LLM hallucinations.
    """
    print("[NODE] Analyzing Intent with Ollama...")
    messages = state.get("messages", [])
    last_user_message = state.get("translated_text") or (messages[-1].content if messages else "")
    
    # UPDATED PROMPT: Added Few-Shot Examples to force the 1B model to behave.
    # Double curly braces {{ }} are used to escape JSON brackets in LangChain format strings.
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
        
        # Validation
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
    """
    Node 3: The Deterministic Math Engine.
    UPDATED: Calculates real total_income dynamically so we don't use hardcoded mock data in the Risk Analyzer.
    """
    print("[NODE] Running Deterministic Math...")
    transactions = state.get("extracted_transactions", [])
    
    total_spent = 0.0
    total_income = 0.0
    balance = 0.0
    
    for tx in transactions:
        amount = float(tx["amount"])
        if tx["tx_type"] == "DEPOSIT":
            balance += amount
            total_income += amount  # Calculate real historical income
        else: # WITHDRAWAL or SUBSCRIPTION
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

# =============================================================================
# NEW NODES FOR PURCHASE EVALUATION
# =============================================================================
async def extract_purchase_info(state: AgentState) -> dict:
    """Node 4a: Extracts structured product details from the user's text."""
    print("[NODE] Extracting Purchase Information...")
    messages = state.get("messages", [])
    text_to_analyze = state.get("translated_text") or (messages[-1].content if messages else "")
    
    # UPDATED PROMPT: Strict instructions to stop hallucinating real-world prices
    # Added slang understanding ("bucks", "grand", "k")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a STRICT data extraction algorithm. 
        Your ONLY job is to extract the EXACT item name and the EXACT price mentioned in the user's text.
        
        CRITICAL RULES:
        1. IGNORE real-world market prices. If the user says a PS5 costs 1 million, you extract 1000000.0.
        2. Words like 'bucks', 'quid', 'grand', 'k' refer to money.
        3. If no price is mentioned, set item_price to 0.0.
        
        EXAMPLES:
        User: "Can I buy a PS5 for 600 bucks?"
        Expected Output: item_name="PS5", item_price=600.0
        
        User: "I want to get a used car for 25k"
        Expected Output: item_name="used car", item_price=25000.0
        """),
        ("user", "{input}")
    ])
    
    try:
        chain = prompt | structured_llm
        result: PurchaseExtraction = await chain.ainvoke({"input": text_to_analyze})
        print(f"[EXTRACTOR] Found item: {result.item_name} for {result.item_price}")
        return {"purchase_data": result.model_dump()}
    except Exception as e:
        print(f"[EXTRACTOR ERROR] {e}")
        # Fallback to prevent crash
        return {"purchase_data": {"item_name": "unknown item", "item_price": 0.0, "is_credit": False, "credit_months": 1}}

def fetch_market_price(state: AgentState) -> dict:
    """Node 4.5: Searches the web for the current average price of the item."""
    print("[NODE] Fetching Market Price from Web...")
    item_name = state.get("purchase_data", {}).get("item_name")
    
    if not item_name or item_name == "unknown item":
        return {"market_context": "No market data available."}

    try:
        search = DuckDuckGoSearchRun()
        # Formulate a search queary to find the average price of the item.
        query = f"average price of {item_name} USD 2026"
        results = search.invoke(query)
        
        # Cut results
        market_info = results[:1000] if results else "No clear pricing found online."
        print(f"[SEARCH] Found context: {market_info[:100]}...")
        
        return {"market_context": market_info}
    except Exception as e:
        print(f"[SEARCH ERROR] {e}")
        return {"market_context": "Could not fetch market data due to search error."}
    
    
def analyze_purchase_risk(state: AgentState) -> dict:
    """
    Node 4b: Evaluates financial safety dynamically based on REAL database metrics,
    strictly eliminating hardcoded mock values.
    """
    print("[NODE] Analyzing Purchase Risk...")
    data = state.get("purchase_data", {})
    metrics = state.get("metrics", {})
    real_balance = state.get("financial_result", 0.0)
    
    # We use actual data from DB instead of fake numbers!
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

# =============================================================================

async def generate_final_response(state: AgentState) -> dict:
    """Node 5: The Chat Responder."""
    print("[NODE] Generating Final Response...")
    messages = state.get("messages", [])
    last_user_message = state.get("translated_text") or (messages[-1].content if messages else "")
    transactions = state.get("extracted_transactions", [])
    balance = state.get("financial_result", 0.0)
    intent = state.get("intent", "general_chat")
    metrics = state.get("metrics") or {}
    
    burn_rate = metrics.get("burn_rate", 0.0)
    runway = metrics.get("runway_months", 0.0)
    
    # UPDATED: Add prompt template for evaluate_purchase
    if intent == "evaluate_purchase":
        purchase = state.get("purchase_data", {})
        risk = state.get("purchase_risk", {})
        market = state.get("market_context", "No data")
        
        system_prompt = f"""You are a strict data reporter. Your ONLY job is to relay the exact numbers provided below. 
        DO NOT invent market prices. DO NOT correct the user's price. DO NOT give financial advice.
        
        [CALCULATOR DATA]
        Requested Item: {purchase.get('item_name')}
        Requested Price: ${purchase.get('item_price')}
        Current Balance: ${balance:.2f}
        Verdict: {'Insufficient Funds' if risk.get('is_risky') else 'Sufficient Funds'}
        Math Details: {risk.get('details')}
        
        [MARKET RESEARCH FROM WEB]
        {market}

       Write a professional response. 
        1. State if the purchase is approved or denied based on the Calculator Verdict and user's balance.
        2. Compare the user's Target Price with the average prices found in the Market Research. Advise if they are overpaying or if it's a good deal.
        Do not invent numbers. Base your market analysis ONLY on the [MARKET RESEARCH FROM WEB] block. Always respond in English."""
        
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
    """Route from Intent Analyzer to the correct next step."""
    # evaluate_purchase ALSO needs database data to know the real balance!
    if state.get("intent") in ["get_balance", "analyze_runway", "evaluate_purchase"]:
        return "fetch_ledger_data"
    return "generate_final_response"

def route_after_math(state: AgentState) -> str:
    """After getting DB data and doing math, decide if we need to extract purchase info."""
    if state.get("intent") == "evaluate_purchase":
        return "extract_purchase_info"
    return "generate_final_response"

# =============================================================================
# 5. Graph Compilation
# =============================================================================
workflow = StateGraph(AgentState)

# Register all nodes
workflow.add_node("translate_input", translate_user_input)
workflow.add_node("analyze_intent", analyze_intent)
workflow.add_node("fetch_ledger_data", fetch_ledger_data)
workflow.add_node("run_math_engine", run_math_engine)
workflow.add_node("extract_purchase_info", extract_purchase_info)     # NEW
workflow.add_node("analyze_purchase_risk", analyze_purchase_risk)     # NEW
workflow.add_node("generate_final_response", generate_final_response)
workflow.add_node("fetch_market_price", fetch_market_price)

# Define the flow path
workflow.set_entry_point("translate_input")
workflow.add_edge("translate_input", "analyze_intent")

# Dynamic branching based on intent
workflow.add_conditional_edges(
    "analyze_intent",
    route_based_on_intent,
    {
        "fetch_ledger_data": "fetch_ledger_data",
        "generate_final_response": "generate_final_response"
    }
)

# Fetch DB -> Run Math
workflow.add_edge("fetch_ledger_data", "run_math_engine")

# After Math, branch again: either go to final response, OR evaluate purchase
workflow.add_conditional_edges(
    "run_math_engine",
    route_after_math,
    {
        "extract_purchase_info": "extract_purchase_info",
        "generate_final_response": "generate_final_response"
    }
)

# Linear flow for purchase evaluation
workflow.add_edge("extract_purchase_info", "fetch_market_price")
workflow.add_edge("fetch_market_price", "analyze_purchase_risk")
workflow.add_edge("analyze_purchase_risk", "generate_final_response") 

#End our graph
workflow.add_edge("generate_final_response", END)

# Compile into an executable application
app_graph = workflow.compile()