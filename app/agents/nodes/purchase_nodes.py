import os
import httpx
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from app.models.schemas import PurchaseExtraction
from app.services.risk_analyzer import RiskAnalyzer
from app.agents.state import AgentState
from app.agents.llms import structured_llm

async def analyze_image(state: AgentState) -> dict:
    """Node 3.5: Vision Processing via Groq API."""
    image_data = state.get("image_base64")
    
    if not image_data:
        return {"vision_context": "No image provided."}

    print("[NODE] Analyzing Image with Groq Vision AI...")
    
    if image_data.startswith("data:image"):
        final_image_url = image_data
    else:
        clean_b64 = image_data.replace("base64,", "").replace("data:image/jpeg;", "")
        final_image_url = f"data:image/jpeg;base64,{clean_b64}"

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
                        "text": "Look at this screenshot. Find the exact product name, its numerical price, and the currency symbol. IMPORTANT: Translate your final answer to English so the next system can understand it. Reply STRICTLY with the product name, price, and currency."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": final_image_url
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
    """Node 4a: Data Extractor."""
    print("[NODE] Extracting Purchase Information...")
    messages = state.get("messages", [])
    
    history_text = "\n".join([
        f"User: {m.content}" 
        for m in messages[-6:] 
        if m.type == 'human'
    ])
    
    vision_info = state.get("vision_context", "No image provided.")
    url_info = state.get("url_context", "No URL provided.") 
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a STRICT data extraction algorithm. 
        [IMAGE DATA]: {vision_info}
        [WEB PAGE TEXT]: {url_info}
        
        CRITICAL RULES:
        1. Extract the exact item name.
        2. Convert price to a float number (e.g., 12456.0).
        3. If no item can be found, use "unknown item".
        4. EXTRACT THE CORRECT CURRENCY! Pay close attention to the text. 
           If the text says "rubles" or "₽", extract "RUB". 
           If it says "UAH" or "грн", extract "UAH". 
           If it says "$", extract "USD".
        
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
    """Node 4.1: Live Currency Converter & Normalizer."""
    print("[NODE] Converting Currency to Base (USD)...")
    purchase = state.get("purchase_data", {})
    
    price = purchase.get("item_price", 0.0)
    raw_currency = purchase.get("currency", "USD")
    
    def normalize_currency(c: str) -> str:
        if not c: return "USD"
        c_lower = c.strip().lower()
        cmap = {
            "₽": "RUB", "руб": "RUB", "ruble": "RUB", "rubles": "RUB", "rub": "RUB",
            "₴": "UAH", "грн": "UAH", "uah": "UAH", "hryvnia": "UAH",
            "$": "USD", "usd": "USD", "dollar": "USD", "bucks": "USD",
            "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR",
            "£": "GBP", "gbp": "GBP", "pound": "GBP",
            "¥": "JPY", "jpy": "JPY", "yen": "JPY", "yuan": "CNY", "cny": "CNY",
            "zł": "PLN", "pln": "PLN", "zloty": "PLN",
            "₣": "CHF", "chf": "CHF", "franc": "CHF",
            "₹": "INR", "inr": "INR", "rupee": "INR",
            "₩": "KRW", "krw": "KRW", "won": "KRW",
            "₪": "ILS", "ils": "ILS", "shekel": "ILS",
            "₺": "TRY", "try": "TRY", "lira": "TRY"
        }
        return cmap.get(c_lower, c_lower).upper()
        
    iso_currency = normalize_currency(raw_currency)
    print(f"[CURRENCY NORMALIZER] '{raw_currency}' normalized to '{iso_currency}'")
    
    if iso_currency == "USD" or price == 0.0:
        print("[CURRENCY] Already in USD or price is 0. Bypassing API.")
        purchase["currency"] = "USD"
        return {"purchase_data": purchase}

    try:
        url = f"https://open.er-api.com/v6/latest/{iso_currency}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            
        usd_rate = data.get("rates", {}).get("USD")
        
        if usd_rate:
            converted_price = round(price * usd_rate, 2)
            print(f"[CURRENCY] Converted {price} {iso_currency} -> ${converted_price} USD (Rate: {usd_rate})")
            purchase["original_price"] = price
            purchase["original_currency"] = iso_currency
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
    item_name = state.get("purchase_data", {}).get("item_name", "").lower()
    generic_terms = ["unknown item", "medicine", "лекарство", "лекарие", "ліки", "товар", "item", "product", "thing"]
    
    if not item_name or any(term in item_name for term in generic_terms):
        print(f"[SEARCH] Item '{item_name}' is too generic. Skipping web search.")
        return {"market_context": "User did not specify an exact product name. No market comparison possible."}

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