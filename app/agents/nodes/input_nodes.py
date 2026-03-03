import re
import httpx
import json
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
from app.agents.state import AgentState

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

async def extract_url_content(state: AgentState) -> dict:
    """Node 0.5: Smart URL Scraper."""
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