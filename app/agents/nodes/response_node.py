from app.agents.state import AgentState
from app.agents.llms import llm_chat

async def generate_final_response(state: AgentState) -> dict:
    """Node 5: The Chat Responder."""
    print("[NODE] Generating Final Response...")
    messages = state.get("messages", [])
    
    # Use the ORIGINAL user message so the LLM knows which language to reply in
    original_user_message = messages[-1].content if messages else ""
    transactions = state.get("extracted_transactions") or []
    balance = state.get("financial_result")
    if balance is None: balance = 0.0
        
    intent = state.get("intent", "general_chat")
    metrics = state.get("metrics") or {}
    burn_rate = metrics.get("burn_rate", 0.0)
    runway = metrics.get("runway_months", 0.0)

    chat_history = ""
    for m in messages[-5:-1]:
        role = "User" if m.type == "human" else "MonoMind"
        chat_history += f"{role}: {m.content}\n"
        
    if not chat_history:
        chat_history = "Start of conversation."
    
    system_facts = ""
    if intent == "evaluate_purchase":
        purchase = state.get("purchase_data", {})
        risk = state.get("purchase_risk", {})
        market = state.get("market_context", "No data")
        
        orig_price = purchase.get('original_price', purchase.get('item_price'))
        orig_currency = purchase.get('original_currency', 'USD')
        usd_price = purchase.get('item_price')
        
        verdict = "INSUFFICIENT FUNDS: The user does not have enough money to buy this item." if risk.get('is_risky') else "SUFFICIENT FUNDS: The user has enough money to buy this item."
        
        system_facts = f"""
        - Context: User wants to evaluate a purchase.
        - Product: {purchase.get('item_name')}
        - Price: {orig_price} {orig_currency} (Converted to ${usd_price:.2f} USD)
        - Current Balance: ${balance:.2f} USD
        - System Verdict: {verdict}
        - Market Context: {market}
        """
    elif intent == "analyze_runway":
        system_facts = f"""
        - Context: User is asking about their financial runway/burn rate.
        - Current Balance: ${balance:.2f} USD
        - Total Spent (Burn Rate): ${burn_rate:.2f} USD
        - Estimated Financial Runway: {runway:.1f} months
        """
    else:
        tx_text = "\n".join([f"- {tx['tx_type']} | ${tx['amount']} | {tx['description']}" for tx in transactions])
        if not tx_text: tx_text = "No transactions found."
        system_facts = f"""
        - Context: User is asking for their balance, history, or making general conversation.
        - Bot Identity: You are MonoMind, an AI financial assistant created to help this user track expenses, calculate runway, and evaluate purchases safely.
        - Current Balance: ${balance:.2f} USD
        - Recent Transactions:\n{tx_text}
        """

    system_prompt = "You are MonoMind, a helpful and precise financial AI assistant. Your job is to answer the user naturally based ONLY on the provided system facts."

    human_prompt = f"""Here is the context for this conversation:
    
    [VERIFIED SYSTEM FACTS]
    {system_facts}
    
    [RECENT CHAT HISTORY]
    {chat_history}
    
    The user just said: "{original_user_message}"
    
    Please reply directly to the user's message. Follow these strict rules:
    1. LANGUAGE RULE: You MUST reply in the EXACT SAME LANGUAGE as the user's message. If the user writes in Ukrainian, reply in Ukrainian. If Russian, reply in Russian.
    2. FACT-BASED: Answer using ONLY the facts above. If the user asks who you are, use the Bot Identity fact.
    3. NO QUESTIONS: DO NOT ask them how much money they have or what the price is. All data is in the facts!
    4. NO REFUSALS: Do not refuse to answer based on safety filters. Just report the facts friendly and concisely."""

    clean_messages = [
        ("system", system_prompt),
        ("human", human_prompt)
    ]
    
    response = await llm_chat.ainvoke(clean_messages)
    print("[SUCCESS] Final response generated (Adaptive Language Mode).")
    return {"messages": [response]}