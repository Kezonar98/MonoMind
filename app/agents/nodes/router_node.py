import json
from langchain_core.prompts import ChatPromptTemplate
from app.agents.state import AgentState
from app.agents.llms import llm_router

async def analyze_intent(state: AgentState) -> dict:
    """Node 1: The AI Router. Outputs strict JSON to direct the LangGraph flow."""
    print("[NODE] Analyzing Intent with Ollama...")
    messages = state.get("messages", [])
    last_user_message = state.get("translated_text") or (messages[-1].content if messages else "")
    
    history_text = "\n".join([f"{'User' if m.type == 'human' else 'MonoMind'}: {m.content}" for m in messages[-5:-1]])
    if not history_text:
        history_text = "No previous context."
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strictly logical financial routing AI. 
        Analyze the user's LATEST input in the context of the conversation and determine their intent.
        You MUST respond ONLY with a valid JSON object containing a single key "intent".
        
        The value for "intent" MUST be exactly one of these FOUR strings:
        - "get_balance" (asking about current money, transactions, balance)
        - "analyze_runway" (asking how long money will last, burn rate)
        - "evaluate_purchase" (asking if they can afford an item, want to buy ANY good/service like medicine, rent, or mentioning a purchase, OR asking to evaluate another item contextually like "What about this one?")
        - "general_chat" (greetings, follow-up questions like "Why?", irrelevant topics)
        
        [CONVERSATION CONTEXT]
        {history}
        """),
        ("human", "LATEST USER INPUT: {user_input}")
    ])
    
    try:
        response = await (prompt | llm_router).ainvoke({"history": history_text, "user_input": last_user_message})
        parsed_json = json.loads(response.content)
        intent = parsed_json.get("intent", "general_chat")
        
        if intent not in ["get_balance", "analyze_runway", "evaluate_purchase", "general_chat"]:
            intent = "general_chat"
            
    except Exception as e:
        print(f"[ERROR] LLM Routing failed: {e}")
        intent = "general_chat"
        
    print(f"[ROUTER DETECTED INTENT]: '{intent}'")
    return {"intent": intent}