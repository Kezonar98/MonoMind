# test_graph.py
import asyncio
from langchain_core.messages import HumanMessage
from app.agents.graph import app_graph

async def run_test():
    print("=== [TEST] Initialization financial agent ===")
    print("Connect to local Ollama (Llama 3)...\n")
    
    # 1.Form start state
    # Imitate that user with id=1 is asking about their financial data. The messages list will grow as the conversation progresses, preserving the full history.
    test_message = "How much did I spend on food last month?"
    print(f"[USER INPUT]: '{test_message}'\n")
    
    initial_state = {
        "messages": [HumanMessage(content=test_message)],
        "user_id": 1,
        "intent": None,
        "extracted_transactions": [],
        "financial_result": None
    }
    
    # 2. Start Graph (LangGraph >= 1.0 use ainvoke for async execution)
    try:
        final_state = await app_graph.ainvoke(initial_state)
        
        # 3. Analyze result
        print("\n=== [TEST] Result of Routing ===")
        print(f"The final Result (Intent):  {final_state.get('intent')}")
       
        final_message = final_state['messages'][-1].content
        print(f"\n[MONOMIND RESPONSE]:\n{final_message}")

        if final_state.get('intent') == 'get_balance':
            print("[SUCCESS] Model succesfully identified the intent!")
        else:
            print("[WARNING] Model failed to identify the financial intent.")
            
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Graph failed during execution: {e}")

if __name__ == "__main__":
    # Start async event loop
    asyncio.run(run_test())