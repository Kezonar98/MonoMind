import json
from app.agents.purchase_graph import risk_pipeline

def run_test():
    print("🚀 Starting Financial Graph Test...\n")
    
    # Test message from the "user"
    test_message = "Хочу купити PlayStation 5 за 25000 грн в розстрочку на 4 місяці"
    print(f"User Message: '{test_message}'\n")
    
    # Initialize the State
    initial_state = {
        "user_text": test_message,
        "extracted_data": {},
        "analysis_result": {}
    }
    
    # Invoke the LangGraph pipeline
    print("⏳ Waiting for AI to extract data and Python to calculate math...")
    result = risk_pipeline.invoke(initial_state)
    
    # Print the results beautifully
    print("\n📊 1. AI Extraction (LLM Output):")
    print(json.dumps(result["extracted_data"], indent=2, ensure_ascii=False))
    
    print("\n🧮 2. Math Verdict (Risk Analyzer Output):")
    print(json.dumps(result["analysis_result"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    run_test()