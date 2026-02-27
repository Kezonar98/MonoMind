import os
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Import our previously created modules
from app.models.schemas import PurchaseExtraction
from app.services.risk_analyzer import RiskAnalyzer

# 1. Define the State of our graph
# This is the "backpack" that agents pass to each other
class RiskAgentState(TypedDict):
    user_text: str             # What the user typed
    extracted_data: dict       # JSON extracted by the LLM
    analysis_result: dict      # Verdict from the mathematical engine

# Configure our LLM (fetch URL from Docker environment)
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL")
llm = ChatOllama(
    base_url=OLLAMA_URL,
    model="llama3.2:1b",
    temperature=0.0 # Temperature 0, because we need precision, not creativity
)

# Connect the Pydantic schema to the LLM (Structured Output)
structured_llm = llm.with_structured_output(PurchaseExtraction)

# 2. Node 1: Extractor Agent (AI)
def extract_purchase_info(state: RiskAgentState) -> Dict[str, Any]:
    """Reads the user's text and converts it into structured JSON."""
    user_text = state["user_text"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a precise financial data extraction agent. Extract the item name, price, and credit details from the user's input. If price is missing, set it to 0.0."),
        ("user", "{input}")
    ])
    
    chain = prompt | structured_llm
    
    # Invoke the LLM
    result: PurchaseExtraction = chain.invoke({"input": user_text})
    
    # Return data to update the "backpack" (State)
    return {"extracted_data": result.model_dump()}

# 3. Node 2: Math Agent (Python)
def analyze_financial_risk(state: RiskAgentState) -> Dict[str, Any]:
    """Takes the JSON from the LLM and runs it through strict mathematics."""
    data = state["extracted_data"]
    
    # THE MAGIC: For now, we hardcode the user's finances. 
    # In the future, we will fetch them from PostgreSQL!
    mock_income = 30000.0
    mock_expenses = 15000.0
    mock_balance = 8000.0
    
    # Initialize our mathematical engine
    analyzer = RiskAnalyzer(
        monthly_income=mock_income, 
        monthly_expenses=mock_expenses, 
        current_balance=mock_balance
    )
    
    # Perform the calculation
    verdict = analyzer.assess_purchase_risk(
        item_name=data["item_name"],
        item_price=data["item_price"],
        is_credit=data["is_credit"],
        credit_months=data["credit_months"] or 1
    )
    
    return {"analysis_result": verdict}

# 4. Assemble the Graph
workflow = StateGraph(RiskAgentState)

# Add nodes
workflow.add_node("extractor", extract_purchase_info)
workflow.add_node("math_analyzer", analyze_financial_risk)

# Build the routing (Edges)
workflow.set_entry_point("extractor") # Always start with data extraction
workflow.add_edge("extractor", "math_analyzer") # After extraction, move to mathematics
workflow.add_edge("math_analyzer", END) # After mathematics, end the graph execution

# Compile our agent graph
risk_pipeline = workflow.compile()