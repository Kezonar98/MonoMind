from typing import Annotated, TypedDict, Sequence, Optional
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.future import select
from app.db.session import AsyncSessionLocal
from app.models import ledger

# -----------------------------------------------------------------------------
# 1. State Definition (Agent memory)
# That dict will be passed through the graph and updated at each node. It contains all the information about the conversation, user context, and intermediate results.
# -----------------------------------------------------------------------------
class AgentState(TypedDict):
    # History of messages (both user and agent) to maintain context across the conversation.
    # New data will be added to this list, not replaced, to preserve the full conversation history.
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Context of user 
    user_id: int
    
    # Analytics of data
    intent: Optional[str]                   # LLM-derived intent from user query
    extracted_transactions: list[dict]      # Row data from Ledger
    financial_result: Optional[float]       # Result of deterministic math 
    metrics: Optional[dict]                 # Dictionary for Analytics (burn rate, runway, etc. - for future expansion)

# -----------------------------------------------------------------------------
# 2. Nodes 
# For now these are just placeholders. The real logic will be implemented in the future.
# -----------------------------------------------------------------------------
# Initialization local model one time

llm_router = ChatOllama(
    model="llama3.2:1b", 
    temperature=0.0, 
    format="json",
    num_predict=50,      # Try to keep the response short since we only need to identify intent, not generate long text
    num_ctx=512          # Reduce context window for faster response since we only need to analyze intent(Some problems with memory)
)

llm_chat = ChatOllama(
    model="llama3.2:1b", 
    temperature=0.3, # Some creativity for generating responses, but not too high to go off track
    num_ctx=512     # Again, we don't need a huge context window for this application, and smaller models can be faster and more cost-effective for our use case.
)

def analyze_intent(state: AgentState) -> dict:
    """Node 1: Analyzes user query through LLM and determines intent."""
    print("[NODE] Analyzing Intent with Ollama...")
    
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "general_chat"}
        
    last_user_message = messages[-1].content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strictly logical financial routing AI. 
        Analyze the user's input and determine their intent.
        You MUST respond ONLY with a valid JSON object containing a single key "intent".
        The value for "intent" MUST be exactly one of these three strings:
        1. "get_balance" (if asking about current money, transactions, ledger, or balance, or spending)
        2. "analyze_runway" (if asking how long their money will last, burn rate, or survival prediction)
        3. "general_chat" (if anything else, greeting, or irrelevant)."""),
        ("human", "{user_input}")
    ])
    
    chain = prompt | llm_router
    
    try:
        response = chain.invoke({"user_input": last_user_message})
        parsed_json = json.loads(response.content)
        intent = parsed_json.get("intent", "general_chat")
        
        if intent not in ["get_balance", "analyze_runway", "general_chat"]:
            intent = "general_chat"
            
    except Exception as e:
        print(f"[ERROR] LLM Routing failed: {e}")
        intent = "general_chat"
        
    print(f"[ROUTER DETECTED INTENT]: '{intent}'")
    return {"intent": intent}

async def fetch_ledger_data(state: AgentState) -> dict:
    """Node 2: Fetches relevant financial data from PostgreSQL based on user_id and intent."""
    print("[NODE] Fetching Ledger Data from PostgreSQL...")
    user_id = state.get("user_id")
    
    # Use our Bank-Grade async database session to fetch transactions for the user. We will extract only the necessary fields to keep it lightweight for the agent's memory.
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ledger.Transaction).where(ledger.Transaction.user_id == user_id)
        )
        transactions = result.scalars().all()
        
        # Object should be converted to dict 
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
    """Node 3: Execute deterministic math on extracted transactions."""
    print("[NODE] Running Deterministic Math...")
    transactions = state.get("extracted_transactions", [])
    
    total_spent = 0.0
    balance = 0.0
    for tx in transactions:
        amount = float(tx["amount"])
        if tx["tx_type"] == "DEPOSIT":
            balance += amount
        else: # WITHDRAWAL or SUBSCRIPTION
            balance -= amount
            total_spent += amount
            
    # Calculate Runway. 
    # Total_spent - is the burn rate (Let it be average monthly expenses)
    runway_months = 0.0
    if total_spent > 0:
        runway_months = balance / total_spent
        
    print(f"[MATH] Balance: ${balance:.2f} | Burn Rate: ${total_spent:.2f} | Runway: {runway_months:.1f} months")
    
    return {
        "financial_result": balance,
        "metrics": {
            "burn_rate": total_spent,
            "runway_months": runway_months
        }
    }
            
def generate_final_response(state: AgentState) -> dict:
    """Node 4: LLM forming a final response to the user based on all the data and calculations."""
    print("[NODE] Generating Final Response...")
    
    # Take all the relevant data from the state to form a comprehensive system prompt for the LLM.
    messages = state.get("messages", [])
    last_user_message = messages[-1].content
    transactions = state.get("extracted_transactions", [])
    balance = state.get("financial_result", 0.0)
    
    # Take the intent
    intent = state.get("intent", "general_chat")
    metrics = state.get("metrics") or {}
    burn_rate = metrics.get("burn_rate", 0.0)
    runway = metrics.get("runway_months", 0.0)
    
    # Dynamically form the system prompt based on the identified intent and the data we have. 
    if intent == "analyze_runway":
        # Prompt for analyze of Runway
        system_prompt = f"""You are MonoMind, an elite financial AI assistant. 
        The user is asking about their financial runway or burn rate.
        Use ONLY these deterministic facts provided by the math engine:
        - Current Balance: ${balance:.2f}
        - Total Spent (Burn Rate): ${burn_rate:.2f}
        - Estimated Financial Runway: {runway:.1f} months
        
        Explain these metrics clearly and professionally. Warn them if the runway is very short. Do not invent any numbers.
        
        CRITICAL: Always respond in English, regardless of the language the user speaks."""
    else:
        # Prompt for general balance inquiry (get_balance)
        tx_text = "\n".join([
            f"- {tx['tx_type']} | ${tx['amount']} | {tx['description']}" 
            for tx in transactions
        ])
        if not tx_text:
            tx_text = "No transactions found."
            
        system_prompt = f"""You are MonoMind, an elite financial AI assistant. 
        Answer the user's question using ONLY the deterministic data provided below. 
        Do NOT make up any numbers. Be concise and professional.
        
        Current Calculated Balance: ${balance:.2f}
        Transaction History:
        {tx_text}
        
        CRITICAL: Always respond in English, regardless of the language the user speaks."""
    
    #Form the prompt and call the model to generate a response.
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{user_input}")
    ])
    
    # Call a model for talking
    response = (prompt | llm_chat).invoke({"user_input": last_user_message})
    print("[SUCCESS] Final response generated.")
    
    return {"messages": [response]}

# -----------------------------------------------------------------------------
# 3. Edges 
# -----------------------------------------------------------------------------
def route_based_on_intent(state: AgentState) -> str:
    """After analyzing the intent, we need to decide where to route the flow next."""
    if state.get("intent") in ["get_balance", "analyze_runway"]:
        return "fetch_ledger_data"
    return "generate_final_response"

# -----------------------------------------------------------------------------
# 4. Graph Compilation 
# -----------------------------------------------------------------------------
workflow = StateGraph(AgentState)

# Add some nodes to the graph
workflow.add_node("analyze_intent", analyze_intent)
workflow.add_node("fetch_ledger_data", fetch_ledger_data)
workflow.add_node("run_math_engine", run_math_engine)
workflow.add_node("generate_final_response", generate_final_response)

# Build the edges
workflow.set_entry_point("analyze_intent")

# After analyzing intent, route conditionally based on the identified intent
workflow.add_conditional_edges(
    "analyze_intent",
    route_based_on_intent,
    {
        "fetch_ledger_data": "fetch_ledger_data",
        "generate_final_response": "generate_final_response"
    }
)

# Linear flow for data processing after fetching ledger data
workflow.add_edge("fetch_ledger_data", "run_math_engine")
workflow.add_edge("run_math_engine", "generate_final_response")
workflow.add_edge("generate_final_response", END)

# Compilation Bank-Grade graph for execution
app_graph = workflow.compile()