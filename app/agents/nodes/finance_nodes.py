from sqlalchemy.future import select
from app.db.session import AsyncSessionLocal
from app.models import ledger
from app.agents.state import AgentState

async def fetch_ledger_data(state: AgentState) -> dict:
    """
    Node 2: The Database Fetcher.
    Retrieves transactions from PostgreSQL. 
    Strictly uses String for user_id to match the DB schema and prevent overflows.
    """
    print("[NODE] Fetching Ledger Data from PostgreSQL...")
    
    # Ensure user_id is a string to match the VARCHAR column in PostgreSQL
    # This prevents the 'operator does not exist: character varying = integer' error
    user_id_raw = state.get("user_id")
    user_id_str = str(user_id_raw) if user_id_raw is not None else ""
    
    async with AsyncSessionLocal() as db:
        # SQLAlchemy will now correctly bind this as a VARCHAR parameter
        result = await db.execute(
            select(ledger.Transaction).where(ledger.Transaction.user_id == user_id_str)
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
        
    print(f"[LEDGER] Retrieved {len(extracted)} transactions for User {user_id_str}")
    return {"extracted_transactions": extracted}

def run_math_engine(state: AgentState) -> dict:
    """
    Node 3: The Deterministic Math Engine.
    Performs precise financial calculations in Python to avoid LLM hallucinations.
    """
    print("[NODE] Running Deterministic Math...")
    transactions = state.get("extracted_transactions", [])
    
    total_spent = 0.0
    total_income = 0.0
    balance = 0.0
    
    for tx in transactions:
        amount = float(tx["amount"])
        # Use tx_type string comparison for stability
        tx_type = tx["tx_type"]
        
        if tx_type == "DEPOSIT":
            balance += amount
            total_income += amount
        else: 
            balance -= amount
            total_spent += amount
            
    # Calculate runway (how many months the current balance will last based on current spending)
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