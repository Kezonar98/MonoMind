from typing import Optional, Union
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.session import AsyncSessionLocal
from app.models.ledger import Transaction, User # <--- IMPORTED User MODEL
from app.agents.graph import app_workflow
from langchain_core.messages import HumanMessage

router = APIRouter()

# =============================================================================
# DEPENDENCIES
# =============================================================================
async def get_db():
    """Dependency to provide a database session for each request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# =============================================================================
# PYDANTIC SCHEMAS (Input data validation for API)
# =============================================================================
class ChatRequest(BaseModel):
    user_id: Union[str, int]
    message: Optional[str] = None
    text: Optional[str] = None
    user_input: Optional[str] = None
    user_text: Optional[str] = None
    image_base64: Optional[str] = None

    class Config:
        extra = "allow" 

class TransactionCreate(BaseModel):
    user_id: Union[str, int]
    amount: float
    currency: str = "USD"
    tx_type: str                        # "DEPOSIT" or "EXPENSE"
    description: str

# =============================================================================
# 1. AI CHAT ENDPOINT (Talking with MonoMind)
# =============================================================================
@router.post("/chat/")
async def chat_with_agent(request: ChatRequest):
    """Main endpoint to talk to the AI assistant."""
    try:
        actual_text = request.message or request.text or request.user_input or request.user_text or ""
        
        # Initialize memory (State) for LangGraph
        initial_state = {
            "messages": [HumanMessage(content=actual_text)],
            "user_id": str(request.user_id),       # Force string for PostgreSQL consistency
            "image_base64": request.image_base64   
        }
        
        # Compile and run our workflow
        graph = app_workflow.compile()
        result = await graph.ainvoke(initial_state)
        
        last_message = result["messages"][-1].content
        detected_intent = result.get("intent", "general_chat")
        
        return {
            "response": last_message,
            "intent": detected_intent
        }
        
    except Exception as e:
        print(f"[API ERROR] Chat generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 2. LEDGER ENDPOINTS (Financial Management)
# =============================================================================
@router.post("/ledger/")
async def add_transaction(tx: TransactionCreate, db: AsyncSession = Depends(get_db)):
    """Add a new transaction (Deposit or Expense) to the database. Auto-creates user if missing."""
    try:
        user_id_str = str(tx.user_id)
        
        # 1. Check if the user already exists in the 'users' table
        user = await db.scalar(select(User).where(User.id == user_id_str))
        
        # 2. If user does not exist, create them automatically
        if not user:
            # We generate a placeholder email since it is required by the User model
            placeholder_email = f"{user_id_str}@telegram.bot"
            new_user = User(
                id=user_id_str,
                email=placeholder_email
            )
            db.add(new_user)
            # Flush sends the insert to the DB so the Transaction can use the foreign key immediately
            await db.flush() 

        # 3. Create the transaction linked to the user
        new_tx = Transaction(
            user_id=user_id_str, 
            amount=tx.amount,
            currency=tx.currency,
            tx_type=tx.tx_type,  
            description=tx.description
        )
        db.add(new_tx)
        
        # 4. Commit all changes (User creation + Transaction creation) to the database
        await db.commit()
        await db.refresh(new_tx)
        
        return {
            "status": "success", 
            "message": f"Added {tx.tx_type} of {tx.amount} {tx.currency}",
            "transaction_id": getattr(new_tx, "id", "created")
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/ledger/{user_id}")
async def get_balance(user_id: str, db: AsyncSession = Depends(get_db)):
    """Get the current balance, total income/expenses, and transaction count for a specific user."""
    try:
        db_user_id = str(user_id) 
        result = await db.execute(select(Transaction).where(Transaction.user_id == db_user_id))
        transactions = result.scalars().all()
        
        balance = 0.0
        total_income = 0.0
        total_spent = 0.0
        
        for tx in transactions:
            amount = float(tx.amount)
            tx_type_str = tx.tx_type.name if hasattr(tx.tx_type, 'name') else str(tx.tx_type)
            
            if tx_type_str == "DEPOSIT":
                balance += amount
                total_income += amount
            else:
                balance -= amount
                total_spent += amount
                
        return {
            "user_id": user_id, 
            "balance": balance, 
            "total_income": total_income,
            "total_spent": total_spent,
            "transaction_count": len(transactions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")