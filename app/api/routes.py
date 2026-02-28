# app/api/routes.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.agents.graph import app_workflow # CHANGED: Importing uncompiled workflow
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver # IMPORTING ASYNC SAVER
from langchain_core.messages import HumanMessage
from app.db.session import get_db
from app.models import ledger, schemas

router = APIRouter()

@router.post("/users/", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: schemas.UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user in the system."""
    
    # 1.Check if user with the same email already exists(dublicates)
    result = await db.execute(select(ledger.User).where(ledger.User.email == user.email))
    existing_user = result.scalars().first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists."
        )
    
    # 2.Create object for DB
    db_user = ledger.User(email=user.email)
    
    # 3. Write to DB
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user) # Get the generated ID and timestamps from DB after commit
    
    return db_user

@router.post("/transactions/", response_model=schemas.TransactionResponse, status_code=status.HTTP_201_CREATED)
async def create_transaction(transaction: schemas.TransactionCreate, db: AsyncSession = Depends(get_db)):
    """Add a new transaction for a user."""
    
    # 1. Validate user existence before creating transaction (foreign key constraint)
    result = await db.execute(select(ledger.User).where(ledger.User.id == transaction.user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found. Cannot attach transaction to a ghost account."
        )
    
    # 2. Create transaction object for DB with strict validation from Pydantic model
    db_transaction = ledger.Transaction(
        user_id=transaction.user_id,
        amount=transaction.amount,
        currency=transaction.currency,
        tx_type=transaction.tx_type,
        description=transaction.description
    )
    
    # 3. Write to DB
    db.add(db_transaction)
    await db.commit()
    await db.refresh(db_transaction)
    
    return db_transaction

@router.post("/chat/", response_model=schemas.ChatResponse)
async def chat_with_agent(request: schemas.ChatRequest):
    """
    The main cognitive endpoint.
    Accepts a text query, passes it through LangGraph (with memory), and returns a response.
    """
    try:
        # 1. Form start state for the graph
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "user_id": request.user_id,
            "intent": None,
            "extracted_transactions": [],
            "financial_result": None
        }
        
        # 2. Create config Thread ID
        config = {"configurable": {"thread_id": str(request.user_id)}}
        
        # 3. NEW: Run the graph with the initial state and dynamic ASYNC memory configuration
        # This prevents the Thread Lock 500 Error by compiling the graph asynchronously per request
        async with AsyncSqliteSaver.from_conn_string("monomind_memory.sqlite") as memory:
            app_graph = app_workflow.compile(checkpointer=memory)
            final_state = await app_graph.ainvoke(initial_state, config=config)
        
        # 4. Take final state and extract the intent and the last message for the response
        intent = final_state.get("intent", "unknown")
        
        # Last message from our model
        final_message = final_state["messages"][-1].content
        
        return schemas.ChatResponse(
            intent=intent, 
            response=final_message
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()

        # If database or AI engine fails, we want to catch it and return a user-friendly error message
        raise HTTPException(status_code=500, detail=f"AI Engine Error: {str(e)}")