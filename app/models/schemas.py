# app/models/schemas.py
from pydantic import BaseModel, ConfigDict, Field, EmailStr
from datetime import datetime
from decimal import Decimal
from typing import Optional

from app.models.ledger import TransactionType

class TransactionBase(BaseModel):
    # Strict validation at the edge before it ever hits the DB
    amount: Decimal = Field(..., max_digits=14, decimal_places=4, description="Exact transaction value")
    currency: str = Field(default="USD", min_length=3, max_length=3)
    tx_type: TransactionType
    description: Optional[str] = Field(default=None, max_length=255)

class TransactionCreate(TransactionBase):
    user_id: int

class TransactionResponse(TransactionBase):
    id: int
    user_id: int
    timestamp: datetime

    # V2 requirement: Maps SQLAlchemy ORM instances to Pydantic responses
    model_config = ConfigDict(from_attributes=True)

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    pass

class UserResponse(UserBase):
    id: int
    created_at: datetime
    transactions: list[TransactionResponse] = []

    model_config = ConfigDict(from_attributes=True)

class ChatRequest(BaseModel):
    user_id: int
    message: str

class ChatResponse(BaseModel):
    intent: str
    response: str

    # --- AI Extractor Schemas ---

class PurchaseExtraction(BaseModel):
    """
    Structured data extracted from user input regarding a potential purchase.
    """
    item_name: str = Field(
        description="The name of the product or service the user wants to buy. E.g., 'iPhone 15', 'shoes', 'course'."
    )
    item_price: float = Field(
        description="The price of the item. Must be a number. If not found, return 0.0."
    )
    is_credit: bool = Field(
        default=False,
        description="True if the user mentions buying on credit, installments, or splitting the payment. Otherwise False."
    )
    credit_months: Optional[int] = Field(
        default=1,
        description="The number of months for the credit or installment. If not specified but is_credit is True, default to 1."
    )