# app/models/ledger.py
import enum
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import String, Numeric, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import AsyncAttrs

class Base(AsyncAttrs, DeclarativeBase):
    """Bank-Grade Declarative Base for all ledger models."""
    pass

class TransactionType(str, enum.Enum):
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    SUBSCRIPTION = "SUBSCRIPTION"

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # selectin loading is required for async relationship fetching
    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )

class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    
    # Financial data requires exact precision (14 digits total, 4 decimal places)
    amount: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    
    tx_type: Mapped[TransactionType] = mapped_column(SQLEnum(TransactionType), nullable=False)
    description: Mapped[str] = mapped_column(String(255), nullable=True)
    
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    user: Mapped["User"] = relationship(back_populates="transactions")