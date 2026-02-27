from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.config import settings

#Initialize Async Engine
engine = create_async_engine(
    settings.async_database_url,
    echo=False,             # Off for production; can be True for development
    pool_size=20,           # Based on expected load; adjust as needed
    max_overflow=10,        # Some extra connections for spikes
    pool_pre_ping=True,     # Check ping 
)

# Session factory for creating AsyncSession instances
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Dependency Injector for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Database session generator.
    Ensures safe connection closure after request completion,
    even if an error occurs during processing.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()