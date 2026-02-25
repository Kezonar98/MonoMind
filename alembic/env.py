import asyncio
import os
import sys
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# -----------------------------------------------------------------------------
# Bank-Grade Path Resolution
# Ensure Alembic can locate the application's root directory for absolute imports.
# -----------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.core.config import settings
from app.models.ledger import Base

# This is the Alembic Config object, which provides access to the values within the .ini file.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# -----------------------------------------------------------------------------
# MetaData Configuration
# Link Alembic to the SQLAlchemy DeclarativeBase metadata for autogeneration.
# -----------------------------------------------------------------------------
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.
    
    This configures the context with just a URL and not an Engine.
    By skipping the Engine creation we don't even need a DBAPI to be available.
    Calls to context.execute() here emit the given string to the script output.
    """
    # Securely inject the database URL from Pydantic settings
    context.configure(
        url=settings.async_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Helper function to run migrations within a synchronous connection context."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Async Engine
    and associate a connection with the context.
    """
    # 1. Retrieve the base configuration dictionary from alembic.ini
    configuration = config.get_section(config.config_ini_section, {})
    
    # 2. Hard Override: Inject the secure, dynamically generated async database URL
    configuration["sqlalchemy.url"] = settings.async_database_url

    # 3. Create the engine using the overridden configuration dictionary
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.
    
    For asynchronous SQLAlchemy execution, this simply initializes the asyncio 
    event loop and delegates execution to the async migration handler.
    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()