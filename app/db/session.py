"""
Async database session factory and helpers.
"""

from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

# Create async engine with sensible pool sizing
engine = create_async_engine(
    settings.postgres_dsn,
    echo=False,
    pool_size=5,      # SQLAlchemy async engines ignore pool_size for some drivers, but kept for clarity
    max_overflow=10,
)

# Async session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_session() -> AsyncSession:
    """
    Dependency-injectable session generator for FastAPI.

    Usage with FastAPI:
        async def endpoint(db: AsyncSession = Depends(get_session)):
            ...
    """
    async with async_session() as session:
        yield session
