"""Async database engine + session factory.

Owns the single async SQLAlchemy engine and a session factory used by API
dependencies. Created/stored on ``app.state`` by the FastAPI lifespan.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from agentos.core.config import Settings
from agentos.core.logging import get_logger

logger = get_logger(__name__)


async def ping_database(engine: AsyncEngine) -> bool:
    """Return True if the database answers ``SELECT 1``."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 - health probe must never raise
        logger.warning("database unreachable")
        return False


def build_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        connect_args={"timeout": 5},
    )


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
