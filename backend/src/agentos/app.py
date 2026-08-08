"""FastAPI application factory.

``create_app`` wires settings, logging, lifespan-managed engine/redis
connections, the API router, and readiness probes. Kept dependency-light so
it can be unit-tested without infrastructure.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine

from agentos.api import api_router
from agentos.core.config import Settings, get_settings
from agentos.core.logging import get_logger
from agentos.db.session import build_engine, ping_database

logger = get_logger(__name__)


def _build_lifespan(
    settings: Settings,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine: AsyncEngine = build_engine(settings)
        redis: aioredis.Redis = aioredis.from_url(
            settings.redis_url,
            socket_connect_timeout=3,
            socket_timeout=3,
        )

        app.state.engine = engine
        app.state.redis = redis
        app.state.health_checks = {
            "postgres": lambda: _probe_db(engine),
            "redis": lambda: _probe_redis(redis),
        }
        logger.info("startup complete")
        try:
            yield
        finally:
            await redis.aclose()
            await engine.dispose()
            logger.info("shutdown complete")

    return lifespan


async def _probe_db(engine: AsyncEngine) -> bool:
    return await _wrap_probe(ping_database(engine), "postgres")


async def _probe_redis(redis: aioredis.Redis) -> bool:
    return await _wrap_probe(ping_redis(redis), "redis")


async def _wrap_probe(coro: Awaitable[bool], name: str) -> bool:
    try:
        return bool(await coro)
    except Exception:  # noqa: BLE001 - health probe must never raise
        logger.warning("dependency unavailable", extra={"extra_fields": {"dependency": name}})
        return False


async def ping_redis(redis: aioredis.Redis) -> bool:
    return await redis.ping()


def create_app() -> FastAPI:
    """Build and configure the AgentOS FastAPI application."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, force=True)

    app = FastAPI(
        title="AgentOS API",
        version="0.1.0",
        lifespan=_build_lifespan(settings),
    )
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
