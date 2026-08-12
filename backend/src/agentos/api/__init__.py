"""API router aggregation: every endpoint namespace mounts here."""

from fastapi import APIRouter

from agentos.api.auth import router as auth_router
from agentos.api.health import router as health_router
from agentos.api.repos import router as repos_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(health_router)
api_router.include_router(repos_router)
