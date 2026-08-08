"""API router aggregation: every endpoint namespace mounts here."""

from fastapi import APIRouter

from agentos.api.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)
