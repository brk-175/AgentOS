"""Health-check endpoints (liveness + readiness).

Liveness (``/live``) reports the process is up. Readiness (``/ready``)
reports whether external dependencies (database, redis) answer; it returns
503 as soon as any dependency is failing so orchestrators can restart us.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def live() -> dict[str, str]:
    """Liveness probe: the process is running and serving requests."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    """Readiness probe: all external dependencies are reachable."""
    checks = getattr(request.app.state, "health_checks", None) or {}
    if not checks:
        return JSONResponse(status_code=503, content={"status": "degraded", "checks": {}})

    results: dict[str, str] = {}
    healthy = True
    for name, probe in checks.items():
        try:
            ok: bool = await probe()
        except Exception:  # noqa: BLE001 - a failing probe must not 500
            ok = False
        results[name] = "ok" if ok else "fail"
        healthy = healthy and ok

    status = "ok" if healthy else "degraded"
    code = 200 if healthy else 503
    return JSONResponse(status_code=code, content={"status": status, "checks": results})


HealthProbe = Callable[[], Awaitable[bool]]
