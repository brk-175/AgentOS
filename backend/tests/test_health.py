"""Health endpoint tests: liveness always 200, readiness degrades to 503."""

import pytest
from httpx import ASGITransport, AsyncClient

from agentos.app import create_app


@pytest.mark.asyncio
async def test_live_returns_ok() -> None:
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_ready_ok_when_all_checks_pass() -> None:
    app = create_app()

    async def ok() -> bool:
        return True

    app.state.health_checks = {"postgres": ok, "redis": ok}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["checks"] == {"postgres": "ok", "redis": "ok"}


@pytest.mark.asyncio
async def test_ready_degrades_when_a_check_fails() -> None:
    app = create_app()

    async def ok_check() -> bool:
        return True

    async def broken() -> bool:
        return False

    app.state.health_checks = {"postgres": broken, "redis": ok_check}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["postgres"] == "fail"
    assert response.json()["checks"]["redis"] == "ok"


@pytest.mark.asyncio
async def test_ready_degrades_when_probe_raises() -> None:
    app = create_app()

    async def exploding() -> bool:
        raise RuntimeError("boom")

    app.state.health_checks = {"postgres": exploding}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["postgres"] == "fail"
