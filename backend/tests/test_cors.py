"""CORS tests: the API must allow credentialed cross-origin reads from the frontend."""

import httpx

from agentos.core.config import get_settings


async def test_allows_frontend_origin_with_credentials(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/auth/me", headers={"origin": get_settings().frontend_url})
    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == get_settings().frontend_url
    assert response.headers["access-control-allow-credentials"] == "true"


async def test_rejects_disallowed_origin(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/v1/auth/me", headers={"origin": "http://evil.example.com"})
    assert response.status_code == 401
    assert "access-control-allow-origin" not in response.headers


async def test_preflight_from_frontend(client: httpx.AsyncClient) -> None:
    response = await client.options(
        "/api/v1/auth/me",
        headers={
            "origin": get_settings().frontend_url,
            "access-control-request-method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-methods"] is not None
    assert response.headers["access-control-allow-credentials"] == "true"
