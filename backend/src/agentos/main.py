"""Entrypoint: ``python -m agentos.main`` or ``agentos-api`` console script."""

import uvicorn

from agentos.core.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "agentos.app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=settings.app_env == "local",
    )


if __name__ == "__main__":
    main()
