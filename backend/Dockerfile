# ---------- build stage ----------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN pip install --no-cache-dir poetry==2.3.4

# backend deps first (cached unless the lock changes)
WORKDIR /src/backend
COPY backend/pyproject.toml backend/poetry.lock ./
RUN poetry install --only main --no-root

# backend package + alembic (wheel, so site-packages stays self-contained)
COPY backend/src ./src
COPY backend/alembic ./alembic
COPY backend/alembic.ini ./
RUN poetry build -f wheel && pip install --no-cache-dir dist/*.whl

# GitHub MCP server (spawned by the worker over stdio)
WORKDIR /src/mcp-servers
COPY mcp-servers/pyproject.toml ./
COPY mcp-servers/src ./src
RUN poetry build -f wheel && pip install --no-cache-dir dist/*.whl

# ---------- runtime stage ----------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

USER appuser
WORKDIR /home/appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=5 \
    CMD celery -A agentos.tasks.celery_app inspect ping --timeout 5 || exit 1

CMD ["celery", "-A", "agentos.tasks.celery_app", "worker", "--loglevel=INFO", "--concurrency=2"]
