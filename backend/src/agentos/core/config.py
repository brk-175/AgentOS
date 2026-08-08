"""Typed application configuration loaded from environment / .env.

All secrets and knobs are centralised here so the rest of the codebase
never touches ``os.environ`` directly.

Security note: settings are mergeable across environments (env vars override
the checked-in ``.env.example`` defaults) and secrets are only read, never
logged.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ENV_FILE = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings, sourced from environment then ``<repo>/.env``."""

    model_config = SettingsConfigDict(
        env_file=_DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Core ---
    app_env: str = "local"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    # --- Database ---
    database_url: str = "postgresql+asyncpg://agentos:change-me@localhost:5432/agentos"

    # --- Redis (cache / queue / SSE bus) ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Celery ---
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # --- GitHub OAuth App ---
    github_client_id: str = ""
    github_client_secret: str = ""
    github_oauth_callback_url: str = ""

    # --- Secrets ---
    secret_key: str = "change-me-session-secret"
    fernet_key: str = ""

    # --- OpenRouter ---
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "anthropic/claude-sonnet-4.5"
    openrouter_embeddings_model: str = "openai/text-embedding-3-small"
    embeddings_dimensions: int = 1536
    openrouter_judge_model: str = "anthropic/claude-sonnet-4.5"

    # --- Rate limiting ---
    rate_limit_per_minute: int = 120
    run_concurrency_per_user: int = 2

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"prod", "production"}


@lru_cache
def get_settings() -> Settings:
    """Return the (cached) process-wide settings object."""
    return Settings()
