"""Typed application configuration sourced exclusively from the environment.

Every value lives in ``<repo>/.env`` (or real environment variables) — no
defaults live in code. If a variable is missing, the process fails fast on
startup instead of silently running with wrong values.

Secrets are only ever read here, never logged.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ENV_FILE = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings; every field is required from the environment."""

    model_config = SettingsConfigDict(
        env_file=_DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Core ---
    app_env: str = Field(init=False)
    log_level: str = Field(init=False)
    api_prefix: str = Field(init=False)

    # --- Database ---
    database_url: str = Field(init=False)

    # --- Redis (cache / queue / SSE bus) ---
    redis_url: str = Field(init=False)

    # --- Celery ---
    celery_broker_url: str = Field(init=False)
    celery_result_backend: str = Field(init=False)

    # --- GitHub OAuth App ---
    github_client_id: str = Field(init=False)
    github_client_secret: str = Field(init=False)
    github_oauth_callback_url: str = Field(init=False)

    # --- Secrets ---
    secret_key: str = Field(init=False)
    fernet_key: str = Field(init=False)

    # --- OpenRouter ---
    openrouter_api_key: str = Field(init=False)
    openrouter_base_url: str = Field(init=False)
    openrouter_model: str = Field(init=False)
    openrouter_embeddings_model: str = Field(init=False)
    embeddings_dimensions: int = Field(init=False)
    openrouter_judge_model: str = Field(init=False)

    # --- Rate limiting ---
    rate_limit_per_minute: int = Field(init=False)
    run_concurrency_per_user: int = Field(init=False)

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"prod", "production"}


@lru_cache
def get_settings() -> Settings:
    """Return the (cached) process-wide settings object."""
    return Settings()
