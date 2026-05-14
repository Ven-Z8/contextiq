"""Application settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for ContextIQ."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="CONTEXTIQ_", extra="ignore")

    data_dir: Path = Path("data")
    qdrant_path: Path = Path("data/qdrant")
    default_model: str = "claude-sonnet-4-5"
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "CONTEXTIQ_ANTHROPIC_API_KEY"),
    )
    answer_max_tokens: int = 1_000


def get_settings() -> Settings:
    """Return application settings."""

    return Settings()
