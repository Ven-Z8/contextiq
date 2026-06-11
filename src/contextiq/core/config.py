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
    qdrant_collection: str = "contextiq_chunks_v2"
    jobs_db_path: Path = Path("data/jobs.db")
    ingest_page_batch_size: int = 50
    ingest_fast_page_threshold: int = 50
    default_model: str = "claude-sonnet-4-5"
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "CONTEXTIQ_ANTHROPIC_API_KEY"),
    )
    answer_max_tokens: int = 1_000
    summary_model: str = "claude-haiku-4-5"
    enable_vlm_extraction: bool = False
    enable_tree_build: bool = True
    enable_node_summaries: bool = False


def get_settings() -> Settings:
    """Return application settings."""

    return Settings()
