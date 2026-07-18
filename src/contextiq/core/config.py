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
    # Answer provider: "openrouter" (default, minimax-m3), "nvidia" (NIM),
    # or "anthropic" (Citations-API seam).
    llm_provider: str = "openrouter"
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "CONTEXTIQ_OPENROUTER_API_KEY"),
    )
    openrouter_model: str = "minimax/minimax-m3"
    # NVIDIA NIM API (Nemotron, etc.)
    nvidia_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("NVIDIA_API_KEY", "CONTEXTIQ_NVIDIA_API_KEY"),
    )
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    # Agentic retrieve: use a smaller/faster model for routing/decomposition/reranking
    nvidia_agentic_model: str = "meta/llama-3.1-70b-instruct"
    # Agentic retrieve (decompose + rerank). Falls back to plain hybrid without a client.
    agentic: bool = True
    answer_max_tokens: int = 4_000
    # Vector backend: "qdrant" (production, file-locked local) or "memory" (dev, no locks)
    vector_backend: str = "memory"


def get_settings() -> Settings:
    """Return application settings."""

    return Settings()
