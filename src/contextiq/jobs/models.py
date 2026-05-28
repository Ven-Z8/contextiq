"""Ingest job models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class IngestJobStatus(StrEnum):
    """Lifecycle states for background ingest jobs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestJob(BaseModel):
    """Persisted ingest job record."""

    job_id: str
    source_path: str
    status: IngestJobStatus = IngestJobStatus.PENDING
    profile_name: str = "fast"
    build_index: bool = True
    pages_total: int = 0
    pages_done: int = 0
    blocks_saved: int = 0
    blocks_indexed: int = 0
    document_id: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class IngestJobCreate(BaseModel):
    """Request payload for starting an ingest job."""

    source_path: str
    profile_name: str | None = None
    build_index: bool = True
