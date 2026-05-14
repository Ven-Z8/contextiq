"""Retrieval result models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from contextiq.ingestion.models import DocumentBlock


class RetrievalHit(BaseModel):
    """A ranked retrieval result with inspectable trace metadata."""

    block: DocumentBlock
    rank: int
    score: float
    stages: list[str] = Field(default_factory=list)
    reason: str
