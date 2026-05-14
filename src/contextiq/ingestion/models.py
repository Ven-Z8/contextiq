"""Ingestion data models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class BlockType(StrEnum):
    """Supported document block types."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    HEADING = "heading"


class DocumentBlock(BaseModel):
    """A citation-preserving document segment."""

    document_id: str
    block_id: str
    source_path: str
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)
    block_type: BlockType = BlockType.TEXT
    text: str
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
