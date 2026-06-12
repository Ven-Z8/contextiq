"""Extractor protocol — the swappable document-reading boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from contextiq.ingestion.models import DocumentBlock


@runtime_checkable
class Extractor(Protocol):
    """Read a document into ordered, citation-preserving blocks."""

    name: str

    def extract(
        self, path: Path, *, page_range: tuple[int, int] | None = None
    ) -> list[DocumentBlock]:
        """Return ordered blocks for the document (optionally a page range)."""
        ...
