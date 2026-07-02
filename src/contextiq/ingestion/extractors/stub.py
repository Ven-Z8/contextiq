"""Deterministic in-memory extractor for tests and swap proofs."""

from __future__ import annotations

from pathlib import Path

from contextiq.ingestion.models import DocumentBlock


class StubExtractor:
    """Returns a fixed block list; records the last requested page range."""

    name = "stub"

    def __init__(self, blocks: list[DocumentBlock]) -> None:
        self._blocks = blocks
        self.last_page_range: tuple[int, int] | None = None

    def extract(
        self, path: Path, *, page_range: tuple[int, int] | None = None
    ) -> list[DocumentBlock]:
        del path
        self.last_page_range = page_range
        return list(self._blocks)
