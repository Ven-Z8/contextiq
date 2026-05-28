"""Resolve parent sections for child retrieval hits."""

from __future__ import annotations

from collections.abc import Callable

from contextiq.ingestion.hierarchy import ParentChunk
from contextiq.ingestion.models import BlockType, DocumentBlock


class ParentResolver:
    """Fetch parent section text for child chunk hits."""

    def __init__(
        self,
        parent_loader: Callable[[str], ParentChunk | None],
        *,
        max_parent_words: int = 1_500,
    ) -> None:
        self.parent_loader = parent_loader
        self.max_parent_words = max_parent_words

    def resolve(self, block: DocumentBlock) -> DocumentBlock | None:
        """Return a parent block when the hit is a child chunk."""

        if block.metadata.get("chunk_level") == "parent":
            return None

        parent_id = block.metadata.get("parent_id")
        if not isinstance(parent_id, str) or not parent_id:
            return None

        parent = self.parent_loader(parent_id)
        if parent is None:
            return None

        text = self._trim_parent_text(parent.text)
        return DocumentBlock(
            document_id=parent.document_id,
            block_id=parent.parent_id,
            source_path=block.source_path,
            page=parent.page_start,
            section_path=parent.section_path,
            block_type=BlockType.TEXT,
            text=text,
            metadata={
                "chunk_level": "parent",
                "section_id": parent.section_id,
                "parent_id": parent.parent_id,
                "resolved_from": block.block_id,
            },
        )

    def enrich_candidates(self, candidates: list[DocumentBlock]) -> list[DocumentBlock]:
        """Add parent context blocks without replacing child hits."""

        enriched: list[DocumentBlock] = []
        seen: set[str] = set()
        for block in candidates:
            if block.block_id not in seen:
                enriched.append(block)
                seen.add(block.block_id)
            parent = self.resolve(block)
            if parent is not None and parent.block_id not in seen:
                enriched.append(parent)
                seen.add(parent.block_id)
        return enriched

    def _trim_parent_text(self, text: str) -> str:
        words = text.split()
        if len(words) <= self.max_parent_words:
            return text
        return " ".join(words[: self.max_parent_words])
