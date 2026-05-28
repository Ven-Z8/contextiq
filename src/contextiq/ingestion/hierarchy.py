"""Parent-child hierarchy for section-scoped retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

from contextiq.ingestion.models import DocumentBlock

_PARENT_SECTION_DEPTH = 2
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ParentChunk:
    """Section-level parent chunk stored outside the vector index."""

    parent_id: str
    document_id: str
    section_id: str
    section_path: list[str]
    text: str
    page_start: int | None
    page_end: int | None
    block_ids: tuple[str, ...]


class HierarchyBuilder:
    """Build parent sections and tag child blocks for retrieval."""

    def __init__(self, section_depth: int = _PARENT_SECTION_DEPTH) -> None:
        self.section_depth = section_depth

    def build_parents(self, blocks: list[DocumentBlock]) -> list[ParentChunk]:
        """Group blocks into section-level parent chunks."""

        if not blocks:
            return []

        parents: list[ParentChunk] = []
        current_key: tuple[str, ...] | None = None
        current_blocks: list[DocumentBlock] = []
        document_id = blocks[0].document_id

        def flush() -> None:
            nonlocal current_blocks, current_key
            if not current_blocks:
                return
            section_path = self._section_path_for_key(current_key or ("Document",))
            section_id = self.section_id(document_id, section_path)
            pages = [block.page for block in current_blocks if block.page is not None]
            parents.append(
                ParentChunk(
                    parent_id=section_id,
                    document_id=document_id,
                    section_id=section_id,
                    section_path=section_path,
                    text="\n\n".join(
                        block.text.strip()
                        for block in current_blocks
                        if block.text.strip()
                    ),
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    block_ids=tuple(block.block_id for block in current_blocks),
                )
            )
            current_blocks = []

        for block in blocks:
            key = self.section_key(block)
            if current_key is None:
                current_key = key
            if key != current_key:
                flush()
                current_key = key
            current_blocks.append(block)
        flush()
        return parents

    def tag_blocks(
        self,
        blocks: list[DocumentBlock],
        parents: list[ParentChunk] | None = None,
    ) -> list[DocumentBlock]:
        """Attach parent and section metadata to child blocks."""

        if not blocks:
            return []

        parent_by_block: dict[str, ParentChunk] = {}
        if parents:
            for parent in parents:
                for block_id in parent.block_ids:
                    parent_by_block[block_id] = parent

        tagged: list[DocumentBlock] = []
        for block in blocks:
            parent = parent_by_block.get(block.block_id)
            if parent is None:
                section_path = self._section_path_for_key(self.section_key(block))
                section_id = self.section_id(block.document_id, section_path)
            else:
                section_path = parent.section_path
                section_id = parent.section_id

            tagged.append(
                block.model_copy(
                    update={
                        "metadata": {
                            **block.metadata,
                            "parent_id": parent.parent_id if parent else section_id,
                            "section_id": section_id,
                            "chunk_level": block.metadata.get("chunk_level", "child"),
                        }
                    }
                )
            )
        return tagged

    def section_key(self, block: DocumentBlock) -> tuple[str, ...]:
        """Return the parent grouping key for a block."""

        if block.section_path:
            return tuple(block.section_path[: self.section_depth])
        return ("Document",)

    def section_id(self, document_id: str, section_path: list[str]) -> str:
        """Return a stable section/parent identifier."""

        slug = _SLUG_PATTERN.sub("-", " ".join(section_path).lower()).strip("-")
        if not slug:
            slug = "document"
        return f"{document_id}:{slug[:120]}"

    def _section_path_for_key(self, key: tuple[str, ...]) -> list[str]:
        return list(key)
