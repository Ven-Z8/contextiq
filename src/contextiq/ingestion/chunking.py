"""Structure-aware chunking for parsed document blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from contextiq.ingestion.models import BlockType, DocumentBlock


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking policy for retrieval-ready blocks."""

    max_text_words: int = 350
    text_overlap_words: int = 50
    max_table_rows: int = 40
    table_overlap_rows: int = 3


class DocumentChunker:
    """Split oversized blocks while preserving structure and metadata."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk_blocks(self, blocks: list[DocumentBlock]) -> list[DocumentBlock]:
        chunks: list[DocumentBlock] = []
        for block in blocks:
            if block.block_type == BlockType.TABLE:
                chunks.extend(self._chunk_table(block))
            elif block.block_type == BlockType.TEXT:
                chunks.extend(self._chunk_text(block))
            else:
                chunks.append(block)
        return chunks

    def _chunk_text(self, block: DocumentBlock) -> list[DocumentBlock]:
        words = block.text.split()
        if len(words) <= self.config.max_text_words:
            return [block]

        windows = self._window_items(
            words,
            max_items=self.config.max_text_words,
            overlap=self.config.text_overlap_words,
        )
        return [
            self._copy_chunk(
                block=block,
                text=" ".join(window),
                chunk_index=index,
                chunk_count=len(windows),
                metadata={
                    "chunk_strategy": "text_window",
                    "word_start": start + 1,
                    "word_end": start + len(window),
                    "word_count": len(window),
                },
            )
            for index, (start, window) in enumerate(windows)
        ]

    def _chunk_table(self, block: DocumentBlock) -> list[DocumentBlock]:
        table = self._parse_markdown_table(block.text)
        if table is None:
            return [block]

        header, rows = table
        if len(rows) <= self.config.max_table_rows:
            return [block]

        windows = self._window_items(
            rows,
            max_items=self.config.max_table_rows,
            overlap=self.config.table_overlap_rows,
        )
        return [
            self._copy_chunk(
                block=block,
                text=self._render_markdown_table(header=header, rows=window),
                chunk_index=index,
                chunk_count=len(windows),
                metadata={
                    "chunk_strategy": "table_row_window",
                    "row_start": start + 1,
                    "row_end": start + len(window),
                    "row_count": len(window),
                },
            )
            for index, (start, window) in enumerate(windows)
        ]

    def _copy_chunk(
        self,
        block: DocumentBlock,
        text: str,
        chunk_index: int,
        chunk_count: int,
        metadata: dict[str, str | int | float | bool | None],
    ) -> DocumentBlock:
        return block.model_copy(
            update={
                "block_id": f"{block.block_id}:chunk-{chunk_index}",
                "text": text,
                "metadata": {
                    **block.metadata,
                    **metadata,
                    "parent_block_id": block.block_id,
                    "chunk_index": chunk_index,
                    "chunk_count": chunk_count,
                },
            }
        )

    def _window_items(
        self,
        items: list,
        max_items: int,
        overlap: int,
    ) -> list[tuple[int, list]]:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        safe_overlap = min(max(overlap, 0), max_items - 1)
        stride = max_items - safe_overlap
        windows: list[tuple[int, list]] = []
        for start in range(0, len(items), stride):
            window = items[start : start + max_items]
            if not window:
                break
            windows.append((start, window))
            if start + max_items >= len(items):
                break
        return windows

    def _parse_markdown_table(self, text: str) -> tuple[list[str], list[list[str]]] | None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
        if len(table_lines) < 3:
            return None

        header = self._parse_table_row(table_lines[0])
        separator = self._parse_table_row(table_lines[1])
        if not all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in separator):
            return None

        rows = [self._parse_table_row(line) for line in table_lines[2:]]
        return header, rows

    def _parse_table_row(self, line: str) -> list[str]:
        return [cell.strip() for cell in line.strip("|").split("|")]

    def _render_markdown_table(self, header: list[str], rows: list[list[str]]) -> str:
        column_count = max([len(header), *(len(row) for row in rows)])
        padded_header = header + [""] * (column_count - len(header))
        padded_rows = [row + [""] * (column_count - len(row)) for row in rows]
        lines = [
            "| " + " | ".join(padded_header) + " |",
            "| " + " | ".join(["---"] * column_count) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in padded_rows)
        return "\n".join(lines)
