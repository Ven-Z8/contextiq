from __future__ import annotations

from contextiq.ingestion.chunking import ChunkingConfig, DocumentChunker
from contextiq.ingestion.models import BlockType, DocumentBlock


def test_chunker_preserves_small_blocks() -> None:
    block = DocumentBlock(
        document_id="doc",
        block_id="doc:1",
        source_path="sample.md",
        text="Short focused paragraph.",
    )

    chunks = DocumentChunker().chunk_blocks([block])

    assert chunks == [block]


def test_chunker_splits_large_text_with_overlap_metadata() -> None:
    block = DocumentBlock(
        document_id="doc",
        block_id="doc:1",
        source_path="sample.md",
        section_path=["Risk"],
        text=" ".join(f"word{index}" for index in range(30)),
    )

    chunks = DocumentChunker(
        ChunkingConfig(max_text_words=10, text_overlap_words=2)
    ).chunk_blocks([block])

    assert [chunk.block_id for chunk in chunks] == [
        "doc:1:chunk-0",
        "doc:1:chunk-1",
        "doc:1:chunk-2",
        "doc:1:chunk-3",
    ]
    assert chunks[1].text.startswith("word8 word9")
    assert chunks[0].metadata["chunk_strategy"] == "text_window"
    assert chunks[0].metadata["parent_block_id"] == "doc:1"
    assert chunks[0].metadata["chunk_index"] == 0


def test_chunker_splits_large_markdown_table_by_row_window() -> None:
    block = DocumentBlock(
        document_id="doc",
        block_id="doc:table",
        source_path="sheet.xlsx",
        section_path=["Objectives"],
        block_type=BlockType.TABLE,
        text="\n".join(
            [
                "| ID | Description |",
                "| --- | --- |",
                "| L-001 | Power systems |",
                "| L-002 | Mobility systems |",
                "| L-003 | Communications |",
                "| L-004 | Autonomy |",
                "| L-005 | Habitation |",
            ]
        ),
        metadata={"parser": "openpyxl", "sheet_name": "Objectives"},
    )

    chunks = DocumentChunker(
        ChunkingConfig(max_table_rows=2, table_overlap_rows=1)
    ).chunk_blocks([block])

    assert [chunk.block_id for chunk in chunks] == [
        "doc:table:chunk-0",
        "doc:table:chunk-1",
        "doc:table:chunk-2",
        "doc:table:chunk-3",
    ]
    assert "| ID | Description |" in chunks[1].text
    assert "L-002" in chunks[1].text
    assert "L-003" in chunks[1].text
    assert chunks[1].metadata["row_start"] == 2
    assert chunks[1].metadata["row_end"] == 3
    assert chunks[1].metadata["chunk_strategy"] == "table_row_window"
