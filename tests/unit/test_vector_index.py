from __future__ import annotations

from pathlib import Path

from contextiq.ingestion.models import DocumentBlock
from contextiq.retrieval.vector_index import VectorIndex


def test_vector_index_accepts_citation_block_ids(tmp_path: Path) -> None:
    index = VectorIndex(path=tmp_path / "qdrant")
    block = DocumentBlock(
        document_id="apple-2025-10k-e8c0f8fec7",
        block_id="apple-2025-10k-e8c0f8fec7:0",
        source_path="data/raw/apple-2025-10k.pdf",
        text="Regulatory risk and antitrust enforcement may affect Apple operations.",
    )

    indexed = index.index_blocks([block])
    results = index.search("regulatory antitrust risk", limit=1)

    assert indexed == 1
    assert results == [block.block_id]


def test_vector_index_replaces_existing_document_vectors(tmp_path: Path) -> None:
    index = VectorIndex(path=tmp_path / "qdrant")
    old_block = DocumentBlock(
        document_id="doc",
        block_id="doc:old",
        source_path="sample.md",
        text="Old markdown parse about risk factors.",
    )
    new_block = DocumentBlock(
        document_id="doc",
        block_id="doc:new",
        source_path="sample.md",
        text="New structured parse about net sales performance.",
    )

    index.index_blocks([old_block])
    index.index_blocks([new_block])

    assert "doc:old" not in index.search("risk factors", limit=5)
    assert index.search("net sales performance", limit=1) == ["doc:new"]


def test_vector_index_surfaces_unexpected_delete_failures(tmp_path: Path) -> None:
    index = VectorIndex(path=tmp_path / "qdrant")

    def fail_delete(*args, **kwargs):
        raise RuntimeError("disk failure")

    index.client.delete = fail_delete  # type: ignore[method-assign]

    try:
        index._delete_existing_documents({"doc"})
    except RuntimeError as exc:
        assert "Failed to delete existing vectors" in str(exc)
    else:
        raise AssertionError("Expected unexpected vector delete failure to raise")
