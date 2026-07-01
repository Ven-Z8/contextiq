"""Phase 2: the one clean hybrid retrieve (no legacy pipeline).

ponytail: fake the vector index so the check runs with no Qdrant/models.
"""

from __future__ import annotations

from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.retrieval.store import LocalDocumentStore


class _FakeHit:
    def __init__(self, block_id: str, score: float) -> None:
        self.block_id = block_id
        self.score = score
        self.section_id = None


class _FakeIndex:
    def search_hybrid(
        self, query, limit, document_id=None, *, group_by_section=False, group_size=2
    ):
        # One id that exists in the store, one that does not.
        return [_FakeHit("d:0", 9.0), _FakeHit("ghost:9", 1.0)]


def test_hybrid_hits_maps_ids_and_drops_missing(tmp_path) -> None:
    store = LocalDocumentStore(path=tmp_path / "blocks.json")
    store.save_blocks(
        [
            DocumentBlock(
                document_id="d",
                block_id="d:0",
                source_path="c.md",
                page=1,
                section_path=[],
                block_type=BlockType.TEXT,
                text="Alpha owes rent.",
                metadata={},
            )
        ]
    )
    store._vector_index = _FakeIndex()  # bypass real Qdrant

    hits = store.hybrid_hits("rent", limit=40)

    assert [h.block.block_id for h in hits] == ["d:0"]  # unknown id dropped
    assert hits[0].stages == ["hybrid"]
    assert hits[0].rank == 0
