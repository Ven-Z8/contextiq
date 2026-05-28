from __future__ import annotations

from pathlib import Path

from contextiq.ingestion.hierarchy import ParentChunk
from contextiq.ingestion.models import DocumentBlock
from contextiq.retrieval.parent_resolver import ParentResolver
from contextiq.retrieval.store import LocalDocumentStore
from contextiq.retrieval.vector_index import VectorIndex


def test_vector_index_hybrid_search_groups_by_section(tmp_path: Path) -> None:
    index = VectorIndex(path=tmp_path / "qdrant")
    blocks = [
        DocumentBlock(
            document_id="doc",
            block_id="doc:0",
            source_path="sample.pdf",
            section_path=["Risk Factors"],
            text="Regulatory risk and antitrust enforcement may affect operations.",
            metadata={
                "section_id": "doc:risk-factors",
                "parent_id": "doc:risk-factors",
                "chunk_level": "child",
            },
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:1",
            source_path="sample.pdf",
            section_path=["Risk Factors"],
            text="Additional regulatory risk disclosures for the quarter.",
            metadata={
                "section_id": "doc:risk-factors",
                "parent_id": "doc:risk-factors",
                "chunk_level": "child",
            },
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:2",
            source_path="sample.pdf",
            section_path=["Financial Results"],
            text="Net sales increased year over year in all segments.",
            metadata={
                "section_id": "doc:financial-results",
                "parent_id": "doc:financial-results",
                "chunk_level": "child",
            },
        ),
    ]

    indexed = index.index_blocks(blocks)
    hits = index.search_hybrid(
        "regulatory antitrust risk",
        limit=2,
        document_id="doc",
        group_by_section=True,
        group_size=1,
    )
    sparse_hits = index.search_sparse("regulatory antitrust", limit=2, document_id="doc")

    assert indexed == 3
    assert hits
    assert {hit.block_id for hit in hits}.issubset({"doc:0", "doc:1", "doc:2"})
    assert sparse_hits


def test_store_persists_parents_and_lazy_loads_document(tmp_path: Path) -> None:
    store = LocalDocumentStore(path=tmp_path / "processed" / "blocks.json")
    blocks = [
        DocumentBlock(
            document_id="doc",
            block_id="doc:0",
            source_path="sample.pdf",
            section_path=["Risk Factors"],
            text="Regulatory risk may affect operations.",
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:1",
            source_path="sample.pdf",
            section_path=["Financial Results"],
            text="Net sales increased year over year.",
        ),
    ]

    store.save_blocks(blocks)
    fresh_store = LocalDocumentStore(path=tmp_path / "processed" / "blocks.json")
    scoped = fresh_store.scoped("doc")
    loaded = scoped.load_blocks()

    assert len(loaded) == 2
    assert loaded[0].metadata["parent_id"]
    parent = scoped.load_parent(str(loaded[0].metadata["parent_id"]))
    assert parent is not None
    assert "Regulatory risk" in parent.text or "Net sales" in parent.text


def test_parent_resolver_enriches_child_hits() -> None:
    parent = ParentChunk(
        parent_id="doc:risk-factors",
        document_id="doc",
        section_id="doc:risk-factors",
        section_path=["Risk Factors"],
        text="Regulatory risk and supply chain risk are disclosed in this section.",
        page_start=1,
        page_end=2,
        block_ids=("doc:0",),
    )
    child = DocumentBlock(
        document_id="doc",
        block_id="doc:0",
        source_path="sample.pdf",
        section_path=["Risk Factors"],
        text="Regulatory risk may affect operations.",
        metadata={
            "parent_id": parent.parent_id,
            "section_id": parent.section_id,
            "chunk_level": "child",
        },
    )
    def load_parent(parent_id: str) -> ParentChunk | None:
        return parent if parent_id == parent.parent_id else None

    resolver = ParentResolver(parent_loader=load_parent)

    enriched = resolver.enrich_candidates([child])

    assert len(enriched) == 2
    assert enriched[0].block_id == "doc:0"
    assert enriched[1].metadata["chunk_level"] == "parent"
