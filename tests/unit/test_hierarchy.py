from __future__ import annotations

from contextiq.ingestion.hierarchy import HierarchyBuilder
from contextiq.ingestion.models import DocumentBlock


def test_hierarchy_builds_section_parents() -> None:
    blocks = [
        DocumentBlock(
            document_id="doc",
            block_id="doc:0",
            source_path="sample.pdf",
            section_path=["Risk Factors"],
            text="Regulatory risk may affect operations.",
            page=1,
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:1",
            source_path="sample.pdf",
            section_path=["Risk Factors"],
            text="Supply chain disruption remains a concern.",
            page=2,
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:2",
            source_path="sample.pdf",
            section_path=["Financial Results"],
            text="Net sales increased year over year.",
            page=3,
        ),
    ]

    builder = HierarchyBuilder()
    parents = builder.build_parents(blocks)

    assert len(parents) == 2
    assert parents[0].section_path == ["Risk Factors"]
    assert "Regulatory risk" in parents[0].text
    assert "Supply chain" in parents[0].text
    assert parents[0].page_start == 1
    assert parents[0].page_end == 2


def test_hierarchy_tags_child_blocks_with_parent_metadata() -> None:
    blocks = [
        DocumentBlock(
            document_id="doc",
            block_id="doc:0",
            source_path="sample.pdf",
            section_path=["Risk Factors"],
            text="Regulatory risk may affect operations.",
        )
    ]
    builder = HierarchyBuilder()
    parents = builder.build_parents(blocks)
    tagged = builder.tag_blocks(blocks, parents)

    assert tagged[0].metadata["chunk_level"] == "child"
    assert tagged[0].metadata["parent_id"] == parents[0].parent_id
    assert tagged[0].metadata["section_id"] == parents[0].section_id
