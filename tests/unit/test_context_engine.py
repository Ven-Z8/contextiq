from __future__ import annotations

import json
from pathlib import Path

from contextiq.context.engine import ContextEngine
from contextiq.context.models import ContextPacket, ContextSource
from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.retrieval.store import LocalDocumentStore


def test_context_engine_selects_matching_blocks(tmp_path: Path) -> None:
    store = LocalDocumentStore(path=tmp_path / "blocks.json")
    store.save_blocks(
        [
            DocumentBlock(
                document_id="doc",
                block_id="doc:1",
                source_path="sample.md",
                text="Regulatory risk and antitrust enforcement may affect platform operations.",
            ),
            DocumentBlock(
                document_id="doc",
                block_id="doc:2",
                source_path="sample.md",
                text="Unrelated hardware warranty details.",
            ),
        ]
    )

    packet = ContextEngine(store=store).build_context("What regulatory risks exist?")

    assert packet.sources
    assert packet.sources[0].block.block_id == "doc:1"


def test_context_engine_carries_retrieval_trace_for_visual_sources(tmp_path: Path) -> None:
    store = LocalDocumentStore(path=tmp_path / "blocks.json")
    store.save_blocks(
        [
            DocumentBlock(
                document_id="doc",
                block_id="doc:figure",
                source_path="sample.md",
                block_type=BlockType.FIGURE,
                text="Figure: Revenue by segment chart.",
                metadata={"visual_kind": "figure", "caption": "Revenue by segment chart"},
            )
        ]
    )

    packet = ContextEngine(store=store).build_context(
        "Find the revenue by segment chart",
        limit=3,
    )

    source = packet.sources[0]
    assert source.block.block_id == "doc:figure"
    assert "lexical" in source.stages
    assert source.score > 0
    assert "figure" in packet.to_markdown().lower()


def test_context_packet_markdown_fences_table_sources() -> None:
    block = DocumentBlock(
        document_id="nasa",
        block_id="nasa:uc",
        source_path="sheet.xlsx",
        block_type=BlockType.TABLE,
        text=(
            "| UC ID | Use Cases |\n"
            "| --- | --- |\n"
            "| UC-T-202 L | Transportation of large cargo |"
        ),
    )
    packet = ContextPacket(
        question="What is UC-T-202 L?",
        sources=[
            ContextSource(
                block=block,
                estimated_tokens=20,
                reason="structured code selected this table evidence block",
                stages=["structured_code"],
                score=73.0,
            )
        ],
        token_budget=6000,
        used_tokens=20,
        dropped_candidates=0,
    )

    markdown = packet.to_markdown()

    assert "```text\n| UC ID | Use Cases |" in markdown
    assert "UC-T-202 L" in markdown


def test_context_packet_markdown_includes_visual_metadata() -> None:
    block = DocumentBlock(
        document_id="doc",
        block_id="doc:figure",
        source_path="report.pdf",
        page=12,
        block_type=BlockType.FIGURE,
        text=(
            "Figure: Revenue trend chart.\n"
            "Visual description: Services revenue increased."
        ),
        metadata={
            "visual_kind": "figure",
            "caption": "Revenue trend chart.",
            "visual_description": "Services revenue increased.",
            "visual_description_provider": "docling-vlm",
            "visual_class": "line_chart",
            "visual_class_confidence": 0.88,
            "image_path": "data/processed/visuals/doc/block-00010.png",
        },
    )
    packet = ContextPacket(
        question="What does the revenue chart show?",
        sources=[
            ContextSource(
                block=block,
                estimated_tokens=20,
                reason="visual retrieval selected this figure",
                stages=["vector"],
                score=18.0,
            )
        ],
        token_budget=6000,
        used_tokens=20,
        dropped_candidates=0,
    )

    markdown = packet.to_markdown()

    assert "Visual: figure" in markdown
    assert "Caption: Revenue trend chart." in markdown
    assert "Visual class: line_chart (confidence 0.88)" in markdown
    assert "Visual description: Services revenue increased." in markdown
    assert "Visual description provider: docling-vlm" in markdown
    assert "Image artifact: data/processed/visuals/doc/block-00010.png" in markdown


def test_store_save_blocks_replaces_existing_document_parse(tmp_path: Path) -> None:
    store = LocalDocumentStore(path=tmp_path / "blocks.json")
    store.save_blocks(
        [
            DocumentBlock(
                document_id="doc",
                block_id="doc:old",
                source_path="sample.md",
                text="Old markdown parse block.",
            ),
            DocumentBlock(
                document_id="other",
                block_id="other:1",
                source_path="other.md",
                text="Other document remains.",
            ),
        ]
    )

    store.save_blocks(
        [
            DocumentBlock(
                document_id="doc",
                block_id="doc:new",
                source_path="sample.md",
                text="New structured parse block.",
            )
        ]
    )

    assert [block.block_id for block in store.load_blocks()] == ["other:1", "doc:new"]


def test_store_save_blocks_preserves_legacy_corpus_when_creating_manifest(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "blocks.json"
    store_path.write_text(
        json.dumps(
            [
                DocumentBlock(
                    document_id="legacy",
                    block_id="legacy:1",
                    source_path="legacy.md",
                    text="Legacy block.",
                ).model_dump(mode="json")
            ]
        ),
        encoding="utf-8",
    )
    store = LocalDocumentStore(path=store_path)

    store.save_blocks(
        [
            DocumentBlock(
                document_id="new",
                block_id="new:1",
                source_path="new.md",
                text="New block.",
            )
        ]
    )

    assert [block.block_id for block in store.load_blocks()] == ["legacy:1", "new:1"]


def test_store_writes_one_json_file_per_document(tmp_path: Path) -> None:
    store = LocalDocumentStore(path=tmp_path / "manifest.json")
    store.save_blocks(
        [
            DocumentBlock(
                document_id="doc/a",
                block_id="doc/a:1",
                source_path="a.md",
                text="First document.",
            ),
            DocumentBlock(
                document_id="doc:b",
                block_id="doc:b:1",
                source_path="b.md",
                text="Second document.",
            ),
        ]
    )

    document_files = sorted((tmp_path / "documents").glob("*.json"))

    assert len(document_files) == 2
    assert (tmp_path / "manifest.json").exists()
    assert [block.block_id for block in store.load_blocks()] == ["doc/a:1", "doc:b:1"]


