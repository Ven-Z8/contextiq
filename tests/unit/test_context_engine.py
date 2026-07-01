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


def test_store_lexical_search_works_without_vector_index(tmp_path: Path) -> None:
    store = LocalDocumentStore(path=tmp_path / "blocks.json")
    store.save_blocks(
        [
            DocumentBlock(
                document_id="doc",
                block_id="doc:1",
                source_path="sample.md",
                text="A termination clause controls renewal rights.",
            )
        ]
    )

    results = store._search_lexical("termination rights", limit=3)

    assert [result.block_id for result in results] == ["doc:1"]


def test_store_strict_vector_search_exposes_index_errors(tmp_path: Path) -> None:
    store = LocalDocumentStore(path=tmp_path / "blocks.json", strict_vector_errors=True)
    store.save_blocks(
        [
            DocumentBlock(
                document_id="doc",
                block_id="doc:1",
                source_path="sample.md",
                text="A termination clause controls renewal rights.",
            )
        ]
    )

    class BrokenVectorIndex:
        def search(self, query: str, limit: int) -> list[str]:
            raise RuntimeError("vector index unavailable")

    store.vector_index_factory = BrokenVectorIndex

    try:
        store._search_vector("termination rights", limit=3)
    except RuntimeError as exc:
        assert "Vector search failed" in str(exc)
    else:
        raise AssertionError("Expected strict vector search to raise")


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


def test_store_expands_heading_candidates_to_nearby_text(tmp_path: Path) -> None:
    store = LocalDocumentStore(path=tmp_path / "blocks.json")
    store.save_blocks(
        [
            DocumentBlock(
                document_id="doc",
                block_id="doc:1",
                source_path="sample.md",
                block_type=BlockType.HEADING,
                text="## Legal and Regulatory Compliance Risks",
            ),
            DocumentBlock(
                document_id="doc",
                block_id="doc:2",
                source_path="sample.md",
                text=(
                    "Government investigations and legal proceedings may adversely "
                    "affect operations."
                ),
            ),
        ]
    )

    expanded = store._expand_candidate_blocks([store.load_blocks()[0]], limit=3)

    assert [block.block_id for block in expanded] == ["doc:1", "doc:2"]


def test_store_expands_candidates_to_same_section_neighbors(tmp_path: Path) -> None:
    store = LocalDocumentStore(path=tmp_path / "blocks.json")
    store.save_blocks(
        [
            DocumentBlock(
                document_id="doc",
                block_id="doc:heading",
                source_path="sample.md",
                block_type=BlockType.HEADING,
                section_path=["Confidentiality"],
                text="## Confidentiality",
            ),
            DocumentBlock(
                document_id="doc",
                block_id="doc:evidence",
                source_path="sample.md",
                section_path=["Confidentiality"],
                text="Each party must protect confidential information using reasonable care.",
            ),
            DocumentBlock(
                document_id="doc",
                block_id="doc:next",
                source_path="sample.md",
                block_type=BlockType.HEADING,
                section_path=["Termination"],
                text="## Termination",
            ),
        ]
    )

    expanded = store._expand_candidate_blocks([store.load_blocks()[1]], limit=5)

    assert [block.block_id for block in expanded] == ["doc:heading", "doc:evidence"]


def test_store_adds_query_matching_section_anchor_candidates(tmp_path: Path) -> None:
    store = LocalDocumentStore(path=tmp_path / "blocks.json")
    store.save_blocks(
        [
            DocumentBlock(
                document_id="doc",
                block_id="doc:segment-heading",
                source_path="sample.md",
                block_type=BlockType.HEADING,
                section_path=["Segment Operating Performance"],
                text="## Segment Operating Performance",
            ),
            DocumentBlock(
                document_id="doc",
                block_id="doc:segment-table",
                source_path="sample.md",
                block_type=BlockType.TABLE,
                section_path=["Segment Operating Performance"],
                text="| Americas | 2025 | 2024 |",
            ),
            DocumentBlock(
                document_id="doc",
                block_id="doc:unrelated",
                source_path="sample.md",
                section_path=["Other"],
                text="Unrelated obligations.",
            ),
        ]
    )

    anchors = store._section_anchor_candidates(
        "What was the segment operating performance?",
        limit=5,
    )

    assert [block.block_id for block in anchors] == ["doc:segment-heading"]


def test_store_reranks_evidence_paragraphs_above_generic_headings(tmp_path: Path) -> None:
    store = LocalDocumentStore(path=tmp_path / "blocks.json")
    blocks = [
        DocumentBlock(
            document_id="doc",
            block_id="doc:1",
            source_path="sample.md",
            block_type=BlockType.HEADING,
            text="## Apple Inc.",
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:2",
            source_path="sample.md",
            block_type=BlockType.TEXT,
            section_path=["Legal and Regulatory Compliance Risks"],
            text=(
                "Government investigations, competition laws, privacy regulations, and legal "
                "proceedings may adversely affect operations."
            ),
        ),
    ]

    ranked = store._rerank_candidates(
        "What are Apple's main regulatory risks? Cite pages.",
        blocks,
    )

    assert [block.block_id for block in ranked] == ["doc:2", "doc:1"]


def test_store_reranks_exact_section_clause_above_broad_obligation_prose(
    tmp_path: Path,
) -> None:
    store = LocalDocumentStore(path=tmp_path / "blocks.json")
    blocks = [
        DocumentBlock(
            document_id="filing",
            block_id="filing:obligations",
            source_path="filing.pdf",
            section_path=["Privacy and Data Obligations"],
            text=(
                "The company is subject to many international obligations, regulations, "
                "policies, and controls relating to personal data."
            ),
        ),
        DocumentBlock(
            document_id="contract",
            block_id="contract:confidentiality",
            source_path="contract.md",
            section_path=["Confidentiality"],
            text=(
                "Each party must protect confidential information using reasonable care "
                "and disclose it only to advisors who need access."
            ),
        ),
    ]

    ranked = store._rerank_candidates(
        "What confidentiality obligations do the parties have?",
        blocks,
    )

    assert [block.block_id for block in ranked] == [
        "contract:confidentiality",
        "filing:obligations",
    ]


def test_store_boosts_exact_section_term_over_repeated_generic_terms(
    tmp_path: Path,
) -> None:
    store = LocalDocumentStore(path=tmp_path / "blocks.json")
    broad = DocumentBlock(
        document_id="filing",
        block_id="filing:broad",
        source_path="filing.pdf",
        section_path=["General Obligations"],
        text=(
            "Parties may have many obligations and must satisfy obligations "
            "under regulations, policies, and controls."
        ),
    )
    exact = DocumentBlock(
        document_id="contract",
        block_id="contract:confidentiality",
        source_path="contract.md",
        section_path=["Confidentiality"],
        text="Each party must protect confidential information using reasonable care.",
    )

    ranked = store._rerank_candidates(
        "What confidentiality obligations do the parties have?",
        [broad, exact],
    )

    assert ranked[0].block_id == "contract:confidentiality"


def test_store_reranks_financial_performance_tables_above_risk_prose(tmp_path: Path) -> None:
    store = LocalDocumentStore(path=tmp_path / "blocks.json")
    blocks = [
        DocumentBlock(
            document_id="doc",
            block_id="doc:risk",
            source_path="sample.md",
            page=11,
            section_path=["Business Risks"],
            block_type=BlockType.TEXT,
            text=(
                "In 2025 the Company must introduce new products and services and manage "
                "performance issues across products and services or its business may be "
                "adversely affected. Products and services can face market performance risks."
            ),
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:table",
            source_path="sample.md",
            page=26,
            section_path=["Products and Services Performance"],
            block_type=BlockType.TABLE,
            text=(
                "| Category | 2025 | Change | 2024 |\n"
                "| iPhone | $209,586 | 4% | $201,183 |\n"
                "| Services | $109,158 | 14% | $96,169 |\n"
                "| Total net sales | $416,161 | 6% | $391,035 |"
            ),
        ),
    ]

    ranked = store._rerank_candidates(
        "What is the Products and Services Performance in 2025?",
        blocks,
    )

    assert [block.block_id for block in ranked] == ["doc:table", "doc:risk"]


def test_store_search_prefers_products_services_section_over_financial_noise(
    tmp_path: Path,
) -> None:
    store = LocalDocumentStore(path=tmp_path / "blocks.json")
    store.save_blocks(
        [
            DocumentBlock(
                document_id="doc",
                block_id="doc:performance-heading",
                source_path="sample.md",
                page=26,
                section_path=["Products and Services Performance"],
                block_type=BlockType.HEADING,
                text="## Products and Services Performance",
            ),
            DocumentBlock(
                document_id="doc",
                block_id="doc:category-table",
                source_path="sample.md",
                page=26,
                section_path=["Products and Services Performance"],
                block_type=BlockType.TABLE,
                text=(
                    "| Category | 2025 | Change | 2024 | Change | 2023 |\n"
                    "| iPhone | $209,586 | 4% | $201,183 | -% | $200,583 |\n"
                    "| Services | $109,158 | 14% | $96,169 | 13% | $85,200 |\n"
                    "| Total net sales | $416,161 | 6% | $391,035 | 2% | $383,285 |"
                ),
            ),
            DocumentBlock(
                document_id="doc",
                block_id="doc:iphone",
                source_path="sample.md",
                page=26,
                section_path=["iPhone"],
                block_type=BlockType.TEXT,
                text=(
                    "iPhone net sales increased during 2025 compared to 2024 due "
                    "to higher net sales of Pro models."
                ),
            ),
            DocumentBlock(
                document_id="doc",
                block_id="doc:services",
                source_path="sample.md",
                page=26,
                section_path=["Services"],
                block_type=BlockType.TEXT,
                text=(
                    "Services net sales increased during 2025 compared to 2024 primarily due "
                    "to higher net sales from advertising, the App Store and cloud services."
                ),
            ),
            DocumentBlock(
                document_id="doc",
                block_id="doc:tariffs",
                source_path="sample.md",
                page=25,
                section_path=["Tariffs and Other Measures"],
                block_type=BlockType.TEXT,
                text=(
                    "In 2025 tariffs and other measures applied to products and services can "
                    "affect pricing, gross margin, business, results of operations and "
                    "financial condition."
                ),
            ),
            DocumentBlock(
                document_id="doc",
                block_id="doc:margin-risk",
                source_path="sample.md",
                page=18,
                section_path=[
                    "The Company's net sales and gross margins are subject to volatility "
                    "and downward pressure due to a variety of factors."
                ],
                block_type=BlockType.TEXT,
                text=(
                    "The Company's gross margins vary significantly across its products, "
                    "services, geographic segments and distribution channels and can change "
                    "over time. These factors could adversely affect results of operations."
                ),
            ),
            DocumentBlock(
                document_id="doc",
                block_id="doc:cash-flow",
                source_path="sample.md",
                page=36,
                section_path=["CONSOLIDATED STATEMENTS OF CASH FLOWS"],
                block_type=BlockType.TABLE,
                text=(
                    "| Cash generated by operating activities | 2025 | 2024 | 2023 |\n"
                    "| Net income | 112,010 | 93,736 | 96,995 |"
                ),
            ),
        ]
    )

    results = store.search(
        "What is the Products and Services Performance in 2025?",
        limit=5,
    )

    result_ids = [block.block_id for block in results]
    assert "doc:category-table" in result_ids[:2]
    assert "doc:iphone" in result_ids
    assert "doc:services" in result_ids
    assert "doc:performance-heading" not in result_ids
    assert "doc:tariffs" not in result_ids
    assert "doc:margin-risk" not in result_ids
    assert "doc:cash-flow" not in result_ids


def test_store_deprioritizes_gross_margin_for_net_sales_category_query(
    tmp_path: Path,
) -> None:
    store = LocalDocumentStore(path=tmp_path / "blocks.json")
    blocks = [
        DocumentBlock(
            document_id="doc",
            block_id="doc:iphone",
            source_path="sample.md",
            section_path=["iPhone"],
            text="iPhone net sales increased during 2025 compared to 2024.",
        ),
        DocumentBlock(
            document_id="doc",
            block_id="doc:gross-margin",
            source_path="sample.md",
            section_path=["Products Gross Margin"],
            text="Products gross margin increased during 2025 compared to 2024.",
        ),
    ]

    ranked = store._rerank_candidates(
        "What is the Products and Services Performance in 2025?",
        blocks,
    )

    assert [block.block_id for block in ranked] == ["doc:iphone", "doc:gross-margin"]
