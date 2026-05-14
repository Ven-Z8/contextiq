from __future__ import annotations

from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.retrieval.query import QueryAnalyzer, QueryIntent
from contextiq.retrieval.ranker import CandidateRanker


def test_query_analyzer_detects_financial_performance_intent() -> None:
    analysis = QueryAnalyzer().analyze(
        "What is the Products and Services Performance in 2025?"
    )

    assert analysis.intent == QueryIntent.FINANCIAL_PERFORMANCE
    assert {"products", "services", "net", "sales", "2025"} <= analysis.terms


def test_query_analyzer_expands_regulatory_risk_terms() -> None:
    analysis = QueryAnalyzer().analyze("What regulatory risks exist?")

    assert analysis.intent == QueryIntent.GENERAL
    assert {"regulatory", "legal", "antitrust", "adverse", "operations"} <= analysis.terms


def test_query_analyzer_extracts_structured_codes() -> None:
    analysis = QueryAnalyzer().analyze("What is UC ID UC-T-202 L ?")

    assert analysis.structured_codes == {"uc-t-202 l", "uc-t-202"}
    assert {"uc-t-202 l", "uc-t-202"} <= analysis.terms


def test_query_analyzer_detects_asset_mapping_query() -> None:
    analysis = QueryAnalyzer().analyze("What HLR functions support for Orion?")

    assert analysis.has_asset_mapping_query
    assert analysis.asset_modes == {"hlr"}
    assert analysis.asset_terms == {"orion"}


def test_query_analyzer_detects_contract_intent() -> None:
    analysis = QueryAnalyzer().analyze("How can either party terminate the agreement?")

    assert analysis.intent == QueryIntent.CONTRACT


def test_candidate_ranker_prefers_financial_table_over_risk_prose() -> None:
    ranker = CandidateRanker()
    risk = DocumentBlock(
        document_id="doc",
        block_id="doc:risk",
        source_path="sample.md",
        section_path=["Business Risks"],
        block_type=BlockType.TEXT,
        text=(
            "In 2025 the Company must introduce new products and services and manage "
            "performance issues or results may be adversely affected."
        ),
    )
    table = DocumentBlock(
        document_id="doc",
        block_id="doc:table",
        source_path="sample.md",
        section_path=["Products and Services Performance"],
        block_type=BlockType.TABLE,
        text=(
            "| Category | 2025 | 2024 | 2023 |\n"
            "| iPhone | $209,586 | $201,183 | $200,583 |\n"
            "| Services | $109,158 | $96,169 | $85,200 |\n"
            "| Total net sales | $416,161 | $391,035 | $383,285 |"
        ),
    )

    ranked = ranker.rerank(
        "What is the Products and Services Performance in 2025?",
        [risk, table],
    )

    assert [block.block_id for block in ranked] == ["doc:table", "doc:risk"]
