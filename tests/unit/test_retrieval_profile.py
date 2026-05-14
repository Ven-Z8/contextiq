from __future__ import annotations

from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.retrieval.profile import RetrievalProfile
from contextiq.retrieval.query import QueryAnalyzer
from contextiq.retrieval.ranker import CandidateRanker


def test_query_analyzer_uses_profile_assets_instead_of_builtins() -> None:
    profile = RetrievalProfile(
        asset_modes={"mode-x"},
        known_assets={"tesla bot"},
    )

    analysis = QueryAnalyzer(profile=profile).analyze("What MODE-X functions support Tesla Bot?")

    assert analysis.has_asset_mapping_query
    assert analysis.asset_modes == {"mode-x"}
    assert analysis.asset_terms == {"tesla bot"}


def test_ranker_uses_profile_source_aliases_for_named_source_precision() -> None:
    profile = RetrievalProfile(
        source_aliases={"tesla": {"tsla", "tesla"}},
    )
    analyzer = QueryAnalyzer(profile=profile)
    ranker = CandidateRanker(analyzer=analyzer)
    tesla = DocumentBlock(
        document_id="TSLA-2025-10K",
        block_id="tesla:income",
        source_path="TSLA-2025-10K.pdf",
        block_type=BlockType.TABLE,
        text="| Revenue | Gross margin | Operating income | Net income |",
    )
    apple = DocumentBlock(
        document_id="apple-2025-10k",
        block_id="apple:income",
        source_path="apple-2025-10k.pdf",
        block_type=BlockType.TABLE,
        text="| Revenue | Gross margin | Operating income | Net income |",
    )

    analysis = analyzer.analyze("What was Tesla revenue and net income in 2025?")
    precise = ranker.source_precision_candidates(analysis, [apple, tesla])

    assert [block.block_id for block in precise] == ["tesla:income"]


def test_ranker_uses_profile_product_markers_without_apple_specific_code() -> None:
    profile = RetrievalProfile(
        product_service_sections={"vehicles", "energy generation and storage"},
        product_markers={"model y", "model 3", "energy generation and storage"},
    )
    ranker = CandidateRanker(analyzer=QueryAnalyzer(profile=profile))
    table = DocumentBlock(
        document_id="tsla",
        block_id="tsla:product-table",
        source_path="tsla.pdf",
        section_path=["Product and Service Performance"],
        block_type=BlockType.TABLE,
        text=(
            "| Category | 2025 | 2024 |\n"
            "| Model Y | 100 | 90 |\n"
            "| Energy Generation and Storage | 50 | 40 |\n"
            "| Total net sales | 150 | 130 |"
        ),
    )

    assert ranker.is_product_services_performance_evidence(table)


def test_query_analyzer_expands_financial_products_from_profile() -> None:
    profile = RetrievalProfile(
        product_service_sections={"vehicles", "energy generation and storage"},
        product_markers={"model y", "model 3", "energy generation and storage"},
        financial_comparison_terms={"2026", "2025"},
    )

    analysis = QueryAnalyzer(profile=profile).analyze(
        "What is the products and services performance in 2025?"
    )

    assert "model y" in analysis.terms
    assert "energy generation and storage" in analysis.terms
    assert "2026" in analysis.terms
