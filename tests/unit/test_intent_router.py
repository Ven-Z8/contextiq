"""Unit tests for QueryIntentRouter — query-to-search-config routing.

All tests use synthetic queries only — no corpus, no vector index needed.
Tests verify routing decisions and IntentSearchConfig correctness.
"""

from __future__ import annotations

import pytest

from contextiq.retrieval.intent_router import (
    IntentSearchConfig,
    QueryIntentRouter,
    _ANALYTICAL_CONFIG,
    _DEFAULT_CONFIG,
    _FINANCIAL_CONFIG,
    _RISK_CONFIG,
    _STRUCTURED_CODE_CONFIG,
)
from contextiq.retrieval.query import QueryAnalyzer, QueryIntent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def router() -> QueryIntentRouter:
    return QueryIntentRouter()


@pytest.fixture
def analyzer() -> QueryAnalyzer:
    return QueryAnalyzer()


# ---------------------------------------------------------------------------
# IntentSearchConfig invariants
# ---------------------------------------------------------------------------

class TestIntentSearchConfigInvariants:
    """Ensure pre-built configs satisfy domain constraints."""

    def test_financial_config_is_sparse_dominant(self):
        assert _FINANCIAL_CONFIG.sparse_limit > _FINANCIAL_CONFIG.dense_limit
        assert _FINANCIAL_CONFIG.sparse_limit >= 60

    def test_analytical_config_is_dense_dominant(self):
        assert _ANALYTICAL_CONFIG.dense_limit > _ANALYTICAL_CONFIG.sparse_limit
        assert _ANALYTICAL_CONFIG.dense_limit >= 60

    def test_structured_code_config_is_very_sparse_dominant(self):
        assert _STRUCTURED_CODE_CONFIG.sparse_limit >= 70
        assert _STRUCTURED_CODE_CONFIG.sparse_limit > _STRUCTURED_CODE_CONFIG.dense_limit

    def test_risk_config_is_balanced_to_dense(self):
        # Risk should favour dense slightly (prose retrieval)
        assert _RISK_CONFIG.dense_limit >= _RISK_CONFIG.sparse_limit

    def test_default_config_is_balanced(self):
        assert _DEFAULT_CONFIG.dense_limit == _DEFAULT_CONFIG.sparse_limit

    def test_all_configs_have_rrf_candidates(self):
        configs = [
            _FINANCIAL_CONFIG, _ANALYTICAL_CONFIG,
            _RISK_CONFIG, _STRUCTURED_CODE_CONFIG, _DEFAULT_CONFIG,
        ]
        for config in configs:
            assert config.rrf_candidates >= 10, (
                f"{config.description} has too few RRF candidates"
            )

    def test_financial_config_biases_table_blocks(self):
        assert _FINANCIAL_CONFIG.block_type_bias == "table"
        assert _FINANCIAL_CONFIG.content_profile_bias == "financial_table"

    def test_analytical_config_biases_narrative_blocks(self):
        assert _ANALYTICAL_CONFIG.content_profile_bias == "narrative_para"

    def test_risk_config_biases_risk_sections(self):
        assert _RISK_CONFIG.content_profile_bias == "risk_section"


# ---------------------------------------------------------------------------
# Financial query routing
# ---------------------------------------------------------------------------

class TestFinancialQueryRouting:
    """Queries about revenue, margins, EPS should get sparse-heavy config."""

    def test_revenue_query_routes_to_financial(self, router):
        config = router.route("What was Apple's total net sales in 2025?")
        assert config.sparse_limit > config.dense_limit

    def test_gross_margin_routes_to_financial(self, router):
        config = router.route("What is the gross margin percentage for the Products segment?")
        assert config.sparse_limit > config.dense_limit

    def test_eps_query_routes_to_financial(self, router):
        config = router.route("What was the diluted earnings per share?")
        assert config.sparse_limit > config.dense_limit

    def test_financial_config_returned(self, router):
        config = router.route("Show me revenue breakdown by segment for fiscal 2025")
        # Should be financial or at minimum sparse-dominant
        assert config.sparse_limit >= config.dense_limit

    def test_financial_query_has_table_bias(self, router):
        config = router.route("What were total net sales in fiscal 2025?")
        assert config.block_type_bias == "table" or config.content_profile_bias == "financial_table"


# ---------------------------------------------------------------------------
# Analytical query routing
# ---------------------------------------------------------------------------

class TestAnalyticalQueryRouting:
    """Why/how explanatory queries should get dense-heavy config."""

    def test_why_query_routes_to_analytical(self, router):
        config = router.route("Why did Services revenue grow faster than Products?")
        assert config.dense_limit > config.sparse_limit

    def test_how_query_routes_to_analytical(self, router):
        config = router.route("How does Apple generate revenue from the Services segment?")
        assert config.dense_limit > config.sparse_limit

    def test_explain_query_routes_to_analytical(self, router):
        config = router.route("Explain the factors that contributed to margin expansion")
        assert config.dense_limit > config.sparse_limit

    def test_cause_query_routes_to_analytical(self, router):
        config = router.route("What caused the decline in iPhone unit sales?")
        assert config.dense_limit > config.sparse_limit

    def test_describe_query_routes_to_analytical(self, router):
        config = router.route("Describe Apple's strategy for the China market")
        assert config.dense_limit > config.sparse_limit


# ---------------------------------------------------------------------------
# Risk query routing
# ---------------------------------------------------------------------------

class TestRiskQueryRouting:
    """Risk/compliance/regulatory queries should get risk config."""

    def test_risk_query_routes_correctly(self, router):
        config = router.route("What are the key risks to Apple's business?")
        assert config.content_profile_bias == "risk_section"

    def test_tariff_query_routes_to_risk(self, router):
        config = router.route("What tariff exposure does Apple face from Chinese imports?")
        assert config.content_profile_bias == "risk_section"

    def test_regulatory_query_routes_to_risk(self, router):
        config = router.route("What regulatory compliance risks exist in the EU?")
        assert config.content_profile_bias == "risk_section"

    def test_litigation_query_routes_to_risk(self, router):
        config = router.route("Is Apple facing any material litigation?")
        assert config.content_profile_bias == "risk_section"

    def test_adverse_query_routes_to_risk(self, router):
        config = router.route("What factors could adversely impact operating income?")
        assert config.content_profile_bias == "risk_section"

    def test_risk_config_has_dense_preference(self, router):
        config = router.route("What are the primary risk factors facing Apple?")
        # Risk config is balanced-to-dense for prose retrieval
        assert config.dense_limit >= config.sparse_limit


# ---------------------------------------------------------------------------
# Structured code routing
# ---------------------------------------------------------------------------

class TestStructuredCodeRouting:
    """Code-like tokens (CN-P-103, IFRS-17, ASC-842) should be sparse-dominant."""

    def test_accounting_standard_routes_to_sparse(self, router, analyzer):
        analysis = analyzer.analyze("What does IFRS-17 require for insurance contracts?")
        if analysis.has_structured_codes:
            config = router.route("What does IFRS-17 require?", analysis=analysis)
            assert config.sparse_limit >= 70

    def test_default_fallback_for_general_query(self, router):
        config = router.route("Tell me about Apple's business")
        # Generic query → default balanced config
        assert config.dense_limit == config.sparse_limit or config.description == "general: balanced hybrid"


# ---------------------------------------------------------------------------
# Router describe() helper
# ---------------------------------------------------------------------------

class TestRouterDescribe:
    """Verify the describe() method returns useful debug strings."""

    def test_describe_returns_string(self, router):
        result = router.describe("What was total revenue?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_describe_contains_limits(self, router):
        result = router.describe("What was total revenue?")
        assert "dense=" in result
        assert "sparse=" in result

    def test_describe_different_for_different_intents(self, router):
        financial = router.describe("What was total revenue in fiscal 2025?")
        analytical = router.describe("Why did revenue grow in fiscal 2025?")
        # Descriptions should differ (different configs)
        assert financial != analytical


# ---------------------------------------------------------------------------
# Pre-supplied analysis passthrough
# ---------------------------------------------------------------------------

class TestPreSuppliedAnalysis:
    """When analysis is pre-supplied, router should use it (not re-analyze)."""

    def test_accepts_pre_analyzed_financial_intent(self, router, analyzer):
        analysis = analyzer.analyze("net sales revenue gross margin")
        config = router.route("net sales revenue gross margin", analysis=analysis)
        assert isinstance(config, IntentSearchConfig)

    def test_structured_code_in_analysis_overrides_intent(self, router, analyzer):
        """Structured code detection takes priority over base intent."""
        analysis = analyzer.analyze("What is ASC-842 lease accounting treatment?")
        if analysis.has_structured_codes:
            config = router.route("ASC-842", analysis=analysis)
            assert config.sparse_limit >= 70

    def test_none_analysis_triggers_fresh_analyze(self, router):
        """Passing None for analysis should still work (router calls analyzer internally)."""
        config = router.route("How did operating expenses change year over year?", analysis=None)
        assert isinstance(config, IntentSearchConfig)
        # "how" → analytical
        assert config.dense_limit > config.sparse_limit
