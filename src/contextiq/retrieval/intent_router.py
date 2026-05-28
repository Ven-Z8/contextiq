"""Query intent routing: map query type to optimal Qdrant search configuration.

Different enterprise queries need different retrieval strategies:
  - Financial tables need sparse/keyword dominance (exact dollar amounts, ticker codes)
  - "Why/How" explanatory queries need semantic dense dominance (prose explanations)
  - Risk/regulatory queries need prose preference + broad coverage
  - Structured code queries (CN-P-103, IFRS-17) need exact match

This router maps a QueryAnalysis → IntentSearchConfig, which VectorIndex
uses to tune prefetch limits, fusion weights, and payload filters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contextiq.retrieval.query import QueryAnalysis, QueryAnalyzer, QueryIntent


@dataclass
class IntentSearchConfig:
    """Qdrant search configuration tuned for a specific query intent.

    dense_limit: Candidates fetched from dense (semantic) prefetch
    sparse_limit: Candidates fetched from sparse (SPLADE) prefetch
    rrf_candidates: Candidates passed from RRF fusion to ColBERT reranking
    block_type_bias: Optional soft filter — prefer blocks of this type in scoring
    content_profile_bias: Optional payload filter on content_profile field
    description: Human-readable description for debugging
    """

    dense_limit: int = 50
    sparse_limit: int = 50
    rrf_candidates: int = 20
    block_type_bias: str | None = None
    content_profile_bias: str | None = None
    description: str = "balanced"
    extra_filters: list[dict] = field(default_factory=list)


# Pre-built configs for each intent
_FINANCIAL_CONFIG = IntentSearchConfig(
    dense_limit=30,
    sparse_limit=70,         # SPLADE dominates: exact dollar amounts, segment names
    rrf_candidates=25,
    block_type_bias="table",
    content_profile_bias="financial_table",
    description="financial_performance: sparse-heavy for exact number matching",
)

_ANALYTICAL_CONFIG = IntentSearchConfig(
    dense_limit=70,          # Dense dominates: semantic understanding for "why/how"
    sparse_limit=30,
    rrf_candidates=20,
    block_type_bias="text",
    content_profile_bias="narrative_para",
    description="analytical: dense-heavy for explanatory prose retrieval",
)

_RISK_CONFIG = IntentSearchConfig(
    dense_limit=60,
    sparse_limit=40,
    rrf_candidates=20,
    content_profile_bias="risk_section",
    description="risk_compliance: balanced with risk section preference",
)

_STRUCTURED_CODE_CONFIG = IntentSearchConfig(
    dense_limit=20,
    sparse_limit=80,         # SPLADE term expansion captures code variants
    rrf_candidates=15,
    description="structured_code: sparse-dominant for exact code matching",
)

_CONTRACT_CONFIG = IntentSearchConfig(
    dense_limit=50,
    sparse_limit=50,
    rrf_candidates=20,
    description="contract: balanced for legal clause retrieval",
)

_DEFAULT_CONFIG = IntentSearchConfig(
    description="general: balanced hybrid",
)

# Intent signals: query terms that override the base intent
_ANALYTICAL_SIGNALS = {
    "why", "how", "explain", "reason", "cause", "because", "drove",
    "contributed", "impact", "effect", "resulted", "led", "factor",
    "describe", "overview", "discuss",
}

_RISK_SIGNALS = {
    "risk", "risks", "compliance", "regulation", "regulatory",
    "exposure", "liability", "uncertainty", "adverse", "material",
    "tariff", "sanction", "lawsuit", "litigation", "penalty", "fine",
}


class QueryIntentRouter:
    """Map a query to the optimal Qdrant search configuration.

    Uses both the coarse QueryIntent from QueryAnalyzer AND fine-grained
    signal detection to handle cases the analyzer misses:
      - "Why did Services revenue grow?" → GENERAL intent but ANALYTICAL config
      - "What tariff risks does Apple face?" → GENERAL intent but RISK config
    """

    def __init__(self, analyzer: QueryAnalyzer | None = None) -> None:
        self.analyzer = analyzer or QueryAnalyzer()

    def route(
        self,
        query: str,
        analysis: QueryAnalysis | None = None,
    ) -> IntentSearchConfig:
        """Return the IntentSearchConfig most appropriate for this query."""
        if analysis is None:
            analysis = self.analyzer.analyze(query)

        # Structured codes always go sparse-heavy for exact matching
        if analysis.has_structured_codes:
            return _STRUCTURED_CODE_CONFIG

        # Financial performance: sparse-heavy for table/number retrieval
        if analysis.intent == QueryIntent.FINANCIAL_PERFORMANCE:
            return _FINANCIAL_CONFIG

        # Contract queries
        if analysis.intent == QueryIntent.CONTRACT:
            return _CONTRACT_CONFIG

        # Signal-based overrides for GENERAL intent
        raw_lower = {t.lower() for t in analysis.raw_terms}

        if raw_lower & _ANALYTICAL_SIGNALS:
            return _ANALYTICAL_CONFIG

        if raw_lower & _RISK_SIGNALS:
            return _RISK_CONFIG

        return _DEFAULT_CONFIG

    def describe(self, query: str) -> str:
        """Return a short description of the routing decision (for debugging)."""
        config = self.route(query)
        return (
            f"Intent: {config.description} | "
            f"dense={config.dense_limit}, sparse={config.sparse_limit}, "
            f"rrf_candidates={config.rrf_candidates}"
        )
