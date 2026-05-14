"""Query analysis primitives for retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from contextiq.retrieval.profile import RetrievalProfile


class QueryIntent(StrEnum):
    """Supported retrieval intents."""

    GENERAL = "general"
    CONTRACT = "contract"
    FINANCIAL_PERFORMANCE = "financial_performance"


@dataclass(frozen=True)
class QueryAnalysis:
    """Normalized query features used by candidate generation and ranking."""

    raw_terms: set[str]
    terms: set[str]
    intent: QueryIntent
    structured_codes: set[str]
    asset_modes: set[str]
    asset_terms: set[str]
    visual_terms: set[str]

    @property
    def mentions_gross_margin(self) -> bool:
        return bool({"gross", "margin"} & self.raw_terms)

    @property
    def mentions_risk(self) -> bool:
        return bool({"risk", "risks"} & self.raw_terms)

    @property
    def has_structured_codes(self) -> bool:
        return bool(self.structured_codes)

    @property
    def has_asset_mapping_query(self) -> bool:
        return bool(self.asset_modes and self.asset_terms)

    @property
    def has_visual_intent(self) -> bool:
        return bool(self.visual_terms)


class QueryAnalyzer:
    """Turn user questions into explicit retrieval features."""

    stopwords = {
        "company",
        "cite",
        "page",
        "pages",
        "main",
        "what",
        "are",
        "the",
        "and",
        "with",
    }

    def __init__(self, profile: RetrievalProfile | None = None) -> None:
        self.profile = profile or RetrievalProfile.default()
        self.structured_code_pattern = self._build_structured_code_pattern()

    def _build_structured_code_pattern(self) -> re.Pattern[str] | None:
        if not self.profile.structured_code_prefixes:
            return None
        prefixes = "|".join(
            re.escape(prefix.upper())
            for prefix in sorted(self.profile.structured_code_prefixes)
        )
        return re.compile(
            rf"\b(?:{prefixes})-[A-Z]-\d{{3}}(?:\s+[A-Z])?\b",
            re.IGNORECASE,
        )

    def analyze(self, query: str) -> QueryAnalysis:
        structured_codes = self.extract_structured_codes(query)
        asset_modes = self.extract_asset_modes(query)
        asset_terms = self.extract_asset_terms(query)
        visual_terms = self.extract_visual_terms(query)
        raw_terms = {
            self.normalize_term(term)
            for term in query.split()
            if self.normalize_term(term)
        }
        intent = self._detect_intent(raw_terms)
        terms = {term for term in raw_terms if len(term) > 3 and term not in self.stopwords}
        terms.update(structured_codes)
        terms.update(self._expanded_terms(raw_terms=raw_terms, intent=intent))
        return QueryAnalysis(
            raw_terms=raw_terms,
            terms=terms,
            intent=intent,
            structured_codes=structured_codes,
            asset_modes=asset_modes,
            asset_terms=asset_terms,
            visual_terms=visual_terms,
        )

    def normalize_term(self, term: str) -> str:
        normalized = term.strip(".,:;!?()[]{}\"'").lower()
        if normalized.endswith("'s"):
            normalized = normalized[:-2]
        return normalized

    def extract_structured_codes(self, query: str) -> set[str]:
        codes: set[str] = set()
        if self.structured_code_pattern is None:
            return codes
        for match in self.structured_code_pattern.finditer(query):
            code = re.sub(r"\s+", " ", match.group(0).strip()).lower()
            codes.add(code)
            if code.endswith(" l"):
                codes.add(code[:-2])
        return codes

    def extract_asset_modes(self, query: str) -> set[str]:
        normalized = f" {query.lower()} "
        return {
            mode
            for mode in self.profile.asset_modes
            if re.search(rf"(?<![a-z0-9]){re.escape(mode)}(?![a-z0-9])", normalized)
        }

    def extract_asset_terms(self, query: str) -> set[str]:
        normalized = " ".join(query.lower().split())
        return {asset for asset in self.profile.known_assets if asset in normalized}

    def extract_visual_terms(self, query: str) -> set[str]:
        raw_terms = {self.normalize_term(term) for term in query.split()}
        return raw_terms & self.profile.visual_markers

    def _detect_intent(self, raw_terms: set[str]) -> QueryIntent:
        if {"performance", "2025"} & raw_terms and (
            {"products", "services"} <= raw_terms or {"sales", "revenue"} & raw_terms
        ):
            return QueryIntent.FINANCIAL_PERFORMANCE
        if raw_terms & self.profile.contract_markers:
            return QueryIntent.CONTRACT
        return QueryIntent.GENERAL

    def _expanded_terms(
        self, raw_terms: set[str], intent: QueryIntent
    ) -> set[str]:
        terms: set[str] = set()
        if "ip" in raw_terms:
            terms.update({"intellectual", "property"})
        if "termination" in raw_terms:
            terms.add("terminate")
        if "confidentiality" in raw_terms:
            terms.add("confidential")
        if "obligations" in raw_terms:
            terms.add("must")
        if "parties" in raw_terms:
            terms.add("party")
        if "regulatory" in raw_terms:
            terms.update(self.profile.regulatory_expansion_terms)
        if "risks" in raw_terms or "risk" in raw_terms:
            terms.update(self.profile.risk_expansion_terms)
        if intent == QueryIntent.FINANCIAL_PERFORMANCE:
            product_terms = set(self.profile.product_markers)
            product_terms.update(self.profile.product_service_sections)
            comparison_terms = set(self.profile.financial_comparison_terms)
            terms.update(
                {
                    "net",
                    "sales",
                    "revenue",
                    "gross",
                    "margin",
                    "category",
                    "categories",
                    "change",
                    "services",
                    "products",
                }
            )
            terms.update(product_terms)
            terms.update(comparison_terms)
        return terms
