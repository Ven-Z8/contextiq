"""Configurable retrieval vocabulary and scoring profile."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocumentTitleAlias:
    """Map a set of query terms to source aliases."""

    terms: set[str]
    aliases: set[str]


@dataclass(frozen=True)
class RetrievalProfile:
    """Domain vocabulary used by query analysis and reranking.

    The defaults are intentionally generic. Corpus-specific terms should live in
    ``config/retrieval_profile.json`` instead of retrieval code.
    """

    source_aliases: dict[str, set[str]] = field(default_factory=dict)
    document_title_aliases: list[DocumentTitleAlias] = field(default_factory=list)
    topic_anchors: set[str] = field(default_factory=set)
    asset_modes: set[str] = field(default_factory=set)
    known_assets: set[str] = field(default_factory=set)
    structured_code_prefixes: set[str] = field(default_factory=set)
    visual_markers: set[str] = field(default_factory=set)
    contract_markers: set[str] = field(default_factory=set)
    regulatory_expansion_terms: set[str] = field(default_factory=set)
    risk_expansion_terms: set[str] = field(default_factory=set)
    structured_source_markers: set[str] = field(default_factory=set)
    financial_anchor_headings: set[str] = field(default_factory=set)
    product_service_anchor_headings: set[str] = field(default_factory=set)
    product_service_sections: set[str] = field(default_factory=set)
    product_markers: set[str] = field(default_factory=set)
    financial_comparison_terms: set[str] = field(default_factory=set)
    financial_metrics: set[str] = field(
        default_factory=lambda: {
            "revenue",
            "net sales",
            "gross margin",
            "operating income",
            "net income",
            "diluted earnings per share",
        }
    )
    financial_noise_sections: set[str] = field(
        default_factory=lambda: {
            "risk",
            "cash flows",
            "operating expenses",
            "income taxes",
            "liquidity and capital resources",
            "debt",
            "leases",
            "purchase obligations",
        }
    )
    financial_risk_markers: set[str] = field(
        default_factory=lambda: {
            "risk factors",
            "subject to volatility",
            "downward pressure",
            "adversely affect",
            "material adverse",
        }
    )
    low_information_frontmatter: set[str] = field(default_factory=set)

    @classmethod
    def default(cls, path: Path | None = None) -> RetrievalProfile:
        profile_path = path or Path("config/retrieval_profile.json")
        if not profile_path.exists():
            return cls()
        return cls.from_json(profile_path.read_text(encoding="utf-8"))

    @classmethod
    def from_json(cls, payload: str) -> RetrievalProfile:
        data = json.loads(payload)
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> RetrievalProfile:
        return cls(
            source_aliases={
                cls._normalize(term): {cls._normalize(alias) for alias in aliases}
                for term, aliases in data.get("source_aliases", {}).items()
            },
            document_title_aliases=[
                DocumentTitleAlias(
                    terms={cls._normalize(term) for term in item.get("terms", [])},
                    aliases={cls._normalize(alias) for alias in item.get("aliases", [])},
                )
                for item in data.get("document_title_aliases", [])
            ],
            topic_anchors=cls._normalized_set(data.get("topic_anchors", [])),
            asset_modes=cls._normalized_set(data.get("asset_modes", [])),
            known_assets=cls._normalized_set(data.get("known_assets", [])),
            structured_code_prefixes=cls._normalized_set(
                data.get("structured_code_prefixes", [])
            ),
            visual_markers=cls._normalized_set(data.get("visual_markers", [])),
            contract_markers=cls._normalized_set(data.get("contract_markers", [])),
            regulatory_expansion_terms=cls._normalized_set(
                data.get("regulatory_expansion_terms", [])
            ),
            risk_expansion_terms=cls._normalized_set(data.get("risk_expansion_terms", [])),
            structured_source_markers=cls._normalized_set(
                data.get("structured_source_markers", [])
            ),
            financial_anchor_headings=cls._normalized_set(
                data.get("financial_anchor_headings", [])
            ),
            product_service_anchor_headings=cls._normalized_set(
                data.get("product_service_anchor_headings", [])
            ),
            product_service_sections=cls._normalized_set(
                data.get("product_service_sections", [])
            ),
            product_markers=cls._normalized_set(data.get("product_markers", [])),
            financial_comparison_terms=cls._normalized_set(
                data.get("financial_comparison_terms", [])
            ),
            financial_metrics=cls._normalized_set(
                data.get(
                    "financial_metrics",
                    [
                        "revenue",
                        "net sales",
                        "gross margin",
                        "operating income",
                        "net income",
                        "diluted earnings per share",
                    ],
                )
            ),
            financial_noise_sections=cls._normalized_set(
                data.get(
                    "financial_noise_sections",
                    [
                        "risk",
                        "cash flows",
                        "operating expenses",
                        "income taxes",
                        "liquidity and capital resources",
                        "debt",
                        "leases",
                        "purchase obligations",
                    ],
                )
            ),
            financial_risk_markers=cls._normalized_set(
                data.get(
                    "financial_risk_markers",
                    [
                        "risk factors",
                        "subject to volatility",
                        "downward pressure",
                        "adversely affect",
                        "material adverse",
                    ],
                )
            ),
            low_information_frontmatter=cls._normalized_set(
                data.get("low_information_frontmatter", [])
            ),
        )

    @staticmethod
    def _normalized_set(values: list[str]) -> set[str]:
        return {RetrievalProfile._normalize(value) for value in values}

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(str(value).lower().split())
