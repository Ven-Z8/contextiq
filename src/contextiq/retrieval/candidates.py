"""Candidate generation for retrieval."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from contextiq.ingestion.models import DocumentBlock
from contextiq.retrieval.query import QueryAnalyzer, QueryIntent
from contextiq.retrieval.ranker import CandidateRanker


@dataclass
class RetrievalCandidate:
    """A first-pass retrieval candidate with source-stage trace."""

    block: DocumentBlock
    stages: list[str] = field(default_factory=list)

    def add_stage(self, stage: str) -> None:
        if stage not in self.stages:
            self.stages.append(stage)


class CandidateGenerator:
    """Collect first-pass candidates from vector, lexical, and structural signals."""

    def __init__(
        self,
        blocks_provider: Callable[[], list[DocumentBlock]],
        vector_search: Callable[[str, int], list[DocumentBlock]],
        analyzer: QueryAnalyzer | None = None,
        ranker: CandidateRanker | None = None,
    ) -> None:
        self.blocks_provider = blocks_provider
        self.vector_search = vector_search
        self.analyzer = analyzer or QueryAnalyzer()
        self.ranker = ranker or CandidateRanker(self.analyzer)

    def generate(self, query: str, limit: int) -> list[DocumentBlock]:
        return [candidate.block for candidate in self.generate_with_trace(query, limit)]

    def generate_with_trace(self, query: str, limit: int) -> list[RetrievalCandidate]:
        analysis = self.analyzer.analyze(query)
        if analysis.has_structured_codes:
            return self._tagged(self.structured_code_candidates(query, limit), "structured_code")
        if analysis.has_asset_mapping_query:
            return self._tagged(self.asset_mapping_candidates(query, limit), "asset_mapping")

        candidates: list[RetrievalCandidate] = []
        if analysis.has_visual_intent:
            candidates.extend(
                self._tagged(
                    self.visual_candidates(query, limit),
                    "visual",
                )
            )
        if analysis.intent == QueryIntent.FINANCIAL_PERFORMANCE:
            candidates.extend(
                self._tagged(
                    self.financial_metric_table_candidates(query, limit),
                    "financial_metric_table",
                )
            )
        candidates.extend(self._tagged(self.vector_search(query, limit), "vector"))
        candidates.extend(self._tagged(self.lexical_candidates(query, limit), "lexical"))
        candidates.extend(
            self._tagged(
                self.section_anchor_candidates(query, limit),
                "section_anchor",
            )
        )
        if analysis.intent == QueryIntent.FINANCIAL_PERFORMANCE:
            candidates.extend(
                self._tagged(
                    self.financial_anchor_candidates(query, limit),
                    "financial_anchor",
                )
            )
        return self._dedupe(candidates)

    def lexical_candidates(self, query: str, limit: int) -> list[DocumentBlock]:
        terms = self.analyzer.analyze(query).terms
        scored: list[tuple[int, int, DocumentBlock]] = []
        for block in self.blocks_provider():
            text = " ".join([block.text, *block.section_path]).lower()
            score = sum(1 for term in terms if term in text)
            if score > 0:
                heading_penalty = 1 if block.block_type.value == "heading" else 0
                scored.append((score, heading_penalty, block))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return [block for _, _, block in scored[:limit]]

    def structured_code_candidates(self, query: str, limit: int) -> list[DocumentBlock]:
        analysis = self.analyzer.analyze(query)
        scored: list[tuple[int, DocumentBlock]] = []
        for block in self.blocks_provider():
            text = self.ranker.block_text_with_sections(block)
            matches = sum(1 for code in analysis.structured_codes if code in text)
            if matches:
                scored.append((matches, block))
        scored.sort(
            key=lambda item: (
                item[0],
                item[1].block_type.value == "table",
                self.ranker.has_structured_source_marker(item[1]),
            ),
            reverse=True,
        )
        return [block for _, block in scored[:limit]]

    def asset_mapping_candidates(self, query: str, limit: int) -> list[DocumentBlock]:
        analysis = self.analyzer.analyze(query)
        scored: list[tuple[float, DocumentBlock]] = []
        for block in self.blocks_provider():
            if not self.ranker.matches_asset_mapping(analysis, block):
                continue
            scored.append((self.ranker.asset_mapping_bonus(analysis, block), block))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [block for _, block in scored[:limit]]

    def visual_candidates(self, query: str, limit: int) -> list[DocumentBlock]:
        analysis = self.analyzer.analyze(query)
        if not analysis.has_visual_intent:
            return []

        scored: list[tuple[float, DocumentBlock]] = []
        for block in self.blocks_provider():
            score = self.ranker.visual_evidence_bonus(analysis, block)
            if score > 0:
                scored.append((score, block))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [block for _, block in scored[:limit]]

    def section_anchor_candidates(self, query: str, limit: int) -> list[DocumentBlock]:
        terms = self.analyzer.analyze(query).terms
        scored: list[tuple[int, DocumentBlock]] = []
        for block in self.blocks_provider():
            if block.block_type.value != "heading":
                continue
            section_text = self.ranker.block_text_with_sections(block)
            score = sum(1 for term in terms if term in section_text)
            if score > 0:
                scored.append((score, block))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [block for _, block in scored[:limit]]

    def financial_anchor_candidates(self, query: str, limit: int) -> list[DocumentBlock]:
        analysis = self.analyzer.analyze(query)
        if analysis.intent != QueryIntent.FINANCIAL_PERFORMANCE:
            return []

        anchored: list[DocumentBlock] = []
        for block in self.blocks_provider():
            if block.block_type.value != "heading":
                continue
            if not self.ranker.is_financial_anchor_heading(block):
                continue
            self._append_unique(anchored, block)
            if len(anchored) >= limit:
                break
        return anchored[:limit]

    def financial_metric_table_candidates(self, query: str, limit: int) -> list[DocumentBlock]:
        analysis = self.analyzer.analyze(query)
        if analysis.intent != QueryIntent.FINANCIAL_PERFORMANCE:
            return []

        source_aliases = self.ranker.source_aliases(analysis)
        scored: list[tuple[float, DocumentBlock]] = []
        for block in self.blocks_provider():
            if block.block_type.value != "table":
                continue
            if source_aliases and not any(
                alias in self.ranker.source_identity_text(block) for alias in source_aliases
            ):
                continue
            text = self.ranker.block_text_with_sections(block)
            if "diluted" in analysis.raw_terms and "diluted" not in text:
                continue
            if {"net", "income"} <= analysis.raw_terms and "net income" not in text:
                continue
            metric_score = sum(
                weight
                for metric, weight in self._financial_metric_weights().items()
                if metric in text
            )
            term_score = sum(1.0 for term in analysis.terms if term in text)
            score = metric_score + term_score
            if score >= 5.0:
                scored.append((score, block))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [block for _, block in scored[:limit]]

    def _tagged(
        self, blocks: list[DocumentBlock], stage: str
    ) -> list[RetrievalCandidate]:
        return [RetrievalCandidate(block=block, stages=[stage]) for block in blocks]

    def _dedupe(self, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        by_id: dict[str, RetrievalCandidate] = {}
        for candidate in candidates:
            existing = by_id.get(candidate.block.block_id)
            if existing is None:
                by_id[candidate.block.block_id] = candidate
                continue
            for stage in candidate.stages:
                existing.add_stage(stage)
        return list(by_id.values())

    def _append_unique(self, blocks: list[DocumentBlock], block: DocumentBlock) -> None:
        if all(existing.block_id != block.block_id for existing in blocks):
            blocks.append(block)

    def _financial_metric_weights(self) -> dict[str, float]:
        weights = {metric: 3.0 for metric in self.ranker.profile.financial_metrics}
        if "diluted earnings per share" in weights:
            weights["diluted earnings per share"] = 2.0
        weights["2025"] = 1.0
        return weights
