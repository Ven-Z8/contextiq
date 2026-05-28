"""End-to-end retrieval pipeline orchestration."""

from __future__ import annotations

from contextiq.ingestion.models import DocumentBlock
from contextiq.retrieval.candidates import CandidateGenerator, RetrievalCandidate
from contextiq.retrieval.expansion import SectionExpander
from contextiq.retrieval.models import RetrievalHit
from contextiq.retrieval.parent_resolver import ParentResolver
from contextiq.retrieval.ranker import CandidateRanker


class RetrievalPipeline:
    """Generate, expand, rerank, and filter retrieval candidates."""

    def __init__(
        self,
        generator: CandidateGenerator,
        expander: SectionExpander,
        ranker: CandidateRanker,
        parent_resolver: ParentResolver | None = None,
    ) -> None:
        self.generator = generator
        self.expander = expander
        self.ranker = ranker
        self.parent_resolver = parent_resolver

    def search(self, query: str, limit: int = 8) -> list[DocumentBlock]:
        return [hit.block for hit in self.search_with_trace(query=query, limit=limit)]

    def search_with_trace(self, query: str, limit: int = 8) -> list[RetrievalHit]:
        candidate_limit = max(limit * 3, 24)
        candidates = self.generator.generate_with_trace(query=query, limit=candidate_limit)
        expanded = self.expander.expand_with_trace(
            candidates=candidates,
            limit=candidate_limit * 3,
            query=query,
        )
        blocks_by_id = {candidate.block.block_id: candidate for candidate in expanded}
        ranked = self.ranker.rerank(
            query=query,
            candidates=[candidate.block for candidate in expanded],
        )
        filtered = self._dedupe_sections(
            self.ranker.apply_intent_precision(query=query, candidates=ranked)
        )
        analysis = self.ranker.analyzer.analyze(query)
        if self.parent_resolver is not None and not (
            analysis.has_structured_codes or analysis.has_asset_mapping_query
        ):
            filtered = self.parent_resolver.enrich_candidates(filtered)

        hits: list[RetrievalHit] = []
        for rank, block in enumerate(filtered[:limit], start=1):
            candidate = blocks_by_id.get(block.block_id)
            if candidate is None:
                stage = "parent" if block.metadata.get("chunk_level") == "parent" else "reranking"
                candidate = RetrievalCandidate(block=block, stages=[stage])
            score = self.ranker.score(analysis, block)
            hits.append(
                RetrievalHit(
                    block=block,
                    rank=rank,
                    score=score,
                    stages=candidate.stages,
                    reason=self._reason(block=block, stages=candidate.stages, score=score),
                )
            )
        return hits

    def _dedupe_sections(self, candidates: list[DocumentBlock]) -> list[DocumentBlock]:
        """Keep the strongest block per section when metadata is available."""

        deduped: list[DocumentBlock] = []
        seen_sections: set[str] = set()
        for block in candidates:
            section_id = block.metadata.get("section_id")
            if not isinstance(section_id, str) or not section_id:
                deduped.append(block)
                continue
            if section_id in seen_sections:
                continue
            seen_sections.add(section_id)
            deduped.append(block)
        return deduped

    def _reason(self, block: DocumentBlock, stages: list[str], score: float) -> str:
        stage_text = ", ".join(stages) if stages else "reranking"
        visual = ""
        if block.block_type.value in {"figure", "table"}:
            visual = f" {block.block_type.value} evidence"
        return f"{stage_text} selected this{visual} block; reranker score {score:.2f}"
