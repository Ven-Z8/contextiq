"""Contextual expansion for retrieval candidates."""

from __future__ import annotations

from collections.abc import Callable

from contextiq.ingestion.models import DocumentBlock
from contextiq.retrieval.candidates import RetrievalCandidate
from contextiq.retrieval.ranker import CandidateRanker


class SectionExpander:
    """Add nearby evidence around selected retrieval candidates."""

    def __init__(
        self,
        blocks_provider: Callable[[], list[DocumentBlock]],
        ranker: CandidateRanker | None = None,
    ) -> None:
        self.blocks_provider = blocks_provider
        self.ranker = ranker or CandidateRanker()

    def expand(
        self,
        candidates: list[DocumentBlock],
        limit: int,
        query: str | None = None,
    ) -> list[DocumentBlock]:
        traced_candidates = [
            RetrievalCandidate(block=candidate, stages=[])
            for candidate in candidates
        ]
        return [
            candidate.block
            for candidate in self.expand_with_trace(
                candidates=traced_candidates,
                limit=limit,
                query=query,
            )
        ]

    def expand_with_trace(
        self,
        candidates: list[RetrievalCandidate],
        limit: int,
        query: str | None = None,
    ) -> list[RetrievalCandidate]:
        if query:
            analysis = self.ranker.analyzer.analyze(query)
            if analysis.has_structured_codes or analysis.has_asset_mapping_query:
                return candidates[:limit]

        blocks = self.blocks_provider()
        by_id = {block.block_id: block for block in blocks}
        positions = {block.block_id: index for index, block in enumerate(blocks)}
        expanded: list[RetrievalCandidate] = []

        for candidate in candidates:
            if candidate.block.block_id not in by_id:
                continue
            self._append_same_section_prefix(
                expanded=expanded,
                blocks=blocks,
                positions=positions,
                candidate=candidate,
            )
            self._append_unique(expanded, candidate)
            self._append_same_section_suffix(
                expanded=expanded,
                blocks=blocks,
                positions=positions,
                candidate=candidate,
            )
            if candidate.block.block_type.value == "heading":
                self._append_heading_window(
                    expanded=expanded,
                    blocks=blocks,
                    positions=positions,
                    candidate=candidate,
                    query=query,
                )
            if len(expanded) >= limit:
                break

        return expanded[:limit]

    def _append_heading_window(
        self,
        expanded: list[RetrievalCandidate],
        blocks: list[DocumentBlock],
        positions: dict[str, int],
        candidate: RetrievalCandidate,
        query: str | None,
    ) -> None:
        start = positions[candidate.block.block_id] + 1
        window = self.ranker.heading_expansion_window(query, candidate.block)
        for neighbor in blocks[start : start + window]:
            if neighbor.document_id != candidate.block.document_id:
                break
            self._append_unique(
                expanded,
                RetrievalCandidate(block=neighbor, stages=["expansion"]),
            )
            if window <= 3 and neighbor.block_type.value != "heading":
                break

    def _append_same_section_prefix(
        self,
        expanded: list[RetrievalCandidate],
        blocks: list[DocumentBlock],
        positions: dict[str, int],
        candidate: RetrievalCandidate,
    ) -> None:
        section_key = self._section_key(candidate.block)
        if not section_key:
            return

        start = positions[candidate.block.block_id]
        for neighbor in reversed(blocks[max(0, start - 2) : start]):
            if neighbor.document_id != candidate.block.document_id:
                break
            if self._section_key(neighbor) != section_key:
                continue
            if neighbor.block_type.value == "heading":
                self._append_unique(
                    expanded,
                    RetrievalCandidate(block=neighbor, stages=["expansion"]),
                )

    def _append_same_section_suffix(
        self,
        expanded: list[RetrievalCandidate],
        blocks: list[DocumentBlock],
        positions: dict[str, int],
        candidate: RetrievalCandidate,
    ) -> None:
        section_key = self._section_key(candidate.block)
        if not section_key or candidate.block.block_type.value == "heading":
            return

        start = positions[candidate.block.block_id] + 1
        for neighbor in blocks[start : start + 2]:
            if neighbor.document_id != candidate.block.document_id:
                break
            if self._section_key(neighbor) != section_key:
                break
            self._append_unique(
                expanded,
                RetrievalCandidate(block=neighbor, stages=["expansion"]),
            )

    def _append_unique(
        self, candidates: list[RetrievalCandidate], candidate: RetrievalCandidate
    ) -> None:
        for existing in candidates:
            if existing.block.block_id != candidate.block.block_id:
                continue
            for stage in candidate.stages:
                existing.add_stage(stage)
            return
        candidates.append(candidate)

    def _section_key(self, block: DocumentBlock) -> tuple[str, ...]:
        return tuple(section.lower() for section in block.section_path)
