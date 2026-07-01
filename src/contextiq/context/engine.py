"""Context engineering engine."""

from __future__ import annotations

import re

from contextiq.context.models import ContextPacket, ContextSource
from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.retrieval.query import QueryAnalysis, QueryAnalyzer
from contextiq.retrieval.store import LocalDocumentStore
from contextiq.utils.tokens import TokenCounter


class ContextEngine:
    """Build token-budgeted context packets."""

    def __init__(self, store: LocalDocumentStore, token_budget: int = 6_000) -> None:
        self.store = store
        self.token_budget = token_budget
        self.encoder = TokenCounter()
        self.analyzer = QueryAnalyzer()

    def build_context(self, question: str, limit: int = 24) -> ContextPacket:
        analysis = self.analyzer.analyze(question)
        # ponytail: new hybrid retrieve (store.hybrid_hits) is built + e2e-proven alongside
        # this legacy path. Cut over here + delete the legacy stack once FinanceBench (Phase 3)
        # shows hybrid_hits wins — same "build-new-alongside-old, delete when it wins" rule.
        hits = self.store.search_with_trace(question, limit=limit)
        sources: list[ContextSource] = []
        used_tokens = 0
        seen_structured_excerpts: set[str] = set()

        for hit in hits:
            block = self._trim_structured_code_table(hit.block, analysis.structured_codes)
            block = self._trim_asset_mapping_table(block, analysis)
            if analysis.has_structured_codes:
                excerpt_key = self._normalized_excerpt_key(block.text)
                if excerpt_key in seen_structured_excerpts:
                    continue
                seen_structured_excerpts.add(excerpt_key)
            estimated = len(self.encoder.encode(block.text))
            if used_tokens + estimated > self.token_budget:
                continue
            sources.append(
                ContextSource(
                    block=block,
                    estimated_tokens=estimated,
                    reason=hit.reason,
                    stages=hit.stages,
                    score=hit.score,
                )
            )
            used_tokens += estimated

        return ContextPacket(
            question=question,
            sources=sources,
            token_budget=self.token_budget,
            used_tokens=used_tokens,
            dropped_candidates=max(len(hits) - len(sources), 0),
        )

    def _trim_structured_code_table(
        self,
        block: DocumentBlock,
        structured_codes: set[str],
    ) -> DocumentBlock:
        if not structured_codes or block.block_type != BlockType.TABLE:
            return block

        lines = block.text.splitlines()
        has_separator = (
            len(lines) >= 2
            and set(lines[1].replace("|", "").strip()) <= {"-", " "}
        )
        header = lines[:2] if has_separator else lines[:1]
        matching_rows = [
            line
            for line in lines[len(header) :]
            if any(code in line.lower() for code in structured_codes)
        ]
        if not matching_rows:
            return block

        metadata = dict(block.metadata)
        metadata["excerpted_for_structured_code"] = True
        return block.model_copy(
            update={
                "text": "\n".join([*header, *matching_rows]),
                "metadata": metadata,
            }
        )

    def _trim_asset_mapping_table(
        self,
        block: DocumentBlock,
        analysis: QueryAnalysis,
    ) -> DocumentBlock:
        if not analysis.has_asset_mapping_query or block.block_type != BlockType.TABLE:
            return block

        lines = block.text.splitlines()
        start_index = self._find_asset_mapping_start(lines, analysis)
        if start_index is None:
            return block

        header = self._table_header(lines, start_index)
        excerpt_rows: list[str] = []
        for line in lines[start_index:]:
            cells = self._table_cells(line)
            if excerpt_rows and self._contains_asset_id(cells):
                break
            excerpt_rows.append(line)

        metadata = dict(block.metadata)
        metadata["excerpted_for_asset_mapping"] = True
        return block.model_copy(
            update={
                "text": "\n".join([*header, *excerpt_rows]),
                "metadata": metadata,
            }
        )

    def _find_asset_mapping_start(
        self,
        lines: list[str],
        analysis: QueryAnalysis,
    ) -> int | None:
        for index, line in enumerate(lines):
            row_text = " ".join(self._table_cells(line)).lower()
            if not any(asset in row_text for asset in analysis.asset_terms):
                continue
            if any(f"-{mode}" in row_text for mode in analysis.asset_modes):
                return index
        return None

    def _table_cells(self, line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    def _table_header(self, lines: list[str], start_index: int) -> list[str]:
        if len(lines) >= 3 and "asset id" in lines[2].lower():
            return [lines[2], lines[1]]
        if (
            len(lines) >= 2
            and set(lines[1].replace("|", "").strip()) <= {"-", " "}
        ):
            return lines[:2]
        return lines[: min(start_index, 1)]

    def _contains_asset_id(self, cells: list[str]) -> bool:
        return any(re.search(r"\bEM-\d{3}-[A-Z]+\b", cell) for cell in cells)

    def _normalized_excerpt_key(self, text: str) -> str:
        return " ".join(text.lower().split())
