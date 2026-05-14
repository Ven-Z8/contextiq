"""Local JSON-backed document store for first-night iteration."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from hashlib import sha1
from pathlib import Path

from pydantic import TypeAdapter

from contextiq.core.config import get_settings
from contextiq.ingestion.models import DocumentBlock
from contextiq.retrieval.candidates import CandidateGenerator
from contextiq.retrieval.expansion import SectionExpander
from contextiq.retrieval.models import RetrievalHit
from contextiq.retrieval.pipeline import RetrievalPipeline
from contextiq.retrieval.query import QueryAnalyzer
from contextiq.retrieval.ranker import CandidateRanker
from contextiq.retrieval.vector_index import VectorIndex

logger = logging.getLogger(__name__)


class LocalDocumentStore:
    """Store parsed document blocks locally before vector indexing is wired in."""

    def __init__(self, path: Path | None = None, strict_vector_errors: bool = False) -> None:
        settings = get_settings()
        self.path = path or settings.data_dir / "processed" / "blocks.json"
        self.documents_dir = self.path.parent / "documents"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self._adapter = TypeAdapter(list[DocumentBlock])
        self._manifest_adapter = TypeAdapter(list[str])
        self.strict_vector_errors = strict_vector_errors
        self.vector_index_factory: Callable[[], VectorIndex] = VectorIndex
        self.query_analyzer = QueryAnalyzer()
        self.ranker = CandidateRanker(self.query_analyzer)
        self.candidate_generator = CandidateGenerator(
            blocks_provider=self.load_blocks,
            vector_search=self._search_vector,
            analyzer=self.query_analyzer,
            ranker=self.ranker,
        )
        self.section_expander = SectionExpander(
            blocks_provider=self.load_blocks,
            ranker=self.ranker,
        )
        self.retrieval_pipeline = RetrievalPipeline(
            generator=self.candidate_generator,
            expander=self.section_expander,
            ranker=self.ranker,
        )

    def save_blocks(self, blocks: list[DocumentBlock]) -> None:
        incoming_grouped = self._group_blocks_by_document(blocks)
        incoming_document_ids = set(incoming_grouped)
        existing_blocks = [
            block
            for block in self.load_blocks()
            if block.document_id not in incoming_document_ids
        ]
        grouped = self._group_blocks_by_document(existing_blocks)
        grouped.update(incoming_grouped)

        for document_id, document_blocks in grouped.items():
            payload = [block.model_dump(mode="json") for block in document_blocks]
            self._document_path(document_id).write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )

        self.path.write_text(json.dumps(list(grouped), indent=2), encoding="utf-8")

    def load_blocks(self) -> list[DocumentBlock]:
        document_ids = self._load_manifest()
        if not document_ids:
            return self._load_legacy_blocks()

        blocks: list[DocumentBlock] = []
        for document_id in document_ids:
            document_path = self._document_path(document_id)
            if not document_path.exists():
                continue
            blocks.extend(
                self._adapter.validate_json(document_path.read_text(encoding="utf-8"))
            )
        return blocks

    def _load_manifest(self) -> list[str]:
        if not self.path.exists():
            return []
        payload = self.path.read_text(encoding="utf-8")
        try:
            return self._manifest_adapter.validate_json(payload)
        except Exception:
            return []

    def _load_legacy_blocks(self) -> list[DocumentBlock]:
        if not self.path.exists():
            return []
        try:
            return self._adapter.validate_json(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _group_blocks_by_document(
        self, blocks: list[DocumentBlock]
    ) -> dict[str, list[DocumentBlock]]:
        grouped: dict[str, list[DocumentBlock]] = {}
        for block in blocks:
            grouped.setdefault(block.document_id, []).append(block)
        return grouped

    def _document_path(self, document_id: str) -> Path:
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", document_id).strip("-")
        if not slug:
            slug = "document"
        digest = sha1(document_id.encode("utf-8")).hexdigest()[:10]
        return self.documents_dir / f"{slug[:80]}-{digest}.json"

    def search(self, query: str, limit: int = 8) -> list[DocumentBlock]:
        return self.retrieval_pipeline.search(query=query, limit=limit)

    def search_with_trace(self, query: str, limit: int = 8) -> list[RetrievalHit]:
        return self.retrieval_pipeline.search_with_trace(query=query, limit=limit)

    def index_blocks(self, blocks: list[DocumentBlock]) -> int:
        return VectorIndex().index_blocks(blocks)

    def _search_vector(self, query: str, limit: int) -> list[DocumentBlock]:
        by_id = {block.block_id: block for block in self.load_blocks()}
        try:
            block_ids = self.vector_index_factory().search(query=query, limit=limit)
        except Exception as exc:
            message = "Vector search failed"
            if self.strict_vector_errors:
                raise RuntimeError(message) from exc
            logger.warning("%s; falling back to lexical search", message, exc_info=exc)
            return []
        return [by_id[block_id] for block_id in block_ids if block_id in by_id]

    def _search_lexical(self, query: str, limit: int) -> list[DocumentBlock]:
        return self.candidate_generator.lexical_candidates(query=query, limit=limit)

    def _expand_candidate_blocks(
        self,
        candidates: list[DocumentBlock],
        limit: int,
        query: str | None = None,
    ) -> list[DocumentBlock]:
        return self.section_expander.expand(
            candidates=candidates,
            limit=limit,
            query=query,
        )

    def _financial_anchor_candidates(
        self, query: str, limit: int
    ) -> list[DocumentBlock]:
        return self.candidate_generator.financial_anchor_candidates(
            query=query,
            limit=limit,
        )

    def _section_anchor_candidates(
        self, query: str, limit: int
    ) -> list[DocumentBlock]:
        return self.candidate_generator.section_anchor_candidates(
            query=query,
            limit=limit,
        )

    def _heading_expansion_window(
        self, query: str | None, heading: DocumentBlock
    ) -> int:
        return self.ranker.heading_expansion_window(query=query, heading=heading)

    def _rerank_candidates(
        self, query: str, candidates: list[DocumentBlock]
    ) -> list[DocumentBlock]:
        return self.ranker.rerank(query=query, candidates=candidates)

    def _apply_intent_precision(
        self, query: str, candidates: list[DocumentBlock]
    ) -> list[DocumentBlock]:
        return self.ranker.apply_intent_precision(query=query, candidates=candidates)

    def stats(self) -> dict[str, int]:
        blocks = self.load_blocks()
        return {"documents": len({block.document_id for block in blocks}), "blocks": len(blocks)}
