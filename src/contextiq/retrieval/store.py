"""Local JSON-backed document store with hybrid retrieval."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from hashlib import sha1
from pathlib import Path

from pydantic import TypeAdapter

from contextiq.core.config import get_settings
from contextiq.ingestion.hierarchy import HierarchyBuilder, ParentChunk
from contextiq.ingestion.models import DocumentBlock
from contextiq.retrieval.models import RetrievalHit
from contextiq.retrieval.vector_index import VectorIndex

logger = logging.getLogger(__name__)


class LocalDocumentStore:
    """Store parsed document blocks locally and retrieve them via hybrid search."""

    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        self.path = path or settings.data_dir / "processed" / "blocks.json"
        self.documents_dir = self.path.parent / "documents"
        self.parents_dir = self.path.parent / "parents"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.parents_dir.mkdir(parents=True, exist_ok=True)
        self._adapter = TypeAdapter(list[DocumentBlock])
        self._manifest_adapter = TypeAdapter(list[str])
        self._parent_adapter = TypeAdapter(list[ParentChunk])
        self._scoped_document_id: str | None = None
        self._document_cache: dict[str, list[DocumentBlock]] = {}
        self._parent_cache: dict[str, dict[str, ParentChunk]] = {}
        self._vector_index: VectorIndex | None = None
        self.hierarchy = HierarchyBuilder()
        self.vector_index_factory: Callable[[], VectorIndex] = VectorIndex

    def scoped(self, document_id: str | None) -> LocalDocumentStore:
        """Return a store view limited to one document when scoped.

        The clone shares the parent's VectorIndex so only one QdrantClient
        lock is held at a time (local Qdrant is single-writer).
        """
        if document_id is None:
            return self
        clone = LocalDocumentStore(path=self.path)
        clone._scoped_document_id = document_id
        clone._document_cache = self._document_cache
        clone._parent_cache = self._parent_cache
        clone._vector_index = self._vector_index
        return clone

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
            parents = self.hierarchy.build_parents(document_blocks)
            tagged_blocks = self.hierarchy.tag_blocks(document_blocks, parents)
            self._save_parents(document_id, parents)
            payload = [block.model_dump(mode="json") for block in tagged_blocks]
            self._document_path(document_id).write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
            self._document_cache[document_id] = tagged_blocks
            self._parent_cache[document_id] = {parent.parent_id: parent for parent in parents}

        self.path.write_text(json.dumps(list(grouped), indent=2), encoding="utf-8")

    def load_blocks(self, document_id: str | None = None) -> list[DocumentBlock]:
        scoped_id = document_id if document_id is not None else self._scoped_document_id
        document_ids = self._load_manifest()
        if not document_ids:
            blocks = self._load_legacy_blocks()
            return self._filter_blocks(blocks, scoped_id)

        blocks: list[DocumentBlock] = []
        target_ids = document_ids
        if scoped_id is not None:
            target_ids = [doc_id for doc_id in document_ids if doc_id == scoped_id]
        for doc_id in target_ids:
            blocks.extend(self._load_document_blocks(doc_id))
        return blocks

    def load_parent(self, parent_id: str) -> ParentChunk | None:
        document_id = parent_id.split(":", 1)[0]
        cached = self._parent_cache.get(document_id)
        if cached is not None:
            return cached.get(parent_id)

        parent_path = self._parent_path(document_id)
        if not parent_path.exists():
            return None
        parents = self._parent_adapter.validate_json(parent_path.read_text(encoding="utf-8"))
        parent_map = {parent.parent_id: parent for parent in parents}
        self._parent_cache[document_id] = parent_map
        return parent_map.get(parent_id)

    def _load_document_blocks(self, document_id: str) -> list[DocumentBlock]:
        cached = self._document_cache.get(document_id)
        if cached is not None:
            return cached

        document_path = self._document_path(document_id)
        if not document_path.exists():
            return []
        blocks = self._adapter.validate_json(document_path.read_text(encoding="utf-8"))
        self._document_cache[document_id] = blocks
        return blocks

    def _save_parents(self, document_id: str, parents: list[ParentChunk]) -> None:
        payload = [
            {
                **parent.__dict__,
                "block_ids": list(parent.block_ids),
            }
            for parent in parents
        ]
        self._parent_path(document_id).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        self._parent_cache[document_id] = {parent.parent_id: parent for parent in parents}

    def _filter_blocks(
        self,
        blocks: list[DocumentBlock],
        document_id: str | None,
    ) -> list[DocumentBlock]:
        if document_id is None:
            return blocks
        return [block for block in blocks if block.document_id == document_id]

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

    def _parent_path(self, document_id: str) -> Path:
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", document_id).strip("-")
        if not slug:
            slug = "document"
        digest = sha1(document_id.encode("utf-8")).hexdigest()[:10]
        return self.parents_dir / f"{slug[:80]}-{digest}.json"

    def hybrid_hits(self, query: str, limit: int = 40) -> list[RetrievalHit]:
        """The live retrieve: Qdrant hybrid dense+BM25 with RRF.

        Validated on FinanceBench (0.476 vs 0.19 naive-RAG). The long-context model
        absorbs generous top-k, so no reranker in v1 (add one only if the eval says
        recall is short). Falls back to a keyword scan only when the vector index is
        empty/unavailable (offline, cold corpus, tests) — a standalone safety net.
        """
        by_id = {block.block_id: block for block in self.load_blocks()}
        index = self._get_vector_index()
        results: list[RetrievalHit] = []
        if index is not None:
            try:
                hits = index.search_hybrid(
                    query=query,
                    limit=limit,
                    document_id=self._scoped_document_id,
                    group_by_section=True,
                )
                for rank, hit in enumerate(hits):
                    block = by_id.get(hit.block_id)
                    if block is not None:
                        results.append(
                            RetrievalHit(
                                block=block,
                                rank=rank,
                                score=hit.score,
                                stages=["hybrid"],
                                reason="hybrid dense+BM25 RRF",
                            )
                        )
            except Exception as exc:
                logger.warning("Hybrid search failed", exc_info=exc)
        if results:
            return results
        logger.warning("Vector index empty/unavailable; using lexical fallback")
        return self._lexical_fallback(query, limit)

    def _lexical_fallback(self, query: str, limit: int) -> list[RetrievalHit]:
        """Keyword-overlap scan for degraded operation (no/empty vector index)."""
        terms = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
        scored: list[tuple[int, DocumentBlock]] = []
        for block in self.load_blocks():
            haystack = (block.text + " " + " ".join(block.section_path)).lower()
            overlap = sum(1 for term in terms if term in haystack)
            if overlap:
                scored.append((overlap, block))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievalHit(
                block=block,
                rank=i,
                score=float(score),
                stages=["lexical"],
                reason="lexical fallback",
            )
            for i, (score, block) in enumerate(scored[:limit])
        ]

    def index_blocks(self, blocks: list[DocumentBlock]) -> int:
        index = self._get_vector_index()
        if index is None:
            return 0
        return index.index_blocks(blocks)

    def _get_vector_index(self) -> VectorIndex | None:
        if self._vector_index is not None:
            return self._vector_index
        try:
            self._vector_index = self.vector_index_factory()
        except RuntimeError as exc:
            logger.warning("Vector index unavailable", exc_info=exc)
            return None
        return self._vector_index

    def stats(self) -> dict[str, int]:
        blocks = self.load_blocks()
        return {
            "documents": len({block.document_id for block in blocks}),
            "blocks": len(blocks),
        }
