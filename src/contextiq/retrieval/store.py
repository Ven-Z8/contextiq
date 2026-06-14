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
from contextiq.ingestion.hierarchy import HierarchyBuilder, ParentChunk
from contextiq.ingestion.models import BlockType, DocumentBlock
from contextiq.retrieval.candidates import CandidateGenerator
from contextiq.retrieval.expansion import SectionExpander
from contextiq.retrieval.models import RetrievalHit
from contextiq.retrieval.parent_resolver import ParentResolver
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
        self.parents_dir = self.path.parent / "parents"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.parents_dir.mkdir(parents=True, exist_ok=True)
        self._adapter = TypeAdapter(list[DocumentBlock])
        self._manifest_adapter = TypeAdapter(list[str])
        self._parent_adapter = TypeAdapter(list[ParentChunk])
        self.strict_vector_errors = strict_vector_errors
        self._scoped_document_id: str | None = None
        self._document_cache: dict[str, list[DocumentBlock]] = {}
        self._parent_cache: dict[str, dict[str, ParentChunk]] = {}
        self._vector_index: VectorIndex | None = None
        self.hierarchy = HierarchyBuilder()
        self.vector_index_factory: Callable[[], VectorIndex] = VectorIndex
        self.query_analyzer = QueryAnalyzer()
        self.ranker = CandidateRanker(self.query_analyzer)
        self.candidate_generator = CandidateGenerator(
            blocks_provider=self.load_blocks,
            vector_search=self._search_vector,
            sparse_search=self._search_sparse,
            analyzer=self.query_analyzer,
            ranker=self.ranker,
        )
        self.section_expander = SectionExpander(
            blocks_provider=self.load_blocks,
            ranker=self.ranker,
        )
        self.parent_resolver = ParentResolver(parent_loader=self.load_parent)
        self.retrieval_pipeline = RetrievalPipeline(
            generator=self.candidate_generator,
            expander=self.section_expander,
            ranker=self.ranker,
            parent_resolver=self.parent_resolver,
        )

    def scoped(self, document_id: str | None) -> LocalDocumentStore:
        """Return a store view limited to one document when scoped.

        The clone shares the parent's VectorIndex so only one QdrantClient
        lock is held at a time (local Qdrant is single-writer).
        """
        if document_id is None:
            return self
        clone = LocalDocumentStore(path=self.path, strict_vector_errors=self.strict_vector_errors)
        clone._scoped_document_id = document_id
        clone._document_cache = self._document_cache
        clone._parent_cache = self._parent_cache
        # Share the parent's vector index — local Qdrant allows only one lock holder
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

    def search(self, query: str, limit: int = 8) -> list[DocumentBlock]:
        return self.retrieval_pipeline.search(query=query, limit=limit)

    def search_with_trace(self, query: str, limit: int = 8) -> list[RetrievalHit]:
        return self.retrieval_pipeline.search_with_trace(query=query, limit=limit)

    def index_blocks(self, blocks: list[DocumentBlock]) -> int:
        index = self._get_vector_index()
        if index is None:
            return 0
        return index.index_blocks(blocks)

    def index_blocks_enterprise(self, blocks: list[DocumentBlock]) -> int:
        """Index blocks into the enterprise 3-vector collection (SPLADE + ColBERT).

        Use this for full enterprise-grade retrieval with:
          - SPLADE neural sparse (term expansion)
          - ColBERT late-interaction reranking (MaxSim)
          - Intent-aware prefetch configuration

        Returns number of blocks indexed.
        """
        index = self._get_vector_index()
        if index is None:
            return 0
        return index.index_blocks_enterprise(blocks)

    def search_enterprise_with_scores(
        self,
        query: str,
        limit: int = 8,
    ) -> list[tuple[DocumentBlock, float]]:
        """Like search_enterprise but returns (block, colbert_score) pairs."""
        index = self._get_vector_index()
        if index is None:
            return []

        scoped_blocks = self.load_blocks()
        by_id = {block.block_id: block for block in scoped_blocks}

        try:
            hits = index.search_hybrid_with_reranking(
                query=query,
                limit=limit,
                document_id=self._scoped_document_id,
            )
        except Exception as exc:
            logger.warning("Enterprise search failed", exc_info=exc)
            return []

        results: list[tuple[DocumentBlock, float]] = []
        for h in hits:
            block = None
            if h.block_id in by_id:
                block = by_id[h.block_id]
                if h.payload:
                    cp = h.payload.get("content_profile")
                    if cp and not block.metadata.get("content_profile"):
                        block = block.model_copy(update={
                            "metadata": {**block.metadata, "content_profile": cp}
                        })
            elif h.payload:
                block = self._reconstruct_block_from_payload(h.payload)
            if block is not None:
                results.append((block, h.score))
        return results

    def search_enterprise(
        self,
        query: str,
        limit: int = 8,
    ) -> list[DocumentBlock]:
        """Search using the full enterprise pipeline (dense + SPLADE → RRF → ColBERT).

        Reconstructs adaptive sub-chunks directly from the Qdrant payload when
        the block_id is not present in the local JSON store (i.e., the block was
        created by AdaptiveChunker and only lives in the enterprise collection).

        Falls back to legacy search if enterprise collection is not indexed.
        """
        index = self._get_vector_index()
        if index is None:
            return []

        scoped_blocks = self.load_blocks()
        by_id = {block.block_id: block for block in scoped_blocks}

        try:
            hits = index.search_hybrid_with_reranking(
                query=query,
                limit=limit,
                document_id=self._scoped_document_id,
            )
        except Exception as exc:
            logger.warning("Enterprise search failed; falling back to legacy", exc_info=exc)
            return self._search_vector(query, limit)

        if not hits:
            logger.debug("Enterprise search returned no hits; falling back to legacy")
            return self._search_vector(query, limit)

        results: list[DocumentBlock] = []
        for h in hits:
            if h.block_id in by_id:
                # Original block found — enrich with content_profile from enterprise payload
                block = by_id[h.block_id]
                if h.payload:
                    cp = h.payload.get("content_profile")
                    if cp and not block.metadata.get("content_profile"):
                        block = block.model_copy(update={
                            "metadata": {**block.metadata, "content_profile": cp}
                        })
                results.append(block)
            elif h.payload:
                # Adaptive sub-chunk — not in JSON store, reconstruct from Qdrant payload
                block = self._reconstruct_block_from_payload(h.payload)
                if block is not None:
                    results.append(block)

        return results

    def _reconstruct_block_from_payload(self, payload: dict) -> DocumentBlock | None:
        """Reconstruct a DocumentBlock from a Qdrant enterprise payload.

        Used when an adaptive sub-chunk's block_id is not present in the
        local JSON store. The enterprise payload carries all necessary fields.
        """
        try:
            # section_path was stored as "A > B > C" — reconstruct as list
            section_path_raw = payload.get("section_path", "")
            section_path = (
                [p.strip() for p in section_path_raw.split(" > ") if p.strip()]
                if section_path_raw
                else []
            )

            # block_type was stored as .value string
            block_type_raw = payload.get("block_type", "text")
            try:
                block_type = BlockType(block_type_raw)
            except ValueError:
                block_type = BlockType.TEXT

            # Reconstruct metadata
            metadata: dict = {}
            for key in ("content_profile", "chunk_strategy", "parent_block_id",
                        "chunk_index", "section_id", "parent_id", "chunk_level"):
                val = payload.get(key)
                if val is not None:
                    metadata[key] = val

            return DocumentBlock(
                document_id=payload["document_id"],
                block_id=payload["block_id"],
                source_path=payload.get("source_path", ""),
                page=payload.get("page"),
                section_path=section_path,
                block_type=block_type,
                text=payload.get("block_text_searchable", ""),
                metadata=metadata,
            )
        except Exception as exc:
            logger.debug("Failed to reconstruct block from payload: %s", exc)
            return None

    def enterprise_status(self) -> dict:
        """Return enterprise collection status for CLI/API diagnostics."""
        index = self._get_vector_index()
        if index is None:
            return {"available": False, "reason": "vector index unavailable"}
        info = index.enterprise_collection_info()
        if info is None:
            return {"available": False, "reason": "enterprise collection not indexed"}
        return {"available": True, **info}

    def _get_vector_index(self) -> VectorIndex | None:
        if self._vector_index is not None:
            return self._vector_index
        try:
            self._vector_index = self.vector_index_factory()
        except RuntimeError as exc:
            logger.warning("Vector index unavailable", exc_info=exc)
            return None
        return self._vector_index

    def _search_vector(self, query: str, limit: int) -> list[DocumentBlock]:
        scoped_blocks = self.load_blocks()
        by_id = {block.block_id: block for block in scoped_blocks}
        index = self._get_vector_index()
        if index is None:
            return []
        try:
            hits = index.search_hybrid(
                query=query,
                limit=limit,
                document_id=self._scoped_document_id,
                group_by_section=True,
            )
            block_ids = [hit.block_id for hit in hits]
        except Exception as exc:
            message = "Vector search failed"
            if self.strict_vector_errors:
                raise RuntimeError(message) from exc
            logger.warning("%s; falling back to lexical search", message, exc_info=exc)
            return []
        return [by_id[block_id] for block_id in block_ids if block_id in by_id]

    def _search_sparse(self, query: str, limit: int) -> list[DocumentBlock]:
        scoped_blocks = self.load_blocks()
        by_id = {block.block_id: block for block in scoped_blocks}
        index = self._get_vector_index()
        if index is None:
            return []
        try:
            block_ids = index.search_sparse(
                query=query,
                limit=limit,
                document_id=self._scoped_document_id,
            )
        except Exception as exc:
            logger.warning("Sparse search failed; falling back to scan", exc_info=exc)
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

    def _section_anchor_candidates(
        self, query: str, limit: int
    ) -> list[DocumentBlock]:
        return self.candidate_generator.section_anchor_candidates(
            query=query,
            limit=limit,
        )

    def _rerank_candidates(
        self, query: str, candidates: list[DocumentBlock]
    ) -> list[DocumentBlock]:
        return self.ranker.rerank(query=query, candidates=candidates)

    def stats(self) -> dict[str, int]:
        blocks = self.load_blocks()
        return {"documents": len({block.document_id for block in blocks}), "blocks": len(blocks)}
