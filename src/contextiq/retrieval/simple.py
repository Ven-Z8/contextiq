"""Minimal two-stage retriever: one dense embedder + one cross-encoder.

Replaces the unablated SPLADE + ColBERT + RRF + ~150-function heuristic stack with
the documented two-stage SOTA shape — strong dense recall (top-N) then
cross-encoder reranking — using FastEmbed (already a dependency; no torch).
Uses query/passage asymmetric embedding (the BGE prefix the old path omitted).
"""

from __future__ import annotations

from pathlib import Path

from contextiq.ingestion.models import DocumentBlock


class SimpleRetriever:
    def __init__(
        self,
        *,
        qdrant_path: Path,
        embed_model: str = "BAAI/bge-large-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
        dim: int = 1024,
        collection: str = "simple",
    ) -> None:
        from fastembed import TextEmbedding  # noqa: PLC0415
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: PLC0415
        from qdrant_client import QdrantClient, models  # noqa: PLC0415

        self._models = models
        self._embed = TextEmbedding(embed_model)
        self._rerank = TextCrossEncoder(reranker_model)
        self.client = QdrantClient(path=str(qdrant_path))
        self.collection = collection
        self.client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        self._blocks: dict[str, DocumentBlock] = {}

    def index(self, blocks: list[DocumentBlock]) -> int:
        models = self._models
        vecs = list(self._embed.embed([b.text for b in blocks]))  # passage embeddings
        points = []
        for i, (b, v) in enumerate(zip(blocks, vecs, strict=True)):
            self._blocks[b.block_id] = b
            points.append(models.PointStruct(id=i, vector=v.tolist(),
                                             payload={"block_id": b.block_id}))
        for j in range(0, len(points), 256):
            self.client.upsert(self.collection, points[j:j + 256], wait=True)
        return len(points)

    def search(self, query: str, limit: int = 30) -> list[DocumentBlock]:
        qv = next(self._embed.query_embed(query))  # query embedding (BGE prefix applied)
        hits = self.client.query_points(
            collection_name=self.collection, query=qv.tolist(),
            limit=limit, with_payload=True,
        ).points
        cands = [self._blocks[h.payload["block_id"]] for h in hits]
        if not cands:
            return []
        scores = list(self._rerank.rerank(query, [c.text for c in cands]))
        order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
        return [cands[i] for i in order]
