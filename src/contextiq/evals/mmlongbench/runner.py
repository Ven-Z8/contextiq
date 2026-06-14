"""Run the MMLongBench-Doc page-level Recall@k baseline.

Each document is ingested into an ISOLATED local index (per-doc temp dir) so
retrieval is naturally scoped to that document — this avoids the broken
document_id payload filter in local Qdrant. We then query the live retrieval
path (LocalDocumentStore.search) and score page-level Recall@k, alongside a
uniform-random page-picker floor so the absolute number is read as LIFT, not in
isolation (short docs have a high random floor by construction).
"""

from __future__ import annotations

import logging
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from contextiq.evals.mmlongbench.dataset import load_docs
from contextiq.evals.mmlongbench.page_recall import page_recall_at_k

logger = logging.getLogger(__name__)


@dataclass
class MMLBResult:
    docs: int = 0
    answerable_questions: int = 0
    failed_docs: list[str] = field(default_factory=list)
    recall_at_5: list[float] = field(default_factory=list)
    recall_at_10: list[float] = field(default_factory=list)
    random_at_5: list[float] = field(default_factory=list)
    random_at_10: list[float] = field(default_factory=list)
    doc_pages: dict[str, int] = field(default_factory=dict)
    empty_result_queries: int = 0

    def summary(self) -> dict:
        def mean(xs: list[float]) -> float:
            return round(statistics.fmean(xs), 4) if xs else 0.0
        r5, r10 = mean(self.recall_at_5), mean(self.recall_at_10)
        f5, f10 = mean(self.random_at_5), mean(self.random_at_10)
        return {
            "docs": self.docs,
            "failed_docs": len(self.failed_docs),
            "answerable_questions": self.answerable_questions,
            "doc_page_counts": self.doc_pages,
            "empty_result_queries": self.empty_result_queries,
            "page_recall@5": r5,
            "random_floor@5": f5,
            "lift@5": round(r5 - f5, 4),
            "page_recall@10": r10,
            "random_floor@10": f10,
            "lift@10": round(r10 - f10, 4),
        }


def _ingest_isolated(pdf: Path, work: Path, pipeline: str = "legacy"):
    """Ingest one PDF into a store+index rooted at `work`, via explicit paths
    (no global os.environ mutation — paths are injected directly)."""
    from contextiq.ingestion.batch import BatchIngestor  # noqa: PLC0415
    from contextiq.ingestion.profiles import FAST  # noqa: PLC0415
    from contextiq.retrieval.store import LocalDocumentStore  # noqa: PLC0415
    from contextiq.retrieval.vector_index import VectorIndex  # noqa: PLC0415

    result = BatchIngestor(profile=FAST).ingest(pdf)
    store = LocalDocumentStore(
        path=work / "processed" / "blocks.json", strict_vector_errors=True
    )
    # Inject the index path directly instead of relying on settings/env.
    store.vector_index_factory = lambda: VectorIndex(path=work / "qdrant")
    store.save_blocks(result.blocks)
    if pipeline == "enterprise":
        from contextiq.ingestion.adaptive_chunker import AdaptiveChunker  # noqa: PLC0415
        store.index_blocks_enterprise(AdaptiveChunker().process_blocks(result.blocks))
    else:
        store.index_blocks(result.blocks)
    pages = [b.page for b in result.blocks if b.page is not None]
    page_count = max(pages) if pages else 0
    doc_id = result.blocks[0].document_id if result.blocks else ""
    return store, doc_id, page_count


@dataclass
class _DocScratch:
    r5: list[float] = field(default_factory=list)
    r10: list[float] = field(default_factory=list)
    f5: list[float] = field(default_factory=list)
    f10: list[float] = field(default_factory=list)


def evaluate(
    limit_docs: int | None = 3, blocks_per_query: int = 30, pipeline: str = "legacy"
) -> MMLBResult:
    docs = load_docs(limit_docs=limit_docs)
    res = MMLBResult()
    cache = Path(tempfile.gettempdir()) / "mmlb_pdfs"
    from contextiq.evals.mmlongbench.dataset import fetch_pdf  # noqa: PLC0415

    for doc in docs:
        res.docs += 1
        try:
            pdf = fetch_pdf(doc.doc_id, cache)
            with tempfile.TemporaryDirectory(prefix="mmlb_idx_") as tmp:
                store, ingested_id, page_count = _ingest_isolated(
                    pdf, Path(tmp), pipeline=pipeline
                )
                scratch = _DocScratch()
                for q in doc.questions:
                    if not q.answerable:
                        continue
                    blocks = (
                        store.search_enterprise(q.question, limit=blocks_per_query)
                        if pipeline == "enterprise"
                        else store.search(q.question, limit=blocks_per_query)
                    )
                    for b in blocks:  # isolation self-check
                        if b.document_id != ingested_id:
                            raise RuntimeError(
                                f"cross-doc leak: {b.document_id} != {ingested_id}"
                            )
                    if not blocks:
                        # search_enterprise silently falls back to an empty standard
                        # collection in enterprise mode — surface it, do not hide a 0.
                        res.empty_result_queries += 1
                        logger.warning("empty retrieval (possible silent fallback): %s",
                                       q.question[:60])
                    pages = [b.page for b in blocks]
                    r5 = page_recall_at_k(q.evidence_pages, pages, 5)
                    r10 = page_recall_at_k(q.evidence_pages, pages, 10)
                    if r5 is None or r10 is None:
                        continue
                    p = page_count or 1
                    scratch.r5.append(r5)
                    scratch.r10.append(r10)
                    scratch.f5.append(min(5 / p, 1.0))
                    scratch.f10.append(min(10 / p, 1.0))
            # merge only after the whole doc succeeded (no partial-recall pollution)
            res.answerable_questions += len(scratch.r5)
            res.recall_at_5.extend(scratch.r5)
            res.recall_at_10.extend(scratch.r10)
            res.random_at_5.extend(scratch.f5)
            res.random_at_10.extend(scratch.f10)
            res.doc_pages[doc.doc_id] = page_count
            logger.info("scored doc %s (%d pages)", doc.doc_id, page_count)
        except Exception as exc:
            logger.warning("doc %s failed (scores discarded): %s", doc.doc_id, exc)
            res.failed_docs.append(doc.doc_id)
    return res
