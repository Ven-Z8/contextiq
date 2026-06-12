"""Run the MMLongBench-Doc page-level Recall@k baseline.

Each document is ingested into an ISOLATED local index (per-doc temp dir) so
retrieval is naturally scoped to that document — this avoids the broken
document_id payload filter in local Qdrant. We then query the live retrieval
path (LocalDocumentStore.search) and score page-level Recall@k.
"""

from __future__ import annotations

import logging
import os
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

    def summary(self) -> dict[str, float | int]:
        def mean(xs: list[float]) -> float:
            return round(statistics.fmean(xs), 4) if xs else 0.0
        return {
            "docs": self.docs,
            "answerable_questions": self.answerable_questions,
            "failed_docs": len(self.failed_docs),
            "page_recall@5": mean(self.recall_at_5),
            "page_recall@10": mean(self.recall_at_10),
        }


def _ingest_isolated(pdf: Path, work: Path):
    """Ingest one PDF into an isolated store+index rooted at `work`."""
    os.environ["CONTEXTIQ_DATA_DIR"] = str(work)
    os.environ["CONTEXTIQ_QDRANT_PATH"] = str(work / "qdrant")
    from contextiq.ingestion.batch import BatchIngestor  # noqa: PLC0415
    from contextiq.ingestion.profiles import FAST  # noqa: PLC0415
    from contextiq.retrieval.store import LocalDocumentStore  # noqa: PLC0415

    result = BatchIngestor(profile=FAST).ingest(pdf)
    store = LocalDocumentStore()
    store.save_blocks(result.blocks)
    store.index_blocks(result.blocks)
    return store


def evaluate(limit_docs: int | None = 3, blocks_per_query: int = 30) -> MMLBResult:
    docs = load_docs(limit_docs=limit_docs)
    res = MMLBResult()
    cache = Path(tempfile.gettempdir()) / "mmlb_pdfs"
    from contextiq.evals.mmlongbench.dataset import fetch_pdf  # noqa: PLC0415

    for doc in docs:
        res.docs += 1
        try:
            pdf = fetch_pdf(doc.doc_id, cache)
            with tempfile.TemporaryDirectory(prefix="mmlb_idx_") as tmp:
                store = _ingest_isolated(pdf, Path(tmp))
                for q in doc.questions:
                    if not q.answerable:
                        continue
                    blocks = store.search(q.question, limit=blocks_per_query)
                    pages = [b.page for b in blocks]
                    r5 = page_recall_at_k(q.evidence_pages, pages, 5)
                    r10 = page_recall_at_k(q.evidence_pages, pages, 10)
                    if r5 is None:
                        continue
                    res.answerable_questions += 1
                    res.recall_at_5.append(r5)
                    res.recall_at_10.append(r10)
            logger.info("scored doc %s", doc.doc_id)
        except Exception as exc:
            logger.warning("doc %s failed: %s", doc.doc_id, exc)
            res.failed_docs.append(doc.doc_id)
    return res
