"""Load MMLongBench-Doc questions and PDFs from HuggingFace at runtime.

The dataset is CC-BY-NC 4.0 (research only), so we never vendor its content into
this repo — questions come from the datasets-server rows API and PDFs are fetched
on demand via huggingface_hub and cached locally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from contextiq.evals.mmlongbench.page_recall import parse_evidence_pages

logger = logging.getLogger(__name__)

_REPO = "yubo2333/MMLongBench-Doc"
_ROWS_API = "https://datasets-server.huggingface.co/rows"


@dataclass
class MMLBQuestion:
    doc_id: str
    question: str
    answer: str
    evidence_pages: set[int]
    evidence_sources: str
    answer_format: str

    @property
    def answerable(self) -> bool:
        return bool(self.evidence_pages)


@dataclass
class MMLBDoc:
    doc_id: str
    questions: list[MMLBQuestion] = field(default_factory=list)


def _fetch_rows(offset: int, length: int) -> list[dict]:
    resp = httpx.get(
        _ROWS_API,
        params={"dataset": _REPO, "config": "default", "split": "train",
                "offset": offset, "length": length},
        timeout=60,
    )
    resp.raise_for_status()
    return [r["row"] for r in resp.json().get("rows", [])]


def load_docs(limit_docs: int | None = None, max_scan: int = 100_000) -> list[MMLBDoc]:
    """Return docs (grouped questions) in dataset order, up to limit_docs.

    Pagination stops on the natural empty-rows break; max_scan is only a runaway
    guard and logs a warning if ever hit (so a grown dataset isn't silently truncated).
    """
    docs: dict[str, MMLBDoc] = {}
    order: list[str] = []
    offset = 0
    while offset < max_scan:
        rows = _fetch_rows(offset, 100)
        if not rows:
            break
        for row in rows:
            doc_id = row["doc_id"]
            if doc_id not in docs:
                if limit_docs is not None and len(order) >= limit_docs:
                    return [docs[d] for d in order]
                docs[doc_id] = MMLBDoc(doc_id=doc_id)
                order.append(doc_id)
            docs[doc_id].questions.append(
                MMLBQuestion(
                    doc_id=doc_id,
                    question=row["question"],
                    answer=row.get("answer", ""),
                    evidence_pages=parse_evidence_pages(row.get("evidence_pages")),
                    evidence_sources=row.get("evidence_sources", ""),
                    answer_format=row.get("answer_format", ""),
                )
            )
        offset += len(rows)
    return [docs[d] for d in order]


def fetch_pdf(doc_id: str, cache_dir: Path) -> Path:
    """Download one document PDF from HF (cached). doc_id is the filename."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    cache_dir.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        repo_id=_REPO, repo_type="dataset", filename=f"documents/{doc_id}",
        local_dir=str(cache_dir),
    )
    return Path(local)
