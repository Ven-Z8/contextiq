"""Retrieval evaluation metrics and qrels runner."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from contextiq.ingestion.models import DocumentBlock


class ContentAnchor(BaseModel):
    """Stable relevance target matched against retrieved block content."""

    id: str
    grade: int = Field(default=1, ge=1)
    text_contains: list[str] = Field(default_factory=list)
    source_contains: str | None = None
    document_id: str | None = None
    block_type: str | None = None

    @model_validator(mode="after")
    def require_matcher(self) -> ContentAnchor:
        """Require at least one non-ID condition so anchors are meaningful."""

        if (
            not self.text_contains
            and self.source_contains is None
            and self.document_id is None
            and self.block_type is None
        ):
            raise ValueError("content anchors need at least one match condition")
        return self


class RetrievalEvalCase(BaseModel):
    """A single retrieval eval query with graded relevant evidence targets."""

    id: str
    question: str
    relevant_blocks: dict[str, int] = Field(default_factory=dict)
    relevant_anchors: list[ContentAnchor] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_relevance_targets(self) -> RetrievalEvalCase:
        """Require at least one block ID or content anchor."""

        if not self.relevant_blocks and not self.relevant_anchors:
            raise ValueError("eval cases need relevant_blocks or relevant_anchors")
        return self


class RetrievalQueryResult(BaseModel):
    """Metrics and hit details for one eval query."""

    id: str
    question: str
    ranked_block_ids: list[str]
    hit_block_ids: list[str]
    missed_block_ids: list[str]
    metrics: dict[str, float]


class RetrievalEvalReport(BaseModel):
    """Aggregate retrieval eval report."""

    query_count: int
    aggregate: dict[str, float]
    queries: list[RetrievalQueryResult]

    def to_markdown(self) -> str:
        """Render a compact markdown report for CLI and portfolio docs."""

        lines = [
            "# Retrieval Evaluation",
            "",
            f"Queries: {self.query_count}",
            "",
            "## Aggregate Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
        for metric, value in self.aggregate.items():
            lines.append(f"| {metric} | {value:.3f} |")

        lines.extend(["", "## Query Results", ""])
        for result in self.queries:
            lines.extend(
                [
                    f"### {result.id}",
                    f"Question: {result.question}",
                    f"Hits: {', '.join(result.hit_block_ids) or '-'}",
                    f"Missed: {', '.join(result.missed_block_ids) or '-'}",
                    "",
                ]
            )
        return "\n".join(lines).strip()


class _QrelsFile(BaseModel):
    queries: list[RetrievalEvalCase]


def load_qrels(path: Path) -> list[RetrievalEvalCase]:
    """Load retrieval qrels from a JSON file."""

    qrels = _QrelsFile.model_validate_json(path.read_text(encoding="utf-8"))
    return qrels.queries


def run_retrieval_eval(
    cases: list[RetrievalEvalCase],
    retrieve: Callable[[str, int], list[DocumentBlock]],
    limit: int = 20,
    k: int = 10,
) -> RetrievalEvalReport:
    """Run a retriever for each eval query and score ranked block IDs."""

    ranked_by_query_id = {
        case.id: retrieve(case.question, limit)
        for case in cases
    }
    return evaluate_ranked_blocks(cases, ranked_by_query_id, k=k)


def evaluate_ranked_ids(
    cases: list[RetrievalEvalCase],
    ranked_by_query_id: dict[str, list[str]],
    k: int = 10,
) -> RetrievalEvalReport:
    """Score ranked retrieval results with Recall@k, Precision@k, MRR, and NDCG@k."""

    query_results = [
        _evaluate_case(case, ranked_by_query_id.get(case.id, []), k=k)
        for case in cases
    ]
    aggregate = _aggregate_metrics([result.metrics for result in query_results])
    return RetrievalEvalReport(
        query_count=len(cases),
        aggregate=aggregate,
        queries=query_results,
    )


def evaluate_ranked_blocks(
    cases: list[RetrievalEvalCase],
    ranked_by_query_id: dict[str, list[DocumentBlock]],
    k: int = 10,
) -> RetrievalEvalReport:
    """Score ranked retrieval results with block IDs and stable content anchors."""

    query_results = [
        _evaluate_case_blocks(case, ranked_by_query_id.get(case.id, []), k=k)
        for case in cases
    ]
    aggregate = _aggregate_metrics([result.metrics for result in query_results])
    return RetrievalEvalReport(
        query_count=len(cases),
        aggregate=aggregate,
        queries=query_results,
    )


def _evaluate_case(
    case: RetrievalEvalCase,
    ranked_block_ids: list[str],
    k: int,
) -> RetrievalQueryResult:
    relevant_ids = set(case.relevant_blocks)
    top_k = ranked_block_ids[:k]
    hit_block_ids = [block_id for block_id in top_k if block_id in relevant_ids]
    missed_block_ids = [
        block_id for block_id in case.relevant_blocks if block_id not in hit_block_ids
    ]
    metrics = {
        f"recall@{k}": _recall_at_k(top_k, relevant_ids),
        f"precision@{k}": _precision_at_k(top_k, relevant_ids, k),
        "mrr": _reciprocal_rank(top_k, relevant_ids),
        f"ndcg@{k}": _ndcg_at_k(top_k, case.relevant_blocks, k),
    }
    return RetrievalQueryResult(
        id=case.id,
        question=case.question,
        ranked_block_ids=ranked_block_ids,
        hit_block_ids=hit_block_ids,
        missed_block_ids=missed_block_ids,
        metrics=metrics,
    )


def _evaluate_case_blocks(
    case: RetrievalEvalCase,
    ranked_blocks: list[DocumentBlock],
    k: int,
) -> RetrievalQueryResult:
    top_k = ranked_blocks[:k]
    matched_targets: set[str] = set()
    hit_block_ids: list[str] = []
    relevance_by_rank: list[int] = []

    for block in top_k:
        block_grade, block_targets = _matched_relevance_targets(case, block)
        relevance_by_rank.append(block_grade)
        if block_targets:
            matched_targets.update(block_targets)
            hit_block_ids.append(block.block_id)

    target_grades = _target_grades(case)
    missed_targets = [
        target_id for target_id in target_grades if target_id not in matched_targets
    ]
    relevant_count = len(target_grades)
    hit_count = len(matched_targets)
    metrics = {
        f"recall@{k}": hit_count / relevant_count if relevant_count else 0.0,
        f"precision@{k}": len([grade for grade in relevance_by_rank if grade > 0]) / k
        if k > 0
        else 0.0,
        "mrr": _reciprocal_rank_by_relevance(relevance_by_rank),
        f"ndcg@{k}": _ndcg_from_relevances(relevance_by_rank, list(target_grades.values()), k),
    }
    return RetrievalQueryResult(
        id=case.id,
        question=case.question,
        ranked_block_ids=[block.block_id for block in ranked_blocks],
        hit_block_ids=hit_block_ids,
        missed_block_ids=missed_targets,
        metrics=metrics,
    )


def _aggregate_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    metric_names = metrics[0].keys()
    return {
        name: sum(item[name] for item in metrics) / len(metrics)
        for name in metric_names
    }


def _recall_at_k(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    return len([block_id for block_id in ranked_ids if block_id in relevant_ids]) / len(
        relevant_ids
    )


def _precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len([block_id for block_id in ranked_ids[:k] if block_id in relevant_ids]) / k


def _reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for index, block_id in enumerate(ranked_ids, start=1):
        if block_id in relevant_ids:
            return 1 / index
    return 0.0


def _reciprocal_rank_by_relevance(relevances: list[int]) -> float:
    for index, relevance in enumerate(relevances, start=1):
        if relevance > 0:
            return 1 / index
    return 0.0


def _ndcg_at_k(ranked_ids: list[str], relevance_by_id: dict[str, int], k: int) -> float:
    dcg = _dcg([relevance_by_id.get(block_id, 0) for block_id in ranked_ids[:k]])
    ideal_relevances = sorted(relevance_by_id.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal_relevances)
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def _ndcg_from_relevances(
    ranked_relevances: list[int],
    ideal_relevances: list[int],
    k: int,
) -> float:
    dcg = _dcg(ranked_relevances[:k])
    ideal_dcg = _dcg(sorted(ideal_relevances, reverse=True)[:k])
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def _dcg(relevances: list[int]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def _target_grades(case: RetrievalEvalCase) -> dict[str, int]:
    target_grades = dict(case.relevant_blocks)
    target_grades.update({anchor.id: anchor.grade for anchor in case.relevant_anchors})
    return target_grades


def _matched_relevance_targets(
    case: RetrievalEvalCase,
    block: DocumentBlock,
) -> tuple[int, set[str]]:
    matches: set[str] = set()
    grade = 0
    if block.block_id in case.relevant_blocks:
        matches.add(block.block_id)
        grade = max(grade, case.relevant_blocks[block.block_id])
    for anchor in case.relevant_anchors:
        if _matches_anchor(block, anchor):
            matches.add(anchor.id)
            grade = max(grade, anchor.grade)
    return grade, matches


def _matches_anchor(block: DocumentBlock, anchor: ContentAnchor) -> bool:
    if anchor.document_id is not None and block.document_id != anchor.document_id:
        return False
    if anchor.block_type is not None and block.block_type != anchor.block_type:
        return False
    if anchor.source_contains is not None and anchor.source_contains not in block.source_path:
        return False

    block_text = block.text.casefold()
    return all(fragment.casefold() in block_text for fragment in anchor.text_contains)


RetrievalEvalCaseList = TypeAdapter(list[RetrievalEvalCase])


def write_qrels(path: Path, cases: list[RetrievalEvalCase]) -> None:
    """Write qrels in the same shape accepted by load_qrels."""

    payload = {"queries": RetrievalEvalCaseList.dump_python(cases, mode="json")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
