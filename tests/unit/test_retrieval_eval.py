from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextiq.evals.retrieval import (
    ContentAnchor,
    RetrievalEvalCase,
    evaluate_ranked_blocks,
    evaluate_ranked_ids,
    load_qrels,
    run_retrieval_eval,
)
from contextiq.ingestion.models import DocumentBlock


def test_evaluate_ranked_ids_reports_retrieval_metrics() -> None:
    cases = [
        RetrievalEvalCase(
            id="q1",
            question="Where is the product revenue table?",
            relevant_blocks={"doc:table": 3, "doc:note": 1},
        ),
        RetrievalEvalCase(
            id="q2",
            question="What are termination rights?",
            relevant_blocks={"contract:termination": 3},
        ),
    ]
    ranked_by_query_id = {
        "q1": ["doc:wrong", "doc:table", "doc:note"],
        "q2": ["contract:unrelated"],
    }

    report = evaluate_ranked_ids(cases, ranked_by_query_id, k=3)

    assert report.query_count == 2
    assert report.aggregate["recall@3"] == pytest.approx(0.5)
    assert report.aggregate["precision@3"] == pytest.approx(1 / 3)
    assert report.aggregate["mrr"] == pytest.approx(0.25)
    assert 0 < report.aggregate["ndcg@3"] < 1
    assert report.queries[0].hit_block_ids == ["doc:table", "doc:note"]
    assert report.queries[1].missed_block_ids == ["contract:termination"]


def test_load_qrels_reads_seed_eval_cases(tmp_path: Path) -> None:
    qrels_path = tmp_path / "qrels.json"
    qrels_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "id": "contract_termination",
                        "question": "How can the agreement terminate?",
                        "relevant_blocks": {"sample-contract:4": 3},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = load_qrels(qrels_path)

    assert len(cases) == 1
    assert cases[0].id == "contract_termination"
    assert cases[0].relevant_blocks == {"sample-contract:4": 3}


def test_run_retrieval_eval_uses_retriever_output() -> None:
    cases = [
        RetrievalEvalCase(
            id="q1",
            question="Find confidentiality obligations.",
            relevant_blocks={"contract:confidentiality": 3},
        )
    ]

    def retrieve(_: str, __: int) -> list[DocumentBlock]:
        return [
            DocumentBlock(
                document_id="contract",
                block_id="contract:other",
                source_path="contract.md",
                text="Other clause.",
            ),
            DocumentBlock(
                document_id="contract",
                block_id="contract:confidentiality",
                source_path="contract.md",
                text="Each party must protect confidential information.",
            ),
        ]

    report = run_retrieval_eval(cases, retrieve=retrieve, limit=5, k=5)

    assert report.query_count == 1
    assert report.aggregate["recall@5"] == 1.0
    assert report.aggregate["mrr"] == pytest.approx(0.5)


def test_evaluate_ranked_blocks_matches_relevant_content_anchors() -> None:
    case = RetrievalEvalCase(
        id="apple_products_services",
        question="What is the Products and Services Performance in 2025?",
        relevant_blocks={},
        relevant_anchors=[
            ContentAnchor(
                id="apple_products_services_table",
                grade=3,
                text_contains=["Total net sales", "Services", "$ 416,161"],
                source_contains="apple-2025-10k.pdf",
                block_type="table",
            )
        ],
    )
    ranked_blocks = [
        DocumentBlock(
            document_id="apple-2025-10k-e8c0f8fec7",
            block_id="apple-2025-10k-e8c0f8fec7:325",
            source_path="data/raw/apple-2025-10k.pdf",
            block_type="table",
            text="| Services | 109,158 |\n| Total net sales | $ 416,161 |",
        )
    ]

    report = evaluate_ranked_blocks([case], {"apple_products_services": ranked_blocks}, k=5)

    assert report.aggregate["recall@5"] == 1.0
    assert report.aggregate["precision@5"] == pytest.approx(0.2)
    assert report.aggregate["mrr"] == 1.0
    assert report.queries[0].hit_block_ids == ["apple-2025-10k-e8c0f8fec7:325"]
    assert report.queries[0].missed_block_ids == []
