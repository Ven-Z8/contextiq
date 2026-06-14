from __future__ import annotations

from contextiq.evals.mmlongbench.page_recall import page_recall_at_k, parse_evidence_pages


def test_parse_evidence_pages_handles_bracketed_lists() -> None:
    assert parse_evidence_pages("[5]") == {5}
    assert parse_evidence_pages("[5, 7, 12]") == {5, 7, 12}
    assert parse_evidence_pages("[]") == set()
    assert parse_evidence_pages("") == set()


def test_page_recall_dedupes_pages_in_rank_order_and_caps_at_k() -> None:
    # ranked block pages (rank order); distinct top-3 pages = [5, 8, 9]
    ranked = [5, 5, 8, 5, 9, 11]
    # gold {5, 9}: page 5 in top-3, page 9 in top-3 -> recall 2/2 = 1.0
    assert page_recall_at_k({5, 9}, ranked, k=3) == 1.0
    # k=2 distinct pages = [5, 8]; gold {5,9}: only 5 hit -> 1/2 = 0.5
    assert page_recall_at_k({5, 9}, ranked, k=2) == 0.5


def test_page_recall_ignores_none_pages_and_empty_gold() -> None:
    assert page_recall_at_k({3}, [None, None, 3], k=5) == 1.0
    # empty gold (unanswerable) -> None (scored separately, not as retrieval recall)
    assert page_recall_at_k(set(), [1, 2], k=5) is None


def test_parse_evidence_pages_rejects_noncanonical_and_keeps_positive() -> None:
    # only positive page ints; no negative leakage from a stray hyphen
    assert parse_evidence_pages("[10, 10]") == {10}
    assert parse_evidence_pages("[1, 2, 3]") == {1, 2, 3}
    assert parse_evidence_pages(None) == set()
