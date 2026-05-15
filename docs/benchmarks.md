# Benchmarks

Current retrieval baseline generated with:

```bash
uv run contextiq eval-retrieval --limit 20 --k 10
```

Dataset: `tests/evals/qrels/retrieval_seed.json`

| Metric | Current Baseline |
| --- | ---: |
| Queries | 12 |
| Recall@10 | 0.736 |
| Precision@10 | 0.233 |
| MRR | 0.735 |
| NDCG@10 | 0.853 |

## Interpretation

This is a seed retrieval eval that uses content-anchor qrels from an early two-document corpus. It is a regression guard for retrieval ranking and context assembly.

The larger local corpus used during development contains external financial filings and NASA technical documents. Large raw documents, processed blocks, and vector index state are intentionally excluded from the public repository.

Current manual smoke checks pass for the main demo flows: Apple product/service performance, Microsoft fiscal 2025 income metrics, NVIDIA fiscal 2025 results, NASA Data and Computing recommendations, exact UC-ID lookup, and Orion HLR asset-function retrieval.
