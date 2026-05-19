# Benchmarks

Current retrieval comparison generated with:

```bash
uv run contextiq eval-retrieval --limit 20 --k 10
```

Dataset: `tests/evals/qrels/retrieval_seed.json`

| Metric | Lexical-only baseline | ContextIQ |
| --- | ---: | ---: |
| Queries | 12 | 12 |
| Recall@10 | 0.556 | 0.736 |
| Precision@10 | 0.233 | 0.233 |
| MRR | 0.667 | 0.735 |
| NDCG@10 | 0.881 | 0.853 |

## Interpretation

This is a seed retrieval eval that uses content-anchor qrels from an early two-document corpus. It is a regression guard for retrieval ranking and context assembly. The lexical-only baseline uses the local lexical candidate generator without ContextIQ's full retrieval pipeline, expansion, and ranking path.

ContextIQ improves Recall@10 and MRR on this seed set while tying Precision@10. The lexical-only baseline has higher NDCG@10 on this small fixture, which is useful signal for future ranking work rather than something to hide.

The larger local corpus used during development contains external financial filings and NASA technical documents. Large raw documents, processed blocks, and vector index state are intentionally excluded from the public repository.

Current manual smoke checks pass for the main demo flows: Apple product/service performance, Microsoft fiscal 2025 income metrics, NVIDIA fiscal 2025 results, NASA Data and Computing recommendations, exact UC-ID lookup, and Orion HLR asset-function retrieval.
