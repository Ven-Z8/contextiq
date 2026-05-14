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

This is a legacy Apple/NASA seed eval that still uses content-anchor qrels from the two-document phase. The live portfolio corpus has grown to five real external documents, so this number is now a regression guard rather than the final portfolio benchmark. It intentionally exposes two next eval tasks: refresh qrels for the five-document corpus and add answer-grounded checks for Microsoft, NVIDIA, and the NASA Data and Computing Architecture Study.

The live portfolio corpus currently contains 5 real external documents and 7,428 blocks: Apple 2025 Form 10-K, NASA Moon to Mars lunar objective decomposition workbook, Microsoft FY25 Q4 Form 10-K, NVIDIA 2025 Annual Report, and NASA Science Mission Directorate Data and Computing Architecture Study.

Current manual smoke checks pass for the main demo flows: Apple product/service performance, Microsoft fiscal 2025 income metrics, NVIDIA fiscal 2025 results, NASA Data and Computing recommendations, exact UC-ID lookup, and Orion HLR asset-function retrieval.
