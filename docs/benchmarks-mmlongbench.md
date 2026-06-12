# MMLongBench-Doc — Page-Level Retrieval Benchmark

> **Headline metric (PRD):** page-level **Recall@5 ≥ 0.85** on MMLongBench-Doc, reproducible from a clean clone.

## What this measures

MMLongBench-Doc ([paper](https://arxiv.org/abs/2407.01523), [HF](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc))
is 135 long PDFs (avg 47.5 pages) with 1,082 questions, each annotated with the **evidence page number(s)**
that contain the answer (text/table/chart/figure/layout). We score *retrieval*, not answer generation:

- For each answerable question, retrieve blocks via the live retrieval path, map each block to its source
  page, dedupe pages in rank order, take the top-k distinct **pages**, and compute
  `Recall@k = |gold_pages ∩ top_k_pages| / |gold_pages|`.
- Unanswerable questions (empty evidence) are excluded from retrieval recall (scored separately later).
- Each document is ingested into an **isolated** local index so retrieval is naturally scoped to it
  (the `document_id` payload filter is a no-op in local Qdrant — see issue #17).

This is page-level **retrieval** recall — a different, upstream quantity from MMLongBench's published
**answer** F1 (where GPT-4o scores 44.9%). We never conflate the two; answer quality is reported separately.

## Reproducing

```bash
make eval-mmlongbench DOCS=3      # subset; DOCS=-1 for the full 135-doc corpus
```

The harness downloads questions (datasets-server API) and PDFs (`huggingface_hub`) at runtime. The dataset is
**CC-BY-NC 4.0 (research only)**, so no dataset content is vendored into this repo — only the eval code.

## Results

| Run | Docs | Answerable Q | page-Recall@5 | page-Recall@10 |
|---|---:|---:|---:|---:|
| **Baseline (legacy live path), 3-doc subset** — 2026-06-12 | 3 | 18 | **0.79** | 0.87 |
| Full 135-doc corpus | — | — | _TODO_ | _TODO_ |

### Honest caveats on the 0.79 baseline
- **Not representative.** The 3 docs are short Pew-research reports (few pages, chart-heavy) where top-5 page
  recall is easy. On long, table-heavy docs (e.g. a 10-K) the live path retrieves poorly — a real query for
  "net sales Products vs Services" returned litigation prose, not the table. **The full-corpus number will be
  materially lower.**
- **Tiny sample** (n=18 questions) → high variance.
- **Legacy pipeline.** This measures the currently-wired lexical/heuristic path, **not** the marketed
  SPLADE+ColBERT enterprise stack, which is dead code (issue #13). Phase 1 wires the real pipeline; we then
  re-baseline.

### Published context (answer accuracy, not retrieval)
| System | MMLongBench-Doc answer F1 |
|---|---:|
| GPT-4o | 44.9% |
| GPT-4V | 31.4% |

_Retrieval-recall leaderboards for MMLongBench-Doc are not standardized; we report our page-level recall
transparently with the methodology above so it is reproducible and falsifiable._
