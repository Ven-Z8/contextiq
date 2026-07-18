# MMLongBench-Doc — Page-Level Retrieval Benchmark

> **Status: historical notes.** The marketed SPLADE+ColBERT enterprise stack and
> `make eval-mmlongbench` / `scripts/eval_mmlongbench.py` runner are not in the
> current tree. Keep this file for measured Phase-0/1 numbers; live eval today is
> FinanceBench via `evals/financebench/run_answers.py` (see `AGENTS.md`).

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

Absolute page-recall on short docs is dominated by a high random floor (taking 10 distinct pages of a
~15-23pp doc is 43-67% of it), so we report **lift over a uniform random page-picker** (`E[recall@k] = min(k/pages, 1)`).

| Run | Docs | Q | pages | Recall@5 | random@5 | **lift@5** | Recall@10 | random@10 | lift@10 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Legacy lexical, 3-doc subset | 3 | 18 | 23/23/15 | 0.79 | 0.26 | +0.53 | 0.87 | 0.53 | +0.35 |
| **Enterprise (SPLADE+ColBERT+adaptive), 3-doc subset** | 3 | 18 | 23/23/15 | **0.88** | 0.26 | **+0.62** | 0.92 | 0.53 | +0.39 |
| **Enterprise, 10-doc representative subset** (docs 15-112pp) | 10 | 64 | 15-112 | **0.67** | 0.19 | +0.48 | 0.79 | 0.37 | +0.42 |

**Phase 1 finding:** the marketed SPLADE+ColBERT+adaptive-chunking stack was dead code (#13/#14). Wiring it into
the eval (`--pipeline enterprise`) lifts the same 3-doc subset from **0.79 → 0.88@5** (lift +0.53 → +0.62) — real
retrieval gain from connecting the real architecture, with `.page` preserved through `search_enterprise`.

**lift@5 = +0.53** is genuine retrieval signal (0.79 vs 0.26 random). lift@10 is smaller (+0.35) because at k=10
on short docs the random floor alone is 0.53 — i.e. @10 here is mostly "the doc is short", which is exactly why
we lead with @5 lift. Run with `strict_vector_errors=True` and a per-question cross-doc isolation assertion
(0 failed docs, no leak), so the number is not propped up by silent vector→lexical fallback or contamination.

### The representative number is 0.67@5, not 0.88
The 3-doc subset (short Pew reports) inflated to 0.88. On a **10-doc representative set** including long docs
(up to 112pp), enterprise page-Recall@5 is **0.67** (lift +0.48 over random, 0 empty-result/fallback queries —
trustworthy). This is the honest current state: **+0.18 below the 0.85 target.** Phase 2 (retrieval-correctness:
intent biasing, AdaptiveChunker fixes, post-retrieval re-rank) must close the gap. Full 135-corpus run is #23.


### Diagnosis: where the misses are (10-doc, recall@5 by evidence source)
| Evidence source | n | Recall@5 |
|---|---:|---:|
| Chart | 22 | 0.79 |
| Pure-text | 11 | 0.78 |
| Multi-source | 14 | 0.62 |
| Table | 3 | 0.67 |
| **Figure** | 11 | **0.47** |
| **Generalized-text (Layout)** | 3 | **0.33** |

Misses concentrate on **Figure (0.47)** and **Layout (0.33)** — visual evidence the FAST text pipeline reduces to
`<figure>`/caption stubs. Text/Chart are already ~0.78, so **even perfect text-ranking caps the overall ≈ 0.79**;
reaching 0.85 requires real visual extraction (the `DoclingVLMExtractor` thesis). page-sum aggregation was tried
and *regressed* (0.67→0.63) — it dilutes precise single-block evidence; best-block-rank is kept.



### NEGATIVE RESULT: granite-docling VLM extraction is worse (do not pursue)
Tested the project's core thesis — swap the FAST standard extractor for the granite-docling-258M VLM (`--extractor vlm`)
on the same 10 docs. It **regressed**: page-Recall@5 **0.67 → 0.59**, and *worsened the visual categories it was meant to fix*
(Figure 0.47 → 0.28, Chart 0.79 → 0.69), while being ~10x slower (a 112-page doc took >15 min). A 258M VLM produces
lower-quality text than the dedicated TableFormer+layout pipeline. **The standard pipeline (0.67) is the better extractor.**
Implication: closing the visual gap needs either a far stronger extractor (e.g. Mistral OCR API) or **multimodal
page-image retrieval (ColPali)** — not a small local VLM. (MLX-runtime fix + harness `--extractor` option retained for future engines.)



### SIMPLIFICATION VALIDATED: ~50 LOC matches ~2,800 LOC
Head-to-head on the same 4 docs (25 answerable Q, 0 empty):
- **Simple** (bge-base-en-v1.5 dense + bge-reranker-base cross-encoder, ~50 LOC): page-Recall@5 **0.795**, @10 0.872
- **Enterprise** (SPLADE+ColBERT+RRF+7-profile adaptive chunker+intent router, ~2,800 LOC / 150 fns): @5 **0.808**, @10 0.888

A ~1-point gap = noise on n=25. The entire heavy stack buys nothing measurable -> **adopt the simple two-stage
pipeline and delete ~2,000 LOC.** The path to 0.85 is NOT architecture (both ~equal); it is a stronger embedder
(bge-large/Qwen3 — OOM'd the local Mac on the 112-page doc) and possibly visual retrieval, both needing more RAM/GPU
than this machine. Local hardware is the wall: heavy embedders can't hold large docs in memory here.

### Honest caveats
- **Short-doc subset, not representative.** 3 dataset-order Pew reports (~15-23pp vs corpus avg 47.5pp), n=18
  (high variance, no CI yet — see #23). The full-corpus number will be lower; long/table-heavy docs are where the
  live path fails (a real 10-K query for "net sales Products vs Services" returned litigation prose, not the table).
- **Legacy pipeline, not the marketed stack.** This measures the currently-wired lexical/heuristic path, **not**
  SPLADE+ColBERT (dead code, #13). Phase 1 wires the real pipeline, then we re-baseline.
- **FAST profile = no visual enrichment** (#25): chart-evidence recall here is carried by same-page prose, not the
  VLM pipeline. **Page-index alignment** (1-indexed, absolute across batches) is verified but not yet test-pinned (#24).
  The live path's section-dedup + parent-resolver slightly **deflate** the number (#26), so 0.79 is conservative.

### Published context (answer accuracy, not retrieval)
| System | MMLongBench-Doc answer F1 |
|---|---:|
| GPT-4o | 44.9% |
| GPT-4V | 31.4% |

_Retrieval-recall leaderboards for MMLongBench-Doc are not standardized; we report our page-level recall
transparently with the methodology above so it is reproducible and falsifiable._
