# Corpus Doc-QA Rebuild — Implementation Plan (ponytail:full)

> Execute task-by-task; check boxes as you go. Ponytail governs: reuse installed deps, delete more than you add, shortest working diff, add levers only when the eval says they're short.

**Goal:** A reliable QA bot over a *corpus* of 150–200pp documents that beats naive-RAG's 19% on FinanceBench and grounds every answer with citations.

**Vendor decision (per Venkat):** one vendor — **OpenRouter** — for both parse and answer. Parse = `mistral-ocr` engine ($2/1k pages; free `cloudflare-ai` engine for dev). Answer = `nvidia/nemotron-3-ultra-550b-a55b:free` (1M ctx, $0). Call OpenRouter over **httpx** (NOT the OpenAI SDK — CLAUDE.md hard rule). Keep the existing `AnthropicLLMClient` as an optional seam for airtight Citations-API grounding.

**Architecture (lazy corpus variant):**
`parse (OpenRouter mistral-ocr) → chunk ~800tok → qdrant hybrid dense+BM25 (installed) → retrieve top ~40 → Nemotron answer with chunk-ID citations + abstain gate.`
**No reranker in v1** — the free 1M-ctx model absorbs generous retrieval. Reranker, contextual-prefixing, and per-doc long-context are **named, not built** (P5).

**Tech stack:** reuse `anthropic` (seam), `qdrant-client[fastembed]`, `docling` (parse fallback), `httpx`. New = REST calls only. No new Python deps.

**Known trade-offs (accepted):**
- Chunk-ID citations, not Anthropic char-span citations — standard RAG grounding, slightly less airtight.
- Free tier: ~20 req/min, 50 req/day (→1,000/day with $10 credits — needed for the 150-Q eval). Some free providers require logging opt-in.

**Verified facts driving this:**
- FinanceBench: naive shared-store RAG = **19%**, long-context = **79%**, oracle = **85%** (arXiv 2311.11944).
- Financial Touchstone: **66.5% of failures are retrieval failures** → retrieval quality + abstain matter most.
- OpenRouter `mistral-ocr` = $2/1k pages; `cloudflare-ai` engine = free ([docs](https://openrouter.ai/docs/features/multimodal/pdfs)).
- Nemotron 3 Ultra free: $0, 1M ctx (200K on free route) ([model](https://openrouter.ai/nvidia/nemotron-3-ultra-550b-a55b:free)).

---

## What we DELETE (once P3's number beats baseline; grep for callers first)

- `src/contextiq/retrieval/`: `candidates.py`, `ranker.py`, `expansion.py`, `parent_resolver.py`, `intent_router.py`, `profile.py`, `pipeline.py`
- `src/contextiq/ingestion/adaptive_chunker.py`, `ingestion/profiles.py`
- SPLADE/ColBERT/`search_hybrid_with_reranking`/`index_blocks_enterprise` paths in `vector_index.py` + their `store.py` wrappers
- Demo hacks in `context/engine.py` (`_trim_asset_mapping_table`, `_trim_structured_code_table`, `EM-###`/`CN-P-103`)
- Drop `UPGRADE_PLAN.md`; rewrite `README.md` to match reality.

~2,000 LOC out. **Delete is a task, not a footnote.**

---

## Phase 1 — Answer path: OpenRouter/Nemotron + chunk-ID citations + abstain

Smallest testable slice. Works on a handful of chunks before corpus retrieval exists.

**Files:** new `llm/openrouter_client.py` (~40 LOC, httpx `LLMClient`); `llm/answerer.py` (select provider via env; parse `[chunk_N]` cites; gate); `llm/prompts.py` (citation + abstain instructions); `core/config.py` (`llm_provider`, `openrouter_api_key`, `openrouter_model`); `tests/unit/test_answerer_grounding.py`.

- [ ] `OpenRouterLLMClient.generate(...)` — httpx POST `https://openrouter.ai/api/v1/chat/completions`, `Authorization: Bearer`, model from env. Return the same `LLMResult`. No OpenAI SDK.
- [ ] Prompt: number the context chunks `[chunk_1..N]`; instruct "cite the chunk id(s) you used inline like `[chunk_3]`; if the answer isn't in the context, reply exactly `NOT_IN_DOCUMENT`."
- [ ] `answerer.py`: pick client by `CONTEXTIQ_LLM_PROVIDER` (default `openrouter`; `anthropic` = Citations-API seam; no key = `ExtractiveFallbackClient`). Parse cited chunk ids → map to source page/path on `GroundedAnswer`.
- [ ] Test: 2 fake chunks. Assert (a) in-context question returns text + a `[chunk_N]` that maps to the right source, (b) out-of-context question returns `NOT_IN_DOCUMENT`. One `test_*.py`, no fixtures.
- [ ] Commit.

**Skipped:** streaming, multi-turn, Anthropic span citations (seam kept).

---

## Phase 2 — one clean retrieve (SHIPPED, cutover staged)

- [x] **retrieve = `store.hybrid_hits(query, k=40)`** — Qdrant hybrid dense+BM25 via `vector_index.search_hybrid`, no rerank. Tested (`tests/unit/test_hybrid_retrieve.py`).
- [x] **End-to-end proven**: ingest `sample-contract.md` → `hybrid_hits` (9 sources) → **Nemotron** grounded answer with a real Confidence/Gaps block; empty-context path correctly returns `NOT_IN_DOCUMENT`.
- [~] **Parse: deferred, not needed for v1.** `loader.load(pdf)` already runs Docling (standard TableFormer — was the best extractor at 0.67, no VLM OOM). Hosted OCR is a P5 lever gated on the eval: OpenRouter `mistral-ocr` markdown extraction is flaky per field reports (annotations live on the Responses API, not chat/completions); if tables are the miss, wire Mistral's **direct** OCR API.
- [~] **Cutover deferred (build-new-alongside-old).** `build_context` still calls legacy `search_with_trace`; flip it to `hybrid_hits` **and** delete the legacy stack together once Phase 3 shows hybrid wins — avoids migrating ~8 legacy-coupled tests (`test_context_engine`, `test_api`) before the number justifies it. Suite is at baseline (only 9 pre-existing enterprise-stack failures; new work adds zero).

---

## Phase 3 — FinanceBench answer-accuracy eval (the gate)

**Files:** port/point the recall harness (main checkout `evals/financebench/`) into this tree; add `evals/financebench/run_answers.py`.

- [ ] **First: retry wrapper on `OpenRouterLLMClient`** (2-3 attempts on transient 429/5xx). Free-tier Nemotron intermittently fails and `_generate_safely` silently returns the extractive fallback — that would silently corrupt a 150-question eval (observed once in the Phase 2 e2e).
- [ ] Add answer-accuracy scoring (LLM-as-judge vs gold `answer`; keep evidence-recall). Reuse its PDF download + ingest. Point the eval at `store.hybrid_hits` so it measures the new retrieve vs the legacy path (build-new-alongside-old).
- [ ] Run AMD/AmEx/Boeing subset first, then full 150 (buy $10 OpenRouter credits → 1,000/day to avoid throttle).
- [ ] Record accuracy + abstain-rate + a manual hallucination spot-check.
- [ ] **Gate:** beat naive-RAG **19%** decisively. Target **≥55%**. If short → P5 levers.
- [ ] Commit real numbers into README/plan (no marketing).

---

## Phase 4 — Delete + honest README

- [ ] Grep each delete-list module for non-test callers; delete dead stack + its tests.
- [ ] Rewrite `README.md`: the pipeline that actually runs + the measured FinanceBench number. Kill the SPLADE/ColBERT narrative and the fabricated results table.
- [ ] `ruff check .` + `pytest`. Commit.

---

## Phase 5 — Add ONLY if P3 is short (named, unbuilt, leverage order)

1. **Dedicated reranker** — Voyage rerank-2.5 or Cohere rerank-3.5 (separate key; not on OpenRouter). Anthropic recipe: −67% retrieval failures. First lever if retrieval is the miss.
2. **Contextual-retrieval prefixing** — 1-sentence LLM context per chunk pre-embed (−49% failures). Ingest-time cost (use free Nemotron to generate).
3. **Per-doc long-context routing** — route to the winning doc, stuff the whole doc (<200K) into Nemotron. The 79% path for single-doc questions.
4. **Airtight citations for demo** — flip `CONTEXTIQ_LLM_PROVIDER=anthropic` to use the Citations API seam.

Each is a knob, not a rewrite. Don't build until the eval demands it.

---

**Self-review:** covers parse (P2), retrieve (P2), grounded answer+abstain (P1), eval gate (P3), cleanup (P4), scale levers (P5). Real signatures only (`LLMClient`/`LLMResult`/`GroundedAnswer`/`ContextEngine.build_context`/`vector_index.search_hybrid` read from the tree). OpenRouter request bodies pinned at implementation against live docs (no stale-API placeholder code).
