# ContextIQ — Product Requirements Document

> **One line:** An open-source, run-it-locally RAG system that answers questions over **200+ page documents** with **reproducibly high retrieval accuracy on a public benchmark** — and lets you **swap modules** (extraction engine today; domain skills later) without touching the core.

**Status:** Living document · v0.2 · Last updated 2026-07-14
**Owner:** Venkat
**Repo:** [Ven-Z8/contextiq](https://github.com/Ven-Z8/contextiq) (public)

---

## 1. The problem

Most RAG systems quietly fall apart on long, messy, real-world documents — a 200-page 10-K, a scanned manual, a multi-column report. They chunk-and-embed uniformly, fragment tables and sections, and retrieve the wrong evidence. Worse, almost none of them prove their quality: results are demoed on cherry-picked questions, never on a benchmark a skeptic already respects.

ContextIQ exists to be the RAG system you can **point at a 200+ page document, run locally, and trust** — because its retrieval quality is measured on a public benchmark and the numbers reproduce.

## 2. Vision

A RAG you run on your own machine, where every layer is a swappable module:

- **Swap the extraction engine** — Docling-VLM today; Nemotron-Parse, Mistral OCR, Reducto, or whatever wins next, behind one interface.
- **Swap the domain skill** *(future)* — finance, legal, medical retrieval behaviors as plug-ins.
- **Keep the core** — a hierarchical document tree + agentic retrieval that doesn't change when you swap a module.

The differentiator is not features — it's **undeniable, reproducible retrieval quality** on long documents, in the open.

## 3. Target users

| User | Need |
|---|---|
| Engineer evaluating RAG approaches | Clone, `pip install`, run on their own PDF, see benchmark numbers reproduce |
| Builder embedding RAG in a product | A clean, swappable retrieval core they can extend per domain |
| Researcher / skeptic | A reproducible eval harness against a recognized public benchmark |

**Non-user (for now):** turnkey SaaS users — this is a local/self-hosted library + app first.

## 4. The headline metric (what "undeniable" means)

The product's credibility rests on **one reproducible primary number**, plus supporting metrics. We measure on **MMLongBench-Doc** (135 long PDFs, avg 47.5 pages; 1,082 questions with annotated `evidence_pages`; deliberately includes cross-page and unanswerable questions).

| Metric | Definition | Target (v0.1) | Why it's undeniable |
|---|---|---|---|
| **Retrieval Recall@k** *(headline)* | Fraction of questions where the retriever surfaces the annotated evidence page(s) within top-k (k=5) | **≥ 0.85** | Page-level, reproducible from public `evidence_pages` labels; matches what "retrieval" actually means |
| Retrieval Recall@10 | Same, k=10 | ≥ 0.92 | Headroom check |
| Answer quality (F1) | End-to-end answer F1 vs gold | **Beats stated baseline** (vanilla RAG; reference: GPT-4o = 44.9% F1) | Reported as a *lift over baseline*, never as a raw "80–90%" — SOTA on this benchmark is ~45%, so we never overclaim answer accuracy |
| Unanswerable handling | Correct refusals on the 223 unanswerable questions | Tracked | Tests the confidence gate / anti-hallucination |
| Ingestion quality | Table fidelity (TEDS), reading-order accuracy | Beats Docling-standard baseline | Garbage extraction caps recall |

**Reproducibility is a hard requirement.** Every number ships with: pinned dataset + model versions, a documented method, a one-command runner, and a baseline beside it. Current financial QA eval: `evals/financebench/run_answers.py` (see `AGENTS.md`). We **never** headline an end-to-end answer-accuracy figure of 80–90% on MMLongBench-Doc — that would be the opposite of undeniable.

## 5. Architecture (Agentic Hierarchical RAG)

Three swappable layers over a stable core:

1. **Ingestion** — pluggable layout-preserving extraction (`Extractor` protocol: Docling-standard default, Docling-VLM optional) → citation-preserving blocks, optional recursive document tree with node summaries.
2. **Retrieval** — local Qdrant hybrid (dense FastEmbed + BM25 sparse, RRF fusion), optional agentic route/decompose/rerank (`CONTEXTIQ_AGENTIC`), then token-budgeted context packing.
3. **Orchestration** — grounded answer synthesis with citations + extractive fallback when no LLM key is configured.

Infra (turbopuffer / cloud vector DB) is a later scale phase, not the headline. Hard constraint: **never the OpenAI SDK**.

## 6. Scope

### v0.1 (this milestone) — "Prove the retrieval"
- **In:** long-document ingestion; hybrid retrieval; FinanceBench answer eval harness; local install (`uv sync`) with no machine-local secrets in the repo.
- **Out:** domain-skill plug-ins; cloud vector DB; hosted SaaS; claiming dead SPLADE/ColBERT enterprise stack as live.

### Later
- Domain-skill plug-in system (finance/legal/medical behaviors).
- Stronger MMLongBench-Doc recall harness restored under a real one-command target.
- turbopuffer/cloud backend; multi-document graph.

## 7. Requirements

**Functional**
- Ingest a 200+ page PDF locally without OOM; produce a navigable document tree.
- Swap the extraction engine via one interface, no core changes.
- Retrieve evidence for a question with page-level citations.
- Run the full benchmark with one command and emit a metrics report.

**Non-functional / "undeniable"**
- **Installable by anyone:** `git clone && uv sync` (or `pip install`) works with zero machine-local path dependencies. *(Current blocker: `ven-obs`/`ven-eval` hard-pinned to local paths — must be fixed.)*
- **CI runs the test suite and (eventually) the eval gate** on every PR.
- Reproducible benchmark numbers with pinned versions + documented method.
- Process/planning docs stay local (public-repo hygiene); product docs (this PRD, README, benchmarks) are public.

## 8. Milestones

| Milestone | Definition of done |
|---|---|
| **M0 — Foundation** *(done)* | Pluggable extraction → document tree, additive to retrieval, tests green (PR #1) |
| **M1 — Installable + CI** | No machine-local deps; `uv sync` works for a fresh clone; CI runs tests + lint on PRs |
| **M2 — Measured** | FinanceBench answer harness green; MMLongBench harness historical (see docs) |
| **M3 — ≥ 0.85** | Retrieval Recall@5 ≥ 0.85 on MMLongBench-Doc; answer F1 beats baseline; numbers in README |

## 9. Open risks

| Risk | Mitigation |
|---|---|
| Machine-local deps block install & CI | Make `ven-obs`/`ven-eval` optional or vendor them (M1, top priority) |
| VLM extraction too slow for 200pp interactively | Async-batch only; MLX on Apple Silicon; standard-extractor fast path |
| Recall@5 plateaus below 0.85 | Reasoning-tree navigation + hybrid fusion + reranking; ablate per component |
| Over-claiming answer accuracy | PRD §4 fixes the metric discipline: retrieval headline, answer-as-lift |

---

*This PRD is the single product source-of-truth. Engineering specs/plans live local-only per the public-repo hygiene policy.*
