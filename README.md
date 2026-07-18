# ContextIQ

Grounded question-answering over financial filings (10-Ks, 10-Qs). Routes a question to the right document, decomposes it into line items, retrieves with hybrid search, and answers with an LLM that **cites every claim** and **refuses when evidence is missing**.

## Why it exists

QA over long financial documents is deceptively hard: the answer to "what's the quick ratio?" lives in a balance-sheet table whose vocabulary ("current assets", "current liabilities") never matches the question. Naive RAG scores ~19% on FinanceBench. ContextIQ is a deliberately lean, *simple-agentic* pipeline that does markedly better while staying honest about what it doesn't know.

## Verified Results

Measured on **FinanceBench** (Patronus AI — human-verified questions over public filings). Answer accuracy is scored by an LLM judge against gold answers.

| Setting | Questions | Correct | Accuracy | Notes |
|---------|-----------|---------|----------|-------|
| **Per-document, agentic** (decompose + rerank) | 21 (3 docs) | 6 | **0.286** | Decompose lifts hardest questions |
| **Per-document, plain hybrid** (no agentic) | 21 (3 docs) | 4 | **0.190** | Matches FinanceBench naive-RAG baseline (0.19) |
| **Corpus, agentic** (router picks filing) | — | — | — | Requires full corpus ingestion; not yet run to completion |

**Honest limitations:**
- Only 3 of 12 FinanceBench companies ingested (AMD, AMEX, Boeing) — 21 questions total
- Agentic decompose helps on ratio/metric questions but adds ~3 LLM calls/question
- ±1 question variance on LLM nondeterminism — treat every number as a range
- Full 150-question corpus not yet validated; broad claims require full ingestion

**Reproduce:**
```bash
# Ingest a filing
uv run contextiq ingest evals/financebench/pdfs/AMD_2022_10K.pdf

# Plain hybrid retrieve (baseline)
PYTHONPATH=src python evals/financebench/run_answers.py --limit 0

# Agentic retrieve (decompose + rerank)
PYTHONPATH=src python evals/financebench/run_answers.py --limit 0 --agentic
```

## Architecture

```
Question
    │
    ▼
Route ───► (if corpus) pick target filing
    │
    ▼
Decompose ───► "quick ratio" → ["current assets", "current liabilities", ...]
    │
    ▼
Retrieve + Merge ───► Qdrant hybrid (BGE dense + BM25, RRF) per sub-query
    │
    ▼
Rerank ───► Table-aware LLM rerank keeps most useful blocks (incl. tables)
    │
    ▼
Answer ───► minimax-m3 (OpenRouter) with strict grounding prompt
    │           • Every claim cited: block_id, page
    │           • Emits NOT_IN_DOCUMENT rather than guess
    ▼
```

No multi-agent orchestration — one route + one decompose + one rerank call. Every added component was measured head-to-head before adoption.

## Quick Start

```bash
uv sync --extra dev --extra ui
cp .env.example .env            # set OPENROUTER_API_KEY or NVIDIA_API_KEY
uv run contextiq ingest data/raw/sample-contract.md
uv run contextiq-ui             # Gradio dashboard (or contextiq-api for FastAPI backend)
```

Without `OPENROUTER_API_KEY` or `NVIDIA_API_KEY`, ContextIQ returns a safe extractive fallback so the demo still runs offline. Agentic retrieve is on by default (`CONTEXTIQ_AGENTIC=1`) and falls back to plain hybrid when no model client is available.

## LLM Provider Options

ContextIQ supports three answer providers via `CONTEXTIQ_LLM_PROVIDER`:

| Provider | Env Key | Default Model | Notes |
|----------|---------|---------------|-------|
| `openrouter` | `OPENROUTER_API_KEY` | `minimax/minimax-m3` | Cheap, strong reasoning, good for finance |
| `nvidia` | `NVIDIA_API_KEY` | `nvidia/nemotron-3-ultra-550b-a55b` | NVIDIA-hosted NIM, free tier available |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` | Citations API for airtight grounding |

Switch providers by setting `CONTEXTIQ_LLM_PROVIDER` in `.env`. Agentic retrieve (decompose + rerank) works with any provider that has a valid API key.

## Stack

Python · Qdrant (hybrid vector index) · BGE-small dense + BM25 sparse (FastEmbed) · Docling (PDF/table parsing) · minimax-m3 via OpenRouter (routing, decomposition, rerank, synthesis) · FastAPI · Gradio · Typer CLI · pytest

Answer model is env-driven: swap via `CONTEXTIQ_OPENROUTER_MODEL`, or set `CONTEXTIQ_LLM_PROVIDER=anthropic` to use Claude with the Anthropic Citations API.

## Evaluation Harness

```
evals/
├── financebench/
│   ├── questions.jsonl      # 150 questions, 12 companies, 8 sectors
│   ├── pdfs/                # 10-K/10-Q PDFs (ingest first)
│   ├── run_answers.py       # Accuracy eval with LLM judge
│   └── run_recall.py        # Retrieval recall@k eval
├── contextiq.yaml           # Promptfoo config for regression tests
└── provider.py              # LLM client abstraction for judges
```

| Metric | Command |
|--------|---------|
| Retrieval Recall@k | `PYTHONPATH=src python evals/financebench/run_recall.py --limit 0` |
| Answer Accuracy | `PYTHONPATH=src python evals/financebench/run_answers.py --limit 0 [--agentic] [--corpus]` |
| Promptfoo Regression | `npx promptfoo eval -c evals/contextiq.yaml` |

## CI

```yaml
# .github/workflows/ci.yml
- ruff check .
- pytest tests/ -q
- PYTHONPATH=src python evals/financebench/run_answers.py --limit 5  # smoke test
```

![CI](https://github.com/Ven-Z8/contextiq/actions/workflows/ci.yml/badge.svg)