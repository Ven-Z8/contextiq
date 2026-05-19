# ContextIQ

ContextIQ is a context-engineered retrieval system for complex enterprise documents. It parses large files, preserves citation metadata, builds token-aware context packets, and returns grounded answers with evidence.

## Results Snapshot

| Metric | Lexical-only baseline | ContextIQ |
| --- | ---: | ---: |
| Retrieval Recall@10 | 0.556 | 0.736 |
| Retrieval Precision@10 | 0.233 | 0.233 |
| Retrieval MRR | 0.667 | 0.735 |
| Retrieval NDCG@10 | 0.881 | 0.853 |

These retrieval metrics come from the seed evaluation set in `tests/evals/qrels/retrieval_seed.json`. See [docs/benchmarks.md](docs/benchmarks.md) for details.

## Quick Install

```bash
uv sync --extra dev
uv run contextiq --help
uv run pytest
```

## Run The App

Run the backend:

```bash
uv run contextiq-api
```

Then open `http://127.0.0.1:8000`.

The dashboard supports two modes:

- `Answer with Evidence`: retrieval plus grounded answer synthesis.
- `Build Context Only`: retrieval trace without an LLM call.

Set `ANTHROPIC_API_KEY` for Anthropic answer synthesis. Without a key, ContextIQ returns a safe extractive fallback so the demo still runs locally.

## Architecture

```text
Document -> Structural Chunking -> Qdrant/Fallback Index -> Intent Router
  -> Context Packet -> Answer Synthesis -> Citations + Eval Trace
```

Corpus-specific retrieval vocabulary lives in
[config/retrieval_profile.json](config/retrieval_profile.json), so source
aliases, product markers, asset names, and structured-document markers can be
updated without changing generic retrieval code.

## Highlights

- Structural chunking for prose, tables, spreadsheets, and figure metadata.
- Citation-preserving retrieval with source, page, block, and chunk metadata.
- Token-aware context packet assembly with selected and dropped candidates.
- Configurable retrieval profile for aliases, source markers, assets, and financial terms.
- Optional Anthropic answer synthesis with extractive fallback when no API key is set.
- Retrieval eval contracts and qrels for repeatable quality checks.

See [docs/code-flow.md](docs/code-flow.md) for the function-level execution map.

The retrieval, MCP, and eval contracts live in:

- [specs/agents.yaml](specs/agents.yaml)
- [specs/mcp-tools.yaml](specs/mcp-tools.yaml)
- [specs/evals.yaml](specs/evals.yaml)

## Demo Commands

```bash
uv run contextiq ingest data/raw/sample-contract.md
uv run contextiq ask "What are the main regulatory risks? Cite pages."
curl -X POST http://127.0.0.1:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"What obligations does the contract mention?","limit":6}'
uv run contextiq inspect-context
```

## Evaluation

```bash
uv run contextiq eval-retrieval --limit 20 --k 10
make eval
```
