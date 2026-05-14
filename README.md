# ContextIQ

ContextIQ is a context-engineered agentic RAG system for complex enterprise documents, with MCP tools and measurable evaluation.

## Benchmark

| Metric | Naive RAG | ContextIQ |
| --- | ---: | ---: |
| Retrieval Recall@10 | pending | 0.736 |
| Retrieval Precision@10 | pending | 0.233 |
| Retrieval MRR | pending | 0.735 |
| Retrieval NDCG@10 | pending | 0.853 |

These numbers are the legacy Apple/NASA seed eval. The current live corpus is larger:
5 real documents, 7,428 blocks, and manual smoke checks across Apple, Microsoft,
NVIDIA, and NASA sources. See [docs/benchmarks.md](docs/benchmarks.md).

## Quick Install

```bash
uv sync --extra dev
uv run contextiq --help
uv run pytest
```

## Local App

Run the backend:

```bash
uv run contextiq-api
```

Then open `http://127.0.0.1:8000`.

The dashboard supports two modes:

- `Answer with Evidence`: retrieval plus Claude grounded answer synthesis.
- `Build Context Only`: retrieval trace without an LLM call.

Set `ANTHROPIC_API_KEY` for Claude synthesis. Without a key, ContextIQ returns a
safe extractive fallback so the demo still runs locally.

## Architecture

```text
Document -> Structural Chunking -> Qdrant/Fallback Index -> Intent Router
  -> Context Packet -> Claude Answer Synthesis -> Citations + Eval Trace
```

Corpus-specific retrieval vocabulary lives in
[config/retrieval_profile.json](config/retrieval_profile.json), so source
aliases, product markers, asset names, and structured-document markers can be
updated without changing generic retrieval code.

## Design Thesis

ContextIQ is built from a local AI engineering pattern library:

- Context window auto-compaction
- Context minimization
- Progressive disclosure for large files
- Spec-driven eval contracts
- Policy-gated MCP tools
- Planner-worker agent orchestration

See [docs/wiki-patterns.md](docs/wiki-patterns.md).
See [docs/code-flow.md](docs/code-flow.md) for the function-level execution map.

The agent, MCP, and eval contracts live in:

- [specs/agents.yaml](specs/agents.yaml)
- [specs/mcp-tools.yaml](specs/mcp-tools.yaml)
- [specs/evals.yaml](specs/evals.yaml)

## First Demo

```bash
uv run contextiq ingest data/raw/apple-2025-10k.pdf
uv run contextiq ask "What are the main regulatory risks? Cite pages."
curl -X POST http://127.0.0.1:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"What HLR functions support for Orion?","limit":6}'
uv run contextiq inspect-context
```

## Evaluation

```bash
uv run contextiq eval-retrieval --limit 20 --k 10
make eval
```
