# AGENTS.md

## Cursor Cloud specific instructions

ContextIQ is a Python 3.12 document-QA (RAG) app managed by [`uv`](https://docs.astral.sh/uv/). It ships three runnable surfaces (all defined under `[project.scripts]` in `pyproject.toml`):

- **`contextiq`** — Typer CLI (`ingest`, `ask`, `inspect-context`, `eval-retrieval`, ...).
- **`contextiq-api`** — FastAPI backend on `127.0.0.1:8000` (uvicorn, `--reload`). It also serves a self-contained dark-themed HTML dashboard at `/` (endpoints: `/health`, `/stats`, `/ingest`, `/context`, `/answer`).
- **`contextiq-ui`** — Gradio dashboard (requires the `ui` extra).

Standard commands are already documented in the `Makefile` (`make api|ui|test|lint`) and `README.md` — use those rather than duplicating them. Prefix commands with `uv run`, e.g. `uv run contextiq-api`.

### Non-obvious caveats

- **Runs fully offline without any API key.** With no LLM key configured, `ask`/`/answer` return a safe **extractive fallback** (mode `extractive_fallback`) that surfaces the retrieved evidence instead of a synthesized answer — this is expected, not a failure. To enable real LLM synthesis, set `OPENROUTER_API_KEY` (default provider, `minimax/minimax-m3`) or set `CONTEXTIQ_LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY`. Copy `.env.example` to `.env` to configure.
- **You must ingest a document before querying.** `ask`/`/answer`/the dashboard return nothing useful until at least one doc is ingested. Quick start: `uv run contextiq ingest data/raw/sample-contract.md`.
- **Qdrant is embedded/local**, persisted under `data/qdrant` (path from `CONTEXTIQ_QDRANT_PATH`). Ingested state persists across restarts; there is no external Qdrant server to run. The log line `UserWarning: Payload indexes have no effect in the local Qdrant` is benign.
- **First ingest downloads FastEmbed models** (BGE-small dense + BM25) from Hugging Face and caches them; the first run is slower and needs network access.
- **Tests pass cleanly with plain `uv run pytest`.** The `--deselect` flags in `.github/workflows/ci.yml` target test files that no longer exist in the tree, so they are unnecessary here.
