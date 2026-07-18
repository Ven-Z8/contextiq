# AGENTS.md

## Cursor Cloud specific instructions

ContextIQ is a Python 3.12 document-QA (RAG) app managed by [`uv`](https://docs.astral.sh/uv/). It ships three runnable surfaces (all defined under `[project.scripts]` in `pyproject.toml`):

- **`contextiq`** — Typer CLI (`ingest`, `ask`, `inspect-context`, `eval-retrieval`, ...).
- **`contextiq-api`** — FastAPI backend on `127.0.0.1:8000` (uvicorn, `--reload`). It also serves a self-contained dark-themed HTML dashboard at `/` (endpoints: `/health`, `/stats`, `/ingest`, `/context`, `/answer`).
- **`contextiq-ui`** — Gradio dashboard (requires the `ui` extra).

Standard commands are already documented in the `Makefile` (`make api|ui|test|lint`) and `README.md` — use those rather than duplicating them. Prefix commands with `uv run`, e.g. `uv run contextiq-api`.

### Non-obvious caveats

- **Runs fully offline without any API key.** With no LLM key configured, `ask`/`/answer` return a safe **extractive fallback** (mode `extractive_fallback`) that surfaces the retrieved evidence instead of a synthesized answer — this is expected, not a failure. To enable real LLM synthesis, set the default provider's API key env var (see `.env.example` for the exact variable names, provider switch, and default model), or switch the provider to `anthropic` with an Anthropic key. Note: the provider/model config values may be registered as redacted secrets in Cloud Agent settings, so avoid writing their literal values into committed files (the commit-time secret scanner will block them).
- **You must ingest a document before querying.** `ask`/`/answer`/the dashboard return nothing useful until at least one doc is ingested. Quick start: `uv run contextiq ingest data/raw/sample-contract.md`.
- **Qdrant is embedded/local**, persisted under `data/qdrant` (path from `CONTEXTIQ_QDRANT_PATH`). Ingested state persists across restarts; there is no external Qdrant server to run. The log line `UserWarning: Payload indexes have no effect in the local Qdrant` is benign.
- **First ingest downloads FastEmbed models** (BGE-small dense + BM25) from Hugging Face and caches them; the first run is slower and needs network access.
- **Tests:** `uv run pytest` locally. CI additionally `--deselect`s two known failing cases in `test_no_corpus_hardcoding.py` and `test_vector_index.py` (issue #11) — those files exist; the deselects are intentional quarantine, not leftovers for deleted tests.
- **FinanceBench evals**: the 10-K PDFs are NOT checked into the repo. Fetch them from `https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/<DOC_NAME>.pdf` into `evals/financebench/pdfs/`, ingest with `uv run contextiq ingest <pdf>`, then run `PYTHONPATH=src uv run python evals/financebench/run_answers.py --limit 0 [--agentic|--corpus]`. Requires the default provider's API key (answerer + LLM judge; see `.env.example`); only questions whose `doc_name` is ingested are scored (`evals/financebench/questions.jsonl` has 150 total). `evals/run.py` (promptfoo) depends on a private `ven_eval` package not in this repo — use `run_answers.py` instead.
