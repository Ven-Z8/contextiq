.PHONY: run api ui test lint bench eval

run:
	uv run contextiq --help

api:
	uv run contextiq-api

ui:
	uv run --extra ui contextiq-ui

test:
	uv run pytest

lint:
	uv run ruff check .

bench:
	uv run python scripts/benchmark.py

eval:
	uv run contextiq eval-retrieval --limit 20 --k 10

eval-mmlongbench:
	uv run --extra dev --extra eval python scripts/eval_mmlongbench.py --limit-docs $(or $(DOCS),3)
