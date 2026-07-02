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
	cd evals && uv run python run.py
