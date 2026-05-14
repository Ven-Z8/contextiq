FROM python:3.12-slim AS runtime

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src ./src
COPY prompts ./prompts
COPY scripts ./scripts

RUN uv sync --no-dev

CMD ["uv", "run", "contextiq", "--help"]
