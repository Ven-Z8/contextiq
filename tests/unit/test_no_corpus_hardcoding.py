from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = [
    PROJECT_ROOT / "src/contextiq/context",
    PROJECT_ROOT / "src/contextiq/ingestion",
    PROJECT_ROOT / "src/contextiq/llm",
    PROJECT_ROOT / "src/contextiq/retrieval",
]
FORBIDDEN_CORPUS_TERMS = [
    "apple",
    "microsoft",
    "msft",
    "nasa",
    "orion",
    "gateway",
    "lunar terrain vehicle",
    "space launch system",
    "exploration ground systems",
    "human landing system",
    "uc-t-202",
    "fn-t-101",
    "em-003-hlr",
    "data-and-computing-architecture-study",
    "lunar-objective-decomposition",
]


def test_core_source_does_not_hardcode_demo_corpus_terms() -> None:
    """Keep demo/corpus vocabulary in config, docs, tests, or ingested data."""

    violations: list[str] = []
    for source_dir in SOURCE_DIRS:
        for path in source_dir.rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            hits = [
                term
                for term in FORBIDDEN_CORPUS_TERMS
                if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text)
            ]
            if hits:
                violations.append(f"{path.relative_to(PROJECT_ROOT)}: {', '.join(hits)}")

    assert violations == []
