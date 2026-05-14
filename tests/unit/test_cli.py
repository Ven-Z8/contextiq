from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from contextiq.cli.app import app


def test_cli_eval_retrieval_prints_report(tmp_path: Path) -> None:
    qrels_path = tmp_path / "qrels.json"
    qrels_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "id": "missing",
                        "question": "What evidence is missing?",
                        "relevant_blocks": {"missing:block": 3},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["eval-retrieval", "--qrels", str(qrels_path), "--limit", "3", "--k", "3"],
    )

    assert result.exit_code == 0
    assert "Retrieval Evaluation" in result.output
    assert "recall@3" in result.output
