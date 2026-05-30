"""Run promptfoo for contextiq and emit the shared EvalRecord file."""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ven_eval.records import JsonFileSink, normalize

HERE = Path(__file__).parent
NATIVE = HERE / "results" / "contextiq.json"
RECORDS = HERE / "results" / "contextiq.records.json"


def main() -> None:
    subprocess.run(
        ["npx", "promptfoo@0.121.13", "eval", "-c", "promptfooconfig.yaml"],
        cwd=HERE,
        check=True,
    )
    data = json.loads(NATIVE.read_text())
    recs = normalize(
        data,
        project="contextiq",
        run_id=str(uuid.uuid4()),
        ts=datetime.now(timezone.utc).isoformat(),
    )
    JsonFileSink(RECORDS).write(recs)
    print(f"Wrote {len(recs)} records -> {RECORDS}")


if __name__ == "__main__":
    main()
