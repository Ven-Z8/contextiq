"""Run promptfoo for contextiq and emit the shared EvalRecord file."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ven_eval.records import JsonFileSink, SqliteSink, normalize

HERE = Path(__file__).parent
NATIVE = HERE / "results" / "contextiq.json"
RECORDS = HERE / "results" / "contextiq.records.json"


def main() -> None:
    # Pin promptfoo's provider python to THIS interpreter — the venv that has
    # ven_eval + contextiq installed. (Assert scripts are intentionally
    # dependency-free, so they run under whatever interpreter promptfoo picks.)
    env = {**os.environ, "PROMPTFOO_PYTHON": sys.executable}
    result = subprocess.run(
        # -j 1: embedded Qdrant is single-writer; parallel workers deadlock the lock.
        ["npx", "promptfoo@0.118.0", "eval", "-c", "promptfooconfig.yaml", "-j", "1"],
        cwd=HERE,
        env=env,
    )
    # promptfoo exits 100 when some test cases fail — that is normal eval signal,
    # not a runner error. Only a missing results file means the run truly broke.
    if result.returncode not in (0, 100):
        raise SystemExit(f"promptfoo failed with exit code {result.returncode}")
    data = json.loads(NATIVE.read_text())
    recs = normalize(
        data,
        project="contextiq",
        run_id=str(uuid.uuid4()),
        ts=datetime.now(UTC).isoformat(),
    )
    JsonFileSink(RECORDS).write(recs)
    print(f"Wrote {len(recs)} records -> {RECORDS}")

    # DB-foundation seam: opt in by setting VEN_EVAL_DB to the shared store
    # (e.g. the workspace ven_master.db). Keeps contextiq decoupled from layout.
    db_path = os.environ.get("VEN_EVAL_DB")
    if db_path:
        SqliteSink(db_path).write(recs)
        print(f"Wrote {len(recs)} records -> {db_path}")


if __name__ == "__main__":
    main()
