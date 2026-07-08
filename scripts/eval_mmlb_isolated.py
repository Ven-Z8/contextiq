"""Per-document isolated MMLongBench eval: each doc in a fresh subprocess so peak
RAM is reclaimed between docs (16GB local can't hold accumulation + strong models).
Aggregates page-Recall@5/@10 + lift over a random page-picker across all docs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import fmean


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", type=int, default=10)
    p.add_argument("--pipeline", default="simple")
    p.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--dim", type=int, default=1024)
    args = p.parse_args()

    r5: list[float] = []
    r10: list[float] = []
    f5: list[float] = []
    f10: list[float] = []
    failed: list[int] = []
    for i in range(args.docs):
        out = Path(tempfile.gettempdir()) / f"mmlb_iso_{i}.json"
        cmd = [
            "uv", "run", "--no-sync", "python", "scripts/eval_mmlongbench.py",
            "--doc-index", str(i), "--pipeline", args.pipeline,
            "--embed-model", args.embed_model, "--dim", str(args.dim),
            "--records-out", str(out),
        ]
        print(f"--- doc {i} ---", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0 or not out.exists():
            print(f"doc {i} FAILED (rc={rc})", flush=True)
            failed.append(i)
            continue
        recs = json.loads(out.read_text())
        for rec in recs:
            r5.append(rec["recall@5"])
            r10.append(rec["recall@10"])
            pc = rec.get("page_count") or 1
            f5.append(min(5 / pc, 1.0))
            f10.append(min(10 / pc, 1.0))

    def m(x): return round(fmean(x), 4) if x else 0.0
    summary = {
        "docs": args.docs, "failed_docs": failed, "questions": len(r5),
        "embed_model": args.embed_model,
        "page_recall@5": m(r5), "random_floor@5": m(f5), "lift@5": round(m(r5) - m(f5), 4),
        "page_recall@10": m(r10), "random_floor@10": m(f10), "lift@10": round(m(r10) - m(f10), 4),
    }
    print("\n=== ISOLATED AGGREGATE ===")
    print(json.dumps(summary, indent=2))
    Path("/tmp/mmlb_isolated_summary.json").write_text(json.dumps(summary, indent=2))
    if failed:
        sys.exit(0)


if __name__ == "__main__":
    main()
