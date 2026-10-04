#!/usr/bin/env python3
"""Find the first random-corpus body that lacks the canonical body identity."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import subprocess
from pathlib import Path


def args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--body-tool", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--jobs", type=int, default=8)
    return ap.parse_args()


def scan(tool: Path, item: tuple[int, dict[str, str]]) -> tuple[int, dict | None]:
    index, row = item
    proc = subprocess.run(
        [str(tool), "scan", row["path"]],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode:
        return index, {
            "index": index,
            "uuid": row["uuid"],
            "path": row["path"],
            "scan_error": proc.stderr[-8000:],
        }

    report = json.loads(proc.stdout)
    fallback = next(
        (solid for solid in report["solids"] if solid["identity_kind"] != "canonical"),
        None,
    )
    if fallback is None:
        return index, None
    return index, {
        "index": index,
        "uuid": row["uuid"],
        "path": row["path"],
        "report_counts": {
            key: report[key]
            for key in (
                "manifold_solids",
                "fallback_identity_solids",
                "skipped_solids",
            )
        },
        "first": fallback,
    }


def main() -> int:
    cfg = args()
    with cfg.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if cfg.start < 0 or cfg.start > len(rows):
        raise SystemExit(f"--start {cfg.start} outside 0..{len(rows)}")

    for start in range(cfg.start, len(rows), cfg.batch):
        batch = list(enumerate(rows[start : start + cfg.batch], start))
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.jobs) as pool:
            results = list(pool.map(lambda item: scan(cfg.body_tool, item), batch))
        failures = [failure for _, failure in results if failure is not None]
        if failures:
            print(json.dumps(min(failures, key=lambda failure: failure["index"]), indent=2))
            return 1
        print(f"canonical through model {start + len(batch) - 1}", flush=True)

    print(json.dumps({"all_canonical": True, "models": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
