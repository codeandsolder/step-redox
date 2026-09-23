#!/usr/bin/env python3
"""Run step-support-scan over a TSV corpus manifest and retain matching solids."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scanner", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--face-count", type=int)
    parser.add_argument("--contains-support", action="append", default=[])
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"uuid", "path"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"{path}: expected TSV columns {sorted(required)}")
    return rows


def scan_one(scanner: Path, row: dict[str, str]) -> dict:
    proc = subprocess.run(
        [str(scanner), row["path"]],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    result = {
        "uuid": row["uuid"],
        "path": row["path"],
        "returncode": proc.returncode,
    }
    if proc.returncode:
        result["error"] = proc.stderr[-4000:]
        result["solids"] = []
        return result
    try:
        result["solids"] = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        result["error"] = f"invalid scanner JSON: {exc}: {proc.stdout[-1000:]}"
        result["solids"] = []
    return result


def matches(solid: dict, face_count: int | None, supports: list[str]) -> bool:
    if face_count is not None and solid.get("face_count") != face_count:
        return False
    counts = solid.get("support_counts", {})
    return all(counts.get(support, 0) > 0 for support in supports)


def main() -> int:
    args = parse_args()
    rows = load_manifest(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    matched = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = pool.map(lambda row: scan_one(args.scanner, row), rows)
        with args.out.open("w") as handle:
            for result in results:
                if result.get("error"):
                    failures += 1
                result["solids"] = [
                    solid
                    for solid in result["solids"]
                    if matches(solid, args.face_count, args.contains_support)
                ]
                if not result["solids"] and not result.get("error"):
                    continue
                matched += len(result["solids"])
                handle.write(json.dumps(result, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "files": len(rows),
                "matching_solids": matched,
                "failures": failures,
                "output": str(args.out),
            },
            separators=(",", ":"),
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
