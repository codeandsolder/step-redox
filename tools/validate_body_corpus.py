#!/usr/bin/env python3
"""Validate a generated step-redox single-body corpus."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--body-tool", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=8)
    return ap.parse_args()


def corpus_fingerprint(uuid: str, solid: dict) -> str:
    geometry = solid.get("geometry_fingerprint")
    if geometry is None:
        payload = (
            f'body-v2:fallback:{uuid}:{solid["data_section"]}:{solid["solid_id"]}'
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    scale = solid.get("unit_scale_mm")
    if scale is None:
        payload = f"body-v2:unknown:{uuid}:{geometry}"
    else:
        payload = f"body-v2:mm-scale:{float(scale).hex()}:{geometry}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def validate_one(corpus: Path, tool: Path, body: dict) -> dict | None:
    rel = body.get("body_path")
    if not body.get("extractable"):
        return None
    if not rel:
        return {"fingerprint": body["fingerprint"], "error": "missing body_path"}

    path = corpus / rel
    if not path.is_file():
        return {"fingerprint": body["fingerprint"], "error": f"missing file {path}"}

    proc = subprocess.run(
        [str(tool), "scan", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode:
        return {
            "fingerprint": body["fingerprint"],
            "error": f"scan failed rc={proc.returncode}: {proc.stderr[-2000:]}",
        }
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {
            "fingerprint": body["fingerprint"],
            "error": f"invalid scan JSON: {exc}",
        }

    if report["manifold_solids"] != 1 or report["analyzed_solids"] != 1:
        return {
            "fingerprint": body["fingerprint"],
            "error": "isolated file does not contain exactly one analyzable manifold solid",
            "report": {
                "manifold_solids": report["manifold_solids"],
                "analyzed_solids": report["analyzed_solids"],
                "skipped_solids": report["skipped_solids"],
            },
        }

    solid = report["solids"][0]
    if body.get("identity_kind") != "file_local_fallback":
        expected = corpus_fingerprint(body["representative"]["uuid"], solid)
        if expected != body["fingerprint"]:
            return {
                "fingerprint": body["fingerprint"],
                "error": "identity changed after extraction",
                "actual": expected,
                "geometry_fingerprint": solid["geometry_fingerprint"],
                "unit_scale_mm": solid.get("unit_scale_mm"),
            }
    elif solid.get("geometry_fingerprint") is not None:
        return {
            "fingerprint": body["fingerprint"],
            "error": "fallback body unexpectedly became canonically identifiable after extraction",
            "geometry_fingerprint": solid["geometry_fingerprint"],
        }

    for key in ("vertices", "edges", "oriented_edges", "faces"):
        if solid[key] != body[key]:
            return {
                "fingerprint": body["fingerprint"],
                "error": f"{key} changed after extraction",
                "expected": body[key],
                "actual": solid[key],
            }
    return None


def main() -> int:
    args = parse_args()
    bodies_path = args.corpus / "bodies.jsonl"
    occurrences_path = args.corpus / "occurrences.jsonl"
    summary_path = args.corpus / "summary.json"

    bodies = [json.loads(line) for line in bodies_path.read_text().splitlines() if line]
    occurrences = [
        json.loads(line) for line in occurrences_path.read_text().splitlines() if line
    ]
    summary = json.loads(summary_path.read_text())

    errors: list[dict] = []
    fingerprints = [body["fingerprint"] for body in bodies]
    if len(fingerprints) != len(set(fingerprints)):
        errors.append({"error": "duplicate fingerprint in bodies manifest"})

    occurrence_total = sum(body["occurrences"] for body in bodies)
    if occurrence_total != len(occurrences):
        errors.append(
            {
                "error": "body occurrence counts do not match occurrences manifest",
                "body_occurrence_total": occurrence_total,
                "occurrence_rows": len(occurrences),
            }
        )

    manifest_counts = {body["fingerprint"]: body["occurrences"] for body in bodies}
    occurrence_counts = Counter(row["fingerprint"] for row in occurrences)
    if manifest_counts != dict(occurrence_counts):
        missing = sorted(set(manifest_counts) - set(occurrence_counts))
        extra = sorted(set(occurrence_counts) - set(manifest_counts))
        mismatched = sorted(
            fingerprint
            for fingerprint in set(manifest_counts) & set(occurrence_counts)
            if manifest_counts[fingerprint] != occurrence_counts[fingerprint]
        )
        errors.append(
            {
                "error": "per-body occurrence counts do not match",
                "missing_fingerprints": missing[:20],
                "extra_fingerprints": extra[:20],
                "mismatched_fingerprints": mismatched[:20],
            }
        )

    objects_materialized = summary.get(
        "objects_materialized", (args.corpus / "objects").is_dir()
    )
    if objects_materialized:
        expected_objects = {
            (args.corpus / body["body_path"]).resolve()
            for body in bodies
            if body.get("extractable") and body.get("body_path")
        }
        objects_dir = args.corpus / "objects"
        actual_objects = (
            {path.resolve() for path in objects_dir.rglob("*.step")}
            if objects_dir.is_dir()
            else set()
        )
        if expected_objects != actual_objects:
            errors.append(
                {
                    "error": "object file set does not match body manifest",
                    "missing_objects": [
                        str(path.relative_to(args.corpus.resolve()))
                        for path in sorted(expected_objects - actual_objects)[:20]
                    ],
                    "extra_objects": [
                        str(path.relative_to(args.corpus.resolve()))
                        for path in sorted(actual_objects - expected_objects)[:20]
                    ],
                }
            )

    if len(bodies) != summary["unique_body_geometries"]:
        errors.append(
            {
                "error": "body count does not match summary",
                "bodies": len(bodies),
                "summary": summary["unique_body_geometries"],
            }
        )

    if objects_materialized:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for error in pool.map(
                lambda body: validate_one(args.corpus, args.body_tool, body), bodies
            ):
                if error is not None:
                    errors.append(error)

    result = {
        "bodies": len(bodies),
        "occurrences": len(occurrences),
        "extractable_bodies": sum(bool(body["extractable"]) for body in bodies),
        "validation_errors": len(errors),
        "errors": errors[:20],
    }
    print(json.dumps(result, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
