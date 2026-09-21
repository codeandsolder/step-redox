#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def fixture_path(cache: Path, spec: dict) -> Path:
    return cache / spec.get("filename", "")


def run_probe(step_redox: Path, input_path: Path, prefix: Path):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(step_redox),
        "--profile", "compact",
        str(input_path),
        str(prefix.with_suffix(".compact.step")),
        "--patterns-json", str(prefix.with_suffix(".patterns.json")),
        "--periodic-bodies-json", str(prefix.with_suffix(".bodies.json")),
        "--count-parameters-json", str(prefix.with_suffix(".counts.json")),
        "--periodic-chains-json", str(prefix.with_suffix(".chains.json")),
        "--json",
    ]
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    prefix.with_suffix(".stdout.txt").write_text(proc.stdout)
    prefix.with_suffix(".stats.json").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{input_path.name}: step-redox semantic probe failed ({proc.returncode})\n"
            f"{proc.stderr[-4000:]}"
        )


def close(a: float, b: float, tol: float = 1e-8) -> bool:
    return math.isfinite(a) and abs(a - b) <= tol


def validate_chain(name: str, chains: list[dict], expected: dict) -> dict:
    candidates = [
        chain for chain in chains
        if chain.get("sites") == expected["sites"]
        and close(chain.get("pitch_mm", float("nan")), expected["pitch_mm"])
    ]
    if len(candidates) != 1:
        raise AssertionError(
            f"{name}: expected exactly one {expected['sites']}-site/"
            f"{expected['pitch_mm']} mm chain, got {len(candidates)} from "
            f"{[(c.get('sites'), c.get('pitch_mm')) for c in chains]}"
        )

    chain = candidates[0]
    checks = {
        "read_only_proven": chain["read_only_proven"] == expected["read_only_proven"],
        "complete_partition": chain["complete_partition"] == expected["complete_partition"],
        "faces_without_geometry": chain["faces_without_geometry"] == expected["faces_without_geometry"],
        "nonmanifold_edges": chain["nonmanifold_edges"] == expected["nonmanifold_edges"],
        "cross_site_edges": chain["cross_site_edges"] == expected["cross_site_edges"],
        "interior_site_face_count": chain["interior_site_face_count"] == expected["interior_site_face_count"],
        "interior_gap_face_count": chain["interior_gap_face_count"] == expected["interior_gap_face_count"],
        "stretch_faces": len(chain["stretch_face_ids"]) == expected["stretch_faces"],
        "fixed_negative_faces": len(chain["fixed_negative_face_ids"]) == expected["fixed_negative_faces"],
        "fixed_positive_faces": len(chain["fixed_positive_face_ids"]) == expected["fixed_positive_faces"],
    }
    failed = [key for key, ok in checks.items() if not ok]
    if failed:
        details = {key: chain.get(key) for key in failed}
        details["stretch_faces"] = len(chain["stretch_face_ids"])
        details["fixed_negative_faces"] = len(chain["fixed_negative_face_ids"])
        details["fixed_positive_faces"] = len(chain["fixed_positive_face_ids"])
        raise AssertionError(f"{name}: periodic-chain checks failed: {failed}; observed={details}")

    slope = chain["interior_site_face_count"] + chain["interior_gap_face_count"]
    return {
        "fixture": name,
        "sites": chain["sites"],
        "pitch_mm": chain["pitch_mm"],
        "lattice_face_family_votes": chain["lattice_face_family_votes"],
        "interior_site_face_count": chain["interior_site_face_count"],
        "interior_gap_face_count": chain["interior_gap_face_count"],
        "faces_per_added_site": slope,
        "stretch_faces": len(chain["stretch_face_ids"]),
        "fixed_negative_faces": len(chain["fixed_negative_face_ids"]),
        "fixed_positive_faces": len(chain["fixed_positive_face_ids"]),
        "repeat_coverage_ratio": chain["repeat_coverage_ratio"],
        "read_only_proven": chain["read_only_proven"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--step-redox", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    results = []
    args.out.mkdir(parents=True, exist_ok=True)

    for name, spec in manifest["fixtures"].items():
        expected = spec.get("periodic_chain_expectation")
        if not expected:
            continue
        input_path = fixture_path(args.cache, spec)
        if not input_path.exists():
            raise FileNotFoundError(
                f"{name}: expected cached fixture {input_path}; run geometry validation first"
            )
        prefix = args.out / name
        run_probe(args.step_redox, input_path, prefix)
        chains = json.loads(prefix.with_suffix(".chains.json").read_text())
        result = validate_chain(name, chains, expected)
        results.append(result)
        print(
            f"{name}: sites={result['sites']} pitch={result['pitch_mm']:.9f} mm "
            f"cell={result['interior_site_face_count']}+{result['interior_gap_face_count']}="
            f"{result['faces_per_added_site']} faces "
            f"votes={result['lattice_face_family_votes']} "
            f"coverage={result['repeat_coverage_ratio']:.3%} proven"
        )

    if not results:
        raise RuntimeError("manifest contains no periodic_chain_expectation fixtures")

    (args.out / "periodic-chain-summary.json").write_text(json.dumps(results, indent=2) + "\n")

    # Family-level cross-check: the independently published siblings should all
    # recover the same pitch and per-added-site topology law.
    pitches = {round(r["pitch_mm"], 9) for r in results}
    slopes = {r["faces_per_added_site"] for r in results}
    if len(pitches) != 1 or len(slopes) != 1:
        raise AssertionError(
            f"periodic family is inconsistent across fixtures: pitches={pitches}, slopes={slopes}"
        )


if __name__ == "__main__":
    main()
