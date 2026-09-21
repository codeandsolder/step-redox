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
        "--cad-fragments-json", str(prefix.with_suffix(".cad-fragments.json")),
        "--formed-sheet-json", str(prefix.with_suffix(".formed-sheet.json")),
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


def validate_cad_fragments(name: str, fragments: list[dict], expected: dict) -> dict:
    expected_count = expected["linear_patterns"]
    if len(fragments) != expected_count:
        raise AssertionError(
            f"{name}: expected {expected_count} CAD fragments, got {len(fragments)}"
        )

    expected_instances = expected["instances_per_pattern"]
    expected_step = expected["step_mm"]
    max_residual = expected["max_residual_mm"]
    observed = []
    for index, fragment in enumerate(fragments):
        source = fragment.get("source", {})
        if source.get("kind") != "instance_pattern":
            raise AssertionError(f"{name}: fragment {index} is not instance_pattern source")

        model = fragment["model"]
        root = fragment["root"]
        if model.get("roots") != [root]:
            raise AssertionError(f"{name}: fragment {index} has unexpected roots {model.get('roots')}")
        try:
            root_node = model["nodes"][root]["Pattern"]
            linear = root_node["pattern"]["Linear"]
        except (KeyError, TypeError, IndexError) as exc:
            raise AssertionError(f"{name}: fragment {index} root is not a linear pattern") from exc

        if linear["count"] != expected_instances:
            raise AssertionError(
                f"{name}: fragment {index} count {linear['count']} != {expected_instances}"
            )
        step = linear["step_mm"]
        if len(step) != 3 or any(not close(a, b, 1e-12) for a, b in zip(step, expected_step)):
            raise AssertionError(
                f"{name}: fragment {index} step {step} != canonical {expected_step}"
            )

        child = root_node["child"]
        try:
            fallback = model["nodes"][child]["BrepFallback"]
        except (KeyError, TypeError, IndexError) as exc:
            raise AssertionError(
                f"{name}: fragment {index} child is not exact B-rep fallback"
            ) from exc
        if len(fallback.get("source_entity_ids", [])) != 1:
            raise AssertionError(
                f"{name}: fragment {index} fallback does not preserve exactly one source occurrence"
            )

        provenance = model["provenance"][str(root)]
        if provenance["proof"] != "WithinTolerance":
            raise AssertionError(
                f"{name}: fragment {index} unexpectedly claims {provenance['proof']} proof"
            )
        residual = provenance["max_residual_mm"]
        if not math.isfinite(residual) or residual > max_residual:
            raise AssertionError(
                f"{name}: fragment {index} residual {residual} exceeds {max_residual}"
            )

        observed.append({
            "count": linear["count"],
            "step_mm": step,
            "max_residual_mm": residual,
            "fallback_entity": fallback["source_entity_ids"][0],
        })

    return {
        "fixture": name,
        "fragments": len(fragments),
        "patterns": observed,
    }


def validate_formed_sheet(name: str, candidates: list[dict], expected: dict) -> dict:
    if len(candidates) != expected["candidates"]:
        raise AssertionError(
            f"{name}: expected {expected['candidates']} formed-sheet candidates, got {len(candidates)}"
        )

    observed = []
    for index, candidate in enumerate(candidates):
        checks = {
            "thickness_mm": close(candidate["thickness_mm"], expected["thickness_mm"], 1e-9),
            "cylindrical_faces": candidate["cylindrical_faces"] == expected["cylindrical_faces"],
            "paired_cylindrical_faces": candidate["paired_cylindrical_faces"] == expected["paired_cylindrical_faces"],
            "coaxial_radius_pairs": candidate["coaxial_radius_pairs"] == expected["coaxial_radius_pairs"],
            "parallel_plane_pairs": candidate["parallel_plane_pairs"] == expected["parallel_plane_pairs"],
        }
        failed = [key for key, ok in checks.items() if not ok]
        if failed:
            raise AssertionError(
                f"{name}: formed-sheet candidate {index} failed {failed}; observed={candidate}"
            )
        expected_ratio = expected["paired_cylindrical_faces"] / expected["cylindrical_faces"]
        if not close(candidate["paired_cylinder_face_ratio"], expected_ratio, 1e-12):
            raise AssertionError(
                f"{name}: formed-sheet candidate {index} paired ratio "
                f"{candidate['paired_cylinder_face_ratio']} != {expected_ratio}"
            )
        observed.append({
            "solid_id": candidate["solid_id"],
            "thickness_mm": candidate["thickness_mm"],
            "paired_cylinder_face_ratio": candidate["paired_cylinder_face_ratio"],
            "coaxial_radius_pairs": candidate["coaxial_radius_pairs"],
            "parallel_plane_pairs": candidate["parallel_plane_pairs"],
        })

    return {"fixture": name, "candidates": observed}


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
    chain_results = []
    cad_results = []
    sheet_results = []
    args.out.mkdir(parents=True, exist_ok=True)

    for name, spec in manifest["fixtures"].items():
        chain_expected = spec.get("periodic_chain_expectation")
        cad_expected = spec.get("cad_fragment_expectation")
        sheet_expected = spec.get("formed_sheet_expectation")
        if not chain_expected and not cad_expected and not sheet_expected:
            continue
        input_path = fixture_path(args.cache, spec)
        if not input_path.exists():
            raise FileNotFoundError(
                f"{name}: expected cached fixture {input_path}; run geometry validation first"
            )
        prefix = args.out / name
        run_probe(args.step_redox, input_path, prefix)

        if chain_expected:
            chains = json.loads(prefix.with_suffix(".chains.json").read_text())
            result = validate_chain(name, chains, chain_expected)
            chain_results.append(result)
            print(
                f"{name}: sites={result['sites']} pitch={result['pitch_mm']:.9f} mm "
                f"cell={result['interior_site_face_count']}+{result['interior_gap_face_count']}="
                f"{result['faces_per_added_site']} faces "
                f"votes={result['lattice_face_family_votes']} "
                f"coverage={result['repeat_coverage_ratio']:.3%} proven"
            )

        if cad_expected:
            fragments = json.loads(prefix.with_suffix(".cad-fragments.json").read_text())
            result = validate_cad_fragments(name, fragments, cad_expected)
            cad_results.append(result)
            first = result["patterns"][0]
            print(
                f"{name}: {result['fragments']} canonical CAD linear patterns x "
                f"{first['count']} at step={first['step_mm']} "
                f"residual<={max(p['max_residual_mm'] for p in result['patterns']):.3g} mm"
            )

        if sheet_expected:
            candidates = json.loads(prefix.with_suffix(".formed-sheet.json").read_text())
            result = validate_formed_sheet(name, candidates, sheet_expected)
            sheet_results.append(result)
            first = result["candidates"][0]
            print(
                f"{name}: {len(result['candidates'])} formed-sheet candidates "
                f"t={first['thickness_mm']} mm, "
                f"paired={first['paired_cylinder_face_ratio']:.1%}, "
                f"bend_pairs={first['coaxial_radius_pairs']}"
            )

    if not chain_results and not cad_results and not sheet_results:
        raise RuntimeError("manifest contains no semantic expectations")

    if chain_results:
        (args.out / "periodic-chain-summary.json").write_text(
            json.dumps(chain_results, indent=2) + "\n"
        )

        # Family-level cross-check: the independently published siblings should all
        # recover the same pitch and per-added-site topology law.
        pitches = {round(r["pitch_mm"], 9) for r in chain_results}
        slopes = {r["faces_per_added_site"] for r in chain_results}
        if len(pitches) != 1 or len(slopes) != 1:
            raise AssertionError(
                f"periodic family is inconsistent across fixtures: pitches={pitches}, slopes={slopes}"
            )

    if cad_results:
        (args.out / "cad-fragment-summary.json").write_text(
            json.dumps(cad_results, indent=2) + "\n"
        )

    if sheet_results:
        (args.out / "formed-sheet-summary.json").write_text(
            json.dumps(sheet_results, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
