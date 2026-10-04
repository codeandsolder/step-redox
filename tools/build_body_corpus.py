#!/usr/bin/env python3
"""Build a deduplicated single-body STEP corpus from a model manifest.

The expensive whole-model corpus remains the source of truth for array/assembly
operations. This derived corpus isolates one representative MANIFOLD_SOLID_BREP
per translation/Z-quarter-turn-invariant geometry identity so feature recovery
can iterate on bodies without reparsing unrelated model structure.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--body-tool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument(
        "--require-canonical",
        action="store_true",
        help="fail the build if any body uses fallback identity or is skipped",
    )
    return ap.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or not {"uuid", "path"}.issubset(rows[0]):
        raise SystemExit(f"{path}: expected TSV columns uuid and path")
    return rows


SI_PREFIX_METRES = {
    None: 1.0,
    "EXA": 1.0e18,
    "PETA": 1.0e15,
    "TERA": 1.0e12,
    "GIGA": 1.0e9,
    "MEGA": 1.0e6,
    "KILO": 1.0e3,
    "HECTO": 1.0e2,
    "DECA": 1.0e1,
    "DECI": 1.0e-1,
    "CENTI": 1.0e-2,
    "MILLI": 1.0e-3,
    "MICRO": 1.0e-6,
    "NANO": 1.0e-9,
    "PICO": 1.0e-12,
    "FEMTO": 1.0e-15,
    "ATTO": 1.0e-18,
}


def detect_model_length_scale_mm(path: str) -> float | None:
    data = Path(path).read_bytes()
    # Compatibility fallback for scanner binaries predating per-representation
    # unit reporting. Only accept one unambiguous SI length scale for the file.
    pattern = re.compile(
        rb"LENGTH_UNIT\s*\(\s*\).*?SI_UNIT\s*\(\s*(\$|\.[A-Z]+\.)\s*,\s*\.METRE\.\s*\)",
        re.DOTALL,
    )
    scales: set[float] = set()
    for match in pattern.finditer(data):
        raw = match.group(1)
        prefix = None if raw == b"$" else raw.strip(b".").decode("ascii")
        metres = SI_PREFIX_METRES.get(prefix)
        if metres is not None:
            scales.add(metres * 1000.0)
    return next(iter(scales)) if len(scales) == 1 else None


def scan_one(
    tool: Path,
    tool_sha256: str,
    cache_dir: Path,
    row: dict[str, str],
) -> dict:
    cache_path = cache_dir / f'{row["uuid"]}.json'
    try:
        cached = json.loads(cache_path.read_text())
        if (
            cached.get("body_tool_sha256") == tool_sha256
            and cached.get("path") == row["path"]
            and isinstance(cached.get("report"), dict)
        ):
            return cached["report"]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass

    proc = subprocess.run(
        [str(tool), "scan", row["path"]],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode:
        return {
            "uuid": row["uuid"],
            "path": row["path"],
            "error": proc.stderr[-8000:],
        }
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {
            "uuid": row["uuid"],
            "path": row["path"],
            "error": f"invalid JSON: {exc}: {proc.stdout[-1000:]}",
        }
    report["uuid"] = row["uuid"]
    report["path"] = row["path"]

    payload = {
        "body_tool_sha256": tool_sha256,
        "path": row["path"],
        "report": report,
    }
    tmp = cache_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")))
    tmp.replace(cache_path)
    return report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_fingerprint(uuid: str, solid: dict) -> str:
    geometry = solid.get("geometry_fingerprint")
    if geometry is None:
        # Canonical geometry analysis did not understand this body. Preserve it
        # in the corpus, but never deduplicate it with another occurrence.
        payload = (
            f'body-v2:fallback:{uuid}:{solid["data_section"]}:{solid["solid_id"]}'
        )
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    scale = solid.get("unit_scale_mm")
    if scale is None:
        # Unknown units are safe but deliberately file-local: never deduplicate
        # raw coordinates across two files unless their physical length scale is known.
        payload = f"body-v2:unknown:{uuid}:{geometry}"
    else:
        payload = f"body-v2:mm-scale:{float(scale).hex()}:{geometry}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def extract_group(tool: Path, out: Path, source: str, bodies: list[dict]) -> dict:
    command = [str(tool), "extract", source, str(out / "objects")]
    for body in bodies:
        rep = body["representative"]
        command.extend(
            [
                "--solid",
                f'{rep["data_section"]}:{rep["solid_id"]}:{body["fingerprint"]}',
            ]
        )
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return {
        "source": source,
        "requested": len(bodies),
        "returncode": proc.returncode,
        "stderr": proc.stderr[-8000:],
    }


def main() -> int:
    args = parse_args()
    rows = load_manifest(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)

    body_tool_sha256 = sha256_file(args.body_tool)
    source_manifest_sha256 = sha256_file(args.manifest)
    generator_sha256 = sha256_file(Path(__file__))

    previous_summary = None
    try:
        previous_summary = json.loads((args.out / "summary.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    same_build = bool(
        previous_summary
        and previous_summary.get("body_tool_sha256") == body_tool_sha256
        and previous_summary.get("source_manifest_sha256") == source_manifest_sha256
        and previous_summary.get("generator_sha256") == generator_sha256
    )

    if not same_build:
        objects = args.out / "objects"
        if objects.exists():
            shutil.rmtree(objects)

    for name in (
        "bodies.jsonl",
        "occurrences.jsonl",
        "model-scans.jsonl",
        "extraction-reports.json",
        "summary.json",
        "COMPLETE",
        "FAILED",
    ):
        try:
            (args.out / name).unlink()
        except FileNotFoundError:
            pass

    (args.out / "BUILDING").write_text(
        json.dumps(
            {
                "source_manifest": str(args.manifest),
                "source_manifest_sha256": source_manifest_sha256,
                "body_tool_sha256": body_tool_sha256,
                "generator_sha256": generator_sha256,
            },
            indent=2,
        )
    )

    scan_cache = args.out / "scan-cache"
    scan_cache.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        reports = list(
            pool.map(
                lambda row: scan_one(
                    args.body_tool,
                    body_tool_sha256,
                    scan_cache,
                    row,
                ),
                rows,
            )
        )

    scan_path = args.out / "model-scans.jsonl"
    with scan_path.open("w") as handle:
        for report in reports:
            handle.write(json.dumps(report, separators=(",", ":")) + "\n")

    failures = [report for report in reports if "error" in report]
    groups: dict[str, list[dict]] = defaultdict(list)
    total_manifold = total_analyzed = total_skipped = total_without_rep = 0
    total_fallback_identity = 0
    total_unknown_units = 0

    occurrence_path = args.out / "occurrences.jsonl"
    with occurrence_path.open("w") as occurrences:
        for report in reports:
            if "error" in report:
                continue
            total_manifold += report["manifold_solids"]
            total_analyzed += report["analyzed_solids"]
            total_skipped += report["skipped_solids"]
            total_fallback_identity += report.get("fallback_identity_solids", 0)
            total_without_rep += report["solids_without_shape_representation"]
            for solid in report["solids"]:
                total_unknown_units += int(solid.get("unit_scale_mm") is None)
                fingerprint = corpus_fingerprint(report["uuid"], solid)
                row = {
                    "fingerprint": fingerprint,
                    "uuid": report["uuid"],
                    "source_path": report["path"],
                    **solid,
                }
                groups[fingerprint].append(row)
                occurrences.write(json.dumps(row, separators=(",", ":")) + "\n")

    body_rows: list[dict] = []
    unextractable = 0
    for fingerprint, members in sorted(groups.items()):
        dimensions = {
            (
                member["vertices"],
                member["edges"],
                member["oriented_edges"],
                member["faces"],
            )
            for member in members
        }
        if len(dimensions) != 1:
            raise SystemExit(
                f"fingerprint collision/inconsistent body metrics for {fingerprint}: {dimensions}"
            )
        extractable = [m for m in members if m["has_shape_representation"]]
        representative = min(
            extractable or members,
            key=lambda m: (
                m["closure_entities"],
                m["uuid"],
                m["data_section"],
                m["solid_id"],
            ),
        )
        if not extractable:
            unextractable += 1
        shard = fingerprint[:2]
        body_rows.append(
            {
                "fingerprint": fingerprint,
                "identity": "translation+z90+locus+unit-v2",
                "occurrences": len(members),
                "extractable": bool(extractable),
                "body_path": f"objects/{shard}/{fingerprint}.step"
                if extractable
                else None,
                "geometry_fingerprint": representative["geometry_fingerprint"],
                "identity_kind": representative.get("identity_kind", "canonical"),
                "unit_scale_mm": representative.get("unit_scale_mm"),
                "vertices": representative["vertices"],
                "edges": representative["edges"],
                "oriented_edges": representative["oriented_edges"],
                "faces": representative["faces"],
                "representative": {
                    "uuid": representative["uuid"],
                    "source_path": representative["source_path"],
                    "data_section": representative["data_section"],
                    "solid_id": representative["solid_id"],
                    "unit_scale_mm": representative.get("unit_scale_mm"),
                    "center_source_units": representative["center_source_units"],
                    "canonical_quarter_turn": representative[
                        "canonical_quarter_turn"
                    ],
                    "closure_entities": representative["closure_entities"],
                },
            }
        )

    bodies_path = args.out / "bodies.jsonl"
    with bodies_path.open("w") as handle:
        for row in body_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    extraction_failures = []
    if not args.no_extract:
        by_source: dict[str, list[dict]] = defaultdict(list)
        for body in body_rows:
            if body["extractable"]:
                by_source[body["representative"]["source_path"]].append(body)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            extraction_reports = list(
                pool.map(
                    lambda item: extract_group(
                        args.body_tool, args.out, item[0], item[1]
                    ),
                    by_source.items(),
                )
            )
        extraction_failures = [
            report for report in extraction_reports if report["returncode"]
        ]
        (args.out / "extraction-reports.json").write_text(
            json.dumps(extraction_reports, indent=2)
        )

    extracted = sum(1 for body in body_rows if body["extractable"])
    summary = {
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": source_manifest_sha256,
        "body_tool": str(args.body_tool),
        "body_tool_sha256": body_tool_sha256,
        "generator_sha256": generator_sha256,
        "scan_cache": str(scan_cache),
        "source_models": len(rows),
        "scan_failures": len(failures),
        "manifold_solid_occurrences": total_manifold,
        "analyzed_solid_occurrences": total_analyzed,
        "skipped_solid_occurrences": total_skipped,
        "fallback_identity_solid_occurrences": total_fallback_identity,
        "solid_occurrences_without_shape_representation": total_without_rep,
        "solid_occurrences_with_unknown_length_units": total_unknown_units,
        "unique_body_geometries": len(body_rows),
        "unique_bodies_with_unknown_length_units": sum(
            body["unit_scale_mm"] is None for body in body_rows
        ),
        "duplicate_occurrences_removed": total_analyzed - len(body_rows),
        "materializable_unique_bodies": extracted,
        "extractable_unique_bodies": extracted,
        "unextractable_unique_bodies": unextractable,
        "objects_materialized": not args.no_extract,
        "extraction_failures": len(extraction_failures),
        "identity": "translation+z90+locus+unit-v2",
        "objects_dir": str(args.out / "objects"),
        "bodies_manifest": str(bodies_path),
        "occurrences_manifest": str(occurrence_path),
        "model_scans": str(scan_path),
    }
    summary_tmp = args.out / "summary.json.tmp"
    summary_tmp.write_text(json.dumps(summary, indent=2))
    summary_tmp.replace(args.out / "summary.json")

    canonicality_failure = bool(
        args.require_canonical and (total_fallback_identity or total_skipped)
    )
    failed = bool(failures or extraction_failures or canonicality_failure)
    try:
        (args.out / "BUILDING").unlink()
    except FileNotFoundError:
        pass
    marker = args.out / ("FAILED" if failed else "COMPLETE")
    marker.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
