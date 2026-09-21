#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import zipfile
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_SOLID, TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

VIEWS = {
    "px": (1, 0, 0),
    "nx": (-1, 0, 0),
    "py": (0, 1, 0),
    "ny": (0, -1, 0),
    "pz": (0, 0, 1),
    "nz": (0, 0, -1),
    "iso_ppp": (1, 1, 1),
    "iso_npp": (-1, 1, 1),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def fetch_fixture(name: str, spec: dict, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    filename = spec.get("filename") or f"{name}.step"
    path = cache / filename
    expected = spec["sha256"].lower()
    if path.exists() and sha256_file(path) == expected:
        return path

    archive_kind = spec.get("archive")
    if archive_kind:
        archive_path = cache / spec.get(
            "archive_filename", filename + "." + archive_kind
        )
        archive_expected = spec.get("archive_sha256", "").lower()
        archive_ok = archive_path.exists()
        if archive_ok and archive_expected:
            archive_ok = sha256_file(archive_path) == archive_expected
        if not archive_ok:
            tmp = archive_path.with_suffix(archive_path.suffix + ".part")
            tmp.unlink(missing_ok=True)
            req = Request(
                spec["url"],
                headers={"User-Agent": "step-redox-validation/0.1"},
            )
            with urlopen(req, timeout=120) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            if archive_expected:
                got = sha256_file(tmp)
                if got != archive_expected:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"fixture {name}: archive SHA-256 mismatch: "
                        f"expected {archive_expected}, got {got}"
                    )
            os.replace(tmp, archive_path)

        if archive_kind != "zip":
            raise RuntimeError(f"fixture {name}: unsupported archive {archive_kind!r}")
        member = spec["member"]
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.unlink(missing_ok=True)
        with zipfile.ZipFile(archive_path) as zf:
            with zf.open(member) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        got = sha256_file(tmp)
        if got != expected:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"fixture {name}: extracted STEP SHA-256 mismatch: "
                f"expected {expected}, got {got}"
            )
        os.replace(tmp, path)
        return path

    tmp = path.with_suffix(path.suffix + ".part")
    tmp.unlink(missing_ok=True)
    req = Request(spec["url"], headers={"User-Agent": "step-redox-validation/0.1"})
    with urlopen(req, timeout=120) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    got = sha256_file(tmp)
    if got != expected:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"fixture {name}: SHA-256 mismatch: expected {expected}, got {got}"
        )
    os.replace(tmp, path)
    return path


def count(shape, kind) -> int:
    ex = TopExp_Explorer(shape, kind)
    n = 0
    while ex.More():
        n += 1
        ex.Next()
    return n


def load_step(path: Path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"{path}: STEPControl_Reader failed: {status}")
    roots = reader.NbRootsForTransfer()
    transferred = reader.TransferRoots()
    return reader.OneShape(), roots, transferred


def props_vec(p):
    return [float(p.X()), float(p.Y()), float(p.Z())]


def inertia_matrix(props: GProp_GProps):
    m = props.MatrixOfInertia()
    return [[float(m.Value(i, j)) for j in range(1, 4)] for i in range(1, 4)]


def bbox_values(box: Bnd_Box):
    # cadquery-ocp bindings changed the Python surface of Bnd_Box across
    # OpenCascade releases. Support the forms used by both our local OCP 8
    # validator and the public OCP 7.9 wheels used in GitHub Actions.
    scalar_names = ("GetXMin", "GetYMin", "GetZMin", "GetXMax", "GetYMax", "GetZMax")
    if all(hasattr(box, name) for name in scalar_names):
        return [float(getattr(box, name)()) for name in scalar_names]
    if hasattr(box, "CornerMin") and hasattr(box, "CornerMax"):
        lo = box.CornerMin()
        hi = box.CornerMax()
        return [
            float(lo.X()), float(lo.Y()), float(lo.Z()),
            float(hi.X()), float(hi.Y()), float(hi.Z()),
        ]
    if hasattr(box, "Get"):
        values = box.Get()
        if len(values) >= 6:
            return [float(value) for value in values[:6]]
    raise RuntimeError("unsupported OCP Bnd_Box binding")

def analyze(path: Path):
    shape, roots, transferred = load_step(path)
    volume = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, volume)
    surface = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, surface)
    linear = GProp_GProps()
    BRepGProp.LinearProperties_s(shape, linear)
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box, True)
    bbox = bbox_values(box)
    center = [(bbox[i] + bbox[i + 3]) * 0.5 for i in range(3)]
    extent = [bbox[i + 3] - bbox[i] for i in range(3)]
    report = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "roots": int(roots),
        "transferred": int(transferred),
        "brep_valid": bool(BRepCheck_Analyzer(shape).IsValid()),
        "volume": float(volume.Mass()),
        "area": float(surface.Mass()),
        "edge_length": float(linear.Mass()),
        "center_of_mass": props_vec(volume.CentreOfMass()),
        "inertia": inertia_matrix(volume),
        "bbox": bbox,
        "bbox_center": center,
        "bbox_extent": extent,
        "faces": count(shape, TopAbs_FACE),
        "edges": count(shape, TopAbs_EDGE),
        "vertices": count(shape, TopAbs_VERTEX),
        "solids": count(shape, TopAbs_SOLID),
        "shells": count(shape, TopAbs_SHELL),
    }
    return report, shape


def tessellate(shape, linear=0.03, angular=0.3) -> np.ndarray:
    BRepMesh_IncrementalMesh(shape, linear, False, angular, True)
    faces = TopExp_Explorer(shape, TopAbs_FACE)
    tris = []
    while faces.More():
        face = TopoDS.Face(faces.Current())
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            for i in range(1, tri.NbTriangles() + 1):
                pts = []
                for node in tri.Triangle(i).Get():
                    p = tri.Node(node).Transformed(trsf)
                    pts.append((p.X(), p.Y(), p.Z()))
                tris.append(pts)
        faces.Next()
    if not tris:
        return np.empty((0, 3, 3), dtype=np.float64)
    return np.asarray(tris, dtype=np.float64)


def strict_mesh_hash(tris: np.ndarray, quantum_mm=1e-9) -> str:
    scale = 1.0 / quantum_mm
    packed = []
    for tri in tris:
        pts = sorted(tuple(int(round(float(x) * scale)) for x in p) for p in tri)
        packed.append(tuple(pts))
    packed.sort()
    h = hashlib.sha256()
    for tri in packed:
        for p in tri:
            h.update(struct.pack("<qqq", *p))
    return h.hexdigest()


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero view vector")
    return v / n


def camera_basis(direction):
    d = unit(direction)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(d, up))) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    u = unit(np.cross(up, d))
    v = unit(np.cross(d, u))
    return u, v, d


def projection_frame(ref_tris: np.ndarray, direction, margin=0.06):
    u, v, d = camera_basis(direction)
    flat = ref_tris.reshape(-1, 3)
    x = flat @ u
    y = flat @ v
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    width = max(xmax - xmin, 1e-9)
    height = max(ymax - ymin, 1e-9)
    return (u, v, d), (
        xmin - width * margin, xmax + width * margin,
        ymin - height * margin, ymax + height * margin,
    )


def render_triangles(tris: np.ndarray, frame, size=320, supersample=2):
    (u, v, d), (xmin, xmax, ymin, ymax) = frame
    n = size * supersample
    sx = (n - 1) / max(xmax - xmin, 1e-12)
    sy = (n - 1) / max(ymax - ymin, 1e-12)
    px = tris @ u
    py = tris @ v
    pz = tris @ d
    centers = pz.mean(axis=1)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nl = np.linalg.norm(normals, axis=1)
    valid = nl > 1e-18
    normals[valid] /= nl[valid, None]
    shade = np.clip(0.28 + 0.67 * np.abs(normals @ d), 0.0, 1.0)
    order = np.argsort(centers)
    image = Image.new("L", (n, n), 255)
    draw = ImageDraw.Draw(image)
    for i in order:
        coords = [
            (
                int(round((float(px[i, j]) - xmin) * sx)),
                int(round((ymax - float(py[i, j])) * sy)),
            )
            for j in range(3)
        ]
        value = int(round(235 - 190 * float(shade[i])))
        draw.polygon(coords, fill=value)
    if supersample != 1:
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def align_translation(ref: dict, cand: dict, mode: str):
    if mode == "identity":
        return np.zeros(3, dtype=np.float64)
    if mode == "bbox_center":
        return np.asarray(ref["bbox_center"]) - np.asarray(cand["bbox_center"])
    if mode == "center_of_mass":
        return np.asarray(ref["center_of_mass"]) - np.asarray(cand["center_of_mass"])
    raise ValueError(f"unknown alignment mode {mode!r}")


def rel_error(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def compare_reports(ref: dict, cand: dict, alignment: str, thresholds: dict):
    shift = align_translation(ref, cand, alignment)
    checks = []

    def add(name, value, limit, unit=""):
        checks.append({
            "name": name, "value": value, "limit": limit, "unit": unit,
            "pass": value <= limit,
        })

    add("brep_valid", 0 if cand["brep_valid"] else 1, 0)

    if thresholds.get("topology_exact", False):
        for key in ["solids", "shells", "faces", "edges", "vertices"]:
            checks.append({
                "name": f"{key}_exact",
                "value": cand[key] - ref[key],
                "reference": ref[key],
                "candidate": cand[key],
                "limit": 0,
                "pass": cand[key] == ref[key],
            })

    for metric, config_key in [
        ("volume", "volume_rel"),
        ("area", "area_rel"),
        ("edge_length", "edge_length_rel"),
    ]:
        if config_key in thresholds:
            add(config_key, rel_error(ref[metric], cand[metric]), thresholds[config_key])

    if "bbox_extent_abs_mm" in thresholds:
        delta = np.max(np.abs(
            np.asarray(ref["bbox_extent"]) - np.asarray(cand["bbox_extent"])
        ))
        add("bbox_extent_abs_mm", float(delta), thresholds["bbox_extent_abs_mm"], "mm")

    if "center_abs_mm" in thresholds:
        adjusted = np.asarray(cand["bbox_center"]) + shift
        delta = np.linalg.norm(np.asarray(ref["bbox_center"]) - adjusted)
        add("center_abs_mm", float(delta), thresholds["center_abs_mm"], "mm")

    if "com_abs_mm" in thresholds:
        adjusted = np.asarray(cand["center_of_mass"]) + shift
        delta = np.linalg.norm(np.asarray(ref["center_of_mass"]) - adjusted)
        add("com_abs_mm", float(delta), thresholds["com_abs_mm"], "mm")

    if "inertia_rel" in thresholds:
        a = np.asarray(ref["inertia"])
        b = np.asarray(cand["inertia"])
        denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-30)
        add("inertia_rel", float(np.linalg.norm(a - b) / denom), thresholds["inertia_rel"])

    return shift, checks


def render_compare(ref_shape, cand_shape, shift, render_dir: Path, thresholds: dict, views):
    render_dir.mkdir(parents=True, exist_ok=True)
    ref_tris = tessellate(ref_shape)
    cand_tris = tessellate(cand_shape) + np.asarray(shift)[None, None, :]
    results = []
    for view_name in views:
        frame = projection_frame(ref_tris, VIEWS[view_name])
        ref_img = render_triangles(ref_tris, frame)
        cand_img = render_triangles(cand_tris, frame)
        diff = ImageChops.difference(ref_img, cand_img)
        ref_path = render_dir / f"{view_name}.reference.png"
        cand_path = render_dir / f"{view_name}.candidate.png"
        diff_path = render_dir / f"{view_name}.diff.png"
        ref_img.save(ref_path)
        cand_img.save(cand_path)
        diff.save(diff_path)
        arr = np.asarray(diff, dtype=np.float64)
        mean_abs = float(arr.mean() / 255.0)
        changed_fraction = float((arr > 8).mean())
        ref_sil = np.asarray(ref_img) < 250
        cand_sil = np.asarray(cand_img) < 250
        silhouette_xor = float(np.logical_xor(ref_sil, cand_sil).mean())
        result = {
            "view": view_name,
            "mean_abs": mean_abs,
            "changed_fraction": changed_fraction,
            "silhouette_xor": silhouette_xor,
            "reference": str(ref_path),
            "candidate": str(cand_path),
            "diff": str(diff_path),
        }
        result["pass"] = (
            mean_abs <= thresholds.get("render_mean_abs", 1.0)
            and changed_fraction <= thresholds.get("render_changed_fraction", 1.0)
            and silhouette_xor <= thresholds.get("silhouette_xor", 1.0)
        )
        results.append(result)

    if results:
        thumb = 220
        label_h = 24
        sheet = Image.new("L", (thumb * 3, (thumb + label_h) * len(results)), 255)
        draw = ImageDraw.Draw(sheet)
        for row, result in enumerate(results):
            y = row * (thumb + label_h)
            draw.text((6, y + 5), result["view"], fill=0)
            for col, key in enumerate(["reference", "candidate", "diff"]):
                image = Image.open(result[key]).convert("L")
                image.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
                x = col * thumb + (thumb - image.width) // 2
                yy = y + label_h + (thumb - image.height) // 2
                sheet.paste(image, (x, yy))
        sheet.save(render_dir / "contact-sheet.png")
    return results


def write_summary(results, path: Path):
    lines = ["# STEP validation", ""]
    lines.append(f"**{sum(r['pass'] for r in results)}/{len(results)} cases passed.**")
    lines.append("")
    for result in results:
        lines.extend([
            f"## {'PASS' if result['pass'] else 'FAIL'}: {result['name']}",
            "",
            "|" + " Check | Value | Limit | Result |",
            "|---|---:|---:|:---:|",
        ])
        for check in result.get("checks", []):
            lines.append(
                f"| {check['name']} | {check.get('value','')} | "
                f"{check.get('limit','')} | {'PASS' if check['pass'] else 'FAIL'} |"
            )
        for view in result.get("renders", []):
            lines.append(
                f"| render:{view['view']}:mean | {view['mean_abs']:.6g} | "
                f"{result['thresholds'].get('render_mean_abs','')} | "
                f"{'PASS' if view['pass'] else 'FAIL'} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def run_manifest(
    manifest_path: Path,
    cache: Path,
    out: Path,
    step_redox: Path,
    step_count_resize: Path | None = None,
):
    manifest = json.loads(manifest_path.read_text())
    fixtures = {
        name: fetch_fixture(name, spec, cache)
        for name, spec in manifest["fixtures"].items()
    }
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for case in manifest.get("cases", []):
        if case.get("enabled", True) is False:
            continue
        name = case["name"]
        case_dir = out / name
        case_dir.mkdir(parents=True, exist_ok=True)
        input_path = fixtures[case["input_fixture"]]
        reference_path = fixtures[case.get("reference_fixture", case["input_fixture"])]
        output_path = case_dir / "candidate.step"
        values = {
            "step_redox": str(step_redox),
            "step_count_resize": str(step_count_resize) if step_count_resize else "",
            "input": str(input_path),
            "reference": str(reference_path),
            "output": str(output_path),
        }
        if any("{step_count_resize}" in part for part in case["command"]) and not step_count_resize:
            raise RuntimeError(
                f"case {name}: command requires --step-count-resize but none was supplied"
            )
        command = [part.format(**values) for part in case["command"]]
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = {
            "name": name,
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-10000:],
            "stderr": proc.stderr[-10000:],
            "thresholds": case.get("thresholds", {}),
        }
        if proc.returncode != 0 or not output_path.exists():
            result["pass"] = False
            result["error"] = "generator command failed"
            results.append(result)
            continue

        ref_report, ref_shape = analyze(reference_path)
        cand_report, cand_shape = analyze(output_path)
        shift, checks = compare_reports(
            ref_report, cand_report, case.get("alignment", "identity"),
            case.get("thresholds", {}),
        )
        renders = render_compare(
            ref_shape, cand_shape, shift, case_dir / "renders",
            case.get("thresholds", {}), case.get("views", list(VIEWS)),
        )
        if case.get("thresholds", {}).get("strict_mesh_hash", False):
            ref_hash = strict_mesh_hash(tessellate(ref_shape, 0.002, 0.2))
            cand_hash = strict_mesh_hash(tessellate(cand_shape, 0.002, 0.2))
            checks.append({
                "name": "strict_mesh_hash",
                "value": cand_hash,
                "reference": ref_hash,
                "limit": ref_hash,
                "pass": cand_hash == ref_hash,
            })
        result.update({
            "reference": ref_report,
            "candidate": cand_report,
            "alignment": case.get("alignment", "identity"),
            "alignment_translation_mm": shift.tolist(),
            "checks": checks,
            "renders": renders,
        })
        result["pass"] = all(c["pass"] for c in checks) and all(r["pass"] for r in renders)
        (case_dir / "report.json").write_text(json.dumps(result, indent=2))
        results.append(result)

    (out / "report.json").write_text(json.dumps(results, indent=2))
    write_summary(results, out / "summary.md")
    return 0 if all(result["pass"] for result in results) else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--cache", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--step-redox", type=Path, required=True)
    run.add_argument("--step-count-resize", type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument(
        "--alignment",
        choices=["identity", "bbox_center", "center_of_mass"],
        default="identity",
    )
    compare.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "run":
        raise SystemExit(
            run_manifest(
                args.manifest,
                args.cache,
                args.out,
                args.step_redox,
                args.step_count_resize,
            )
        )

    thresholds = {
        "topology_exact": True,
        "volume_rel": 2e-5,
        "area_rel": 1e-8,
        "edge_length_rel": 1e-8,
        "bbox_extent_abs_mm": 1e-6,
        "center_abs_mm": 1e-6,
        "com_abs_mm": 2.5e-4,
        "inertia_rel": 2e-5,
        "render_mean_abs": 0.004,
        "render_changed_fraction": 0.015,
        "silhouette_xor": 0.0015,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    ref_report, ref_shape = analyze(args.reference)
    cand_report, cand_shape = analyze(args.candidate)
    shift, checks = compare_reports(ref_report, cand_report, args.alignment, thresholds)
    renders = render_compare(
        ref_shape, cand_shape, shift, args.out / "renders", thresholds, list(VIEWS)
    )
    report = {
        "reference": ref_report,
        "candidate": cand_report,
        "alignment": args.alignment,
        "alignment_translation_mm": shift.tolist(),
        "checks": checks,
        "renders": renders,
        "pass": all(c["pass"] for c in checks) and all(r["pass"] for r in renders),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
