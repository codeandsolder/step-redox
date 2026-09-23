import argparse
import json
import math
import re
from pathlib import Path

from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS


ABS_GEOM_TOL = 1.0e-9
ABS_VOLUME_TOL = 1.0e-10
ABS_AREA_TOL = 1.0e-9
REL_TOL = 1.0e-10


def load(path):
    reader = STEPControl_Reader()
    assert reader.ReadFile(path) == IFSelect_RetDone
    assert reader.TransferRoots() > 0
    return reader.OneShape()


def entity_rank_for_step_id(path, entity_id):
    # XSControl's TransferOne() takes the 1-based model rank, not the textual
    # STEP #id. Resolve explicitly so sparse/reordered entity IDs remain safe.
    pattern = re.compile(rb"(?m)^#(\d+)\s*=")
    for rank, match in enumerate(pattern.finditer(Path(path).read_bytes()), start=1):
        if int(match.group(1)) == entity_id:
            return rank
    raise ValueError(f"STEP entity #{entity_id} not found in {path}")


def load_entity_id(path, entity_id):
    rank = entity_rank_for_step_id(path, entity_id)
    reader = STEPControl_Reader()
    assert reader.ReadFile(path) == IFSelect_RetDone
    assert reader.TransferOne(rank)
    assert reader.NbShapes() == 1
    return reader.Shape(1), rank


def count(shape, kind):
    explorer = TopExp_Explorer(shape, kind)
    n = 0
    while explorer.More():
        n += 1
        explorer.Next()
    return n


def prop(solid):
    volume = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, volume)
    area = GProp_GProps()
    BRepGProp.SurfaceProperties_s(solid, area)
    center = volume.CentreOfMass()
    box = Bnd_Box()
    BRepBndLib.Add_s(solid, box, True)
    return {
        "volume": float(volume.Mass()),
        "area": float(area.Mass()),
        "center": [float(center.X()), float(center.Y()), float(center.Z())],
        "bbox": [float(value) for value in box.Get()],
        "faces": count(solid, TopAbs_FACE),
        "edges": count(solid, TopAbs_EDGE),
        "vertices": count(solid, TopAbs_VERTEX),
    }


def solids_from_shape(shape):
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    out = []
    while explorer.More():
        out.append(prop(TopoDS.Solid(explorer.Current())))
        explorer.Next()

    # STEP mapped-item transforms can introduce ~1e-12 coordinate roundoff.
    # Pair solids by a coarser geometric key, then compare with explicit
    # tolerances instead of requiring decimal-string identity.
    return sorted(
        out,
        key=lambda x: (
            tuple(round(v, 9) for v in x["center"]),
            round(x["volume"], 9),
            round(x["area"], 9),
            tuple(round(v, 9) for v in x["bbox"]),
        ),
    )


def solids(path):
    return solids_from_shape(load(path))


def cut_volume(a, b):
    operation = BRepAlgoAPI_Cut(a, b)
    operation.Build()
    assert operation.IsDone()
    volume = GProp_GProps()
    BRepGProp.VolumeProperties_s(operation.Shape(), volume)
    return abs(float(volume.Mass()))


def volume_tolerance(a, b):
    return max(ABS_VOLUME_TOL, REL_TOL * max(abs(a), abs(b)))


def close(a, b, abs_tol):
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=abs_tol)


def compare_solid(a, b):
    errors = {}
    if not close(a["volume"], b["volume"], ABS_VOLUME_TOL):
        errors["volume"] = abs(a["volume"] - b["volume"])
    if not close(a["area"], b["area"], ABS_AREA_TOL):
        errors["area"] = abs(a["area"] - b["area"])

    center_error = max(abs(x - y) for x, y in zip(a["center"], b["center"]))
    if center_error > ABS_GEOM_TOL:
        errors["center"] = center_error

    bbox_error = max(abs(x - y) for x, y in zip(a["bbox"], b["bbox"]))
    if bbox_error > ABS_GEOM_TOL:
        errors["bbox"] = bbox_error

    for field in ("faces", "edges", "vertices"):
        if a[field] != b[field]:
            errors[field] = [a[field], b[field]]

    return errors


parser = argparse.ArgumentParser()
parser.add_argument("a")
parser.add_argument("b")
parser.add_argument(
    "--a-entity-id",
    type=int,
    help=(
        "compare one exact textual STEP entity ID (#N) from A against B. "
        "Useful when A contains repeated/overlapping solids; avoids heuristic pairing."
    ),
)
args = parser.parse_args()

if args.a_entity_id is not None:
    a_shape, a_entity_rank = load_entity_id(args.a, args.a_entity_id)
    b_shape = load(args.b)
    a_solids = solids_from_shape(a_shape)
    b_solids = solids_from_shape(b_shape)
    assert len(a_solids) == 1
    assert len(b_solids) == 1
    a_solid = a_solids[0]
    b_solid = b_solids[0]

    volume_error = abs(a_solid["volume"] - b_solid["volume"])
    area_error = abs(a_solid["area"] - b_solid["area"])
    center_error = max(
        abs(x - y) for x, y in zip(a_solid["center"], b_solid["center"])
    )
    bbox_error = max(abs(x - y) for x, y in zip(a_solid["bbox"], b_solid["bbox"]))
    source_minus_candidate = cut_volume(a_shape, b_shape)
    candidate_minus_source = cut_volume(b_shape, a_shape)
    boolean_volume_tolerance = volume_tolerance(a_solid["volume"], b_solid["volume"])

    # Analytic OCCT bounding boxes can be conservative (notably trimmed tori),
    # and CAD kernels are free to split faces differently. For entity-rank
    # validation, occupied geometry is the contract: mass properties plus both
    # directional Boolean residuals must agree.
    geometry_equal = (
        close(a_solid["volume"], b_solid["volume"], ABS_VOLUME_TOL)
        and close(a_solid["area"], b_solid["area"], ABS_AREA_TOL)
        and center_error <= ABS_GEOM_TOL
        and source_minus_candidate <= boolean_volume_tolerance
        and candidate_minus_source <= boolean_volume_tolerance
    )
    print(
        json.dumps(
            {
                "a_entity_id": args.a_entity_id,
                "a_entity_rank": a_entity_rank,
                "a": a_solid,
                "b": b_solid,
                "geometry_equal": geometry_equal,
                "errors": {
                    "volume": volume_error,
                    "area": area_error,
                    "center": center_error,
                    "bbox_report_only": bbox_error,
                },
                "boolean_residual_volume": {
                    "a_minus_b": source_minus_candidate,
                    "b_minus_a": candidate_minus_source,
                },
                "tolerances": {
                    "geometry_abs": ABS_GEOM_TOL,
                    "volume_abs": ABS_VOLUME_TOL,
                    "boolean_volume_effective": boolean_volume_tolerance,
                    "area_abs": ABS_AREA_TOL,
                    "relative": REL_TOL,
                },
            },
            indent=2,
        )
    )
    raise SystemExit(0 if geometry_equal else 1)

a_solids = solids(args.a)
b_solids = solids(args.b)
mismatches = []

if len(a_solids) != len(b_solids):
    mismatches.append({"solid_count": [len(a_solids), len(b_solids)]})
else:
    for index, (a_solid, b_solid) in enumerate(zip(a_solids, b_solids)):
        errors = compare_solid(a_solid, b_solid)
        if errors:
            mismatches.append(
                {
                    "index": index,
                    "errors": errors,
                    "a": a_solid,
                    "b": b_solid,
                }
            )

print(
    json.dumps(
        {
            "a": a_solids,
            "b": b_solids,
            "equal": not mismatches,
            "mismatches": mismatches,
            "tolerances": {
                "geometry_abs": ABS_GEOM_TOL,
                "volume_abs": ABS_VOLUME_TOL,
                "area_abs": ABS_AREA_TOL,
                "relative": REL_TOL,
            },
        },
        indent=2,
    )
)
