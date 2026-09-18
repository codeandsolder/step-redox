import json
import math
import sys

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
    reader.TransferRoots()
    return reader.OneShape()


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
        "bbox": [
            float(box.GetXMin()),
            float(box.GetYMin()),
            float(box.GetZMin()),
            float(box.GetXMax()),
            float(box.GetYMax()),
            float(box.GetZMax()),
        ],
        "faces": count(solid, TopAbs_FACE),
        "edges": count(solid, TopAbs_EDGE),
        "vertices": count(solid, TopAbs_VERTEX),
    }


def solids(path):
    shape = load(path)
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


a_path, b_path = sys.argv[1:3]
a_solids = solids(a_path)
b_solids = solids(b_path)
mismatches = []

if len(a_solids) != len(b_solids):
    mismatches.append(
        {"solid_count": [len(a_solids), len(b_solids)]}
    )
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
