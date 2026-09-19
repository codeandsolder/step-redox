import argparse, hashlib, json, struct
from pathlib import Path

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


def count(shape, kind):
    ex = TopExp_Explorer(shape, kind)
    n = 0
    while ex.More():
        n += 1
        ex.Next()
    return n


def q(x, scale=1e9):
    return int(round(float(x) * scale))


def mesh_hash(shape, linear=0.002, angular=0.2):
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
                t = tri.Triangle(i)
                ids = t.Get()
                pts = []
                for ni in ids:
                    p = tri.Node(ni).Transformed(trsf)
                    pts.append((q(p.X()), q(p.Y()), q(p.Z())))
                pts.sort()
                tris.append(tuple(pts))
        faces.Next()
    tris.sort()
    h = hashlib.sha256()
    for tri in tris:
        for p in tri:
            h.update(struct.pack("<qqq", *p))
    return h.hexdigest(), len(tris)


def load(path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"ReadFile failed: {status}")
    roots = reader.NbRootsForTransfer()
    transferred = reader.TransferRoots()
    return reader.OneShape(), roots, transferred


def analyze(path):
    shape, roots, transferred = load(path)
    vol = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, vol)
    surf = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, surf)
    linear = GProp_GProps()
    BRepGProp.LinearProperties_s(shape, linear)
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box, True)
    bbox = [box.GetXMin(), box.GetYMin(), box.GetZMin(), box.GetXMax(), box.GetYMax(), box.GetZMax()]
    mh, ntri = mesh_hash(shape)
    return {
        "path": str(path),
        "bytes": Path(path).stat().st_size,
        "roots": roots,
        "transferred": transferred,
        "volume": vol.Mass(),
        "area": surf.Mass(),
        "edge_length": linear.Mass(),
        "bbox": bbox,
        "faces": count(shape, TopAbs_FACE),
        "edges": count(shape, TopAbs_EDGE),
        "vertices": count(shape, TopAbs_VERTEX),
        "solids": count(shape, TopAbs_SOLID),
        "shells": count(shape, TopAbs_SHELL),
        "triangles": ntri,
        "mesh_hash": mh,
        "brep_valid": bool(BRepCheck_Analyzer(shape).IsValid()),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    print(json.dumps([analyze(Path(p)) for p in args.paths], indent=2))
