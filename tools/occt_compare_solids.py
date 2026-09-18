import json, sys
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS


def load(path):
    r=STEPControl_Reader()
    assert r.ReadFile(path)==IFSelect_RetDone
    r.TransferRoots()
    return r.OneShape()


def count(shape, kind):
    e=TopExp_Explorer(shape,kind); n=0
    while e.More(): n+=1; e.Next()
    return n


def prop(s):
    v=GProp_GProps(); BRepGProp.VolumeProperties_s(s,v)
    a=GProp_GProps(); BRepGProp.SurfaceProperties_s(s,a)
    c=v.CentreOfMass()
    b=Bnd_Box(); BRepBndLib.Add_s(s,b,True)
    q=lambda x: round(float(x),12)
    return {
      "volume":q(v.Mass()), "area":q(a.Mass()),
      "center":[q(c.X()),q(c.Y()),q(c.Z())],
      "bbox":[q(b.GetXMin()),q(b.GetYMin()),q(b.GetZMin()),q(b.GetXMax()),q(b.GetYMax()),q(b.GetZMax())],
      "faces":count(s,TopAbs_FACE),"edges":count(s,TopAbs_EDGE),"vertices":count(s,TopAbs_VERTEX),
    }


def solids(path):
    sh=load(path); e=TopExp_Explorer(sh,TopAbs_SOLID); out=[]
    while e.More():
        out.append(prop(TopoDS.Solid(e.Current())))
        e.Next()
    return sorted(out,key=lambda x:(x["center"],x["volume"],x["area"],x["bbox"]))


a,b=sys.argv[1:3]
aa,bb=solids(a),solids(b)
print(json.dumps({"a":aa,"b":bb,"equal":aa==bb},indent=2))
