import sys, math
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common


def load(p):
    r=STEPControl_Reader(); assert r.ReadFile(p)==IFSelect_RetDone
    r.TransferRoots(); return r.OneShape()


def mass(s):
    g=GProp_GProps(); BRepGProp.VolumeProperties_s(s,g); return g.Mass()


def center(s):
    g=GProp_GProps(); BRepGProp.VolumeProperties_s(s,g); c=g.CentreOfMass()
    return (c.X(),c.Y(),c.Z())


def solids(s):
    e=TopExp_Explorer(s,TopAbs_SOLID); out=[]
    while e.More():
        x=TopoDS.Solid(e.Current()); out.append(x); e.Next()
    return sorted(out,key=lambda s:tuple(round(x,9) for x in center(s)))


aa=solids(load(sys.argv[1])); bb=solids(load(sys.argv[2]))
assert len(aa)==len(bb)
worst=0.0
for i,(a,b) in enumerate(zip(aa,bb)):
    va=mass(a); vb=mass(b)
    op=BRepAlgoAPI_Common(a,b); op.Build()
    if not op.IsDone():
        raise RuntimeError(f"common failed for solid {i}")
    vc=mass(op.Shape())
    miss=max(abs(va-vc),abs(vb-vc))
    worst=max(worst,miss)
    print(i,repr(va),repr(vb),repr(vc),"missing",repr(miss))
print("count",len(aa),"worst_missing_volume",repr(worst))
