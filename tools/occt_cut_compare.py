import sys
from OCP.STEPControl import STEPControl_Reader
from OCP.IFSelect import IFSelect_RetDone
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

def load(p):
 r=STEPControl_Reader(); assert r.ReadFile(p)==IFSelect_RetDone; r.TransferRoots(); return r.OneShape()
def prop(s):
 g=GProp_GProps(); BRepGProp.VolumeProperties_s(s,g); c=g.CentreOfMass(); return g.Mass(),(c.X(),c.Y(),c.Z())
def solids(shape):
 e=TopExp_Explorer(shape,TopAbs_SOLID); out=[]
 while e.More():
  s=TopoDS.Solid(e.Current()); v,c=prop(s); out.append((c,v,s)); e.Next()
 return sorted(out,key=lambda x:tuple(round(v,9) for v in x[0]))
def cutvol(a,b):
 op=BRepAlgoAPI_Cut(a,b); op.Build()
 return prop(op.Shape())[0] if op.IsDone() else float('nan')

A=solids(load(sys.argv[1])); B=solids(load(sys.argv[2]))
for i,(a,b) in enumerate(zip(A,B)):
 ca,va,sa=a; cb,vb,sb=b
 if abs(va-vb)>1e-7: continue
 ab=cutvol(sa,sb); ba=cutvol(sb,sa)
 if max(abs(ab),abs(ba))>1e-12:
  print(i,ca,cb,"vol",va,vb,"A-B",ab,"B-A",ba)
