#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, pathlib, sys

from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Fuse
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Trsf, gp_Vec
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS

def bbox(shape):
    b=Bnd_Box();BRepBndLib.Add_s(shape,b)
    lo=b.CornerMin();hi=b.CornerMax()
    return [lo.X(),lo.Y(),lo.Z(),hi.X(),hi.Y(),hi.Z()]

def volume(shape):
    g=GProp_GProps();BRepGProp.VolumeProperties_s(shape,g);return g.Mass()

def solids(shape):
    out=[];e=TopExp_Explorer(shape,TopAbs_SOLID)
    while e.More():
        s=TopoDS.Solid(e.Current())
        out.append(s);e.Next()
    return out

def largest_solid(path):
    r=STEPControl_Reader()
    if r.ReadFile(str(path)) != 1: raise RuntimeError(f"read failed: {path}")
    r.TransferRoots();shape=r.OneShape()
    ss=solids(shape)
    if not ss:raise RuntimeError("no solids")
    return max(ss,key=volume)

def translated(shape,dx):
    t=gp_Trsf();t.SetTranslation(gp_Vec(dx,0,0))
    return BRepBuilderAPI_Transform(shape,t,True).Shape()

def slab(shape,x0,x1,b):
    # Oversize Y/Z deliberately.
    y0=b[1]-10.;z0=b[2]-10.;dy=(b[4]-b[1])+20.;dz=(b[5]-b[2])+20.
    box=BRepPrimAPI_MakeBox(gp_Pnt(x0,y0,z0),gp_Pnt(x1,y0+dy,z0+dz)).Shape()
    op=BRepAlgoAPI_Common(shape,box);op.Build()
    if not op.IsDone():raise RuntimeError(f"common failed {x0} {x1}")
    return op.Shape()

# gp_Pnt import separated to keep old OCP variants happy.
from OCP.gp import gp_Pnt

def fuse_many(shapes):
    cur=shapes[0]
    for i,s in enumerate(shapes[1:],1):
        op=BRepAlgoAPI_Fuse(cur,s)
        op.Build()
        if not op.IsDone():raise RuntimeError(f"fuse failed at {i}")
        cur=op.Shape()
    return cur

def face_count(shape):
    from OCP.TopAbs import TopAbs_FACE
    e=TopExp_Explorer(shape,TopAbs_FACE);n=0
    while e.More():n+=1;e.Next()
    return n

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--target-sites",type=int,required=True)
    ap.add_argument("--pitch",type=float,default=2.0)
    ap.add_argument("--end-margin",type=float,default=0.25)
    args=ap.parse_args()

    source=largest_solid(pathlib.Path(args.source))
    b=bbox(source);L=b[3]-b[0]
    source_sites=round((L-2*args.end_margin)/args.pitch)
    if source_sites<2 or abs(L-(source_sites*args.pitch+2*args.end_margin))>2e-3:
        raise RuntimeError((L,source_sites))
    # Keep actual asymmetric exported endpoints; only the nominal cell region
    # is inferred from pitch + end margin.
    left_boundary=b[0]+args.end_margin
    right_boundary=left_boundary+source_sites*args.pitch

    left=slab(source,b[0]-1e-6,left_boundary,b)
    right=slab(source,right_boundary,b[3]+1e-6,b)

    # Use a central source cell to avoid end-specific topology.
    canonical_site=source_sites//2-1
    x0=left_boundary+canonical_site*args.pitch
    cell=slab(source,x0,x0+args.pitch,b)
    cell_center=(x0+x0+args.pitch)*0.5

    target_sites=args.target_sites
    target_cell_length=target_sites*args.pitch
    target_left_boundary=-target_cell_length*0.5
    target_right_boundary=target_cell_length*0.5
    # Source is essentially centered but preserve cap widths exactly.
    left_width=left_boundary-b[0]
    right_width=b[3]-right_boundary
    target_min=target_left_boundary-left_width
    target_max=target_right_boundary+right_width

    pieces=[]
    pieces.append(translated(left,target_min-b[0]))
    for i in range(target_sites):
        c=target_left_boundary+(i+0.5)*args.pitch
        pieces.append(translated(cell,c-cell_center))
    pieces.append(translated(right,target_right_boundary-right_boundary))

    out=fuse_many(pieces)
    info={
        "source_bbox":b,
        "source_volume":volume(source),
        "source_faces":face_count(source),
        "source_sites":source_sites,
        "canonical_cell_bbox":bbox(cell),
        "canonical_cell_volume":volume(cell),
        "canonical_cell_faces":face_count(cell),
        "left_cap_bbox":bbox(left),"left_cap_volume":volume(left),"left_cap_faces":face_count(left),
        "right_cap_bbox":bbox(right),"right_cap_volume":volume(right),"right_cap_faces":face_count(right),
        "target_sites":target_sites,
        "target_bbox":bbox(out),
        "target_volume":volume(out),
        "target_faces":face_count(out),
        "valid":BRepCheck_Analyzer(out).IsValid(),
    }
    print(json.dumps(info,indent=2))

    w=STEPControl_Writer()
    w.Transfer(out,STEPControl_AsIs)
    status=w.Write(str(args.output))
    if status != 1:raise RuntimeError(f"write status {status}")

if __name__=="__main__":
    main()
