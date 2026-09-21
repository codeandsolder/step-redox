#!/usr/bin/env python3
"""Detect regular arrays of existing MAPPED_ITEM placements."""
from __future__ import annotations
import argparse,collections,json,math,pathlib,re,sys
import numpy as np
import prototype_detect_trim_lattices as lattice

def target_frame(g,target):
    t=g.typ.get(target)
    rs=g.refs.get(target,())
    if t=="AXIS2_PLACEMENT_3D":
        p=next((g.points[x] for x in rs if x in g.points),None)
        ds=[g.dirs[x] for x in rs if x in g.dirs]
        if p is None:return None
        z=lattice.norm(ds[0]) if ds else np.array([0.,0.,1.])
        x=lattice.norm(ds[1]) if len(ds)>1 else np.array([1.,0.,0.])
        x=x-z*np.dot(x,z);x=lattice.norm(x);y=lattice.norm(np.cross(z,x))
        return np.asarray(p),np.column_stack([x,y,z])
    if t=="CARTESIAN_TRANSFORMATION_OPERATOR_3D":
        p=next((g.points[x] for x in rs if x in g.points),None)
        ds=[g.dirs[x] for x in rs if x in g.dirs]
        if p is None:return None
        x=lattice.norm(ds[0]) if len(ds)>0 else np.array([1.,0.,0.])
        y=lattice.norm(ds[1]) if len(ds)>1 else np.array([0.,1.,0.])
        z=lattice.norm(ds[2]) if len(ds)>2 else lattice.norm(np.cross(x,y))
        return np.asarray(p),np.column_stack([x,y,z])
    return None

def qmat(M):
    return tuple(int(round(float(x)/1e-10)) for x in M.reshape(-1))

def parent_reps(g):
    inbound=collections.defaultdict(list)
    for p,rs in g.refs.items():
        for c in rs:inbound[c].append(p)
    out={}
    for i,t in g.typ.items():
        if t!="MAPPED_ITEM":continue
        out[i]=tuple(sorted(
            p for p in inbound.get(i,())
            if "SHAPE_REPRESENTATION" in (g.typ.get(p) or "")
        ))
    return out,inbound

def style_map(g):
    s=collections.defaultdict(list)
    for i,t in g.typ.items():
        if t!="STYLED_ITEM":continue
        rs=g.refs.get(i,())
        if rs:
            s[rs[-1]].append(tuple(rs[:-1]))
    return {k:tuple(sorted(v)) for k,v in s.items()}

def fit_positions(P,tol):
    P=np.asarray(P,dtype=float)
    if len(P)<2:return None
    C=P-P.mean(axis=0)
    _,S,Vt=np.linalg.svd(C,full_matrices=False)
    rank=sum(float(x)>tol for x in S)
    if rank<=0:return None
    dim=1 if rank==1 else 2
    basis=Vt[:dim]
    uv=C@basis.T
    if dim==1:
        uv=np.column_stack([uv[:,0],np.zeros(len(P))])
    fit=lattice.fit_lattice(uv,tol)
    if fit is None:return None
    b3=[]
    for b2 in fit["basis"]:
        if dim==1:
            b3.append((basis[0]*b2[0]).tolist())
        else:
            b3.append((basis[0]*b2[0]+basis[1]*b2[1]).tolist())
    fit["basis_3d"]=b3
    fit["svd_rank"]=rank
    return fit

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--tol",type=float,default=1e-5)
    ap.add_argument("--min-items",type=int,default=4)
    args=ap.parse_args()
    path=pathlib.Path(args.input);g=lattice.parse(path)
    parents,inbound=parent_reps(g);styles=style_map(g)
    groups=collections.defaultdict(list)
    rejected=collections.Counter()
    for i,t in g.typ.items():
        if t!="MAPPED_ITEM":continue
        rs=g.refs.get(i,())
        if len(rs)<2:continue
        repmap=rs[0];target=rs[-1]
        fr=target_frame(g,target)
        if fr is None:
            rejected[g.typ.get(target) or "<NONE>"]+=1;continue
        p,M=fr
        ps=parents.get(i,())
        # One parent representation is the common useful case. A style root is
        # not part of the shape hierarchy and is tracked separately.
        if len(ps)!=1:
            rejected[f"parents:{len(ps)}"]+=1;continue
        key=(ps[0],repmap,qmat(M),styles.get(i,()))
        groups[key].append((i,p.tolist(),target))
    hits=[]
    for (parent,repmap,ori,sty),members in groups.items():
        if len(members)<args.min_items:continue
        fit=fit_positions([m[1] for m in members],args.tol)
        if fit is None:continue
        fit.pop("integer_sites",None)
        hits.append({
            "parent_representation":parent,
            "representation_map":repmap,
            "style":sty,
            "items":len(members),
            "lattice":fit,
            "sample_items":[m[0] for m in members[:8]],
        })
    hits.sort(key=lambda x:-x["items"])
    print(json.dumps({
        "file":str(path),"bytes":path.stat().st_size,"entities":len(g.raw),
        "tolerance_mm":args.tol,"mapped_items":sum(t=="MAPPED_ITEM" for t in g.typ.values()),
        "array_families":len(hits),"rejected_target_or_parent":dict(rejected),
        "hits":hits,
    },indent=2))

if __name__=="__main__":main()
