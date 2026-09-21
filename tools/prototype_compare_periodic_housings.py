#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, hashlib, json, pathlib

import prototype_deboolean_repeated_components as db

def dominant_housing(path: pathlib.Path):
    db._GEOM_PAYLOAD_CACHE.clear()
    lines,raw,typ,refs,pts,inb,maxid=db.parse(path)
    solids=[i for i,t in typ.items() if t=="MANIFOLD_SOLID_BREP"]
    if not solids:
        raise RuntimeError(f"{path}: no MANIFOLD_SOLID_BREP")
    ranked=[]
    for solid in solids:
        cl=db.descendants(solid,refs)
        faces=[x for x in cl if typ.get(x)=="ADVANCED_FACE"]
        ranked.append((len(faces),solid,faces))
    ranked.sort(reverse=True)
    _,solid,faces=ranked[0]
    fes={f:db.face_edges(f,typ,refs) for f in faces}
    style=db.parse_style_map(raw,typ,refs)
    return dict(path=path,raw=raw,typ=typ,refs=refs,pts=pts,faces=faces,
                fes=fes,style=style,solid=solid)

def face_center(face,g):
    vids=db.face_vertex_ids(face,g["fes"],g["typ"],g["refs"],g["pts"])
    xyz=sorted(set(g["pts"][v] for v in vids))
    if not xyz:return None
    lo=[min(p[k] for p in xyz) for k in range(3)]
    hi=[max(p[k] for p in xyz) for k in range(3)]
    return tuple((lo[k]+hi[k])*0.5 for k in range(3))

def local_signature(face,g,tol):
    c=face_center(face,g)
    if c is None:return None
    delta=tuple(-x for x in c)
    key=db.face_translation_key(face,delta,g["raw"],g["typ"],g["refs"],g["pts"],g["fes"],tol)
    digest=hashlib.sha256(repr(key).encode()).hexdigest()
    return digest,c,key

def summarize(g,tol):
    families=collections.defaultdict(list)
    missing=0
    for f in g["faces"]:
        x=local_signature(f,g,tol)
        if x is None:
            missing+=1;continue
        digest,c,key=x
        families[digest].append((f,c,key))
    return families,missing

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol",type=float,default=1e-5)
    ap.add_argument("--expected-added-cells",type=int,default=None)
    args=ap.parse_args()

    ga=dominant_housing(pathlib.Path(args.a))
    gb=dominant_housing(pathlib.Path(args.b))
    fa,ma=summarize(ga,args.tol);fb,mb=summarize(gb,args.tol)
    ca=collections.Counter({k:len(v) for k,v in fa.items()})
    cb=collections.Counter({k:len(v) for k,v in fb.items()})
    keys=set(ca)|set(cb)

    records=[]
    for k in keys:
        na,nb=ca[k],cb[k]
        if na==nb:continue
        va=fa.get(k,[]);vb=fb.get(k,[])
        rec={
            "signature":k[:16],
            "a_count":na,
            "b_count":nb,
            "delta":nb-na,
            "surface_type": (vb or va)[0][2][0],
            "edges": (vb or va)[0][2][2],
            "a_centers":[list(c) for _,c,_ in va],
            "b_centers":[list(c) for _,c,_ in vb],
        }
        if args.expected_added_cells:
            rec["delta_per_added_cell"]=(nb-na)/args.expected_added_cells
        records.append(rec)
    records.sort(key=lambda x:(-abs(x["delta"]),x["signature"]))

    positive=sum(max(0,x["delta"]) for x in records)
    negative=sum(max(0,-x["delta"]) for x in records)
    result={
      "a":str(ga["path"]),"b":str(gb["path"]),
      "a_housing_solid":ga["solid"],"b_housing_solid":gb["solid"],
      "a_faces":len(ga["faces"]),"b_faces":len(gb["faces"]),
      "a_local_signature_families":len(fa),"b_local_signature_families":len(fb),
      "a_missing":ma,"b_missing":mb,
      "positive_face_delta":positive,
      "negative_face_delta":negative,
      "net_face_delta":len(gb["faces"])-len(ga["faces"]),
      "changed_families":len(records),
      "records":records,
    }
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
