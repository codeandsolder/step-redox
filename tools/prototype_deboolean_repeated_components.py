#!/usr/bin/env python3
"""
Experimental generic de-booleaning / feature-instancing pass for STEP AP214-ish
ECAD models.

Strategy:
  1. Find the dominant BREP geometry root and its shape representation.
  2. Cut face adjacency at presentation-style/material boundaries.
  3. Find connected components repeated by a pure translation.
  4. Replace repeated component copies by one shell source + MAPPED_ITEMs.
  5. Re-express the remaining faces as shell-based surface geometry.
  6. Repair presentation references and mark/sweep entities detached by rewrite.

This intentionally changes "one boolean-fused solid" into visually equivalent
surface models. It is for ECAD/display geometry, not manufacturing BREP export.
"""
from __future__ import annotations
import argparse, collections, math, pathlib, re

ER=re.compile(r"^#(\d+)=(.*);$")
RR=re.compile(r"#(\d+)")
CP=re.compile(r"CARTESIAN_POINT\('[^']*',\(([^)]*)\)\)")

SURF_TYPES={
    "PLANE","CYLINDRICAL_SURFACE","CONICAL_SURFACE","SPHERICAL_SURFACE",
    "TOROIDAL_SURFACE","B_SPLINE_SURFACE_WITH_KNOTS",
    "SURFACE_OF_LINEAR_EXTRUSION","SURFACE_OF_REVOLUTION",
}

# One input file per process. Cache the expensive ID-independent dependency
# payload of surfaces/curves; candidate transforms only change point coords.
_GEOM_PAYLOAD_CACHE={}
GEOM_ROOT_TYPES={"MANIFOLD_SOLID_BREP","BREP_WITH_VOIDS","SHELL_BASED_SURFACE_MODEL"}
ROOT_TYPES={
    "APPLICATION_PROTOCOL_DEFINITION",
    "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION",
    "PRESENTATION_LAYER_ASSIGNMENT",
    "PRODUCT_RELATED_PRODUCT_CATEGORY",
    "SHAPE_DEFINITION_REPRESENTATION",
    "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION",
    "SHAPE_REPRESENTATION_RELATIONSHIP",
}

def parse(path):
    lines=path.read_text(errors="replace").splitlines()
    raw={};typ={};refs={};pts={};inb=collections.defaultdict(list);maxid=0

    # Part 21 allows entity instances to span physical lines. EasyEDA usually
    # emits one entity per line, but several first-party vendor exporters
    # (notably TE) wrap large CLOSED_SHELL / ADVANCED_FACE aggregates. Assemble
    # complete #id=...; statements for analysis while keeping the original
    # physical lines unchanged for rewrite-oriented prototypes.
    statements=[]
    pending=None
    for line in lines:
        s=line.strip()
        if pending is None:
            if not s.startswith("#"):
                continue
            pending=s
        else:
            pending+=s
        if pending.endswith(";"):
            statements.append(pending)
            pending=None

    for statement in statements:
        m=ER.match(statement)
        if not m:continue
        i=int(m.group(1));maxid=max(maxid,i);b=m.group(2)
        raw[i]=b;typ[i]="COMPLEX" if b.startswith("(") else b.split("(",1)[0]
        rs=list(map(int,RR.findall(b)));refs[i]=rs
        for x in rs:inb[x].append(i)
        q=CP.fullmatch(b)
        if q:pts[i]=tuple(map(float,q.group(1).split(",")))
    return lines,raw,typ,refs,pts,inb,maxid

def descendants(start,refs):
    seen=set();stack=list(start if isinstance(start,(list,tuple,set)) else [start])
    while stack:
        x=stack.pop()
        if x in seen:continue
        seen.add(x);stack.extend(refs.get(x,()))
    return seen

def face_edges(f,typ,refs):
    out=set()
    for b in refs.get(f,()):
        if typ.get(b) not in ("FACE_BOUND","FACE_OUTER_BOUND"):continue
        for loop in refs.get(b,()):
            if typ.get(loop)!="EDGE_LOOP":continue
            for oe in refs.get(loop,()):
                if typ.get(oe)!="ORIENTED_EDGE":continue
                out.update(x for x in refs.get(oe,()) if typ.get(x)=="EDGE_CURVE")
    return out

def face_surface(f,typ,refs):
    for x in refs.get(f,()):
        t=typ.get(x) or ""
        if t in SURF_TYPES or "B_SPLINE_SURFACE" in t:
            return x
    return None

def face_vertex_ids(f,face_edges_map,typ,refs,pts):
    out=set()
    for e in face_edges_map[f]:
        for v in refs.get(e,())[:2]:
            if typ.get(v)=="VERTEX_POINT":
                out.update(x for x in refs.get(v,()) if x in pts)
    return out

def componentize(faces,edge_faces):
    faces=list(faces);S=set(faces);par={f:f for f in faces}
    def F(x):
        while par[x]!=x:
            par[x]=par[par[x]];x=par[x]
        return x
    def U(a,b):
        a,b=F(a),F(b)
        if a!=b:par[b]=a
    for allf in edge_faces.values():
        same=[f for f in allf if f in S]
        for f in same[1:]:U(same[0],f)
    C=collections.defaultdict(list)
    for f in faces:C[F(f)].append(f)
    return list(C.values())

def parse_style_map(raw,typ,refs):
    style={}
    for i,t in typ.items():
        if t!="STYLED_ITEM":continue
        rs=refs.get(i,())
        if rs and typ.get(rs[-1])=="ADVANCED_FACE":
            style[rs[-1]]=tuple(rs[:-1])
    return style

def component_info(fs,face_edges_map,typ,refs,pts):
    verts=set();sh=collections.Counter();edgeuse=collections.Counter()
    for f in fs:
        sid=face_surface(f,typ,refs);sh[typ.get(sid) or "<NONE>"]+=1
        for e in face_edges_map[f]:
            edgeuse[e]+=1
        verts.update(face_vertex_ids(f,face_edges_map,typ,refs,pts))
    # Topological vertex entities are often duplicated at identical coordinates
    # by exporters/boolean operations. Congruence is geometric, so collapse
    # coincident coordinates before comparing components.
    xyz=sorted(set(pts[v] for v in verts))
    lo=tuple(min(p[k] for p in xyz) for k in range(3))
    hi=tuple(max(p[k] for p in xyz) for k in range(3))
    center=tuple((lo[k]+hi[k])/2 for k in range(3))
    size=tuple(hi[k]-lo[k] for k in range(3))
    return {
        "faces":tuple(sorted(fs)),"verts":tuple(sorted(verts)),"xyz":xyz,
        "surface_hist":tuple(sorted(sh.items())),
        "boundary_edges":sum(n==1 for n in edgeuse.values()),
        "closed":bool(edgeuse) and all(n==2 for n in edgeuse.values()),
        "center":center,"size":size,
    }

def normalized_body(body):
    """Entity syntax invariant to referenced entity numbering."""
    return RR.sub("#", body)


def descendant_geometry_signature(entity, delta, raw, typ, refs, pts, tol):
    """Translation-normalized geometric payload reachable from an entity."""
    payload=_GEOM_PAYLOAD_CACHE.get(entity)
    if payload is None:
        seen=set()
        stack=[entity]
        bodies=[]
        raw_coords=set()
        while stack:
            x=stack.pop()
            if x in seen:
                continue
            seen.add(x)
            if x in pts:
                raw_coords.add(pts[x])
                continue
            body=raw.get(x)
            if body is None:
                continue
            bodies.append((typ.get(x) or "<NONE>", normalized_body(body)))
            stack.extend(refs.get(x,()))
        payload=(tuple(sorted(bodies)),tuple(sorted(raw_coords)))
        _GEOM_PAYLOAD_CACHE[entity]=payload
    bodies,raw_coords=payload
    coords=tuple(sorted(
        tuple(round((p[k]+delta[k])/tol) for k in range(3))
        for p in raw_coords
    ))
    return bodies,coords


def face_translation_key(f, delta, raw, typ, refs, pts, face_edges_map, tol):
    sid=face_surface(f,typ,refs)
    bounds=[x for x in refs.get(f,()) if typ.get(x) in ("FACE_BOUND","FACE_OUTER_BOUND")]
    edge_geometries=[]
    boundary_coords=set()
    for e in face_edges_map[f]:
        # EDGE_CURVE = (name, vertex1, vertex2, curve, same_sense)
        curve=None
        for x in refs.get(e,()):
            if typ.get(x) not in ("VERTEX_POINT",None):
                curve=x
        if curve is not None:
            edge_geometries.append(
                descendant_geometry_signature(curve,delta,raw,typ,refs,pts,tol)
            )
        for v in refs.get(e,())[:2]:
            if typ.get(v)=="VERTEX_POINT":
                for pp in refs.get(v,()):
                    if pp in pts:
                        p=pts[pp]
                        boundary_coords.add(
                            tuple(round((p[k]+delta[k])/tol) for k in range(3))
                        )
    support=(
        descendant_geometry_signature(sid,delta,raw,typ,refs,pts,tol)
        if sid is not None else None
    )
    # Preserve face orientation semantics in addition to its locus.
    sense=".T." if raw.get(f,"").rstrip().endswith(".T.)") else (
        ".F." if raw.get(f,"").rstrip().endswith(".F.)") else "?"
    )
    return (
        typ.get(sid) or "<NONE>",
        len(bounds),
        len(face_edges_map[f]),
        sense,
        support,
        tuple(sorted(edge_geometries)),
        tuple(sorted(boundary_coords)),
    )


def component_translation_key(info, delta, raw, typ, refs, pts, face_edges_map, tol):
    return collections.Counter(
        face_translation_key(f,delta,raw,typ,refs,pts,face_edges_map,tol)
        for f in info["faces"]
    )


def translation_match(a,b,raw,typ,refs,pts,face_edges_map,tol):
    d=tuple(b["center"][k]-a["center"][k] for k in range(3))
    return (
        component_translation_key(a,d,raw,typ,refs,pts,face_edges_map,tol)
        == component_translation_key(b,(0.0,0.0,0.0),raw,typ,refs,pts,face_edges_map,tol)
    )

def parse_rep_body(body):
    # REP_TYPE('name',(#items),#context)
    p=body.find("(")
    if p<0 or p+1>=len(body) or body[p+1]!="'":raise ValueError(body[:100])
    i=p+2;name=[]
    while i<len(body):
        if body[i]=="'":
            if i+1<len(body) and body[i+1]=="'":
                name.append("''");i+=2;continue
            i+=1;break
        name.append(body[i]);i+=1
    while i<len(body) and body[i] in " ,":i+=1
    if i>=len(body) or body[i]!="(":raise ValueError("missing item aggregate")
    start=i;depth=0
    while i<len(body):
        if body[i]=="(":depth+=1
        elif body[i]==")":
            depth-=1
            if depth==0:
                end=i;i+=1;break
        i+=1
    items=list(map(int,RR.findall(body[start:end+1])))
    tail=body[i:]
    m=re.search(r"#(\d+)\)\s*$",tail)
    if not m:raise ValueError("missing representation context")
    return "".join(name),items,int(m.group(1))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("src");ap.add_argument("dst")
    ap.add_argument("--tol",type=float,default=1e-5)
    ap.add_argument("--min-instances",type=int,default=4)
    ap.add_argument("--min-saved-faces",type=int,default=100)
    args=ap.parse_args()
    src=pathlib.Path(args.src);dst=pathlib.Path(args.dst)
    lines,raw,typ,refs,pts,inb,maxid=parse(src)

    # Pick geometry root / representation that owns the most ADVANCED_FACEs.
    geom_candidates=[]
    for g,t in typ.items():
        if t not in GEOM_ROOT_TYPES:continue
        ds=descendants(g,refs)
        fs={x for x in ds if typ.get(x)=="ADVANCED_FACE"}
        reps=[x for x in inb.get(g,()) if "SHAPE_REPRESENTATION" in (typ.get(x) or "")]
        if fs and reps:
            geom_candidates.append((len(fs),g,reps[0],fs,ds))
    if not geom_candidates:raise RuntimeError("no represented BREP geometry root")
    geom_candidates.sort(reverse=True)
    face_count,geom_root,main_rep,model_faces,old_geom_desc=geom_candidates[0]
    rep_name,rep_items,context=parse_rep_body(raw[main_rep])
    print("geometry root",geom_root,typ[geom_root],"main rep",main_rep,typ[main_rep],
          "faces",face_count,"name",rep_name)

    face_edges_map={f:face_edges(f,typ,refs) for f in model_faces}
    edge_faces=collections.defaultdict(list)
    for f,es in face_edges_map.items():
        for e in es:edge_faces[e].append(f)
    style=parse_style_map(raw,typ,refs)

    # Split by style first, then topology; group components by coarse invariant.
    style_faces=collections.defaultdict(list)
    for f in model_faces:style_faces[style.get(f,())].append(f)
    coarse=collections.defaultdict(list)
    for st,fs in style_faces.items():
        for comp in componentize(fs,edge_faces):
            info=component_info(comp,face_edges_map,typ,refs,pts)
            key=(
                st,len(info["faces"]),info["surface_hist"],
                info["boundary_edges"],
                tuple(round(x/args.tol) for x in info["size"]),
            )
            coarse[key].append(info)

    # Greedy exact-ish translation clustering within the coarse buckets.
    families=[]
    for key,members in coarse.items():
        if len(members)<args.min_instances:continue
        clusters=[]
        for m in members:
            for c in clusters:
                if translation_match(c[0],m,raw,typ,refs,pts,face_edges_map,args.tol):
                    c.append(m);break
            else:
                clusters.append([m])
        for c in clusters:
            if len(c)<args.min_instances:continue
            saved=(len(c)-1)*len(c[0]["faces"])
            if saved<args.min_saved_faces:continue
            families.append({
                "style":key[0],"members":c,"faces_each":len(c[0]["faces"]),
                "saved_faces":saved,
                "proof_tolerance":args.tol,
            })
    families.sort(key=lambda x:-x["saved_faces"])
    if not families:raise RuntimeError("no repeated translation component families")
    for n,fam in enumerate(families[:30]):
        c=fam["members"][0]
        print("family",n,"instances",len(fam["members"]),"faces",fam["faces_each"],
              "saved",fam["saved_faces"],"style",fam["style"],
              "face-proof-tol",fam["proof_tolerance"],"size",tuple(round(x,6) for x in c["size"]))

    repeated_faces={f for fam in families for m in fam["members"] for f in m["faces"]}
    retained_faces=set(model_faces)-repeated_faces
    retained_comps=componentize(retained_faces,edge_faces) if retained_faces else []

    # Original geometry still needed by retained faces and canonical source faces.
    source_faces=set(retained_faces)
    for fam in families:source_faces.update(fam["members"][0]["faces"])
    needed_original=descendants(source_faces,refs)

    nextid=maxid+1;added=[]
    def push(body):
        nonlocal nextid
        i=nextid;nextid+=1;added.append((i,body));return i
    def reflist(ids):return "(" + ",".join(f"#{i}" for i in ids) + ")"

    p0=push("CARTESIAN_POINT('',(0.,0.,0.))")
    dz=push("DIRECTION('',(0.,0.,1.))")
    dx=push("DIRECTION('',(1.,0.,0.))")
    origin=push(f"AXIS2_PLACEMENT_3D('',#{p0},#{dz},#{dx})")

    retained_shells=[]
    for fs in sorted(retained_comps,key=lambda x:(-len(x),min(x))):
        info=component_info(fs,face_edges_map,typ,refs,pts)
        kind="CLOSED_SHELL" if info["closed"] else "OPEN_SHELL"
        retained_shells.append(push(f"{kind}('step-redox residual',{reflist(sorted(fs))})"))
    residual_model=None
    if retained_shells:
        residual_model=push(
            f"SHELL_BASED_SURFACE_MODEL('step-redox residual',{reflist(retained_shells)})"
        )

    mapped=[];mapped_styles=[];family_stats=[]
    for fi,fam in enumerate(families):
        canonical=fam["members"][0]
        kind="CLOSED_SHELL" if canonical["closed"] else "OPEN_SHELL"
        sh=push(f"{kind}('step-redox feature source',{reflist(canonical['faces'])})")
        sm=push(f"SHELL_BASED_SURFACE_MODEL('step-redox feature source',(#{sh}))")
        sr=push(f"SHAPE_REPRESENTATION('step-redox feature source',(#{sm},#{origin}),#{context})")
        rm=push(f"REPRESENTATION_MAP(#{origin},#{sr})")
        for member in fam["members"]:
            delta=tuple(member["center"][k]-canonical["center"][k] for k in range(3))
            if max(abs(x) for x in delta)<1e-15:
                ax=origin
            else:
                pp=push(f"CARTESIAN_POINT('',({delta[0]:.17g},{delta[1]:.17g},{delta[2]:.17g}))")
                ax=push(f"AXIS2_PLACEMENT_3D('',#{pp},#{dz},#{dx})")
            mi=push(f"MAPPED_ITEM('',#{rm},#{ax})");mapped.append(mi)
            if fam["style"]:
                si=push(f"STYLED_ITEM('',{reflist(fam['style'])},#{mi})")
                mapped_styles.append(si)
        family_stats.append((len(fam["members"]),fam["faces_each"],fam["proof_tolerance"]))

    # Replace direct geometry item(s) in the owning representation.
    keep_items=[x for x in rep_items if typ.get(x) not in GEOM_ROOT_TYPES]
    new_items=([residual_model] if residual_model else [])+mapped+keep_items
    qname=rep_name.replace("'","''")
    new_rep_body=f"{typ[main_rep]}('{qname}',{reflist(new_items)},#{context})"

    # Remove presentation references that solely kept obsolete fused geometry
    # and repeated copied faces alive. Mapped-item styles replace repeated face styles.
    stale_style_items=set()
    for i,t in typ.items():
        if t!="STYLED_ITEM":continue
        rs=refs.get(i,())
        if not rs:continue
        target=rs[-1]
        if target==geom_root or (target in old_geom_desc and target not in needed_original):
            stale_style_items.add(i)
    drop_refs=stale_style_items | (old_geom_desc-needed_original)

    def rewrite_first_aggregate(body,drop,append=()):
        l=body.find("(#")
        if l<0:return body
        i=l;depth=0
        while i<len(body):
            if body[i]=="(":depth+=1
            elif body[i]==")":
                depth-=1
                if depth==0:
                    end=i;break
            i+=1
        ids=list(map(int,RR.findall(body[l:end+1])))
        ids=[x for x in ids if x not in drop]
        ids.extend(x for x in append if x not in ids)
        return body[:l]+reflist(ids)+body[end+1:]

    rewritten={}
    for i,t in typ.items():
        if t=="MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION":
            rewritten[i]=rewrite_first_aggregate(raw[i],drop_refs,mapped_styles)
        elif t=="PRESENTATION_LAYER_ASSIGNMENT":
            rewritten[i]=rewrite_first_aggregate(raw[i],drop_refs)

    out=[];in_data=False;inserted=False
    for line in lines:
        m=ER.match(line.strip())
        if m:
            i=int(m.group(1))
            if i==main_rep:
                out.append(f"#{i}={new_rep_body};");continue
            if i in rewritten:
                out.append(f"#{i}={rewritten[i]};");continue
        if line.strip()=="DATA;":in_data=True
        if in_data and line.strip()=="ENDSEC;" and not inserted:
            out.extend(f"#{i}={b};" for i,b in added);inserted=True;in_data=False
        out.append(line)
    if not inserted:raise RuntimeError("DATA section not found")

    # Generic conservative graph GC from document/product/presentation roots.
    traw={};trefs={};ttyp={}
    for line in out:
        m=ER.match(line.strip())
        if not m:continue
        i=int(m.group(1));b=m.group(2)
        traw[i]=b;trefs[i]=list(map(int,RR.findall(b)))
        ttyp[i]="COMPLEX" if b.startswith("(") else b.split("(",1)[0]
    roots=[i for i,t in ttyp.items() if t in ROOT_TYPES]
    live=set();stack=list(roots)
    while stack:
        x=stack.pop()
        if x in live or x not in traw:continue
        live.add(x);stack.extend(trefs.get(x,()))
    gc=[];removed=0
    for line in out:
        m=ER.match(line.strip())
        if m and int(m.group(1)) not in live:
            removed+=1;continue
        gc.append(line)
    dst.write_text("\n".join(gc)+"\n")
    print("families",len(families),"instances",sum(len(x["members"]) for x in families),
          "repeated faces",len(repeated_faces),"retained faces",len(retained_faces))
    print("stale styles",len(stale_style_items),"mapped styles",len(mapped_styles))
    print("gc roots",len(roots),"removed",removed,"live",len(live))
    print("bytes",src.stat().st_size,"->",dst.stat().st_size,
          "ratio",dst.stat().st_size/src.stat().st_size)

if __name__=="__main__":main()
