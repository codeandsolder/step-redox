#!/usr/bin/env python3
"""
Detect repeated trimming-loop lattices on planar STEP faces.

No package/connector knowledge is used.  The detector:
  * finds planar ADVANCED_FACEs with many inner bounds,
  * clusters inner loops by translation-invariant topology/geometry,
  * infers a primitive 1-D/2-D lattice in the plane's own coordinates,
  * records occupancy, row-pattern/run structure, and whether the outer
    boundary is line-only (eligible for exact cell clipping).

Geometric equivalence is quantized at 1e-5 mm by default.  Parameterization
metadata stays part of the loop signature; this detector does not silently
turn an arbitrary trim into a different curve.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re
from dataclasses import dataclass

import numpy as np

ER = re.compile(r"^#(\d+)=(.*);$")
RR = re.compile(r"#(\d+)")
CP = re.compile(r"CARTESIAN_POINT\('[^']*',\(([^)]*)\)\)")
DIR = re.compile(r"DIRECTION\('[^']*',\(([^)]*)\)\)")
NUM = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", re.I)

def compact_step_syntax(text: str) -> str:
    parts = text.split("'")
    for i in range(0, len(parts), 2):
        parts[i] = "".join(parts[i].split())
    return "'".join(parts)

def unquoted_semicolon(text: str) -> int | None:
    start = 0
    while True:
        idx = text.find(";", start)
        if idx < 0:
            return None
        if text.count("'", 0, idx) % 2 == 0:
            return idx
        start = idx + 1

def iter_step_entities(path: pathlib.Path):
    pending = ""
    start_re = re.compile(r"#\s*\d+\s*=")
    with path.open(errors="replace") as stream:
        for line in stream:
            pending += line
            while True:
                m = start_re.search(pending)
                if m is None:
                    pending = pending[-64:]
                    break
                if m.start():
                    pending = pending[m.start():]
                end = unquoted_semicolon(pending)
                if end is None:
                    break
                rec = compact_step_syntax(pending[:end+1])
                pending = pending[end+1:]
                q = ER.match(rec)
                if q:
                    yield int(q.group(1)), q.group(2)

def etype(body: str) -> str:
    return "COMPLEX" if body.startswith("(") else body.split("(", 1)[0]

def norm(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n == 0:
        raise ValueError("zero vector")
    return v / n

@dataclass
class Graph:
    raw: dict[int, str]
    typ: dict[int, str]
    refs: dict[int, list[int]]
    points: dict[int, tuple[float,float,float]]
    dirs: dict[int, tuple[float,float,float]]

def parse(path: pathlib.Path) -> Graph:
    raw = {}; typ = {}; refs = {}; points = {}; dirs = {}
    for i,b in iter_step_entities(path):
        raw[i]=b; typ[i]=etype(b); refs[i]=list(map(int,RR.findall(b)))
        m=CP.fullmatch(b)
        if m:
            vals=tuple(map(float,m.group(1).split(",")))
            if len(vals)==3: points[i]=vals
        m=DIR.fullmatch(b)
        if m:
            vals=tuple(map(float,m.group(1).split(",")))
            if len(vals)==3: dirs[i]=vals
    return Graph(raw,typ,refs,points,dirs)

def descendants(g: Graph, roots):
    seen=set(); stack=list(roots if isinstance(roots,(list,tuple,set)) else [roots])
    while stack:
        x=stack.pop()
        if x in seen or x not in g.raw: continue
        seen.add(x); stack.extend(g.refs.get(x,()))
    return seen

def bound_loop(g: Graph, bound: int) -> int | None:
    return next((x for x in g.refs.get(bound,()) if g.typ.get(x)=="EDGE_LOOP"),None)

def loop_edges(g: Graph, bound: int):
    loop=bound_loop(g,bound)
    if loop is None:return []
    out=[]
    for oe in g.refs.get(loop,()):
        if g.typ.get(oe)!="ORIENTED_EDGE":continue
        e=next((x for x in g.refs.get(oe,()) if g.typ.get(x)=="EDGE_CURVE"),None)
        if e is not None:out.append((oe,e))
    return out

def edge_vertices(g: Graph, edge: int):
    out=[]
    for x in g.refs.get(edge,())[:3]:
        if g.typ.get(x)!="VERTEX_POINT":continue
        p=next((y for y in g.refs.get(x,()) if y in g.points),None)
        if p is not None:out.append(g.points[p])
    return out

def edge_curve(g: Graph, edge: int):
    # EDGE_CURVE(name,start,end,curve,same_sense)
    rs=g.refs.get(edge,())
    for x in rs:
        if g.typ.get(x) not in (None,"VERTEX_POINT"):
            return x
    return None

def normalized_body(body: str) -> str:
    return RR.sub("#",body)

def qtuple(p, tol):
    return tuple(int(round(float(v)/tol)) for v in p)

def loop_anchor(g: Graph, bound: int):
    coords=[]
    for oe,e in loop_edges(g,bound):
        coords.extend(edge_vertices(g,e))
        c=edge_curve(g,e)
        if c is not None:
            ds=descendants(g,c)
            coords.extend(g.points[x] for x in ds if x in g.points)
    if not coords:return None
    a=np.asarray(coords,dtype=float)
    lo=a.min(axis=0);hi=a.max(axis=0)
    return tuple(((lo+hi)/2).tolist())

def circle_locus(g: Graph, circle: int, anchor, tol):
    if g.typ.get(circle)!="CIRCLE":return None
    rs=g.refs.get(circle,())
    axis=next((x for x in rs if g.typ.get(x)=="AXIS2_PLACEMENT_3D"),None)
    if axis is None:return None
    ar=g.refs.get(axis,())
    center=next((g.points[x] for x in ar if x in g.points),None)
    normal=next((g.dirs[x] for x in ar if x in g.dirs),None)
    if center is None or normal is None:return None
    n=norm(normal)
    # Circle radius is the last numeric literal in its record.  Deliberately
    # ignore AXIS2_PLACEMENT ref_direction: it only chooses parameter zero and
    # does not change the circle locus.
    nums=[float(x) for x in NUM.findall(g.raw[circle])]
    if not nums:return None
    radius=nums[-1]
    rel=qtuple((center[k]-anchor[k] for k in range(3)),tol)
    # Preserve oriented plane normal; reversing it changes curve orientation.
    nq=tuple(int(round(float(x)/1e-10)) for x in n)
    return ("CIRCLE",rel,nq,int(round(radius/tol)))

def line_polygon_locus(g: Graph, bound: int, anchor, tol):
    """Canonical closed polygon locus, independent of line segmentation."""
    loop=bound_loop(g,bound)
    if loop is None:return None
    seq=[]
    for oe in g.refs.get(loop,()):
        if g.typ.get(oe)!="ORIENTED_EDGE":return None
        edge=next((x for x in g.refs.get(oe,()) if g.typ.get(x)=="EDGE_CURVE"),None)
        if edge is None:return None
        curve=edge_curve(g,edge)
        if not curve_is_straight_locus(g,curve,edge,tol):return None
        verts=[x for x in g.refs.get(edge,()) if g.typ.get(x)=="VERTEX_POINT"][:2]
        if len(verts)!=2:return None
        ps=[]
        for v in verts:
            p=next((g.points[x] for x in g.refs.get(v,()) if x in g.points),None)
            if p is None:return None
            ps.append(np.asarray(p,dtype=float))
        # ORIENTED_EDGE's final boolean selects traversal of EDGE_CURVE.
        forward=g.raw[oe].rstrip().endswith(".T.)")
        seq.append(ps[0] if forward else ps[1])
    if len(seq)<3:return None

    # Remove duplicate and collinear vertices.  Cross product is normalized by
    # adjacent segment lengths so the criterion is scale-independent.
    changed=True
    while changed and len(seq)>=3:
        changed=False;out=[]
        n=len(seq)
        for i,p in enumerate(seq):
            a=seq[i-1];b=seq[(i+1)%n]
            u=p-a;v=b-p
            lu=float(np.linalg.norm(u));lv=float(np.linalg.norm(v))
            if lu<=tol or lv<=tol:
                changed=True
                continue
            if float(np.linalg.norm(np.cross(u,v)))/(lu*lv) <= 1e-10:
                # Collinear. Keep a reversal/cusp, drop only same-direction
                # subdivisions of one straight edge.
                if float(np.dot(u,v))>0:
                    changed=True
                    continue
            out.append(p)
        if len(out)==len(seq):break
        seq=out
    if len(seq)<3:return None

    rel=[qtuple((p[k]-anchor[k] for k in range(3)),tol) for p in seq]
    # Starting vertex is arbitrary.  Boundary locus is also independent of
    # traversal direction; FACE_BOUND/face sense retain orientation semantics.
    rots=[]
    for s in (rel,list(reversed(rel))):
        rots.extend(tuple(s[k:]+s[:k]) for k in range(len(s)))
    return ("LINE_POLYGON_LOCUS",min(rots))

def loop_signature(g: Graph, bound: int, anchor, tol):
    edges=loop_edges(g,bound)
    if not edges:return None

    # First collapse exporter segmentation of a complete circular loop.
    # Two half-arcs, four quarter-arcs, etc. are the same closed boundary
    # when every EDGE_CURVE lies on the same circle locus.
    circles=[]
    for oe,e in edges:
        c=edge_curve(g,e)
        loc=circle_locus(g,c,anchor,tol) if c is not None else None
        if loc is None:
            circles=[]
            break
        circles.append(loc)
    if circles and len(set(circles))==1:
        return ("CLOSED_CURVE_LOCUS",circles[0])

    polygon=line_polygon_locus(g,bound,anchor,tol)
    if polygon is not None:
        return polygon

    # Generic fallback: ordered oriented-edge structure + ID-independent
    # dependency payload.  This remains conservative for arbitrary splines.
    uses=[]
    all_rel=[]
    for oe,e in edges:
        curve=edge_curve(g,e)
        if curve is None:return None
        d=descendants(g,curve)
        bodies=[]
        for x in d:
            if x in g.points:
                p=g.points[x]
                all_rel.append(qtuple((p[k]-anchor[k] for k in range(3)),tol))
            else:
                body=g.raw.get(x)
                if body is not None:
                    bodies.append((g.typ.get(x),normalized_body(body)))
        vv=[]
        for p in edge_vertices(g,e):
            vv.append(qtuple((p[k]-anchor[k] for k in range(3)),tol))
        uses.append((
            normalized_body(g.raw[oe]),
            normalized_body(g.raw[e]),
            tuple(vv),
            tuple(sorted(bodies)),
        ))
    rots=[tuple(uses[k:]+uses[:k]) for k in range(len(uses))]
    cycle=min(rots,key=repr)
    return ("EXPORTED_LOOP",cycle,tuple(sorted(set(all_rel))))

def plane_frame(g: Graph, plane: int):
    if g.typ.get(plane)!="PLANE":return None
    axis=next((x for x in g.refs.get(plane,()) if g.typ.get(x)=="AXIS2_PLACEMENT_3D"),None)
    if axis is None:return None
    rs=g.refs.get(axis,())
    loc=next((g.points[x] for x in rs if x in g.points),None)
    ds=[g.dirs[x] for x in rs if x in g.dirs]
    if loc is None or not ds:return None
    z=norm(ds[0]); x=norm(ds[1] if len(ds)>1 else (1,0,0))
    x=x-z*np.dot(x,z);x=norm(x)
    y=norm(np.cross(z,x))
    return np.asarray(loc),x,y,z

def uv_of(p, frame):
    o,x,y,z=frame;d=np.asarray(p)-o
    return np.asarray([np.dot(d,x),np.dot(d,y)])

def curve_is_straight_locus(g: Graph, curve: int, edge: int | None, tol: float):
    """True when the exported curve's locus is one straight segment.

    B-spline exporters frequently spell a perfectly straight edge as a cubic
    with four collinear poles.  We recognize the locus but keep the original
    curve entity when rewriting topology.
    """
    t=g.typ.get(curve)
    if t=="LINE":
        return True
    if t!="B_SPLINE_CURVE_WITH_KNOTS" or edge is None:
        return False
    verts=edge_vertices(g,edge)
    if len(verts)!=2:
        return False
    a=np.asarray(verts[0],dtype=float);b=np.asarray(verts[1],dtype=float)
    d=b-a;L=float(np.linalg.norm(d))
    if L<=tol:
        return False
    u=d/L
    poles=[g.points[x] for x in g.refs.get(curve,()) if x in g.points]
    if len(poles)<2:
        return False
    last=-float("inf")
    for p in poles:
        v=np.asarray(p,dtype=float)-a
        s=float(np.dot(v,u))
        perp=float(np.linalg.norm(v-s*u))
        if perp>tol or s < -tol or s > L+tol:
            return False
        # Require monotonic traversal; a collinear spline that doubles back is
        # not equivalent to a simple line boundary.
        if s+tol < last:
            return False
        last=s
    return True

def line_only_outer(g: Graph, outer: int, tol: float=1e-5):
    curves=[]
    pts=[]
    for oe,e in loop_edges(g,outer):
        c=edge_curve(g,e)
        if c is None:return False,[]
        curves.append(curve_is_straight_locus(g,c,e,tol))
        vv=edge_vertices(g,e)
        if vv:
            pts.extend(vv)
    return all(curves),pts

def canonical_vec(v,tol):
    q=np.rint(np.asarray(v)/tol).astype(np.int64)
    if np.all(q==0):return None
    for x in q:
        if x:
            if x<0:q=-q
            break
    return tuple(int(x) for x in q)

def local_difference_candidates(P,tol,limit=24):
    n=len(P)
    if n<2:return []
    # Sample enough anchors to see the primitive neighbor directions without
    # O(n^2) memory.  For each sample keep only its nearest 12 neighbors.
    idx=np.linspace(0,n-1,min(n,384),dtype=int)
    counter=collections.Counter()
    sums=collections.defaultdict(lambda:np.zeros(2,dtype=float))
    for i in idx:
        d=P-P[i]
        dist2=np.einsum("ij,ij->i",d,d)
        k=min(13,n)
        near=np.argpartition(dist2,k-1)[:k]
        for j in near:
            if j==i:continue
            actual=np.asarray(d[j],dtype=float)
            cv=canonical_vec(actual,tol)
            if cv is None:continue
            # canonical_vec may flip sign; accumulate the actual vector with
            # the same canonical sign rather than reconstructing q*tol.
            q=np.asarray(cv,dtype=float)
            if np.dot(actual,q)<0:actual=-actual
            counter[cv]+=1
            sums[cv]+=actual
    cands=[]
    for q,c in counter.items():
        # Tolerance clusters neighbors; it must not quantize the fitted pitch.
        # Otherwise a few-micrometre basis rounding error accumulates across a
        # long array and falsely rejects an exact lattice.
        v=sums[q]/c
        cands.append((float(np.linalg.norm(v)), -c, v, c))
    cands.sort(key=lambda x:(x[0],x[1]))
    # unique directions/lengths already quantized
    return cands[:limit]

def fit_lattice(P,tol):
    P=np.asarray(P,dtype=float)
    if len(P)<2:return None
    cands=local_difference_candidates(P,tol)
    if not cands:return None
    origin=P[0]
    best=None

    # 2-D candidates first.
    for ia,(_,_,a,ca) in enumerate(cands):
        for _,_,b,cb in cands[ia+1:]:
            det=a[0]*b[1]-a[1]*b[0]
            if abs(det) < tol*tol*10:continue
            B=np.column_stack([a,b])
            C=np.linalg.solve(B,P.T).T
            C-=np.linalg.solve(B,origin)
            I=np.rint(C)
            R=(C-I)@B.T
            err=np.linalg.norm(R,axis=1)
            aligned=err<=tol*1.25
            score=(int(aligned.sum()), -(np.linalg.norm(a)+np.linalg.norm(b)))
            if best is None or score>best[0]:
                best=(score,a,b,I.astype(int),err)
    if best is not None and best[0][0]>=max(4,int(math.ceil(.98*len(P)))):
        _,a,b,I,err=best
        # Canonicalize integer origin.
        I=I-I.min(axis=0)
        occ={tuple(map(int,x)) for x,e in zip(I,err) if e<=tol*1.25}
        nx=max(i for i,j in occ)+1;ny=max(j for i,j in occ)+1
        rows=collections.defaultdict(set)
        for i,j in occ:rows[j].add(i)
        row_patterns=collections.defaultdict(list)
        run_lengths=collections.Counter()
        for j in range(ny):
            s=tuple(i in rows[j] for i in range(nx))
            row_patterns[s].append(j)
            start=None
            for i,hit in enumerate(s+(False,)):
                if hit and start is None:start=i
                elif not hit and start is not None:
                    run_lengths[i-start]+=1;start=None
        return {
            "dimension":2,
            "basis":[a.tolist(),b.tolist()],
            "pitch":[float(np.linalg.norm(a)),float(np.linalg.norm(b))],
            "aligned":int(sum(err<=tol*1.25)),
            "max_residual":float(err.max()),
            "occupancy_count":len(occ),
            "grid_shape":[nx,ny],
            "fill_ratio":len(occ)/(nx*ny),
            "unique_row_patterns":len(row_patterns),
            "row_pattern_multiplicities":sorted((len(v),sum(k)) for k,v in row_patterns.items()),
            "run_lengths":dict(sorted(run_lengths.items())),
            "integer_sites":sorted([list(x) for x in occ]) if len(occ)<=256 else None,
        }

    # 1-D fallback.
    for _,_,a,ca in cands:
        aa=float(np.dot(a,a))
        if aa==0:continue
        C=((P-origin)@a)/aa
        I=np.rint(C)
        recon=origin+I[:,None]*a
        err=np.linalg.norm(P-recon,axis=1)
        aligned=err<=tol*1.25
        if aligned.sum()>=max(4,int(math.ceil(.98*len(P)))):
            vals=I[aligned].astype(int); vals-=vals.min()
            return {
                "dimension":1,"basis":[a.tolist()],"pitch":[float(np.linalg.norm(a))],
                "aligned":int(aligned.sum()),"max_residual":float(err.max()),
                "occupancy_count":len(set(map(int,vals))),
                "grid_shape":[int(vals.max())+1],
                "fill_ratio":len(set(map(int,vals)))/(int(vals.max())+1),
            }
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--tol",type=float,default=1e-5)
    ap.add_argument("--min-inner-bounds",type=int,default=8)
    ap.add_argument("--min-family",type=int,default=8)
    ap.add_argument("--include-sites",action="store_true")
    args=ap.parse_args()
    path=pathlib.Path(args.input);g=parse(path)
    hits=[]
    for f,t in g.typ.items():
        if t!="ADVANCED_FACE":continue
        bounds=[x for x in g.refs.get(f,()) if g.typ.get(x) in ("FACE_BOUND","FACE_OUTER_BOUND")]
        inner=[x for x in bounds if g.typ.get(x)=="FACE_BOUND"]
        outer=[x for x in bounds if g.typ.get(x)=="FACE_OUTER_BOUND"]
        if len(inner)<args.min_inner_bounds or len(outer)!=1:continue
        plane=next((x for x in g.refs.get(f,()) if g.typ.get(x)=="PLANE"),None)
        frame=plane_frame(g,plane) if plane is not None else None
        if frame is None:continue
        fam=collections.defaultdict(list)
        for b in inner:
            a=loop_anchor(g,b)
            if a is None:continue
            sig=loop_signature(g,b,a,args.tol)
            if sig is None:continue
            fam[sig].append((b,a))
        outer_linear,outer_pts=line_only_outer(g,outer[0])
        for sig,items in fam.items():
            if len(items)<args.min_family:continue
            uv=np.asarray([uv_of(a,frame) for b,a in items])
            lat=fit_lattice(uv,args.tol)
            if lat is None:continue
            if not args.include_sites:lat.pop("integer_sites",None)
            # Distinct edge-curve classes are useful for seeing whether the
            # repeated trim is a circle, rectangle, arbitrary spline, etc.
            curve_hist=collections.Counter()
            for oe,e in loop_edges(g,items[0][0]):
                c=edge_curve(g,e);curve_hist[g.typ.get(c) or "<NONE>"]+=1
            hit={
                "face":f,"plane":plane,"bounds":len(bounds),"inner_bounds":len(inner),
                "family_bounds":len(items),
                "curve_hist":dict(curve_hist),
                "outer_line_only":outer_linear,
                "outer_edges":len(loop_edges(g,outer[0])),
                "lattice":lat,
                "sample_bounds":[x[0] for x in items[:8]],
            }
            hits.append(hit)
    hits.sort(key=lambda h:(-h["family_bounds"],h["face"]))
    print(json.dumps({
        "file":str(path),"bytes":path.stat().st_size,
        "entities":len(g.raw),"tolerance_mm":args.tol,
        "faces_with_lattice_trim_families":len(hits),
        "hits":hits,
    },indent=2))

if __name__=="__main__":
    main()
