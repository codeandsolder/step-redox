#!/usr/bin/env python3
from __future__ import annotations
import argparse, collections, json, math, pathlib, sys

import prototype_deboolean_repeated_components as db

def q(v, tol):
    return int(round(float(v) / tol))

def face_info(face, raw, typ, refs, pts, fes, style, tol):
    verts = sorted(set(pts[v] for v in db.face_vertex_ids(face, fes, typ, refs, pts)))
    if not verts:
        return None
    lo = [min(p[k] for p in verts) for k in range(3)]
    hi = [max(p[k] for p in verts) for k in range(3)]
    center = tuple((lo[k] + hi[k]) * 0.5 for k in range(3))
    size = tuple(hi[k] - lo[k] for k in range(3))
    local = tuple(sorted(tuple(q(p[k] - center[k], tol) for k in range(3)) for p in verts))
    sid = db.face_surface(face, typ, refs)
    bounds = sum(typ.get(x) in ("FACE_BOUND", "FACE_OUTER_BOUND") for x in refs.get(face, ()))
    sense = ".T." if raw.get(face, "").rstrip().endswith(".T.)") else (
        ".F." if raw.get(face, "").rstrip().endswith(".F.)") else "?"
    )
    return {
        "face": face,
        "center": center,
        "size": size,
        "cheap": (
            style.get(face, ()),
            typ.get(sid) or "<NONE>",
            len(fes[face]),
            bounds,
            sense,
            tuple(q(x, tol) for x in size),
            local,
        ),
    }

def exact_clusters(items, raw, typ, refs, pts, fes, tol):
    clusters = []
    for item in items:
        placed = False
        for cluster in clusters:
            canon = cluster[0]
            delta = tuple(item["center"][k] - canon["center"][k] for k in range(3))
            if db.face_translation_key(canon["face"], delta, raw, typ, refs, pts, fes, tol) ==                db.face_translation_key(item["face"], (0.0,0.0,0.0), raw, typ, refs, pts, fes, tol):
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
    return clusters

def lattice_1d(centers, tol):
    if len(centers) < 2:
        return None
    spans = [max(c[k] for c in centers) - min(c[k] for c in centers) for k in range(3)]
    axis = max(range(3), key=lambda k: spans[k])
    if spans[axis] <= tol:
        return None
    # A single exact translated-face family should move along only one axis.
    other = [k for k in range(3) if k != axis]
    if any(spans[k] > tol * 4 for k in other):
        return None
    coords = sorted(set(q(c[axis], tol) for c in centers))
    if len(coords) != len(centers):
        return None
    diffs = [coords[i+1] - coords[i] for i in range(len(coords)-1) if coords[i+1] != coords[i]]
    if not diffs:
        return None
    step = 0
    for d in diffs:
        step = math.gcd(step, abs(d))
    if step <= 0:
        return None
    lo, hi = coords[0], coords[-1]
    span_sites = (hi - lo) // step + 1
    if (hi - lo) % step:
        return None
    sites = [(x - lo) // step for x in coords]
    return {
        "axis": axis,
        "pitch_mm": step * tol,
        "count": len(coords),
        "grid_shape": int(span_sites),
        "fill_ratio": len(coords) / span_sites,
        "sites": sites,
        "start_mm": lo * tol,
        "end_mm": hi * tol,
        "other_span_mm": [spans[k] for k in other],
    }

def split_parallel_rows(cluster, tol, min_instances):
    """Split one exact congruence class into parallel axis-aligned rows.

    Exporters commonly reuse the same local face geometry at several Y/Z
    offsets. Exact translation congruence correctly groups all of those faces
    together, but that aggregate is not itself 1-D. Choose the dominant span
    axis and partition by the two orthogonal center coordinates before fitting
    a lattice.
    """
    if len(cluster) < min_instances:
        return []
    spans = [
        max(x["center"][k] for x in cluster) - min(x["center"][k] for x in cluster)
        for k in range(3)
    ]
    axis = max(range(3), key=lambda k: spans[k])
    if spans[axis] <= tol:
        return []
    other = [k for k in range(3) if k != axis]
    rows = collections.defaultdict(list)
    for item in cluster:
        rows[tuple(q(item["center"][k], tol) for k in other)].append(item)
    return [row for row in rows.values() if len(row) >= min_instances]

def analyze(path: pathlib.Path, tol: float, min_instances: int):
    db._GEOM_PAYLOAD_CACHE.clear()
    lines,raw,typ,refs,pts,inb,maxid = db.parse(path)
    solids=[i for i,t in typ.items() if t=="MANIFOLD_SOLID_BREP"]
    if not solids:
        raise RuntimeError("no MANIFOLD_SOLID_BREP")

    solid_faces=[]
    for s in solids:
        cl=db.descendants(s,refs)
        fs=[x for x in cl if typ.get(x)=="ADVANCED_FACE"]
        solid_faces.append((len(fs),s,fs))
    solid_faces.sort(reverse=True)
    face_count, housing, faces = solid_faces[0]

    fes={f:db.face_edges(f,typ,refs) for f in faces}
    style=db.parse_style_map(raw,typ,refs)
    buckets=collections.defaultdict(list)
    rejected=0
    for f in faces:
        info=face_info(f,raw,typ,refs,pts,fes,style,tol)
        if info is None:
            rejected+=1
            continue
        buckets[info["cheap"]].append(info)

    families=[]
    singleton_exact=0
    for cheap,items in buckets.items():
        if len(items)<min_instances:
            continue
        for cluster in exact_clusters(items,raw,typ,refs,pts,fes,tol):
            if len(cluster)<min_instances:
                singleton_exact+=len(cluster)
                continue

            rows=[cluster]
            if lattice_1d([x["center"] for x in cluster],tol) is None:
                rows=split_parallel_rows(cluster,tol,min_instances)

            for row in rows:
                lat=lattice_1d([x["center"] for x in row],tol)
                if lat is None:
                    continue
                families.append({
                    "instances":len(row),
                    "surface_type":cheap[1],
                    "edges":cheap[2],
                    "bounds":cheap[3],
                    "sense":cheap[4],
                    "size_mm":[x*tol for x in cheap[5]],
                    "lattice":lat,
                    "sample_faces":[x["face"] for x in row[:12]],
                    "sample_centers":[list(x["center"]) for x in row[:12]],
                })

    families.sort(key=lambda x:(-x["instances"],x["lattice"]["pitch_mm"],x["surface_type"]))
    return {
        "file":str(path),
        "bytes":path.stat().st_size,
        "solids":len(solids),
        "housing_solid":housing,
        "housing_faces":face_count,
        "faces_without_vertices":rejected,
        "cheap_buckets":len(buckets),
        "periodic_face_families":len(families),
        "periodic_face_instances":sum(x["instances"] for x in families),
        "families":families,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("steps",nargs="+")
    ap.add_argument("--tol",type=float,default=1e-5)
    ap.add_argument("--min-instances",type=int,default=3)
    args=ap.parse_args()
    for p in args.steps:
        result=analyze(pathlib.Path(p),args.tol,args.min_instances)
        print(json.dumps(result,separators=(",",":")))

if __name__=="__main__":
    main()
