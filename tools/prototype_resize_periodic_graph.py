#!/usr/bin/env python3
"""
Prototype pure STEP graph expansion for a proven PeriodicBodyPattern.

Current scope: grow a 1-D body at its positive-axis end. The source graph must
already be semantically normalized (whole-solid instancing before curve
replicas) and accompanied by step-redox periodic-body JSON.

No geometry kernel or boolean operation is used. One isolated repeat-cell face
patch is cloned by translation; the positive fixed/end-cap face graph is cloned
and translated; the eight-ish spanning faces are retained and receive freshly
traced boundary loops over the new shared EDGE_CURVE graph.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import prototype_deboolean_repeated_components as db

RR = re.compile(r"#(\d+)")
CP_FULL = re.compile(r"CARTESIAN_POINT\('([^']*)',\(([^)]*)\)\)")
TOL = 1e-7


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def addv(a, b):
    return tuple(a[i] + b[i] for i in range(3))


def subv(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def scale(a, s):
    return tuple(a[i] * s for i in range(3))


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    n = norm(a)
    if n <= 1e-15:
        raise RuntimeError("zero vector")
    return tuple(x / n for x in a)


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def qcoord(p, tol=TOL):
    return tuple(round(x / tol) for x in p)


class Graph:
    def __init__(self, path: pathlib.Path):
        (
            self.lines,
            self.raw,
            self.typ,
            self.refs,
            self.pts,
            self.inb,
            self.maxid,
        ) = db.parse(path)
        self.path = path
        self.added = []
        self.overrides = {}

    def alloc(self):
        self.maxid += 1
        return self.maxid

    def register(self, i: int, body: str, added=True):
        self.raw[i] = body
        self.typ[i] = "COMPLEX" if body.startswith("(") else body.split("(", 1)[0]
        rs = list(map(int, RR.findall(body)))
        self.refs[i] = rs
        for x in rs:
            self.inb[x].append(i)
        m = CP_FULL.fullmatch(body)
        if m:
            vals = tuple(map(float, m.group(2).split(",")))
            self.pts[i] = vals
        if added:
            self.added.append(i)

    def push(self, body: str):
        i = self.alloc()
        self.register(i, body, True)
        return i

    def override(self, i: int, body: str):
        self.overrides[i] = body
        # Keep query maps coherent for operations after this rewrite.
        self.raw[i] = body
        self.typ[i] = "COMPLEX" if body.startswith("(") else body.split("(", 1)[0]
        self.refs[i] = list(map(int, RR.findall(body)))

    def replace_refs(self, body: str, mapping: dict[int, int]):
        return RR.sub(lambda m: f"#{mapping.get(int(m.group(1)), int(m.group(1)))}", body)

    def clone_descendants(self, seeds, delta):
        ids = sorted(db.descendants(set(seeds), self.refs))
        mapping = {old: self.alloc() for old in ids}
        for old in ids:
            body = self.replace_refs(self.raw[old], mapping)
            if self.typ.get(old) == "CARTESIAN_POINT":
                m = CP_FULL.fullmatch(body)
                if not m:
                    raise RuntimeError(f"cannot parse cloned point #{old}: {body}")
                xyz = tuple(map(float, m.group(2).split(",")))
                xyz = addv(xyz, delta)
                def step_real(value):
                    text = format(value, ".17g")
                    if "e" in text:
                        text = text.replace("e", "E")
                    if "E" not in text and "." not in text:
                        text += "."
                    return text
                body = (
                    f"CARTESIAN_POINT('{m.group(1)}',"
                    f"({step_real(xyz[0])},{step_real(xyz[1])},{step_real(xyz[2])}))"
                )
            self.register(mapping[old], body, True)
        return mapping

    def vertex_coord(self, vertex):
        if self.typ.get(vertex) != "VERTEX_POINT":
            raise RuntimeError(f"#{vertex} is not VERTEX_POINT")
        p = next((x for x in self.refs[vertex] if x in self.pts), None)
        if p is None:
            raise RuntimeError(f"VERTEX_POINT #{vertex} has no point")
        return self.pts[p]

    def edge_vertices(self, edge):
        if self.typ.get(edge) != "EDGE_CURVE":
            raise RuntimeError(f"#{edge} is not EDGE_CURVE")
        vs = [x for x in self.refs[edge] if self.typ.get(x) == "VERTEX_POINT"]
        if len(vs) < 2:
            raise RuntimeError(f"EDGE_CURVE #{edge} lacks endpoints")
        return vs[:2]

    def make_edge_like(self, prototype, va, vb):
        if self.typ.get(prototype) != "EDGE_CURVE":
            raise RuntimeError("edge prototype is not EDGE_CURVE")
        rs = self.refs[prototype]
        curve = next(
            (x for x in rs if self.typ.get(x) not in ("VERTEX_POINT", None)), None
        )
        if curve is None:
            raise RuntimeError(f"edge #{prototype} has no curve support")
        sense = ".T." if self.raw[prototype].rstrip().endswith(".T.)") else ".F."
        return self.push(f"EDGE_CURVE('',#{va},#{vb},#{curve},{sense})")

    def write(self, path: pathlib.Path):
        out = []
        in_data = False
        inserted = False
        for line in self.lines:
            m = db.ER.match(line.strip())
            if m:
                i = int(m.group(1))
                if i in self.overrides:
                    out.append(f"#{i}={self.overrides[i]};")
                    continue
            if line.strip() == "DATA;":
                in_data = True
            if in_data and line.strip() == "ENDSEC;" and not inserted:
                out.extend(f"#{i}={self.raw[i]};" for i in self.added)
                inserted = True
                in_data = False
            out.append(line)
        if not inserted:
            raise RuntimeError("DATA section not found")
        path.write_text("\n".join(out) + "\n")


def face_edges_ordered(g: Graph, face):
    out = []
    for b in g.refs.get(face, ()):
        if g.typ.get(b) not in ("FACE_BOUND", "FACE_OUTER_BOUND"):
            continue
        loop = next((x for x in g.refs[b] if g.typ.get(x) == "EDGE_LOOP"), None)
        if loop is None:
            continue
        for oe in g.refs.get(loop, ()):
            if g.typ.get(oe) != "ORIENTED_EDGE":
                continue
            edge = next((x for x in g.refs[oe] if g.typ.get(x) == "EDGE_CURVE"), None)
            if edge is not None:
                out.append(edge)
    return out


def face_vertices(g: Graph, face):
    out = set()
    for edge in face_edges_ordered(g, face):
        out.update(g.edge_vertices(edge))
    return out


def face_center(g: Graph, face):
    xyz = [g.vertex_coord(v) for v in face_vertices(g, face)]
    if not xyz:
        raise RuntimeError(f"face #{face} has no vertices")
    lo = [min(p[k] for p in xyz) for k in range(3)]
    hi = [max(p[k] for p in xyz) for k in range(3)]
    return tuple((lo[k] + hi[k]) * 0.5 for k in range(3))


def edge_center_span(g: Graph, edge, axis):
    a, b = map(g.vertex_coord, g.edge_vertices(edge))
    pa, pb = dot(a, axis), dot(b, axis)
    return (pa + pb) * 0.5, abs(pb - pa)


def housing_edge_faces(g: Graph, faces):
    out = collections.defaultdict(list)
    for f in faces:
        for e in set(face_edges_ordered(g, f)):
            out[e].append(f)
    return out


def oriented_loop_edges(g: Graph, face):
    result = []
    for b in g.refs.get(face, ()):
        if g.typ.get(b) not in ("FACE_BOUND", "FACE_OUTER_BOUND"):
            continue
        loop = next((x for x in g.refs[b] if g.typ.get(x) == "EDGE_LOOP"), None)
        if loop is None:
            continue
        seq = []
        for oe in g.refs.get(loop, ()):
            if g.typ.get(oe) != "ORIENTED_EDGE":
                continue
            edge = next((x for x in g.refs[oe] if g.typ.get(x) == "EDGE_CURVE"), None)
            if edge is None:
                continue
            va, vb = g.edge_vertices(edge)
            forward = g.raw[oe].rstrip().endswith(".T.)")
            seq.append((edge, va if forward else vb, vb if forward else va))
        result.append((g.typ[b], seq))
    return result


def area_vector(g: Graph, traversal):
    pts = [g.vertex_coord(start) for _, start, _ in traversal]
    if len(pts) < 3:
        return (0.0, 0.0, 0.0)
    acc = (0.0, 0.0, 0.0)
    for a, b in zip(pts, pts[1:] + pts[:1]):
        acc = addv(acc, cross(a, b))
    return scale(acc, 0.5)


def trace_cycles(g: Graph, edges):
    incident = collections.defaultdict(list)
    for e in edges:
        a, b = g.edge_vertices(e)
        incident[a].append(e)
        incident[b].append(e)
    bad = {v: es for v, es in incident.items() if len(es) != 2}
    if bad:
        sample = list(bad.items())[:12]
        raise RuntimeError(f"boundary graph not cycles; bad vertex degrees {sample}")

    unused = set(edges)
    cycles = []
    while unused:
        e0 = next(iter(unused))
        a0, b0 = g.edge_vertices(e0)
        cur_v = a0
        cur_e = e0
        cycle = []
        while True:
            unused.discard(cur_e)
            va, vb = g.edge_vertices(cur_e)
            if cur_v == va:
                next_v = vb
            elif cur_v == vb:
                next_v = va
            else:
                raise RuntimeError("cycle traversal lost endpoint")
            cycle.append((cur_e, cur_v, next_v))
            cur_v = next_v
            if cur_v == a0:
                break
            candidates = [e for e in incident[cur_v] if e in unused]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"cycle continuation at #{cur_v}: {candidates}, incident={incident[cur_v]}"
                )
            cur_e = candidates[0]
        cycles.append(cycle)
    return cycles


def reverse_cycle(cycle):
    return [(e, b, a) for e, a, b in reversed(cycle)]


def rewrite_face_bounds(g: Graph, face, target_edges):
    source_loops = oriented_loop_edges(g, face)
    source_outer = next((seq for kind, seq in source_loops if kind == "FACE_OUTER_BOUND"), None)
    if source_outer is None:
        raise RuntimeError(f"stretch face #{face} lacks outer bound")
    ref_area = area_vector(g, source_outer)
    if norm(ref_area) <= 1e-12:
        raise RuntimeError(f"stretch face #{face} has degenerate outer area")

    hole_sign = -1.0
    source_hole = next((seq for kind, seq in source_loops if kind == "FACE_BOUND"), None)
    if source_hole:
        hole_sign = 1.0 if dot(area_vector(g, source_hole), ref_area) > 0 else -1.0

    cycles = trace_cycles(g, target_edges)
    scored = [(abs(dot(area_vector(g, c), ref_area)), c) for c in cycles]
    outer_index = max(range(len(scored)), key=lambda i: scored[i][0])

    new_bounds = []
    for i, (_, cycle) in enumerate(scored):
        desired = 1.0 if i == outer_index else hole_sign
        actual = dot(area_vector(g, cycle), ref_area)
        if actual == 0:
            raise RuntimeError(f"zero-area cycle on face #{face}")
        if (actual > 0) != (desired > 0):
            cycle = reverse_cycle(cycle)

        oes = []
        for edge, start, end in cycle:
            va, vb = g.edge_vertices(edge)
            if start == va and end == vb:
                orient = ".T."
            elif start == vb and end == va:
                orient = ".F."
            else:
                raise RuntimeError("cycle endpoint mismatch")
            oes.append(g.push(f"ORIENTED_EDGE('',*,*,#{edge},{orient})"))
        loop = g.push("EDGE_LOOP('',(" + ",".join(f"#{x}" for x in oes) + "))")
        kind = "FACE_OUTER_BOUND" if i == outer_index else "FACE_BOUND"
        new_bounds.append(g.push(f"{kind}('',#{loop},.T.)"))

    body = g.raw[face]
    start = body.find("(#")
    if start < 0:
        raise RuntimeError(f"cannot find bounds aggregate in face #{face}")
    depth = 0
    end = None
    for i in range(start, len(body)):
        if body[i] == "(":
            depth += 1
        elif body[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        raise RuntimeError("unterminated face aggregate")
    agg = "(" + ",".join(f"#{x}" for x in new_bounds) + ")"
    g.override(face, body[:start] + agg + body[end + 1 :])


def clone_styles_for_faces(g: Graph, face_mapping):
    # Presentation is non-geometric, but preserving it makes the prototype a
    # useful product fixture too. Clone STYLED_ITEMs and append them to each
    # presentation representation that owned the source style.
    new_styles = []
    style_owner_add = collections.defaultdict(list)
    for sid, t in list(g.typ.items()):
        if t != "STYLED_ITEM":
            continue
        rs = g.refs.get(sid, ())
        if not rs:
            continue
        old_face = rs[-1]
        if old_face not in face_mapping:
            continue
        body = g.raw[sid]
        # Only retarget the final face reference; style assignment graph remains shared.
        matches = list(RR.finditer(body))
        if not matches:
            continue
        last = matches[-1]
        new_body = body[: last.start()] + f"#{face_mapping[old_face]}" + body[last.end() :]
        ns = g.push(new_body)
        new_styles.append(ns)
        for owner in g.inb.get(sid, ()):
            if g.typ.get(owner) == "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION":
                style_owner_add[owner].append(ns)

    for owner, ids in style_owner_add.items():
        body = g.raw[owner]
        start = body.find("(#")
        if start < 0:
            continue
        depth = 0
        end = None
        for i in range(start, len(body)):
            if body[i] == "(":
                depth += 1
            elif body[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        old_ids = list(map(int, RR.findall(body[start : end + 1])))
        agg = "(" + ",".join(f"#{x}" for x in old_ids + ids) + ")"
        g.override(owner, body[:start] + agg + body[end + 1 :])
    return len(new_styles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("body_json", type=pathlib.Path)
    ap.add_argument("dst", type=pathlib.Path)
    ap.add_argument("--target-sites", type=int, required=True)
    args = ap.parse_args()

    body = json.loads(args.body_json.read_text())
    if isinstance(body, list):
        if len(body) != 1:
            raise RuntimeError("prototype requires exactly one periodic body")
        body = body[0]

    g = Graph(args.src)
    solid = body["solid_id"]
    axis = unit(tuple(body["axis"]))
    pitch = float(body["pitch_mm"])
    source_sites = int(body["sites"])
    target_sites = args.target_sites
    if target_sites <= source_sites:
        raise RuntimeError("prototype currently supports expansion only")
    extra = target_sites - source_sites
    delta_total = scale(axis, extra * pitch)

    shell = next((x for x in g.refs[solid] if g.typ.get(x) == "CLOSED_SHELL"), None)
    if shell is None:
        raise RuntimeError("periodic body solid lacks CLOSED_SHELL")
    shell_faces = list(g.refs[shell])

    families = body["repeat_face_families"]
    repeat_faces = {f for fam in families for f in fam["face_ids"]}
    stretch_faces = set(body["stretch_face_ids"])
    fixed_faces = set(body["fixed_face_ids"])
    housing_faces = repeat_faces | stretch_faces | fixed_faces

    edge_faces = housing_edge_faces(g, housing_faces)

    # Site face table and one interior canonical patch.
    site_faces = [
        [fam["face_ids"][site] for fam in families] for site in range(source_sites)
    ]
    canonical_site = source_sites // 2
    canonical_faces = site_faces[canonical_site]

    # Project approximate site location from the first family's face center;
    # only differences matter for translation.
    canonical_projection = dot(face_center(g, canonical_faces[0]), axis)

    # Fixed-face end classification. The periodic cells occupy an ordered band;
    # everything fixed beyond its positive end is the movable positive cap.
    site_projection = [
        sum(dot(face_center(g, f), axis) for f in site_faces[i]) / len(site_faces[i])
        for i in range(source_sites)
    ]
    right_threshold = site_projection[-1] + pitch * 0.5
    right_fixed = {
        f for f in fixed_faces if dot(face_center(g, f), axis) > right_threshold - TOL
    }
    left_fixed = fixed_faces - right_fixed
    if not right_fixed or not left_fixed:
        raise RuntimeError(
            f"failed to split fixed faces into ends: left={len(left_fixed)} right={len(right_fixed)}"
        )

    # Clone the positive end cap as a rigid graph.
    cap_map = g.clone_descendants(right_fixed, delta_total)
    new_right_fixed = {cap_map[f] for f in right_fixed}

    # Clone canonical site patch for every newly-added site.
    new_site_faces = []
    site_maps = {}
    for site in range(source_sites, target_sites):
        d = scale(axis, (site - canonical_site) * pitch)
        mapping = g.clone_descendants(canonical_faces, d)
        site_maps[site] = mapping
        faces = [mapping[f] for f in canonical_faces]
        new_site_faces.append(faces)

    # Build coordinate->vertex lookup only from the target housing parts that
    # physically define stretch-face boundaries.
    target_patch_faces = set().union(*map(set, site_faces))
    for fs in new_site_faces:
        target_patch_faces.update(fs)
    target_patch_faces.update(left_fixed)
    target_patch_faces.update(new_right_fixed)

    coord_vertex = collections.defaultdict(list)
    for f in target_patch_faces:
        for v in face_vertices(g, f):
            coord_vertex[qcoord(g.vertex_coord(v))].append(v)

    def find_vertex(p):
        candidates = coord_vertex.get(qcoord(p), ())
        if not candidates:
            raise RuntimeError(f"no target vertex at {p}")
        # All source cell coordinates were unique; cloned cap can occasionally
        # expose the same topological corner via several faces but should share
        # one VERTEX_POINT entity.
        unique = sorted(set(candidates))
        if len(unique) != 1:
            raise RuntimeError(f"ambiguous target vertex at {p}: {unique}")
        return unique[0]

    # Source adjacency tells which cell boundary curves belong to which stretch
    # face. Existing cells remain untouched; cloned cells get mapped curves.
    stretch_edges = {f: set() for f in stretch_faces}
    canonical_boundary = []
    for edge, fs in edge_faces.items():
        r = [f for f in fs if f in canonical_faces]
        s = [f for f in fs if f in stretch_faces]
        if r and s:
            canonical_boundary.append((edge, s[0]))

    # Existing repeat->stretch edges.
    for edge, fs in edge_faces.items():
        if any(f in repeat_faces for f in fs):
            for s in (f for f in fs if f in stretch_faces):
                stretch_edges[s].add(edge)

    # New repeat->stretch edges.
    for site, mapping in site_maps.items():
        for source_edge, stretch in canonical_boundary:
            stretch_edges[stretch].add(mapping[source_edge])

    # Fixed->stretch edges. Positive-cap curves are cloned; negative-cap curves
    # remain the original graph.
    for edge, fs in edge_faces.items():
        fixed = [f for f in fs if f in fixed_faces]
        stretches = [f for f in fs if f in stretch_faces]
        if not fixed or not stretches:
            continue
        if fixed[0] in right_fixed:
            target_edge = cap_map[edge]
        else:
            target_edge = edge
        for s in stretches:
            stretch_edges[s].add(target_edge)

    # Stretch<->stretch edges: six full-length rails plus, for the two complex
    # side pairs, [left end][N-1 gaps][right end].
    ss_by_pair = collections.defaultdict(list)
    for edge, fs in edge_faces.items():
        ss = sorted(f for f in fs if f in stretch_faces)
        if len(ss) == 2 and len(fs) == 2:
            center, span = edge_center_span(g, edge, axis)
            ss_by_pair[tuple(ss)].append((center, span, edge))

    new_ss_edges = []
    for pair, rows in ss_by_pair.items():
        rows.sort()
        full = [r for r in rows if r[1] > (source_sites - 1) * pitch]
        short = [r for r in rows if r not in full]

        for center, span, edge in full:
            va, vb = g.edge_vertices(edge)
            pa, pb = g.vertex_coord(va), g.vertex_coord(vb)
            if dot(pa, axis) > dot(pb, axis):
                va, vb = vb, va
                pa, pb = pb, pa
            target_right = addv(pb, delta_total)
            nv = find_vertex(target_right)
            ne = g.make_edge_like(edge, va, nv)
            new_ss_edges.append(ne)
            for s in pair:
                stretch_edges[s].add(ne)

        if not short:
            continue
        if len(short) != source_sites + 1:
            raise RuntimeError(
                f"unexpected short stretch-edge grammar for pair {pair}: {len(short)}"
            )
        left_end = short[0]
        right_end = short[-1]
        gaps = short[1:-1]

        # Keep left end and all original inter-site gaps.
        for row in [left_end] + gaps:
            for s in pair:
                stretch_edges[s].add(row[2])

        # Add one new gap for each added site, translated from the rightmost
        # existing gap by +pitch, +2*pitch, ...
        prototype = gaps[-1]
        for j in range(1, extra + 1):
            d = scale(axis, j * pitch)
            edge = prototype[2]
            va, vb = g.edge_vertices(edge)
            pva, pvb = addv(g.vertex_coord(va), d), addv(g.vertex_coord(vb), d)
            nva, nvb = find_vertex(pva), find_vertex(pvb)
            ne = g.make_edge_like(edge, nva, nvb)
            new_ss_edges.append(ne)
            for s in pair:
                stretch_edges[s].add(ne)

        # Translate the right-end segment; its cell-side endpoint comes from
        # the last cloned site and its cap-side endpoint from the cloned cap.
        edge = right_end[2]
        va, vb = g.edge_vertices(edge)
        pva = addv(g.vertex_coord(va), delta_total)
        pvb = addv(g.vertex_coord(vb), delta_total)
        nva, nvb = find_vertex(pva), find_vertex(pvb)
        ne = g.make_edge_like(edge, nva, nvb)
        new_ss_edges.append(ne)
        for s in pair:
            stretch_edges[s].add(ne)

    # Rebuild stretch-face bounds solely from their target shared EDGE_CURVE set.
    for f in sorted(stretch_faces):
        rewrite_face_bounds(g, f, stretch_edges[f])

    # Replace right fixed faces in the shell and append new repeat-cell faces.
    shell_target = []
    for f in shell_faces:
        if f in right_fixed:
            shell_target.append(cap_map[f])
        else:
            shell_target.append(f)
    for fs in new_site_faces:
        shell_target.extend(fs)
    g.override(
        shell,
        "CLOSED_SHELL('',(" + ",".join(f"#{f}" for f in shell_target) + "))",
    )

    # Clone presentation styles for new cells and the translated cap.
    face_map = {old: cap_map[old] for old in right_fixed}
    for site, mapping in site_maps.items():
        for old in canonical_faces:
            face_map[old] = mapping[old]
    style_clones = clone_styles_for_faces(g, face_map)

    g.write(args.dst)

    print(
        json.dumps(
            {
                "source_sites": source_sites,
                "target_sites": target_sites,
                "pitch_mm": pitch,
                "axis": axis,
                "canonical_site": canonical_site,
                "canonical_faces": len(canonical_faces),
                "right_fixed_faces": len(right_fixed),
                "left_fixed_faces": len(left_fixed),
                "new_repeat_faces": sum(len(x) for x in new_site_faces),
                "new_ss_edges": len(new_ss_edges),
                "stretch_edge_counts": {
                    str(f): len(edges) for f, edges in sorted(stretch_edges.items())
                },
                "style_clones": style_clones,
                "added_entities": len(g.added),
                "output_bytes": args.dst.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
