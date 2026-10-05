use crate::math3::{add, cross, distance, dot, mul, norm, normalize as normalize3, sub};
use crate::step_graph::{build_index, entity_id, entity_ref_value, simple_record};
use ruststep::ast::{EntityInstance, Parameter, Record};
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

const GEOM_TOL_MM: f64 = 1.0e-5;
const DIR_TOL: f64 = 1.0e-10;
const MIN_COAXIAL_PAIRS: usize = 3;
const MIN_PARALLEL_PLANE_PAIRS: usize = 2;
const MIN_PROJECTED_OVERLAP_RATIO: f64 = 0.5;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FormedSheetEvidence {
    pub solid_id: u64,
    pub thickness_mm: f64,
    pub total_faces: usize,
    pub cylindrical_faces: usize,
    pub paired_cylindrical_faces: usize,
    pub paired_planar_faces: usize,
    pub coaxial_radius_pairs: usize,
    /// Raw count of parallel plane-support pairs at the dominant thickness.
    /// This can exceed paired_planar_patches because spatially disjoint
    /// faces can happen to lie one material thickness apart.
    pub parallel_plane_pairs: usize,
    pub paired_planar_patches: usize,
    pub paired_support_patch_count: usize,
    pub patch_adjacencies: Vec<SheetPatchAdjacency>,
    pub patch_graph_connected: bool,
    pub patch_graph_is_tree: bool,
    pub reference_skin: Option<SheetReferenceSkin>,
    pub sidewall_faces: usize,
    pub paired_cylinder_face_ratio: f64,
    pub cylinder_pairs: Vec<SheetCylinderPair>,
    pub plane_pairs: Vec<SheetPlanePair>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SheetCylinderPair {
    pub inner_face_id: u64,
    pub outer_face_id: u64,
    pub inner_radius_mm: f64,
    pub outer_radius_mm: f64,
    pub mid_radius_mm: f64,
    /// Canonical point on the cylinder axis: the point nearest world origin.
    pub axis_origin_mm: [f64; 3],
    pub axis: [f64; 3],
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SheetPlanePair {
    pub negative_face_id: u64,
    pub positive_face_id: u64,
    /// Canonical unit normal shared by both support planes.
    pub normal: [f64; 3],
    /// Arithmetic midpoint between paired support planes. Kept as useful geometry
    /// evidence; the canonical sheet does not assume this is its reference surface.
    pub mid_offset_mm: f64,
    pub separation_mm: f64,
    /// AABB overlap after both trimmed faces are projected onto their common plane.
    /// It is evidence for matching the two skins, not a claim that their trims are
    /// literal translations.
    pub projected_overlap_ratio: f64,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash)]
#[serde(tag = "kind", content = "index", rename_all = "snake_case")]
pub enum SheetPatchId {
    Plane(usize),
    Cylinder(usize),
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SheetPatchAdjacency {
    pub patch_a: SheetPatchId,
    pub patch_b: SheetPatchId,
    /// True when the two source-skin face pairings cross relative to each patch's
    /// local ordering (negative/positive for planes, inner/outer for cylinders).
    pub crossed_source_sides: bool,
    /// Source EDGE_CURVEs proving the adjacency, normally one on each skin.
    pub source_edge_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SheetReferenceSkin {
    pub face_ids: Vec<u64>,
    pub patches: Vec<SheetReferencePatch>,
    pub unique_edges: usize,
    pub boundary_loops: Vec<SheetReferenceLoop>,
    pub internal_seams: Vec<SheetReferenceCurve>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SheetReferencePatch {
    pub patch: SheetPatchId,
    pub source_face_id: u64,
    pub paired_face_id: u64,
    pub surface: SheetReferenceSurface,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SheetReferenceSurface {
    Plane {
        normal: [f64; 3],
        offset_mm: f64,
        paired_offset_mm: f64,
    },
    Cylinder {
        axis_origin_mm: [f64; 3],
        axis: [f64; 3],
        radius_mm: f64,
        paired_radius_mm: f64,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SheetReferenceLoop {
    pub curves: Vec<SheetReferenceCurve>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SheetReferenceCurve {
    Line {
        source_edge_id: u64,
        patches: Vec<SheetPatchId>,
        start_mm: [f64; 3],
        end_mm: [f64; 3],
    },
    CircleArc {
        source_edge_id: u64,
        patches: Vec<SheetPatchId>,
        center_mm: [f64; 3],
        normal: [f64; 3],
        radius_mm: f64,
        start_mm: [f64; 3],
        end_mm: [f64; 3],
        sweep_angle_rad: f64,
    },
    EllipseArc {
        source_edge_id: u64,
        patches: Vec<SheetPatchId>,
        center_mm: [f64; 3],
        u_axis_mm: [f64; 3],
        v_axis_mm: [f64; 3],
        start_angle_rad: f64,
        sweep_angle_rad: f64,
        start_mm: [f64; 3],
        end_mm: [f64; 3],
    },
}

#[derive(Debug, Clone)]
struct PlaneFace {
    face_id: u64,
    origin: [f64; 3],
    normal: [f64; 3],
    vertices: Vec<[f64; 3]>,
}

#[derive(Debug, Clone)]
struct CylinderFace {
    face_id: u64,
    origin: [f64; 3],
    axis: [f64; 3],
    radius_mm: f64,
}

#[derive(Debug, Clone)]
struct PlanePairCandidate {
    first: usize,
    second: usize,
    overlap_area: f64,
    overlap_ratio: f64,
    separation_mm: f64,
}

#[must_use]
pub fn detect_formed_sheet_evidence(entities: &[EntityInstance]) -> Vec<FormedSheetEvidence> {
    let index = build_index(entities);
    let mut out = Vec::new();

    for entity in entities {
        let Some(record) = simple_record(entity) else {
            continue;
        };
        if record.name != "MANIFOLD_SOLID_BREP" {
            continue;
        }
        let solid_id = entity_id(entity);
        let Some(face_ids) = solid_faces(solid_id, entities, &index) else {
            continue;
        };

        let mut planes = Vec::new();
        let mut cylinders = Vec::new();
        for &face_id in &face_ids {
            let Some((surface_id, surface)) = face_surface(face_id, entities, &index) else {
                continue;
            };
            match surface.name.as_str() {
                "PLANE" => {
                    if let (Some((origin, normal)), Some(vertices)) = (
                        surface_axis(surface_id, entities, &index),
                        face_vertices(face_id, entities, &index),
                    ) {
                        if vertices.len() >= 3 {
                            planes.push(PlaneFace {
                                face_id,
                                origin,
                                normal,
                                vertices,
                            });
                        }
                    }
                }
                "CYLINDRICAL_SURFACE" => {
                    let Some((origin, axis)) = surface_axis(surface_id, entities, &index) else {
                        continue;
                    };
                    let Some(radius_mm) = surface_radius(surface) else {
                        continue;
                    };
                    if radius_mm.is_finite() && radius_mm > GEOM_TOL_MM {
                        cylinders.push(CylinderFace {
                            face_id,
                            origin,
                            axis,
                            radius_mm,
                        });
                    }
                }
                _ => {}
            }
        }

        if cylinders.len() < MIN_COAXIAL_PAIRS * 2 || planes.len() < 2 {
            continue;
        }

        let cylinder_candidates = coaxial_radius_differences(&cylinders);
        if cylinder_candidates.is_empty() {
            continue;
        }
        let plane_separations = parallel_plane_separations(&planes);

        // First infer the dominant material thickness from two independent geometric
        // signals. Candidate pairs are intentionally not yet made one-to-one here:
        // this stage is only a thickness vote.
        let mut thickness_scores = BTreeMap::<i64, (usize, usize)>::new();
        for pair in &cylinder_candidates {
            let thickness = pair.outer_radius_mm - pair.inner_radius_mm;
            let tick = quantize_mm(thickness);
            if tick > 0 {
                thickness_scores.entry(tick).or_default().0 += 1;
            }
        }
        for separation in plane_separations {
            let tick = quantize_mm(separation);
            if let Some(entry) = thickness_scores.get_mut(&tick) {
                entry.1 += 1;
            }
        }

        let Some((&tick, &(raw_cylinder_pairs, raw_plane_pairs))) = thickness_scores
            .iter()
            .max_by_key(|(tick, (cyl, plane))| (*cyl, *plane, -tick.abs()))
        else {
            continue;
        };
        if raw_cylinder_pairs < MIN_COAXIAL_PAIRS || raw_plane_pairs < MIN_PARALLEL_PLANE_PAIRS {
            continue;
        }
        let thickness_mm = tick as f64 * GEOM_TOL_MM;

        let cylinder_pairs = select_cylinder_pairs(cylinder_candidates, thickness_mm);
        let plane_pairs = select_plane_pairs(&planes, thickness_mm);
        if cylinder_pairs.len() < MIN_COAXIAL_PAIRS || plane_pairs.len() < MIN_PARALLEL_PLANE_PAIRS
        {
            continue;
        }

        let paired_cylindrical_faces = cylinder_pairs.len() * 2;
        let paired_planar_faces = plane_pairs.len() * 2;
        let paired_cylinder_face_ratio = paired_cylindrical_faces as f64 / cylinders.len() as f64;

        let face_edge_map = face_edge_map(&face_ids, entities, &index);
        let patch_adjacencies = patch_adjacencies(&plane_pairs, &cylinder_pairs, &face_edge_map);
        let paired_support_patch_count = plane_pairs.len() + cylinder_pairs.len();
        let patch_graph_connected =
            graph_connected(plane_pairs.len(), cylinder_pairs.len(), &patch_adjacencies);
        let patch_graph_is_tree =
            patch_graph_connected && patch_adjacencies.len() + 1 == paired_support_patch_count;
        let paired_surface_faces = paired_cylindrical_faces + paired_planar_faces;
        let sidewall_faces = face_ids.len().saturating_sub(paired_surface_faces);
        let reference_skin = recover_reference_skin(
            &plane_pairs,
            &cylinder_pairs,
            &patch_adjacencies,
            &face_edge_map,
            entities,
            &index,
        );

        out.push(FormedSheetEvidence {
            solid_id,
            thickness_mm,
            total_faces: face_ids.len(),
            cylindrical_faces: cylinders.len(),
            paired_cylindrical_faces,
            paired_planar_faces,
            coaxial_radius_pairs: cylinder_pairs.len(),
            parallel_plane_pairs: raw_plane_pairs,
            paired_planar_patches: plane_pairs.len(),
            paired_support_patch_count,
            patch_adjacencies,
            patch_graph_connected,
            patch_graph_is_tree,
            reference_skin,
            sidewall_faces,
            paired_cylinder_face_ratio,
            cylinder_pairs,
            plane_pairs,
        });
    }

    out.sort_by(|a, b| {
        b.paired_cylinder_face_ratio
            .total_cmp(&a.paired_cylinder_face_ratio)
            .then_with(|| {
                b.paired_support_patch_count
                    .cmp(&a.paired_support_patch_count)
            })
            .then_with(|| a.solid_id.cmp(&b.solid_id))
    });
    out
}

fn coaxial_radius_differences(cylinders: &[CylinderFace]) -> Vec<SheetCylinderPair> {
    let mut out = Vec::new();
    for (index, a) in cylinders.iter().enumerate() {
        for b in &cylinders[index + 1..] {
            if !parallel(a.axis, b.axis) {
                continue;
            }
            if axis_line_distance(a.origin, a.axis, b.origin) > GEOM_TOL_MM {
                continue;
            }
            let (inner, outer) = if a.radius_mm <= b.radius_mm {
                (a, b)
            } else {
                (b, a)
            };
            let thickness = outer.radius_mm - inner.radius_mm;
            if thickness <= GEOM_TOL_MM {
                continue;
            }
            let axis = canonical_axis(inner.axis);
            out.push(SheetCylinderPair {
                inner_face_id: inner.face_id,
                outer_face_id: outer.face_id,
                inner_radius_mm: inner.radius_mm,
                outer_radius_mm: outer.radius_mm,
                mid_radius_mm: (inner.radius_mm + outer.radius_mm) * 0.5,
                axis_origin_mm: canonical_line_origin(inner.origin, axis),
                axis,
            });
        }
    }
    out
}

fn select_cylinder_pairs(
    mut candidates: Vec<SheetCylinderPair>,
    thickness_mm: f64,
) -> Vec<SheetCylinderPair> {
    candidates.retain(|pair| {
        ((pair.outer_radius_mm - pair.inner_radius_mm) - thickness_mm).abs() <= GEOM_TOL_MM
    });
    candidates.sort_by(|a, b| {
        a.inner_face_id
            .cmp(&b.inner_face_id)
            .then_with(|| a.outer_face_id.cmp(&b.outer_face_id))
    });

    let mut used = HashSet::new();
    let mut out = Vec::new();
    for pair in candidates {
        if used.contains(&pair.inner_face_id) || used.contains(&pair.outer_face_id) {
            continue;
        }
        used.insert(pair.inner_face_id);
        used.insert(pair.outer_face_id);
        out.push(pair);
    }
    out
}

fn parallel_plane_separations(planes: &[PlaneFace]) -> Vec<f64> {
    let mut out = Vec::new();
    for (index, a) in planes.iter().enumerate() {
        for b in &planes[index + 1..] {
            if !parallel(a.normal, b.normal) {
                continue;
            }
            let normal = canonical_axis(a.normal);
            let separation = (dot(normal, b.origin) - dot(normal, a.origin)).abs();
            if separation > GEOM_TOL_MM && separation.is_finite() {
                out.push(separation);
            }
        }
    }
    out
}

fn select_plane_pairs(planes: &[PlaneFace], thickness_mm: f64) -> Vec<SheetPlanePair> {
    let mut candidates = Vec::new();
    for first in 0..planes.len() {
        for second in first + 1..planes.len() {
            let a = &planes[first];
            let b = &planes[second];
            if !parallel(a.normal, b.normal) {
                continue;
            }
            let normal = canonical_axis(a.normal);
            let separation = (dot(normal, b.origin) - dot(normal, a.origin)).abs();
            if (separation - thickness_mm).abs() > GEOM_TOL_MM {
                continue;
            }
            let Some((overlap_area, overlap_ratio)) =
                projected_overlap(&a.vertices, &b.vertices, normal)
            else {
                continue;
            };
            if overlap_ratio + f64::EPSILON < MIN_PROJECTED_OVERLAP_RATIO {
                continue;
            }
            candidates.push(PlanePairCandidate {
                first,
                second,
                overlap_area,
                overlap_ratio,
                separation_mm: separation,
            });
        }
    }

    // Broad-face pairs generally maximize common projected area. Greedy matching is
    // deterministic and also prevents an intermediate support plane from being paired
    // to both skins when several parallel faces are one thickness apart.
    candidates.sort_by(|a, b| {
        b.overlap_area
            .total_cmp(&a.overlap_area)
            .then_with(|| b.overlap_ratio.total_cmp(&a.overlap_ratio))
            .then_with(|| planes[a.first].face_id.cmp(&planes[b.first].face_id))
            .then_with(|| planes[a.second].face_id.cmp(&planes[b.second].face_id))
    });

    let mut used = HashSet::new();
    let mut out = Vec::new();
    for candidate in candidates {
        let a = &planes[candidate.first];
        let b = &planes[candidate.second];
        if used.contains(&a.face_id) || used.contains(&b.face_id) {
            continue;
        }

        let normal = canonical_axis(a.normal);
        let da = dot(normal, a.origin);
        let db = dot(normal, b.origin);
        let (negative_face_id, positive_face_id, negative_offset, positive_offset) = if da <= db {
            (a.face_id, b.face_id, da, db)
        } else {
            (b.face_id, a.face_id, db, da)
        };

        used.insert(a.face_id);
        used.insert(b.face_id);
        out.push(SheetPlanePair {
            negative_face_id,
            positive_face_id,
            normal,
            mid_offset_mm: (negative_offset + positive_offset) * 0.5,
            separation_mm: candidate.separation_mm,
            projected_overlap_ratio: candidate.overlap_ratio,
        });
    }
    out.sort_by_key(|pair| (pair.negative_face_id, pair.positive_face_id));
    out
}

fn projected_overlap(
    first: &[[f64; 3]],
    second: &[[f64; 3]],
    normal: [f64; 3],
) -> Option<(f64, f64)> {
    let (u, v) = plane_basis(normal)?;
    let a = projected_bbox(first, u, v)?;
    let b = projected_bbox(second, u, v)?;

    let overlap_u = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let overlap_v = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let overlap_area = overlap_u * overlap_v;
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let smaller = area_a.min(area_b);
    if !overlap_area.is_finite() || smaller <= GEOM_TOL_MM * GEOM_TOL_MM {
        return None;
    }
    Some((overlap_area, overlap_area / smaller))
}

fn plane_basis(normal: [f64; 3]) -> Option<([f64; 3], [f64; 3])> {
    let reference = if normal[0].abs() < 0.8 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = normalize(cross(normal, reference))?;
    let v = normalize(cross(normal, u))?;
    Some((u, v))
}

fn projected_bbox(points: &[[f64; 3]], u: [f64; 3], v: [f64; 3]) -> Option<[f64; 4]> {
    if points.is_empty() {
        return None;
    }
    let mut lo_u = f64::INFINITY;
    let mut lo_v = f64::INFINITY;
    let mut hi_u = f64::NEG_INFINITY;
    let mut hi_v = f64::NEG_INFINITY;
    for &point in points {
        let pu = dot(point, u);
        let pv = dot(point, v);
        lo_u = lo_u.min(pu);
        lo_v = lo_v.min(pv);
        hi_u = hi_u.max(pu);
        hi_v = hi_v.max(pv);
    }
    [lo_u, lo_v, hi_u, hi_v]
        .iter()
        .all(|value| value.is_finite())
        .then_some([lo_u, lo_v, hi_u, hi_v])
}

fn patch_adjacencies(
    plane_pairs: &[SheetPlanePair],
    cylinder_pairs: &[SheetCylinderPair],
    face_edges: &HashMap<u64, Vec<u64>>,
) -> Vec<SheetPatchAdjacency> {
    let mut patches = Vec::<(SheetPatchId, [u64; 2])>::new();
    for (index, pair) in plane_pairs.iter().enumerate() {
        patches.push((
            SheetPatchId::Plane(index),
            [pair.negative_face_id, pair.positive_face_id],
        ));
    }
    for (index, pair) in cylinder_pairs.iter().enumerate() {
        patches.push((
            SheetPatchId::Cylinder(index),
            [pair.inner_face_id, pair.outer_face_id],
        ));
    }

    let mut out = Vec::new();
    for first in 0..patches.len() {
        for second in first + 1..patches.len() {
            let (patch_a, faces_a) = patches[first];
            let (patch_b, faces_b) = patches[second];

            let straight0 = shared_edges(faces_a[0], faces_b[0], face_edges);
            let straight1 = shared_edges(faces_a[1], faces_b[1], face_edges);
            let crossed0 = shared_edges(faces_a[0], faces_b[1], face_edges);
            let crossed1 = shared_edges(faces_a[1], faces_b[0], face_edges);

            let straight_score =
                usize::from(!straight0.is_empty()) + usize::from(!straight1.is_empty());
            let crossed_score =
                usize::from(!crossed0.is_empty()) + usize::from(!crossed1.is_empty());
            let (crossed_source_sides, mut source_edge_ids) =
                match straight_score.cmp(&crossed_score) {
                    Ordering::Greater if straight_score == 2 => {
                        let mut edges = straight0;
                        edges.extend(straight1);
                        (false, edges)
                    }
                    Ordering::Less if crossed_score == 2 => {
                        let mut edges = crossed0;
                        edges.extend(crossed1);
                        (true, edges)
                    }
                    Ordering::Equal if straight_score == 2 => {
                        let mut straight_edges = straight0;
                        straight_edges.extend(straight1);
                        let mut crossed_edges = crossed0;
                        crossed_edges.extend(crossed1);
                        if crossed_edges.len() > straight_edges.len() {
                            (true, crossed_edges)
                        } else {
                            (false, straight_edges)
                        }
                    }
                    _ => continue,
                };
            source_edge_ids.sort_unstable();
            source_edge_ids.dedup();
            out.push(SheetPatchAdjacency {
                patch_a,
                patch_b,
                crossed_source_sides,
                source_edge_ids,
            });
        }
    }
    out
}

fn graph_connected(
    plane_count: usize,
    cylinder_count: usize,
    adjacency: &[SheetPatchAdjacency],
) -> bool {
    let patch_count = plane_count + cylinder_count;
    if patch_count == 0 {
        return false;
    }
    let mut graph = vec![Vec::<usize>::new(); patch_count];
    for edge in adjacency {
        let a = patch_linear_index(edge.patch_a, plane_count);
        let b = patch_linear_index(edge.patch_b, plane_count);
        if a >= patch_count || b >= patch_count {
            return false;
        }
        graph[a].push(b);
        graph[b].push(a);
    }

    let mut seen = vec![false; patch_count];
    let mut queue = VecDeque::from([0usize]);
    seen[0] = true;
    while let Some(node) = queue.pop_front() {
        for &neighbor in &graph[node] {
            if !seen[neighbor] {
                seen[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }
    seen.into_iter().all(|value| value)
}

fn patch_linear_index(patch: SheetPatchId, plane_count: usize) -> usize {
    match patch {
        SheetPatchId::Plane(index) => index,
        SheetPatchId::Cylinder(index) => plane_count + index,
    }
}

fn recover_reference_skin(
    plane_pairs: &[SheetPlanePair],
    cylinder_pairs: &[SheetCylinderPair],
    adjacency: &[SheetPatchAdjacency],
    face_edges: &HashMap<u64, Vec<u64>>,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<SheetReferenceSkin> {
    let first = select_skin_faces(plane_pairs, cylinder_pairs, adjacency)?;
    let second = complementary_skin(&first, plane_pairs, cylinder_pairs)?;

    [first, second]
        .into_iter()
        .filter_map(|skin| {
            build_reference_skin(
                &skin,
                plane_pairs,
                cylinder_pairs,
                face_edges,
                entities,
                index,
            )
        })
        .min_by_key(|skin| {
            (
                skin.unique_edges,
                skin.boundary_loops
                    .iter()
                    .map(|loop_| loop_.curves.len())
                    .sum::<usize>(),
                skin.internal_seams.len(),
            )
        })
}

fn complementary_skin(
    skin: &[(SheetPatchId, usize, u64)],
    plane_pairs: &[SheetPlanePair],
    cylinder_pairs: &[SheetCylinderPair],
) -> Option<Vec<(SheetPatchId, usize, u64)>> {
    skin.iter()
        .map(|&(patch, side, _)| {
            if side > 1 {
                return None;
            }
            let other_side = 1 - side;
            let face = match patch {
                SheetPatchId::Plane(index) => {
                    let pair = plane_pairs.get(index)?;
                    [pair.negative_face_id, pair.positive_face_id][other_side]
                }
                SheetPatchId::Cylinder(index) => {
                    let pair = cylinder_pairs.get(index)?;
                    [pair.inner_face_id, pair.outer_face_id][other_side]
                }
            };
            Some((patch, other_side, face))
        })
        .collect()
}

fn build_reference_skin(
    skin: &[(SheetPatchId, usize, u64)],
    plane_pairs: &[SheetPlanePair],
    cylinder_pairs: &[SheetCylinderPair],
    face_edges: &HashMap<u64, Vec<u64>>,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<SheetReferenceSkin> {
    let mut face_to_patch = HashMap::<u64, (SheetPatchId, usize)>::new();
    let mut patch_side = HashMap::<SheetPatchId, usize>::new();
    let mut patches = Vec::with_capacity(skin.len());
    let mut face_ids = Vec::with_capacity(skin.len());

    for &(patch, side, face_id) in skin {
        if side > 1
            || face_to_patch.insert(face_id, (patch, side)).is_some()
            || patch_side.insert(patch, side).is_some()
        {
            return None;
        }
        face_ids.push(face_id);
        let (paired_face_id, surface) = match patch {
            SheetPatchId::Plane(patch_index) => {
                let pair = plane_pairs.get(patch_index)?;
                let offsets = [
                    pair.mid_offset_mm - pair.separation_mm * 0.5,
                    pair.mid_offset_mm + pair.separation_mm * 0.5,
                ];
                let faces = [pair.negative_face_id, pair.positive_face_id];
                (
                    faces[1 - side],
                    SheetReferenceSurface::Plane {
                        normal: pair.normal,
                        offset_mm: offsets[side],
                        paired_offset_mm: offsets[1 - side],
                    },
                )
            }
            SheetPatchId::Cylinder(patch_index) => {
                let pair = cylinder_pairs.get(patch_index)?;
                let faces = [pair.inner_face_id, pair.outer_face_id];
                let radii = [pair.inner_radius_mm, pair.outer_radius_mm];
                (
                    faces[1 - side],
                    SheetReferenceSurface::Cylinder {
                        axis_origin_mm: pair.axis_origin_mm,
                        axis: pair.axis,
                        radius_mm: radii[side],
                        paired_radius_mm: radii[1 - side],
                    },
                )
            }
        };
        patches.push(SheetReferencePatch {
            patch,
            source_face_id: face_id,
            paired_face_id,
            surface,
        });
    }

    let mut edge_patches = HashMap::<u64, Vec<SheetPatchId>>::new();
    for &(patch, _, face_id) in skin {
        for &edge in face_edges.get(&face_id)? {
            edge_patches.entry(edge).or_default().push(patch);
        }
    }
    if edge_patches
        .values()
        .any(|patches| patches.is_empty() || patches.len() > 2)
    {
        return None;
    }

    let boundary_edges = edge_patches
        .iter()
        .filter_map(|(&edge, patches)| (patches.len() == 1).then_some(edge))
        .collect::<BTreeSet<_>>();
    let internal_edges = edge_patches
        .iter()
        .filter_map(|(&edge, patches)| (patches.len() == 2).then_some(edge))
        .collect::<BTreeSet<_>>();
    if boundary_edges.is_empty() {
        return None;
    }

    let mut vertex_edges = HashMap::<u64, Vec<u64>>::new();
    for &edge in &boundary_edges {
        let (start, end, _, _) = edge_curve_info(edge, entities, index)?;
        vertex_edges.entry(start).or_default().push(edge);
        vertex_edges.entry(end).or_default().push(edge);
    }
    if vertex_edges.values().any(|edges| edges.len() != 2) {
        return None;
    }

    let mut unused = boundary_edges.clone();
    let mut boundary_loops = Vec::new();
    while let Some(&first_edge) = unused.iter().next() {
        let (first_start, first_end, _, _) = edge_curve_info(first_edge, entities, index)?;
        let start_vertex = first_start.min(first_end);
        let mut current_vertex = start_vertex;
        let mut current_edge = first_edge;
        let mut curves = Vec::new();

        loop {
            if !unused.remove(&current_edge) {
                return None;
            }
            let (edge_start, edge_end, _, _) = edge_curve_info(current_edge, entities, index)?;
            let next_vertex = if edge_start == current_vertex {
                edge_end
            } else if edge_end == current_vertex {
                edge_start
            } else {
                return None;
            };
            curves.push(reference_curve(
                current_edge,
                current_vertex,
                next_vertex,
                edge_patches.get(&current_edge)?,
                &patch_side,
                cylinder_pairs,
                entities,
                index,
            )?);

            current_vertex = next_vertex;
            if current_vertex == start_vertex {
                break;
            }
            let next = vertex_edges
                .get(&current_vertex)?
                .iter()
                .copied()
                .filter(|edge| unused.contains(edge))
                .collect::<Vec<_>>();
            if next.len() != 1 {
                return None;
            }
            current_edge = next[0];
        }
        boundary_loops.push(SheetReferenceLoop { curves });
    }
    boundary_loops.sort_by_key(|loop_| std::cmp::Reverse(loop_.curves.len()));

    let mut internal_seams = Vec::with_capacity(internal_edges.len());
    for edge in internal_edges {
        let (start, end, _, _) = edge_curve_info(edge, entities, index)?;
        internal_seams.push(reference_curve(
            edge,
            start,
            end,
            edge_patches.get(&edge)?,
            &patch_side,
            cylinder_pairs,
            entities,
            index,
        )?);
    }
    internal_seams.sort_by_key(reference_curve_source_edge);

    Some(SheetReferenceSkin {
        face_ids,
        patches,
        unique_edges: edge_patches.len(),
        boundary_loops,
        internal_seams,
    })
}

fn reference_curve_source_edge(curve: &SheetReferenceCurve) -> u64 {
    match curve {
        SheetReferenceCurve::Line { source_edge_id, .. }
        | SheetReferenceCurve::CircleArc { source_edge_id, .. }
        | SheetReferenceCurve::EllipseArc { source_edge_id, .. } => *source_edge_id,
    }
}

fn reference_curve(
    edge_id: u64,
    traversal_start: u64,
    traversal_end: u64,
    patches: &[SheetPatchId],
    patch_side: &HashMap<SheetPatchId, usize>,
    cylinder_pairs: &[SheetCylinderPair],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<SheetReferenceCurve> {
    let (edge_start, edge_end, curve_id, same_sense) = edge_curve_info(edge_id, entities, index)?;
    let start_mm = vertex_point(traversal_start, entities, index)?;
    let end_mm = vertex_point(traversal_end, entities, index)?;
    let traversal_forward = traversal_start == edge_start && traversal_end == edge_end;
    let traversal_reverse = traversal_start == edge_end && traversal_end == edge_start;
    if !traversal_forward && !traversal_reverse {
        return None;
    }
    let parameter_forward = if traversal_forward {
        same_sense
    } else {
        !same_sense
    };

    match entities.get(*index.get(&curve_id)?)? {
        EntityInstance::Simple { record, .. } if record.name == "LINE" => {
            Some(SheetReferenceCurve::Line {
                source_edge_id: edge_id,
                patches: patches.to_vec(),
                start_mm,
                end_mm,
            })
        }
        EntityInstance::Simple { record, .. } if record.name == "CIRCLE" => {
            let (center_mm, normal, x_direction, radius_mm) =
                circle_support(curve_id, entities, index)?;
            let y_direction = normalize(cross(normal, x_direction))?;
            if (distance(start_mm, center_mm) - radius_mm).abs() > GEOM_TOL_MM * 5.0
                || (distance(end_mm, center_mm) - radius_mm).abs() > GEOM_TOL_MM * 5.0
            {
                return None;
            }
            let start_angle = circle_angle(start_mm, center_mm, x_direction, y_direction)?;
            let end_angle = circle_angle(end_mm, center_mm, x_direction, y_direction)?;
            Some(SheetReferenceCurve::CircleArc {
                source_edge_id: edge_id,
                patches: patches.to_vec(),
                center_mm,
                normal,
                radius_mm,
                start_mm,
                end_mm,
                sweep_angle_rad: circle_sweep(start_angle, end_angle, parameter_forward),
            })
        }
        EntityInstance::Complex { .. } => reference_ellipse_seam(
            edge_id,
            curve_id,
            traversal_start,
            traversal_end,
            patches,
            patch_side,
            cylinder_pairs,
            entities,
            index,
        ),
        _ => None,
    }
}

fn reference_ellipse_seam(
    edge_id: u64,
    curve_id: u64,
    traversal_start: u64,
    traversal_end: u64,
    patches: &[SheetPatchId],
    patch_side: &HashMap<SheetPatchId, usize>,
    cylinder_pairs: &[SheetCylinderPair],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<SheetReferenceCurve> {
    if patches.len() != 2 {
        return None;
    }
    let SheetPatchId::Cylinder(first_index) = patches[0] else {
        return None;
    };
    let SheetPatchId::Cylinder(second_index) = patches[1] else {
        return None;
    };
    let first_pair = cylinder_pairs.get(first_index)?;
    let second_pair = cylinder_pairs.get(second_index)?;
    let first_side = *patch_side.get(&patches[0])?;
    let second_side = *patch_side.get(&patches[1])?;
    let first_radius = [first_pair.inner_radius_mm, first_pair.outer_radius_mm][first_side];
    let second_radius = [second_pair.inner_radius_mm, second_pair.outer_radius_mm][second_side];
    if (first_radius - second_radius).abs() > GEOM_TOL_MM {
        return None;
    }
    let radius_mm = (first_radius + second_radius) * 0.5;
    let first_axis = first_pair.axis;
    let second_axis = second_pair.axis;
    if dot(first_axis, second_axis).abs() > DIR_TOL * 100.0 {
        return None;
    }
    let (center_mm, axis_gap) = closest_axis_intersection(
        first_pair.axis_origin_mm,
        first_axis,
        second_pair.axis_origin_mm,
        second_axis,
    )?;
    if axis_gap > GEOM_TOL_MM {
        return None;
    }

    let start_mm = vertex_point(traversal_start, entities, index)?;
    let end_mm = vertex_point(traversal_end, entities, index)?;
    let first_coordinate = dot(sub(start_mm, center_mm), first_axis);
    let second_coordinate = dot(sub(start_mm, center_mm), second_axis);
    if first_coordinate.abs() <= GEOM_TOL_MM || second_coordinate.abs() <= GEOM_TOL_MM {
        return None;
    }
    let branch_sign = if first_coordinate * second_coordinate >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let u_axis_mm = mul(add(mul(first_axis, branch_sign), second_axis), radius_mm);
    let v_axis_mm = mul(normalize(cross(first_axis, second_axis))?, radius_mm);

    let start_angle = ellipse_angle(start_mm, center_mm, u_axis_mm, v_axis_mm)?;
    let end_angle = ellipse_angle(end_mm, center_mm, u_axis_mm, v_axis_mm)?;
    if distance(
        ellipse_point(center_mm, u_axis_mm, v_axis_mm, start_angle),
        start_mm,
    ) > GEOM_TOL_MM * 5.0
        || distance(
            ellipse_point(center_mm, u_axis_mm, v_axis_mm, end_angle),
            end_mm,
        ) > GEOM_TOL_MM * 5.0
    {
        return None;
    }

    let samples = rational_single_span_samples(curve_id, entities, index)?;
    for sample in &samples {
        if (axis_line_distance(first_pair.axis_origin_mm, first_axis, *sample) - first_radius).abs()
            > GEOM_TOL_MM * 5.0
            || (axis_line_distance(second_pair.axis_origin_mm, second_axis, *sample)
                - second_radius)
                .abs()
                > GEOM_TOL_MM * 5.0
        {
            return None;
        }
        let u = dot(sub(*sample, center_mm), first_axis);
        let v = dot(sub(*sample, center_mm), second_axis);
        if (u - branch_sign * v).abs() > GEOM_TOL_MM * 5.0 {
            return None;
        }
    }

    let midpoint = samples
        .get(1)
        .copied()
        .or_else(|| samples.first().copied())?;
    let midpoint_angle = ellipse_angle(midpoint, center_mm, u_axis_mm, v_axis_mm)?;
    let sweep_angle_rad = ellipse_arc_sweep(start_angle, end_angle, midpoint_angle)?;

    Some(SheetReferenceCurve::EllipseArc {
        source_edge_id: edge_id,
        patches: patches.to_vec(),
        center_mm,
        u_axis_mm,
        v_axis_mm,
        start_angle_rad: start_angle,
        sweep_angle_rad,
        start_mm,
        end_mm,
    })
}

fn closest_axis_intersection(
    first_origin: [f64; 3],
    first_axis: [f64; 3],
    second_origin: [f64; 3],
    second_axis: [f64; 3],
) -> Option<([f64; 3], f64)> {
    let b = dot(first_axis, second_axis);
    let denominator = 1.0 - b * b;
    if denominator.abs() <= DIR_TOL {
        return None;
    }
    let delta = sub(first_origin, second_origin);
    let d = dot(first_axis, delta);
    let e = dot(second_axis, delta);
    let first_t = (b * e - d) / denominator;
    let second_t = (e - b * d) / denominator;
    let first_point = add(first_origin, mul(first_axis, first_t));
    let second_point = add(second_origin, mul(second_axis, second_t));
    Some((
        mul(add(first_point, second_point), 0.5),
        distance(first_point, second_point),
    ))
}

fn ellipse_point(center: [f64; 3], u_axis: [f64; 3], v_axis: [f64; 3], angle: f64) -> [f64; 3] {
    add(
        center,
        add(mul(u_axis, angle.cos()), mul(v_axis, angle.sin())),
    )
}

fn ellipse_angle(
    point: [f64; 3],
    center: [f64; 3],
    u_axis: [f64; 3],
    v_axis: [f64; 3],
) -> Option<f64> {
    let relative = sub(point, center);
    let u_norm_sq = dot(u_axis, u_axis);
    let v_norm_sq = dot(v_axis, v_axis);
    if u_norm_sq <= DIR_TOL || v_norm_sq <= DIR_TOL {
        return None;
    }
    let cosine = dot(relative, u_axis) / u_norm_sq;
    let sine = dot(relative, v_axis) / v_norm_sq;
    Some(sine.atan2(cosine))
}

fn ellipse_arc_sweep(start: f64, end: f64, sample: f64) -> Option<f64> {
    let tau = std::f64::consts::TAU;
    let positive = (end - start).rem_euclid(tau);
    let negative = positive - tau;
    let positive_progress = (sample - start).rem_euclid(tau);
    let negative_progress = -((start - sample).rem_euclid(tau));
    let on_positive = positive_progress <= positive + 1.0e-8;
    let on_negative = negative_progress >= negative - 1.0e-8;
    match (on_positive, on_negative) {
        (true, false) => Some(positive),
        (false, true) => Some(negative),
        (true, true) => {
            if positive.abs() <= negative.abs() {
                Some(positive)
            } else {
                Some(negative)
            }
        }
        (false, false) => None,
    }
}

fn rational_single_span_samples(
    curve_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<[f64; 3]>> {
    let EntityInstance::Complex { subsuper, .. } = entities.get(*index.get(&curve_id)?)? else {
        return None;
    };
    let records = &subsuper.0;
    let record = |name: &str| {
        let mut matches = records.iter().filter(|record| record.name == name);
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    };
    let bspline = record("B_SPLINE_CURVE")?;
    let bspline_params = list_params(bspline)?;
    let degree = match bspline_params.first()? {
        Parameter::Integer(value) if *value >= 1 => *value as usize,
        _ => return None,
    };
    let Parameter::List(pole_params) = bspline_params.get(1)? else {
        return None;
    };
    if pole_params.len() != degree + 1 {
        return None;
    }
    let poles = pole_params
        .iter()
        .map(|parameter| {
            let point_id = entity_ref_value(parameter)?;
            cartesian_point(point_id, entities, index)
        })
        .collect::<Option<Vec<_>>>()?;

    let knots = record("B_SPLINE_CURVE_WITH_KNOTS")?;
    let knot_params = list_params(knots)?;
    let Parameter::List(multiplicities) = knot_params.first()? else {
        return None;
    };
    if multiplicities.len() != 2
        || !multiplicities.iter().all(
            |parameter| matches!(parameter, Parameter::Integer(value) if *value == degree as i64 + 1),
        )
    {
        return None;
    }

    let rational = record("RATIONAL_B_SPLINE_CURVE")?;
    let rational_params = list_params(rational)?;
    let Parameter::List(weight_params) = rational_params.first()? else {
        return None;
    };
    if weight_params.len() != poles.len() {
        return None;
    }
    let weights = weight_params
        .iter()
        .map(|parameter| numeric_value(parameter).filter(|value| value.is_finite() && *value > 0.0))
        .collect::<Option<Vec<_>>>()?;

    [0.25, 0.5, 0.75]
        .into_iter()
        .map(|parameter| rational_bezier_point(&poles, &weights, parameter))
        .collect()
}

fn rational_bezier_point(poles: &[[f64; 3]], weights: &[f64], parameter: f64) -> Option<[f64; 3]> {
    if poles.len() != weights.len() || poles.is_empty() {
        return None;
    }
    let degree = poles.len() - 1;
    let mut numerator = [0.0; 3];
    let mut denominator = 0.0;
    for (index, (&pole, &weight)) in poles.iter().zip(weights).enumerate() {
        let bernstein = binomial(degree, index) as f64
            * parameter.powi(index as i32)
            * (1.0 - parameter).powi((degree - index) as i32);
        let weighted = bernstein * weight;
        denominator += weighted;
        for axis in 0..3 {
            numerator[axis] += pole[axis] * weighted;
        }
    }
    if denominator.abs() <= 1.0e-15 {
        return None;
    }
    Some([
        numerator[0] / denominator,
        numerator[1] / denominator,
        numerator[2] / denominator,
    ])
}

fn binomial(n: usize, k: usize) -> usize {
    let k = k.min(n - k);
    (0..k).fold(1usize, |accumulator, index| {
        accumulator * (n - index) / (index + 1)
    })
}

fn select_skin_faces(
    plane_pairs: &[SheetPlanePair],
    cylinder_pairs: &[SheetCylinderPair],
    adjacency: &[SheetPatchAdjacency],
) -> Option<Vec<(SheetPatchId, usize, u64)>> {
    let plane_count = plane_pairs.len();
    let patch_count = plane_count + cylinder_pairs.len();
    if patch_count == 0 {
        return None;
    }
    let mut graph = vec![Vec::<(usize, bool)>::new(); patch_count];
    for edge in adjacency {
        let a = patch_linear_index(edge.patch_a, plane_count);
        let b = patch_linear_index(edge.patch_b, plane_count);
        if a >= patch_count || b >= patch_count {
            return None;
        }
        graph[a].push((b, edge.crossed_source_sides));
        graph[b].push((a, edge.crossed_source_sides));
    }

    let mut side = vec![None::<usize>; patch_count];
    side[0] = Some(0);
    let mut queue = VecDeque::from([0usize]);
    while let Some(patch) = queue.pop_front() {
        let current = side[patch]?;
        for &(neighbor, crossed) in &graph[patch] {
            let expected = current ^ usize::from(crossed);
            match side[neighbor] {
                Some(existing) if existing != expected => return None,
                Some(_) => {}
                None => {
                    side[neighbor] = Some(expected);
                    queue.push_back(neighbor);
                }
            }
        }
    }
    if side.iter().any(Option::is_none) {
        return None;
    }

    let mut out = Vec::with_capacity(patch_count);
    for (index, selected) in side.into_iter().enumerate() {
        let selected = selected?;
        if index < plane_count {
            let pair = &plane_pairs[index];
            let faces = [pair.negative_face_id, pair.positive_face_id];
            out.push((SheetPatchId::Plane(index), selected, faces[selected]));
        } else {
            let cylinder_index = index - plane_count;
            let pair = &cylinder_pairs[cylinder_index];
            let faces = [pair.inner_face_id, pair.outer_face_id];
            out.push((
                SheetPatchId::Cylinder(cylinder_index),
                selected,
                faces[selected],
            ));
        }
    }
    Some(out)
}

fn edge_curve_info(
    edge_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<(u64, u64, u64, bool)> {
    let edge = simple_record(entities.get(*index.get(&edge_id)?)?)?;
    if edge.name != "EDGE_CURVE" {
        return None;
    }
    let params = list_params(edge)?;
    let start = params.get(1).and_then(entity_ref_value)?;
    let end = params.get(2).and_then(entity_ref_value)?;
    let curve = params.get(3).and_then(entity_ref_value)?;
    let same_sense = enumeration_bool(params.get(4)?)?;
    Some((start, end, curve, same_sense))
}

fn vertex_point(
    vertex_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let vertex = simple_record(entities.get(*index.get(&vertex_id)?)?)?;
    if vertex.name != "VERTEX_POINT" {
        return None;
    }
    let params = list_params(vertex)?;
    let point_id = params.get(1).and_then(entity_ref_value)?;
    cartesian_point(point_id, entities, index)
}

fn circle_support(
    curve_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<([f64; 3], [f64; 3], [f64; 3], f64)> {
    let circle = simple_record(entities.get(*index.get(&curve_id)?)?)?;
    if circle.name != "CIRCLE" {
        return None;
    }
    let params = list_params(circle)?;
    let placement_id = params.get(1).and_then(entity_ref_value)?;
    let radius_mm = numeric_value(params.get(2)?)?;
    let (center, normal, x_direction) = placement_frame(placement_id, entities, index)?;
    Some((center, normal, x_direction, radius_mm))
}

fn placement_frame(
    placement_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<([f64; 3], [f64; 3], [f64; 3])> {
    let placement = simple_record(entities.get(*index.get(&placement_id)?)?)?;
    if placement.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let params = list_params(placement)?;
    let origin_id = params.get(1).and_then(entity_ref_value)?;
    let axis_id = params.get(2).and_then(entity_ref_value)?;
    let reference_id = params.get(3).and_then(entity_ref_value)?;
    let origin = cartesian_point(origin_id, entities, index)?;
    let normal = normalize(direction(axis_id, entities, index)?)?;
    let reference = direction(reference_id, entities, index)?;
    let projected = sub(reference, mul(normal, dot(reference, normal)));
    let x_direction = normalize(projected)?;
    Some((origin, normal, x_direction))
}

fn circle_angle(
    point: [f64; 3],
    center: [f64; 3],
    x_direction: [f64; 3],
    y_direction: [f64; 3],
) -> Option<f64> {
    let radial = sub(point, center);
    if norm(radial) <= GEOM_TOL_MM * 0.01 {
        return None;
    }
    Some(dot(radial, y_direction).atan2(dot(radial, x_direction)))
}

fn circle_sweep(start: f64, end: f64, forward: bool) -> f64 {
    let tau = std::f64::consts::TAU;
    if forward {
        (end - start).rem_euclid(tau)
    } else {
        -((start - end).rem_euclid(tau))
    }
}

fn enumeration_bool(parameter: &Parameter) -> Option<bool> {
    match parameter {
        Parameter::Enumeration(value) if value == "T" => Some(true),
        Parameter::Enumeration(value) if value == "F" => Some(false),
        _ => None,
    }
}

fn shared_edges(
    first_face: u64,
    second_face: u64,
    face_edges: &HashMap<u64, Vec<u64>>,
) -> Vec<u64> {
    let Some(first) = face_edges.get(&first_face) else {
        return Vec::new();
    };
    let Some(second) = face_edges.get(&second_face) else {
        return Vec::new();
    };
    let second = second.iter().copied().collect::<HashSet<_>>();
    first
        .iter()
        .copied()
        .filter(|edge| second.contains(edge))
        .collect()
}

fn face_edge_map(
    face_ids: &[u64],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> HashMap<u64, Vec<u64>> {
    face_ids
        .iter()
        .filter_map(|&face_id| face_edges(face_id, entities, index).map(|edges| (face_id, edges)))
        .collect()
}

fn solid_faces(
    solid_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<u64>> {
    let solid = simple_record(entities.get(*index.get(&solid_id)?)?)?;
    let params = list_params(solid)?;
    let shell_id = params.get(1).and_then(entity_ref_value)?;
    let shell = simple_record(entities.get(*index.get(&shell_id)?)?)?;
    if shell.name != "CLOSED_SHELL" && shell.name != "OPEN_SHELL" {
        return None;
    }
    let shell_params = list_params(shell)?;
    let Parameter::List(items) = shell_params.get(1)? else {
        return None;
    };
    items.iter().map(entity_ref_value).collect()
}

fn face_surface<'a>(
    face_id: u64,
    entities: &'a [EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<(u64, &'a Record)> {
    let face = simple_record(entities.get(*index.get(&face_id)?)?)?;
    if face.name != "ADVANCED_FACE" {
        return None;
    }
    let params = list_params(face)?;
    let surface_id = params.get(2).and_then(entity_ref_value)?;
    let surface = simple_record(entities.get(*index.get(&surface_id)?)?)?;
    Some((surface_id, surface))
}

fn face_edges(
    face_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<u64>> {
    let face = simple_record(entities.get(*index.get(&face_id)?)?)?;
    if face.name != "ADVANCED_FACE" {
        return None;
    }
    let params = list_params(face)?;
    let Parameter::List(bounds) = params.get(1)? else {
        return None;
    };
    let mut out = Vec::new();
    for bound_param in bounds {
        let bound_id = entity_ref_value(bound_param)?;
        let bound = simple_record(entities.get(*index.get(&bound_id)?)?)?;
        if bound.name != "FACE_OUTER_BOUND" && bound.name != "FACE_BOUND" {
            return None;
        }
        let bound_params = list_params(bound)?;
        let loop_id = bound_params.get(1).and_then(entity_ref_value)?;
        let loop_record = simple_record(entities.get(*index.get(&loop_id)?)?)?;
        if loop_record.name != "EDGE_LOOP" {
            return None;
        }
        let loop_params = list_params(loop_record)?;
        let Parameter::List(oriented_edges) = loop_params.get(1)? else {
            return None;
        };
        for edge_param in oriented_edges {
            let oriented_id = entity_ref_value(edge_param)?;
            let oriented = simple_record(entities.get(*index.get(&oriented_id)?)?)?;
            if oriented.name != "ORIENTED_EDGE" {
                return None;
            }
            let oriented_params = list_params(oriented)?;
            out.push(oriented_params.get(3).and_then(entity_ref_value)?);
        }
    }
    Some(out)
}

fn face_vertices(
    face_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<[f64; 3]>> {
    let mut out = Vec::<[f64; 3]>::new();
    for edge_id in face_edges(face_id, entities, index)? {
        let edge = simple_record(entities.get(*index.get(&edge_id)?)?)?;
        if edge.name != "EDGE_CURVE" {
            return None;
        }
        let params = list_params(edge)?;
        for vertex_param in [params.get(1)?, params.get(2)?] {
            let vertex_id = entity_ref_value(vertex_param)?;
            let vertex = simple_record(entities.get(*index.get(&vertex_id)?)?)?;
            if vertex.name != "VERTEX_POINT" {
                return None;
            }
            let vertex_params = list_params(vertex)?;
            let point_id = vertex_params.get(1).and_then(entity_ref_value)?;
            let point = cartesian_point(point_id, entities, index)?;
            if !out
                .iter()
                .any(|&existing| distance(existing, point) <= GEOM_TOL_MM * 0.1)
            {
                out.push(point);
            }
        }
    }
    Some(out)
}

fn surface_axis(
    surface_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<([f64; 3], [f64; 3])> {
    let surface = simple_record(entities.get(*index.get(&surface_id)?)?)?;
    let params = list_params(surface)?;
    let placement_id = params.get(1).and_then(entity_ref_value)?;
    let placement = simple_record(entities.get(*index.get(&placement_id)?)?)?;
    if placement.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let placement_params = list_params(placement)?;
    let origin_id = placement_params.get(1).and_then(entity_ref_value)?;
    let axis_id = placement_params.get(2).and_then(entity_ref_value)?;
    let origin = cartesian_point(origin_id, entities, index)?;
    let axis = direction(axis_id, entities, index)?;
    Some((origin, normalize(axis)?))
}

fn surface_radius(surface: &Record) -> Option<f64> {
    let params = list_params(surface)?;
    numeric_value(params.get(2)?)
}

fn cartesian_point(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let record = simple_record(entities.get(*index.get(&id)?)?)?;
    if record.name != "CARTESIAN_POINT" {
        return None;
    }
    let params = list_params(record)?;
    let Parameter::List(coords) = params.get(1)? else {
        return None;
    };
    if coords.len() != 3 {
        return None;
    }
    Some([
        numeric_value(&coords[0])?,
        numeric_value(&coords[1])?,
        numeric_value(&coords[2])?,
    ])
}

fn direction(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let record = simple_record(entities.get(*index.get(&id)?)?)?;
    if record.name != "DIRECTION" {
        return None;
    }
    let params = list_params(record)?;
    let Parameter::List(coords) = params.get(1)? else {
        return None;
    };
    if coords.len() != 3 {
        return None;
    }
    Some([
        numeric_value(&coords[0])?,
        numeric_value(&coords[1])?,
        numeric_value(&coords[2])?,
    ])
}

fn parallel(a: [f64; 3], b: [f64; 3]) -> bool {
    (dot(a, b).abs() - 1.0).abs() <= DIR_TOL
}

fn axis_line_distance(origin: [f64; 3], axis: [f64; 3], other: [f64; 3]) -> f64 {
    norm(cross(sub(other, origin), axis))
}

fn canonical_axis(mut axis: [f64; 3]) -> [f64; 3] {
    for value in axis {
        if value.abs() > DIR_TOL {
            if value < 0.0 {
                axis = [-axis[0], -axis[1], -axis[2]];
            }
            break;
        }
    }
    axis
}

fn canonical_line_origin(origin: [f64; 3], axis: [f64; 3]) -> [f64; 3] {
    sub(origin, mul(axis, dot(origin, axis)))
}

fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    normalize3(vector, DIR_TOL)
}

fn quantize_mm(value: f64) -> i64 {
    (value / GEOM_TOL_MM).round() as i64
}

const fn list_params(record: &Record) -> Option<&[Parameter]> {
    match &record.parameter {
        Parameter::List(params) => Some(params.as_slice()),
        _ => None,
    }
}

fn numeric_value(parameter: &Parameter) -> Option<f64> {
    match parameter {
        Parameter::Real(value) => Some(*value),
        Parameter::Integer(value) => crate::numeric::exact_i64_to_f64(*value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(face_id: u64, x: f64, normal: [f64; 3], vertices: Vec<[f64; 3]>) -> PlaneFace {
        PlaneFace {
            face_id,
            origin: [x, 0.0, 0.0],
            normal,
            vertices,
        }
    }

    #[test]
    fn coaxial_radius_pairs_recover_wall_thickness() {
        let cylinders = vec![
            CylinderFace {
                face_id: 1,
                origin: [0.0, 0.0, 0.0],
                axis: [0.0, 0.0, 1.0],
                radius_mm: 0.4,
            },
            CylinderFace {
                face_id: 2,
                origin: [0.0, 0.0, 8.0],
                axis: [0.0, 0.0, -1.0],
                radius_mm: 0.6,
            },
            CylinderFace {
                face_id: 3,
                origin: [5.0, 0.0, 0.0],
                axis: [0.0, 0.0, 1.0],
                radius_mm: 0.9,
            },
        ];
        let pairs = coaxial_radius_differences(&cylinders);
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].outer_radius_mm - pairs[0].inner_radius_mm - 0.2).abs() < 1.0e-12);
        assert!((pairs[0].mid_radius_mm - 0.5).abs() < 1.0e-12);
        assert_eq!(pairs[0].axis_origin_mm, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn plane_separation_is_orientation_invariant() {
        let planes = vec![
            plane(
                1,
                0.0,
                [1.0, 0.0, 0.0],
                vec![[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 1.0]],
            ),
            plane(
                2,
                0.2,
                [-1.0, 0.0, 0.0],
                vec![[0.2, 0.0, 0.0], [0.2, 2.0, 0.0], [0.2, 2.0, 1.0]],
            ),
        ];
        let separations = parallel_plane_separations(&planes);
        assert_eq!(separations.len(), 1);
        assert!((separations[0] - 0.2).abs() < 1.0e-12);
    }

    #[test]
    fn plane_pairing_rejects_disjoint_same_thickness_faces() {
        let rectangle =
            |x: f64, y0: f64, y1: f64| vec![[x, y0, 0.0], [x, y1, 0.0], [x, y1, 1.0], [x, y0, 1.0]];
        let planes = vec![
            plane(1, 0.0, [1.0, 0.0, 0.0], rectangle(0.0, 0.0, 4.0)),
            plane(2, 0.2, [-1.0, 0.0, 0.0], rectangle(0.2, 0.2, 3.8)),
            plane(3, 0.2, [-1.0, 0.0, 0.0], rectangle(0.2, 10.0, 12.0)),
        ];
        let pairs = select_plane_pairs(&planes, 0.2);
        assert_eq!(pairs.len(), 1);
        assert_eq!(
            (pairs[0].negative_face_id, pairs[0].positive_face_id),
            (1, 2)
        );
        assert!(pairs[0].projected_overlap_ratio > 0.9);
    }
}
