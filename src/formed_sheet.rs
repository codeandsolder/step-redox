use crate::math3::{
    canonical_direction, closest_point_on_unit_line_to_origin, cross, distance, dot,
    normalize as normalize3, point_to_unit_line_distance,
};
use crate::step_entities::{
    cartesian_point, direction_components as direction, number as numeric_value,
};
use crate::step_graph::{build_index, entity_id, entity_ref_value, simple_record};
use ruststep::ast::{EntityInstance, Parameter, Record};
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

mod reference_skin;
use reference_skin::recover_reference_skin;
pub use reference_skin::{
    SheetReferenceCurve, SheetReferenceLoop, SheetReferencePatch, SheetReferenceSkin,
    SheetReferenceSurface,
};
#[cfg(test)]
use reference_skin::{closest_axis_intersection, rational_bezier_point};

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
    detect_formed_sheet_evidence_with_min_pairs(
        entities,
        MIN_COAXIAL_PAIRS,
        MIN_PARALLEL_PLANE_PAIRS,
    )
}

/// Analyze formed-sheet candidates with caller-selected evidence thresholds.
///
/// This keeps the public high-confidence diagnostic policy separate from
/// constructive recovery, whose own grammar may legitimately prove a body from
/// a single bend pair plus independent topology/trim evidence.
#[must_use]
pub(crate) fn detect_formed_sheet_evidence_with_min_pairs(
    entities: &[EntityInstance],
    min_coaxial_pairs: usize,
    min_parallel_plane_pairs: usize,
) -> Vec<FormedSheetEvidence> {
    if min_coaxial_pairs == 0 || min_parallel_plane_pairs == 0 {
        return Vec::new();
    }
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

        if cylinders.len() < min_coaxial_pairs * 2 || planes.len() < 2 {
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
        if raw_cylinder_pairs < min_coaxial_pairs || raw_plane_pairs < min_parallel_plane_pairs {
            continue;
        }
        let thickness_mm = tick as f64 * GEOM_TOL_MM;

        let cylinder_pairs = select_cylinder_pairs(cylinder_candidates, thickness_mm);
        let plane_pairs = select_plane_pairs(&planes, thickness_mm);
        if cylinder_pairs.len() < min_coaxial_pairs || plane_pairs.len() < min_parallel_plane_pairs
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

/// Rebuild a reference skin from an explicitly selected set of paired supports.
///
/// Constructive recovery uses this after permissive evidence discovery to discard
/// geometrically irrelevant same-thickness pairs (for example the two width faces
/// of a square-section bent pin) before proving a feature grammar.
pub(crate) fn recover_reference_skin_for_pairs(
    solid_id: u64,
    plane_pairs: &[SheetPlanePair],
    cylinder_pairs: &[SheetCylinderPair],
    entities: &[EntityInstance],
) -> Option<SheetReferenceSkin> {
    if plane_pairs.is_empty() || cylinder_pairs.is_empty() {
        return None;
    }
    let index = build_index(entities);
    let face_ids = solid_faces(solid_id, entities, &index)?;
    let face_edges = face_edge_map(&face_ids, entities, &index);
    let adjacency = patch_adjacencies(plane_pairs, cylinder_pairs, &face_edges);
    if !graph_connected(plane_pairs.len(), cylinder_pairs.len(), &adjacency) {
        return None;
    }
    recover_reference_skin(
        plane_pairs,
        cylinder_pairs,
        &adjacency,
        &face_edges,
        entities,
        &index,
    )
}

fn coaxial_radius_differences(cylinders: &[CylinderFace]) -> Vec<SheetCylinderPair> {
    let mut out = Vec::new();
    for (index, a) in cylinders.iter().enumerate() {
        for b in &cylinders[index + 1..] {
            if !parallel(a.axis, b.axis) {
                continue;
            }
            if point_to_unit_line_distance(b.origin, a.origin, a.axis) > GEOM_TOL_MM {
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
            let axis = canonical_direction(inner.axis);
            out.push(SheetCylinderPair {
                inner_face_id: inner.face_id,
                outer_face_id: outer.face_id,
                inner_radius_mm: inner.radius_mm,
                outer_radius_mm: outer.radius_mm,
                mid_radius_mm: f64::midpoint(inner.radius_mm, outer.radius_mm),
                axis_origin_mm: closest_point_on_unit_line_to_origin(inner.origin, axis),
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
            let normal = canonical_direction(a.normal);
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
            let normal = canonical_direction(a.normal);
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

        let normal = canonical_direction(a.normal);
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
            mid_offset_mm: f64::midpoint(negative_offset, positive_offset),
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

fn parallel(a: [f64; 3], b: [f64; 3]) -> bool {
    (dot(a, b).abs() - 1.0).abs() <= DIR_TOL
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
    fn closest_axis_intersection_handles_near_parallel_axes() {
        let angle: f64 = 2.0e-5;
        let second_axis = [angle.cos(), -angle.sin(), 0.0];
        let (center, gap) = closest_axis_intersection(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            second_axis,
        )
        .expect("test lines should have a well-defined closest approach");
        let expected_x = 1.0 / angle.tan();
        assert!(gap < 1.0e-9);
        assert!((center[0] - expected_x).abs() < 1.0e-5);
        assert!(center[1].abs() < 1.0e-9);
    }

    #[test]
    fn rational_bezier_de_casteljau_handles_extreme_weights() {
        let scale = 1.0e100;
        let root_half = std::f64::consts::FRAC_1_SQRT_2;
        let poles = [[scale, 0.0, 0.0], [scale, scale, 0.0], [0.0, scale, 0.0]];
        let weights = [1.0e300, root_half * 1.0e300, 1.0e300];
        let point = rational_bezier_point(&poles, &weights, 0.5)
            .expect("extreme finite weights should remain evaluable");
        let expected = root_half * scale;
        assert!(point[0].is_finite() && point[1].is_finite());
        assert!((point[0] - expected).abs() / scale < 1.0e-14);
        assert!((point[1] - expected).abs() / scale < 1.0e-14);
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
