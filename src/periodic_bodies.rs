use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

use crate::patterns::InstancePattern;

const GEOM_TOL_MM: f64 = 1.0e-5;
const DIR_TOL: f64 = 1.0e-10;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PeriodicFaceFamily {
    pub face_ids: Vec<u64>,
    pub phase_mm: f64,
    pub span_mm: f64,
    pub orthogonal_center_mm: [f64; 2],
    pub bounds: usize,
    pub edges: usize,
    pub max_residual_mm: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PeriodicBodyPattern {
    pub solid_id: u64,
    pub axis: [f64; 3],
    pub pitch_mm: f64,
    pub sites: usize,
    pub coupled_instance_patterns: Vec<usize>,
    pub repeat_face_families: Vec<PeriodicFaceFamily>,
    pub repeat_faces: usize,
    pub faces_per_site: usize,
    pub stretch_face_ids: Vec<u64>,
    pub fixed_face_ids: Vec<u64>,
    pub repeat_coverage_ratio: f64,
    pub max_residual_mm: f64,
}

#[derive(Debug, Clone)]
struct LatticeCandidate {
    axis: [f64; 3],
    pitch: f64,
    sites: usize,
    pattern_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct FaceInfo {
    id: u64,
    projected: f64,
    span: f64,
    key: FaceKey,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct FaceKey {
    orthogonal_center: [i64; 2],
    normal: [i64; 3],
    bounds: usize,
    edges: usize,
    same_sense: bool,
    local_points: Vec<[i64; 3]>,
}

pub fn detect_periodic_bodies(
    entities: &[EntityInstance],
    instance_patterns: &[InstancePattern],
) -> Vec<PeriodicBodyPattern> {
    let candidates = lattice_candidates(instance_patterns);
    if candidates.is_empty() {
        return Vec::new();
    }

    let index = build_index(entities);
    let solids = entities
        .iter()
        .filter_map(|entity| {
            let record = simple_record(entity)?;
            (record.name == "MANIFOLD_SOLID_BREP").then_some(entity_id(entity))
        })
        .collect::<Vec<_>>();

    let mut out = Vec::new();
    for candidate in candidates {
        let Some((v, w)) = orthogonal_basis(candidate.axis) else {
            continue;
        };
        let pitch_ticks = (candidate.pitch / GEOM_TOL_MM).round() as i64;
        if pitch_ticks <= 0 {
            continue;
        }

        for &solid in &solids {
            let Some(face_ids) = solid_faces(solid, entities, &index) else {
                continue;
            };
            if face_ids.len() < candidate.sites.saturating_mul(4) {
                continue;
            }

            let mut grouped: HashMap<(FaceKey, i64), Vec<FaceInfo>> = HashMap::new();
            let mut all_face_ids = Vec::with_capacity(face_ids.len());
            let mut analyzable = HashMap::<u64, FaceInfo>::new();

            for face in face_ids.iter().copied() {
                all_face_ids.push(face);
                let Some(info) = analyze_planar_line_face(
                    face,
                    candidate.axis,
                    v,
                    w,
                    entities,
                    &index,
                ) else {
                    continue;
                };
                let projected_ticks = (info.projected / GEOM_TOL_MM).round() as i64;
                let phase = projected_ticks.rem_euclid(pitch_ticks);
                grouped
                    .entry((info.key.clone(), phase))
                    .or_default()
                    .push(info.clone());
                analyzable.insert(face, info);
            }

            let mut families = Vec::new();
            let mut repeat_set = HashSet::new();
            let mut max_residual = 0.0f64;

            for ((_key, phase_ticks), mut group) in grouped {
                if group.len() != candidate.sites {
                    continue;
                }
                group.sort_by(|a, b| a.projected.total_cmp(&b.projected));
                let start = group[0].projected;
                let mut residual = 0.0f64;
                let mut okay = true;
                for (site, face) in group.iter().enumerate() {
                    let expected = start + site as f64 * candidate.pitch;
                    let err = (face.projected - expected).abs();
                    residual = residual.max(err);
                    if err > GEOM_TOL_MM {
                        okay = false;
                        break;
                    }
                }
                if !okay {
                    continue;
                }

                let span0 = group[0].span;
                if group
                    .iter()
                    .any(|face| (face.span - span0).abs() > GEOM_TOL_MM)
                {
                    continue;
                }

                let orth = [
                    group[0].key.orthogonal_center[0] as f64 * GEOM_TOL_MM,
                    group[0].key.orthogonal_center[1] as f64 * GEOM_TOL_MM,
                ];
                let ids = group.iter().map(|face| face.id).collect::<Vec<_>>();
                repeat_set.extend(ids.iter().copied());
                max_residual = max_residual.max(residual);
                families.push(PeriodicFaceFamily {
                    face_ids: ids,
                    phase_mm: phase_ticks as f64 * GEOM_TOL_MM,
                    span_mm: span0,
                    orthogonal_center_mm: orth,
                    bounds: group[0].key.bounds,
                    edges: group[0].key.edges,
                    max_residual_mm: residual,
                });
            }

            if families.len() < 4 || repeat_set.len() < candidate.sites.saturating_mul(4) {
                continue;
            }
            if repeat_set.len() % candidate.sites != 0 {
                continue;
            }

            let min_stretch = (candidate.sites.saturating_sub(1)) as f64 * candidate.pitch
                - GEOM_TOL_MM;
            let mut stretch = Vec::new();
            let mut fixed = Vec::new();
            for face in all_face_ids {
                if repeat_set.contains(&face) {
                    continue;
                }
                if let Some(info) = analyzable.get(&face) {
                    if info.span >= min_stretch {
                        stretch.push(face);
                        continue;
                    }
                }
                fixed.push(face);
            }

            families.sort_by(|a, b| {
                a.phase_mm
                    .total_cmp(&b.phase_mm)
                    .then_with(|| a.orthogonal_center_mm[0].total_cmp(&b.orthogonal_center_mm[0]))
                    .then_with(|| a.orthogonal_center_mm[1].total_cmp(&b.orthogonal_center_mm[1]))
                    .then_with(|| a.span_mm.total_cmp(&b.span_mm))
            });
            stretch.sort_unstable();
            fixed.sort_unstable();

            let repeat_faces = repeat_set.len();
            out.push(PeriodicBodyPattern {
                solid_id: solid,
                axis: candidate.axis,
                pitch_mm: candidate.pitch,
                sites: candidate.sites,
                coupled_instance_patterns: candidate.pattern_indices.clone(),
                repeat_face_families: families,
                repeat_faces,
                faces_per_site: repeat_faces / candidate.sites,
                stretch_face_ids: stretch,
                fixed_face_ids: fixed,
                repeat_coverage_ratio: repeat_faces as f64 / face_ids.len().max(1) as f64,
                max_residual_mm: max_residual,
            });
        }
    }

    out.sort_by(|a, b| {
        b.repeat_faces
            .cmp(&a.repeat_faces)
            .then_with(|| a.solid_id.cmp(&b.solid_id))
    });
    out.dedup_by(|a, b| a.solid_id == b.solid_id
        && a.sites == b.sites
        && (a.pitch_mm - b.pitch_mm).abs() <= GEOM_TOL_MM
        && parallel(a.axis, b.axis));
    out
}

fn lattice_candidates(patterns: &[InstancePattern]) -> Vec<LatticeCandidate> {
    let mut out = Vec::<LatticeCandidate>::new();
    for (index, pattern) in patterns.iter().enumerate() {
        if pattern.dimension != 1
            || pattern.item_ids.len() < 4
            || pattern.grid_shape.len() != 1
            || pattern.grid_shape[0] != pattern.item_ids.len()
            || (pattern.fill_ratio - 1.0).abs() > 1.0e-12
            || pattern.basis.len() != 1
        {
            continue;
        }
        let Some(mut axis) = normalize(pattern.basis[0]) else {
            continue;
        };
        canonicalize_axis(&mut axis);
        let pitch = norm(pattern.basis[0]);
        if !pitch.is_finite() || pitch <= GEOM_TOL_MM {
            continue;
        }
        let sites = pattern.item_ids.len();

        if let Some(existing) = out.iter_mut().find(|existing| {
            existing.sites == sites
                && (existing.pitch - pitch).abs() <= GEOM_TOL_MM
                && parallel(existing.axis, axis)
        }) {
            existing.pattern_indices.push(index);
        } else {
            out.push(LatticeCandidate {
                axis,
                pitch,
                sites,
                pattern_indices: vec![index],
            });
        }
    }
    out
}

fn analyze_planar_line_face(
    face: u64,
    axis: [f64; 3],
    v: [f64; 3],
    w: [f64; 3],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<FaceInfo> {
    let record = simple_record(entities.get(*index.get(&face)?)?)?;
    if record.name != "ADVANCED_FACE" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let bounds = match params.get(1)? {
        Parameter::List(items) => items.len(),
        _ => return None,
    };
    let surface = entity_ref_value(params.get(2)?)?;
    let same_sense = matches!(params.get(3)?, Parameter::Enumeration(value) if value == "T");

    let surface_record = simple_record(entities.get(*index.get(&surface)?)?)?;
    if surface_record.name != "PLANE" {
        return None;
    }
    let normal = plane_normal(surface_record, entities, index)?;
    let normal_q = quantize_dir(normal)?;

    let edge_ids = face_edges(face, entities, index)?;
    if edge_ids.is_empty() {
        return None;
    }
    let mut point_ids = HashSet::new();
    for &edge in &edge_ids {
        let edge_record = simple_record(entities.get(*index.get(&edge)?)?)?;
        if edge_record.name != "EDGE_CURVE" {
            return None;
        }
        let Parameter::List(edge_params) = &edge_record.parameter else {
            return None;
        };
        let start = entity_ref_value(edge_params.get(1)?)?;
        let end = entity_ref_value(edge_params.get(2)?)?;
        let curve = entity_ref_value(edge_params.get(3)?)?;
        let curve_record = simple_record(entities.get(*index.get(&curve)?)?)?;
        if curve_record.name != "LINE" {
            return None;
        }
        for vertex in [start, end] {
            let vertex_record = simple_record(entities.get(*index.get(&vertex)?)?)?;
            if vertex_record.name != "VERTEX_POINT" {
                return None;
            }
            let Parameter::List(vertex_params) = &vertex_record.parameter else {
                return None;
            };
            point_ids.insert(entity_ref_value(vertex_params.get(1)?)?);
        }
    }
    if point_ids.len() < 3 {
        return None;
    }

    let mut points = Vec::with_capacity(point_ids.len());
    for point in point_ids {
        points.push(cartesian_point(point, entities, index)?);
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for point in &points {
        for k in 0..3 {
            lo[k] = lo[k].min(point[k]);
            hi[k] = hi[k].max(point[k]);
        }
    }
    let center = [
        (lo[0] + hi[0]) * 0.5,
        (lo[1] + hi[1]) * 0.5,
        (lo[2] + hi[2]) * 0.5,
    ];
    let projected_values = points.iter().map(|&point| dot(point, axis)).collect::<Vec<_>>();
    let min_projected = projected_values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_projected = projected_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let projected = dot(center, axis);
    let span = max_projected - min_projected;

    let mut local_points = points
        .iter()
        .map(|point| {
            [
                quantize_mm(point[0] - center[0]),
                quantize_mm(point[1] - center[1]),
                quantize_mm(point[2] - center[2]),
            ]
        })
        .collect::<Vec<_>>();
    local_points.sort_unstable();
    local_points.dedup();

    Some(FaceInfo {
        id: face,
        projected,
        span,
        key: FaceKey {
            orthogonal_center: [quantize_mm(dot(center, v)), quantize_mm(dot(center, w))],
            normal: normal_q,
            bounds,
            edges: edge_ids.len(),
            same_sense,
            local_points,
        },
    })
}

fn solid_faces(
    solid: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<u64>> {
    let record = simple_record(entities.get(*index.get(&solid)?)?)?;
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let shell = entity_ref_value(params.get(1)?)?;
    let shell_record = simple_record(entities.get(*index.get(&shell)?)?)?;
    if shell_record.name != "CLOSED_SHELL" {
        return None;
    }
    let Parameter::List(shell_params) = &shell_record.parameter else {
        return None;
    };
    let Parameter::List(face_refs) = shell_params.get(1)? else {
        return None;
    };
    face_refs.iter().map(entity_ref_value).collect()
}

fn face_edges(
    face: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<u64>> {
    let record = simple_record(entities.get(*index.get(&face)?)?)?;
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let Parameter::List(bound_refs) = params.get(1)? else {
        return None;
    };
    let mut edges = HashSet::new();
    for bound_ref in bound_refs {
        let bound = entity_ref_value(bound_ref)?;
        let bound_record = simple_record(entities.get(*index.get(&bound)?)?)?;
        if bound_record.name != "FACE_BOUND" && bound_record.name != "FACE_OUTER_BOUND" {
            return None;
        }
        let Parameter::List(bound_params) = &bound_record.parameter else {
            return None;
        };
        let loop_id = entity_ref_value(bound_params.get(1)?)?;
        let loop_record = simple_record(entities.get(*index.get(&loop_id)?)?)?;
        if loop_record.name != "EDGE_LOOP" {
            return None;
        }
        let Parameter::List(loop_params) = &loop_record.parameter else {
            return None;
        };
        let Parameter::List(oriented_refs) = loop_params.get(1)? else {
            return None;
        };
        for oriented_ref in oriented_refs {
            let oriented = entity_ref_value(oriented_ref)?;
            let oriented_record = simple_record(entities.get(*index.get(&oriented)?)?)?;
            if oriented_record.name != "ORIENTED_EDGE" {
                return None;
            }
            let Parameter::List(oriented_params) = &oriented_record.parameter else {
                return None;
            };
            edges.insert(entity_ref_value(oriented_params.get(3)?)?);
        }
    }
    let mut edges = edges.into_iter().collect::<Vec<_>>();
    edges.sort_unstable();
    Some(edges)
}

fn plane_normal(
    plane: &Record,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let Parameter::List(params) = &plane.parameter else {
        return None;
    };
    let placement = entity_ref_value(params.get(1)?)?;
    let placement_record = simple_record(entities.get(*index.get(&placement)?)?)?;
    if placement_record.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let Parameter::List(place_params) = &placement_record.parameter else {
        return None;
    };
    let direction = entity_ref_value(place_params.get(2)?)?;
    let direction_record = simple_record(entities.get(*index.get(&direction)?)?)?;
    if direction_record.name != "DIRECTION" {
        return None;
    }
    let Parameter::List(dir_params) = &direction_record.parameter else {
        return None;
    };
    let Parameter::List(coords) = dir_params.get(1)? else {
        return None;
    };
    if coords.len() != 3 {
        return None;
    }
    normalize([
        numeric_value(&coords[0])?,
        numeric_value(&coords[1])?,
        numeric_value(&coords[2])?,
    ])
}

fn cartesian_point(
    point: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let record = simple_record(entities.get(*index.get(&point)?)?)?;
    if record.name != "CARTESIAN_POINT" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
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

fn orthogonal_basis(axis: [f64; 3]) -> Option<([f64; 3], [f64; 3])> {
    let reference = if axis[0].abs() < 0.8 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let v = normalize(cross(axis, reference))?;
    let w = normalize(cross(axis, v))?;
    Some((v, w))
}

fn quantize_mm(value: f64) -> i64 {
    (value / GEOM_TOL_MM).round() as i64
}

fn quantize_dir(v: [f64; 3]) -> Option<[i64; 3]> {
    let mut out = [0i64; 3];
    for (i, value) in v.into_iter().enumerate() {
        let q = (value / DIR_TOL).round();
        if !q.is_finite() {
            return None;
        }
        out[i] = q as i64;
    }
    Some(out)
}

fn numeric_value(param: &Parameter) -> Option<f64> {
    match param {
        Parameter::Integer(value) => Some(*value as f64),
        Parameter::Real(value) => Some(*value),
        _ => None,
    }
}

fn entity_ref_value(param: &Parameter) -> Option<u64> {
    match param {
        Parameter::Ref(Name::Entity(id)) => Some(*id),
        _ => None,
    }
}

fn build_index(entities: &[EntityInstance]) -> HashMap<u64, usize> {
    entities
        .iter()
        .enumerate()
        .map(|(idx, entity)| (entity_id(entity), idx))
        .collect()
}

fn simple_record(entity: &EntityInstance) -> Option<&Record> {
    match entity {
        EntityInstance::Simple { record, .. } => Some(record),
        EntityInstance::Complex { .. } => None,
    }
}

fn entity_id(entity: &EntityInstance) -> u64 {
    match entity {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id,
    }
}

fn canonicalize_axis(axis: &mut [f64; 3]) {
    for value in axis.iter() {
        if value.abs() > 1.0e-12 {
            if *value < 0.0 {
                *axis = scale(*axis, -1.0);
            }
            return;
        }
    }
}

fn parallel(a: [f64; 3], b: [f64; 3]) -> bool {
    norm(cross(a, b)) <= 1.0e-8
}

fn scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

fn normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = norm(v);
    if !n.is_finite() || n <= 1.0e-15 {
        return None;
    }
    Some(scale(v, 1.0 / n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_axis_has_stable_sign() {
        let mut axis = [-1.0, 0.0, 0.0];
        canonicalize_axis(&mut axis);
        assert_eq!(axis, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn orthogonal_basis_is_perpendicular() {
        let axis = normalize([1.0, 2.0, 3.0]).unwrap();
        let (v, w) = orthogonal_basis(axis).unwrap();
        assert!(dot(axis, v).abs() < 1e-12);
        assert!(dot(axis, w).abs() < 1e-12);
        assert!(dot(v, w).abs() < 1e-12);
    }
}
