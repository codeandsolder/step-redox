use crate::instances::{
    build_index, cartesian_point, entity_id, entity_ref_value, number, simple_record,
};
use ruststep::ast::{EntityInstance, Parameter};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

const GEOM_TOL_MM: f64 = 1.0e-7;
const DIR_TOL: f64 = 1.0e-10;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecoveredSolidExtrusion {
    pub solid_id: u64,
    pub cap_face_ids: [u64; 2],
    pub side_face_ids: Vec<u64>,
    pub profile_points_mm: Vec<[f64; 2]>,
    pub origin_mm: [f64; 3],
    pub x_axis: [f64; 3],
    pub y_axis: [f64; 3],
    pub z_axis: [f64; 3],
    pub height_mm: f64,
    pub max_residual_mm: f64,
}

#[derive(Debug, Clone)]
struct FaceInfo {
    id: u64,
    plane: PlaneSupport,
    loop_edges: Vec<EdgeUse>,
}

#[derive(Debug, Clone, Copy)]
struct PlaneSupport {
    origin: [f64; 3],
    normal: [f64; 3],
}

#[derive(Debug, Clone)]
struct CanonicalProfile {
    origin_mm: [f64; 3],
    x_axis: [f64; 3],
    y_axis: [f64; 3],
    points_mm: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, Copy)]
struct EdgeUse {
    edge_id: u64,
    start_vertex: u64,
    end_vertex: u64,
    start_mm: [f64; 3],
    end_mm: [f64; 3],
}

pub fn detect_solid_extrusions(entities: &[EntityInstance]) -> Vec<RecoveredSolidExtrusion> {
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
        if let Some(candidate) = detect_one_solid(solid_id, entities, &index) {
            out.push(candidate);
        }
    }

    out.sort_by_key(|candidate| candidate.solid_id);
    out
}

fn detect_one_solid(
    solid_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<RecoveredSolidExtrusion> {
    let face_ids = solid_faces(solid_id, entities, index)?;
    if face_ids.len() < 5 {
        return None;
    }

    let faces = face_ids
        .iter()
        .copied()
        .map(|face_id| face_info(face_id, entities, index))
        .collect::<Option<Vec<_>>>()?;

    let mut edge_faces = HashMap::<u64, Vec<usize>>::new();
    for (face_index, face) in faces.iter().enumerate() {
        for edge in &face.loop_edges {
            edge_faces.entry(edge.edge_id).or_default().push(face_index);
        }
    }
    if edge_faces.values().any(|attached| attached.len() != 2) {
        return None;
    }

    let mut candidates = Vec::new();
    for first in 0..faces.len() {
        for second in first + 1..faces.len() {
            if let Some(candidate) =
                cap_pair_candidate(solid_id, first, second, &faces, &edge_faces)
            {
                candidates.push(candidate);
            }
        }
    }
    candidates.into_iter().min_by(candidate_order)
}

fn candidate_order(
    first: &RecoveredSolidExtrusion,
    second: &RecoveredSolidExtrusion,
) -> std::cmp::Ordering {
    first
        .profile_points_mm
        .len()
        .cmp(&second.profile_points_mm.len())
        .then_with(|| first.z_axis[0].total_cmp(&second.z_axis[0]))
        .then_with(|| first.z_axis[1].total_cmp(&second.z_axis[1]))
        .then_with(|| first.z_axis[2].total_cmp(&second.z_axis[2]))
        .then_with(|| first.height_mm.total_cmp(&second.height_mm))
}

fn cap_pair_candidate(
    solid_id: u64,
    first_index: usize,
    second_index: usize,
    faces: &[FaceInfo],
    edge_faces: &HashMap<u64, Vec<usize>>,
) -> Option<RecoveredSolidExtrusion> {
    let first = &faces[first_index];
    let second = &faces[second_index];
    if first.loop_edges.len() != second.loop_edges.len() || first.loop_edges.len() < 3 {
        return None;
    }
    if !parallel(first.plane.normal, second.plane.normal) {
        return None;
    }

    let z_axis = canonical_axis(first.plane.normal);
    let first_offset = dot(z_axis, first.plane.origin);
    let second_offset = dot(z_axis, second.plane.origin);
    let separation = second_offset - first_offset;
    if !separation.is_finite() || separation.abs() <= GEOM_TOL_MM {
        return None;
    }
    let (bottom_index, top_index, height_mm) = if separation > 0.0 {
        (first_index, second_index, separation)
    } else {
        (second_index, first_index, -separation)
    };
    let bottom = &faces[bottom_index];
    let top = &faces[top_index];
    let extrusion = mul(z_axis, height_mm);

    if !face_lies_on_plane(bottom) || !face_lies_on_plane(top) {
        return None;
    }

    let bottom_edges = bottom
        .loop_edges
        .iter()
        .map(|edge| edge.edge_id)
        .collect::<HashSet<_>>();
    let top_edges = top
        .loop_edges
        .iter()
        .map(|edge| edge.edge_id)
        .collect::<HashSet<_>>();
    if !bottom_edges.is_disjoint(&top_edges) {
        return None;
    }

    let side_indices = (0..faces.len())
        .filter(|&index| index != bottom_index && index != top_index)
        .collect::<Vec<_>>();
    if side_indices.len() != bottom.loop_edges.len() {
        return None;
    }

    let mut used_bottom_edges = HashSet::new();
    let mut used_top_edges = HashSet::new();
    let mut max_residual_mm = 0.0_f64;

    for &side_index in &side_indices {
        let side = &faces[side_index];
        if dot(side.plane.normal, z_axis).abs() > DIR_TOL * 100.0 {
            return None;
        }
        if side.loop_edges.len() != 4 {
            return None;
        }

        let side_edge_ids = side
            .loop_edges
            .iter()
            .map(|edge| edge.edge_id)
            .collect::<HashSet<_>>();
        let shared_bottom = side_edge_ids
            .intersection(&bottom_edges)
            .copied()
            .collect::<Vec<_>>();
        let shared_top = side_edge_ids
            .intersection(&top_edges)
            .copied()
            .collect::<Vec<_>>();
        if shared_bottom.len() != 1 || shared_top.len() != 1 {
            return None;
        }
        let bottom_edge_id = shared_bottom[0];
        let top_edge_id = shared_top[0];
        if !used_bottom_edges.insert(bottom_edge_id) || !used_top_edges.insert(top_edge_id) {
            return None;
        }

        let bottom_edge = bottom
            .loop_edges
            .iter()
            .find(|edge| edge.edge_id == bottom_edge_id)?;
        let top_edge = top
            .loop_edges
            .iter()
            .find(|edge| edge.edge_id == top_edge_id)?;
        let residual = translated_edge_residual(bottom_edge, top_edge, extrusion)?;
        max_residual_mm = max_residual_mm.max(residual);
        if residual > GEOM_TOL_MM {
            return None;
        }

        let lateral_edges = side
            .loop_edges
            .iter()
            .filter(|edge| edge.edge_id != bottom_edge_id && edge.edge_id != top_edge_id)
            .collect::<Vec<_>>();
        if lateral_edges.len() != 2 {
            return None;
        }
        for lateral in lateral_edges {
            let residual = connector_residual(lateral, extrusion)?;
            max_residual_mm = max_residual_mm.max(residual);
            if residual > GEOM_TOL_MM {
                return None;
            }
        }
    }

    if used_bottom_edges != bottom_edges || used_top_edges != top_edges {
        return None;
    }
    if edge_faces.iter().any(|(edge, attached)| {
        (bottom_edges.contains(edge) || top_edges.contains(edge)) && attached.len() != 2
    }) {
        return None;
    }

    let profile = canonical_profile(bottom, z_axis)?;

    Some(RecoveredSolidExtrusion {
        solid_id,
        cap_face_ids: [bottom.id, top.id],
        side_face_ids: side_indices.iter().map(|&index| faces[index].id).collect(),
        profile_points_mm: profile.points_mm,
        origin_mm: profile.origin_mm,
        x_axis: profile.x_axis,
        y_axis: profile.y_axis,
        z_axis,
        height_mm,
        max_residual_mm,
    })
}

fn canonical_profile(cap: &FaceInfo, z_axis: [f64; 3]) -> Option<CanonicalProfile> {
    let reference = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        .into_iter()
        .min_by(|a, b| dot(*a, z_axis).abs().total_cmp(&dot(*b, z_axis).abs()))?;
    let x_axis = normalize(sub(reference, mul(z_axis, dot(reference, z_axis))))?;
    let y_axis = normalize(cross(z_axis, x_axis))?;

    let world_points = cap
        .loop_edges
        .iter()
        .map(|edge| edge.start_mm)
        .collect::<Vec<_>>();
    if world_points.len() < 3 {
        return None;
    }
    for index in 0..cap.loop_edges.len() {
        let current = cap.loop_edges[index];
        let next = cap.loop_edges[(index + 1) % cap.loop_edges.len()];
        if current.end_vertex != next.start_vertex
            || distance(current.end_mm, next.start_mm) > GEOM_TOL_MM
        {
            return None;
        }
    }

    let projected = world_points
        .iter()
        .map(|&point| [dot(point, x_axis), dot(point, y_axis)])
        .collect::<Vec<_>>();
    let start_index = (0..projected.len()).min_by(|&a, &b| {
        projected[a][0]
            .total_cmp(&projected[b][0])
            .then_with(|| projected[a][1].total_cmp(&projected[b][1]))
    })?;

    let mut ordered = (0..projected.len())
        .map(|offset| projected[(start_index + offset) % projected.len()])
        .collect::<Vec<_>>();
    let origin_mm = world_points[start_index];
    let origin_2d = ordered[0];
    for point in &mut ordered {
        point[0] -= origin_2d[0];
        point[1] -= origin_2d[1];
        if point[0].abs() <= 1.0e-14 {
            point[0] = 0.0;
        }
        if point[1].abs() <= 1.0e-14 {
            point[1] = 0.0;
        }
    }

    if signed_area(&ordered) < 0.0 {
        let first = ordered[0];
        ordered.reverse();
        let pos = ordered.iter().position(|point| *point == first)?;
        ordered.rotate_left(pos);
    }

    Some(CanonicalProfile {
        origin_mm,
        x_axis,
        y_axis,
        points_mm: ordered,
    })
}

fn solid_faces(
    solid_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<u64>> {
    let solid = simple_record(entities.get(*index.get(&solid_id)?)?)?;
    let Parameter::List(params) = &solid.parameter else {
        return None;
    };
    let shell_id = entity_ref_value(params.get(1)?)?;
    let shell = simple_record(entities.get(*index.get(&shell_id)?)?)?;
    if shell.name != "CLOSED_SHELL" {
        return None;
    }
    let Parameter::List(shell_params) = &shell.parameter else {
        return None;
    };
    let Parameter::List(faces) = shell_params.get(1)? else {
        return None;
    };
    faces.iter().map(entity_ref_value).collect()
}

fn face_info(
    face_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<FaceInfo> {
    let face = simple_record(entities.get(*index.get(&face_id)?)?)?;
    if face.name != "ADVANCED_FACE" && face.name != "FACE_SURFACE" {
        return None;
    }
    let Parameter::List(params) = &face.parameter else {
        return None;
    };
    let Parameter::List(bounds) = params.get(1)? else {
        return None;
    };
    if bounds.len() != 1 {
        return None;
    }
    let surface_id = entity_ref_value(params.get(2)?)?;
    let plane = plane_support(surface_id, entities, index)?;

    let bound_id = entity_ref_value(bounds.first()?)?;
    let bound = simple_record(entities.get(*index.get(&bound_id)?)?)?;
    if bound.name != "FACE_BOUND" && bound.name != "FACE_OUTER_BOUND" {
        return None;
    }
    let Parameter::List(bound_params) = &bound.parameter else {
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
    let Parameter::List(oriented_edges) = loop_params.get(1)? else {
        return None;
    };

    let mut loop_edges = Vec::with_capacity(oriented_edges.len());
    for oriented in oriented_edges {
        let oriented_id = entity_ref_value(oriented)?;
        let oriented_record = simple_record(entities.get(*index.get(&oriented_id)?)?)?;
        if oriented_record.name != "ORIENTED_EDGE" {
            return None;
        }
        let Parameter::List(oriented_params) = &oriented_record.parameter else {
            return None;
        };
        let edge_id = entity_ref_value(oriented_params.get(3)?)?;
        let forward = enumeration_bool(oriented_params.get(4)?)?;

        let edge_record = simple_record(entities.get(*index.get(&edge_id)?)?)?;
        if edge_record.name != "EDGE_CURVE" {
            return None;
        }
        let Parameter::List(edge_params) = &edge_record.parameter else {
            return None;
        };
        let start_vertex = entity_ref_value(edge_params.get(1)?)?;
        let end_vertex = entity_ref_value(edge_params.get(2)?)?;
        let curve_id = entity_ref_value(edge_params.get(3)?)?;
        let curve = simple_record(entities.get(*index.get(&curve_id)?)?)?;
        if curve.name != "LINE" {
            return None;
        }

        let start_mm = vertex_point(start_vertex, entities, index)?;
        let end_mm = vertex_point(end_vertex, entities, index)?;
        let edge_use = if forward {
            EdgeUse {
                edge_id,
                start_vertex,
                end_vertex,
                start_mm,
                end_mm,
            }
        } else {
            EdgeUse {
                edge_id,
                start_vertex: end_vertex,
                end_vertex: start_vertex,
                start_mm: end_mm,
                end_mm: start_mm,
            }
        };
        loop_edges.push(edge_use);
    }

    Some(FaceInfo {
        id: face_id,
        plane,
        loop_edges,
    })
}

fn plane_support(
    surface_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<PlaneSupport> {
    let surface = simple_record(entities.get(*index.get(&surface_id)?)?)?;
    if surface.name != "PLANE" {
        return None;
    }
    let Parameter::List(params) = &surface.parameter else {
        return None;
    };
    let placement_id = entity_ref_value(params.get(1)?)?;
    let placement = simple_record(entities.get(*index.get(&placement_id)?)?)?;
    if placement.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let Parameter::List(placement_params) = &placement.parameter else {
        return None;
    };
    let origin = cartesian_point(entity_ref_value(placement_params.get(1)?)?, entities, index)?;
    let normal = direction(entity_ref_value(placement_params.get(2)?)?, entities, index)?;
    Some(PlaneSupport {
        origin,
        normal: normalize(normal)?,
    })
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
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let Parameter::List(values) = params.get(1)? else {
        return None;
    };
    if values.len() != 3 {
        return None;
    }
    Some([
        number(&values[0])?,
        number(&values[1])?,
        number(&values[2])?,
    ])
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
    let Parameter::List(params) = &vertex.parameter else {
        return None;
    };
    cartesian_point(entity_ref_value(params.get(1)?)?, entities, index)
}

fn face_lies_on_plane(face: &FaceInfo) -> bool {
    face.loop_edges.iter().all(|edge| {
        point_plane_distance(edge.start_mm, face.plane) <= GEOM_TOL_MM
            && point_plane_distance(edge.end_mm, face.plane) <= GEOM_TOL_MM
    })
}

fn translated_edge_residual(bottom: &EdgeUse, top: &EdgeUse, extrusion: [f64; 3]) -> Option<f64> {
    let direct = distance(add(bottom.start_mm, extrusion), top.start_mm)
        .max(distance(add(bottom.end_mm, extrusion), top.end_mm));
    let reverse = distance(add(bottom.start_mm, extrusion), top.end_mm)
        .max(distance(add(bottom.end_mm, extrusion), top.start_mm));
    Some(direct.min(reverse))
}

fn connector_residual(edge: &EdgeUse, extrusion: [f64; 3]) -> Option<f64> {
    let forward = distance(sub(edge.end_mm, edge.start_mm), extrusion);
    let reverse = distance(sub(edge.start_mm, edge.end_mm), extrusion);
    Some(forward.min(reverse))
}

fn point_plane_distance(point: [f64; 3], plane: PlaneSupport) -> f64 {
    dot(plane.normal, sub(point, plane.origin)).abs()
}

fn enumeration_bool(parameter: &Parameter) -> Option<bool> {
    match parameter {
        Parameter::Enumeration(value) if value == "T" => Some(true),
        Parameter::Enumeration(value) if value == "F" => Some(false),
        _ => None,
    }
}

fn parallel(a: [f64; 3], b: [f64; 3]) -> bool {
    (dot(a, b).abs() - 1.0).abs() <= DIR_TOL
}

fn canonical_axis(mut axis: [f64; 3]) -> [f64; 3] {
    for value in axis {
        if value.abs() > DIR_TOL {
            if value < 0.0 {
                axis = mul(axis, -1.0);
            }
            break;
        }
    }
    axis
}

fn signed_area(points: &[[f64; 2]]) -> f64 {
    let mut twice_area = 0.0;
    for index in 0..points.len() {
        let next = (index + 1) % points.len();
        twice_area += points[index][0] * points[next][1] - points[next][0] * points[index][1];
    }
    twice_area * 0.5
}

fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    let length = norm(vector);
    if !length.is_finite() || length <= DIR_TOL {
        return None;
    }
    Some(mul(vector, 1.0 / length))
}

fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mul(vector: [f64; 3], scalar: f64) -> [f64; 3] {
    [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_axis_has_deterministic_sign() {
        assert_eq!(canonical_axis([-1.0, 0.0, 0.0]), [1.0, -0.0, -0.0]);
        assert_eq!(canonical_axis([0.0, -1.0, 0.0]), [-0.0, 1.0, -0.0]);
    }

    #[test]
    fn translated_edge_residual_ignores_edge_orientation() {
        let bottom = EdgeUse {
            edge_id: 1,
            start_vertex: 10,
            end_vertex: 11,
            start_mm: [0.0, 0.0, 0.0],
            end_mm: [2.0, 0.0, 0.0],
        };
        let top = EdgeUse {
            edge_id: 2,
            start_vertex: 21,
            end_vertex: 20,
            start_mm: [2.0, 0.0, 3.0],
            end_mm: [0.0, 0.0, 3.0],
        };
        assert_eq!(
            translated_edge_residual(&bottom, &top, [0.0, 0.0, 3.0]),
            Some(0.0)
        );
    }
}
