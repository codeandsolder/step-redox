use crate::brep::{self, CurveSupport, SurfaceSupport};
use crate::instances::{build_index, entity_id, simple_record};
use ruststep::ast::EntityInstance;
use serde::Serialize;
use std::collections::HashMap;

const GEOM_TOL_MM: f64 = 1.0e-7;
const DIR_TOL: f64 = 1.0e-9;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecoveredSolidRevolution {
    pub solid_id: u64,
    pub face_ids: Vec<u64>,
    /// Closed meridian polygon in local [radius, axial] coordinates.
    pub profile_points_mm: Vec<[f64; 2]>,
    /// Canonical closest point on the revolution axis to global origin.
    pub axis_origin_mm: [f64; 3],
    /// Canonical axis direction.
    pub axis_direction: [f64; 3],
    /// Deterministic local +X radial direction, perpendicular to the axis.
    pub radial_direction: [f64; 3],
    pub max_residual_mm: f64,
}

#[derive(Debug, Clone, Copy)]
struct Segment2 {
    a: [f64; 2],
    b: [f64; 2],
}

type ProfileGraph = (Vec<[f64; 2]>, Vec<(usize, usize)>);

#[derive(Debug, Clone)]
struct FaceInfo {
    surface: SurfaceSupport,
    loops: Vec<brep::FaceLoop>,
}

pub fn detect_solid_revolutions(entities: &[EntityInstance]) -> Vec<RecoveredSolidRevolution> {
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
) -> Option<RecoveredSolidRevolution> {
    let face_ids = brep::solid_face_ids(solid_id, entities, index)?;
    if face_ids.len() < 3 {
        return None;
    }
    let faces = face_ids
        .iter()
        .copied()
        .map(|id| {
            Some(FaceInfo {
                surface: brep::surface_support(
                    brep::face_surface(id, entities, index)?,
                    entities,
                    index,
                ),
                loops: brep::face_loops(id, entities, index)?,
            })
        })
        .collect::<Option<Vec<_>>>()?;

    // Closed two-manifold proof. A STEP seam may occur twice on one face; counting
    // oriented uses rather than distinct faces handles that case correctly.
    let mut edge_faces = HashMap::<u64, Vec<usize>>::new();
    for (face_index, face) in faces.iter().enumerate() {
        if face.loops.is_empty() {
            return None;
        }
        for edge in face.loops.iter().flat_map(|loop_| &loop_.edges) {
            edge_faces.entry(edge.edge_id).or_default().push(face_index);
        }
    }
    if edge_faces.values().any(|attached| attached.len() != 2)
        || !shell_faces_connected(faces.len(), &edge_faces)
    {
        return None;
    }

    let first_cylinder = faces.iter().find_map(|face| match face.surface {
        SurfaceSupport::Cylinder(cylinder) => Some(cylinder),
        _ => None,
    })?;
    let axis_direction = canonical_axis(first_cylinder.axis);
    let axis_origin_mm =
        closest_axis_point_to_global_origin(first_cylinder.axis_origin_mm, axis_direction);
    let radial_direction = radial_basis(axis_direction)?;

    let mut max_residual_mm = 0.0_f64;
    let mut segments = Vec::new();
    for (face_index, face) in faces.iter().enumerate() {
        let (segment, residual) = match face.surface {
            SurfaceSupport::Cylinder(cylinder) => cylinder_profile_segment(
                face_index,
                face,
                cylinder,
                axis_origin_mm,
                axis_direction,
                &faces,
                &edge_faces,
            )?,
            SurfaceSupport::Plane(plane) => plane_profile_segment(
                face_index,
                face,
                plane,
                axis_origin_mm,
                axis_direction,
                &faces,
                &edge_faces,
            )?,
            _ => return None,
        };
        max_residual_mm = max_residual_mm.max(residual);
        if max_residual_mm > GEOM_TOL_MM {
            return None;
        }
        push_unique_segment(&mut segments, segment);
    }

    let profile_points_mm = closed_profile_from_segments(segments)?;
    Some(RecoveredSolidRevolution {
        solid_id,
        face_ids,
        profile_points_mm,
        axis_origin_mm,
        axis_direction,
        radial_direction,
        max_residual_mm,
    })
}

fn cylinder_profile_segment(
    face_index: usize,
    face: &FaceInfo,
    cylinder: brep::CylinderSupport,
    axis_origin: [f64; 3],
    axis: [f64; 3],
    faces: &[FaceInfo],
    edge_faces: &HashMap<u64, Vec<usize>>,
) -> Option<(Segment2, f64)> {
    if face.loops.len() != 1
        || !parallel(cylinder.axis, axis)
        || axis_distance(cylinder.axis_origin_mm, axis_origin, axis) > GEOM_TOL_MM
        || !cylinder.radius_mm.is_finite()
        || cylinder.radius_mm <= GEOM_TOL_MM
    {
        return None;
    }

    let mut min_t = f64::INFINITY;
    let mut max_t = f64::NEG_INFINITY;
    let mut max_residual = axis_distance(cylinder.axis_origin_mm, axis_origin, axis);

    for edge in &face.loops[0].edges {
        for point in [edge.start_mm, edge.end_mm] {
            let radius = point_axis_distance(point, axis_origin, axis);
            max_residual = max_residual.max((radius - cylinder.radius_mm).abs());
            let t = axial_coordinate(point, axis_origin, axis);
            min_t = min_t.min(t);
            max_t = max_t.max(t);
        }

        match &edge.support {
            CurveSupport::Line(line) => {
                let alignment = 1.0 - dot(line.direction, axis).abs();
                if alignment > DIR_TOL {
                    return None;
                }
                max_residual = max_residual.max(
                    (point_axis_distance(line.origin_mm, axis_origin, axis) - cylinder.radius_mm)
                        .abs(),
                );
            }
            CurveSupport::Circle(circle) => {
                if !parallel(circle.normal, axis) {
                    return None;
                }
                max_residual = max_residual
                    .max(axis_distance(circle.center_mm, axis_origin, axis))
                    .max((circle.radius_mm - cylinder.radius_mm).abs());
            }
            CurveSupport::BSpline(_) => {
                let neighbor = unique_neighbor_face(face_index, edge.edge_id, edge_faces)?;
                let SurfaceSupport::Plane(plane) = faces.get(neighbor)?.surface else {
                    return None;
                };
                if !parallel(plane.normal, axis) {
                    return None;
                }
                let start_axial = axial_coordinate(edge.start_mm, axis_origin, axis);
                let end_axial = axial_coordinate(edge.end_mm, axis_origin, axis);
                let plane_axial = axial_coordinate(plane.origin_mm, axis_origin, axis);
                max_residual = max_residual
                    .max((start_axial - end_axial).abs())
                    .max((start_axial - plane_axial).abs())
                    .max((end_axial - plane_axial).abs())
                    .max(plane.max_residual_mm);
            }
            CurveSupport::Other { .. } => return None,
        }
    }

    if !min_t.is_finite() || max_t - min_t <= GEOM_TOL_MM || max_residual > GEOM_TOL_MM {
        return None;
    }
    Some((
        Segment2 {
            a: [cylinder.radius_mm, min_t],
            b: [cylinder.radius_mm, max_t],
        },
        max_residual,
    ))
}

fn plane_profile_segment(
    face_index: usize,
    face: &FaceInfo,
    plane: brep::PlaneSupport,
    axis_origin: [f64; 3],
    axis: [f64; 3],
    faces: &[FaceInfo],
    edge_faces: &HashMap<u64, Vec<usize>>,
) -> Option<(Segment2, f64)> {
    if face.loops.is_empty() || face.loops.len() > 2 || !parallel(plane.normal, axis) {
        return None;
    }
    let t = axial_coordinate(plane.origin_mm, axis_origin, axis);
    let mut min_r = f64::INFINITY;
    let mut max_r = f64::NEG_INFINITY;
    let mut max_residual = plane.max_residual_mm;
    let mut only_circles = true;

    for loop_ in &face.loops {
        if loop_.edges.is_empty() {
            return None;
        }
        for edge in &loop_.edges {
            for point in [edge.start_mm, edge.end_mm] {
                max_residual =
                    max_residual.max((axial_coordinate(point, axis_origin, axis) - t).abs());
                let radius = point_axis_distance(point, axis_origin, axis);
                min_r = min_r.min(radius);
                max_r = max_r.max(radius);
            }

            match &edge.support {
                CurveSupport::Circle(circle) => {
                    if !parallel(circle.normal, axis) {
                        return None;
                    }
                    max_residual =
                        max_residual.max(axis_distance(circle.center_mm, axis_origin, axis));
                    min_r = min_r.min(circle.radius_mm);
                    max_r = max_r.max(circle.radius_mm);
                }
                CurveSupport::Line(line) => {
                    only_circles = false;
                    if dot(line.direction, axis).abs() > DIR_TOL {
                        return None;
                    }
                    // A radial meridian line must intersect the revolution axis.
                    let normal = cross(axis, line.direction);
                    let normal_len = norm(normal);
                    if normal_len <= DIR_TOL {
                        return None;
                    }
                    let line_axis_distance = dot(
                        sub(line.origin_mm, axis_origin),
                        mul(normal, 1.0 / normal_len),
                    )
                    .abs();
                    max_residual = max_residual.max(line_axis_distance);
                }
                CurveSupport::BSpline(_) => {
                    let neighbor = unique_neighbor_face(face_index, edge.edge_id, edge_faces)?;
                    let SurfaceSupport::Cylinder(cylinder) = faces.get(neighbor)?.surface else {
                        return None;
                    };
                    if !parallel(cylinder.axis, axis)
                        || axis_distance(cylinder.axis_origin_mm, axis_origin, axis) > GEOM_TOL_MM
                    {
                        return None;
                    }
                    let start_radius = point_axis_distance(edge.start_mm, axis_origin, axis);
                    let end_radius = point_axis_distance(edge.end_mm, axis_origin, axis);
                    max_residual = max_residual
                        .max((start_radius - cylinder.radius_mm).abs())
                        .max((end_radius - cylinder.radius_mm).abs())
                        .max(axis_distance(cylinder.axis_origin_mm, axis_origin, axis));
                    min_r = min_r.min(cylinder.radius_mm);
                    max_r = max_r.max(cylinder.radius_mm);
                }
                CurveSupport::Other { .. } => return None,
            }
        }
    }

    if !min_r.is_finite() || !max_r.is_finite() || max_residual > GEOM_TOL_MM {
        return None;
    }
    if max_r - min_r <= GEOM_TOL_MM {
        // A single closed coaxial circular boundary on an axis-normal plane is a disk.
        if face.loops.len() == 1 && only_circles {
            min_r = 0.0;
        } else {
            return None;
        }
    }
    if min_r <= GEOM_TOL_MM {
        min_r = 0.0;
    }
    if max_r - min_r <= GEOM_TOL_MM {
        return None;
    }
    Some((
        Segment2 {
            a: [min_r, t],
            b: [max_r, t],
        },
        max_residual,
    ))
}

fn shell_faces_connected(face_count: usize, edge_faces: &HashMap<u64, Vec<usize>>) -> bool {
    if face_count == 0 {
        return false;
    }
    let mut adjacency = vec![Vec::<usize>::new(); face_count];
    for attached in edge_faces.values() {
        let [a, b] = attached.as_slice() else {
            return false;
        };
        if *a >= face_count || *b >= face_count {
            return false;
        }
        if a != b {
            adjacency[*a].push(*b);
            adjacency[*b].push(*a);
        }
    }

    let mut seen = vec![false; face_count];
    let mut stack = vec![0usize];
    seen[0] = true;
    while let Some(face) = stack.pop() {
        for &neighbor in &adjacency[face] {
            if !seen[neighbor] {
                seen[neighbor] = true;
                stack.push(neighbor);
            }
        }
    }
    seen.into_iter().all(|connected| connected)
}

fn unique_neighbor_face(
    face_index: usize,
    edge_id: u64,
    edge_faces: &HashMap<u64, Vec<usize>>,
) -> Option<usize> {
    let neighbors = edge_faces
        .get(&edge_id)?
        .iter()
        .copied()
        .filter(|&index| index != face_index)
        .collect::<Vec<_>>();
    let [neighbor] = neighbors.as_slice() else {
        return None;
    };
    Some(*neighbor)
}

fn closed_profile_from_segments(mut segments: Vec<Segment2>) -> Option<Vec<[f64; 2]>> {
    if segments.len() < 3 {
        return None;
    }
    for segment in &mut segments {
        canonicalize_segment(segment);
        if distance2(segment.a, segment.b) <= GEOM_TOL_MM {
            return None;
        }
        let horizontal = (segment.a[1] - segment.b[1]).abs() <= GEOM_TOL_MM;
        let vertical = (segment.a[0] - segment.b[0]).abs() <= GEOM_TOL_MM;
        if !horizontal && !vertical {
            return None;
        }
        if segment.a[0] < -GEOM_TOL_MM || segment.b[0] < -GEOM_TOL_MM {
            return None;
        }
    }

    let (nodes, mut edges) = graph_from_segments(&segments)?;
    let degree_one = nodes
        .iter()
        .enumerate()
        .filter(|(index, _)| node_degree(*index, &edges) == 1)
        .map(|(index, point)| (index, *point))
        .collect::<Vec<_>>();
    if !degree_one.is_empty() {
        let [(first_index, first), (second_index, second)] = degree_one.as_slice() else {
            return None;
        };
        if first[0].abs() > GEOM_TOL_MM
            || second[0].abs() > GEOM_TOL_MM
            || (first[1] - second[1]).abs() <= GEOM_TOL_MM
        {
            return None;
        }
        edges.push((*first_index, *second_index));
    }

    if edges.len() != nodes.len()
        || nodes
            .iter()
            .enumerate()
            .any(|(index, _)| node_degree(index, &edges) != 2)
    {
        return None;
    }

    let start = (0..nodes.len()).min_by(|&a, &b| point_order(nodes[a], nodes[b]))?;
    let mut incident = edges
        .iter()
        .enumerate()
        .filter_map(|(edge_index, &(a, b))| {
            if a == start {
                Some((edge_index, b))
            } else if b == start {
                Some((edge_index, a))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    incident.sort_by(|(_, a), (_, b)| point_order(nodes[*a], nodes[*b]));
    let &(mut previous_edge, mut current) = incident.first()?;

    let mut order = vec![start];
    while current != start {
        if order.len() > nodes.len() {
            return None;
        }
        order.push(current);
        let next = edges.iter().enumerate().find_map(|(edge_index, &(a, b))| {
            if edge_index == previous_edge {
                None
            } else if a == current {
                Some((edge_index, b))
            } else if b == current {
                Some((edge_index, a))
            } else {
                None
            }
        })?;
        previous_edge = next.0;
        current = next.1;
    }
    if order.len() != nodes.len() {
        return None;
    }

    let mut points = order
        .into_iter()
        .map(|index| nodes[index])
        .collect::<Vec<_>>();
    if polygon_self_intersects(&points) {
        return None;
    }
    let area = signed_area(&points);
    if !area.is_finite() || area.abs() <= GEOM_TOL_MM * GEOM_TOL_MM {
        return None;
    }
    if area < 0.0 {
        points.reverse();
    }
    rotate_points_to_minimum(&mut points);
    for point in &mut points {
        if point[0].abs() <= GEOM_TOL_MM {
            point[0] = 0.0;
        }
    }
    Some(points)
}

fn graph_from_segments(segments: &[Segment2]) -> Option<ProfileGraph> {
    let mut nodes = Vec::<[f64; 2]>::new();
    let mut edges = Vec::<(usize, usize)>::new();
    for segment in segments {
        let a = intern_point(&mut nodes, segment.a);
        let b = intern_point(&mut nodes, segment.b);
        if a == b {
            return None;
        }
        if !edges
            .iter()
            .any(|&(x, y)| (x == a && y == b) || (x == b && y == a))
        {
            edges.push((a, b));
        }
    }
    Some((nodes, edges))
}

fn intern_point(nodes: &mut Vec<[f64; 2]>, point: [f64; 2]) -> usize {
    if let Some(index) = nodes
        .iter()
        .position(|existing| distance2(*existing, point) <= GEOM_TOL_MM)
    {
        index
    } else {
        nodes.push(point);
        nodes.len() - 1
    }
}

fn node_degree(node: usize, edges: &[(usize, usize)]) -> usize {
    edges
        .iter()
        .filter(|&&(a, b)| a == node || b == node)
        .count()
}

fn polygon_self_intersects(points: &[[f64; 2]]) -> bool {
    for first in 0..points.len() {
        let first_next = (first + 1) % points.len();
        for second in first + 1..points.len() {
            let second_next = (second + 1) % points.len();
            if first == second
                || first_next == second
                || second_next == first
                || (first == 0 && second_next == 0)
            {
                continue;
            }
            if axis_aligned_segments_intersect(
                points[first],
                points[first_next],
                points[second],
                points[second_next],
            ) {
                return true;
            }
        }
    }
    false
}

fn axis_aligned_segments_intersect(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let ab_horizontal = (a[1] - b[1]).abs() <= GEOM_TOL_MM;
    let cd_horizontal = (c[1] - d[1]).abs() <= GEOM_TOL_MM;
    if ab_horizontal && cd_horizontal {
        if (a[1] - c[1]).abs() > GEOM_TOL_MM {
            return false;
        }
        intervals_overlap(a[0], b[0], c[0], d[0])
    } else if !ab_horizontal && !cd_horizontal {
        if (a[0] - c[0]).abs() > GEOM_TOL_MM {
            return false;
        }
        intervals_overlap(a[1], b[1], c[1], d[1])
    } else {
        let (h0, h1, v0, v1) = if ab_horizontal {
            (a, b, c, d)
        } else {
            (c, d, a, b)
        };
        between(v0[0], h0[0], h1[0]) && between(h0[1], v0[1], v1[1])
    }
}

fn intervals_overlap(a: f64, b: f64, c: f64, d: f64) -> bool {
    let (a0, a1) = (a.min(b), a.max(b));
    let (c0, c1) = (c.min(d), c.max(d));
    a0 <= c1 + GEOM_TOL_MM && c0 <= a1 + GEOM_TOL_MM
}

fn between(value: f64, a: f64, b: f64) -> bool {
    value >= a.min(b) - GEOM_TOL_MM && value <= a.max(b) + GEOM_TOL_MM
}

fn push_unique_segment(segments: &mut Vec<Segment2>, mut candidate: Segment2) {
    canonicalize_segment(&mut candidate);
    if segments.iter().any(|existing| {
        distance2(existing.a, candidate.a) <= GEOM_TOL_MM
            && distance2(existing.b, candidate.b) <= GEOM_TOL_MM
    }) {
        return;
    }
    segments.push(candidate);
}

fn canonicalize_segment(segment: &mut Segment2) {
    if point_order(segment.b, segment.a).is_lt() {
        std::mem::swap(&mut segment.a, &mut segment.b);
    }
}

fn rotate_points_to_minimum(points: &mut [[f64; 2]]) {
    if let Some(index) = (0..points.len()).min_by(|&a, &b| point_order(points[a], points[b])) {
        points.rotate_left(index);
    }
}

fn point_order(a: [f64; 2], b: [f64; 2]) -> std::cmp::Ordering {
    a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1]))
}

fn signed_area(points: &[[f64; 2]]) -> f64 {
    0.5 * (0..points.len())
        .map(|index| {
            let next = (index + 1) % points.len();
            points[index][0] * points[next][1] - points[next][0] * points[index][1]
        })
        .sum::<f64>()
}

fn canonical_axis(axis: [f64; 3]) -> [f64; 3] {
    let index = (0..3)
        .max_by(|&a, &b| axis[a].abs().total_cmp(&axis[b].abs()))
        .unwrap_or(0);
    if axis[index] < 0.0 {
        mul(axis, -1.0)
    } else {
        axis
    }
}

fn closest_axis_point_to_global_origin(origin: [f64; 3], axis: [f64; 3]) -> [f64; 3] {
    sub(origin, mul(axis, dot(origin, axis)))
}

fn radial_basis(axis: [f64; 3]) -> Option<[f64; 3]> {
    let reference = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        .into_iter()
        .min_by(|a, b| dot(*a, axis).abs().total_cmp(&dot(*b, axis).abs()))?;
    normalize(sub(reference, mul(axis, dot(reference, axis))))
}

fn axial_coordinate(point: [f64; 3], axis_origin: [f64; 3], axis: [f64; 3]) -> f64 {
    dot(sub(point, axis_origin), axis)
}

fn point_axis_distance(point: [f64; 3], axis_origin: [f64; 3], axis: [f64; 3]) -> f64 {
    norm(cross(sub(point, axis_origin), axis))
}

fn axis_distance(origin: [f64; 3], axis_origin: [f64; 3], axis: [f64; 3]) -> f64 {
    point_axis_distance(origin, axis_origin, axis)
}

fn parallel(a: [f64; 3], b: [f64; 3]) -> bool {
    1.0 - dot(a, b).abs() <= DIR_TOL
}

fn distance2(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    let length = norm(vector);
    (length.is_finite() && length > DIR_TOL).then(|| mul(vector, 1.0 / length))
}

fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mul(a: [f64; 3], scalar: f64) -> [f64; 3] {
    [a[0] * scalar, a[1] * scalar, a[2] * scalar]
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
    fn shell_connectivity_rejects_disconnected_two_manifolds() {
        let connected = HashMap::from([(1, vec![0, 1]), (2, vec![1, 2]), (3, vec![2, 0])]);
        assert!(shell_faces_connected(3, &connected));

        let disconnected = HashMap::from([
            (1, vec![0, 1]),
            (2, vec![0, 1]),
            (3, vec![2, 3]),
            (4, vec![2, 3]),
        ]);
        assert!(!shell_faces_connected(4, &disconnected));
    }

    #[test]
    fn closes_axis_touching_step_profile() {
        let profile = closed_profile_from_segments(vec![
            Segment2 {
                a: [0.0, 0.0],
                b: [2.0, 0.0],
            },
            Segment2 {
                a: [2.0, 0.0],
                b: [2.0, 3.0],
            },
            Segment2 {
                a: [0.0, 3.0],
                b: [2.0, 3.0],
            },
        ])
        .unwrap();
        assert_eq!(profile.len(), 4);
        assert!(profile.iter().any(|point| *point == [0.0, 0.0]));
        assert!(profile.iter().any(|point| *point == [0.0, 3.0]));
        assert!(signed_area(&profile) > 0.0);
    }

    #[test]
    fn closes_hollow_step_profile_and_deduplicates_patches() {
        let mut segments = Vec::new();
        for _ in 0..4 {
            push_unique_segment(
                &mut segments,
                Segment2 {
                    a: [1.0, 0.0],
                    b: [3.0, 0.0],
                },
            );
            push_unique_segment(
                &mut segments,
                Segment2 {
                    a: [3.0, 0.0],
                    b: [3.0, 2.0],
                },
            );
            push_unique_segment(
                &mut segments,
                Segment2 {
                    a: [1.0, 2.0],
                    b: [3.0, 2.0],
                },
            );
            push_unique_segment(
                &mut segments,
                Segment2 {
                    a: [1.0, 0.0],
                    b: [1.0, 2.0],
                },
            );
        }
        assert_eq!(segments.len(), 4);
        let profile = closed_profile_from_segments(segments).unwrap();
        assert_eq!(profile.len(), 4);
        assert!(profile.iter().all(|point| point[0] >= 1.0));
    }

    #[test]
    fn rejects_branching_or_self_intersecting_profiles() {
        assert!(
            closed_profile_from_segments(vec![
                Segment2 {
                    a: [1.0, 0.0],
                    b: [3.0, 0.0]
                },
                Segment2 {
                    a: [3.0, 0.0],
                    b: [3.0, 2.0]
                },
                Segment2 {
                    a: [3.0, 2.0],
                    b: [1.0, 2.0]
                },
                Segment2 {
                    a: [1.0, 2.0],
                    b: [1.0, 0.0]
                },
                Segment2 {
                    a: [2.0, 0.0],
                    b: [2.0, 2.0]
                },
            ])
            .is_none()
        );
    }
}
