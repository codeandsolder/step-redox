use super::{GEOM_TOL_MM, Segment2, distance2};
use std::collections::HashMap;

pub(super) fn segment_radius_at_axial(segment: Segment2, axial: f64) -> Option<f64> {
    let delta = segment.b[1] - segment.a[1];
    if delta.abs() <= GEOM_TOL_MM || !between(axial, segment.a[1], segment.b[1]) {
        return None;
    }
    let fraction = (axial - segment.a[1]) / delta;
    let radius = fraction.mul_add(segment.b[0] - segment.a[0], segment.a[0]);
    (radius.is_finite() && radius >= -GEOM_TOL_MM).then_some(radius.max(0.0))
}

pub(super) fn shell_faces_connected(
    face_count: usize,
    edge_faces: &HashMap<u64, Vec<usize>>,
) -> bool {
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

pub(super) fn unique_neighbor_face(
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

pub(super) fn between(value: f64, a: f64, b: f64) -> bool {
    value >= a.min(b) - GEOM_TOL_MM && value <= a.max(b) + GEOM_TOL_MM
}

pub(super) fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

pub(super) fn push_unique_segment(segments: &mut Vec<Segment2>, mut candidate: Segment2) {
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

fn point_order(a: [f64; 2], b: [f64; 2]) -> std::cmp::Ordering {
    a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1]))
}
