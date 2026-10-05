use super::{GEOM_TOL_MM, ProfileGraph, Segment2, angle_on_arc, distance2};
use crate::profile_curves::RecoveredProfileCurve;
use std::collections::HashMap;

pub(super) fn segment_radius_at_axial(segment: Segment2, axial: f64) -> Option<f64> {
    let delta = segment.b[1] - segment.a[1];
    if delta.abs() <= GEOM_TOL_MM || !between(axial, segment.a[1], segment.b[1]) {
        return None;
    }
    let fraction = (axial - segment.a[1]) / delta;
    let radius = segment.a[0] + fraction * (segment.b[0] - segment.a[0]);
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

pub(super) fn closed_profile_from_segments(mut segments: Vec<Segment2>) -> Option<Vec<[f64; 2]>> {
    if segments.len() < 3 {
        return None;
    }
    for segment in &mut segments {
        canonicalize_segment(segment);
        if distance2(segment.a, segment.b) <= GEOM_TOL_MM {
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

pub(super) fn closed_profile_from_curves(
    mut curves: Vec<RecoveredProfileCurve>,
) -> Option<Vec<RecoveredProfileCurve>> {
    if curves.len() < 3 || curves.iter().any(RecoveredProfileCurve::is_spline) {
        return None;
    }

    let mut nodes = Vec::<[f64; 2]>::new();
    let mut edges = Vec::<(usize, usize, usize)>::new();
    for (curve_index, curve) in curves.iter().enumerate() {
        let start = curve.start_point()?;
        let end = curve.end_point()?;
        if start[0] < -GEOM_TOL_MM || end[0] < -GEOM_TOL_MM || distance2(start, end) <= GEOM_TOL_MM
        {
            return None;
        }
        let a = intern_point(&mut nodes, start);
        let b = intern_point(&mut nodes, end);
        if a == b {
            return None;
        }
        edges.push((a, b, curve_index));
    }

    let degree_one = nodes
        .iter()
        .enumerate()
        .filter(|(index, _)| curve_node_degree(*index, &edges) == 1)
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
        let curve_index = curves.len();
        curves.push(RecoveredProfileCurve::Line {
            source_edge_ids: Vec::new(),
            start_mm: [0.0, first[1]],
            end_mm: [0.0, second[1]],
        });
        edges.push((*first_index, *second_index, curve_index));
    }

    if edges.len() != nodes.len()
        || nodes
            .iter()
            .enumerate()
            .any(|(index, _)| curve_node_degree(index, &edges) != 2)
    {
        return None;
    }

    let start = (0..nodes.len()).min_by(|&a, &b| point_order(nodes[a], nodes[b]))?;
    let mut incident = edges
        .iter()
        .enumerate()
        .filter_map(|(edge_index, &(a, b, _))| {
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
    let &(first_edge, _) = incident.first()?;

    let mut used = vec![false; edges.len()];
    let mut ordered = Vec::<RecoveredProfileCurve>::with_capacity(edges.len());
    let mut current = start;
    let mut edge_index = first_edge;
    loop {
        if used[edge_index] {
            return None;
        }
        used[edge_index] = true;
        let (a, b, curve_index) = edges[edge_index];
        let (next, curve) = if a == current {
            (b, curves[curve_index].clone())
        } else if b == current {
            (a, curves[curve_index].reversed())
        } else {
            return None;
        };
        ordered.push(curve);
        current = next;
        if current == start {
            break;
        }
        edge_index = edges
            .iter()
            .enumerate()
            .find_map(|(candidate, &(ea, eb, _))| {
                (!used[candidate] && (ea == current || eb == current)).then_some(candidate)
            })?;
        if ordered.len() > edges.len() {
            return None;
        }
    }
    if used.iter().any(|used| !used) || mixed_profile_self_intersects(&ordered) {
        return None;
    }

    let area = signed_curve_area(&ordered)?;
    if !area.is_finite() || area.abs() <= GEOM_TOL_MM * GEOM_TOL_MM {
        return None;
    }
    if area < 0.0 {
        ordered = ordered
            .into_iter()
            .rev()
            .map(|curve| curve.reversed())
            .collect();
    }
    rotate_curves_to_minimum(&mut ordered)?;
    Some(ordered)
}

fn curve_node_degree(node: usize, edges: &[(usize, usize, usize)]) -> usize {
    edges
        .iter()
        .filter(|&&(a, b, _)| a == node || b == node)
        .count()
}

fn rotate_curves_to_minimum(curves: &mut [RecoveredProfileCurve]) -> Option<()> {
    let index = (0..curves.len()).min_by(|&a, &b| {
        let a = curves[a].start_point().unwrap_or([f64::INFINITY; 2]);
        let b = curves[b].start_point().unwrap_or([f64::INFINITY; 2]);
        point_order(a, b)
    })?;
    curves.rotate_left(index);
    Some(())
}

fn mixed_profile_self_intersects(curves: &[RecoveredProfileCurve]) -> bool {
    for first in 0..curves.len() {
        for second in first + 1..curves.len() {
            let adjacent = second == first + 1 || (first == 0 && second + 1 == curves.len());
            match (&curves[first], &curves[second]) {
                (
                    RecoveredProfileCurve::Line {
                        start_mm: a,
                        end_mm: b,
                        ..
                    },
                    RecoveredProfileCurve::Line {
                        start_mm: c,
                        end_mm: d,
                        ..
                    },
                ) => {
                    if line_pair_has_extra_intersection(*a, *b, *c, *d, adjacent) {
                        return true;
                    }
                }
                (RecoveredProfileCurve::Line { .. }, RecoveredProfileCurve::CircleArc { .. }) => {
                    if line_arc_has_extra_intersection(&curves[first], &curves[second], adjacent) {
                        return true;
                    }
                }
                (RecoveredProfileCurve::CircleArc { .. }, RecoveredProfileCurve::Line { .. }) => {
                    if line_arc_has_extra_intersection(&curves[second], &curves[first], adjacent) {
                        return true;
                    }
                }
                (
                    RecoveredProfileCurve::CircleArc { .. },
                    RecoveredProfileCurve::CircleArc { .. },
                ) => {
                    if arc_pair_has_extra_intersection(&curves[first], &curves[second], adjacent) {
                        return true;
                    }
                }
                _ => return true,
            }
        }
    }
    false
}

fn line_pair_has_extra_intersection(
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    d: [f64; 2],
    adjacent: bool,
) -> bool {
    if !segments_intersect(a, b, c, d) {
        return false;
    }
    if !adjacent {
        return true;
    }

    let shared = [a, b]
        .into_iter()
        .find(|point| distance2(*point, c) <= GEOM_TOL_MM || distance2(*point, d) <= GEOM_TOL_MM);
    let Some(shared) = shared else {
        return true;
    };
    let other_ab = if distance2(shared, a) <= GEOM_TOL_MM {
        b
    } else {
        a
    };
    let other_cd = if distance2(shared, c) <= GEOM_TOL_MM {
        d
    } else {
        c
    };
    let u = sub2(other_ab, shared);
    let v = sub2(other_cd, shared);
    let scale = (distance2(shared, other_ab) * distance2(shared, other_cd)).max(GEOM_TOL_MM);
    cross2(u, v).abs() <= GEOM_TOL_MM * scale && dot2(u, v) > GEOM_TOL_MM.powi(2)
}

fn line_arc_has_extra_intersection(
    line: &RecoveredProfileCurve,
    arc: &RecoveredProfileCurve,
    adjacent: bool,
) -> bool {
    let RecoveredProfileCurve::Line {
        start_mm, end_mm, ..
    } = line
    else {
        return true;
    };
    let RecoveredProfileCurve::CircleArc {
        center_mm,
        radius_mm,
        start_angle_rad,
        end_angle_rad,
        ..
    } = arc
    else {
        return true;
    };

    for point in line_arc_intersections(
        *start_mm,
        *end_mm,
        *center_mm,
        *radius_mm,
        *start_angle_rad,
        *end_angle_rad,
    ) {
        if !adjacent || !curve_shared_endpoint_at(line, arc, point) {
            return true;
        }
    }
    false
}

fn arc_pair_has_extra_intersection(
    first: &RecoveredProfileCurve,
    second: &RecoveredProfileCurve,
    adjacent: bool,
) -> bool {
    let (
        RecoveredProfileCurve::CircleArc {
            center_mm: first_center,
            radius_mm: first_radius,
            start_angle_rad: first_start,
            end_angle_rad: first_end,
            ..
        },
        RecoveredProfileCurve::CircleArc {
            center_mm: second_center,
            radius_mm: second_radius,
            start_angle_rad: second_start,
            end_angle_rad: second_end,
            ..
        },
    ) = (first, second)
    else {
        return true;
    };
    if !first_radius.is_finite()
        || !second_radius.is_finite()
        || *first_radius <= GEOM_TOL_MM
        || *second_radius <= GEOM_TOL_MM
    {
        return true;
    }

    let center_distance = distance2(*first_center, *second_center);
    if center_distance <= GEOM_TOL_MM && (*first_radius - *second_radius).abs() <= GEOM_TOL_MM {
        let mut shared_endpoints = Vec::<[f64; 2]>::new();
        for point in [
            first.start_point(),
            first.end_point(),
            second.start_point(),
            second.end_point(),
        ]
        .into_iter()
        .flatten()
        {
            if point_on_arc(
                point,
                *first_center,
                *first_radius,
                *first_start,
                *first_end,
            ) && point_on_arc(
                point,
                *second_center,
                *second_radius,
                *second_start,
                *second_end,
            ) && !shared_endpoints
                .iter()
                .any(|existing| distance2(*existing, point) <= GEOM_TOL_MM)
            {
                shared_endpoints.push(point);
            }
        }
        return !adjacent
            || shared_endpoints.len() != 1
            || !curve_shared_endpoint_at(first, second, shared_endpoints[0]);
    }

    for point in
        circle_circle_intersections(*first_center, *first_radius, *second_center, *second_radius)
    {
        if !point_on_arc(
            point,
            *first_center,
            *first_radius,
            *first_start,
            *first_end,
        ) || !point_on_arc(
            point,
            *second_center,
            *second_radius,
            *second_start,
            *second_end,
        ) {
            continue;
        }
        if !adjacent || !curve_shared_endpoint_at(first, second, point) {
            return true;
        }
    }
    false
}

fn circle_circle_intersections(
    first_center: [f64; 2],
    first_radius: f64,
    second_center: [f64; 2],
    second_radius: f64,
) -> Vec<[f64; 2]> {
    let delta = sub2(second_center, first_center);
    let distance = norm2(delta);
    if !distance.is_finite()
        || distance <= GEOM_TOL_MM
        || distance > first_radius + second_radius + GEOM_TOL_MM
        || distance < (first_radius - second_radius).abs() - GEOM_TOL_MM
    {
        return Vec::new();
    }

    let along =
        (first_radius.powi(2) - second_radius.powi(2) + distance.powi(2)) / (2.0 * distance);
    let height_sq = first_radius.powi(2) - along.powi(2);
    let scale = first_radius
        .abs()
        .max(second_radius.abs())
        .max(distance)
        .max(1.0);
    let height_tol = GEOM_TOL_MM * scale;
    if height_sq < -height_tol {
        return Vec::new();
    }

    let unit = mul2(delta, 1.0 / distance);
    let base = add2(first_center, mul2(unit, along));
    let height = height_sq.max(0.0).sqrt();
    let perpendicular = [-unit[1], unit[0]];
    let first = add2(base, mul2(perpendicular, height));
    if height <= GEOM_TOL_MM {
        return vec![first];
    }
    let second = add2(base, mul2(perpendicular, -height));
    vec![first, second]
}

fn line_arc_intersections(
    start: [f64; 2],
    end: [f64; 2],
    center: [f64; 2],
    radius: f64,
    arc_start: f64,
    arc_end: f64,
) -> Vec<[f64; 2]> {
    let direction = sub2(end, start);
    let offset = sub2(start, center);
    let a = dot2(direction, direction);
    if a <= GEOM_TOL_MM.powi(2) {
        return Vec::new();
    }
    let b = 2.0 * dot2(offset, direction);
    let c = dot2(offset, offset) - radius.powi(2);
    let discriminant = b * b - 4.0 * a * c;
    let discriminant_tol = GEOM_TOL_MM.powi(2) * (b.abs().max((4.0 * a * c).abs()).max(1.0));
    if discriminant < -discriminant_tol {
        return Vec::new();
    }

    let sqrt_discriminant = discriminant.max(0.0).sqrt();
    let mut out = Vec::new();
    for numerator in [-b - sqrt_discriminant, -b + sqrt_discriminant] {
        let t = numerator / (2.0 * a);
        let parameter_tol = GEOM_TOL_MM / a.sqrt();
        if t < -parameter_tol || t > 1.0 + parameter_tol {
            continue;
        }
        let point = [start[0] + t * direction[0], start[1] + t * direction[1]];
        if point_on_arc(point, center, radius, arc_start, arc_end)
            && !out
                .iter()
                .any(|existing| distance2(*existing, point) <= GEOM_TOL_MM)
        {
            out.push(point);
        }
    }
    out
}

fn point_on_arc(point: [f64; 2], center: [f64; 2], radius: f64, start: f64, end: f64) -> bool {
    if (distance2(point, center) - radius).abs() > GEOM_TOL_MM {
        return false;
    }
    let angle = (point[1] - center[1]).atan2(point[0] - center[0]);
    angle_on_arc(
        angle,
        start,
        end,
        (GEOM_TOL_MM / radius.max(GEOM_TOL_MM)).max(1.0e-12),
    )
}

fn curve_shared_endpoint_at(
    a: &RecoveredProfileCurve,
    b: &RecoveredProfileCurve,
    point: [f64; 2],
) -> bool {
    let Some(a_start) = a.start_point() else {
        return false;
    };
    let Some(a_end) = a.end_point() else {
        return false;
    };
    let Some(b_start) = b.start_point() else {
        return false;
    };
    let Some(b_end) = b.end_point() else {
        return false;
    };
    (distance2(point, a_start) <= GEOM_TOL_MM || distance2(point, a_end) <= GEOM_TOL_MM)
        && (distance2(point, b_start) <= GEOM_TOL_MM || distance2(point, b_end) <= GEOM_TOL_MM)
}

fn add2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn mul2(a: [f64; 2], factor: f64) -> [f64; 2] {
    [a[0] * factor, a[1] * factor]
}

fn norm2(a: [f64; 2]) -> f64 {
    dot2(a, a).sqrt()
}

fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn signed_curve_area(curves: &[RecoveredProfileCurve]) -> Option<f64> {
    let mut twice_area = 0.0_f64;
    for curve in curves {
        match curve {
            RecoveredProfileCurve::Line {
                start_mm, end_mm, ..
            } => {
                twice_area += start_mm[0] * end_mm[1] - end_mm[0] * start_mm[1];
            }
            RecoveredProfileCurve::CircleArc {
                center_mm,
                radius_mm,
                start_angle_rad,
                end_angle_rad,
                ..
            } => {
                let a = *start_angle_rad;
                let b = *end_angle_rad;
                twice_area += radius_mm * center_mm[0] * (b.sin() - a.sin())
                    + radius_mm * center_mm[1] * (a.cos() - b.cos())
                    + radius_mm.powi(2) * (b - a);
            }
            _ => return None,
        }
    }
    Some(0.5 * twice_area)
}

pub(super) fn line_profile_curves(points: &[[f64; 2]]) -> Vec<RecoveredProfileCurve> {
    (0..points.len())
        .map(|index| RecoveredProfileCurve::Line {
            source_edge_ids: Vec::new(),
            start_mm: points[index],
            end_mm: points[(index + 1) % points.len()],
        })
        .collect()
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
            if segments_intersect(
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

fn segments_intersect(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let ab = sub2(b, a);
    let cd = sub2(d, c);
    let scale = (distance2(a, b) + distance2(c, d)).max(GEOM_TOL_MM);
    let epsilon = GEOM_TOL_MM * scale;
    let o1 = cross2(ab, sub2(c, a));
    let o2 = cross2(ab, sub2(d, a));
    let o3 = cross2(cd, sub2(a, c));
    let o4 = cross2(cd, sub2(b, c));

    if o1.abs() <= epsilon && point_on_segment(c, a, b) {
        return true;
    }
    if o2.abs() <= epsilon && point_on_segment(d, a, b) {
        return true;
    }
    if o3.abs() <= epsilon && point_on_segment(a, c, d) {
        return true;
    }
    if o4.abs() <= epsilon && point_on_segment(b, c, d) {
        return true;
    }
    ((o1 > epsilon && o2 < -epsilon) || (o1 < -epsilon && o2 > epsilon))
        && ((o3 > epsilon && o4 < -epsilon) || (o3 < -epsilon && o4 > epsilon))
}

fn point_on_segment(point: [f64; 2], a: [f64; 2], b: [f64; 2]) -> bool {
    between(point[0], a[0], b[0]) && between(point[1], a[1], b[1])
}

pub(super) fn between(value: f64, a: f64, b: f64) -> bool {
    value >= a.min(b) - GEOM_TOL_MM && value <= a.max(b) + GEOM_TOL_MM
}

pub(super) fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn cross2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
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
