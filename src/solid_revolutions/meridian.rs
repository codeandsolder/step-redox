use crate::math2::{cross, distance, dot, sub};
use crate::profile_curves::RecoveredProfileCurve;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct MeridianLine {
    pub start: [f64; 2],
    pub end: [f64; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct MeridianArc {
    pub source_edge_ids: Vec<u64>,
    pub center: [f64; 2],
    pub radius: f64,
    pub start_angle: f64,
    pub end_angle: f64,
}

impl MeridianArc {
    fn point_at(&self, angle: f64) -> [f64; 2] {
        [
            self.radius.mul_add(angle.cos(), self.center[0]),
            self.radius.mul_add(angle.sin(), self.center[1]),
        ]
    }

    fn endpoints(&self) -> ([f64; 2], [f64; 2]) {
        (
            self.point_at(self.start_angle),
            self.point_at(self.end_angle),
        )
    }

    fn angle_tolerance(&self, tolerance: f64) -> f64 {
        (tolerance / self.radius.max(tolerance)).max(1.0e-12)
    }

    fn contains_point(&self, point: [f64; 2], tolerance: f64) -> bool {
        angle(point, self.center, self.radius, tolerance).is_some_and(|point_angle| {
            angle_on_arc(
                point_angle,
                self.start_angle,
                self.end_angle,
                self.angle_tolerance(tolerance),
            )
        })
    }

    fn has_endpoint(&self, point: [f64; 2], tolerance: f64) -> bool {
        let (start, end) = self.endpoints();
        distance(point, start) <= tolerance || distance(point, end) <= tolerance
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum MeridianCurve {
    Line(MeridianLine),
    Arc(MeridianArc),
}

impl MeridianCurve {
    #[cfg(test)]
    pub fn from_recovered(curve: &RecoveredProfileCurve) -> Option<Self> {
        match curve {
            RecoveredProfileCurve::Line {
                start_mm, end_mm, ..
            } => Some(Self::Line(MeridianLine {
                start: *start_mm,
                end: *end_mm,
            })),
            RecoveredProfileCurve::CircleArc {
                source_edge_ids,
                center_mm,
                radius_mm,
                start_angle_rad,
                end_angle_rad,
            } => Some(Self::Arc(MeridianArc {
                source_edge_ids: source_edge_ids.clone(),
                center: *center_mm,
                radius: *radius_mm,
                start_angle: *start_angle_rad,
                end_angle: *end_angle_rad,
            })),
            RecoveredProfileCurve::Bezier { .. } | RecoveredProfileCurve::BSpline { .. } => None,
        }
    }

    pub fn endpoints(&self) -> ([f64; 2], [f64; 2]) {
        match self {
            Self::Line(line) => (line.start, line.end),
            Self::Arc(arc) => arc.endpoints(),
        }
    }

    pub const fn reversed(mut self) -> Self {
        match &mut self {
            Self::Line(line) => std::mem::swap(&mut line.start, &mut line.end),
            Self::Arc(arc) => std::mem::swap(&mut arc.start_angle, &mut arc.end_angle),
        }
        self
    }

    fn is_degenerate(&self, tolerance: f64) -> bool {
        match self {
            Self::Line(line) => distance(line.start, line.end) <= tolerance,
            Self::Arc(arc) => {
                !arc.radius.is_finite()
                    || arc.radius <= tolerance
                    || (arc.end_angle - arc.start_angle).abs()
                        <= (tolerance / arc.radius.max(tolerance)).max(1.0e-12)
            }
        }
    }

    pub fn min_radius(&self, tolerance: f64) -> f64 {
        match self {
            Self::Line(line) => line.start[0].min(line.end[0]),
            Self::Arc(arc) => {
                let (start, end) = arc.endpoints();
                let mut minimum = start[0].min(end[0]);
                if angle_on_arc(
                    std::f64::consts::PI,
                    arc.start_angle,
                    arc.end_angle,
                    arc.angle_tolerance(tolerance),
                ) {
                    minimum = minimum.min(arc.center[0] - arc.radius);
                }
                minimum
            }
        }
    }

    pub fn into_recovered(self) -> RecoveredProfileCurve {
        match self {
            Self::Line(line) => RecoveredProfileCurve::Line {
                source_edge_ids: Vec::new(),
                start_mm: line.start,
                end_mm: line.end,
            },
            Self::Arc(arc) => RecoveredProfileCurve::CircleArc {
                source_edge_ids: arc.source_edge_ids,
                center_mm: arc.center,
                radius_mm: arc.radius,
                start_angle_rad: arc.start_angle,
                end_angle_rad: arc.end_angle,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct MeridianProfile {
    curves: Vec<MeridianCurve>,
}

impl MeridianProfile {
    #[cfg(test)]
    pub fn from_recovered(curves: &[RecoveredProfileCurve], tolerance: f64) -> Option<Self> {
        Self::closed(
            curves
                .iter()
                .map(MeridianCurve::from_recovered)
                .collect::<Option<Vec<_>>>()?,
            tolerance,
        )
    }

    pub fn from_lines(
        lines: impl IntoIterator<Item = MeridianLine>,
        tolerance: f64,
    ) -> Option<Self> {
        Self::closed(
            lines.into_iter().map(MeridianCurve::Line).collect(),
            tolerance,
        )
    }

    pub fn closed(mut curves: Vec<MeridianCurve>, tolerance: f64) -> Option<Self> {
        for curve in &mut curves {
            if let MeridianCurve::Line(line) = curve {
                for point in [&mut line.start, &mut line.end] {
                    if point[0].abs() <= tolerance {
                        point[0] = 0.0;
                    }
                }
            }
        }

        if curves.is_empty()
            || curves.iter().any(|curve| {
                curve.min_radius(tolerance) < -tolerance || curve.is_degenerate(tolerance)
            })
        {
            return None;
        }

        if curves.len() == 1 {
            let (start, end) = curves[0].endpoints();
            if distance(start, end) > tolerance || profile_self_intersects(&curves, tolerance) {
                return None;
            }
            canonicalize_orientation(&mut curves, tolerance)?;
            return Some(Self { curves });
        }

        let mut ordered = order_closed_curves(curves, tolerance)?;
        if profile_self_intersects(&ordered, tolerance) {
            return None;
        }
        canonicalize_orientation(&mut ordered, tolerance)?;
        Some(Self { curves: ordered })
    }

    pub fn into_recovered(self) -> Vec<RecoveredProfileCurve> {
        self.curves
            .into_iter()
            .map(MeridianCurve::into_recovered)
            .collect()
    }
}

fn order_closed_curves(curves: Vec<MeridianCurve>, tolerance: f64) -> Option<Vec<MeridianCurve>> {
    let mut nodes = Vec::<[f64; 2]>::new();
    let mut edges = Vec::<(usize, usize, MeridianCurve)>::with_capacity(curves.len() + 1);
    for curve in curves {
        let (start, end) = curve.endpoints();
        let first_node = intern_point(&mut nodes, start, tolerance);
        let second_node = intern_point(&mut nodes, end, tolerance);
        if first_node == second_node {
            return None;
        }
        edges.push((first_node, second_node, curve));
    }

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
        if first[0].abs() > tolerance
            || second[0].abs() > tolerance
            || (first[1] - second[1]).abs() <= tolerance
        {
            return None;
        }
        edges.push((
            *first_index,
            *second_index,
            MeridianCurve::Line(MeridianLine {
                start: [0.0, first[1]],
                end: [0.0, second[1]],
            }),
        ));
    }

    if edges.len() != nodes.len()
        || nodes
            .iter()
            .enumerate()
            .any(|(index, _)| node_degree(index, &edges) != 2)
    {
        return None;
    }

    let start_node = (0..nodes.len()).min_by(|&a, &b| point_order(nodes[a], nodes[b]))?;
    let first_edge = edges
        .iter()
        .enumerate()
        .filter_map(|(edge_index, (first, second, _))| {
            if *first == start_node {
                Some((edge_index, *second))
            } else if *second == start_node {
                Some((edge_index, *first))
            } else {
                None
            }
        })
        .min_by(|(_, a), (_, b)| point_order(nodes[*a], nodes[*b]))?
        .0;

    let mut used = vec![false; edges.len()];
    let mut ordered = Vec::with_capacity(edges.len());
    let mut current = start_node;
    let mut edge_index = first_edge;
    loop {
        if used[edge_index] {
            return None;
        }
        used[edge_index] = true;
        let (first, second, curve) = &edges[edge_index];
        let (next, curve) = if *first == current {
            (*second, curve.clone())
        } else if *second == current {
            (*first, curve.clone().reversed())
        } else {
            return None;
        };
        ordered.push(curve);
        current = next;
        if current == start_node {
            break;
        }
        edge_index = edges
            .iter()
            .enumerate()
            .find_map(|(candidate, (first, second, _))| {
                (!used[candidate] && (*first == current || *second == current)).then_some(candidate)
            })?;
        if ordered.len() > edges.len() {
            return None;
        }
    }
    used.into_iter().all(|used| used).then_some(ordered)
}

pub(super) fn angle(point: [f64; 2], center: [f64; 2], radius: f64, tolerance: f64) -> Option<f64> {
    if !point[0].is_finite()
        || !point[1].is_finite()
        || !radius.is_finite()
        || radius <= tolerance
        || (distance(point, center) - radius).abs() > tolerance
    {
        return None;
    }
    Some((point[1] - center[1]).atan2(point[0] - center[0]))
}

pub(super) fn positive_angle_delta(start: f64, end: f64) -> f64 {
    (end - start).rem_euclid(std::f64::consts::TAU)
}

pub(super) fn angle_on_arc(angle: f64, start: f64, end: f64, tolerance: f64) -> bool {
    if end >= start {
        positive_angle_delta(start, angle) <= end - start + tolerance
    } else {
        positive_angle_delta(angle, start) <= start - end + tolerance
    }
}

fn intern_point(nodes: &mut Vec<[f64; 2]>, point: [f64; 2], tolerance: f64) -> usize {
    let existing = nodes
        .iter()
        .position(|candidate| distance(*candidate, point) <= tolerance);
    existing.unwrap_or_else(|| {
        nodes.push(point);
        nodes.len() - 1
    })
}

fn node_degree(node: usize, edges: &[(usize, usize, MeridianCurve)]) -> usize {
    edges
        .iter()
        .filter(|(a, b, _)| *a == node || *b == node)
        .count()
}

fn canonicalize_orientation(curves: &mut Vec<MeridianCurve>, tolerance: f64) -> Option<()> {
    let area = signed_area(curves);
    if !area.is_finite() || area.abs() <= tolerance * tolerance {
        return None;
    }
    if area < 0.0 {
        *curves = curves
            .drain(..)
            .rev()
            .map(MeridianCurve::reversed)
            .collect();
    }
    if curves.len() > 1 {
        let index = (0..curves.len()).min_by(|&a, &b| {
            let (a, _) = curves[a].endpoints();
            let (b, _) = curves[b].endpoints();
            point_order(a, b)
        })?;
        curves.rotate_left(index);
    }
    Some(())
}

fn signed_area(curves: &[MeridianCurve]) -> f64 {
    let mut twice_area = 0.0_f64;
    for curve in curves {
        match curve {
            MeridianCurve::Line(line) => {
                twice_area += cross(line.start, line.end);
            }
            MeridianCurve::Arc(arc) => {
                let start = arc.start_angle;
                let end = arc.end_angle;
                let center_term = arc.center[1].mul_add(
                    start.cos() - end.cos(),
                    arc.center[0] * (end.sin() - start.sin()),
                );
                let integrand = arc.radius.mul_add(end - start, center_term);
                twice_area = arc.radius.mul_add(integrand, twice_area);
            }
        }
    }
    0.5 * twice_area
}

fn point_order(a: [f64; 2], b: [f64; 2]) -> std::cmp::Ordering {
    a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1]))
}

fn profile_self_intersects(curves: &[MeridianCurve], tolerance: f64) -> bool {
    for first in 0..curves.len() {
        for second in (first + 1)..curves.len() {
            let adjacent = second == first + 1 || (first == 0 && second + 1 == curves.len());
            if curves_intersect_away_from_shared_endpoint(
                &curves[first],
                &curves[second],
                adjacent,
                tolerance,
            ) {
                return true;
            }
        }
    }
    false
}

pub(super) fn curves_intersect_away_from_shared_endpoint(
    first: &MeridianCurve,
    second: &MeridianCurve,
    adjacent: bool,
    tolerance: f64,
) -> bool {
    match (first, second) {
        (MeridianCurve::Line(a), MeridianCurve::Line(b)) => {
            line_line_has_extra_intersection(*a, *b, adjacent, tolerance)
        }
        (MeridianCurve::Line(line), MeridianCurve::Arc(arc))
        | (MeridianCurve::Arc(arc), MeridianCurve::Line(line)) => {
            line_arc_has_extra_intersection(*line, arc, adjacent, tolerance)
        }
        (MeridianCurve::Arc(a), MeridianCurve::Arc(b)) => {
            arc_arc_has_extra_intersection(a, b, adjacent, tolerance)
        }
    }
}

fn line_line_has_extra_intersection(
    first: MeridianLine,
    second: MeridianLine,
    adjacent: bool,
    tolerance: f64,
) -> bool {
    let first_direction = sub(first.end, first.start);
    let second_direction = sub(second.end, second.start);
    let first_length = distance(first.start, first.end);
    let second_length = distance(second.start, second.end);
    if first_length <= tolerance || second_length <= tolerance {
        return true;
    }

    let delta = sub(second.start, first.start);
    let denominator = cross(first_direction, second_direction);
    let cross_tolerance = tolerance * (first_length + second_length).max(tolerance);
    let first_parameter_tolerance = tolerance / first_length;
    let second_parameter_tolerance = tolerance / second_length;

    if denominator.abs() <= cross_tolerance {
        if cross(delta, first_direction).abs() > tolerance * first_length.max(tolerance) {
            return false;
        }

        let first_length_squared = dot(first_direction, first_direction);
        let second_start_parameter = dot(delta, first_direction) / first_length_squared;
        let second_end_parameter =
            second_start_parameter + dot(second_direction, first_direction) / first_length_squared;
        let overlap_start = 0.0_f64.max(second_start_parameter.min(second_end_parameter));
        let overlap_end = 1.0_f64.min(second_start_parameter.max(second_end_parameter));
        if overlap_end < overlap_start - first_parameter_tolerance {
            return false;
        }
        if !adjacent {
            return true;
        }

        let overlap_length = (overlap_end - overlap_start).max(0.0) * first_length;
        let shared_endpoint = [first.start, first.end].into_iter().any(|first_point| {
            [second.start, second.end]
                .into_iter()
                .any(|second_point| distance(first_point, second_point) <= tolerance)
        });
        return !(shared_endpoint && overlap_length <= tolerance);
    }

    let first_parameter = cross(delta, second_direction) / denominator;
    let second_parameter = cross(delta, first_direction) / denominator;
    if first_parameter < -first_parameter_tolerance
        || first_parameter > 1.0 + first_parameter_tolerance
        || second_parameter < -second_parameter_tolerance
        || second_parameter > 1.0 + second_parameter_tolerance
    {
        return false;
    }
    if !adjacent {
        return true;
    }

    let first_endpoint = first_parameter <= first_parameter_tolerance
        || first_parameter >= 1.0 - first_parameter_tolerance;
    let second_endpoint = second_parameter <= second_parameter_tolerance
        || second_parameter >= 1.0 - second_parameter_tolerance;
    !(first_endpoint && second_endpoint)
}

fn line_arc_has_extra_intersection(
    line: MeridianLine,
    arc: &MeridianArc,
    adjacent: bool,
    tolerance: f64,
) -> bool {
    if !arc.radius.is_finite() || arc.radius <= tolerance {
        return true;
    }

    let direction = sub(line.end, line.start);
    let line_length = distance(line.start, line.end);
    if line_length <= tolerance {
        return true;
    }
    let line_length_squared = dot(direction, direction);
    let center_relative = sub(arc.center, line.start);
    let center_parameter = dot(center_relative, direction) / line_length_squared;
    let closest = [
        center_parameter.mul_add(direction[0], line.start[0]),
        center_parameter.mul_add(direction[1], line.start[1]),
    ];
    let perpendicular_distance = distance(closest, arc.center);
    if perpendicular_distance > arc.radius + tolerance {
        return false;
    }

    let half_chord_squared = perpendicular_distance
        .mul_add(-perpendicular_distance, arc.radius.powi(2))
        .max(0.0);
    let half_chord = half_chord_squared.sqrt();
    let parameter_offset = half_chord / line_length;
    let parameter_tolerance = tolerance / line_length;
    let parameters = [
        center_parameter - parameter_offset,
        center_parameter + parameter_offset,
    ];

    parameters
        .into_iter()
        .enumerate()
        .any(|(index, parameter)| {
            if index == 1 && half_chord <= tolerance {
                return false;
            }
            if parameter < -parameter_tolerance || parameter > 1.0 + parameter_tolerance {
                return false;
            }
            let point = [
                parameter.mul_add(direction[0], line.start[0]),
                parameter.mul_add(direction[1], line.start[1]),
            ];
            if !arc.contains_point(point, tolerance) {
                return false;
            }
            if !adjacent {
                return true;
            }
            let line_endpoint =
                parameter <= parameter_tolerance || parameter >= 1.0 - parameter_tolerance;
            !(line_endpoint && arc.has_endpoint(point, tolerance))
        })
}

fn arc_ccw_intervals(arc: &MeridianArc, tolerance: f64) -> Vec<(f64, f64)> {
    let sweep = arc.end_angle - arc.start_angle;
    let angle_tolerance = arc.angle_tolerance(tolerance);
    if sweep.abs() >= std::f64::consts::TAU - angle_tolerance {
        return vec![(0.0, std::f64::consts::TAU)];
    }

    let (start, length) = if sweep >= 0.0 {
        (arc.start_angle.rem_euclid(std::f64::consts::TAU), sweep)
    } else {
        (arc.end_angle.rem_euclid(std::f64::consts::TAU), -sweep)
    };
    let end = start + length;
    if end <= std::f64::consts::TAU {
        vec![(start, end)]
    } else {
        vec![
            (start, std::f64::consts::TAU),
            (0.0, end - std::f64::consts::TAU),
        ]
    }
}

fn cocircular_arcs_share_interior(
    first: &MeridianArc,
    second: &MeridianArc,
    tolerance: f64,
) -> bool {
    let angle_tolerance = first
        .angle_tolerance(tolerance)
        .max(second.angle_tolerance(tolerance));
    arc_ccw_intervals(first, tolerance)
        .into_iter()
        .any(|first_interval| {
            arc_ccw_intervals(second, tolerance)
                .into_iter()
                .any(|second_interval| {
                    first_interval.1.min(second_interval.1)
                        - first_interval.0.max(second_interval.0)
                        > angle_tolerance
                })
        })
}

fn arcs_share_endpoint(first: &MeridianArc, second: &MeridianArc, tolerance: f64) -> bool {
    let first_endpoints: [[f64; 2]; 2] = first.endpoints().into();
    let second_endpoints: [[f64; 2]; 2] = second.endpoints().into();
    first_endpoints.into_iter().any(|first_point| {
        second_endpoints
            .into_iter()
            .any(|second_point| distance(first_point, second_point) <= tolerance)
    })
}

fn arc_arc_has_extra_intersection(
    first: &MeridianArc,
    second: &MeridianArc,
    adjacent: bool,
    tolerance: f64,
) -> bool {
    if !first.radius.is_finite()
        || !second.radius.is_finite()
        || first.radius <= tolerance
        || second.radius <= tolerance
    {
        return true;
    }

    let center_delta = sub(second.center, first.center);
    let center_distance = center_delta[0].hypot(center_delta[1]);
    if center_distance <= tolerance {
        if (first.radius - second.radius).abs() > tolerance {
            return false;
        }
        if cocircular_arcs_share_interior(first, second, tolerance) {
            return true;
        }
        return !adjacent && arcs_share_endpoint(first, second, tolerance);
    }

    if center_distance > first.radius + second.radius + tolerance
        || center_distance < (first.radius - second.radius).abs() - tolerance
    {
        return false;
    }

    let radius_square_difference = second.radius.mul_add(-second.radius, first.radius.powi(2));
    let along_numerator = center_distance.mul_add(center_distance, radius_square_difference);
    let along = along_numerator / (2.0 * center_distance);
    let height_squared = along.mul_add(-along, first.radius.powi(2)).max(0.0);
    let height = height_squared.sqrt();
    let unit = [
        center_delta[0] / center_distance,
        center_delta[1] / center_distance,
    ];
    let base = [
        along.mul_add(unit[0], first.center[0]),
        along.mul_add(unit[1], first.center[1]),
    ];
    let perpendicular = [-unit[1], unit[0]];
    let points = [
        [
            height.mul_add(perpendicular[0], base[0]),
            height.mul_add(perpendicular[1], base[1]),
        ],
        [
            height.mul_add(-perpendicular[0], base[0]),
            height.mul_add(-perpendicular[1], base[1]),
        ],
    ];

    points.into_iter().enumerate().any(|(index, point)| {
        if index == 1 && height <= tolerance {
            return false;
        }
        if !first.contains_point(point, tolerance) || !second.contains_point(point, tolerance) {
            return false;
        }
        if !adjacent {
            return true;
        }
        !(first.has_endpoint(point, tolerance) && second.has_endpoint(point, tolerance))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders_and_orients_mixed_closed_profile() {
        let profile = MeridianProfile::closed(
            vec![
                MeridianCurve::Line(MeridianLine {
                    start: [2.0, 1.0],
                    end: [0.0, 1.0],
                }),
                MeridianCurve::Line(MeridianLine {
                    start: [0.0, 0.0],
                    end: [2.0, 0.0],
                }),
                MeridianCurve::Line(MeridianLine {
                    start: [0.0, 1.0],
                    end: [0.0, 0.0],
                }),
                MeridianCurve::Line(MeridianLine {
                    start: [2.0, 0.0],
                    end: [2.0, 1.0],
                }),
            ],
            1.0e-9,
        );
        assert!(profile.is_some());
    }

    #[test]
    fn accepts_single_full_circle_profile() {
        let profile = MeridianProfile::closed(
            vec![MeridianCurve::Arc(MeridianArc {
                source_edge_ids: Vec::new(),
                center: [3.0, 0.0],
                radius: 1.0,
                start_angle: 0.0,
                end_angle: std::f64::consts::TAU,
            })],
            1.0e-9,
        );
        assert!(profile.is_some());
    }

    #[test]
    fn rejects_curve_that_crosses_negative_radius() {
        let arc = MeridianCurve::Arc(MeridianArc {
            source_edge_ids: Vec::new(),
            center: [0.5, 0.0],
            radius: 1.0,
            start_angle: 0.0,
            end_angle: std::f64::consts::TAU,
        });
        assert!(arc.min_radius(1.0e-9) < 0.0);
    }

    #[test]
    fn snaps_near_axis_line_endpoints() -> anyhow::Result<()> {
        let tolerance = 1.0e-9;
        let profile = MeridianProfile::from_lines(
            [
                MeridianLine {
                    start: [-0.5 * tolerance, 0.0],
                    end: [2.0, 0.0],
                },
                MeridianLine {
                    start: [2.0, 0.0],
                    end: [2.0, 1.0],
                },
                MeridianLine {
                    start: [2.0, 1.0],
                    end: [0.5 * tolerance, 1.0],
                },
            ],
            tolerance,
        )
        .ok_or_else(|| anyhow::anyhow!("near-axis line noise should canonicalize"))?
        .into_recovered();

        let axis_points = profile
            .iter()
            .flat_map(|curve| [curve.start_point(), curve.end_point()])
            .flatten()
            .filter(|point| point[0].abs() <= tolerance)
            .collect::<Vec<_>>();
        assert!(!axis_points.is_empty());
        assert!(axis_points.iter().all(|point| point[0] == 0.0));
        Ok(())
    }

    #[test]
    fn rejects_self_intersecting_profile() {
        let profile = MeridianProfile::from_lines(
            [
                MeridianLine {
                    start: [0.0, 0.0],
                    end: [2.0, 2.0],
                },
                MeridianLine {
                    start: [2.0, 2.0],
                    end: [0.0, 2.0],
                },
                MeridianLine {
                    start: [0.0, 2.0],
                    end: [2.0, 0.0],
                },
                MeridianLine {
                    start: [2.0, 0.0],
                    end: [0.0, 0.0],
                },
            ],
            1.0e-9,
        );
        assert!(profile.is_none());
    }

    #[test]
    fn rejects_cocircular_arc_containment() {
        let outer = MeridianArc {
            source_edge_ids: Vec::new(),
            center: [2.0, 0.0],
            radius: 1.0,
            start_angle: 0.0,
            end_angle: std::f64::consts::PI,
        };
        let contained = MeridianArc {
            source_edge_ids: Vec::new(),
            center: [2.0, 0.0],
            radius: 1.0,
            start_angle: std::f64::consts::FRAC_PI_4,
            end_angle: 3.0 * std::f64::consts::FRAC_PI_4,
        };
        assert!(arc_arc_has_extra_intersection(
            &outer, &contained, true, 1.0e-9
        ));
    }

    #[test]
    fn allows_complementary_cocircular_arcs_to_meet_at_endpoints() {
        let upper = MeridianArc {
            source_edge_ids: Vec::new(),
            center: [2.0, 0.0],
            radius: 1.0,
            start_angle: 0.0,
            end_angle: std::f64::consts::PI,
        };
        let lower = MeridianArc {
            source_edge_ids: Vec::new(),
            center: [2.0, 0.0],
            radius: 1.0,
            start_angle: std::f64::consts::PI,
            end_angle: std::f64::consts::TAU,
        };
        assert!(!arc_arc_has_extra_intersection(
            &upper, &lower, true, 1.0e-9
        ));
        assert!(arc_arc_has_extra_intersection(
            &upper, &lower, false, 1.0e-9
        ));
    }
}
