use crate::brep::{
    self, CircleSupport, CurveSupport, OrientedEdgeUse, PlaneSupport, SurfaceSupport,
};
use crate::instances::{build_index, entity_id, simple_record};
use ruststep::ast::EntityInstance;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::f64::consts::TAU;

const GEOM_TOL_MM: f64 = 1.0e-7;
const DIR_TOL: f64 = 1.0e-10;
const ANGLE_TOL_RAD: f64 = 1.0e-10;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecoveredSolidExtrusion {
    pub solid_id: u64,
    pub cap_face_ids: [u64; 2],
    pub side_face_ids: Vec<u64>,
    pub profile_curves: Vec<RecoveredProfileCurve>,
    pub origin_mm: [f64; 3],
    pub x_axis: [f64; 3],
    pub y_axis: [f64; 3],
    pub z_axis: [f64; 3],
    pub height_mm: f64,
    pub max_residual_mm: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RecoveredProfileCurve {
    Line {
        source_edge_ids: Vec<u64>,
        start_mm: [f64; 2],
        end_mm: [f64; 2],
    },
    CircleArc {
        source_edge_ids: Vec<u64>,
        center_mm: [f64; 2],
        radius_mm: f64,
        start_angle_rad: f64,
        end_angle_rad: f64,
    },
}

impl RecoveredProfileCurve {
    fn first_source_edge_id(&self) -> u64 {
        match self {
            Self::Line {
                source_edge_ids, ..
            }
            | Self::CircleArc {
                source_edge_ids, ..
            } => source_edge_ids.first().copied().unwrap_or(u64::MAX),
        }
    }

    pub fn source_edge_ids(&self) -> &[u64] {
        match self {
            Self::Line {
                source_edge_ids, ..
            }
            | Self::CircleArc {
                source_edge_ids, ..
            } => source_edge_ids,
        }
    }

    fn complexity(&self) -> usize {
        match self {
            Self::Line { .. } => 1,
            Self::CircleArc { .. } => 2,
        }
    }

    fn start_point(&self) -> [f64; 2] {
        match self {
            Self::Line { start_mm, .. } => *start_mm,
            Self::CircleArc {
                center_mm,
                radius_mm,
                start_angle_rad,
                ..
            } => [
                center_mm[0] + radius_mm * start_angle_rad.cos(),
                center_mm[1] + radius_mm * start_angle_rad.sin(),
            ],
        }
    }

    fn end_point(&self) -> [f64; 2] {
        match self {
            Self::Line { end_mm, .. } => *end_mm,
            Self::CircleArc {
                center_mm,
                radius_mm,
                end_angle_rad,
                ..
            } => [
                center_mm[0] + radius_mm * end_angle_rad.cos(),
                center_mm[1] + radius_mm * end_angle_rad.sin(),
            ],
        }
    }

    fn reversed(&self) -> Self {
        match self {
            Self::Line {
                source_edge_ids,
                start_mm,
                end_mm,
            } => Self::Line {
                source_edge_ids: source_edge_ids.clone(),
                start_mm: *end_mm,
                end_mm: *start_mm,
            },
            Self::CircleArc {
                source_edge_ids,
                center_mm,
                radius_mm,
                start_angle_rad,
                end_angle_rad,
            } => Self::CircleArc {
                source_edge_ids: source_edge_ids.clone(),
                center_mm: *center_mm,
                radius_mm: *radius_mm,
                start_angle_rad: *end_angle_rad,
                end_angle_rad: *start_angle_rad,
            },
        }
    }
}

#[derive(Debug, Clone)]
struct FaceInfo {
    id: u64,
    surface: SurfaceSupport,
    loop_edges: Vec<OrientedEdgeUse>,
}

#[derive(Debug, Clone)]
struct CanonicalProfile {
    origin_mm: [f64; 3],
    x_axis: [f64; 3],
    y_axis: [f64; 3],
    curves: Vec<RecoveredProfileCurve>,
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
    let face_ids = brep::solid_face_ids(solid_id, entities, index)?;
    if face_ids.len() < 3 {
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
    let first_cost = first
        .profile_curves
        .iter()
        .map(RecoveredProfileCurve::complexity)
        .sum::<usize>();
    let second_cost = second
        .profile_curves
        .iter()
        .map(RecoveredProfileCurve::complexity)
        .sum::<usize>();

    first_cost
        .cmp(&second_cost)
        .then_with(|| first.profile_curves.len().cmp(&second.profile_curves.len()))
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
    let (SurfaceSupport::Plane(first_plane), SurfaceSupport::Plane(second_plane)) =
        (first.surface, second.surface)
    else {
        return None;
    };
    if first.loop_edges.is_empty() || first.loop_edges.len() != second.loop_edges.len() {
        return None;
    }
    if !parallel(first_plane.normal, second_plane.normal) {
        return None;
    }

    let z_axis = canonical_axis(first_plane.normal);
    let first_offset = dot(z_axis, first_plane.origin_mm);
    let second_offset = dot(z_axis, second_plane.origin_mm);
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
    let SurfaceSupport::Plane(bottom_plane) = bottom.surface else {
        return None;
    };
    let SurfaceSupport::Plane(top_plane) = top.surface else {
        return None;
    };
    let extrusion = mul(z_axis, height_mm);

    if !face_lies_on_plane(bottom, bottom_plane) || !face_lies_on_plane(top, top_plane) {
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
        let residual = translated_profile_edge_residual(bottom_edge, top_edge, extrusion)?;
        max_residual_mm = max_residual_mm.max(residual);
        if residual > GEOM_TOL_MM {
            return None;
        }

        if !side_support_matches_profile(side, bottom_edge, z_axis) {
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
        profile_curves: profile.curves,
        origin_mm: profile.origin_mm,
        x_axis: profile.x_axis,
        y_axis: profile.y_axis,
        z_axis,
        height_mm,
        max_residual_mm,
    })
}

fn face_info(
    face_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<FaceInfo> {
    let surface_id = brep::face_surface(face_id, entities, index)?;
    let surface = brep::surface_support(surface_id, entities, index);
    if matches!(surface, SurfaceSupport::Other { .. }) {
        return None;
    }

    let loops = brep::face_loops(face_id, entities, index)?;
    if loops.len() != 1 {
        return None;
    }
    let loop_edges = loops.into_iter().next()?.edges;
    if loop_edges.is_empty()
        || loop_edges
            .iter()
            .any(|edge| matches!(edge.support, CurveSupport::Other { .. }))
    {
        return None;
    }

    Some(FaceInfo {
        id: face_id,
        surface,
        loop_edges,
    })
}

fn face_lies_on_plane(face: &FaceInfo, plane: PlaneSupport) -> bool {
    face.loop_edges
        .iter()
        .all(|edge| edge_lies_on_plane(edge, plane))
}

fn edge_lies_on_plane(edge: &OrientedEdgeUse, plane: PlaneSupport) -> bool {
    if point_plane_distance(edge.start_mm, plane) > GEOM_TOL_MM
        || point_plane_distance(edge.end_mm, plane) > GEOM_TOL_MM
    {
        return false;
    }
    match edge.support {
        CurveSupport::Line(_) => true,
        CurveSupport::Circle(circle) => {
            point_plane_distance(circle.center_mm, plane) <= GEOM_TOL_MM
                && parallel(circle.normal, plane.normal)
        }
        CurveSupport::Other { .. } => false,
    }
}

fn translated_profile_edge_residual(
    bottom: &OrientedEdgeUse,
    top: &OrientedEdgeUse,
    extrusion: [f64; 3],
) -> Option<f64> {
    let endpoint_residual = translated_endpoints_residual(bottom, top, extrusion);
    match (bottom.support, top.support) {
        (CurveSupport::Line(_), CurveSupport::Line(_)) => Some(endpoint_residual),
        (CurveSupport::Circle(bottom_circle), CurveSupport::Circle(top_circle)) => {
            if !parallel(bottom_circle.normal, top_circle.normal) {
                return None;
            }
            if (bottom_circle.radius_mm - top_circle.radius_mm).abs() > GEOM_TOL_MM {
                return None;
            }
            let bottom_sweep = circle_edge_sweep(bottom, bottom_circle)?;
            let top_sweep = circle_edge_sweep(top, top_circle)?;
            if (bottom_sweep.abs() - top_sweep.abs()).abs() > ANGLE_TOL_RAD {
                return None;
            }
            Some(
                endpoint_residual
                    .max(distance(
                        add(bottom_circle.center_mm, extrusion),
                        top_circle.center_mm,
                    ))
                    .max((bottom_circle.radius_mm - top_circle.radius_mm).abs()),
            )
        }
        _ => None,
    }
}

fn translated_endpoints_residual(
    bottom: &OrientedEdgeUse,
    top: &OrientedEdgeUse,
    extrusion: [f64; 3],
) -> f64 {
    let direct = distance(add(bottom.start_mm, extrusion), top.start_mm)
        .max(distance(add(bottom.end_mm, extrusion), top.end_mm));
    let reverse = distance(add(bottom.start_mm, extrusion), top.end_mm)
        .max(distance(add(bottom.end_mm, extrusion), top.start_mm));
    direct.min(reverse)
}

fn side_support_matches_profile(
    side: &FaceInfo,
    profile_edge: &OrientedEdgeUse,
    z_axis: [f64; 3],
) -> bool {
    match (profile_edge.support, side.surface) {
        (CurveSupport::Line(_), SurfaceSupport::Plane(plane)) => {
            let direction = sub(profile_edge.end_mm, profile_edge.start_mm);
            let Some(direction) = normalize(direction) else {
                return false;
            };
            let expected_normal = cross(direction, z_axis);
            let Some(expected_normal) = normalize(expected_normal) else {
                return false;
            };
            parallel(plane.normal, expected_normal)
                && side
                    .loop_edges
                    .iter()
                    .all(|edge| edge_lies_on_plane(edge, plane))
        }
        (CurveSupport::Circle(circle), SurfaceSupport::Cylinder(cylinder)) => {
            parallel(circle.normal, z_axis)
                && parallel(cylinder.axis, z_axis)
                && (circle.radius_mm - cylinder.radius_mm).abs() <= GEOM_TOL_MM
                && axis_line_distance(cylinder.axis_origin_mm, cylinder.axis, circle.center_mm)
                    <= GEOM_TOL_MM
        }
        _ => false,
    }
}

fn connector_residual(edge: &OrientedEdgeUse, extrusion: [f64; 3]) -> Option<f64> {
    if !matches!(edge.support, CurveSupport::Line(_)) {
        return None;
    }
    let forward = distance(sub(edge.end_mm, edge.start_mm), extrusion);
    let reverse = distance(sub(edge.start_mm, edge.end_mm), extrusion);
    Some(forward.min(reverse))
}

fn canonical_profile(cap: &FaceInfo, z_axis: [f64; 3]) -> Option<CanonicalProfile> {
    if cap.loop_edges.is_empty() {
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

    let reference = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        .into_iter()
        .min_by(|a, b| dot(*a, z_axis).abs().total_cmp(&dot(*b, z_axis).abs()))?;
    let x_axis = normalize(sub(reference, mul(z_axis, dot(reference, z_axis))))?;
    let y_axis = normalize(cross(z_axis, x_axis))?;

    if cap.loop_edges.len() == 1
        && let CurveSupport::Circle(circle) = cap.loop_edges[0].support
        && distance(cap.loop_edges[0].start_mm, cap.loop_edges[0].end_mm) <= GEOM_TOL_MM
    {
        if !parallel(circle.normal, z_axis) {
            return None;
        }
        return Some(CanonicalProfile {
            origin_mm: circle.center_mm,
            x_axis,
            y_axis,
            curves: vec![RecoveredProfileCurve::CircleArc {
                source_edge_ids: vec![cap.loop_edges[0].edge_id],
                center_mm: [0.0, 0.0],
                radius_mm: circle.radius_mm,
                start_angle_rad: 0.0,
                end_angle_rad: TAU,
            }],
        });
    }

    let origin_mm = cap
        .loop_edges
        .iter()
        .flat_map(|edge| [edge.start_mm, edge.end_mm])
        .min_by(|a, b| {
            let aa = [dot(*a, x_axis), dot(*a, y_axis)];
            let bb = [dot(*b, x_axis), dot(*b, y_axis)];
            aa[0]
                .total_cmp(&bb[0])
                .then_with(|| aa[1].total_cmp(&bb[1]))
        })?;

    let mut curves = cap
        .loop_edges
        .iter()
        .map(|edge| recovered_profile_curve(edge, origin_mm, x_axis, y_axis, z_axis))
        .collect::<Option<Vec<_>>>()?;

    if signed_area_curves(&curves) < 0.0 {
        curves = curves
            .iter()
            .rev()
            .map(RecoveredProfileCurve::reversed)
            .collect();
    }

    curves = simplify_profile_curves(curves);

    if curves.len() == 1
        && let RecoveredProfileCurve::CircleArc {
            source_edge_ids,
            center_mm,
            radius_mm,
            start_angle_rad,
            end_angle_rad,
        } = &curves[0]
        && (((end_angle_rad - start_angle_rad).abs() - TAU).abs()) <= ANGLE_TOL_RAD
    {
        let center_world = add(
            origin_mm,
            add(mul(x_axis, center_mm[0]), mul(y_axis, center_mm[1])),
        );
        return Some(CanonicalProfile {
            origin_mm: center_world,
            x_axis,
            y_axis,
            curves: vec![RecoveredProfileCurve::CircleArc {
                source_edge_ids: source_edge_ids.clone(),
                center_mm: [0.0, 0.0],
                radius_mm: *radius_mm,
                start_angle_rad: 0.0,
                end_angle_rad: TAU,
            }],
        });
    }

    let start_index = (0..curves.len()).min_by(|&a, &b| {
        let aa = curves[a].start_point();
        let bb = curves[b].start_point();
        aa[0]
            .total_cmp(&bb[0])
            .then_with(|| aa[1].total_cmp(&bb[1]))
            .then_with(|| {
                curves[a]
                    .first_source_edge_id()
                    .cmp(&curves[b].first_source_edge_id())
            })
    })?;
    curves.rotate_left(start_index);

    if curves.iter().enumerate().any(|(index, curve)| {
        distance2(
            curve.end_point(),
            curves[(index + 1) % curves.len()].start_point(),
        ) > GEOM_TOL_MM
    }) {
        return None;
    }

    Some(CanonicalProfile {
        origin_mm,
        x_axis,
        y_axis,
        curves,
    })
}

fn recovered_profile_curve(
    edge: &OrientedEdgeUse,
    origin_mm: [f64; 3],
    x_axis: [f64; 3],
    y_axis: [f64; 3],
    z_axis: [f64; 3],
) -> Option<RecoveredProfileCurve> {
    let start_mm = project2(edge.start_mm, origin_mm, x_axis, y_axis);
    let end_mm = project2(edge.end_mm, origin_mm, x_axis, y_axis);
    match edge.support {
        CurveSupport::Line(_) => Some(RecoveredProfileCurve::Line {
            source_edge_ids: vec![edge.edge_id],
            start_mm,
            end_mm,
        }),
        CurveSupport::Circle(circle) => {
            if !parallel(circle.normal, z_axis) {
                return None;
            }
            let center_mm = project2(circle.center_mm, origin_mm, x_axis, y_axis);
            let start_angle = point_angle_2d(start_mm, center_mm)?;
            let end_angle = point_angle_2d(end_mm, center_mm)?;
            let increasing = if dot(circle.normal, z_axis) >= 0.0 {
                edge.parameter_forward
            } else {
                !edge.parameter_forward
            };
            let sweep = directional_sweep(start_angle, end_angle, increasing, false);
            Some(RecoveredProfileCurve::CircleArc {
                source_edge_ids: vec![edge.edge_id],
                center_mm,
                radius_mm: circle.radius_mm,
                start_angle_rad: start_angle,
                end_angle_rad: start_angle + sweep,
            })
        }
        CurveSupport::Other { .. } => None,
    }
}

fn simplify_profile_curves(mut curves: Vec<RecoveredProfileCurve>) -> Vec<RecoveredProfileCurve> {
    if curves.len() <= 1 {
        return curves;
    }

    loop {
        let mut changed = false;
        let mut linear = Vec::with_capacity(curves.len());
        let mut index = 0;
        while index < curves.len() {
            if index + 1 < curves.len()
                && let Some(merged) = merge_profile_curves(&curves[index], &curves[index + 1])
            {
                linear.push(merged);
                index += 2;
                changed = true;
                continue;
            }
            linear.push(curves[index].clone());
            index += 1;
        }
        curves = linear;

        if curves.len() > 1 {
            let last = curves.len() - 1;
            if let Some(merged) = merge_profile_curves(&curves[last], &curves[0]) {
                let mut wrapped = Vec::with_capacity(curves.len() - 1);
                wrapped.push(merged);
                wrapped.extend(curves[1..last].iter().cloned());
                curves = wrapped;
                changed = true;
            }
        }

        if !changed || curves.len() <= 1 {
            return curves;
        }
    }
}

fn merge_profile_curves(
    first: &RecoveredProfileCurve,
    second: &RecoveredProfileCurve,
) -> Option<RecoveredProfileCurve> {
    if distance2(first.end_point(), second.start_point()) > GEOM_TOL_MM {
        return None;
    }

    match (first, second) {
        (
            RecoveredProfileCurve::Line {
                source_edge_ids: first_ids,
                start_mm,
                end_mm: first_end,
            },
            RecoveredProfileCurve::Line {
                source_edge_ids: second_ids,
                start_mm: second_start,
                end_mm,
            },
        ) => {
            let first_direction = normalize2(sub2(*first_end, *start_mm))?;
            let second_direction = normalize2(sub2(*end_mm, *second_start))?;
            if cross2(first_direction, second_direction).abs() > DIR_TOL
                || dot2(first_direction, second_direction) < 1.0 - DIR_TOL
            {
                return None;
            }
            Some(RecoveredProfileCurve::Line {
                source_edge_ids: merged_source_ids(first_ids, second_ids),
                start_mm: *start_mm,
                end_mm: *end_mm,
            })
        }
        (
            RecoveredProfileCurve::CircleArc {
                source_edge_ids: first_ids,
                center_mm: first_center,
                radius_mm: first_radius,
                start_angle_rad,
                end_angle_rad: first_end,
            },
            RecoveredProfileCurve::CircleArc {
                source_edge_ids: second_ids,
                center_mm: second_center,
                radius_mm: second_radius,
                start_angle_rad: second_start,
                end_angle_rad: second_end,
            },
        ) => {
            if distance2(*first_center, *second_center) > GEOM_TOL_MM
                || (first_radius - second_radius).abs() > GEOM_TOL_MM
            {
                return None;
            }
            let first_sweep = first_end - start_angle_rad;
            let second_sweep = second_end - second_start;
            if first_sweep.signum() != second_sweep.signum() {
                return None;
            }
            let sweep = first_sweep + second_sweep;
            if sweep.abs() > TAU + ANGLE_TOL_RAD {
                return None;
            }
            Some(RecoveredProfileCurve::CircleArc {
                source_edge_ids: merged_source_ids(first_ids, second_ids),
                center_mm: *first_center,
                radius_mm: (*first_radius + *second_radius) * 0.5,
                start_angle_rad: *start_angle_rad,
                end_angle_rad: *start_angle_rad + sweep,
            })
        }
        _ => None,
    }
}

fn merged_source_ids(first: &[u64], second: &[u64]) -> Vec<u64> {
    let mut ids = first.iter().chain(second).copied().collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    ids
}

fn circle_edge_sweep(edge: &OrientedEdgeUse, circle: CircleSupport) -> Option<f64> {
    let start = circle_angle_3d(edge.start_mm, circle)?;
    let end = circle_angle_3d(edge.end_mm, circle)?;
    let closed = distance(edge.start_mm, edge.end_mm) <= GEOM_TOL_MM;
    Some(directional_sweep(
        start,
        end,
        edge.parameter_forward,
        closed,
    ))
}

fn directional_sweep(start: f64, end: f64, increasing: bool, closed: bool) -> f64 {
    if closed {
        return if increasing { TAU } else { -TAU };
    }
    if increasing {
        (end - start).rem_euclid(TAU)
    } else {
        -((start - end).rem_euclid(TAU))
    }
}

fn circle_angle_3d(point: [f64; 3], circle: CircleSupport) -> Option<f64> {
    let y_direction = normalize(cross(circle.normal, circle.x_direction))?;
    let relative = sub(point, circle.center_mm);
    if dot(relative, circle.normal).abs() > GEOM_TOL_MM {
        return None;
    }
    let radial = norm(relative);
    if (radial - circle.radius_mm).abs() > GEOM_TOL_MM {
        return None;
    }
    Some(dot(relative, y_direction).atan2(dot(relative, circle.x_direction)))
}

fn point_angle_2d(point: [f64; 2], center: [f64; 2]) -> Option<f64> {
    let dx = point[0] - center[0];
    let dy = point[1] - center[1];
    let radius = (dx * dx + dy * dy).sqrt();
    if !radius.is_finite() || radius <= DIR_TOL {
        return None;
    }
    Some(dy.atan2(dx))
}

fn signed_area_curves(curves: &[RecoveredProfileCurve]) -> f64 {
    curves
        .iter()
        .map(|curve| match curve {
            RecoveredProfileCurve::Line {
                start_mm, end_mm, ..
            } => 0.5 * (start_mm[0] * end_mm[1] - end_mm[0] * start_mm[1]),
            RecoveredProfileCurve::CircleArc {
                center_mm,
                radius_mm,
                start_angle_rad,
                end_angle_rad,
                ..
            } => {
                let theta0 = *start_angle_rad;
                let theta1 = *end_angle_rad;
                let r = *radius_mm;
                0.5 * (r * center_mm[0] * (theta1.sin() - theta0.sin())
                    - r * center_mm[1] * (theta1.cos() - theta0.cos())
                    + r * r * (theta1 - theta0))
            }
        })
        .sum()
}

fn project2(point: [f64; 3], origin: [f64; 3], x_axis: [f64; 3], y_axis: [f64; 3]) -> [f64; 2] {
    let relative = sub(point, origin);
    let mut result = [dot(relative, x_axis), dot(relative, y_axis)];
    for value in &mut result {
        if value.abs() <= 1.0e-14 {
            *value = 0.0;
        }
    }
    result
}

fn axis_line_distance(origin: [f64; 3], axis: [f64; 3], point: [f64; 3]) -> f64 {
    norm(cross(sub(point, origin), axis))
}

fn point_plane_distance(point: [f64; 3], plane: PlaneSupport) -> f64 {
    dot(plane.normal, sub(point, plane.origin_mm)).abs()
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

fn distance2(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn cross2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn normalize2(vector: [f64; 2]) -> Option<[f64; 2]> {
    let length = (vector[0] * vector[0] + vector[1] * vector[1]).sqrt();
    if !length.is_finite() || length <= DIR_TOL {
        return None;
    }
    Some([vector[0] / length, vector[1] / length])
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

    fn line_edge(
        edge_id: u64,
        start_vertex: u64,
        end_vertex: u64,
        start_mm: [f64; 3],
        end_mm: [f64; 3],
    ) -> OrientedEdgeUse {
        OrientedEdgeUse {
            oriented_edge_id: edge_id + 100,
            edge_id,
            curve_id: edge_id + 200,
            curve_same_sense: true,
            parameter_forward: true,
            start_vertex,
            end_vertex,
            start_mm,
            end_mm,
            support: CurveSupport::Line(brep::LineSupport {
                origin_mm: start_mm,
                direction: normalize(sub(end_mm, start_mm)).unwrap(),
            }),
        }
    }

    #[test]
    fn canonical_axis_has_deterministic_sign() {
        assert_eq!(canonical_axis([-1.0, 0.0, 0.0]), [1.0, -0.0, -0.0]);
        assert_eq!(canonical_axis([0.0, -1.0, 0.0]), [-0.0, 1.0, -0.0]);
    }

    #[test]
    fn translated_line_residual_ignores_edge_orientation() {
        let bottom = line_edge(1, 10, 11, [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        let mut top = line_edge(2, 21, 20, [2.0, 0.0, 3.0], [0.0, 0.0, 3.0]);
        top.parameter_forward = false;
        assert_eq!(
            translated_profile_edge_residual(&bottom, &top, [0.0, 0.0, 3.0]),
            Some(0.0)
        );
    }

    #[test]
    fn coalesces_two_semicircles_into_full_circle() {
        let curves = vec![
            RecoveredProfileCurve::CircleArc {
                source_edge_ids: vec![10],
                center_mm: [0.0, 0.0],
                radius_mm: 2.0,
                start_angle_rad: 0.0,
                end_angle_rad: std::f64::consts::PI,
            },
            RecoveredProfileCurve::CircleArc {
                source_edge_ids: vec![11],
                center_mm: [0.0, 0.0],
                radius_mm: 2.0,
                start_angle_rad: std::f64::consts::PI,
                end_angle_rad: TAU,
            },
        ];
        let simplified = simplify_profile_curves(curves);
        assert_eq!(simplified.len(), 1);
        let RecoveredProfileCurve::CircleArc {
            source_edge_ids,
            start_angle_rad,
            end_angle_rad,
            ..
        } = &simplified[0]
        else {
            panic!("expected circle");
        };
        assert_eq!(source_edge_ids, &vec![10, 11]);
        assert!((*start_angle_rad).abs() < 1.0e-12);
        assert!((*end_angle_rad - TAU).abs() < 1.0e-12);
    }

    #[test]
    fn signed_area_handles_semicircle() {
        let curves = vec![
            RecoveredProfileCurve::Line {
                source_edge_ids: vec![1],
                start_mm: [-1.0, 0.0],
                end_mm: [1.0, 0.0],
            },
            RecoveredProfileCurve::CircleArc {
                source_edge_ids: vec![2],
                center_mm: [0.0, 0.0],
                radius_mm: 1.0,
                start_angle_rad: 0.0,
                end_angle_rad: std::f64::consts::PI,
            },
        ];
        assert!((signed_area_curves(&curves) - std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);
    }
}
