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

#[derive(Debug, Clone, Copy)]
struct LinearMeridianSupport {
    reference_t: f64,
    reference_signed_radius: f64,
    slope: f64,
}

type ProfileGraph = (Vec<[f64; 2]>, Vec<(usize, usize)>);

#[derive(Debug, Clone)]
struct FaceInfo {
    surface: SurfaceSupport,
    loops: Vec<brep::FaceLoop>,
}

struct TopologyContext<'a> {
    faces: &'a [FaceInfo],
    edge_faces: &'a HashMap<u64, Vec<usize>>,
    entities: &'a [EntityInstance],
    index: &'a HashMap<u64, usize>,
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

    let (axis_reference_origin_mm, axis_reference_direction) =
        faces.iter().find_map(|face| match face.surface {
            SurfaceSupport::Cylinder(cylinder) => Some((cylinder.axis_origin_mm, cylinder.axis)),
            SurfaceSupport::Cone(cone) => Some((cone.reference_origin_mm, cone.axis)),
            SurfaceSupport::Revolution(revolution) => {
                Some((revolution.axis_origin_mm, revolution.axis))
            }
            _ => None,
        })?;
    let axis_direction = canonical_axis(axis_reference_direction);
    let axis_origin_mm =
        closest_axis_point_to_global_origin(axis_reference_origin_mm, axis_direction);
    let radial_direction = radial_basis(axis_direction)?;
    let context = TopologyContext {
        faces: &faces,
        edge_faces: &edge_faces,
        entities,
        index,
    };

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
            SurfaceSupport::Cone(cone) => cone_profile_segment(
                face_index,
                face,
                cone,
                axis_origin_mm,
                axis_direction,
                &faces,
                &edge_faces,
            )?,
            SurfaceSupport::Revolution(revolution) => revolution_line_profile_segment(
                face_index,
                face,
                revolution,
                axis_origin_mm,
                axis_direction,
                &context,
            )?,
            SurfaceSupport::Plane(plane) => plane_profile_segment(
                face_index,
                face,
                plane,
                axis_origin_mm,
                axis_direction,
                &context,
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

fn cone_profile_segment(
    face_index: usize,
    face: &FaceInfo,
    cone: brep::ConeSupport,
    axis_origin: [f64; 3],
    axis: [f64; 3],
    faces: &[FaceInfo],
    edge_faces: &HashMap<u64, Vec<usize>>,
) -> Option<(Segment2, f64)> {
    if face.loops.len() != 1
        || !parallel(cone.axis, axis)
        || axis_distance(cone.reference_origin_mm, axis_origin, axis) > GEOM_TOL_MM
        || !cone.reference_radius_mm.is_finite()
        || cone.reference_radius_mm < 0.0
        || !cone.semi_angle_rad.is_finite()
        || cone.semi_angle_rad <= 0.0
        || cone.semi_angle_rad >= std::f64::consts::FRAC_PI_2
    {
        return None;
    }

    let mut min_sample = [f64::INFINITY, f64::INFINITY];
    let mut max_sample = [f64::NEG_INFINITY, f64::NEG_INFINITY];
    for edge in &face.loops[0].edges {
        for point in [edge.start_mm, edge.end_mm] {
            let sample = [
                point_axis_distance(point, axis_origin, axis),
                axial_coordinate(point, axis_origin, axis),
            ];
            if !sample[0].is_finite() || !sample[1].is_finite() {
                return None;
            }
            if sample[1] < min_sample[1] {
                min_sample = sample;
            }
            if sample[1] > max_sample[1] {
                max_sample = sample;
            }
        }
    }
    let delta_t = max_sample[1] - min_sample[1];
    if delta_t <= GEOM_TOL_MM || min_sample[0] <= GEOM_TOL_MM || max_sample[0] <= GEOM_TOL_MM {
        // Keep the first curved-meridian pass deliberately to non-degenerate
        // frusta. Apex topology can be admitted separately once proven.
        return None;
    }

    let slope = (max_sample[0] - min_sample[0]) / delta_t;
    let expected_delta_r = delta_t * cone.semi_angle_rad.tan();
    let reference_t = axial_coordinate(cone.reference_origin_mm, axis_origin, axis);
    let radius_at = |t: f64| min_sample[0] + slope * (t - min_sample[1]);
    if !slope.is_finite()
        || !expected_delta_r.is_finite()
        || !reference_t.is_finite()
        || !radius_at(reference_t).is_finite()
    {
        return None;
    }
    let mut max_residual = axis_distance(cone.reference_origin_mm, axis_origin, axis)
        .max(((max_sample[0] - min_sample[0]).abs() - expected_delta_r).abs())
        .max((radius_at(reference_t) - cone.reference_radius_mm).abs());

    for edge in &face.loops[0].edges {
        for point in [edge.start_mm, edge.end_mm] {
            let radius = point_axis_distance(point, axis_origin, axis);
            let t = axial_coordinate(point, axis_origin, axis);
            if !radius.is_finite() || !t.is_finite() {
                return None;
            }
            max_residual = max_residual.max((radius - radius_at(t)).abs());
        }
    }
    if max_residual > GEOM_TOL_MM {
        return None;
    }

    for edge in &face.loops[0].edges {
        match &edge.support {
            CurveSupport::Line(line) => {
                let direction = normalize(line.direction)?;
                let axial = dot(direction, axis).abs();
                let radial = norm(sub(direction, mul(axis, dot(direction, axis))));
                if (radial.atan2(axial) - cone.semi_angle_rad).abs() > DIR_TOL {
                    return None;
                }
            }
            CurveSupport::Circle(circle) => {
                if !parallel(circle.normal, axis) {
                    return None;
                }
                let t = axial_coordinate(circle.center_mm, axis_origin, axis);
                max_residual = max_residual
                    .max(axis_distance(circle.center_mm, axis_origin, axis))
                    .max((circle.radius_mm - radius_at(t)).abs());
            }
            CurveSupport::BSpline(_) => {
                let neighbor = unique_neighbor_face(face_index, edge.edge_id, edge_faces)?;
                let SurfaceSupport::Plane(plane) = faces.get(neighbor)?.surface else {
                    return None;
                };
                if !parallel(plane.normal, axis) {
                    return None;
                }
                let plane_t = axial_coordinate(plane.origin_mm, axis_origin, axis);
                for point in [edge.start_mm, edge.end_mm] {
                    let t = axial_coordinate(point, axis_origin, axis);
                    let radius = point_axis_distance(point, axis_origin, axis);
                    max_residual = max_residual
                        .max((t - plane_t).abs())
                        .max((radius - radius_at(t)).abs());
                }
                max_residual = max_residual.max(plane.max_residual_mm);
            }
            CurveSupport::Other { .. } => return None,
        }
    }

    if max_residual > GEOM_TOL_MM {
        return None;
    }
    Some((
        Segment2 {
            a: [radius_at(min_sample[1]), min_sample[1]],
            b: [radius_at(max_sample[1]), max_sample[1]],
        },
        max_residual,
    ))
}

fn revolution_line_profile_segment(
    face_index: usize,
    face: &FaceInfo,
    revolution: brep::RevolutionSurfaceSupport,
    axis_origin: [f64; 3],
    axis: [f64; 3],
    context: &TopologyContext<'_>,
) -> Option<(Segment2, f64)> {
    if face.loops.len() != 1 {
        return None;
    }
    let (support, mut max_residual) = revolution_linear_support(
        revolution,
        axis_origin,
        axis,
        context.entities,
        context.index,
    )?;

    let mut min_t = f64::INFINITY;
    let mut max_t = f64::NEG_INFINITY;
    for edge in &face.loops[0].edges {
        for point in [edge.start_mm, edge.end_mm] {
            let t = axial_coordinate(point, axis_origin, axis);
            let radius = point_axis_distance(point, axis_origin, axis);
            let expected_radius = linear_radius_at(support, t)?;
            if !t.is_finite() || !radius.is_finite() {
                return None;
            }
            min_t = min_t.min(t);
            max_t = max_t.max(t);
            max_residual = max_residual.max((radius - expected_radius).abs());
        }
    }
    if !min_t.is_finite() || max_t - min_t <= GEOM_TOL_MM {
        return None;
    }
    let min_signed = linear_signed_radius_at(support, min_t)?;
    let max_signed = linear_signed_radius_at(support, max_t)?;
    if min_signed.abs() <= GEOM_TOL_MM
        || max_signed.abs() <= GEOM_TOL_MM
        || min_signed.signum() != max_signed.signum()
    {
        // Crossing the axis creates an apex / double-cone topology. Keep this
        // first generic-revolution pass to one non-degenerate meridian branch.
        return None;
    }

    for edge in &face.loops[0].edges {
        match &edge.support {
            CurveSupport::Line(line) => {
                let edge_support = linear_meridian_support(*line, axis_origin, axis)?;
                let generator_angle = support.slope.abs().atan();
                let edge_angle = edge_support.slope.abs().atan();
                if (generator_angle - edge_angle).abs() > DIR_TOL {
                    return None;
                }
            }
            CurveSupport::Circle(circle) => {
                if !parallel(circle.normal, axis) {
                    return None;
                }
                let t = axial_coordinate(circle.center_mm, axis_origin, axis);
                max_residual = max_residual
                    .max(axis_distance(circle.center_mm, axis_origin, axis))
                    .max((circle.radius_mm - linear_radius_at(support, t)?).abs());
            }
            CurveSupport::BSpline(_) => {
                let neighbor = unique_neighbor_face(face_index, edge.edge_id, context.edge_faces)?;
                let SurfaceSupport::Plane(plane) = context.faces.get(neighbor)?.surface else {
                    return None;
                };
                if !parallel(plane.normal, axis) {
                    return None;
                }
                let plane_t = axial_coordinate(plane.origin_mm, axis_origin, axis);
                for point in [edge.start_mm, edge.end_mm] {
                    let t = axial_coordinate(point, axis_origin, axis);
                    let radius = point_axis_distance(point, axis_origin, axis);
                    max_residual = max_residual
                        .max((t - plane_t).abs())
                        .max((radius - linear_radius_at(support, t)?).abs());
                }
                max_residual = max_residual.max(plane.max_residual_mm);
            }
            CurveSupport::Other { .. } => return None,
        }
    }
    if max_residual > GEOM_TOL_MM {
        return None;
    }
    Some((
        Segment2 {
            a: [linear_radius_at(support, min_t)?, min_t],
            b: [linear_radius_at(support, max_t)?, max_t],
        },
        max_residual,
    ))
}

fn revolution_linear_support(
    revolution: brep::RevolutionSurfaceSupport,
    axis_origin: [f64; 3],
    axis: [f64; 3],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<(LinearMeridianSupport, f64)> {
    if !parallel(revolution.axis, axis)
        || axis_distance(revolution.axis_origin_mm, axis_origin, axis) > GEOM_TOL_MM
    {
        return None;
    }
    let CurveSupport::Line(line) = brep::curve_support(revolution.swept_curve_id, entities, index)
    else {
        return None;
    };
    let support = linear_meridian_support(line, axis_origin, axis)?;
    Some((
        support,
        axis_distance(revolution.axis_origin_mm, axis_origin, axis),
    ))
}

fn linear_meridian_support(
    line: brep::LineSupport,
    axis_origin: [f64; 3],
    axis: [f64; 3],
) -> Option<LinearMeridianSupport> {
    let direction = normalize(line.direction)?;
    let axial = dot(direction, axis);
    if axial.abs() <= DIR_TOL {
        return None;
    }
    let relative = sub(line.origin_mm, axis_origin);
    let reference_t = dot(relative, axis);
    let perpendicular = sub(direction, mul(axis, axial));
    let radial_direction_magnitude = norm(perpendicular);
    let (reference_signed_radius, slope) = if radial_direction_magnitude <= DIR_TOL {
        (point_axis_distance(line.origin_mm, axis_origin, axis), 0.0)
    } else {
        let radial_direction = mul(perpendicular, 1.0 / radial_direction_magnitude);
        let meridian_normal = normalize(cross(axis, radial_direction))?;
        if dot(relative, meridian_normal).abs() > GEOM_TOL_MM {
            return None;
        }
        (
            dot(relative, radial_direction),
            radial_direction_magnitude / axial,
        )
    };
    if !reference_t.is_finite() || !reference_signed_radius.is_finite() || !slope.is_finite() {
        return None;
    }
    Some(LinearMeridianSupport {
        reference_t,
        reference_signed_radius,
        slope,
    })
}

fn linear_signed_radius_at(support: LinearMeridianSupport, axial: f64) -> Option<f64> {
    let radius = support.reference_signed_radius + support.slope * (axial - support.reference_t);
    radius.is_finite().then_some(radius)
}

fn linear_radius_at(support: LinearMeridianSupport, axial: f64) -> Option<f64> {
    Some(linear_signed_radius_at(support, axial)?.abs())
}

fn plane_profile_segment(
    face_index: usize,
    face: &FaceInfo,
    plane: brep::PlaneSupport,
    axis_origin: [f64; 3],
    axis: [f64; 3],
    context: &TopologyContext<'_>,
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
                    let neighbor =
                        unique_neighbor_face(face_index, edge.edge_id, context.edge_faces)?;
                    let neighbor_face = context.faces.get(neighbor)?;
                    let expected_radius = match neighbor_face.surface {
                        SurfaceSupport::Cylinder(cylinder) => {
                            if !parallel(cylinder.axis, axis)
                                || axis_distance(cylinder.axis_origin_mm, axis_origin, axis)
                                    > GEOM_TOL_MM
                            {
                                return None;
                            }
                            max_residual = max_residual.max(axis_distance(
                                cylinder.axis_origin_mm,
                                axis_origin,
                                axis,
                            ));
                            cylinder.radius_mm
                        }
                        SurfaceSupport::Cone(cone) => {
                            let (segment, residual) = cone_profile_segment(
                                neighbor,
                                neighbor_face,
                                cone,
                                axis_origin,
                                axis,
                                context.faces,
                                context.edge_faces,
                            )?;
                            max_residual = max_residual.max(residual);
                            segment_radius_at_axial(segment, t)?
                        }
                        SurfaceSupport::Revolution(revolution) => {
                            let (support, residual) = revolution_linear_support(
                                revolution,
                                axis_origin,
                                axis,
                                context.entities,
                                context.index,
                            )?;
                            max_residual = max_residual.max(residual);
                            linear_radius_at(support, t)?
                        }
                        _ => return None,
                    };
                    let start_radius = point_axis_distance(edge.start_mm, axis_origin, axis);
                    let end_radius = point_axis_distance(edge.end_mm, axis_origin, axis);
                    max_residual = max_residual
                        .max((start_radius - expected_radius).abs())
                        .max((end_radius - expected_radius).abs());
                    min_r = min_r.min(expected_radius);
                    max_r = max_r.max(expected_radius);
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

fn segment_radius_at_axial(segment: Segment2, axial: f64) -> Option<f64> {
    let delta = segment.b[1] - segment.a[1];
    if delta.abs() <= GEOM_TOL_MM || !between(axial, segment.a[1], segment.b[1]) {
        return None;
    }
    let fraction = (axial - segment.a[1]) / delta;
    let radius = segment.a[0] + fraction * (segment.b[0] - segment.a[0]);
    (radius.is_finite() && radius >= -GEOM_TOL_MM).then_some(radius.max(0.0))
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

fn between(value: f64, a: f64, b: f64) -> bool {
    value >= a.min(b) - GEOM_TOL_MM && value <= a.max(b) + GEOM_TOL_MM
}

fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn cross2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
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
    fn closes_sloped_frustum_profile() {
        let profile = closed_profile_from_segments(vec![
            Segment2 {
                a: [0.0, -1.0],
                b: [2.0, -1.0],
            },
            Segment2 {
                a: [2.0, -1.0],
                b: [1.0, 1.0],
            },
            Segment2 {
                a: [0.0, 1.0],
                b: [1.0, 1.0],
            },
        ])
        .unwrap();
        assert_eq!(profile.len(), 4);
        assert!(profile.iter().any(|point| *point == [2.0, -1.0]));
        assert!(profile.iter().any(|point| *point == [1.0, 1.0]));
        assert!(signed_area(&profile) > 0.0);
    }

    #[test]
    fn rejects_crossing_sloped_profiles() {
        assert!(segments_intersect(
            [1.0, 0.0],
            [3.0, 2.0],
            [3.0, 0.0],
            [1.0, 2.0]
        ));
        assert!(!segments_intersect(
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [4.0, 1.0]
        ));
    }

    #[test]
    fn recovers_native_conical_frustum_fixture() {
        let bytes = include_bytes!("../validation/fixtures/native_conical_frustum.step");
        let recovered = crate::detect_solid_revolutions_bytes(bytes).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(
            recovered[0].profile_points_mm,
            vec![[0.0, 0.0], [2.0, 0.0], [1.0, 2.0], [0.0, 2.0]]
        );
        assert!(recovered[0].max_residual_mm < 1.0e-9);

        let tampered = String::from_utf8_lossy(bytes).replacen(
            "CONICAL_SURFACE('',#32,2.,0.463647609001)",
            "CONICAL_SURFACE('',#32,2.25,0.463647609001)",
            1,
        );
        assert!(
            crate::detect_solid_revolutions_bytes(tampered.as_bytes())
                .unwrap()
                .is_empty()
        );

        #[cfg(feature = "cad-kernel-monstertruck")]
        {
            use crate::cad_kernel::CadKernel;
            let fragment =
                crate::cad_recovery::recover_solid_revolution_fragment(&recovered[0]).unwrap();
            let kernel = crate::cad_kernel::monstertruck::MonstertruckKernel;
            let rebuilt = kernel.evaluate(&fragment.model, fragment.root).unwrap();
            assert!(kernel.summarize(&rebuilt).geometrically_consistent);
            ruststep::parser::parse(&kernel.to_step(&rebuilt).unwrap()).unwrap();
        }
    }

    #[test]
    fn recovers_native_hollow_conical_frustum_fixture() {
        let bytes = include_bytes!("../validation/fixtures/native_hollow_conical_frustum.step");
        let recovered = crate::detect_solid_revolutions_bytes(bytes).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(
            recovered[0].profile_points_mm,
            vec![[0.5, 2.0], [1.0, 0.0], [3.0, 0.0], [2.0, 2.0]]
        );
        assert!(recovered[0].max_residual_mm < 1.0e-9);

        #[cfg(feature = "cad-kernel-monstertruck")]
        {
            use crate::cad_kernel::CadKernel;
            let fragment =
                crate::cad_recovery::recover_solid_revolution_fragment(&recovered[0]).unwrap();
            let kernel = crate::cad_kernel::monstertruck::MonstertruckKernel;
            let rebuilt = kernel.evaluate(&fragment.model, fragment.root).unwrap();
            assert!(kernel.summarize(&rebuilt).geometrically_consistent);
            ruststep::parser::parse(&kernel.to_step(&rebuilt).unwrap()).unwrap();
        }
    }

    #[test]
    fn recovers_native_line_surface_of_revolution_fixture() {
        let bytes =
            include_bytes!("../validation/fixtures/native_line_surface_of_revolution_frustum.step");
        let recovered = crate::detect_solid_revolutions_bytes(bytes).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(
            recovered[0].profile_points_mm,
            vec![[0.0, 0.0], [2.0, 0.0], [1.0, 2.0], [0.0, 2.0]]
        );
        assert!(recovered[0].max_residual_mm < 1.0e-9);

        let skewed = String::from_utf8_lossy(bytes).replacen(
            "CARTESIAN_POINT('',(2.,-4.898587196589E-16,0.))",
            "CARTESIAN_POINT('',(2.,0.25,0.))",
            1,
        );
        assert!(
            crate::detect_solid_revolutions_bytes(skewed.as_bytes())
                .unwrap()
                .is_empty()
        );

        #[cfg(feature = "cad-kernel-monstertruck")]
        {
            use crate::cad_kernel::CadKernel;
            let fragment =
                crate::cad_recovery::recover_solid_revolution_fragment(&recovered[0]).unwrap();
            let kernel = crate::cad_kernel::monstertruck::MonstertruckKernel;
            let rebuilt = kernel.evaluate(&fragment.model, fragment.root).unwrap();
            assert!(kernel.summarize(&rebuilt).geometrically_consistent);
        }
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
