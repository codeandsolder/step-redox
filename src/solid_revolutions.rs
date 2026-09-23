use crate::brep::{self, CurveSupport, SurfaceSupport};
use crate::instances::{build_index, entity_id, simple_record};
use crate::profile_curves::RecoveredProfileCurve;
use ruststep::ast::EntityInstance;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};

const GEOM_TOL_MM: f64 = 1.0e-7;
const DIR_TOL: f64 = 1.0e-9;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecoveredSolidRevolution {
    pub solid_id: u64,
    pub face_ids: Vec<u64>,
    /// Closed meridian profile in local [radius, axial] coordinates.
    pub profile_curves: Vec<RecoveredProfileCurve>,
    /// Canonical closest point on the revolution axis to global origin.
    pub axis_origin_mm: [f64; 3],
    /// Canonical axis direction.
    pub axis_direction: [f64; 3],
    /// Deterministic local +X radial direction, perpendicular to the axis.
    pub radial_direction: [f64; 3],
    pub max_residual_mm: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SolidSurfaceSignature {
    pub solid_id: u64,
    pub face_count: usize,
    pub support_counts: BTreeMap<String, usize>,
    pub faces: Vec<SolidFaceSignature>,
    pub unique_edge_count: usize,
    pub edge_use_count: usize,
    pub closed_two_manifold: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SolidFaceSignature {
    pub face_id: u64,
    pub support: String,
    pub geometry: SolidSurfaceGeometrySignature,
    pub loops: Vec<SolidLoopSignature>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SolidSurfaceGeometrySignature {
    Plane {
        origin_mm: [f64; 3],
        normal: [f64; 3],
        max_residual_mm: f64,
    },
    Cylinder {
        axis_origin_mm: [f64; 3],
        axis: [f64; 3],
        radius_mm: f64,
    },
    Cone {
        reference_origin_mm: [f64; 3],
        axis: [f64; 3],
        reference_radius_mm: f64,
        semi_angle_rad: f64,
    },
    Sphere {
        center_mm: [f64; 3],
        axis: [f64; 3],
        radius_mm: f64,
    },
    Torus {
        center_mm: [f64; 3],
        axis: [f64; 3],
        major_radius_mm: f64,
        minor_radius_mm: f64,
    },
    Revolution {
        axis_origin_mm: [f64; 3],
        axis: [f64; 3],
        swept_curve_id: u64,
    },
    SplineExtrusion {
        extrusion_mm: [f64; 3],
        max_residual_mm: f64,
    },
    Other {
        entity_id: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SolidLoopSignature {
    pub outer: bool,
    pub edge_count: usize,
    pub curve_counts: BTreeMap<String, usize>,
}

impl RecoveredSolidRevolution {
    pub fn polygon_points(&self) -> Option<Vec<[f64; 2]>> {
        if self.profile_curves.len() < 3 {
            return None;
        }
        let mut points = Vec::with_capacity(self.profile_curves.len());
        let mut previous_end = None;
        for curve in &self.profile_curves {
            let RecoveredProfileCurve::Line {
                start_mm, end_mm, ..
            } = curve
            else {
                return None;
            };
            if previous_end.is_some_and(|end| distance2(end, *start_mm) > GEOM_TOL_MM) {
                return None;
            }
            points.push(*start_mm);
            previous_end = Some(*end_mm);
        }
        if previous_end.is_some_and(|end| distance2(end, points[0]) > GEOM_TOL_MM) {
            return None;
        }
        Some(points)
    }
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

pub fn detect_solid_surface_signatures(entities: &[EntityInstance]) -> Vec<SolidSurfaceSignature> {
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
        let Some(face_ids) = brep::solid_face_ids(solid_id, entities, &index) else {
            continue;
        };
        let Some(faces) = face_ids
            .iter()
            .copied()
            .map(|face_id| {
                let surface_id = brep::face_surface(face_id, entities, &index)?;
                let surface = brep::surface_support(surface_id, entities, &index);
                let loops = brep::face_loops(face_id, entities, &index)?;
                Some((face_id, surface, loops))
            })
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };

        let mut support_counts = BTreeMap::new();
        let mut signatures = Vec::with_capacity(faces.len());
        let mut edge_faces = HashMap::<u64, Vec<usize>>::new();
        let mut unique_edges = HashSet::new();
        let mut edge_use_count = 0_usize;
        for (face_index, (face_id, surface, loops)) in faces.iter().enumerate() {
            let support = surface_kind(surface).to_owned();
            *support_counts.entry(support.clone()).or_insert(0) += 1;
            let loop_signatures = loops
                .iter()
                .map(|loop_| {
                    let mut curve_counts = BTreeMap::new();
                    for edge in &loop_.edges {
                        unique_edges.insert(edge.edge_id);
                        edge_faces.entry(edge.edge_id).or_default().push(face_index);
                        edge_use_count += 1;
                        *curve_counts
                            .entry(curve_kind(&edge.support).to_owned())
                            .or_insert(0) += 1;
                    }
                    SolidLoopSignature {
                        outer: loop_.outer,
                        edge_count: loop_.edges.len(),
                        curve_counts,
                    }
                })
                .collect();
            signatures.push(SolidFaceSignature {
                face_id: *face_id,
                support,
                geometry: surface_geometry(surface),
                loops: loop_signatures,
            });
        }

        let closed_two_manifold = edge_faces.values().all(|attached| attached.len() == 2)
            && shell_faces_connected(faces.len(), &edge_faces);
        out.push(SolidSurfaceSignature {
            solid_id,
            face_count: face_ids.len(),
            support_counts,
            faces: signatures,
            unique_edge_count: unique_edges.len(),
            edge_use_count,
            closed_two_manifold,
        });
    }
    out.sort_by_key(|signature| signature.solid_id);
    out
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
    let context = TopologyContext {
        faces: &faces,
        edge_faces: &edge_faces,
        entities,
        index,
    };

    if let Some(torus) = detect_full_ring_torus(solid_id, &face_ids, &faces) {
        return Some(torus);
    }
    if let Some(cap) = detect_spherical_cap(solid_id, &face_ids, &faces) {
        return Some(cap);
    }
    if let Some(end) = detect_hemispherical_end(solid_id, &face_ids, &faces, &context) {
        return Some(end);
    }
    if let Some(mixed) = detect_mixed_torus_revolution(solid_id, &face_ids, &faces, &context) {
        return Some(mixed);
    }
    if face_ids.len() < 3 {
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
    let profile_curves = line_profile_curves(&profile_points_mm);
    Some(RecoveredSolidRevolution {
        solid_id,
        face_ids,
        profile_curves,
        axis_origin_mm,
        axis_direction,
        radial_direction,
        max_residual_mm,
    })
}

fn surface_geometry(surface: &SurfaceSupport) -> SolidSurfaceGeometrySignature {
    match surface {
        SurfaceSupport::Plane(plane) => SolidSurfaceGeometrySignature::Plane {
            origin_mm: plane.origin_mm,
            normal: plane.normal,
            max_residual_mm: plane.max_residual_mm,
        },
        SurfaceSupport::Cylinder(cylinder) => SolidSurfaceGeometrySignature::Cylinder {
            axis_origin_mm: cylinder.axis_origin_mm,
            axis: cylinder.axis,
            radius_mm: cylinder.radius_mm,
        },
        SurfaceSupport::Cone(cone) => SolidSurfaceGeometrySignature::Cone {
            reference_origin_mm: cone.reference_origin_mm,
            axis: cone.axis,
            reference_radius_mm: cone.reference_radius_mm,
            semi_angle_rad: cone.semi_angle_rad,
        },
        SurfaceSupport::Sphere(sphere) => SolidSurfaceGeometrySignature::Sphere {
            center_mm: sphere.center_mm,
            axis: sphere.axis,
            radius_mm: sphere.radius_mm,
        },
        SurfaceSupport::Torus(torus) => SolidSurfaceGeometrySignature::Torus {
            center_mm: torus.center_mm,
            axis: torus.axis,
            major_radius_mm: torus.major_radius_mm,
            minor_radius_mm: torus.minor_radius_mm,
        },
        SurfaceSupport::Revolution(revolution) => SolidSurfaceGeometrySignature::Revolution {
            axis_origin_mm: revolution.axis_origin_mm,
            axis: revolution.axis,
            swept_curve_id: revolution.swept_curve_id,
        },
        SurfaceSupport::SplineExtrusion(spline) => SolidSurfaceGeometrySignature::SplineExtrusion {
            extrusion_mm: spline.extrusion_mm,
            max_residual_mm: spline.max_residual_mm,
        },
        SurfaceSupport::Other { entity_id } => SolidSurfaceGeometrySignature::Other {
            entity_id: *entity_id,
        },
    }
}

fn curve_kind(curve: &CurveSupport) -> &'static str {
    match curve {
        CurveSupport::Line(_) => "line",
        CurveSupport::Circle(_) => "circle",
        CurveSupport::BSpline(_) => "bspline",
        CurveSupport::Other { .. } => "other",
    }
}

fn surface_kind(surface: &SurfaceSupport) -> &'static str {
    match surface {
        SurfaceSupport::Plane(_) => "plane",
        SurfaceSupport::Cylinder(_) => "cylinder",
        SurfaceSupport::Cone(_) => "cone",
        SurfaceSupport::Sphere(_) => "sphere",
        SurfaceSupport::Torus(_) => "torus",
        SurfaceSupport::Revolution(_) => "surface_of_revolution",
        SurfaceSupport::SplineExtrusion(_) => "spline_extrusion",
        SurfaceSupport::Other { .. } => "other",
    }
}

fn detect_full_ring_torus(
    solid_id: u64,
    face_ids: &[u64],
    faces: &[FaceInfo],
) -> Option<RecoveredSolidRevolution> {
    let first = match faces.first()?.surface {
        SurfaceSupport::Torus(torus) => torus,
        _ => return None,
    };
    if !first.major_radius_mm.is_finite()
        || !first.minor_radius_mm.is_finite()
        || first.minor_radius_mm <= GEOM_TOL_MM
        || first.major_radius_mm <= first.minor_radius_mm + GEOM_TOL_MM
    {
        return None;
    }

    let axis_direction = canonical_axis(first.axis);
    let axis_origin_mm = closest_axis_point_to_global_origin(first.center_mm, axis_direction);
    let center_t = axial_coordinate(first.center_mm, axis_origin_mm, axis_direction);
    let radial_direction = radial_basis(axis_direction)?;
    let mut max_residual_mm = axis_distance(first.center_mm, axis_origin_mm, axis_direction);

    for face in faces {
        let SurfaceSupport::Torus(torus) = face.surface else {
            return None;
        };
        if !torus.major_radius_mm.is_finite()
            || !torus.minor_radius_mm.is_finite()
            || !parallel(torus.axis, axis_direction)
        {
            return None;
        }
        max_residual_mm = max_residual_mm
            .max(norm(sub(torus.center_mm, first.center_mm)))
            .max((torus.major_radius_mm - first.major_radius_mm).abs())
            .max((torus.minor_radius_mm - first.minor_radius_mm).abs());

        for edge in face.loops.iter().flat_map(|loop_| &loop_.edges) {
            let CurveSupport::Circle(circle) = edge.support else {
                return None;
            };
            for point in [edge.start_mm, edge.end_mm] {
                max_residual_mm = max_residual_mm
                    .max(circle_point_residual(point, circle))
                    .max(torus_point_residual(point, first, axis_direction));
            }

            let circle_y = normalize(cross(circle.normal, circle.x_direction))?;
            for sample in 0..16 {
                let angle = std::f64::consts::TAU * sample as f64 / 16.0;
                let point = add(
                    circle.center_mm,
                    add(
                        mul(circle.x_direction, circle.radius_mm * angle.cos()),
                        mul(circle_y, circle.radius_mm * angle.sin()),
                    ),
                );
                max_residual_mm =
                    max_residual_mm.max(torus_point_residual(point, first, axis_direction));
            }
        }
    }
    if max_residual_mm > GEOM_TOL_MM {
        return None;
    }

    Some(RecoveredSolidRevolution {
        solid_id,
        face_ids: face_ids.to_vec(),
        profile_curves: vec![RecoveredProfileCurve::CircleArc {
            source_edge_ids: Vec::new(),
            center_mm: [first.major_radius_mm, center_t],
            radius_mm: first.minor_radius_mm,
            start_angle_rad: 0.0,
            end_angle_rad: std::f64::consts::TAU,
        }],
        axis_origin_mm,
        axis_direction,
        radial_direction,
        max_residual_mm,
    })
}

fn detect_spherical_cap(
    solid_id: u64,
    face_ids: &[u64],
    faces: &[FaceInfo],
) -> Option<RecoveredSolidRevolution> {
    if faces.len() < 2 {
        return None;
    }

    let mut sphere_faces = Vec::new();
    let mut plane_face = None;
    for face in faces {
        match face.surface {
            SurfaceSupport::Sphere(sphere) => sphere_faces.push((face, sphere)),
            SurfaceSupport::Plane(plane) if plane_face.is_none() => {
                plane_face = Some((face, plane))
            }
            _ => return None,
        }
    }
    let (plane_face, plane) = plane_face?;
    let (_, first_sphere) = *sphere_faces.first()?;
    if !first_sphere.radius_mm.is_finite() || first_sphere.radius_mm <= GEOM_TOL_MM {
        return None;
    }

    let axis_direction = canonical_axis(normalize(plane.normal)?);
    let axis_origin_mm =
        closest_axis_point_to_global_origin(first_sphere.center_mm, axis_direction);
    let center_t = axial_coordinate(first_sphere.center_mm, axis_origin_mm, axis_direction);
    let plane_t = axial_coordinate(plane.origin_mm, axis_origin_mm, axis_direction);
    let plane_offset = plane_t - center_t;
    let cap_radius_squared = first_sphere.radius_mm.powi(2) - plane_offset.powi(2);
    if !cap_radius_squared.is_finite() || cap_radius_squared <= GEOM_TOL_MM.powi(2) {
        return None;
    }
    let cap_radius = cap_radius_squared.sqrt();
    let radial_direction = radial_basis(axis_direction)?;
    let mut max_residual_mm = plane.max_residual_mm;
    let mut side = 0_i8;
    let mut reached_pole = false;
    let mut sphere_cap_edges = 0_usize;

    for (face, sphere) in &sphere_faces {
        if face.loops.len() != 1
            || !sphere.radius_mm.is_finite()
            || (sphere.radius_mm - first_sphere.radius_mm).abs() > GEOM_TOL_MM
        {
            return None;
        }
        max_residual_mm = max_residual_mm
            .max(norm(sub(sphere.center_mm, first_sphere.center_mm)))
            .max((sphere.radius_mm - first_sphere.radius_mm).abs());

        for edge in &face.loops[0].edges {
            let CurveSupport::Circle(circle) = edge.support else {
                return None;
            };
            max_residual_mm = max_residual_mm.max(sphere_circle_residual(circle, first_sphere)?);
            if cap_circle_residual(circle, axis_origin_mm, axis_direction, plane_t, cap_radius)
                .is_some_and(|residual| residual <= GEOM_TOL_MM)
            {
                sphere_cap_edges += 1;
            }

            for point in [edge.start_mm, edge.end_mm] {
                max_residual_mm = max_residual_mm
                    .max(circle_point_residual(point, circle))
                    .max(sphere_point_residual(point, first_sphere));
                let signed = axial_coordinate(point, axis_origin_mm, axis_direction) - plane_t;
                if signed.abs() > GEOM_TOL_MM {
                    let point_side = if signed > 0.0 { 1 } else { -1 };
                    if side != 0 && side != point_side {
                        return None;
                    }
                    side = point_side;
                }
            }
        }
    }

    if plane_face.loops.len() != 1 || plane_face.loops[0].edges.is_empty() {
        return None;
    }
    for edge in &plane_face.loops[0].edges {
        let CurveSupport::Circle(circle) = edge.support else {
            return None;
        };
        max_residual_mm = max_residual_mm.max(cap_circle_residual(
            circle,
            axis_origin_mm,
            axis_direction,
            plane_t,
            cap_radius,
        )?);
        for point in [edge.start_mm, edge.end_mm] {
            max_residual_mm = max_residual_mm
                .max(circle_point_residual(point, circle))
                .max(sphere_point_residual(point, first_sphere))
                .max((axial_coordinate(point, axis_origin_mm, axis_direction) - plane_t).abs());
        }
    }

    if side == 0 || sphere_cap_edges == 0 || max_residual_mm > GEOM_TOL_MM {
        return None;
    }
    let pole_t = center_t + f64::from(side) * first_sphere.radius_mm;
    for (face, _) in &sphere_faces {
        for edge in &face.loops[0].edges {
            for point in [edge.start_mm, edge.end_mm] {
                if (axial_coordinate(point, axis_origin_mm, axis_direction) - pole_t).abs()
                    <= GEOM_TOL_MM
                    && point_axis_distance(point, axis_origin_mm, axis_direction) <= GEOM_TOL_MM
                {
                    reached_pole = true;
                }
            }
        }
    }
    if !reached_pole {
        return None;
    }

    let cap_angle = plane_offset.atan2(cap_radius);
    let pole_angle = if side > 0 {
        std::f64::consts::FRAC_PI_2
    } else {
        -std::f64::consts::FRAC_PI_2
    };
    Some(RecoveredSolidRevolution {
        solid_id,
        face_ids: face_ids.to_vec(),
        profile_curves: vec![
            RecoveredProfileCurve::Line {
                source_edge_ids: Vec::new(),
                start_mm: [0.0, plane_t],
                end_mm: [cap_radius, plane_t],
            },
            RecoveredProfileCurve::CircleArc {
                source_edge_ids: Vec::new(),
                center_mm: [0.0, center_t],
                radius_mm: first_sphere.radius_mm,
                start_angle_rad: cap_angle,
                end_angle_rad: pole_angle,
            },
            RecoveredProfileCurve::Line {
                source_edge_ids: Vec::new(),
                start_mm: [0.0, pole_t],
                end_mm: [0.0, plane_t],
            },
        ],
        axis_origin_mm,
        axis_direction,
        radial_direction,
        max_residual_mm,
    })
}

fn detect_hemispherical_end(
    solid_id: u64,
    face_ids: &[u64],
    faces: &[FaceInfo],
    context: &TopologyContext<'_>,
) -> Option<RecoveredSolidRevolution> {
    if faces.len() != 5 {
        return None;
    }

    let mut sphere_faces = Vec::new();
    let mut cylinder_faces = Vec::new();
    let mut plane_face = None;
    for (face_index, face) in faces.iter().enumerate() {
        match face.surface {
            SurfaceSupport::Sphere(sphere) => sphere_faces.push((face_index, face, sphere)),
            SurfaceSupport::Cylinder(cylinder) => {
                cylinder_faces.push((face_index, face, cylinder));
            }
            SurfaceSupport::Plane(plane) if plane_face.is_none() => {
                plane_face = Some((face_index, face, plane));
            }
            _ => return None,
        }
    }
    let [
        (sphere_index_a, sphere_face_a, first_sphere),
        (sphere_index_b, sphere_face_b, sphere_b),
    ] = sphere_faces.as_slice()
    else {
        return None;
    };
    let [
        (cylinder_index_a, cylinder_face_a, first_cylinder),
        (cylinder_index_b, cylinder_face_b, cylinder_b),
    ] = cylinder_faces.as_slice()
    else {
        return None;
    };
    let (plane_index, plane_face, plane) = plane_face?;

    if !first_sphere.radius_mm.is_finite()
        || first_sphere.radius_mm <= GEOM_TOL_MM
        || !first_cylinder.radius_mm.is_finite()
        || first_cylinder.radius_mm <= GEOM_TOL_MM
    {
        return None;
    }

    let axis_direction = canonical_axis(normalize(first_cylinder.axis)?);
    let axis_origin_mm =
        closest_axis_point_to_global_origin(first_sphere.center_mm, axis_direction);
    let radial_direction = radial_basis(axis_direction)?;
    let center_t = axial_coordinate(first_sphere.center_mm, axis_origin_mm, axis_direction);
    let plane_t = axial_coordinate(plane.origin_mm, axis_origin_mm, axis_direction);
    if (plane_t - center_t).abs() <= GEOM_TOL_MM || !parallel(plane.normal, axis_direction) {
        return None;
    }

    let radius_mm = first_sphere.radius_mm;
    let mut max_residual_mm = plane
        .max_residual_mm
        .max((first_cylinder.radius_mm - radius_mm).abs())
        .max(norm(sub(sphere_b.center_mm, first_sphere.center_mm)))
        .max((sphere_b.radius_mm - radius_mm).abs())
        .max(axis_distance(
            first_sphere.center_mm,
            axis_origin_mm,
            axis_direction,
        ))
        .max(axis_distance(
            first_cylinder.axis_origin_mm,
            axis_origin_mm,
            axis_direction,
        ))
        .max(axis_distance(
            cylinder_b.axis_origin_mm,
            axis_origin_mm,
            axis_direction,
        ))
        .max((cylinder_b.radius_mm - radius_mm).abs());
    if !parallel(first_cylinder.axis, axis_direction)
        || !parallel(cylinder_b.axis, axis_direction)
        || max_residual_mm > GEOM_TOL_MM
    {
        return None;
    }

    let (plane_segment, plane_residual) = plane_profile_segment(
        plane_index,
        plane_face,
        plane,
        axis_origin_mm,
        axis_direction,
        context,
    )?;
    max_residual_mm = max_residual_mm.max(plane_residual);
    let plane_radii = [plane_segment.a[0], plane_segment.b[0]];
    if (plane_segment.a[1] - plane_t).abs() > GEOM_TOL_MM
        || (plane_segment.b[1] - plane_t).abs() > GEOM_TOL_MM
        || !plane_radii.iter().any(|radius| radius.abs() <= GEOM_TOL_MM)
        || !plane_radii
            .iter()
            .any(|radius| (*radius - radius_mm).abs() <= GEOM_TOL_MM)
    {
        return None;
    }

    for (face_index, face, cylinder) in [
        (*cylinder_index_a, *cylinder_face_a, *first_cylinder),
        (*cylinder_index_b, *cylinder_face_b, *cylinder_b),
    ] {
        let (segment, residual) = cylinder_profile_segment(
            face_index,
            face,
            cylinder,
            axis_origin_mm,
            axis_direction,
            faces,
            context.edge_faces,
        )?;
        max_residual_mm = max_residual_mm.max(residual);
        let axial = [segment.a[1], segment.b[1]];
        if (segment.a[0] - radius_mm).abs() > GEOM_TOL_MM
            || (segment.b[0] - radius_mm).abs() > GEOM_TOL_MM
            || !axial
                .iter()
                .any(|value| (*value - center_t).abs() <= GEOM_TOL_MM)
            || !axial
                .iter()
                .any(|value| (*value - plane_t).abs() <= GEOM_TOL_MM)
        {
            return None;
        }
    }

    let plane_side = if plane_t > center_t { 1_i8 } else { -1_i8 };
    let sphere_side = -plane_side;
    let pole_t = center_t + f64::from(sphere_side) * radius_mm;
    let mut reached_pole = false;
    let mut sphere_cylinder_edges = 0_usize;
    let mut plane_cylinder_edges = 0_usize;

    for (face_index, face, _sphere) in [
        (*sphere_index_a, *sphere_face_a, *first_sphere),
        (*sphere_index_b, *sphere_face_b, *sphere_b),
    ] {
        if face.loops.len() != 1 {
            return None;
        }
        for edge in &face.loops[0].edges {
            let CurveSupport::Circle(circle) = edge.support else {
                return None;
            };
            max_residual_mm = max_residual_mm.max(sphere_circle_residual(circle, *first_sphere)?);
            let neighbor = unique_neighbor_face(face_index, edge.edge_id, context.edge_faces)?;
            match faces.get(neighbor)?.surface {
                SurfaceSupport::Sphere(neighbor_sphere) => {
                    max_residual_mm = max_residual_mm
                        .max(norm(sub(neighbor_sphere.center_mm, first_sphere.center_mm)))
                        .max((neighbor_sphere.radius_mm - radius_mm).abs());
                }
                SurfaceSupport::Cylinder(neighbor_cylinder) => {
                    max_residual_mm = max_residual_mm
                        .max((neighbor_cylinder.radius_mm - radius_mm).abs())
                        .max(cap_circle_residual(
                            circle,
                            axis_origin_mm,
                            axis_direction,
                            center_t,
                            radius_mm,
                        )?);
                    sphere_cylinder_edges += 1;
                }
                _ => return None,
            }

            for point in [edge.start_mm, edge.end_mm] {
                max_residual_mm = max_residual_mm
                    .max(circle_point_residual(point, circle))
                    .max(sphere_point_residual(point, *first_sphere));
                let signed = axial_coordinate(point, axis_origin_mm, axis_direction) - center_t;
                if signed.abs() > GEOM_TOL_MM && signed.signum() != f64::from(sphere_side) {
                    return None;
                }
                if (axial_coordinate(point, axis_origin_mm, axis_direction) - pole_t).abs()
                    <= GEOM_TOL_MM
                    && point_axis_distance(point, axis_origin_mm, axis_direction) <= GEOM_TOL_MM
                {
                    reached_pole = true;
                }
            }
        }
    }

    for (face_index, face, _) in [
        (*cylinder_index_a, *cylinder_face_a, *first_cylinder),
        (*cylinder_index_b, *cylinder_face_b, *cylinder_b),
    ] {
        for edge in &face.loops[0].edges {
            let neighbor = unique_neighbor_face(face_index, edge.edge_id, context.edge_faces)?;
            match faces.get(neighbor)?.surface {
                SurfaceSupport::Cylinder(_) => {
                    if !matches!(edge.support, CurveSupport::Line(_)) {
                        return None;
                    }
                }
                SurfaceSupport::Sphere(_) => {
                    let CurveSupport::Circle(circle) = edge.support else {
                        return None;
                    };
                    max_residual_mm = max_residual_mm.max(cap_circle_residual(
                        circle,
                        axis_origin_mm,
                        axis_direction,
                        center_t,
                        radius_mm,
                    )?);
                }
                SurfaceSupport::Plane(_) => {
                    let CurveSupport::Circle(circle) = edge.support else {
                        return None;
                    };
                    max_residual_mm = max_residual_mm.max(cap_circle_residual(
                        circle,
                        axis_origin_mm,
                        axis_direction,
                        plane_t,
                        radius_mm,
                    )?);
                    plane_cylinder_edges += 1;
                }
                _ => return None,
            }
        }
    }

    if sphere_cylinder_edges == 0
        || plane_cylinder_edges == 0
        || !reached_pole
        || max_residual_mm > GEOM_TOL_MM
    {
        return None;
    }

    let cap_angle = f64::from(sphere_side) * std::f64::consts::FRAC_PI_2;
    Some(RecoveredSolidRevolution {
        solid_id,
        face_ids: face_ids.to_vec(),
        profile_curves: vec![
            RecoveredProfileCurve::Line {
                source_edge_ids: Vec::new(),
                start_mm: [0.0, plane_t],
                end_mm: [radius_mm, plane_t],
            },
            RecoveredProfileCurve::Line {
                source_edge_ids: Vec::new(),
                start_mm: [radius_mm, plane_t],
                end_mm: [radius_mm, center_t],
            },
            RecoveredProfileCurve::CircleArc {
                source_edge_ids: Vec::new(),
                center_mm: [0.0, center_t],
                radius_mm,
                start_angle_rad: 0.0,
                end_angle_rad: cap_angle,
            },
            RecoveredProfileCurve::Line {
                source_edge_ids: Vec::new(),
                start_mm: [0.0, pole_t],
                end_mm: [0.0, plane_t],
            },
        ],
        axis_origin_mm,
        axis_direction,
        radial_direction,
        max_residual_mm,
    })
}

fn detect_mixed_torus_revolution(
    solid_id: u64,
    face_ids: &[u64],
    faces: &[FaceInfo],
    context: &TopologyContext<'_>,
) -> Option<RecoveredSolidRevolution> {
    let torus_count = faces
        .iter()
        .filter(|face| matches!(face.surface, SurfaceSupport::Torus(_)))
        .count();
    if torus_count == 0 || torus_count == faces.len() {
        return None;
    }

    let (axis_reference_origin_mm, axis_reference_direction) =
        faces.iter().find_map(|face| match face.surface {
            SurfaceSupport::Cylinder(cylinder) => Some((cylinder.axis_origin_mm, cylinder.axis)),
            SurfaceSupport::Cone(cone) => Some((cone.reference_origin_mm, cone.axis)),
            SurfaceSupport::Revolution(revolution) => {
                Some((revolution.axis_origin_mm, revolution.axis))
            }
            SurfaceSupport::Torus(torus) => Some((torus.center_mm, torus.axis)),
            _ => None,
        })?;
    let axis_direction = canonical_axis(normalize(axis_reference_direction)?);
    let axis_origin_mm =
        closest_axis_point_to_global_origin(axis_reference_origin_mm, axis_direction);
    let radial_direction = radial_basis(axis_direction)?;

    let mut max_residual_mm = 0.0_f64;
    let mut segments = Vec::new();
    let mut torus_faces = Vec::<(usize, brep::TorusSupport)>::new();
    for (face_index, face) in faces.iter().enumerate() {
        match face.surface {
            SurfaceSupport::Cylinder(cylinder) => {
                let (segment, residual) = cylinder_profile_segment(
                    face_index,
                    face,
                    cylinder,
                    axis_origin_mm,
                    axis_direction,
                    faces,
                    context.edge_faces,
                )?;
                max_residual_mm = max_residual_mm.max(residual);
                push_unique_segment(&mut segments, segment);
            }
            SurfaceSupport::Cone(cone) => {
                let (segment, residual) = cone_profile_segment(
                    face_index,
                    face,
                    cone,
                    axis_origin_mm,
                    axis_direction,
                    faces,
                    context.edge_faces,
                )?;
                max_residual_mm = max_residual_mm.max(residual);
                push_unique_segment(&mut segments, segment);
            }
            SurfaceSupport::Revolution(revolution) => {
                let (segment, residual) = revolution_line_profile_segment(
                    face_index,
                    face,
                    revolution,
                    axis_origin_mm,
                    axis_direction,
                    context,
                )?;
                max_residual_mm = max_residual_mm.max(residual);
                push_unique_segment(&mut segments, segment);
            }
            SurfaceSupport::Plane(plane) => {
                let (segment, residual) = plane_profile_segment(
                    face_index,
                    face,
                    plane,
                    axis_origin_mm,
                    axis_direction,
                    context,
                )?;
                max_residual_mm = max_residual_mm.max(residual);
                push_unique_segment(&mut segments, segment);
            }
            SurfaceSupport::Torus(torus) => torus_faces.push((face_index, torus)),
            _ => return None,
        }
    }
    if max_residual_mm > GEOM_TOL_MM {
        return None;
    }

    let mut groups = Vec::<Vec<usize>>::new();
    let mut supports = Vec::<brep::TorusSupport>::new();
    for (face_index, torus) in torus_faces {
        if !torus.major_radius_mm.is_finite()
            || !torus.minor_radius_mm.is_finite()
            || torus.minor_radius_mm <= GEOM_TOL_MM
            || torus.major_radius_mm <= torus.minor_radius_mm + GEOM_TOL_MM
            || !parallel(torus.axis, axis_direction)
            || axis_distance(torus.center_mm, axis_origin_mm, axis_direction) > GEOM_TOL_MM
        {
            return None;
        }
        if let Some(group_index) = supports
            .iter()
            .position(|existing| same_torus_support(*existing, torus, axis_direction))
        {
            groups[group_index].push(face_index);
        } else {
            supports.push(torus);
            groups.push(vec![face_index]);
        }
    }

    // Start with one curved support in the profile. This covers the dominant
    // torus-fillet clusters while keeping arc/arc intersections fail-closed.
    let [group] = groups.as_slice() else {
        return None;
    };
    let [torus] = supports.as_slice() else {
        return None;
    };
    let (arc, residual) = torus_profile_arc(
        group,
        *torus,
        axis_origin_mm,
        axis_direction,
        faces,
        context.edge_faces,
    )?;
    max_residual_mm = max_residual_mm.max(residual);
    if max_residual_mm > GEOM_TOL_MM {
        return None;
    }

    let mut curves = segments
        .into_iter()
        .map(|segment| RecoveredProfileCurve::Line {
            source_edge_ids: Vec::new(),
            start_mm: segment.a,
            end_mm: segment.b,
        })
        .collect::<Vec<_>>();
    curves.push(arc);
    let profile_curves = closed_profile_from_curves(curves)?;
    Some(RecoveredSolidRevolution {
        solid_id,
        face_ids: face_ids.to_vec(),
        profile_curves,
        axis_origin_mm,
        axis_direction,
        radial_direction,
        max_residual_mm,
    })
}

fn same_torus_support(a: brep::TorusSupport, b: brep::TorusSupport, axis: [f64; 3]) -> bool {
    parallel(a.axis, axis)
        && parallel(b.axis, axis)
        && norm(sub(a.center_mm, b.center_mm)) <= GEOM_TOL_MM
        && (a.major_radius_mm - b.major_radius_mm).abs() <= GEOM_TOL_MM
        && (a.minor_radius_mm - b.minor_radius_mm).abs() <= GEOM_TOL_MM
}

fn torus_profile_arc(
    face_indices: &[usize],
    torus: brep::TorusSupport,
    axis_origin_mm: [f64; 3],
    axis_direction: [f64; 3],
    faces: &[FaceInfo],
    edge_faces: &HashMap<u64, Vec<usize>>,
) -> Option<(RecoveredProfileCurve, f64)> {
    if face_indices.is_empty() || torus.major_radius_mm <= torus.minor_radius_mm + GEOM_TOL_MM {
        return None;
    }
    let center_t = axial_coordinate(torus.center_mm, axis_origin_mm, axis_direction);
    let center_mm = [torus.major_radius_mm, center_t];
    let mut boundary_points = Vec::<[f64; 2]>::new();
    let mut witness_points = Vec::<[f64; 2]>::new();
    let mut source_edge_ids = Vec::<u64>::new();
    let mut max_residual_mm = axis_distance(torus.center_mm, axis_origin_mm, axis_direction);

    for &face_index in face_indices {
        let face = faces.get(face_index)?;
        let SurfaceSupport::Torus(face_torus) = face.surface else {
            return None;
        };
        if !same_torus_support(torus, face_torus, axis_direction) || face.loops.len() != 1 {
            return None;
        }
        for edge in &face.loops[0].edges {
            let CurveSupport::Circle(circle) = edge.support else {
                return None;
            };
            source_edge_ids.push(edge.edge_id);
            for point in [edge.start_mm, edge.end_mm] {
                max_residual_mm = max_residual_mm
                    .max(circle_point_residual(point, circle))
                    .max(torus_point_residual(point, torus, axis_direction));
            }
            let neighbor = unique_neighbor_face(face_index, edge.edge_id, edge_faces)?;
            match faces.get(neighbor)?.surface {
                SurfaceSupport::Torus(neighbor_torus)
                    if same_torus_support(torus, neighbor_torus, axis_direction) =>
                {
                    max_residual_mm = max_residual_mm.max(torus_meridian_circle_residual(
                        circle,
                        torus,
                        axis_origin_mm,
                        axis_direction,
                    )?);
                    let midpoint = circle_trim_midpoint(edge, circle)?;
                    max_residual_mm =
                        max_residual_mm.max(torus_point_residual(midpoint, torus, axis_direction));
                    push_unique_point(
                        &mut witness_points,
                        [
                            point_axis_distance(midpoint, axis_origin_mm, axis_direction),
                            axial_coordinate(midpoint, axis_origin_mm, axis_direction),
                        ],
                    );
                }
                _ => {
                    let (point, residual) = torus_parallel_boundary_point(
                        circle,
                        torus,
                        axis_origin_mm,
                        axis_direction,
                    )?;
                    max_residual_mm = max_residual_mm.max(residual);
                    push_unique_point(&mut boundary_points, point);
                }
            }
        }
    }

    let [start, end] = boundary_points.as_slice() else {
        return None;
    };
    if witness_points.is_empty() || max_residual_mm > GEOM_TOL_MM {
        return None;
    }
    let start_angle = meridian_angle(*start, center_mm, torus.minor_radius_mm)?;
    let end_angle = meridian_angle(*end, center_mm, torus.minor_radius_mm)?;
    let ccw_delta = positive_angle_delta(start_angle, end_angle);
    let cw_delta = std::f64::consts::TAU - ccw_delta;
    let angle_tol = (GEOM_TOL_MM / torus.minor_radius_mm).max(1.0e-12);
    if ccw_delta <= angle_tol || cw_delta <= angle_tol {
        return None;
    }
    let candidates = [
        (start_angle, start_angle + ccw_delta),
        (start_angle, start_angle - cw_delta),
    ];
    let valid = candidates
        .into_iter()
        .filter(|&(arc_start, arc_end)| {
            witness_points.iter().all(|point| {
                meridian_angle(*point, center_mm, torus.minor_radius_mm)
                    .is_some_and(|angle| angle_on_arc(angle, arc_start, arc_end, angle_tol))
            })
        })
        .collect::<Vec<_>>();
    let [(start_angle_rad, end_angle_rad)] = valid.as_slice() else {
        return None;
    };

    source_edge_ids.sort_unstable();
    source_edge_ids.dedup();
    Some((
        RecoveredProfileCurve::CircleArc {
            source_edge_ids,
            center_mm,
            radius_mm: torus.minor_radius_mm,
            start_angle_rad: *start_angle_rad,
            end_angle_rad: *end_angle_rad,
        },
        max_residual_mm,
    ))
}

fn torus_parallel_boundary_point(
    circle: brep::CircleSupport,
    torus: brep::TorusSupport,
    axis_origin_mm: [f64; 3],
    axis_direction: [f64; 3],
) -> Option<([f64; 2], f64)> {
    if !parallel(circle.normal, axis_direction) || circle.radius_mm <= GEOM_TOL_MM {
        return None;
    }
    let axial = axial_coordinate(circle.center_mm, axis_origin_mm, axis_direction);
    let radial = circle.radius_mm;
    let center_t = axial_coordinate(torus.center_mm, axis_origin_mm, axis_direction);
    let meridian_residual =
        (((radial - torus.major_radius_mm).powi(2) + (axial - center_t).powi(2)).sqrt()
            - torus.minor_radius_mm)
            .abs();
    let residual = axis_distance(circle.center_mm, axis_origin_mm, axis_direction)
        .max(meridian_residual)
        .max((norm(circle.normal) - 1.0).abs());
    (residual <= GEOM_TOL_MM).then_some(([radial, axial], residual))
}

fn torus_meridian_circle_residual(
    circle: brep::CircleSupport,
    torus: brep::TorusSupport,
    axis_origin_mm: [f64; 3],
    axis_direction: [f64; 3],
) -> Option<f64> {
    let normal = normalize(circle.normal)?;
    if dot(normal, axis_direction).abs() > DIR_TOL {
        return None;
    }
    let center_t = axial_coordinate(torus.center_mm, axis_origin_mm, axis_direction);
    let axis_point = add(axis_origin_mm, mul(axis_direction, center_t));
    let plane_residual = dot(sub(axis_point, circle.center_mm), normal).abs();
    let mut residual = (circle.radius_mm - torus.minor_radius_mm)
        .abs()
        .max((axial_coordinate(circle.center_mm, axis_origin_mm, axis_direction) - center_t).abs())
        .max(
            (point_axis_distance(circle.center_mm, axis_origin_mm, axis_direction)
                - torus.major_radius_mm)
                .abs(),
        )
        .max(plane_residual)
        .max((norm(circle.normal) - 1.0).abs());

    let x_direction = normalize(circle.x_direction)?;
    if dot(x_direction, normal).abs() > DIR_TOL {
        return None;
    }
    let y_direction = normalize(cross(normal, x_direction))?;
    for sample in 0..16 {
        let angle = std::f64::consts::TAU * sample as f64 / 16.0;
        let point = add(
            circle.center_mm,
            add(
                mul(x_direction, circle.radius_mm * angle.cos()),
                mul(y_direction, circle.radius_mm * angle.sin()),
            ),
        );
        residual = residual.max(torus_point_residual(point, torus, axis_direction));
    }
    (residual <= GEOM_TOL_MM).then_some(residual)
}

fn circle_trim_midpoint(
    edge: &brep::OrientedEdgeUse,
    circle: brep::CircleSupport,
) -> Option<[f64; 3]> {
    let normal = normalize(circle.normal)?;
    let x_direction = normalize(circle.x_direction)?;
    if dot(normal, x_direction).abs() > DIR_TOL {
        return None;
    }
    let y_direction = normalize(cross(normal, x_direction))?;
    let parameter = |point: [f64; 3]| {
        let relative = sub(point, circle.center_mm);
        dot(relative, y_direction).atan2(dot(relative, x_direction))
    };
    let start = parameter(edge.start_mm);
    let end = parameter(edge.end_mm);
    let delta = if edge.parameter_forward {
        positive_angle_delta(start, end)
    } else {
        -positive_angle_delta(end, start)
    };
    let angle_tol = (GEOM_TOL_MM / circle.radius_mm.max(GEOM_TOL_MM)).max(1.0e-12);
    if delta.abs() <= angle_tol || (std::f64::consts::TAU - delta.abs()) <= angle_tol {
        return None;
    }
    let midpoint = start + 0.5 * delta;
    Some(add(
        circle.center_mm,
        add(
            mul(x_direction, circle.radius_mm * midpoint.cos()),
            mul(y_direction, circle.radius_mm * midpoint.sin()),
        ),
    ))
}

fn meridian_angle(point: [f64; 2], center: [f64; 2], radius: f64) -> Option<f64> {
    if !point[0].is_finite()
        || !point[1].is_finite()
        || !radius.is_finite()
        || radius <= GEOM_TOL_MM
        || (distance2(point, center) - radius).abs() > GEOM_TOL_MM
    {
        return None;
    }
    Some((point[1] - center[1]).atan2(point[0] - center[0]))
}

fn positive_angle_delta(start: f64, end: f64) -> f64 {
    (end - start).rem_euclid(std::f64::consts::TAU)
}

fn angle_on_arc(angle: f64, start: f64, end: f64, tolerance: f64) -> bool {
    if end >= start {
        positive_angle_delta(start, angle) <= end - start + tolerance
    } else {
        positive_angle_delta(angle, start) <= start - end + tolerance
    }
}

fn push_unique_point(points: &mut Vec<[f64; 2]>, candidate: [f64; 2]) {
    if !points
        .iter()
        .any(|point| distance2(*point, candidate) <= GEOM_TOL_MM)
    {
        points.push(candidate);
    }
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

fn closed_profile_from_curves(
    mut curves: Vec<RecoveredProfileCurve>,
) -> Option<Vec<RecoveredProfileCurve>> {
    if curves.len() < 3
        || curves
            .iter()
            .filter(|curve| matches!(curve, RecoveredProfileCurve::CircleArc { .. }))
            .count()
            != 1
        || curves.iter().any(RecoveredProfileCurve::is_spline)
    {
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
                    return true;
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

fn line_profile_curves(points: &[[f64; 2]]) -> Vec<RecoveredProfileCurve> {
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

fn circle_point_residual(point: [f64; 3], circle: brep::CircleSupport) -> f64 {
    let relative = sub(point, circle.center_mm);
    let plane_residual = dot(relative, circle.normal).abs();
    let radius_residual = (norm(relative) - circle.radius_mm).abs();
    plane_residual.max(radius_residual)
}

fn sphere_point_residual(point: [f64; 3], sphere: brep::SphereSupport) -> f64 {
    (norm(sub(point, sphere.center_mm)) - sphere.radius_mm).abs()
}

fn sphere_circle_residual(circle: brep::CircleSupport, sphere: brep::SphereSupport) -> Option<f64> {
    let normal = normalize(circle.normal)?;
    let relative = sub(circle.center_mm, sphere.center_mm);
    let offset = dot(relative, normal);
    let lateral = norm(sub(relative, mul(normal, offset)));
    let expected_squared = sphere.radius_mm.powi(2) - offset.powi(2);
    if expected_squared < -GEOM_TOL_MM.powi(2) {
        return None;
    }
    let expected_radius = expected_squared.max(0.0).sqrt();
    Some(
        lateral
            .max((circle.radius_mm - expected_radius).abs())
            .max((norm(circle.normal) - 1.0).abs()),
    )
}

fn cap_circle_residual(
    circle: brep::CircleSupport,
    axis_origin: [f64; 3],
    axis: [f64; 3],
    plane_t: f64,
    radius: f64,
) -> Option<f64> {
    if !parallel(circle.normal, axis) {
        return None;
    }
    Some(
        axis_distance(circle.center_mm, axis_origin, axis)
            .max((axial_coordinate(circle.center_mm, axis_origin, axis) - plane_t).abs())
            .max((circle.radius_mm - radius).abs()),
    )
}

fn torus_point_residual(point: [f64; 3], torus: brep::TorusSupport, axis: [f64; 3]) -> f64 {
    let relative = sub(point, torus.center_mm);
    let axial = dot(relative, axis);
    let radial_vector = sub(relative, mul(axis, axial));
    let radial = norm(radial_vector);
    let tube_distance = ((radial - torus.major_radius_mm).powi(2) + axial.powi(2)).sqrt();
    (tube_distance - torus.minor_radius_mm).abs()
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

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
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

    fn test_circle_edge(
        edge_id: u64,
        start_mm: [f64; 3],
        end_mm: [f64; 3],
        center_mm: [f64; 3],
        normal: [f64; 3],
        radius_mm: f64,
    ) -> brep::OrientedEdgeUse {
        brep::OrientedEdgeUse {
            oriented_edge_id: edge_id + 10_000,
            edge_id,
            curve_id: edge_id + 20_000,
            curve_same_sense: true,
            parameter_forward: true,
            start_vertex: edge_id + 30_000,
            end_vertex: edge_id + 40_000,
            start_mm,
            end_mm,
            support: CurveSupport::Circle(brep::CircleSupport {
                center_mm,
                normal,
                x_direction: [1.0, 0.0, 0.0],
                radius_mm,
            }),
        }
    }

    fn test_line_edge(edge_id: u64, start_mm: [f64; 3], end_mm: [f64; 3]) -> brep::OrientedEdgeUse {
        brep::OrientedEdgeUse {
            oriented_edge_id: edge_id + 10_000,
            edge_id,
            curve_id: edge_id + 20_000,
            curve_same_sense: true,
            parameter_forward: true,
            start_vertex: edge_id + 30_000,
            end_vertex: edge_id + 40_000,
            start_mm,
            end_mm,
            support: CurveSupport::Line(brep::LineSupport {
                origin_mm: start_mm,
                direction: normalize(sub(end_mm, start_mm)).unwrap(),
            }),
        }
    }

    fn test_face(surface: SurfaceSupport, edges: Vec<brep::OrientedEdgeUse>) -> FaceInfo {
        FaceInfo {
            surface,
            loops: vec![brep::FaceLoop {
                bound_id: 1,
                loop_id: 2,
                outer: true,
                orientation: true,
                edges,
            }],
        }
    }

    fn split_hemisphere_faces(cylinder_radius_mm: f64) -> Vec<FaceInfo> {
        let sphere = brep::SphereSupport {
            center_mm: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 1.0],
            x_direction: [1.0, 0.0, 0.0],
            radius_mm: 1.0,
        };
        let cylinder = brep::CylinderSupport {
            axis_origin_mm: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 1.0],
            x_direction: [1.0, 0.0, 0.0],
            radius_mm: cylinder_radius_mm,
        };
        let plane = brep::PlaneSupport {
            origin_mm: [0.0, 0.0, 2.0],
            normal: [0.0, 0.0, 1.0],
            max_residual_mm: 0.0,
        };

        let equator_a = test_circle_edge(
            10,
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            1.0,
        );
        let equator_b = test_circle_edge(
            11,
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            1.0,
        );
        let sphere_seam_a = test_circle_edge(
            12,
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
        );
        let sphere_seam_b = test_circle_edge(
            13,
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
        );
        let disk_a = test_circle_edge(
            14,
            [1.0, 0.0, 2.0],
            [-1.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0],
            1.0,
        );
        let disk_b = test_circle_edge(
            15,
            [-1.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0],
            1.0,
        );
        let cylinder_seam_a = test_line_edge(16, [1.0, 0.0, 0.0], [1.0, 0.0, 2.0]);
        let cylinder_seam_b = test_line_edge(17, [-1.0, 0.0, 0.0], [-1.0, 0.0, 2.0]);

        vec![
            test_face(
                SurfaceSupport::Sphere(sphere),
                vec![
                    equator_a.clone(),
                    sphere_seam_a.clone(),
                    sphere_seam_b.clone(),
                ],
            ),
            test_face(
                SurfaceSupport::Sphere(sphere),
                vec![equator_b.clone(), sphere_seam_a, sphere_seam_b],
            ),
            test_face(
                SurfaceSupport::Cylinder(cylinder),
                vec![
                    equator_a,
                    disk_a.clone(),
                    cylinder_seam_a.clone(),
                    cylinder_seam_b.clone(),
                ],
            ),
            test_face(
                SurfaceSupport::Cylinder(cylinder),
                vec![equator_b, disk_b.clone(), cylinder_seam_a, cylinder_seam_b],
            ),
            test_face(SurfaceSupport::Plane(plane), vec![disk_a, disk_b]),
        ]
    }

    fn edge_face_map(faces: &[FaceInfo]) -> HashMap<u64, Vec<usize>> {
        let mut edge_faces = HashMap::<u64, Vec<usize>>::new();
        for (face_index, face) in faces.iter().enumerate() {
            for edge in face.loops.iter().flat_map(|loop_| &loop_.edges) {
                edge_faces.entry(edge.edge_id).or_default().push(face_index);
            }
        }
        edge_faces
    }

    #[test]
    fn recovers_split_hemispherical_end_topology() {
        let faces = split_hemisphere_faces(1.0);
        let edge_faces = edge_face_map(&faces);
        let entities = Vec::new();
        let index = HashMap::new();
        let context = TopologyContext {
            faces: &faces,
            edge_faces: &edge_faces,
            entities: &entities,
            index: &index,
        };
        let recovered = detect_hemispherical_end(99, &[1, 2, 3, 4, 5], &faces, &context).unwrap();

        assert_eq!(recovered.profile_curves.len(), 4);
        assert!(recovered.max_residual_mm <= 1.0e-12);
        assert_eq!(recovered.axis_direction, [0.0, 0.0, 1.0]);

        assert!(matches!(
            recovered.profile_curves.as_slice(),
            [
                RecoveredProfileCurve::Line {
                    start_mm: [0.0, 2.0],
                    end_mm: [1.0, 2.0],
                    ..
                },
                RecoveredProfileCurve::Line {
                    start_mm: [1.0, 2.0],
                    end_mm: [1.0, 0.0],
                    ..
                },
                RecoveredProfileCurve::CircleArc {
                    center_mm: [0.0, 0.0],
                    radius_mm: 1.0,
                    start_angle_rad: 0.0,
                    end_angle_rad,
                    ..
                },
                RecoveredProfileCurve::Line {
                    start_mm: [0.0, -1.0],
                    end_mm: [0.0, 2.0],
                    ..
                }
            ] if (*end_angle_rad + std::f64::consts::FRAC_PI_2).abs() <= 1.0e-12
        ));

        #[cfg(feature = "cad-kernel-monstertruck")]
        {
            use crate::cad_kernel::CadKernel;
            let fragment =
                crate::cad_recovery::recover_solid_revolution_fragment(&recovered).unwrap();
            let kernel = crate::cad_kernel::monstertruck::MonstertruckKernel;
            let rebuilt = kernel.evaluate(&fragment.model, fragment.root).unwrap();
            assert!(kernel.summarize(&rebuilt).geometrically_consistent);
            ruststep::parser::parse(&kernel.to_step(&rebuilt).unwrap()).unwrap();
        }

        let tampered = split_hemisphere_faces(1.01);
        let edge_faces = edge_face_map(&tampered);
        let context = TopologyContext {
            faces: &tampered,
            edge_faces: &edge_faces,
            entities: &entities,
            index: &index,
        };
        assert!(detect_hemispherical_end(99, &[1, 2, 3, 4, 5], &tampered, &context).is_none());
    }

    fn quarter_fillet_profile() -> Vec<RecoveredProfileCurve> {
        vec![
            RecoveredProfileCurve::Line {
                source_edge_ids: Vec::new(),
                start_mm: [0.0, 1.0],
                end_mm: [1.0, 1.0],
            },
            RecoveredProfileCurve::Line {
                source_edge_ids: Vec::new(),
                start_mm: [1.0, 1.0],
                end_mm: [1.0, 1.4],
            },
            RecoveredProfileCurve::CircleArc {
                source_edge_ids: Vec::new(),
                center_mm: [0.9, 1.4],
                radius_mm: 0.1,
                start_angle_rad: 0.0,
                end_angle_rad: std::f64::consts::FRAC_PI_2,
            },
            RecoveredProfileCurve::Line {
                source_edge_ids: Vec::new(),
                start_mm: [0.9, 1.5],
                end_mm: [0.0, 1.5],
            },
            RecoveredProfileCurve::Line {
                source_edge_ids: Vec::new(),
                start_mm: [0.0, 1.5],
                end_mm: [0.0, 1.0],
            },
        ]
    }

    #[test]
    fn orders_mixed_line_arc_profile_and_rejects_line_arc_crossing() {
        let profile = quarter_fillet_profile();
        let scrambled = vec![
            profile[2].reversed(),
            profile[4].clone(),
            profile[1].clone(),
            profile[3].reversed(),
            profile[0].clone(),
        ];
        let ordered = closed_profile_from_curves(scrambled).unwrap();
        assert_eq!(ordered.len(), 5);
        assert!(matches!(
            &ordered[2],
            RecoveredProfileCurve::CircleArc {
                center_mm: [0.9, 1.4],
                radius_mm: 0.1,
                start_angle_rad,
                end_angle_rad,
                ..
            } if start_angle_rad.abs() <= 1.0e-12
                && (*end_angle_rad - std::f64::consts::FRAC_PI_2).abs() <= 1.0e-12
        ));

        let crossing = RecoveredProfileCurve::Line {
            source_edge_ids: Vec::new(),
            start_mm: [0.95, 1.2],
            end_mm: [0.95, 1.6],
        };
        assert!(line_arc_has_extra_intersection(
            &crossing,
            &profile[2],
            false
        ));
    }

    fn split_quarter_torus_faces(torus_minor_radius_mm: f64) -> Vec<FaceInfo> {
        let torus = brep::TorusSupport {
            center_mm: [0.0, 0.0, 1.4],
            axis: [0.0, 0.0, 1.0],
            x_direction: [1.0, 0.0, 0.0],
            major_radius_mm: 0.9,
            minor_radius_mm: torus_minor_radius_mm,
        };
        let cylinder = brep::CylinderSupport {
            axis_origin_mm: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 1.0],
            x_direction: [1.0, 0.0, 0.0],
            radius_mm: 1.0,
        };
        let bottom_plane = brep::PlaneSupport {
            origin_mm: [0.0, 0.0, 1.0],
            normal: [0.0, 0.0, 1.0],
            max_residual_mm: 0.0,
        };
        let top_plane = brep::PlaneSupport {
            origin_mm: [0.0, 0.0, 1.5],
            normal: [0.0, 0.0, 1.0],
            max_residual_mm: 0.0,
        };

        let top_a = test_circle_edge(
            110,
            [0.9, 0.0, 1.5],
            [-0.9, 0.0, 1.5],
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 1.0],
            0.9,
        );
        let top_b = test_circle_edge(
            111,
            [-0.9, 0.0, 1.5],
            [0.9, 0.0, 1.5],
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 1.0],
            0.9,
        );
        let side_a = test_circle_edge(
            112,
            [1.0, 0.0, 1.4],
            [-1.0, 0.0, 1.4],
            [0.0, 0.0, 1.4],
            [0.0, 0.0, 1.0],
            1.0,
        );
        let side_b = test_circle_edge(
            113,
            [-1.0, 0.0, 1.4],
            [1.0, 0.0, 1.4],
            [0.0, 0.0, 1.4],
            [0.0, 0.0, 1.0],
            1.0,
        );
        let bottom_a = test_circle_edge(
            114,
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            1.0,
        );
        let bottom_b = test_circle_edge(
            115,
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            1.0,
        );
        let torus_seam_positive = test_circle_edge(
            116,
            [0.9, 0.0, 1.5],
            [1.0, 0.0, 1.4],
            [0.9, 0.0, 1.4],
            [0.0, 1.0, 0.0],
            0.1,
        );
        let torus_seam_negative = test_circle_edge(
            117,
            [-0.9, 0.0, 1.5],
            [-1.0, 0.0, 1.4],
            [-0.9, 0.0, 1.4],
            [0.0, -1.0, 0.0],
            0.1,
        );
        let cylinder_seam_positive = test_line_edge(118, [1.0, 0.0, 1.0], [1.0, 0.0, 1.4]);
        let cylinder_seam_negative = test_line_edge(119, [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.4]);

        vec![
            test_face(
                SurfaceSupport::Torus(torus),
                vec![
                    top_a.clone(),
                    torus_seam_positive.clone(),
                    side_a.clone(),
                    torus_seam_negative.clone(),
                ],
            ),
            test_face(
                SurfaceSupport::Torus(torus),
                vec![
                    top_b.clone(),
                    torus_seam_positive,
                    side_b.clone(),
                    torus_seam_negative,
                ],
            ),
            test_face(
                SurfaceSupport::Cylinder(cylinder),
                vec![
                    bottom_a.clone(),
                    cylinder_seam_positive.clone(),
                    side_a,
                    cylinder_seam_negative.clone(),
                ],
            ),
            test_face(
                SurfaceSupport::Cylinder(cylinder),
                vec![
                    bottom_b.clone(),
                    cylinder_seam_positive,
                    side_b,
                    cylinder_seam_negative,
                ],
            ),
            test_face(
                SurfaceSupport::Plane(bottom_plane),
                vec![bottom_a, bottom_b],
            ),
            test_face(SurfaceSupport::Plane(top_plane), vec![top_a, top_b]),
        ]
    }

    #[test]
    fn recovers_split_quarter_torus_fillet_topology() {
        let faces = split_quarter_torus_faces(0.1);
        let edge_faces = edge_face_map(&faces);
        let entities = Vec::new();
        let index = HashMap::new();
        let context = TopologyContext {
            faces: &faces,
            edge_faces: &edge_faces,
            entities: &entities,
            index: &index,
        };
        let recovered =
            detect_mixed_torus_revolution(99, &[1, 2, 3, 4, 5, 6], &faces, &context).unwrap();
        assert_eq!(recovered.profile_curves.len(), 5);
        assert!(recovered.profile_curves.iter().any(|curve| matches!(
            curve,
            RecoveredProfileCurve::CircleArc {
                center_mm: [0.9, 1.4],
                radius_mm: 0.1,
                start_angle_rad,
                end_angle_rad,
                ..
            } if start_angle_rad.abs() <= 1.0e-12
                && (*end_angle_rad - std::f64::consts::FRAC_PI_2).abs() <= 1.0e-12
        )));

        let mut spindle = split_quarter_torus_faces(0.1);
        for face in spindle.iter_mut().take(2) {
            let SurfaceSupport::Torus(mut torus) = face.surface else {
                panic!("expected torus test face");
            };
            torus.major_radius_mm = 0.05;
            face.surface = SurfaceSupport::Torus(torus);
        }
        let edge_faces = edge_face_map(&spindle);
        let context = TopologyContext {
            faces: &spindle,
            edge_faces: &edge_faces,
            entities: &entities,
            index: &index,
        };
        assert!(
            detect_mixed_torus_revolution(99, &[1, 2, 3, 4, 5, 6], &spindle, &context).is_none()
        );

        let tampered = split_quarter_torus_faces(0.11);
        let edge_faces = edge_face_map(&tampered);
        let context = TopologyContext {
            faces: &tampered,
            edge_faces: &edge_faces,
            entities: &entities,
            index: &index,
        };
        assert!(
            detect_mixed_torus_revolution(99, &[1, 2, 3, 4, 5, 6], &tampered, &context).is_none()
        );
    }

    #[test]
    fn solid_surface_signature_reports_native_spherical_cap() {
        let bytes = include_bytes!("../validation/fixtures/native_spherical_cap.step");
        let signatures = crate::detect_solid_surface_signatures_bytes(bytes).unwrap();
        assert_eq!(signatures.len(), 1);
        assert_eq!(signatures[0].face_count, 2);
        assert_eq!(signatures[0].support_counts.get("sphere"), Some(&1));
        assert_eq!(signatures[0].support_counts.get("plane"), Some(&1));
        assert!(signatures[0].closed_two_manifold);
    }

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
            recovered[0].polygon_points().unwrap(),
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
            recovered[0].polygon_points().unwrap(),
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
            recovered[0].polygon_points().unwrap(),
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
    fn recovers_native_spherical_cap_fixture() {
        let bytes = include_bytes!("../validation/fixtures/native_spherical_cap.step");
        let recovered = crate::detect_solid_revolutions_bytes(bytes).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].face_ids.len(), 2);
        assert_eq!(recovered[0].profile_curves.len(), 3);

        let RecoveredProfileCurve::Line {
            start_mm: plane_axis,
            end_mm: cap_edge,
            ..
        } = &recovered[0].profile_curves[0]
        else {
            panic!("expected planar radial segment");
        };
        assert!(plane_axis[0].abs() < 1.0e-12);
        assert!((plane_axis[1] - 0.073).abs() < 1.0e-12);
        assert!((cap_edge[0] - 0.107568582774).abs() < 1.0e-12);
        assert!((cap_edge[1] - 0.073).abs() < 1.0e-12);

        let RecoveredProfileCurve::CircleArc {
            center_mm,
            radius_mm,
            start_angle_rad,
            end_angle_rad,
            ..
        } = &recovered[0].profile_curves[1]
        else {
            panic!("expected spherical meridian arc");
        };
        assert!(center_mm[0].abs() < 1.0e-12);
        assert!(center_mm[1].abs() < 1.0e-12);
        assert!((*radius_mm - 0.13).abs() < 1.0e-12);
        assert!((*start_angle_rad - 0.596243908486).abs() < 1.0e-12);
        assert!((*end_angle_rad + std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);

        let RecoveredProfileCurve::Line {
            start_mm: pole,
            end_mm: close,
            ..
        } = &recovered[0].profile_curves[2]
        else {
            panic!("expected axis closure");
        };
        assert!(pole[0].abs() < 1.0e-12);
        assert!((pole[1] + 0.13).abs() < 1.0e-12);
        assert_eq!(*close, *plane_axis);
        assert!(recovered[0].max_residual_mm < 1.0e-9);

        let malformed = String::from_utf8_lossy(bytes).replacen(
            "CIRCLE('',#26,0.107568582774)",
            "CIRCLE('',#26,0.117568582774)",
            1,
        );
        assert!(
            crate::detect_solid_revolutions_bytes(malformed.as_bytes())
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
    fn recovers_positive_spherical_cap_fixture() {
        let bytes = include_bytes!("../validation/fixtures/native_spherical_cap_positive.step");
        let recovered = crate::detect_solid_revolutions_bytes(bytes).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].profile_curves.len(), 3);
        let RecoveredProfileCurve::CircleArc { end_angle_rad, .. } =
            &recovered[0].profile_curves[1]
        else {
            panic!("expected spherical meridian arc");
        };
        assert!((*end_angle_rad - std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);
        let RecoveredProfileCurve::Line { start_mm: pole, .. } = &recovered[0].profile_curves[2]
        else {
            panic!("expected axis closure");
        };
        assert!((pole[1] - 0.13).abs() < 1.0e-12);
        assert!(recovered[0].max_residual_mm < 1.0e-10);

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
    fn recovers_native_ring_torus_fixture() {
        let bytes = include_bytes!("../validation/fixtures/native_torus.step");
        let recovered = crate::detect_solid_revolutions_bytes(bytes).unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].face_ids.len(), 1);
        assert_eq!(recovered[0].profile_curves.len(), 1);
        let RecoveredProfileCurve::CircleArc {
            center_mm,
            radius_mm,
            start_angle_rad,
            end_angle_rad,
            ..
        } = &recovered[0].profile_curves[0]
        else {
            panic!("expected full-circle torus meridian");
        };
        assert!((center_mm[0] - 1.2).abs() < 1.0e-12);
        assert!(center_mm[1].abs() < 1.0e-12);
        assert!((*radius_mm - 0.09).abs() < 1.0e-12);
        assert_eq!(*start_angle_rad, 0.0);
        assert_eq!(*end_angle_rad, std::f64::consts::TAU);

        let malformed = String::from_utf8_lossy(bytes).replacen(
            "CIRCLE('',#26,1.29)",
            "CIRCLE('',#26,1.31)",
            1,
        );
        assert!(
            crate::detect_solid_revolutions_bytes(malformed.as_bytes())
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
