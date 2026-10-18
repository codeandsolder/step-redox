use crate::brep::{self, CurveSupport, SurfaceSupport};
use crate::math3::{
    add, canonical_direction, cross, dot, mul, norm, normalize, point_to_unit_line_distance, sub,
};
use crate::step_graph::{build_index, entity_id, simple_record};
use ruststep::ast::EntityInstance;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

const GEOM_TOL_MM: f64 = 1.0e-7;
const DIR_TOL: f64 = 1.0e-8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeTreatmentKind {
    CylindricalFillet,
    PlanarChamfer,
    ToroidalFillet,
    ConicalChamfer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeTreatmentEvidenceStrength {
    AnalyticTangency,
    ContinuationCandidate,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EdgeTreatmentVirtualEdge {
    Segment {
        start_mm: [f64; 3],
        end_mm: [f64; 3],
    },
    Circle {
        center_mm: [f64; 3],
        normal: [f64; 3],
        radius_mm: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EdgeTreatmentCarrier {
    pub face_id: u64,
    pub support: String,
    pub shared_edge_ids: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EdgeTreatmentEvidence {
    pub solid_id: u64,
    pub treatment_face_id: u64,
    pub kind: EdgeTreatmentKind,
    pub strength: EdgeTreatmentEvidenceStrength,
    pub carrier_pair_unique: bool,
    pub treatment_support: String,
    pub carriers: [EdgeTreatmentCarrier; 2],
    pub inferred_radius_mm: Option<f64>,
    pub inferred_angle_rad: Option<f64>,
    pub carrier_angle_rad: f64,
    pub virtual_edge: EdgeTreatmentVirtualEdge,
    pub max_boundary_residual_mm: f64,
}

#[derive(Debug, Clone)]
struct FaceInfo {
    face_id: u64,
    surface: SurfaceSupport,
    loops: Vec<brep::FaceLoop>,
}

#[derive(Debug, Clone, Copy)]
struct MeridianLine {
    a: f64,
    b: f64,
    c: f64,
}

#[must_use]
pub fn detect_edge_treatment_evidence(entities: &[EntityInstance]) -> Vec<EdgeTreatmentEvidence> {
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
        out.extend(detect_solid_edge_treatments(
            solid_id, &face_ids, entities, &index,
        ));
    }
    out.sort_by(|a, b| {
        (
            a.solid_id,
            a.treatment_face_id,
            a.kind,
            a.carriers[0].face_id,
            a.carriers[1].face_id,
        )
            .cmp(&(
                b.solid_id,
                b.treatment_face_id,
                b.kind,
                b.carriers[0].face_id,
                b.carriers[1].face_id,
            ))
    });
    out.dedup_by(|a, b| {
        a.solid_id == b.solid_id
            && a.treatment_face_id == b.treatment_face_id
            && a.kind == b.kind
            && a.carriers == b.carriers
    });
    mark_unique_carrier_pairs(&mut out);
    out
}

fn detect_solid_edge_treatments(
    solid_id: u64,
    face_ids: &[u64],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Vec<EdgeTreatmentEvidence> {
    let Some(faces) = face_ids
        .iter()
        .copied()
        .map(|face_id| {
            let surface_id = brep::face_surface(face_id, entities, index)?;
            Some(FaceInfo {
                face_id,
                surface: brep::surface_support(surface_id, entities, index),
                loops: brep::face_loops(face_id, entities, index)?,
            })
        })
        .collect::<Option<Vec<_>>>()
    else {
        return Vec::new();
    };

    let mut edge_faces = HashMap::<u64, Vec<usize>>::new();
    let mut edge_geometry = HashMap::<u64, brep::OrientedEdgeUse>::new();
    for (face_index, face) in faces.iter().enumerate() {
        for edge in face.loops.iter().flat_map(|loop_| &loop_.edges) {
            edge_faces.entry(edge.edge_id).or_default().push(face_index);
            edge_geometry
                .entry(edge.edge_id)
                .or_insert_with(|| edge.clone());
        }
    }
    if edge_faces.values().any(|attached| attached.len() != 2) {
        return Vec::new();
    }

    let mut out = Vec::new();
    for treatment_index in 0..faces.len() {
        let neighbors = neighbor_shared_edges(treatment_index, &faces, &edge_faces);
        let entries = neighbors.into_iter().collect::<Vec<_>>();
        for first_index in 0..entries.len() {
            for second_index in first_index + 1..entries.len() {
                let (first_face_index, first_edges) = &entries[first_index];
                let (second_face_index, second_edges) = &entries[second_index];
                let first = &faces[*first_face_index];
                let second = &faces[*second_face_index];
                let treatment = &faces[treatment_index];
                let evidence = match &treatment.surface {
                    SurfaceSupport::Cylinder(cylinder) => prove_cylindrical_fillet(
                        solid_id,
                        treatment,
                        *cylinder,
                        first,
                        first_edges,
                        second,
                        second_edges,
                        &edge_geometry,
                    ),
                    SurfaceSupport::Plane(plane) => prove_planar_chamfer(
                        solid_id,
                        treatment,
                        *plane,
                        first,
                        first_edges,
                        second,
                        second_edges,
                        &edge_geometry,
                    ),
                    SurfaceSupport::Torus(torus) => prove_toroidal_fillet(
                        solid_id,
                        treatment,
                        *torus,
                        first,
                        first_edges,
                        second,
                        second_edges,
                        &edge_geometry,
                    ),
                    SurfaceSupport::Cone(cone) => prove_conical_chamfer(
                        solid_id,
                        treatment,
                        *cone,
                        first,
                        first_edges,
                        second,
                        second_edges,
                        &edge_geometry,
                    ),
                    SurfaceSupport::Sphere(_)
                    | SurfaceSupport::Revolution(_)
                    | SurfaceSupport::SplineExtrusion(_)
                    | SurfaceSupport::Other { .. } => None,
                };
                if let Some(evidence) = evidence {
                    out.push(evidence);
                }
            }
        }
    }
    out
}

fn mark_unique_carrier_pairs(out: &mut [EdgeTreatmentEvidence]) {
    let mut counts = HashMap::<(u64, u64, EdgeTreatmentKind), usize>::new();
    for evidence in out.iter() {
        *counts
            .entry((evidence.solid_id, evidence.treatment_face_id, evidence.kind))
            .or_default() += 1;
    }
    for evidence in out {
        evidence.carrier_pair_unique =
            counts[&(evidence.solid_id, evidence.treatment_face_id, evidence.kind)] == 1;
    }
}

fn neighbor_shared_edges(
    treatment_index: usize,
    faces: &[FaceInfo],
    edge_faces: &HashMap<u64, Vec<usize>>,
) -> BTreeMap<usize, Vec<u64>> {
    let mut neighbors = BTreeMap::<usize, Vec<u64>>::new();
    for edge in faces[treatment_index]
        .loops
        .iter()
        .flat_map(|loop_| &loop_.edges)
    {
        let Some(attached) = edge_faces.get(&edge.edge_id) else {
            continue;
        };
        for &neighbor in attached {
            if neighbor != treatment_index {
                neighbors.entry(neighbor).or_default().push(edge.edge_id);
            }
        }
    }
    for edges in neighbors.values_mut() {
        edges.sort_unstable();
        edges.dedup();
    }
    neighbors
}

#[allow(
    clippy::too_many_arguments,
    reason = "proof helpers keep carrier evidence explicit"
)]
fn prove_cylindrical_fillet(
    solid_id: u64,
    treatment: &FaceInfo,
    cylinder: brep::CylinderSupport,
    first: &FaceInfo,
    first_edges: &[u64],
    second: &FaceInfo,
    second_edges: &[u64],
    edge_geometry: &HashMap<u64, brep::OrientedEdgeUse>,
) -> Option<EdgeTreatmentEvidence> {
    let (SurfaceSupport::Plane(first_plane), SurfaceSupport::Plane(second_plane)) =
        (&first.surface, &second.surface)
    else {
        return None;
    };
    let axis = normalize(cylinder.axis, DIR_TOL)?;
    let first_normal = normalize(first_plane.normal, DIR_TOL)?;
    let second_normal = normalize(second_plane.normal, DIR_TOL)?;
    let virtual_direction = normalize(cross(first_normal, second_normal), DIR_TOL)?;
    if !parallel(axis, virtual_direction)
        || dot(first_normal, axis).abs() > DIR_TOL
        || dot(second_normal, axis).abs() > DIR_TOL
    {
        return None;
    }

    let first_tangent_residual = (dot(
        first_normal,
        sub(cylinder.axis_origin_mm, first_plane.origin_mm),
    )
    .abs()
        - cylinder.radius_mm)
        .abs();
    let second_tangent_residual = (dot(
        second_normal,
        sub(cylinder.axis_origin_mm, second_plane.origin_mm),
    )
    .abs()
        - cylinder.radius_mm)
        .abs();
    let mut max_boundary_residual_mm = first_tangent_residual.max(second_tangent_residual);
    if max_boundary_residual_mm > GEOM_TOL_MM {
        return None;
    }

    max_boundary_residual_mm = max_boundary_residual_mm.max(prove_line_boundary(
        first_edges,
        edge_geometry,
        axis,
        &treatment.surface,
        &first.surface,
    )?);
    max_boundary_residual_mm = max_boundary_residual_mm.max(prove_line_boundary(
        second_edges,
        edge_geometry,
        axis,
        &treatment.surface,
        &second.surface,
    )?);
    if max_boundary_residual_mm > GEOM_TOL_MM {
        return None;
    }
    let virtual_edge = plane_intersection_segment(
        *first_plane,
        *second_plane,
        axis,
        cylinder.axis_origin_mm,
        first_edges,
        second_edges,
        edge_geometry,
    )?;

    Some(evidence(
        solid_id,
        treatment,
        EdgeTreatmentKind::CylindricalFillet,
        first,
        first_edges,
        second,
        second_edges,
        Some(cylinder.radius_mm),
        None,
        plane_angle(first_normal, second_normal),
        virtual_edge,
        max_boundary_residual_mm,
    ))
}

#[allow(
    clippy::too_many_arguments,
    reason = "proof helpers keep carrier evidence explicit"
)]
fn prove_planar_chamfer(
    solid_id: u64,
    treatment: &FaceInfo,
    treatment_plane: brep::PlaneSupport,
    first: &FaceInfo,
    first_edges: &[u64],
    second: &FaceInfo,
    second_edges: &[u64],
    edge_geometry: &HashMap<u64, brep::OrientedEdgeUse>,
) -> Option<EdgeTreatmentEvidence> {
    let (SurfaceSupport::Plane(first_plane), SurfaceSupport::Plane(second_plane)) =
        (&first.surface, &second.surface)
    else {
        return None;
    };
    let treatment_normal = normalize(treatment_plane.normal, DIR_TOL)?;
    let first_normal = normalize(first_plane.normal, DIR_TOL)?;
    let second_normal = normalize(second_plane.normal, DIR_TOL)?;
    let virtual_direction = normalize(cross(first_normal, second_normal), DIR_TOL)?;
    if dot(treatment_normal, virtual_direction).abs() > DIR_TOL {
        return None;
    }

    let mut max_boundary_residual_mm = prove_line_boundary(
        first_edges,
        edge_geometry,
        virtual_direction,
        &treatment.surface,
        &first.surface,
    )?;
    max_boundary_residual_mm = max_boundary_residual_mm.max(prove_line_boundary(
        second_edges,
        edge_geometry,
        virtual_direction,
        &treatment.surface,
        &second.surface,
    )?);
    if max_boundary_residual_mm > GEOM_TOL_MM {
        return None;
    }
    let reference_origin = first_edges
        .first()
        .and_then(|edge_id| edge_geometry.get(edge_id))?
        .start_mm;
    let virtual_edge = plane_intersection_segment(
        *first_plane,
        *second_plane,
        virtual_direction,
        reference_origin,
        first_edges,
        second_edges,
        edge_geometry,
    )?;

    Some(evidence(
        solid_id,
        treatment,
        EdgeTreatmentKind::PlanarChamfer,
        first,
        first_edges,
        second,
        second_edges,
        None,
        Some(plane_angle(treatment_normal, first_normal)),
        plane_angle(first_normal, second_normal),
        virtual_edge,
        max_boundary_residual_mm,
    ))
}

#[allow(
    clippy::too_many_arguments,
    reason = "proof helpers keep carrier evidence explicit"
)]
fn prove_toroidal_fillet(
    solid_id: u64,
    treatment: &FaceInfo,
    torus: brep::TorusSupport,
    first: &FaceInfo,
    first_edges: &[u64],
    second: &FaceInfo,
    second_edges: &[u64],
    edge_geometry: &HashMap<u64, brep::OrientedEdgeUse>,
) -> Option<EdgeTreatmentEvidence> {
    let axis = canonical_direction(normalize(torus.axis, DIR_TOL)?);
    let first_line = axisymmetric_meridian_line(&first.surface, torus.center_mm, axis)?;
    let second_line = axisymmetric_meridian_line(&second.surface, torus.center_mm, axis)?;
    let (virtual_t, virtual_radius) = meridian_line_intersection(first_line, second_line)?;
    if virtual_radius <= GEOM_TOL_MM {
        return None;
    }

    let first_tangent_residual =
        meridian_point_line_distance([0.0, torus.major_radius_mm], first_line);
    let second_tangent_residual =
        meridian_point_line_distance([0.0, torus.major_radius_mm], second_line);
    let mut max_boundary_residual_mm = (first_tangent_residual - torus.minor_radius_mm)
        .abs()
        .max((second_tangent_residual - torus.minor_radius_mm).abs());
    if max_boundary_residual_mm > GEOM_TOL_MM {
        return None;
    }

    max_boundary_residual_mm = max_boundary_residual_mm.max(prove_circle_boundary(
        first_edges,
        edge_geometry,
        axis,
        &treatment.surface,
        &first.surface,
    )?);
    max_boundary_residual_mm = max_boundary_residual_mm.max(prove_circle_boundary(
        second_edges,
        edge_geometry,
        axis,
        &treatment.surface,
        &second.surface,
    )?);
    if max_boundary_residual_mm > GEOM_TOL_MM {
        return None;
    }
    let virtual_edge = EdgeTreatmentVirtualEdge::Circle {
        center_mm: add(torus.center_mm, mul(axis, virtual_t)),
        normal: axis,
        radius_mm: virtual_radius,
    };

    Some(evidence(
        solid_id,
        treatment,
        EdgeTreatmentKind::ToroidalFillet,
        first,
        first_edges,
        second,
        second_edges,
        Some(torus.minor_radius_mm),
        None,
        meridian_line_angle(first_line, second_line),
        virtual_edge,
        max_boundary_residual_mm,
    ))
}

#[allow(
    clippy::too_many_arguments,
    reason = "proof helpers keep carrier evidence explicit"
)]
fn prove_conical_chamfer(
    solid_id: u64,
    treatment: &FaceInfo,
    cone: brep::ConeSupport,
    first: &FaceInfo,
    first_edges: &[u64],
    second: &FaceInfo,
    second_edges: &[u64],
    edge_geometry: &HashMap<u64, brep::OrientedEdgeUse>,
) -> Option<EdgeTreatmentEvidence> {
    let axis = canonical_direction(normalize(cone.axis, DIR_TOL)?);
    let origin = cone.reference_origin_mm;
    let treatment_line = axisymmetric_meridian_line(&treatment.surface, origin, axis)?;
    let first_line = axisymmetric_meridian_line(&first.surface, origin, axis)?;
    let second_line = axisymmetric_meridian_line(&second.surface, origin, axis)?;
    let (virtual_t, virtual_radius) = meridian_line_intersection(first_line, second_line)?;
    if virtual_radius <= GEOM_TOL_MM
        || meridian_lines_parallel(treatment_line, first_line)
        || meridian_lines_parallel(treatment_line, second_line)
    {
        return None;
    }

    let mut max_boundary_residual_mm = prove_circle_boundary(
        first_edges,
        edge_geometry,
        axis,
        &treatment.surface,
        &first.surface,
    )?;
    max_boundary_residual_mm = max_boundary_residual_mm.max(prove_circle_boundary(
        second_edges,
        edge_geometry,
        axis,
        &treatment.surface,
        &second.surface,
    )?);
    if max_boundary_residual_mm > GEOM_TOL_MM {
        return None;
    }
    let virtual_edge = EdgeTreatmentVirtualEdge::Circle {
        center_mm: add(origin, mul(axis, virtual_t)),
        normal: axis,
        radius_mm: virtual_radius,
    };

    Some(evidence(
        solid_id,
        treatment,
        EdgeTreatmentKind::ConicalChamfer,
        first,
        first_edges,
        second,
        second_edges,
        None,
        Some(cone.semi_angle_rad),
        meridian_line_angle(first_line, second_line),
        virtual_edge,
        max_boundary_residual_mm,
    ))
}

#[allow(
    clippy::too_many_arguments,
    reason = "proof helpers keep carrier evidence explicit"
)]
fn evidence(
    solid_id: u64,
    treatment: &FaceInfo,
    kind: EdgeTreatmentKind,
    first: &FaceInfo,
    first_edges: &[u64],
    second: &FaceInfo,
    second_edges: &[u64],
    inferred_radius_mm: Option<f64>,
    inferred_angle_rad: Option<f64>,
    carrier_angle_rad: f64,
    virtual_edge: EdgeTreatmentVirtualEdge,
    max_boundary_residual_mm: f64,
) -> EdgeTreatmentEvidence {
    let mut carriers = [
        EdgeTreatmentCarrier {
            face_id: first.face_id,
            support: support_name(&first.surface).to_owned(),
            shared_edge_ids: first_edges.to_vec(),
        },
        EdgeTreatmentCarrier {
            face_id: second.face_id,
            support: support_name(&second.surface).to_owned(),
            shared_edge_ids: second_edges.to_vec(),
        },
    ];
    carriers.sort_by_key(|carrier| carrier.face_id);
    EdgeTreatmentEvidence {
        solid_id,
        treatment_face_id: treatment.face_id,
        kind,
        strength: match kind {
            EdgeTreatmentKind::CylindricalFillet | EdgeTreatmentKind::ToroidalFillet => {
                EdgeTreatmentEvidenceStrength::AnalyticTangency
            }
            EdgeTreatmentKind::PlanarChamfer | EdgeTreatmentKind::ConicalChamfer => {
                EdgeTreatmentEvidenceStrength::ContinuationCandidate
            }
        },
        carrier_pair_unique: false,
        treatment_support: support_name(&treatment.surface).to_owned(),
        carriers,
        inferred_radius_mm,
        inferred_angle_rad,
        carrier_angle_rad,
        virtual_edge,
        max_boundary_residual_mm,
    }
}

fn plane_intersection_segment(
    first: brep::PlaneSupport,
    second: brep::PlaneSupport,
    direction: [f64; 3],
    reference_origin: [f64; 3],
    first_edges: &[u64],
    second_edges: &[u64],
    edge_geometry: &HashMap<u64, brep::OrientedEdgeUse>,
) -> Option<EdgeTreatmentVirtualEdge> {
    let first_normal = normalize(first.normal, DIR_TOL)?;
    let second_normal = normalize(second.normal, DIR_TOL)?;
    let direction = canonical_direction(normalize(direction, DIR_TOL)?);
    let point = three_plane_intersection(
        first_normal,
        dot(first_normal, first.origin_mm),
        second_normal,
        dot(second_normal, second.origin_mm),
        direction,
        dot(direction, reference_origin),
    )?;
    let first_range =
        boundary_axial_range(first_edges, edge_geometry, reference_origin, direction)?;
    let second_range =
        boundary_axial_range(second_edges, edge_geometry, reference_origin, direction)?;
    if (first_range.0 - second_range.0).abs() > GEOM_TOL_MM
        || (first_range.1 - second_range.1).abs() > GEOM_TOL_MM
    {
        return None;
    }
    let start_t = first_range.0.midpoint(second_range.0);
    let end_t = first_range.1.midpoint(second_range.1);
    if end_t - start_t <= GEOM_TOL_MM {
        return None;
    }
    Some(EdgeTreatmentVirtualEdge::Segment {
        start_mm: add(point, mul(direction, start_t)),
        end_mm: add(point, mul(direction, end_t)),
    })
}

fn boundary_axial_range(
    edge_ids: &[u64],
    edge_geometry: &HashMap<u64, brep::OrientedEdgeUse>,
    origin: [f64; 3],
    direction: [f64; 3],
) -> Option<(f64, f64)> {
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for edge_id in edge_ids {
        let edge = edge_geometry.get(edge_id)?;
        for point in [edge.start_mm, edge.end_mm] {
            let parameter = dot(sub(point, origin), direction);
            minimum = minimum.min(parameter);
            maximum = maximum.max(parameter);
        }
    }
    (minimum.is_finite() && maximum.is_finite()).then_some((minimum, maximum))
}

fn three_plane_intersection(
    first_normal: [f64; 3],
    first_offset: f64,
    second_normal: [f64; 3],
    second_offset: f64,
    third_normal: [f64; 3],
    third_offset: f64,
) -> Option<[f64; 3]> {
    let second_cross_third = cross(second_normal, third_normal);
    let third_cross_first = cross(third_normal, first_normal);
    let first_cross_second = cross(first_normal, second_normal);
    let determinant = dot(first_normal, second_cross_third);
    if !determinant.is_finite() || determinant.abs() <= DIR_TOL {
        return None;
    }
    let numerator = add(
        add(
            mul(second_cross_third, first_offset),
            mul(third_cross_first, second_offset),
        ),
        mul(first_cross_second, third_offset),
    );
    Some(mul(numerator, determinant.recip()))
}

fn prove_line_boundary(
    edge_ids: &[u64],
    edge_geometry: &HashMap<u64, brep::OrientedEdgeUse>,
    expected_direction: [f64; 3],
    first_surface: &SurfaceSupport,
    second_surface: &SurfaceSupport,
) -> Option<f64> {
    if edge_ids.is_empty() {
        return None;
    }
    let mut max_residual = 0.0_f64;
    for edge_id in edge_ids {
        let edge = edge_geometry.get(edge_id)?;
        let CurveSupport::Line(line) = edge.support else {
            return None;
        };
        let direction = normalize(line.direction, DIR_TOL)?;
        if !parallel(direction, expected_direction) {
            return None;
        }
        for point in [edge.start_mm, edge.end_mm] {
            max_residual = max_residual
                .max(surface_point_residual(point, first_surface)?)
                .max(surface_point_residual(point, second_surface)?);
        }
    }
    Some(max_residual)
}

fn prove_circle_boundary(
    edge_ids: &[u64],
    edge_geometry: &HashMap<u64, brep::OrientedEdgeUse>,
    axis: [f64; 3],
    first_surface: &SurfaceSupport,
    second_surface: &SurfaceSupport,
) -> Option<f64> {
    if edge_ids.is_empty() {
        return None;
    }
    let mut max_residual = 0.0_f64;
    for edge_id in edge_ids {
        let edge = edge_geometry.get(edge_id)?;
        let CurveSupport::Circle(circle) = edge.support else {
            return None;
        };
        if !parallel(normalize(circle.normal, DIR_TOL)?, axis) {
            return None;
        }
        for point in [edge.start_mm, edge.end_mm] {
            max_residual = max_residual
                .max(surface_point_residual(point, first_surface)?)
                .max(surface_point_residual(point, second_surface)?);
        }
    }
    Some(max_residual)
}

fn surface_point_residual(point: [f64; 3], surface: &SurfaceSupport) -> Option<f64> {
    match surface {
        SurfaceSupport::Plane(plane) => {
            let normal = normalize(plane.normal, DIR_TOL)?;
            Some(dot(normal, sub(point, plane.origin_mm)).abs())
        }
        SurfaceSupport::Cylinder(cylinder) => {
            let axis = normalize(cylinder.axis, DIR_TOL)?;
            Some(
                (point_to_unit_line_distance(point, cylinder.axis_origin_mm, axis)
                    - cylinder.radius_mm)
                    .abs(),
            )
        }
        SurfaceSupport::Cone(cone) => {
            let axis = normalize(cone.axis, DIR_TOL)?;
            let axial = dot(sub(point, cone.reference_origin_mm), axis);
            let radial = point_to_unit_line_distance(point, cone.reference_origin_mm, axis);
            let expected = cone
                .semi_angle_rad
                .tan()
                .mul_add(axial, cone.reference_radius_mm);
            expected.is_finite().then_some((radial - expected).abs())
        }
        SurfaceSupport::Sphere(sphere) => {
            Some((norm(sub(point, sphere.center_mm)) - sphere.radius_mm).abs())
        }
        SurfaceSupport::Torus(torus) => {
            let axis = normalize(torus.axis, DIR_TOL)?;
            let relative = sub(point, torus.center_mm);
            let axial = dot(relative, axis);
            let radial_vector = sub(relative, axis.map(|component| component * axial));
            let radial = norm(radial_vector);
            Some(((radial - torus.major_radius_mm).hypot(axial) - torus.minor_radius_mm).abs())
        }
        SurfaceSupport::Revolution(_)
        | SurfaceSupport::SplineExtrusion(_)
        | SurfaceSupport::Other { .. } => None,
    }
}

fn axisymmetric_meridian_line(
    surface: &SurfaceSupport,
    origin: [f64; 3],
    axis: [f64; 3],
) -> Option<MeridianLine> {
    match surface {
        SurfaceSupport::Plane(plane) => {
            let normal = normalize(plane.normal, DIR_TOL)?;
            if !parallel(normal, axis) {
                return None;
            }
            let t = dot(sub(plane.origin_mm, origin), axis);
            Some(MeridianLine {
                a: 1.0,
                b: 0.0,
                c: -t,
            })
        }
        SurfaceSupport::Cylinder(cylinder) => {
            let source_axis = normalize(cylinder.axis, DIR_TOL)?;
            if !parallel(source_axis, axis)
                || point_to_unit_line_distance(cylinder.axis_origin_mm, origin, axis) > GEOM_TOL_MM
            {
                return None;
            }
            Some(MeridianLine {
                a: 0.0,
                b: 1.0,
                c: -cylinder.radius_mm,
            })
        }
        SurfaceSupport::Cone(cone) => {
            let source_axis = normalize(cone.axis, DIR_TOL)?;
            let alignment = dot(source_axis, axis);
            if 1.0 - alignment.abs() > DIR_TOL
                || point_to_unit_line_distance(cone.reference_origin_mm, origin, axis) > GEOM_TOL_MM
            {
                return None;
            }
            let reference_t = dot(sub(cone.reference_origin_mm, origin), axis);
            let slope = alignment.signum() * cone.semi_angle_rad.tan();
            let intercept = slope.mul_add(-reference_t, cone.reference_radius_mm);
            if !slope.is_finite() || !intercept.is_finite() {
                return None;
            }
            Some(MeridianLine {
                a: -slope,
                b: 1.0,
                c: -intercept,
            })
        }
        SurfaceSupport::Sphere(_)
        | SurfaceSupport::Torus(_)
        | SurfaceSupport::Revolution(_)
        | SurfaceSupport::SplineExtrusion(_)
        | SurfaceSupport::Other { .. } => None,
    }
}

fn meridian_line_intersection(first: MeridianLine, second: MeridianLine) -> Option<(f64, f64)> {
    let determinant = first.a.mul_add(second.b, -second.a * first.b);
    if !determinant.is_finite() || determinant.abs() <= DIR_TOL {
        return None;
    }
    let t = (first.b.mul_add(second.c, -second.b * first.c)) / determinant;
    let radius = (first.c.mul_add(second.a, -second.c * first.a)) / determinant;
    (t.is_finite() && radius.is_finite()).then_some((t, radius))
}

fn meridian_point_line_distance(point: [f64; 2], line: MeridianLine) -> f64 {
    line.a
        .mul_add(point[0], line.b.mul_add(point[1], line.c))
        .abs()
        / line.a.hypot(line.b)
}

fn meridian_lines_parallel(first: MeridianLine, second: MeridianLine) -> bool {
    first.a.mul_add(second.b, -second.a * first.b).abs() <= DIR_TOL
}

fn meridian_line_angle(first: MeridianLine, second: MeridianLine) -> f64 {
    let first_norm = first.a.hypot(first.b);
    let second_norm = second.a.hypot(second.b);
    let cosine = (first.a.mul_add(second.a, first.b * second.b) / (first_norm * second_norm))
        .abs()
        .clamp(-1.0, 1.0);
    cosine.acos()
}

fn plane_angle(first: [f64; 3], second: [f64; 3]) -> f64 {
    dot(first, second).abs().clamp(-1.0, 1.0).acos()
}

fn parallel(first: [f64; 3], second: [f64; 3]) -> bool {
    let Some(first) = normalize(first, DIR_TOL) else {
        return false;
    };
    let Some(second) = normalize(second, DIR_TOL) else {
        return false;
    };
    1.0 - dot(first, second).abs() <= DIR_TOL
}

const fn support_name(surface: &SurfaceSupport) -> &'static str {
    match surface {
        SurfaceSupport::Plane(_) => "plane",
        SurfaceSupport::Cylinder(_) => "cylinder",
        SurfaceSupport::Cone(_) => "cone",
        SurfaceSupport::Sphere(_) => "sphere",
        SurfaceSupport::Torus(_) => "torus",
        SurfaceSupport::Revolution(_) => "revolution",
        SurfaceSupport::SplineExtrusion(_) => "spline_extrusion",
        SurfaceSupport::Other { .. } => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meridian_lines_intersect_deterministically() {
        let plane = MeridianLine {
            a: 1.0,
            b: 0.0,
            c: -2.0,
        };
        let cylinder = MeridianLine {
            a: 0.0,
            b: 1.0,
            c: -3.0,
        };
        assert_eq!(
            meridian_line_intersection(plane, cylinder),
            Some((2.0, 3.0))
        );
        assert!((meridian_line_angle(plane, cylinder) - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    }

    #[test]
    fn torus_meridian_tangency_has_exact_radius() {
        let plane = MeridianLine {
            a: 1.0,
            b: 0.0,
            c: -1.0,
        };
        let cylinder = MeridianLine {
            a: 0.0,
            b: 1.0,
            c: -4.0,
        };
        let center = [0.75, 3.75];
        assert!((meridian_point_line_distance(center, plane) - 0.25).abs() < 1e-15);
        assert!((meridian_point_line_distance(center, cylinder) - 0.25).abs() < 1e-15);
    }
}
