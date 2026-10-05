use crate::brep::{self, CurveSupport, SurfaceSupport};
use crate::math3::{add, cross, dot, mul, norm, normalize as normalize3, sub};
use crate::profile_curves::RecoveredProfileCurve;
use crate::step_entities::representation_items_and_context;
use crate::step_graph::{build_index, entity_id, entity_ref_value, simple_record};
use ruststep::ast::{EntityInstance, Parameter, Record};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};

mod profile_graph;
use profile_graph::{
    between, push_unique_segment, segment_radius_at_axial, shell_faces_connected, sub2,
    unique_neighbor_face,
};

mod meridian;
use meridian::{MeridianArc, MeridianCurve, MeridianLine, MeridianProfile};

const GEOM_TOL_MM: f64 = 1.0e-7;
/// Default source-evidence tolerance when STEP provides no trusted uncertainty context.
pub(crate) const REVOLUTION_SOURCE_SUPPORT_TOL_MM: f64 = GEOM_TOL_MM;
/// Hard ceiling for source-declared length uncertainty used by revolution recovery.
/// This is 10 nm / 100x the strict default: enough for known exporter roundoff,
/// still far below any dimension we want to approximate semantically.
pub(crate) const MAX_REVOLUTION_SOURCE_UNCERTAINTY_MM: f64 = 1.0e-5;
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
    /// Worst observed disagreement between source trim evidence and the recovered
    /// analytic/profile geometry.
    pub max_residual_mm: f64,
    /// Maximum source-evidence residual admitted for this recovery. Exact/legacy
    /// paths use GEOM_TOL_MM; a straight analytic path may inherit a larger,
    /// explicitly declared representation-context uncertainty, capped by
    /// MAX_REVOLUTION_SOURCE_UNCERTAINTY_MM.
    pub source_tolerance_mm: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecoveredRadialSlotRevolution {
    /// The axisymmetric host body reconstructed before the slot is applied.
    pub base: RecoveredSolidRevolution,
    /// Source planes that prove the slot: the two side walls followed by the
    /// axis-containing back plane.
    pub slot_face_ids: [u64; 3],
    /// Unit radial direction from the revolution axis toward the slot opening.
    pub slot_outward_direction: [f64; 3],
    /// Unit in-plane direction across the slot width.
    pub slot_side_direction: [f64; 3],
    /// Half the distance between the two symmetric slot side planes.
    pub slot_half_width_mm: f64,
    /// Axial coordinate where the slot terminates inside the turned body.
    pub slot_root_axial_mm: f64,
    /// Axial coordinate of the source end through which the slot is open.
    pub slot_open_end_axial_mm: f64,
    /// Worst residual in the slot-plane symmetry/axis proof.
    pub slot_max_residual_mm: f64,
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

#[derive(Clone, Copy)]
struct ProfileSegmentContext<'ctx, 'data> {
    topology: &'ctx TopologyContext<'data>,
    axis_origin_mm: [f64; 3],
    axis_direction: [f64; 3],
    source_tolerance_mm: f64,
    angular_trim_faces: Option<&'ctx HashSet<usize>>,
}

impl<'ctx, 'data> ProfileSegmentContext<'ctx, 'data> {
    const fn without_angular_trims(
        topology: &'ctx TopologyContext<'data>,
        axis_origin_mm: [f64; 3],
        axis_direction: [f64; 3],
        source_tolerance_mm: f64,
    ) -> Self {
        Self {
            topology,
            axis_origin_mm,
            axis_direction,
            source_tolerance_mm,
            angular_trim_faces: None,
        }
    }
}

fn source_tolerance_by_representation_item(
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> HashMap<u64, f64> {
    let mut context_cache = HashMap::<u64, Option<f64>>::new();
    let mut out = HashMap::<u64, f64>::new();

    for entity in entities {
        let Some(record) = simple_record(entity) else {
            continue;
        };
        if record.name != "ADVANCED_BREP_SHAPE_REPRESENTATION" {
            continue;
        }
        let Some((items, context_id)) = representation_items_and_context(entity) else {
            continue;
        };
        let tolerance = *context_cache.entry(context_id).or_insert_with(|| {
            representation_context_length_uncertainty_mm(context_id, entities, index)
        });
        let Some(tolerance) = tolerance else {
            continue;
        };

        for item in items {
            out.entry(item)
                .and_modify(|existing| *existing = existing.min(tolerance))
                .or_insert(tolerance);
        }
    }

    out
}

fn representation_context_length_uncertainty_mm(
    context_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<f64> {
    let entity = entities.get(*index.get(&context_id)?)?;
    let record = entity_record_named(entity, "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT")?;
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let [Parameter::List(uncertainties)] = params.as_slice() else {
        return None;
    };

    let [uncertainty] = uncertainties.as_slice() else {
        return None;
    };
    let uncertainty_id = entity_ref_value(uncertainty)?;
    let value = uncertainty_measure_mm(uncertainty_id, entities, index)?;
    if !value.is_finite() || value <= 0.0 || value > MAX_REVOLUTION_SOURCE_UNCERTAINTY_MM {
        return None;
    }
    Some(value.max(GEOM_TOL_MM))
}

fn uncertainty_measure_mm(
    uncertainty_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<f64> {
    let entity = entities.get(*index.get(&uncertainty_id)?)?;
    let record = entity_record_named(entity, "UNCERTAINTY_MEASURE_WITH_UNIT")?;
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let [
        Parameter::Typed { keyword, parameter },
        unit,
        Parameter::String(name),
        _,
    ] = params.as_slice()
    else {
        return None;
    };
    if keyword != "LENGTH_MEASURE" || name != "distance_accuracy_value" {
        return None;
    }
    let value = parameter_number(parameter)?;
    let unit_id = entity_ref_value(unit)?;
    let scale_mm = crate::units::length_unit_scale_mm(unit_id, entities, index)?;
    let result = value * scale_mm;
    result.is_finite().then_some(result)
}

fn entity_record_named<'a>(entity: &'a EntityInstance, name: &str) -> Option<&'a Record> {
    match entity {
        EntityInstance::Simple { record, .. } => (record.name == name).then_some(record),
        EntityInstance::Complex { subsuper, .. } => {
            subsuper.0.iter().find(|record| record.name == name)
        }
    }
}

fn parameter_number(parameter: &Parameter) -> Option<f64> {
    match parameter {
        Parameter::Real(value) => Some(*value),
        Parameter::Integer(value) => crate::numeric::exact_i64_to_f64(*value),
        Parameter::Typed { parameter, .. } => parameter_number(parameter),
        _ => None,
    }
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
    let source_tolerances = source_tolerance_by_representation_item(entities, &index);
    let mut out = Vec::new();
    for entity in entities {
        let Some(record) = simple_record(entity) else {
            continue;
        };
        if record.name != "MANIFOLD_SOLID_BREP" {
            continue;
        }
        let solid_id = entity_id(entity);
        let source_tolerance_mm = source_tolerances
            .get(&solid_id)
            .copied()
            .unwrap_or(REVOLUTION_SOURCE_SUPPORT_TOL_MM);
        if let Some(candidate) = detect_one_solid(solid_id, entities, &index, source_tolerance_mm) {
            out.push(candidate);
        }
    }
    out.sort_by_key(|candidate| candidate.solid_id);
    out
}

pub fn detect_radial_slot_revolutions(
    entities: &[EntityInstance],
) -> Vec<RecoveredRadialSlotRevolution> {
    let index = build_index(entities);
    let source_tolerances = source_tolerance_by_representation_item(entities, &index);
    let mut out = Vec::new();
    for entity in entities {
        let Some(record) = simple_record(entity) else {
            continue;
        };
        if record.name != "MANIFOLD_SOLID_BREP" {
            continue;
        }
        let solid_id = entity_id(entity);
        let source_tolerance_mm = source_tolerances
            .get(&solid_id)
            .copied()
            .unwrap_or(REVOLUTION_SOURCE_SUPPORT_TOL_MM);
        if let Some(candidate) =
            detect_one_radial_slot(solid_id, entities, &index, source_tolerance_mm)
        {
            out.push(candidate);
        }
    }
    out.sort_by_key(|candidate| candidate.base.solid_id);
    out
}

fn detect_one_solid(
    solid_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    source_tolerance_mm: f64,
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
    if let Some(mixed) = detect_mixed_curved_revolution(solid_id, &face_ids, &faces, &context) {
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
    let segment_context = ProfileSegmentContext::without_angular_trims(
        &context,
        axis_origin_mm,
        axis_direction,
        source_tolerance_mm,
    );
    for (face_index, face) in faces.iter().enumerate() {
        let (segment, residual) = linear_profile_segment(face_index, face, segment_context)?;
        max_residual_mm = max_residual_mm.max(residual);
        if max_residual_mm > source_tolerance_mm {
            return None;
        }
        push_unique_segment(&mut segments, segment);
    }
    let profile = MeridianProfile::from_lines(
        segments.into_iter().map(|segment| MeridianLine {
            start: segment.a,
            end: segment.b,
        }),
        GEOM_TOL_MM,
    )?;
    let profile_curves = profile.into_recovered();
    Some(RecoveredSolidRevolution {
        solid_id,
        face_ids,
        profile_curves,
        axis_origin_mm,
        axis_direction,
        radial_direction,
        max_residual_mm,
        source_tolerance_mm,
    })
}

fn detect_one_radial_slot(
    solid_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    source_tolerance_mm: f64,
) -> Option<RecoveredRadialSlotRevolution> {
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

    let (axis_reference_origin_mm, axis_reference_direction) =
        faces.iter().find_map(|face| match face.surface {
            SurfaceSupport::Cylinder(cylinder) => Some((cylinder.axis_origin_mm, cylinder.axis)),
            SurfaceSupport::Cone(cone) => Some((cone.reference_origin_mm, cone.axis)),
            _ => None,
        })?;
    let axis_direction = canonical_axis(axis_reference_direction);
    let axis_origin_mm =
        closest_axis_point_to_global_origin(axis_reference_origin_mm, axis_direction);

    let mut radial_plane_indices = Vec::<usize>::new();
    for (face_index, face) in faces.iter().enumerate() {
        match face.surface {
            SurfaceSupport::Cylinder(cylinder) => {
                if !parallel(cylinder.axis, axis_direction)
                    || axis_distance(cylinder.axis_origin_mm, axis_origin_mm, axis_direction)
                        > GEOM_TOL_MM
                {
                    return None;
                }
            }
            SurfaceSupport::Cone(cone) => {
                if !parallel(cone.axis, axis_direction)
                    || axis_distance(cone.reference_origin_mm, axis_origin_mm, axis_direction)
                        > GEOM_TOL_MM
                {
                    return None;
                }
            }
            SurfaceSupport::Plane(plane) => {
                let normal = normalize(plane.normal)?;
                let alignment = dot(normal, axis_direction).abs();
                if 1.0 - alignment <= DIR_TOL {
                    continue;
                }
                if alignment <= DIR_TOL {
                    radial_plane_indices.push(face_index);
                } else {
                    return None;
                }
            }
            _ => return None,
        }
    }
    let [plane_a, plane_b, plane_c] = radial_plane_indices.as_slice() else {
        return None;
    };
    let radial_plane_indices = [*plane_a, *plane_b, *plane_c];

    let plane_axis_distance = |face_index: usize| -> Option<f64> {
        let SurfaceSupport::Plane(plane) = faces.get(face_index)?.surface else {
            return None;
        };
        let normal = normalize(plane.normal)?;
        Some(dot(sub(axis_origin_mm, plane.origin_mm), normal).abs())
    };
    let on_axis = radial_plane_indices
        .iter()
        .copied()
        .filter(|&face_index| plane_axis_distance(face_index).is_some_and(|d| d <= GEOM_TOL_MM))
        .collect::<Vec<_>>();
    let [back_face_index] = on_axis.as_slice() else {
        return None;
    };
    let side_face_indices = radial_plane_indices
        .iter()
        .copied()
        .filter(|face_index| face_index != back_face_index)
        .collect::<Vec<_>>();
    let [side_a, side_b] = side_face_indices.as_slice() else {
        return None;
    };

    let SurfaceSupport::Plane(back_plane) = faces[*back_face_index].surface else {
        return None;
    };
    let SurfaceSupport::Plane(side_plane_a) = faces[*side_a].surface else {
        return None;
    };
    let SurfaceSupport::Plane(side_plane_b) = faces[*side_b].surface else {
        return None;
    };
    let back_normal = canonical_axis(back_plane.normal);
    let side_direction = canonical_axis(side_plane_a.normal);
    if !parallel(side_plane_a.normal, side_plane_b.normal)
        || dot(back_normal, side_direction).abs() > DIR_TOL
        || dot(back_normal, axis_direction).abs() > DIR_TOL
        || dot(side_direction, axis_direction).abs() > DIR_TOL
    {
        return None;
    }

    let side_offset_a = dot(sub(side_plane_a.origin_mm, axis_origin_mm), side_direction);
    let side_offset_b = dot(sub(side_plane_b.origin_mm, axis_origin_mm), side_direction);
    if !side_offset_a.is_finite()
        || !side_offset_b.is_finite()
        || side_offset_a * side_offset_b >= 0.0
    {
        return None;
    }
    let slot_half_width_mm = 0.5 * (side_offset_a.abs() + side_offset_b.abs());
    let mut slot_max_residual_mm = plane_axis_distance(*back_face_index)?
        .max((side_offset_a.abs() - side_offset_b.abs()).abs())
        .max((side_offset_a + side_offset_b).abs());
    if !slot_half_width_mm.is_finite()
        || slot_half_width_mm <= GEOM_TOL_MM
        || slot_max_residual_mm > source_tolerance_mm
    {
        return None;
    }

    let projected_range = |direction: [f64; 3]| -> Option<(f64, f64)> {
        let mut min_value = f64::INFINITY;
        let mut max_value = f64::NEG_INFINITY;
        for &face_index in &[*side_a, *side_b] {
            for edge in faces[face_index]
                .loops
                .iter()
                .flat_map(|loop_| &loop_.edges)
            {
                for point in [edge.start_mm, edge.end_mm] {
                    let radial = sub(
                        sub(point, axis_origin_mm),
                        mul(
                            axis_direction,
                            axial_coordinate(point, axis_origin_mm, axis_direction),
                        ),
                    );
                    let value = dot(radial, direction);
                    min_value = min_value.min(value);
                    max_value = max_value.max(value);
                }
            }
        }
        (min_value.is_finite() && max_value.is_finite()).then_some((min_value, max_value))
    };
    let candidate_outward = back_normal;
    let (candidate_min, candidate_max) = projected_range(candidate_outward)?;
    let slot_outward_direction =
        if candidate_min >= -source_tolerance_mm && candidate_max > GEOM_TOL_MM {
            candidate_outward
        } else if candidate_max <= source_tolerance_mm && candidate_min < -GEOM_TOL_MM {
            mul(candidate_outward, -1.0)
        } else {
            return None;
        };

    let mut excluded_profile_faces = radial_plane_indices.into_iter().collect::<HashSet<_>>();
    let angular_trim_faces = [*side_a, *side_b].into_iter().collect::<HashSet<_>>();

    let mut slot_min_t = f64::INFINITY;
    let mut slot_max_t = f64::NEG_INFINITY;
    for &face_index in &[*side_a, *side_b] {
        for edge in faces[face_index]
            .loops
            .iter()
            .flat_map(|loop_| &loop_.edges)
        {
            for point in [edge.start_mm, edge.end_mm] {
                let t = axial_coordinate(point, axis_origin_mm, axis_direction);
                slot_min_t = slot_min_t.min(t);
                slot_max_t = slot_max_t.max(t);
            }
        }
    }
    let mut solid_min_t = f64::INFINITY;
    let mut solid_max_t = f64::NEG_INFINITY;
    for face in &faces {
        for edge in face.loops.iter().flat_map(|loop_| &loop_.edges) {
            for point in [edge.start_mm, edge.end_mm] {
                let t = axial_coordinate(point, axis_origin_mm, axis_direction);
                solid_min_t = solid_min_t.min(t);
                solid_max_t = solid_max_t.max(t);
            }
        }
    }
    if !slot_min_t.is_finite()
        || !slot_max_t.is_finite()
        || !solid_min_t.is_finite()
        || !solid_max_t.is_finite()
        || slot_max_t - slot_min_t <= GEOM_TOL_MM
    {
        return None;
    }
    let matches_min = (slot_min_t - solid_min_t).abs() <= source_tolerance_mm;
    let matches_max = (slot_max_t - solid_max_t).abs() <= source_tolerance_mm;
    let (slot_root_axial_mm, slot_open_end_axial_mm) = match (matches_min, matches_max) {
        (true, false) => (slot_max_t, slot_min_t),
        (false, true) => (slot_min_t, slot_max_t),
        _ => return None,
    };

    let shares_edge = |first: usize, second: usize| {
        edge_faces
            .values()
            .any(|attached| attached.contains(&first) && attached.contains(&second))
    };
    let root_plane_faces = faces
        .iter()
        .enumerate()
        .filter_map(|(face_index, face)| {
            let SurfaceSupport::Plane(plane) = face.surface else {
                return None;
            };
            (parallel(plane.normal, axis_direction)
                && (axial_coordinate(plane.origin_mm, axis_origin_mm, axis_direction)
                    - slot_root_axial_mm)
                    .abs()
                    <= source_tolerance_mm
                && shares_edge(face_index, *side_a)
                && shares_edge(face_index, *side_b))
            .then_some(face_index)
        })
        .collect::<Vec<_>>();
    let [root_plane_face] = root_plane_faces.as_slice() else {
        return None;
    };
    // This axis-normal plane is the closed end of the slot cut, not a
    // boundary of the axisymmetric host that existed before the cut.
    excluded_profile_faces.insert(*root_plane_face);

    let mut max_residual_mm = 0.0_f64;
    let mut segments = Vec::<Segment2>::new();
    let mut base_face_ids = Vec::<u64>::new();
    let segment_context = ProfileSegmentContext {
        topology: &context,
        axis_origin_mm,
        axis_direction,
        source_tolerance_mm,
        angular_trim_faces: Some(&angular_trim_faces),
    };
    for (face_index, face) in faces.iter().enumerate() {
        if excluded_profile_faces.contains(&face_index) {
            continue;
        }
        let segment_result = linear_profile_segment(face_index, face, segment_context);
        let Some((segment, residual)) = segment_result else {
            #[cfg(test)]
            eprintln!(
                "radial-slot base profile rejected face #{} ({}) at index {face_index}",
                face_ids[face_index],
                surface_kind(&face.surface)
            );
            return None;
        };
        max_residual_mm = max_residual_mm.max(residual);
        if max_residual_mm > source_tolerance_mm {
            #[cfg(test)]
            eprintln!(
                "radial-slot base profile residual {:.12e} exceeds source tolerance {:.12e} after face #{}",
                max_residual_mm, source_tolerance_mm, face_ids[face_index]
            );
            return None;
        }
        push_unique_segment(&mut segments, segment);
        base_face_ids.push(face_ids[face_index]);
    }
    #[cfg(test)]
    let debug_segments = segments.clone();
    let Some(profile) = MeridianProfile::from_lines(
        segments.into_iter().map(|segment| MeridianLine {
            start: segment.a,
            end: segment.b,
        }),
        GEOM_TOL_MM,
    ) else {
        #[cfg(test)]
        eprintln!("radial-slot base profile failed closure: {debug_segments:?}");
        return None;
    };
    let profile_curves = profile.into_recovered();
    let Some(radial_direction) = radial_basis(axis_direction) else {
        #[cfg(test)]
        eprintln!("radial-slot base profile failed radial basis for axis {axis_direction:?}");
        return None;
    };
    let base = RecoveredSolidRevolution {
        solid_id,
        face_ids: base_face_ids,
        profile_curves,
        axis_origin_mm,
        axis_direction,
        radial_direction,
        max_residual_mm,
        source_tolerance_mm,
    };

    slot_max_residual_mm = slot_max_residual_mm.max(
        (slot_min_t - solid_min_t)
            .abs()
            .min((slot_max_t - solid_max_t).abs()),
    );
    if slot_max_residual_mm > source_tolerance_mm {
        #[cfg(test)]
        eprintln!(
            "radial-slot slot residual {:.12e} exceeds source tolerance {:.12e}",
            slot_max_residual_mm, source_tolerance_mm
        );
        return None;
    }
    let mut side_face_ids = [face_ids[*side_a], face_ids[*side_b]];
    side_face_ids.sort_unstable();

    Some(RecoveredRadialSlotRevolution {
        base,
        slot_face_ids: [
            side_face_ids[0],
            side_face_ids[1],
            face_ids[*back_face_index],
        ],
        slot_outward_direction,
        slot_side_direction: side_direction,
        slot_half_width_mm,
        slot_root_axial_mm,
        slot_open_end_axial_mm,
        slot_max_residual_mm,
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

    let profile_curves = MeridianProfile::closed(
        vec![MeridianCurve::Arc(MeridianArc {
            source_edge_ids: Vec::new(),
            center: [first.major_radius_mm, center_t],
            radius: first.minor_radius_mm,
            start_angle: 0.0,
            end_angle: std::f64::consts::TAU,
        })],
        GEOM_TOL_MM,
    )?
    .into_recovered();
    Some(RecoveredSolidRevolution {
        solid_id,
        face_ids: face_ids.to_vec(),
        profile_curves,
        axis_origin_mm,
        axis_direction,
        radial_direction,
        max_residual_mm,
        source_tolerance_mm: REVOLUTION_SOURCE_SUPPORT_TOL_MM,
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
    let profile_curves = MeridianProfile::closed(
        vec![
            MeridianCurve::Line(MeridianLine {
                start: [0.0, plane_t],
                end: [cap_radius, plane_t],
            }),
            MeridianCurve::Arc(MeridianArc {
                source_edge_ids: Vec::new(),
                center: [0.0, center_t],
                radius: first_sphere.radius_mm,
                start_angle: cap_angle,
                end_angle: pole_angle,
            }),
            MeridianCurve::Line(MeridianLine {
                start: [0.0, pole_t],
                end: [0.0, plane_t],
            }),
        ],
        GEOM_TOL_MM,
    )?
    .into_recovered();
    Some(RecoveredSolidRevolution {
        solid_id,
        face_ids: face_ids.to_vec(),
        profile_curves,
        axis_origin_mm,
        axis_direction,
        radial_direction,
        max_residual_mm,
        source_tolerance_mm: REVOLUTION_SOURCE_SUPPORT_TOL_MM,
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

    let segment_context = ProfileSegmentContext::without_angular_trims(
        context,
        axis_origin_mm,
        axis_direction,
        REVOLUTION_SOURCE_SUPPORT_TOL_MM,
    );
    let (plane_segment, plane_residual) =
        linear_profile_segment(plane_index, plane_face, segment_context)?;
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

    for (face_index, face) in [
        (*cylinder_index_a, *cylinder_face_a),
        (*cylinder_index_b, *cylinder_face_b),
    ] {
        let (segment, residual) = linear_profile_segment(face_index, face, segment_context)?;
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
    let profile_curves = MeridianProfile::closed(
        vec![
            MeridianCurve::Line(MeridianLine {
                start: [0.0, plane_t],
                end: [radius_mm, plane_t],
            }),
            MeridianCurve::Line(MeridianLine {
                start: [radius_mm, plane_t],
                end: [radius_mm, center_t],
            }),
            MeridianCurve::Arc(MeridianArc {
                source_edge_ids: Vec::new(),
                center: [0.0, center_t],
                radius: radius_mm,
                start_angle: 0.0,
                end_angle: cap_angle,
            }),
            MeridianCurve::Line(MeridianLine {
                start: [0.0, pole_t],
                end: [0.0, plane_t],
            }),
        ],
        GEOM_TOL_MM,
    )?
    .into_recovered();
    Some(RecoveredSolidRevolution {
        solid_id,
        face_ids: face_ids.to_vec(),
        profile_curves,
        axis_origin_mm,
        axis_direction,
        radial_direction,
        max_residual_mm,
        source_tolerance_mm: REVOLUTION_SOURCE_SUPPORT_TOL_MM,
    })
}

fn detect_mixed_curved_revolution(
    solid_id: u64,
    face_ids: &[u64],
    faces: &[FaceInfo],
    context: &TopologyContext<'_>,
) -> Option<RecoveredSolidRevolution> {
    let curved_count = faces
        .iter()
        .filter(|face| {
            matches!(
                face.surface,
                SurfaceSupport::Torus(_) | SurfaceSupport::Sphere(_)
            )
        })
        .count();
    if curved_count == 0 || curved_count == faces.len() {
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
    let mut sphere_faces = Vec::<(usize, brep::SphereSupport)>::new();
    let segment_context = ProfileSegmentContext::without_angular_trims(
        context,
        axis_origin_mm,
        axis_direction,
        REVOLUTION_SOURCE_SUPPORT_TOL_MM,
    );
    for (face_index, face) in faces.iter().enumerate() {
        match face.surface {
            SurfaceSupport::Cylinder(_)
            | SurfaceSupport::Cone(_)
            | SurfaceSupport::Revolution(_)
            | SurfaceSupport::Plane(_) => {
                let (segment, residual) =
                    linear_profile_segment(face_index, face, segment_context)?;
                max_residual_mm = max_residual_mm.max(residual);
                push_unique_segment(&mut segments, segment);
            }
            SurfaceSupport::Torus(torus) => torus_faces.push((face_index, torus)),
            SurfaceSupport::Sphere(sphere) => sphere_faces.push((face_index, sphere)),
            _ => return None,
        }
    }
    if max_residual_mm > REVOLUTION_SOURCE_SUPPORT_TOL_MM {
        return None;
    }

    let mut torus_groups = Vec::<Vec<usize>>::new();
    let mut torus_supports = Vec::<brep::TorusSupport>::new();
    for (face_index, torus) in torus_faces {
        if !torus.major_radius_mm.is_finite()
            || !torus.minor_radius_mm.is_finite()
            || torus.minor_radius_mm <= GEOM_TOL_MM
            || torus.major_radius_mm <= GEOM_TOL_MM
            || !parallel(torus.axis, axis_direction)
            || axis_distance(torus.center_mm, axis_origin_mm, axis_direction) > GEOM_TOL_MM
        {
            return None;
        }
        if let Some(group_index) = torus_supports
            .iter()
            .position(|existing| same_torus_support(*existing, torus, axis_direction))
        {
            torus_groups[group_index].push(face_index);
        } else {
            torus_supports.push(torus);
            torus_groups.push(vec![face_index]);
        }
    }

    let mut sphere_groups = Vec::<Vec<usize>>::new();
    let mut sphere_supports = Vec::<brep::SphereSupport>::new();
    for (face_index, sphere) in sphere_faces {
        if !sphere.radius_mm.is_finite()
            || sphere.radius_mm <= GEOM_TOL_MM
            || axis_distance(sphere.center_mm, axis_origin_mm, axis_direction) > GEOM_TOL_MM
        {
            return None;
        }
        if let Some(group_index) = sphere_supports
            .iter()
            .position(|existing| same_sphere_support(*existing, sphere))
        {
            sphere_groups[group_index].push(face_index);
        } else {
            sphere_supports.push(sphere);
            sphere_groups.push(vec![face_index]);
        }
    }

    let mut arcs = Vec::with_capacity(torus_groups.len() + sphere_groups.len());
    for (group, torus) in torus_groups.iter().zip(&torus_supports) {
        let (arc, residual) = torus_profile_arc(
            group,
            *torus,
            axis_origin_mm,
            axis_direction,
            faces,
            context.edge_faces,
        )?;
        max_residual_mm = max_residual_mm.max(residual);
        arcs.push(arc);
    }
    for (group, sphere) in sphere_groups.iter().zip(&sphere_supports) {
        let (arc, residual) = sphere_profile_arc(
            group,
            *sphere,
            axis_origin_mm,
            axis_direction,
            faces,
            context.edge_faces,
        )?;
        max_residual_mm = max_residual_mm.max(residual);
        arcs.push(arc);
    }
    if arcs.is_empty() || max_residual_mm > REVOLUTION_SOURCE_SUPPORT_TOL_MM {
        return None;
    }

    let mut curves = segments
        .into_iter()
        .map(|segment| {
            MeridianCurve::Line(MeridianLine {
                start: segment.a,
                end: segment.b,
            })
        })
        .collect::<Vec<_>>();
    curves.extend(arcs);
    let profile_curves = MeridianProfile::closed(curves, GEOM_TOL_MM)?.into_recovered();
    Some(RecoveredSolidRevolution {
        solid_id,
        face_ids: face_ids.to_vec(),
        profile_curves,
        axis_origin_mm,
        axis_direction,
        radial_direction,
        max_residual_mm,
        source_tolerance_mm: REVOLUTION_SOURCE_SUPPORT_TOL_MM,
    })
}

fn same_sphere_support(a: brep::SphereSupport, b: brep::SphereSupport) -> bool {
    norm(sub(a.center_mm, b.center_mm)) <= GEOM_TOL_MM
        && (a.radius_mm - b.radius_mm).abs() <= GEOM_TOL_MM
}

fn sphere_profile_arc(
    face_indices: &[usize],
    sphere: brep::SphereSupport,
    axis_origin_mm: [f64; 3],
    axis_direction: [f64; 3],
    faces: &[FaceInfo],
    edge_faces: &HashMap<u64, Vec<usize>>,
) -> Option<(MeridianCurve, f64)> {
    if face_indices.is_empty()
        || !sphere.radius_mm.is_finite()
        || sphere.radius_mm <= GEOM_TOL_MM
        || axis_distance(sphere.center_mm, axis_origin_mm, axis_direction) > GEOM_TOL_MM
    {
        return None;
    }

    let center_t = axial_coordinate(sphere.center_mm, axis_origin_mm, axis_direction);
    let center_mm = [0.0, center_t];
    let mut boundary_points = Vec::<[f64; 2]>::new();
    let mut witness_points = Vec::<[f64; 2]>::new();
    let mut source_edge_ids = Vec::<u64>::new();
    let mut max_residual_mm = axis_distance(sphere.center_mm, axis_origin_mm, axis_direction);

    for &face_index in face_indices {
        let face = faces.get(face_index)?;
        let SurfaceSupport::Sphere(face_sphere) = face.surface else {
            return None;
        };
        if !same_sphere_support(sphere, face_sphere) || face.loops.len() != 1 {
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
                    .max(sphere_point_residual(point, sphere));
            }
            let neighbor = unique_neighbor_face(face_index, edge.edge_id, edge_faces)?;
            match faces.get(neighbor)?.surface {
                SurfaceSupport::Sphere(neighbor_sphere)
                    if same_sphere_support(sphere, neighbor_sphere) =>
                {
                    max_residual_mm = max_residual_mm.max(sphere_meridian_circle_residual(
                        circle,
                        sphere,
                        axis_origin_mm,
                        axis_direction,
                    )?);
                    let midpoint = circle_trim_midpoint(edge, circle)?;
                    max_residual_mm = max_residual_mm.max(sphere_point_residual(midpoint, sphere));
                    push_unique_point(
                        &mut witness_points,
                        [
                            point_axis_distance(midpoint, axis_origin_mm, axis_direction),
                            axial_coordinate(midpoint, axis_origin_mm, axis_direction),
                        ],
                    );
                }
                _ => {
                    let (point, residual) = sphere_parallel_boundary_point(
                        circle,
                        sphere,
                        axis_origin_mm,
                        axis_direction,
                    )?;
                    max_residual_mm = max_residual_mm.max(residual);
                    push_unique_point(&mut boundary_points, point);
                }
            }
        }
    }

    if boundary_points.is_empty()
        || boundary_points.len() > 2
        || witness_points.is_empty()
        || max_residual_mm > GEOM_TOL_MM
    {
        return None;
    }

    let angle_tol = (GEOM_TOL_MM / sphere.radius_mm).max(1.0e-12);
    let mut candidates = Vec::<(f64, f64)>::new();
    if boundary_points.len() == 1 {
        let start = meridian::angle(boundary_points[0], center_mm, sphere.radius_mm, GEOM_TOL_MM)?;
        candidates.push((start, std::f64::consts::FRAC_PI_2));
        candidates.push((start, -std::f64::consts::FRAC_PI_2));
    } else {
        let start = meridian::angle(boundary_points[0], center_mm, sphere.radius_mm, GEOM_TOL_MM)?;
        let end = meridian::angle(boundary_points[1], center_mm, sphere.radius_mm, GEOM_TOL_MM)?;
        let ccw_delta = meridian::positive_angle_delta(start, end);
        let cw_delta = std::f64::consts::TAU - ccw_delta;
        if ccw_delta <= angle_tol || cw_delta <= angle_tol {
            return None;
        }
        candidates.push((start, start + ccw_delta));
        candidates.push((start, start - cw_delta));
    }

    let valid = candidates
        .into_iter()
        .filter(|&(arc_start, arc_end)| {
            let curve = MeridianCurve::Arc(MeridianArc {
                source_edge_ids: Vec::new(),
                center: center_mm,
                radius: sphere.radius_mm,
                start_angle: arc_start,
                end_angle: arc_end,
            });
            curve.min_radius(GEOM_TOL_MM) >= -GEOM_TOL_MM
                && witness_points.iter().all(|point| {
                    meridian::angle(*point, center_mm, sphere.radius_mm, GEOM_TOL_MM).is_some_and(
                        |angle| meridian::angle_on_arc(angle, arc_start, arc_end, angle_tol),
                    )
                })
        })
        .collect::<Vec<_>>();
    let [(start_angle, end_angle)] = valid.as_slice() else {
        return None;
    };
    source_edge_ids.sort_unstable();
    source_edge_ids.dedup();
    Some((
        MeridianCurve::Arc(MeridianArc {
            source_edge_ids,
            center: center_mm,
            radius: sphere.radius_mm,
            start_angle: *start_angle,
            end_angle: *end_angle,
        }),
        max_residual_mm,
    ))
}

fn sphere_parallel_boundary_point(
    circle: brep::CircleSupport,
    sphere: brep::SphereSupport,
    axis_origin_mm: [f64; 3],
    axis_direction: [f64; 3],
) -> Option<([f64; 2], f64)> {
    if !parallel(circle.normal, axis_direction) || circle.radius_mm <= GEOM_TOL_MM {
        return None;
    }
    let axial = axial_coordinate(circle.center_mm, axis_origin_mm, axis_direction);
    let residual = sphere_circle_residual(circle, sphere)?.max(axis_distance(
        circle.center_mm,
        axis_origin_mm,
        axis_direction,
    ));
    (residual <= GEOM_TOL_MM).then_some(([circle.radius_mm, axial], residual))
}

fn sphere_meridian_circle_residual(
    circle: brep::CircleSupport,
    sphere: brep::SphereSupport,
    axis_origin_mm: [f64; 3],
    axis_direction: [f64; 3],
) -> Option<f64> {
    let normal = normalize(circle.normal)?;
    if dot(normal, axis_direction).abs() > DIR_TOL {
        return None;
    }
    let center_t = axial_coordinate(sphere.center_mm, axis_origin_mm, axis_direction);
    let axis_point = add(axis_origin_mm, mul(axis_direction, center_t));
    let plane_residual = dot(sub(axis_point, circle.center_mm), normal).abs();
    let residual = sphere_circle_residual(circle, sphere)?
        .max(norm(sub(circle.center_mm, sphere.center_mm)))
        .max((circle.radius_mm - sphere.radius_mm).abs())
        .max(plane_residual);
    (residual <= GEOM_TOL_MM).then_some(residual)
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
) -> Option<(MeridianCurve, f64)> {
    if face_indices.is_empty()
        || torus.major_radius_mm <= GEOM_TOL_MM
        || torus.minor_radius_mm <= GEOM_TOL_MM
    {
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
    let start_angle = meridian::angle(*start, center_mm, torus.minor_radius_mm, GEOM_TOL_MM)?;
    let end_angle = meridian::angle(*end, center_mm, torus.minor_radius_mm, GEOM_TOL_MM)?;
    let ccw_delta = meridian::positive_angle_delta(start_angle, end_angle);
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
            let curve = MeridianCurve::Arc(MeridianArc {
                source_edge_ids: Vec::new(),
                center: center_mm,
                radius: torus.minor_radius_mm,
                start_angle: arc_start,
                end_angle: arc_end,
            });
            curve.min_radius(GEOM_TOL_MM) >= -GEOM_TOL_MM
                && witness_points.iter().all(|point| {
                    meridian::angle(*point, center_mm, torus.minor_radius_mm, GEOM_TOL_MM)
                        .is_some_and(|angle| {
                            meridian::angle_on_arc(angle, arc_start, arc_end, angle_tol)
                        })
                })
        })
        .collect::<Vec<_>>();
    let [(start_angle, end_angle)] = valid.as_slice() else {
        return None;
    };

    source_edge_ids.sort_unstable();
    source_edge_ids.dedup();
    Some((
        MeridianCurve::Arc(MeridianArc {
            source_edge_ids,
            center: center_mm,
            radius: torus.minor_radius_mm,
            start_angle: *start_angle,
            end_angle: *end_angle,
        }),
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
        meridian::positive_angle_delta(start, end)
    } else {
        -meridian::positive_angle_delta(end, start)
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

fn push_unique_point(points: &mut Vec<[f64; 2]>, candidate: [f64; 2]) {
    if !points
        .iter()
        .any(|point| distance2(*point, candidate) <= GEOM_TOL_MM)
    {
        points.push(candidate);
    }
}

fn linear_profile_segment(
    face_index: usize,
    face: &FaceInfo,
    segment_context: ProfileSegmentContext<'_, '_>,
) -> Option<(Segment2, f64)> {
    match face.surface {
        SurfaceSupport::Cylinder(cylinder) => {
            cylinder_profile_segment_with_angular_trims(face_index, face, cylinder, segment_context)
        }
        SurfaceSupport::Cone(cone) => {
            cone_profile_segment_with_angular_trims(face_index, face, cone, segment_context)
        }
        SurfaceSupport::Revolution(revolution) => {
            revolution_line_profile_segment(face_index, face, revolution, segment_context)
        }
        SurfaceSupport::Plane(plane) => {
            plane_profile_segment_with_angular_trims(face_index, face, plane, segment_context)
        }
        _ => None,
    }
}

fn cylinder_profile_segment_with_angular_trims(
    face_index: usize,
    face: &FaceInfo,
    cylinder: brep::CylinderSupport,
    segment_context: ProfileSegmentContext<'_, '_>,
) -> Option<(Segment2, f64)> {
    let ProfileSegmentContext {
        topology: context,
        axis_origin_mm: axis_origin,
        axis_direction: axis,
        source_tolerance_mm,
        angular_trim_faces,
    } = segment_context;
    if face.loops.len() != 1
        || !parallel(cylinder.axis, axis)
        || axis_distance(cylinder.axis_origin_mm, axis_origin, axis) > GEOM_TOL_MM
        || !cylinder.radius_mm.is_finite()
        || cylinder.radius_mm <= GEOM_TOL_MM
    {
        return None;
    }

    let mut boundary_t = Vec::<f64>::new();
    let mut raw_t = Vec::<f64>::new();
    let mut geometry_residual = axis_distance(cylinder.axis_origin_mm, axis_origin, axis);
    let mut source_support_residual = 0.0_f64;

    for edge in &face.loops[0].edges {
        for point in [edge.start_mm, edge.end_mm] {
            let radius = point_axis_distance(point, axis_origin, axis);
            let t = axial_coordinate(point, axis_origin, axis);
            if !radius.is_finite() || !t.is_finite() {
                return None;
            }
            source_support_residual =
                source_support_residual.max((radius - cylinder.radius_mm).abs());
            raw_t.push(t);
        }

        if angular_trim_faces.is_some_and(|trim_faces| {
            unique_neighbor_face(face_index, edge.edge_id, context.edge_faces)
                .is_some_and(|neighbor| trim_faces.contains(&neighbor))
        }) {
            continue;
        }

        match &edge.support {
            CurveSupport::Line(line) => {
                let alignment = 1.0 - dot(line.direction, axis).abs();
                if alignment > DIR_TOL {
                    return None;
                }
                source_support_residual = source_support_residual.max(
                    (point_axis_distance(line.origin_mm, axis_origin, axis) - cylinder.radius_mm)
                        .abs(),
                );
                let direction = normalize(line.direction)?;
                for point in [edge.start_mm, edge.end_mm] {
                    source_support_residual = source_support_residual
                        .max(norm(cross(sub(point, line.origin_mm), direction)));
                }
            }
            CurveSupport::Circle(circle) => {
                if !parallel(circle.normal, axis)
                    || axis_distance(circle.center_mm, axis_origin, axis) > GEOM_TOL_MM
                    || (circle.radius_mm - cylinder.radius_mm).abs() > GEOM_TOL_MM
                {
                    return None;
                }
                let t = axial_coordinate(circle.center_mm, axis_origin, axis);
                if !t.is_finite() {
                    return None;
                }
                push_unique_scalar(&mut boundary_t, t);
                for point in [edge.start_mm, edge.end_mm] {
                    source_support_residual =
                        source_support_residual.max(circle_point_residual(point, *circle));
                }
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
                if !plane_t.is_finite() {
                    return None;
                }
                push_unique_scalar(&mut boundary_t, plane_t);
                for point in [edge.start_mm, edge.end_mm] {
                    source_support_residual = source_support_residual
                        .max((axial_coordinate(point, axis_origin, axis) - plane_t).abs())
                        .max(
                            (point_axis_distance(point, axis_origin, axis) - cylinder.radius_mm)
                                .abs(),
                        );
                }
                geometry_residual = geometry_residual.max(plane.max_residual_mm);
            }
            CurveSupport::Other { .. } => return None,
        }
    }

    if source_support_residual > source_tolerance_mm {
        return None;
    }

    let (min_t, max_t) = if boundary_t.len() >= 2 {
        let min_t = boundary_t.iter().copied().reduce(f64::min)?;
        let max_t = boundary_t.iter().copied().reduce(f64::max)?;
        (min_t, max_t)
    } else {
        let min_t = raw_t.iter().copied().reduce(f64::min)?;
        let max_t = raw_t.iter().copied().reduce(f64::max)?;
        (min_t, max_t)
    };
    if !min_t.is_finite() || !max_t.is_finite() || max_t - min_t <= GEOM_TOL_MM {
        return None;
    }

    if boundary_t.len() >= 2
        && raw_t
            .iter()
            .any(|&t| t < min_t - source_tolerance_mm || t > max_t + source_tolerance_mm)
    {
        return None;
    }

    Some((
        Segment2 {
            a: [cylinder.radius_mm, min_t],
            b: [cylinder.radius_mm, max_t],
        },
        geometry_residual.max(source_support_residual),
    ))
}

fn push_unique_scalar(values: &mut Vec<f64>, candidate: f64) {
    if !values
        .iter()
        .any(|existing| (*existing - candidate).abs() <= GEOM_TOL_MM)
    {
        values.push(candidate);
    }
}

fn cone_profile_segment_with_angular_trims(
    face_index: usize,
    face: &FaceInfo,
    cone: brep::ConeSupport,
    segment_context: ProfileSegmentContext<'_, '_>,
) -> Option<(Segment2, f64)> {
    cone_profile_segment_with_angular_trims_inner(face_index, face, cone, segment_context, true)
}

fn cone_profile_segment_with_angular_trims_inner(
    face_index: usize,
    face: &FaceInfo,
    cone: brep::ConeSupport,
    segment_context: ProfileSegmentContext<'_, '_>,
    allow_sibling_fallback: bool,
) -> Option<(Segment2, f64)> {
    let ProfileSegmentContext {
        topology: context,
        axis_origin_mm: axis_origin,
        axis_direction: axis,
        source_tolerance_mm,
        angular_trim_faces,
    } = segment_context;
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

    let source_axis = normalize(cone.axis)?;
    let source_axis_alignment = dot(source_axis, axis);
    if (1.0 - source_axis_alignment.abs()) > DIR_TOL {
        return None;
    }
    let source_reference_t = axial_coordinate(cone.reference_origin_mm, axis_origin, axis);
    let source_slope = source_axis_alignment.signum() * cone.semi_angle_rad.tan();
    if !source_reference_t.is_finite() || !source_slope.is_finite() {
        return None;
    }
    let source_radius_at =
        |t: f64| cone.reference_radius_mm + source_slope * (t - source_reference_t);

    let mut boundary_samples = Vec::<[f64; 2]>::new();
    let mut raw_samples = Vec::<[f64; 2]>::new();
    let mut apex_vertices = HashMap::<u64, (usize, [f64; 2])>::new();
    let mut source_support_residual = axis_distance(cone.reference_origin_mm, axis_origin, axis);

    for edge in &face.loops[0].edges {
        let angular_trim = angular_trim_faces.is_some_and(|trim_faces| {
            unique_neighbor_face(face_index, edge.edge_id, context.edge_faces)
                .is_some_and(|neighbor| trim_faces.contains(&neighbor))
        });
        for (vertex_id, point) in [
            (edge.start_vertex, edge.start_mm),
            (edge.end_vertex, edge.end_mm),
        ] {
            let sample = [
                point_axis_distance(point, axis_origin, axis),
                axial_coordinate(point, axis_origin, axis),
            ];
            if !sample[0].is_finite() || !sample[1].is_finite() {
                return None;
            }
            let source_radius = source_radius_at(sample[1]);
            if !source_radius.is_finite() || source_radius < -source_tolerance_mm {
                return None;
            }
            source_support_residual =
                source_support_residual.max((sample[0] - source_radius).abs());
            raw_samples.push(sample);

            if !angular_trim
                && matches!(edge.support, CurveSupport::Line(_))
                && sample[0] <= GEOM_TOL_MM
            {
                let entry = apex_vertices.entry(vertex_id).or_insert((0, sample));
                entry.0 += 1;
                entry.1[0] = entry.1[0].min(sample[0]);
                entry.1[1] = sample[1];
            }
        }

        if angular_trim {
            continue;
        }

        match &edge.support {
            CurveSupport::Line(line) => {
                let direction = normalize(line.direction)?;
                let edge_length = norm(sub(edge.end_mm, edge.start_mm));
                if edge_length <= GEOM_TOL_MM {
                    return None;
                }

                let signed_axial = dot(direction, axis);
                let radial_vector = sub(direction, mul(axis, signed_axial));
                let radial = norm(radial_vector);
                if radial <= DIR_TOL {
                    return None;
                }
                let meridian_normal = normalize(cross(axis, radial_vector))?;
                source_support_residual = source_support_residual
                    .max(dot(sub(line.origin_mm, axis_origin), meridian_normal).abs());

                let line_angle = radial.atan2(signed_axial.abs());
                let angle_error = (line_angle - cone.semi_angle_rad).abs();
                source_support_residual =
                    source_support_residual.max(edge_length * angle_error.sin().abs());

                for point in [edge.start_mm, edge.end_mm] {
                    source_support_residual = source_support_residual
                        .max(norm(cross(sub(point, line.origin_mm), direction)));
                }
            }
            CurveSupport::Circle(circle) => {
                if !circle.radius_mm.is_finite()
                    || circle.radius_mm <= GEOM_TOL_MM
                    || !parallel(circle.normal, axis)
                    || axis_distance(circle.center_mm, axis_origin, axis) > GEOM_TOL_MM
                {
                    return None;
                }
                let circle_t = axial_coordinate(circle.center_mm, axis_origin, axis);
                if !circle_t.is_finite() {
                    return None;
                }
                let mut boundary_sample = [circle.radius_mm, circle_t];

                if let Some(neighbor) =
                    unique_neighbor_face(face_index, edge.edge_id, context.edge_faces)
                {
                    match context.faces.get(neighbor)?.surface {
                        SurfaceSupport::Plane(plane) => {
                            if !parallel(plane.normal, axis) {
                                return None;
                            }
                            let plane_t = axial_coordinate(plane.origin_mm, axis_origin, axis);
                            if !plane_t.is_finite() {
                                return None;
                            }
                            source_support_residual = source_support_residual
                                .max((circle_t - plane_t).abs())
                                .max(plane.max_residual_mm);
                            boundary_sample[1] = plane_t;
                        }
                        SurfaceSupport::Cylinder(cylinder) => {
                            if !parallel(cylinder.axis, axis)
                                || axis_distance(cylinder.axis_origin_mm, axis_origin, axis)
                                    > GEOM_TOL_MM
                                || !cylinder.radius_mm.is_finite()
                                || cylinder.radius_mm <= GEOM_TOL_MM
                            {
                                return None;
                            }
                            source_support_residual = source_support_residual
                                .max((circle.radius_mm - cylinder.radius_mm).abs());
                            boundary_sample[0] = cylinder.radius_mm;
                        }
                        _ => {}
                    }
                }

                let source_radius = source_radius_at(boundary_sample[1]);
                if !source_radius.is_finite() || source_radius < -source_tolerance_mm {
                    return None;
                }
                source_support_residual =
                    source_support_residual.max((boundary_sample[0] - source_radius).abs());
                for point in [edge.start_mm, edge.end_mm] {
                    source_support_residual =
                        source_support_residual.max(circle_point_residual(point, *circle));
                }
                push_unique_point(&mut boundary_samples, boundary_sample);
            }
            CurveSupport::BSpline(_) => {
                let neighbor = unique_neighbor_face(face_index, edge.edge_id, context.edge_faces)?;
                let SurfaceSupport::Plane(plane) = context.faces.get(neighbor)?.surface else {
                    return None;
                };
                if !parallel(plane.normal, axis) {
                    return None;
                }
                let t = axial_coordinate(plane.origin_mm, axis_origin, axis);
                let source_radius = source_radius_at(t);
                if !source_radius.is_finite() || source_radius < -source_tolerance_mm {
                    return None;
                }
                for point in [edge.start_mm, edge.end_mm] {
                    source_support_residual = source_support_residual
                        .max((axial_coordinate(point, axis_origin, axis) - t).abs())
                        .max((point_axis_distance(point, axis_origin, axis) - source_radius).abs());
                }
                source_support_residual = source_support_residual.max(plane.max_residual_mm);
                push_unique_point(&mut boundary_samples, [source_radius.max(0.0), t]);
            }
            CurveSupport::Other { .. } => return None,
        }
    }

    if source_support_residual > source_tolerance_mm {
        return None;
    }

    let apexes = apex_vertices
        .into_values()
        .filter_map(|(count, sample)| (count >= 2).then_some([0.0, sample[1]]))
        .collect::<Vec<_>>();
    match boundary_samples.len() {
        0 => return None,
        1 => {
            if let [apex] = apexes.as_slice() {
                if (apex[1] - boundary_samples[0][1]).abs() <= GEOM_TOL_MM {
                    return None;
                }
                push_unique_point(&mut boundary_samples, *apex);
            } else {
                if !allow_sibling_fallback {
                    return None;
                }
                for (sibling_index, sibling_face) in context.faces.iter().enumerate() {
                    if sibling_index == face_index {
                        continue;
                    }
                    let SurfaceSupport::Cone(sibling_cone) = sibling_face.surface else {
                        continue;
                    };
                    if !same_cone_meridian_support(cone, sibling_cone, axis_origin, axis) {
                        continue;
                    }
                    let Some((sibling_segment, sibling_residual)) =
                        cone_profile_segment_with_angular_trims_inner(
                            sibling_index,
                            sibling_face,
                            sibling_cone,
                            segment_context,
                            false,
                        )
                    else {
                        continue;
                    };
                    let coverage_residual = raw_samples
                        .iter()
                        .copied()
                        .map(|sample| profile_point_segment_distance(sample, sibling_segment))
                        .fold(0.0_f64, f64::max);
                    if coverage_residual <= source_tolerance_mm {
                        return Some((
                            sibling_segment,
                            source_support_residual
                                .max(sibling_residual)
                                .max(coverage_residual),
                        ));
                    }
                }
                return None;
            }
        }
        _ => {
            if !apexes.is_empty() {
                // A trimmed cone crossing an apex would require a V-shaped
                // absolute-radius meridian, not one straight generatrix.
                return None;
            }
        }
    }

    let min_sample = *boundary_samples
        .iter()
        .min_by(|a, b| a[1].total_cmp(&b[1]))?;
    let max_sample = *boundary_samples
        .iter()
        .max_by(|a, b| a[1].total_cmp(&b[1]))?;
    let delta_t = max_sample[1] - min_sample[1];
    if delta_t <= GEOM_TOL_MM || (min_sample[0] <= GEOM_TOL_MM && max_sample[0] <= GEOM_TOL_MM) {
        return None;
    }

    let slope = (max_sample[0] - min_sample[0]) / delta_t;
    if !slope.is_finite() {
        return None;
    }
    let radius_at = |t: f64| min_sample[0] + slope * (t - min_sample[1]);

    let mut profile_residual = 0.0_f64;
    for sample in &boundary_samples {
        profile_residual = profile_residual.max((sample[0] - radius_at(sample[1])).abs());
    }
    if profile_residual > GEOM_TOL_MM {
        return None;
    }

    for sample in raw_samples {
        let radius = radius_at(sample[1]);
        if !radius.is_finite() || radius < -source_tolerance_mm {
            return None;
        }
        source_support_residual = source_support_residual.max((sample[0] - radius.max(0.0)).abs());
    }
    if source_support_residual > source_tolerance_mm {
        return None;
    }

    Some((
        Segment2 {
            a: [min_sample[0].max(0.0), min_sample[1]],
            b: [max_sample[0].max(0.0), max_sample[1]],
        },
        profile_residual.max(source_support_residual),
    ))
}

fn same_cone_meridian_support(
    first: brep::ConeSupport,
    second: brep::ConeSupport,
    axis_origin: [f64; 3],
    axis: [f64; 3],
) -> bool {
    if axis_distance(first.reference_origin_mm, axis_origin, axis) > GEOM_TOL_MM
        || axis_distance(second.reference_origin_mm, axis_origin, axis) > GEOM_TOL_MM
    {
        return false;
    }
    let Some(first_axis) = normalize(first.axis) else {
        return false;
    };
    let Some(second_axis) = normalize(second.axis) else {
        return false;
    };
    let first_alignment = dot(first_axis, axis);
    let second_alignment = dot(second_axis, axis);
    if 1.0 - first_alignment.abs() > DIR_TOL || 1.0 - second_alignment.abs() > DIR_TOL {
        return false;
    }

    let first_t = axial_coordinate(first.reference_origin_mm, axis_origin, axis);
    let second_t = axial_coordinate(second.reference_origin_mm, axis_origin, axis);
    let first_slope = first_alignment.signum() * first.semi_angle_rad.tan();
    let second_slope = second_alignment.signum() * second.semi_angle_rad.tan();
    let first_intercept = first.reference_radius_mm - first_slope * first_t;
    let second_intercept = second.reference_radius_mm - second_slope * second_t;
    first_slope.is_finite()
        && second_slope.is_finite()
        && first_intercept.is_finite()
        && second_intercept.is_finite()
        && (first_slope - second_slope).abs() <= DIR_TOL
        && (first_intercept - second_intercept).abs() <= GEOM_TOL_MM
}

fn profile_point_segment_distance(point: [f64; 2], segment: Segment2) -> f64 {
    let delta = sub2(segment.b, segment.a);
    let denominator = delta[0].mul_add(delta[0], delta[1] * delta[1]);
    if !denominator.is_finite() || denominator <= GEOM_TOL_MM.powi(2) {
        return distance2(point, segment.a);
    }
    let relative = sub2(point, segment.a);
    let fraction = relative[0].mul_add(delta[0], relative[1] * delta[1]) / denominator;
    let fraction = fraction.clamp(0.0, 1.0);
    let projected = [
        delta[0].mul_add(fraction, segment.a[0]),
        delta[1].mul_add(fraction, segment.a[1]),
    ];
    distance2(point, projected)
}

fn revolution_line_profile_segment(
    face_index: usize,
    face: &FaceInfo,
    revolution: brep::RevolutionSurfaceSupport,
    segment_context: ProfileSegmentContext<'_, '_>,
) -> Option<(Segment2, f64)> {
    let ProfileSegmentContext {
        topology: context,
        axis_origin_mm: axis_origin,
        axis_direction: axis,
        ..
    } = segment_context;
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

fn plane_profile_segment_with_angular_trims(
    face_index: usize,
    face: &FaceInfo,
    plane: brep::PlaneSupport,
    segment_context: ProfileSegmentContext<'_, '_>,
) -> Option<(Segment2, f64)> {
    let ProfileSegmentContext {
        topology: context,
        axis_origin_mm: axis_origin,
        axis_direction: axis,
        source_tolerance_mm,
        angular_trim_faces,
    } = segment_context;
    if face.loops.is_empty() || face.loops.len() > 2 || !parallel(plane.normal, axis) {
        return None;
    }
    let t = axial_coordinate(plane.origin_mm, axis_origin, axis);
    if !t.is_finite() {
        return None;
    }

    let mut support_radii = Vec::<f64>::new();
    let mut raw_min_r = f64::INFINITY;
    let mut raw_max_r = f64::NEG_INFINITY;
    let mut raw_axial_residual = 0.0_f64;
    let mut geometry_residual = plane.max_residual_mm;
    let mut source_support_residual = 0.0_f64;
    let mut only_circles = true;
    let mut touches_axis = false;

    for loop_ in &face.loops {
        if loop_.edges.is_empty() {
            return None;
        }
        for edge in &loop_.edges {
            for point in [edge.start_mm, edge.end_mm] {
                let axial_residual = (axial_coordinate(point, axis_origin, axis) - t).abs();
                let radius = point_axis_distance(point, axis_origin, axis);
                if !axial_residual.is_finite() || !radius.is_finite() {
                    return None;
                }
                raw_axial_residual = raw_axial_residual.max(axial_residual);
                source_support_residual = source_support_residual.max(axial_residual);
                raw_min_r = raw_min_r.min(radius);
                raw_max_r = raw_max_r.max(radius);
            }

            if angular_trim_faces.is_some_and(|trim_faces| {
                unique_neighbor_face(face_index, edge.edge_id, context.edge_faces)
                    .is_some_and(|neighbor| trim_faces.contains(&neighbor))
            }) {
                continue;
            }

            match &edge.support {
                CurveSupport::Circle(circle) => {
                    if !circle.radius_mm.is_finite()
                        || circle.radius_mm <= GEOM_TOL_MM
                        || !parallel(circle.normal, axis)
                        || axis_distance(circle.center_mm, axis_origin, axis) > GEOM_TOL_MM
                    {
                        return None;
                    }
                    let circle_t = axial_coordinate(circle.center_mm, axis_origin, axis);
                    source_support_residual = source_support_residual.max((circle_t - t).abs());
                    geometry_residual =
                        geometry_residual.max(axis_distance(circle.center_mm, axis_origin, axis));
                    for point in [edge.start_mm, edge.end_mm] {
                        source_support_residual =
                            source_support_residual.max(circle_point_residual(point, *circle));
                    }
                    push_unique_scalar(&mut support_radii, circle.radius_mm);
                }
                CurveSupport::Line(line) => {
                    only_circles = false;
                    if dot(line.direction, axis).abs() > DIR_TOL {
                        return None;
                    }
                    source_support_residual = source_support_residual
                        .max((axial_coordinate(line.origin_mm, axis_origin, axis) - t).abs());
                    let direction = normalize(line.direction)?;
                    for point in [edge.start_mm, edge.end_mm] {
                        source_support_residual = source_support_residual
                            .max(norm(cross(sub(point, line.origin_mm), direction)));
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
                    geometry_residual = geometry_residual.max(line_axis_distance);
                    if line_axis_distance <= GEOM_TOL_MM {
                        // The finite edge may cross the revolution axis between
                        // two off-axis vertices.  Testing only the endpoints
                        // misses exactly that valid radial-boundary topology.
                        let line_t = axial_coordinate(line.origin_mm, axis_origin, axis);
                        let axis_point = add(axis_origin, mul(axis, line_t));
                        let axis_parameter = dot(sub(axis_point, line.origin_mm), direction);
                        let start_parameter = dot(sub(edge.start_mm, line.origin_mm), direction);
                        let end_parameter = dot(sub(edge.end_mm, line.origin_mm), direction);
                        if between(axis_parameter, start_parameter, end_parameter) {
                            touches_axis = true;
                        }
                    }
                }
                CurveSupport::BSpline(_) => {
                    only_circles = false;
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
                            geometry_residual = geometry_residual.max(axis_distance(
                                cylinder.axis_origin_mm,
                                axis_origin,
                                axis,
                            ));
                            cylinder.radius_mm
                        }
                        SurfaceSupport::Cone(cone) => {
                            let (segment, cone_source_residual) =
                                cone_profile_segment_with_angular_trims(
                                    neighbor,
                                    neighbor_face,
                                    cone,
                                    segment_context,
                                )?;
                            source_support_residual =
                                source_support_residual.max(cone_source_residual);
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
                            geometry_residual = geometry_residual.max(residual);
                            linear_radius_at(support, t)?
                        }
                        _ => return None,
                    };
                    if !expected_radius.is_finite() || expected_radius < -GEOM_TOL_MM {
                        return None;
                    }
                    for point in [edge.start_mm, edge.end_mm] {
                        source_support_residual = source_support_residual.max(
                            (point_axis_distance(point, axis_origin, axis) - expected_radius).abs(),
                        );
                    }
                    push_unique_scalar(&mut support_radii, expected_radius.max(0.0));
                }
                CurveSupport::Other { .. } => return None,
            }
        }
    }

    if geometry_residual > GEOM_TOL_MM || source_support_residual > source_tolerance_mm {
        return None;
    }

    let (mut min_r, max_r) = if support_radii.is_empty() {
        // No analytic radial boundary evidence: preserve the old strict
        // vertex-based behavior rather than broadening acceptance.
        if raw_axial_residual > GEOM_TOL_MM {
            return None;
        }
        (raw_min_r, raw_max_r)
    } else {
        let min_r = support_radii.iter().copied().reduce(f64::min)?;
        let max_r = support_radii.iter().copied().reduce(f64::max)?;
        (min_r, max_r)
    };

    if !min_r.is_finite() || !max_r.is_finite() {
        return None;
    }
    if touches_axis {
        min_r = 0.0;
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
        geometry_residual.max(source_support_residual),
    ))
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
    let outer_tube_distance = ((radial - torus.major_radius_mm).powi(2) + axial.powi(2)).sqrt();
    let inner_tube_distance = ((radial + torus.major_radius_mm).powi(2) + axial.powi(2)).sqrt();
    (outer_tube_distance - torus.minor_radius_mm)
        .abs()
        .min((inner_tube_distance - torus.minor_radius_mm).abs())
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
    normalize3(vector, DIR_TOL)
}

#[cfg(test)]
mod tests;
