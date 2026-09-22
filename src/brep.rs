use crate::instances::{
    cartesian_point, entity_ref, entity_ref_value, number, simple_record, simple_record_mut,
};
use crate::surface_recovery;
use ruststep::ast::{EntityInstance, Parameter, Record, SubSuperRecord};
use std::collections::{HashMap, HashSet};

const DIRECTION_TOLERANCE: f64 = 1.0e-15;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PlaneSupport {
    pub origin_mm: [f64; 3],
    pub normal: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CylinderSupport {
    pub axis_origin_mm: [f64; 3],
    pub axis: [f64; 3],
    pub x_direction: [f64; 3],
    pub radius_mm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BSplineSupport {
    pub degree: usize,
    pub control_points_mm: Vec<[f64; 3]>,
    pub knots: Vec<f64>,
    pub weights: Option<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SplineExtrusionSupport {
    pub profile: BSplineSupport,
    pub extrusion_mm: [f64; 3],
    pub max_residual_mm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SurfaceSupport {
    Plane(PlaneSupport),
    Cylinder(CylinderSupport),
    SplineExtrusion(SplineExtrusionSupport),
    Other { entity_id: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LineSupport {
    pub origin_mm: [f64; 3],
    pub direction: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CircleSupport {
    pub center_mm: [f64; 3],
    pub normal: [f64; 3],
    pub x_direction: [f64; 3],
    pub radius_mm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CurveSupport {
    Line(LineSupport),
    Circle(CircleSupport),
    BSpline(BSplineSupport),
    Other { entity_id: u64 },
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OrientedEdgeUse {
    pub oriented_edge_id: u64,
    pub edge_id: u64,
    pub curve_id: u64,
    pub curve_same_sense: bool,
    pub parameter_forward: bool,
    pub start_vertex: u64,
    pub end_vertex: u64,
    pub start_mm: [f64; 3],
    pub end_mm: [f64; 3],
    pub support: CurveSupport,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FaceLoop {
    pub bound_id: u64,
    pub loop_id: u64,
    pub outer: bool,
    pub orientation: bool,
    pub edges: Vec<OrientedEdgeUse>,
}

pub(crate) fn solid_face_ids(
    solid_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<u64>> {
    let shell_id = manifold_shell(solid_id, entities, index)?;
    ref_list_param(shell_id, 1, entities, index)
}

pub(crate) fn face_loops(
    face_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<FaceLoop>> {
    let record = simple_record(&entities[*index.get(&face_id)?])?;
    if record.name != "ADVANCED_FACE" && record.name != "FACE_SURFACE" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let Parameter::List(bounds) = params.get(1)? else {
        return None;
    };
    bounds
        .iter()
        .map(|bound| ordered_bound_loop(entity_ref_value(bound)?, entities, index))
        .collect()
}

pub(crate) fn ordered_bound_loop(
    bound_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<FaceLoop> {
    let record = simple_record(&entities[*index.get(&bound_id)?])?;
    let outer = match record.name.as_str() {
        "FACE_OUTER_BOUND" => true,
        "FACE_BOUND" => false,
        _ => return None,
    };
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let loop_id = entity_ref_value(params.get(1)?)?;
    let orientation = enumeration_bool(params.get(2)?)?;

    let loop_record = simple_record(&entities[*index.get(&loop_id)?])?;
    if loop_record.name != "EDGE_LOOP" {
        return None;
    }
    let Parameter::List(loop_params) = &loop_record.parameter else {
        return None;
    };
    let Parameter::List(oriented_edges) = loop_params.get(1)? else {
        return None;
    };

    let edges = oriented_edges
        .iter()
        .map(|oriented| oriented_edge_use(entity_ref_value(oriented)?, entities, index))
        .collect::<Option<Vec<_>>>()?;

    Some(FaceLoop {
        bound_id,
        loop_id,
        outer,
        orientation,
        edges,
    })
}

pub(crate) fn oriented_edge_use(
    oriented_edge_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<OrientedEdgeUse> {
    let oriented_record = simple_record(&entities[*index.get(&oriented_edge_id)?])?;
    if oriented_record.name != "ORIENTED_EDGE" {
        return None;
    }
    let Parameter::List(oriented_params) = &oriented_record.parameter else {
        return None;
    };
    let edge_id = entity_ref_value(oriented_params.get(3)?)?;
    let forward = enumeration_bool(oriented_params.get(4)?)?;

    let edge_record = simple_record(&entities[*index.get(&edge_id)?])?;
    if edge_record.name != "EDGE_CURVE" {
        return None;
    }
    let Parameter::List(edge_params) = &edge_record.parameter else {
        return None;
    };
    let raw_start = entity_ref_value(edge_params.get(1)?)?;
    let raw_end = entity_ref_value(edge_params.get(2)?)?;
    let curve_id = entity_ref_value(edge_params.get(3)?)?;
    let curve_same_sense = enumeration_bool(edge_params.get(4)?)?;
    let raw_start_mm = vertex_point(raw_start, entities, index)?;
    let raw_end_mm = vertex_point(raw_end, entities, index)?;

    let (start_vertex, end_vertex, start_mm, end_mm) = if forward {
        (raw_start, raw_end, raw_start_mm, raw_end_mm)
    } else {
        (raw_end, raw_start, raw_end_mm, raw_start_mm)
    };

    Some(OrientedEdgeUse {
        oriented_edge_id,
        edge_id,
        curve_id,
        curve_same_sense,
        parameter_forward: if forward {
            curve_same_sense
        } else {
            !curve_same_sense
        },
        start_vertex,
        end_vertex,
        start_mm,
        end_mm,
        support: curve_support(curve_id, entities, index),
    })
}

pub(crate) fn surface_support(
    surface_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> SurfaceSupport {
    if let Some(record) = index
        .get(&surface_id)
        .and_then(|&idx| simple_record(&entities[idx]))
        && let Parameter::List(params) = &record.parameter
    {
        match record.name.as_str() {
            "PLANE" => {
                let support = (|| {
                    let placement = entity_ref_value(params.get(1)?)?;
                    let (origin_mm, normal, _) = axis2_placement_3d(placement, entities, index)?;
                    Some(PlaneSupport { origin_mm, normal })
                })();
                if let Some(support) = support {
                    return SurfaceSupport::Plane(support);
                }
            }
            "CYLINDRICAL_SURFACE" => {
                let support = (|| {
                    let placement = entity_ref_value(params.get(1)?)?;
                    let (axis_origin_mm, axis, x_direction) =
                        axis2_placement_3d(placement, entities, index)?;
                    Some(CylinderSupport {
                        axis_origin_mm,
                        axis,
                        x_direction,
                        radius_mm: number(params.get(2)?)?,
                    })
                })();
                if let Some(support) = support {
                    return SurfaceSupport::Cylinder(support);
                }
            }
            _ => {}
        }
    }

    surface_recovery::analyze_v_extrusion_surface(surface_id, entities, index, 1.0e-7)
        .map(|evidence| {
            SurfaceSupport::SplineExtrusion(SplineExtrusionSupport {
                profile: BSplineSupport {
                    degree: evidence.degree,
                    control_points_mm: evidence.control_points_mm,
                    knots: evidence.knots,
                    weights: evidence.weights.and_then(normalize_weights),
                },
                extrusion_mm: evidence.extrusion_mm,
                max_residual_mm: evidence.max_residual_mm,
            })
        })
        .unwrap_or(SurfaceSupport::Other {
            entity_id: surface_id,
        })
}

pub(crate) fn curve_support(
    curve_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> CurveSupport {
    curve_support_inner(curve_id, entities, index, 0)
}

fn curve_support_inner(
    curve_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    wrapper_depth: usize,
) -> CurveSupport {
    if wrapper_depth >= 8 {
        return CurveSupport::Other {
            entity_id: curve_id,
        };
    }
    let Some(&entity_index) = index.get(&curve_id) else {
        return CurveSupport::Other {
            entity_id: curve_id,
        };
    };
    let entity = &entities[entity_index];

    if let Some(record) = simple_record(entity)
        && matches!(record.name.as_str(), "SURFACE_CURVE" | "SEAM_CURVE")
        && let Parameter::List(params) = &record.parameter
        && let Some(inner_curve) = params.get(1).and_then(entity_ref_value)
        && inner_curve != curve_id
    {
        return curve_support_inner(inner_curve, entities, index, wrapper_depth + 1);
    }

    if let Some(record) = simple_record(entity)
        && let Parameter::List(params) = &record.parameter
    {
        match record.name.as_str() {
            "LINE" => {
                let support = (|| {
                    let origin_mm =
                        cartesian_point(entity_ref_value(params.get(1)?)?, entities, index)?;
                    let vector_id = entity_ref_value(params.get(2)?)?;
                    let vector = simple_record(&entities[*index.get(&vector_id)?])?;
                    if vector.name != "VECTOR" {
                        return None;
                    }
                    let Parameter::List(vector_params) = &vector.parameter else {
                        return None;
                    };
                    let direction = direction_components(
                        entity_ref_value(vector_params.get(1)?)?,
                        entities,
                        index,
                    )?;
                    Some(LineSupport {
                        origin_mm,
                        direction,
                    })
                })();
                return support
                    .map(CurveSupport::Line)
                    .unwrap_or(CurveSupport::Other {
                        entity_id: curve_id,
                    });
            }
            "CIRCLE" => {
                let support = (|| {
                    let (center_mm, normal, x_direction) =
                        axis2_placement_3d(entity_ref_value(params.get(1)?)?, entities, index)?;
                    Some(CircleSupport {
                        center_mm,
                        normal,
                        x_direction,
                        radius_mm: number(params.get(2)?)?,
                    })
                })();
                return support
                    .map(CurveSupport::Circle)
                    .unwrap_or(CurveSupport::Other {
                        entity_id: curve_id,
                    });
            }
            "BEZIER_CURVE" | "B_SPLINE_CURVE_WITH_KNOTS" => {
                if let Some(spline) = parse_simple_bspline(record, entities, index) {
                    return CurveSupport::BSpline(spline);
                }
            }
            _ => {}
        }
    } else if let EntityInstance::Complex { subsuper, .. } = entity
        && let Some(spline) = parse_complex_rational_bspline(subsuper, entities, index)
    {
        return CurveSupport::BSpline(spline);
    }

    CurveSupport::Other {
        entity_id: curve_id,
    }
}

fn parse_simple_bspline(
    record: &Record,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<BSplineSupport> {
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    match record.name.as_str() {
        "BEZIER_CURVE" => {
            if params.len() != 6 {
                return None;
            }
            let degree = positive_degree(params.get(1)?)?;
            let order = degree.checked_add(1)?;
            let control_points_mm = control_points_3d(params.get(2)?, entities, index)?;
            if control_points_mm.len() != order
                || !not_true_enum(params.get(4)?)
                || !not_true_enum(params.get(5)?)
            {
                return None;
            }
            let mut knots = vec![0.0; order];
            knots.extend(std::iter::repeat_n(1.0, order));
            Some(BSplineSupport {
                degree,
                control_points_mm,
                knots,
                weights: None,
            })
        }
        "B_SPLINE_CURVE_WITH_KNOTS" => {
            if params.len() != 9 {
                return None;
            }
            let degree = positive_degree(params.get(1)?)?;
            let control_points_mm = control_points_3d(params.get(2)?, entities, index)?;
            if !not_true_enum(params.get(4)?) || !not_true_enum(params.get(5)?) {
                return None;
            }
            let multiplicities = positive_integer_list(params.get(6)?)?;
            let values = finite_numeric_list(params.get(7)?)?;
            let knots = normalized_expanded_knots(
                degree,
                control_points_mm.len(),
                &multiplicities,
                &values,
            )?;
            Some(BSplineSupport {
                degree,
                control_points_mm,
                knots,
                weights: None,
            })
        }
        _ => None,
    }
}

fn parse_complex_rational_bspline(
    subsuper: &SubSuperRecord,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<BSplineSupport> {
    const REQUIRED: &[&str] = &[
        "BOUNDED_CURVE",
        "B_SPLINE_CURVE",
        "B_SPLINE_CURVE_WITH_KNOTS",
        "CURVE",
        "GEOMETRIC_REPRESENTATION_ITEM",
        "RATIONAL_B_SPLINE_CURVE",
        "REPRESENTATION_ITEM",
    ];
    if subsuper.0.len() != REQUIRED.len()
        || REQUIRED.iter().any(|name| {
            subsuper
                .0
                .iter()
                .filter(|record| record.name == *name)
                .count()
                != 1
        })
    {
        return None;
    }

    let bspline = record_by_name(subsuper, "B_SPLINE_CURVE")?;
    let Parameter::List(base) = &bspline.parameter else {
        return None;
    };
    if base.len() != 5 {
        return None;
    }
    let degree = positive_degree(base.first()?)?;
    let control_points_mm = control_points_3d(base.get(1)?, entities, index)?;
    if !not_true_enum(base.get(3)?) || !not_true_enum(base.get(4)?) {
        return None;
    }

    let knot_record = record_by_name(subsuper, "B_SPLINE_CURVE_WITH_KNOTS")?;
    let Parameter::List(knot_params) = &knot_record.parameter else {
        return None;
    };
    if knot_params.len() != 3 {
        return None;
    }
    let multiplicities = positive_integer_list(knot_params.first()?)?;
    let values = finite_numeric_list(knot_params.get(1)?)?;
    let knots =
        normalized_expanded_knots(degree, control_points_mm.len(), &multiplicities, &values)?;

    let rational = record_by_name(subsuper, "RATIONAL_B_SPLINE_CURVE")?;
    let Parameter::List(rational_params) = &rational.parameter else {
        return None;
    };
    let [Parameter::List(weight_params)] = rational_params.as_slice() else {
        return None;
    };
    let raw_weights = weight_params
        .iter()
        .map(|parameter| number(parameter).filter(|weight| weight.is_finite() && *weight > 0.0))
        .collect::<Option<Vec<_>>>()?;
    if raw_weights.len() != control_points_mm.len() {
        return None;
    }

    Some(BSplineSupport {
        degree,
        control_points_mm,
        knots,
        weights: Some(normalize_weights(raw_weights)?),
    })
}

fn record_by_name<'a>(subsuper: &'a SubSuperRecord, name: &str) -> Option<&'a Record> {
    let mut matches = subsuper.0.iter().filter(|record| record.name == name);
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

fn positive_degree(parameter: &Parameter) -> Option<usize> {
    match parameter {
        Parameter::Integer(value) if *value >= 1 => usize::try_from(*value).ok(),
        _ => None,
    }
}

fn not_true_enum(parameter: &Parameter) -> bool {
    matches!(parameter, Parameter::Enumeration(value) if value == "F" || value == "U")
}

fn control_points_3d(
    parameter: &Parameter,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<[f64; 3]>> {
    let Parameter::List(items) = parameter else {
        return None;
    };
    let points = items
        .iter()
        .map(|item| cartesian_point(entity_ref_value(item)?, entities, index))
        .collect::<Option<Vec<_>>>()?;
    (!points.is_empty()
        && points
            .iter()
            .flatten()
            .all(|coordinate| coordinate.is_finite()))
    .then_some(points)
}

fn positive_integer_list(parameter: &Parameter) -> Option<Vec<usize>> {
    let Parameter::List(items) = parameter else {
        return None;
    };
    items
        .iter()
        .map(|item| match item {
            Parameter::Integer(value) if *value >= 1 => usize::try_from(*value).ok(),
            _ => None,
        })
        .collect()
}

fn finite_numeric_list(parameter: &Parameter) -> Option<Vec<f64>> {
    let Parameter::List(items) = parameter else {
        return None;
    };
    items
        .iter()
        .map(|item| number(item).filter(|value| value.is_finite()))
        .collect()
}

fn normalized_expanded_knots(
    degree: usize,
    control_points: usize,
    multiplicities: &[usize],
    values: &[f64],
) -> Option<Vec<f64>> {
    let order = degree.checked_add(1)?;
    let expected_knots = control_points.checked_add(order)?;
    let total_multiplicity = multiplicities
        .iter()
        .try_fold(0usize, |sum, &value| sum.checked_add(value))?;
    if control_points < order
        || multiplicities.len() != values.len()
        || multiplicities.len() < 2
        || multiplicities.first().copied()? != order
        || multiplicities.last().copied()? != order
        || total_multiplicity != expected_knots
        || values.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return None;
    }
    let start = *values.first()?;
    let end = *values.last()?;
    let span = end - start;
    if !span.is_finite() || span <= 0.0 {
        return None;
    }
    let mut knots = Vec::with_capacity(expected_knots);
    for (&value, &multiplicity) in values.iter().zip(multiplicities) {
        let normalized = (value - start) / span;
        knots.extend(std::iter::repeat_n(normalized, multiplicity));
    }
    Some(knots)
}

fn normalize_weights(weights: Vec<f64>) -> Option<Vec<f64>> {
    let scale = *weights.first()?;
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let normalized = weights
        .into_iter()
        .map(|weight| weight / scale)
        .collect::<Vec<_>>();
    normalized
        .iter()
        .all(|weight| weight.is_finite() && *weight > 0.0)
        .then_some(normalized)
}

pub(crate) fn axis2_placement_3d(
    placement_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<([f64; 3], [f64; 3], [f64; 3])> {
    let placement = simple_record(&entities[*index.get(&placement_id)?])?;
    if placement.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let Parameter::List(params) = &placement.parameter else {
        return None;
    };
    let origin_mm = cartesian_point(entity_ref_value(params.get(1)?)?, entities, index)?;
    let axis = match params.get(2)? {
        Parameter::Ref(_) => {
            direction_components(entity_ref_value(params.get(2)?)?, entities, index)?
        }
        Parameter::Omitted => [0.0, 0.0, 1.0],
        _ => return None,
    };
    let axis = normalize(axis)?;
    let raw_x = match params.get(3)? {
        Parameter::Ref(_) => {
            direction_components(entity_ref_value(params.get(3)?)?, entities, index)?
        }
        Parameter::Omitted => default_ref_direction(axis),
        _ => return None,
    };
    let x_direction = normalize(sub(raw_x, mul(axis, dot(raw_x, axis))))?;
    Some((origin_mm, axis, x_direction))
}

pub(crate) fn direction_components(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let record = simple_record(&entities[*index.get(&id)?])?;
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
    normalize([
        number(&values[0])?,
        number(&values[1])?,
        number(&values[2])?,
    ])
}

pub(crate) fn vertex_point(
    vertex_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let vertex = simple_record(&entities[*index.get(&vertex_id)?])?;
    if vertex.name != "VERTEX_POINT" {
        return None;
    }
    let Parameter::List(params) = &vertex.parameter else {
        return None;
    };
    cartesian_point(entity_ref_value(params.get(1)?)?, entities, index)
}

pub(crate) fn enumeration_bool(parameter: &Parameter) -> Option<bool> {
    match parameter {
        Parameter::Enumeration(value) if value == "T" => Some(true),
        Parameter::Enumeration(value) if value == "F" => Some(false),
        _ => None,
    }
}

fn default_ref_direction(axis: [f64; 3]) -> [f64; 3] {
    if axis[0].abs() <= axis[1].abs() && axis[0].abs() <= axis[2].abs() {
        [1.0, 0.0, 0.0]
    } else if axis[1].abs() <= axis[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    let length = dot(vector, vector).sqrt();
    if !length.is_finite() || length <= DIRECTION_TOLERANCE {
        return None;
    }
    Some(mul(vector, 1.0 / length))
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

pub(crate) fn manifold_shell(
    solid_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<u64> {
    let record = simple_record(&entities[*index.get(&solid_id)?])?;
    if record.name != "MANIFOLD_SOLID_BREP" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    entity_ref_value(params.get(1)?)
}

pub(crate) fn face_surface(
    face_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<u64> {
    let record = simple_record(&entities[*index.get(&face_id)?])?;
    if record.name != "ADVANCED_FACE" && record.name != "FACE_SURFACE" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    entity_ref_value(params.get(2)?)
}

pub(crate) fn face_sense(
    face_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<String> {
    let record = simple_record(&entities[*index.get(&face_id)?])?;
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    enumeration_value(params.get(3)?)
}

pub(crate) fn face_edge_curves(
    face_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<HashSet<u64>> {
    let record = simple_record(&entities[*index.get(&face_id)?])?;
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let Parameter::List(bounds) = params.get(1)? else {
        return None;
    };

    let mut edges = HashSet::new();
    for bound in bounds {
        let bound_id = entity_ref_value(bound)?;
        let (_, bound_edges, _) = bound_loop_edges(bound_id, entities, index)?;
        edges.extend(bound_edges);
    }
    Some(edges)
}

pub(crate) fn bound_loop_edges(
    bound_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<(u64, HashSet<u64>, String)> {
    let loop_ = ordered_bound_loop(bound_id, entities, index)?;
    Some((
        loop_.loop_id,
        loop_.edges.iter().map(|edge| edge.edge_id).collect(),
        if loop_.orientation { "T" } else { "F" }.to_string(),
    ))
}

pub(crate) fn matching_plane_bound(
    plane_face: u64,
    interface_edges: &HashSet<u64>,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<(u64, u64, String)> {
    let bounds = ref_list_param(plane_face, 1, entities, index)?;
    let mut matched = None;
    for bound in bounds {
        let record = simple_record(&entities[*index.get(&bound)?])?;
        if record.name != "FACE_BOUND" {
            continue;
        }
        let Some((loop_id, edges, orientation)) = bound_loop_edges(bound, entities, index) else {
            continue;
        };
        if edges == *interface_edges {
            if matched.is_some() {
                return None;
            }
            matched = Some((bound, loop_id, orientation));
        }
    }
    matched
}

pub(crate) fn ref_list_param(
    id: u64,
    param_index: usize,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<u64>> {
    let record = simple_record(&entities[*index.get(&id)?])?;
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let Parameter::List(items) = params.get(param_index)? else {
        return None;
    };
    items.iter().map(entity_ref_value).collect()
}

pub(crate) fn remove_refs_from_list_param(
    entity: &mut EntityInstance,
    param_index: usize,
    remove: &HashSet<u64>,
) -> bool {
    let Some(record) = simple_record_mut(entity) else {
        return false;
    };
    let Parameter::List(params) = &mut record.parameter else {
        return false;
    };
    let Some(Parameter::List(items)) = params.get_mut(param_index) else {
        return false;
    };
    let before = items.len();
    items.retain(|item| entity_ref_value(item).is_none_or(|id| !remove.contains(&id)));
    items.len() < before
}

pub(crate) fn append_refs_to_list_param(
    entity: &mut EntityInstance,
    param_index: usize,
    append: &[u64],
) -> bool {
    let Some(record) = simple_record_mut(entity) else {
        return false;
    };
    let Parameter::List(params) = &mut record.parameter else {
        return false;
    };
    let Some(Parameter::List(items)) = params.get_mut(param_index) else {
        return false;
    };
    items.extend(append.iter().copied().map(entity_ref));
    true
}

pub(crate) fn enumeration_value(parameter: &Parameter) -> Option<String> {
    match parameter {
        Parameter::Enumeration(value) => Some(value.clone()),
        _ => None,
    }
}

pub(crate) fn toggle_tf(value: &str) -> String {
    match value {
        "T" => "F".to_string(),
        "F" => "T".to_string(),
        _ => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggles_step_boolean_enumerations() {
        assert_eq!(toggle_tf("T"), "F");
        assert_eq!(toggle_tf("F"), "T");
    }
}
