use crate::math2::rotate_quarter_xy as rotate_xy;
use crate::step_entities::{cartesian_point, closure_from, nth_entity_ref, number};
use crate::step_graph::{entity_ref_value, simple_record};
use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ShapeKey {
    pub(crate) vertices: usize,
    pub(crate) edges: usize,
    pub(crate) oriented_edges: usize,
    pub(crate) faces: usize,
    pub(crate) points: Vec<[i64; 3]>,
    pub(crate) edge_geometry: Vec<(String, Vec<i64>)>,
    pub(crate) face_geometry: Vec<(String, Vec<i64>)>,
    pub(crate) topology: String,
}

#[derive(Debug, Clone)]
pub(crate) struct SolidIdentity {
    pub(crate) root: u64,
    pub(crate) closure: HashSet<u64>,
    pub(crate) center: [f64; 3],
    pub(crate) vertex_points: Vec<[f64; 3]>,
    pub(crate) face_ids: Vec<u64>,
    pub(crate) key: ShapeKey,
    pub(crate) canonical_quarter: u8,
}

/// Build the canonical geometry/topology identity once, independently of any
/// rewrite policy such as presentation-style preservation.
pub(crate) fn solid_identity(
    root: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<SolidIdentity> {
    let closure = semantic_solid_closure(root, entities, index)?;
    let face_ids = manifold_solid_face_ids(root, entities, index)?;
    let mut vertex_points = Vec::new();
    let mut edge_count = 0usize;
    let mut oriented_edge_count = 0usize;
    let mut face_geometry_ids = Vec::with_capacity(face_ids.len());
    let mut edge_geometry_ids = Vec::new();

    for &id in &closure {
        let &idx = index.get(&id)?;
        let Some(record) = simple_record(&entities[idx]) else {
            continue;
        };
        match record.name.as_str() {
            "ADVANCED_FACE" => {
                face_geometry_ids.push(nth_entity_ref(&record.parameter, 2)?);
            }
            "EDGE_CURVE" => {
                edge_count += 1;
                edge_geometry_ids.push(nth_entity_ref(&record.parameter, 3)?);
            }
            "EDGE_LOOP" => {
                let Parameter::List(params) = &record.parameter else {
                    return None;
                };
                let Parameter::List(edge_uses) = params.get(1)? else {
                    return None;
                };
                oriented_edge_count += edge_uses.len();
            }
            "VERTEX_POINT" => {
                let point = nth_entity_ref(&record.parameter, 1)?;
                vertex_points.push(cartesian_point(point, entities, index)?);
            }
            _ => {}
        }
    }

    if face_ids.is_empty() || vertex_points.is_empty() || edge_count == 0 {
        return None;
    }

    let center = centroid(&vertex_points);
    let (points, topology, canonical_quarter) =
        canonical_z90_solid_signature(root, &vertex_points, entities, index, center)?;
    let mut face_geometry = resolved_geometry_signatures(
        &face_geometry_ids,
        entities,
        index,
        center,
        canonical_quarter,
    )?;
    let mut edge_geometry = resolved_geometry_signatures(
        &edge_geometry_ids,
        entities,
        index,
        center,
        canonical_quarter,
    )?;
    face_geometry.sort();
    edge_geometry.sort();

    let key = ShapeKey {
        vertices: vertex_points.len(),
        edges: edge_count,
        oriented_edges: oriented_edge_count,
        faces: face_ids.len(),
        points,
        edge_geometry,
        face_geometry,
        topology,
    };
    Some(SolidIdentity {
        root,
        closure,
        center,
        vertex_points,
        face_ids,
        key,
        canonical_quarter,
    })
}

/// Translation- and Z-quarter-turn-invariant identity for one manifold solid.
pub fn solid_shape_key(
    root: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<(ShapeKey, [f64; 3], u8, usize)> {
    let identity = solid_identity(root, entities, index)?;
    let closure_len = identity.closure.len();
    Some((
        identity.key,
        identity.center,
        identity.canonical_quarter,
        closure_len,
    ))
}

pub fn semantic_solid_closure(
    root: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<HashSet<u64>> {
    let root_record = simple_record(&entities[*index.get(&root)?])?;
    if root_record.name != "MANIFOLD_SOLID_BREP" {
        return None;
    }
    let shell_id = nth_entity_ref(&root_record.parameter, 1)?;
    let face_ids = manifold_solid_face_ids(root, entities, index)?;

    let mut closure = HashSet::new();
    closure.insert(root);
    closure.insert(shell_id);
    for face_id in face_ids {
        closure.extend(closure_from(face_id, entities, index));
    }
    Some(closure)
}

pub fn manifold_solid_face_ids(
    root: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<u64>> {
    let root_record = simple_record(&entities[*index.get(&root)?])?;
    if root_record.name != "MANIFOLD_SOLID_BREP" {
        return None;
    }
    let shell_id = nth_entity_ref(&root_record.parameter, 1)?;
    let shell_record = simple_record(&entities[*index.get(&shell_id)?])?;
    if shell_record.name != "CLOSED_SHELL" {
        return None;
    }
    let Parameter::List(shell_params) = &shell_record.parameter else {
        return None;
    };
    let Parameter::List(member_refs) = shell_params.get(1)? else {
        return None;
    };

    let mut faces = Vec::new();
    for member_ref in member_refs {
        let id = entity_ref_value(member_ref)?;
        let record = simple_record(&entities[*index.get(&id)?])?;
        if matches!(record.name.as_str(), "ADVANCED_FACE" | "FACE_SURFACE") {
            faces.push(id);
        }
    }
    (!faces.is_empty()).then_some(faces)
}

pub(crate) fn solid_topology_signature(
    root: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let face_ids = manifold_solid_face_ids(root, entities, index)?;

    let mut faces = Vec::with_capacity(face_ids.len());
    for face_id in face_ids {
        faces.push(face_topology_signature(
            face_id, entities, index, center, quarter,
        )?);
    }
    faces.sort();
    Some(format!("SHELL[{}]", faces.join("|")))
}

pub fn face_topology_signature(
    face_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let record = simple_record(&entities[*index.get(&face_id)?])?;
    if record.name != "ADVANCED_FACE" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let Parameter::List(bound_refs) = params.get(1)? else {
        return None;
    };
    let surface_id = entity_ref_value(params.get(2)?)?;
    let same_sense = parameter_literal_signature(params.get(3)?)?;

    let mut bounds = Vec::with_capacity(bound_refs.len());
    for bound_ref in bound_refs {
        bounds.push(bound_topology_signature(
            entity_ref_value(bound_ref)?,
            entities,
            index,
            center,
            quarter,
        )?);
    }
    bounds.sort();

    let surface = support_entity_signature(
        surface_id,
        entities,
        index,
        center,
        quarter,
        &mut HashSet::new(),
        0,
    )?;

    Some(format!("FACE({same_sense};{surface};{})", bounds.join("&")))
}

fn bound_topology_signature(
    bound_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let record = simple_record(&entities[*index.get(&bound_id)?])?;
    if record.name != "FACE_OUTER_BOUND" && record.name != "FACE_BOUND" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let loop_id = entity_ref_value(params.get(1)?)?;
    let orientation = parameter_literal_signature(params.get(2)?)?;
    let loop_sig = edge_loop_signature(loop_id, entities, index, center, quarter)?;
    Some(format!("{}({orientation};{loop_sig})", record.name))
}

fn edge_loop_signature(
    loop_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let record = simple_record(&entities[*index.get(&loop_id)?])?;
    if record.name != "EDGE_LOOP" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let Parameter::List(edge_refs) = params.get(1)? else {
        return None;
    };

    let mut uses = Vec::with_capacity(edge_refs.len());
    for edge_ref in edge_refs {
        uses.push(oriented_edge_signature(
            entity_ref_value(edge_ref)?,
            entities,
            index,
            center,
            quarter,
        )?);
    }
    Some(format!("LOOP[{}]", canonical_cycle(&uses).join(">")))
}

/// Resolve a possibly nested `ORIENTED_EDGE` chain to its base `EDGE_CURVE` and
/// effective traversal direction. A direct `EDGE_CURVE` in `EDGE_LOOP` is accepted
/// as the de-facto shorthand for an orientation=.T. wrapper used by real
/// exporters and tolerated by `OpenCascade`.
pub fn resolve_edge_curve_use(
    use_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<(u64, bool)> {
    let mut current = use_id;
    let mut forward = true;
    let mut seen = HashSet::new();

    for _ in 0..32 {
        if !seen.insert(current) {
            return None;
        }
        let record = simple_record(&entities[*index.get(&current)?])?;
        match record.name.as_str() {
            "EDGE_CURVE" => return Some((current, forward)),
            "ORIENTED_EDGE" => {
                let Parameter::List(params) = &record.parameter else {
                    return None;
                };
                current = entity_ref_value(params.get(3)?)?;
                let orientation = match params.get(4)? {
                    Parameter::Enumeration(value) if value == "T" => true,
                    Parameter::Enumeration(value) if value == "F" => false,
                    _ => return None,
                };
                if !orientation {
                    forward = !forward;
                }
            }
            _ => return None,
        }
    }
    None
}

pub(crate) fn oriented_edge_signature(
    oriented_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let (edge_id, forward) = resolve_edge_curve_use(oriented_id, entities, index)?;
    let orientation = if forward { ".T." } else { ".F." };

    let edge_record = simple_record(&entities[*index.get(&edge_id)?])?;
    if edge_record.name != "EDGE_CURVE" {
        return None;
    }
    let Parameter::List(edge_params) = &edge_record.parameter else {
        return None;
    };
    let start_id = entity_ref_value(edge_params.get(1)?)?;
    let end_id = entity_ref_value(edge_params.get(2)?)?;
    let curve_id = entity_ref_value(edge_params.get(3)?)?;
    let same_sense = parameter_literal_signature(edge_params.get(4)?)?;

    let mut start = vertex_signature(start_id, entities, index, center, quarter)?;
    let mut end = vertex_signature(end_id, entities, index, center, quarter)?;
    if orientation == ".F." {
        std::mem::swap(&mut start, &mut end);
    }

    let curve = support_entity_signature(
        curve_id,
        entities,
        index,
        center,
        quarter,
        &mut HashSet::new(),
        0,
    )?;

    // ORIENTED_EDGE.orientation and EDGE_CURVE.same_sense are two
    // serialization choices describing one semantic relation: whether this
    // edge use traverses the underlying curve in its parameter direction.
    // Equivalent exporters may flip both while swapping the stored edge
    // endpoints. Signature the combined meaning, not the two raw flags.
    let curve_forward = orientation == same_sense;
    Some(format!(
        "OE(CF{};{start}->{end};{curve})",
        if curve_forward { "T" } else { "F" }
    ))
}

fn vertex_signature(
    vertex_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let record = simple_record(&entities[*index.get(&vertex_id)?])?;
    if record.name != "VERTEX_POINT" {
        return None;
    }
    let point_id = nth_entity_ref(&record.parameter, 1)?;
    let point = cartesian_point(point_id, entities, index)?;
    let q = transform_point(point, center, quarter);
    Some(format!("P({},{},{})", q[0], q[1], q[2]))
}

pub(crate) fn support_entity_signature(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
    visiting: &mut HashSet<u64>,
    depth: usize,
) -> Option<String> {
    if depth > 48 || !visiting.insert(id) {
        return None;
    }
    let entity = &entities[*index.get(&id)?];

    let result = match entity {
        EntityInstance::Simple { record, .. } => match record.name.as_str() {
            "CARTESIAN_POINT" => {
                let p = cartesian_point(id, entities, index)?;
                let q = transform_point(p, center, quarter);
                Some(format!("POINT({},{},{})", q[0], q[1], q[2]))
            }
            "DIRECTION" => {
                let d = direction_components(record)?;
                let q = transform_direction(d, quarter);
                Some(format!("DIR({},{},{})", q[0], q[1], q[2]))
            }
            "LINE" => line_support_signature(record, entities, index, center, quarter),
            "CIRCLE" => circle_support_signature(record, entities, index, center, quarter),
            "PLANE" => plane_support_signature(record, entities, index, center, quarter),
            "CYLINDRICAL_SURFACE" => {
                cylindrical_surface_signature(record, entities, index, center, quarter)
            }
            "CURVE_REPLICA" => curve_replica_support_signature(
                record,
                entities,
                index,
                center,
                quarter,
                visiting,
                depth + 1,
            ),
            _ if is_topology_type(&record.name) => None,
            _ => support_record_signature(
                record,
                entities,
                index,
                center,
                quarter,
                visiting,
                depth + 1,
            ),
        },
        EntityInstance::Complex { subsuper, .. } => {
            let mut parts = Vec::with_capacity(subsuper.0.len());
            for record in &subsuper.0 {
                if is_topology_type(&record.name) {
                    return None;
                }
                parts.push(support_record_signature(
                    record,
                    entities,
                    index,
                    center,
                    quarter,
                    visiting,
                    depth + 1,
                )?);
            }
            Some(format!("COMPLEX[{}]", parts.join("|")))
        }
    };
    visiting.remove(&id);
    result
}

fn curve_replica_support_signature(
    record: &Record,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
    visiting: &mut HashSet<u64>,
    depth: usize,
) -> Option<String> {
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    if params.len() != 3 {
        return None;
    }
    let parent = entity_ref_value(params.get(1)?)?;
    let transform_id = entity_ref_value(params.get(2)?)?;
    let transform = simple_record(&entities[*index.get(&transform_id)?])?;
    if transform.name != "CARTESIAN_TRANSFORMATION_OPERATOR_3D" {
        return None;
    }
    let Parameter::List(transform_params) = &transform.parameter else {
        return None;
    };
    if transform_params.len() != 8
        || [3usize, 4, 6, 7].iter().any(|&slot| {
            !matches!(
                transform_params.get(slot),
                Some(Parameter::NotProvided | Parameter::Omitted)
            )
        })
    {
        return None;
    }
    let origin = cartesian_point(entity_ref_value(transform_params.get(5)?)?, entities, index)?;
    if origin.iter().any(|value| !value.is_finite()) {
        return None;
    }

    // CURVE_REPLICA applies the transformation local_origin as the parent ->
    // replica translation.  Evaluating the parent relative to (center - delta)
    // is exactly equivalent to translating every parent point by delta first,
    // while leaving all direction vectors unchanged.
    let shifted_center = [
        center[0] - origin[0],
        center[1] - origin[1],
        center[2] - origin[2],
    ];
    support_entity_signature(
        parent,
        entities,
        index,
        shifted_center,
        quarter,
        visiting,
        depth,
    )
}

fn line_support_signature(
    record: &Record,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let point_id = entity_ref_value(params.get(1)?)?;
    let vector_id = entity_ref_value(params.get(2)?)?;
    let point = cartesian_point(point_id, entities, index)?;

    let vector = simple_record(&entities[*index.get(&vector_id)?])?;
    if vector.name != "VECTOR" {
        return None;
    }
    let Parameter::List(vector_params) = &vector.parameter else {
        return None;
    };
    let direction_id = entity_ref_value(vector_params.get(1)?)?;
    let magnitude = number(vector_params.get(2)?)?;
    let direction_record = simple_record(&entities[*index.get(&direction_id)?])?;
    let direction = direction_components(direction_record)?;

    let offset = canonical_axis_offset(point, direction, center, quarter)?;
    let q_dir = transform_direction(direction, quarter);
    Some(format!(
        "LINE_LOCUS(P({},{},{});DIR({},{},{});MAG{})",
        offset[0],
        offset[1],
        offset[2],
        q_dir[0],
        q_dir[1],
        q_dir[2],
        (magnitude * 1.0e9).round() as i64,
    ))
}

pub(crate) fn circle_support_signature(
    record: &Record,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let placement_id = entity_ref_value(params.get(1)?)?;
    let radius = number(params.get(2)?)?;
    let placement = simple_record(&entities[*index.get(&placement_id)?])?;
    if placement.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let Parameter::List(place_params) = &placement.parameter else {
        return None;
    };
    let point = cartesian_point(entity_ref_value(place_params.get(1)?)?, entities, index)?;
    let axis_id = entity_ref_value(place_params.get(2)?)?;
    let axis_record = simple_record(&entities[*index.get(&axis_id)?])?;
    let axis = direction_components(axis_record)?;
    let q_center = transform_point(point, center, quarter);
    let q_axis = transform_direction(axis, quarter);
    Some(format!(
        "CIRCLE_LOCUS(C({},{},{});AXIS({},{},{});R{})",
        q_center[0],
        q_center[1],
        q_center[2],
        q_axis[0],
        q_axis[1],
        q_axis[2],
        (radius * 1.0e9).round() as i64,
    ))
}

pub(crate) fn cylindrical_surface_signature(
    record: &Record,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let placement_id = entity_ref_value(params.get(1)?)?;
    let radius = number(params.get(2)?)?;
    let placement = simple_record(&entities[*index.get(&placement_id)?])?;
    if placement.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let Parameter::List(place_params) = &placement.parameter else {
        return None;
    };
    let point_id = entity_ref_value(place_params.get(1)?)?;
    let axis_id = entity_ref_value(place_params.get(2)?)?;
    let point = cartesian_point(point_id, entities, index)?;
    let axis_record = simple_record(&entities[*index.get(&axis_id)?])?;
    let axis = direction_components(axis_record)?;

    // Sliding the placement origin along the cylinder axis changes only the
    // parameter-space V origin. Rotating AXIS2_PLACEMENT_3D.ref_direction
    // around that axis only changes the U=0 seam. Neither changes the 3-D
    // cylindrical locus, so compare axis line + oriented axis + radius only.
    let offset = canonical_axis_offset(point, axis, center, quarter)?;
    let q_axis = transform_direction(axis, quarter);
    Some(format!(
        "CYLINDER_LOCUS(P({},{},{});AXIS({},{},{});R{})",
        offset[0],
        offset[1],
        offset[2],
        q_axis[0],
        q_axis[1],
        q_axis[2],
        (radius * 1.0e9).round() as i64,
    ))
}

pub(crate) fn canonical_axis_offset(
    point: [f64; 3],
    direction: [f64; 3],
    center: [f64; 3],
    quarter: u8,
) -> Option<[i64; 3]> {
    let norm2 = direction[2].mul_add(
        direction[2],
        direction[1].mul_add(direction[1], direction[0] * direction[0]),
    );
    if !norm2.is_finite() || norm2 <= 1.0e-24 {
        return None;
    }
    let rel = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2],
    ];
    let along = rel[2].mul_add(
        direction[2],
        rel[1].mul_add(direction[1], rel[0] * direction[0]),
    ) / norm2;
    let perpendicular = [
        along.mul_add(-direction[0], rel[0]),
        along.mul_add(-direction[1], rel[1]),
        along.mul_add(-direction[2], rel[2]),
    ];
    let (x, y) = rotate_xy(perpendicular[0], perpendicular[1], quarter);
    Some([
        (x * 1.0e5).round() as i64,
        (y * 1.0e5).round() as i64,
        (perpendicular[2] * 1.0e5).round() as i64,
    ])
}

fn plane_support_signature(
    record: &Record,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let placement_id = entity_ref_value(params.get(1)?)?;
    let placement = simple_record(&entities[*index.get(&placement_id)?])?;
    if placement.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let Parameter::List(place_params) = &placement.parameter else {
        return None;
    };
    let point_id = entity_ref_value(place_params.get(1)?)?;
    let axis_id = entity_ref_value(place_params.get(2)?)?;
    let point = cartesian_point(point_id, entities, index)?;
    let axis_record = simple_record(&entities[*index.get(&axis_id)?])?;
    let axis = direction_components(axis_record)?;

    // A plane is invariant to sliding AXIS2_PLACEMENT_3D's origin within
    // itself, and to rotating ref_direction around the normal. Signature the
    // actual oriented geometric locus: normal + signed perpendicular offset.
    let q_axis = transform_direction(axis, quarter);
    let rel = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2],
    ];
    let (rx, ry) = rotate_xy(rel[0], rel[1], quarter);
    let q_rel = [rx, ry, rel[2]];
    let (anx, any) = rotate_xy(axis[0], axis[1], quarter);
    let q_axis_f = [anx, any, axis[2]];
    // Shape-key positions use the same 1e-5 mm equivalence floor as
    // normalized vertices and axis-line offsets. A tighter plane-only key
    // spuriously splits translated copies on exporter floating-point noise.
    let offset = q_rel[2].mul_add(
        q_axis_f[2],
        q_rel[1].mul_add(q_axis_f[1], q_rel[0] * q_axis_f[0]),
    ) * 1.0e5;
    Some(format!(
        "PLANE(OFFSET{};DIR({},{},{}))",
        offset.round() as i64,
        q_axis[0],
        q_axis[1],
        q_axis[2],
    ))
}

fn support_record_signature(
    record: &Record,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
    visiting: &mut HashSet<u64>,
    depth: usize,
) -> Option<String> {
    let ignored_self_intersect = match record.name.as_str() {
        // self_intersect is exporter metadata, not part of the mathematical
        // B-spline definition. Keep closed-curve/surface flags strict.
        "B_SPLINE_CURVE" => Some(4usize),
        "B_SPLINE_CURVE_WITH_KNOTS" => Some(5usize),
        "B_SPLINE_SURFACE" => Some(6usize),
        "B_SPLINE_SURFACE_WITH_KNOTS" => Some(7usize),
        _ => None,
    };

    let params = if let (Some(ignore), Parameter::List(items)) =
        (ignored_self_intersect, &record.parameter)
    {
        if ignore < items.len() {
            let mut parts = Vec::with_capacity(items.len());
            for (idx, item) in items.iter().enumerate() {
                if idx == ignore {
                    parts.push("SELF_INTERSECT_IGNORED".to_string());
                } else {
                    parts.push(support_param_signature(
                        item,
                        entities,
                        index,
                        center,
                        quarter,
                        visiting,
                        depth + 1,
                    )?);
                }
            }
            format!("({})", parts.join(","))
        } else {
            // Complex STEP entities split inherited B-spline fields across
            // subrecords, so a WITH_KNOTS record may not carry this field.
            support_param_signature(
                &record.parameter,
                entities,
                index,
                center,
                quarter,
                visiting,
                depth + 1,
            )?
        }
    } else {
        support_param_signature(
            &record.parameter,
            entities,
            index,
            center,
            quarter,
            visiting,
            depth + 1,
        )?
    };
    Some(format!("{}{params}", record.name))
}

fn support_param_signature(
    parameter: &Parameter,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
    visiting: &mut HashSet<u64>,
    depth: usize,
) -> Option<String> {
    match parameter {
        Parameter::Ref(Name::Entity(id)) => {
            support_entity_signature(*id, entities, index, center, quarter, visiting, depth)
        }
        Parameter::Ref(Name::Value(id)) => Some(format!("@{id}")),
        Parameter::Ref(Name::ConstantEntity(value)) => Some(format!("#{value}")),
        Parameter::Ref(Name::ConstantValue(value)) => Some(format!("@{value}")),
        Parameter::Real(value) => Some(format!("R{}", (value * 1.0e9).round() as i64)),
        Parameter::Integer(value) => Some(format!("I{value}")),
        Parameter::String(value) => Some(format!("S{value:?}")),
        Parameter::Enumeration(value) => Some(format!(".{value}.")),
        Parameter::List(items) => {
            let mut parts = Vec::with_capacity(items.len());
            for item in items {
                parts.push(support_param_signature(
                    item,
                    entities,
                    index,
                    center,
                    quarter,
                    visiting,
                    depth + 1,
                )?);
            }
            Some(format!("({})", parts.join(",")))
        }
        Parameter::Typed { keyword, parameter } => Some(format!(
            "{keyword}({})",
            support_param_signature(
                parameter,
                entities,
                index,
                center,
                quarter,
                visiting,
                depth + 1,
            )?
        )),
        Parameter::NotProvided => Some("$".to_string()),
        Parameter::Omitted => Some("*".to_string()),
    }
}

fn is_topology_type(name: &str) -> bool {
    matches!(
        name,
        "MANIFOLD_SOLID_BREP"
            | "CLOSED_SHELL"
            | "OPEN_SHELL"
            | "ADVANCED_FACE"
            | "FACE_SURFACE"
            | "FACE_OUTER_BOUND"
            | "FACE_BOUND"
            | "EDGE_LOOP"
            | "ORIENTED_EDGE"
            | "EDGE_CURVE"
            | "VERTEX_POINT"
    )
}

fn direction_components(record: &Record) -> Option<[f64; 3]> {
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let Parameter::List(coords) = params.get(1)? else {
        return None;
    };
    if coords.len() != 3 {
        return None;
    }
    Some([
        number(&coords[0])?,
        number(&coords[1])?,
        number(&coords[2])?,
    ])
}

fn transform_point(point: [f64; 3], center: [f64; 3], quarter: u8) -> [i64; 3] {
    let x = point[0] - center[0];
    let y = point[1] - center[1];
    let z = point[2] - center[2];
    let (rx, ry) = rotate_xy(x, y, quarter);
    [
        (rx * 1.0e5).round() as i64,
        (ry * 1.0e5).round() as i64,
        (z * 1.0e5).round() as i64,
    ]
}

fn transform_direction(direction: [f64; 3], quarter: u8) -> [i64; 3] {
    let (x, y) = rotate_xy(direction[0], direction[1], quarter);
    [
        (x * 1.0e9).round() as i64,
        (y * 1.0e9).round() as i64,
        (direction[2] * 1.0e9).round() as i64,
    ]
}

fn parameter_literal_signature(parameter: &Parameter) -> Option<String> {
    match parameter {
        Parameter::Enumeration(value) => Some(format!(".{value}.")),
        Parameter::Integer(value) => Some(value.to_string()),
        // Scalar geometry (radii, lengths, knot literals in geometric
        // signatures) uses the same 1e-5 mm equivalence floor as points.
        Parameter::Real(value) => Some(format!("{}", (value * 1.0e5).round() as i64)),
        Parameter::Omitted => Some("*".to_string()),
        Parameter::NotProvided => Some("$".to_string()),
        _ => None,
    }
}

fn canonical_cycle(items: &[String]) -> Vec<String> {
    if items.is_empty() {
        return Vec::new();
    }
    let mut best = items.to_vec();
    for shift in 1..items.len() {
        let candidate: Vec<String> = (0..items.len())
            .map(|i| items[(i + shift) % items.len()].clone())
            .collect();
        if candidate < best {
            best = candidate;
        }
    }
    best
}

pub(crate) fn resolved_geometry_signatures(
    ids: &[u64],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<Vec<(String, Vec<i64>)>> {
    let mut out = Vec::with_capacity(ids.len());
    for &id in ids {
        let signature =
            support_entity_signature(id, entities, index, center, quarter, &mut HashSet::new(), 0)?;
        out.push((signature, Vec::new()));
    }
    Some(out)
}

pub(crate) fn canonical_z90_solid_signature(
    root: u64,
    points: &[[f64; 3]],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
) -> Option<(Vec<[i64; 3]>, String, u8)> {
    let mut best: Option<(Vec<[i64; 3]>, String, u8)> = None;
    for quarter in 0..4u8 {
        let point_key = normalized_points(points, center, quarter);
        let topology = solid_topology_signature(root, entities, index, center, quarter)?;
        let candidate = (point_key, topology, quarter);
        if best
            .as_ref()
            .is_none_or(|current| (&candidate.0, &candidate.1) < (&current.0, &current.1))
        {
            best = Some(candidate);
        }
    }
    best
}

pub(crate) fn normalized_points(
    points: &[[f64; 3]],
    center: [f64; 3],
    quarter: u8,
) -> Vec<[i64; 3]> {
    let mut candidate: Vec<[i64; 3]> = points
        .iter()
        .map(|point| transform_point(*point, center, quarter))
        .collect();
    candidate.sort_unstable();
    candidate
}

#[cfg(test)]
pub(crate) fn canonical_z90_points(points: &[[f64; 3]], center: [f64; 3]) -> (Vec<[i64; 3]>, u8) {
    let mut best: Option<Vec<[i64; 3]>> = None;
    let mut best_rotation = 0u8;
    for quarter in 0..4u8 {
        let candidate = normalized_points(points, center, quarter);
        if best.as_ref().is_none_or(|current| candidate < *current) {
            best = Some(candidate);
            best_rotation = quarter;
        }
    }
    (best.unwrap_or_default(), best_rotation)
}

pub(crate) fn centroid(points: &[[f64; 3]]) -> [f64; 3] {
    // Points are normally collected by walking a HashSet closure. Summing in
    // that randomized iteration order made the final few bits of the centroid
    // process-dependent and could flip coordinates across the 1e-9 canonical
    // quantization boundary. Sort first so identity is reproducible across
    // processes and extraction/reparse cycles.
    let mut ordered = points.to_vec();
    ordered.sort_by(|a, b| {
        a[0].total_cmp(&b[0])
            .then_with(|| a[1].total_cmp(&b[1]))
            .then_with(|| a[2].total_cmp(&b[2]))
    });

    let mut sum = [0.0; 3];
    let mut compensation = [0.0; 3];
    for point in &ordered {
        for axis in 0..3 {
            let value = point[axis] - compensation[axis];
            let next = sum[axis] + value;
            compensation[axis] = (next - sum[axis]) - value;
            sum[axis] = next;
        }
    }
    let n = ordered.len() as f64;
    [sum[0] / n, sum[1] / n, sum[2] / n]
}
