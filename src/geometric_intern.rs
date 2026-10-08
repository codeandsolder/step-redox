use crate::math3::{dot, mul, norm, sub};
use crate::step_entities::{cartesian_point, direction_components as direction, number};
use crate::step_graph::{
    ReferenceGraph, build_index, entity_id, entity_ref_value, rewrite_entity_refs, simple_record,
};
use ruststep::ast::{EntityInstance, Parameter};
use std::collections::{HashMap, HashSet};

const POS_TOL_MM: f64 = 1.0e-5;
const SCALAR_TOL_MM: f64 = 1.0e-5;
// Direction is dimensionless. Keep this much tighter than position: an angular
// error can accumulate over a long surface even when its origin matches.
const DIR_TOL: f64 = 1.0e-10;
// Reusing the same infinite line with a very distant exporter-local origin can
// make tolerant STEP importers retrim otherwise unchanged EDGE_CURVEs
// differently. Keep the canonical origin reasonably local to every edge.
const LINE_MAX_ORIGIN_SHIFT_EDGE_CHORDS: f64 = 64.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SupportKey {
    Plane {
        normal: [i64; 3],
        offset: i64,
    },
    Line {
        direction: [i64; 3],
        closest: [i64; 3],
    },
    Cylinder {
        direction: [i64; 3],
        closest: [i64; 3],
        radius: i64,
    },
}

#[derive(Debug, Default, Clone)]
pub struct GeometricInternStats {
    pub supports_merged: usize,
    pub entities_removed: usize,
    pub planes_merged: usize,
    pub lines_merged: usize,
    pub cylinders_merged: usize,
}

/// Merge support geometry by geometric locus, not exporter-local placement
/// frames.  This is deliberately narrower than ordinary value interning:
///
/// * PLANE / `CYLINDRICAL_SURFACE` must be referenced only as `ADVANCED_FACE`
///   support geometry.
/// * LINE must be referenced only by `EDGE_CURVE`.
///
/// That restriction keeps parameter-space consumers (PCURVE, `TRIMMED_CURVE`,
/// etc.) out of the pass. In those safe roles the topology supplies the
/// trimming endpoints/loops. LINE aliases get one additional guard because
/// tolerant importers may still retrim an `EDGE_CURVE` differently when its
/// underlying line is replaced by the same locus with a very remote parameter
/// origin.
pub fn intern_geometric_supports(entities: &mut Vec<EntityInstance>) -> GeometricInternStats {
    let mut stats = GeometricInternStats::default();
    if entities.is_empty() {
        return stats;
    }

    let index = build_index(entities);
    let references = ReferenceGraph::new(entities);
    let inbound = references.inbound();

    let mut seen: HashMap<SupportKey, u64> = HashMap::new();
    let mut alias: HashMap<u64, u64> = HashMap::new();

    for entity in entities.iter() {
        let id = entity_id(entity);
        let Some(record) = simple_record(entity) else {
            continue;
        };
        let safe_parent = match record.name.as_str() {
            "PLANE" | "CYLINDRICAL_SURFACE" => inbound.get(&id).is_some_and(|parents| {
                !parents.is_empty()
                    && parents.iter().all(|parent| {
                        index
                            .get(parent)
                            .and_then(|&i| simple_record(&entities[i]))
                            .is_some_and(|r| r.name == "ADVANCED_FACE")
                    })
            }),
            "LINE" => inbound.get(&id).is_some_and(|parents| {
                !parents.is_empty()
                    && parents.iter().all(|parent| {
                        index
                            .get(parent)
                            .and_then(|&i| simple_record(&entities[i]))
                            .is_some_and(|r| r.name == "EDGE_CURVE")
                    })
            }),
            _ => false,
        };
        if !safe_parent {
            continue;
        }

        let Some(key) = support_key(id, entities, &index) else {
            continue;
        };
        if let Some(&canonical) = seen.get(&key) {
            if canonical != id {
                if record.name == "LINE"
                    && !line_alias_is_safe(id, canonical, entities, &index, &inbound)
                {
                    continue;
                }
                alias.insert(id, canonical);
                stats.supports_merged += 1;
                match record.name.as_str() {
                    "PLANE" => stats.planes_merged += 1,
                    "LINE" => stats.lines_merged += 1,
                    "CYLINDRICAL_SURFACE" => stats.cylinders_merged += 1,
                    _ => {}
                }
            }
        } else {
            seen.insert(key, id);
        }
    }

    if alias.is_empty() {
        return stats;
    }

    // Only descendants of duplicate support roots are eligible for GC.
    let duplicate_roots: HashSet<u64> = alias.keys().copied().collect();
    let mut candidate = duplicate_roots.clone();
    let mut stack: Vec<u64> = duplicate_roots.iter().copied().collect();
    while let Some(id) = stack.pop() {
        for &child in references.refs(id).iter() {
            if index.contains_key(&child) && candidate.insert(child) {
                stack.push(child);
            }
        }
    }

    for entity in entities.iter_mut() {
        rewrite_entity_refs(entity, &alias);
    }

    let rewritten_references = ReferenceGraph::new(entities);
    let rewritten_inbound = rewritten_references.inbound();
    let mut delete = duplicate_roots;

    // Fixed-point orphan collection within the duplicate support closures.
    loop {
        let mut changed = false;
        for &id in &candidate {
            if delete.contains(&id) {
                continue;
            }
            let all_dead = rewritten_inbound
                .get(&id)
                .is_none_or(|parents| parents.iter().all(|p| delete.contains(p)));
            if all_dead {
                delete.insert(id);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    stats.entities_removed = delete.len();
    entities.retain(|entity| !delete.contains(&entity_id(entity)));
    stats
}

fn support_key(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<SupportKey> {
    let record = simple_record(&entities[*index.get(&id)?])?;
    match record.name.as_str() {
        "PLANE" => {
            let axis = nth_ref(&record.parameter, 1)?;
            let (origin, z, _x) = axis3(axis, entities, index)?;
            let n = unit(z)?;
            let d = dot(n, origin);
            Some(SupportKey::Plane {
                normal: [q(n[0], DIR_TOL), q(n[1], DIR_TOL), q(n[2], DIR_TOL)],
                offset: q(d, POS_TOL_MM),
            })
        }
        "LINE" => {
            let (p, d, _magnitude) = line_geometry(id, entities, index)?;
            // Closest point on the infinite line to the global origin.
            let c = sub(p, mul(d, dot(d, p)));
            Some(SupportKey::Line {
                direction: [q(d[0], DIR_TOL), q(d[1], DIR_TOL), q(d[2], DIR_TOL)],
                closest: [
                    q(c[0], POS_TOL_MM),
                    q(c[1], POS_TOL_MM),
                    q(c[2], POS_TOL_MM),
                ],
            })
        }
        "CYLINDRICAL_SURFACE" => {
            let axis = nth_ref(&record.parameter, 1)?;
            let radius = nth_number(&record.parameter, 2)?;
            let (origin, z, _x) = axis3(axis, entities, index)?;
            let d = unit(z)?;
            let c = sub(origin, mul(d, dot(d, origin)));
            Some(SupportKey::Cylinder {
                direction: [q(d[0], DIR_TOL), q(d[1], DIR_TOL), q(d[2], DIR_TOL)],
                closest: [
                    q(c[0], POS_TOL_MM),
                    q(c[1], POS_TOL_MM),
                    q(c[2], POS_TOL_MM),
                ],
                radius: q(radius, SCALAR_TOL_MM),
            })
        }
        _ => None,
    }
}

fn line_alias_is_safe(
    duplicate: u64,
    canonical: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    inbound: &HashMap<u64, Vec<u64>>,
) -> bool {
    let Some((duplicate_origin, _duplicate_direction, duplicate_magnitude)) =
        line_geometry(duplicate, entities, index)
    else {
        return false;
    };
    let Some((canonical_origin, canonical_direction, canonical_magnitude)) =
        line_geometry(canonical, entities, index)
    else {
        return false;
    };

    if (duplicate_magnitude - canonical_magnitude).abs() > SCALAR_TOL_MM {
        return false;
    }

    let axial_shift = dot(sub(duplicate_origin, canonical_origin), canonical_direction).abs();
    let Some(parents) = inbound.get(&duplicate) else {
        return false;
    };

    for &edge_id in parents {
        let Some(&edge_idx) = index.get(&edge_id) else {
            return false;
        };
        let Some(record) = simple_record(&entities[edge_idx]) else {
            return false;
        };
        if record.name != "EDGE_CURVE" {
            return false;
        }
        let Parameter::List(params) = &record.parameter else {
            return false;
        };

        let mut endpoints = [[0.0_f64; 3]; 2];
        for (slot, vertex_param) in [1_usize, 2].into_iter().enumerate() {
            let Some(vertex_id) = params.get(vertex_param).and_then(entity_ref_value) else {
                return false;
            };
            let Some(point) = vertex_point(vertex_id, entities, index) else {
                return false;
            };
            if point_line_distance(point, canonical_origin, canonical_direction) > POS_TOL_MM {
                return false;
            }
            endpoints[slot] = point;
        }

        let edge_chord = norm(sub(endpoints[0], endpoints[1]));
        if !edge_chord.is_finite()
            || edge_chord <= f64::EPSILON
            || axial_shift > LINE_MAX_ORIGIN_SHIFT_EDGE_CHORDS * edge_chord
        {
            return false;
        }
    }

    true
}

fn line_geometry(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<([f64; 3], [f64; 3], f64)> {
    let record = simple_record(&entities[*index.get(&id)?])?;
    if record.name != "LINE" {
        return None;
    }
    let point_id = nth_ref(&record.parameter, 1)?;
    let vector_id = nth_ref(&record.parameter, 2)?;
    let origin = cartesian_point(point_id, entities, index)?;
    let vector = simple_record(&entities[*index.get(&vector_id)?])?;
    if vector.name != "VECTOR" {
        return None;
    }
    let direction_id = nth_ref(&vector.parameter, 1)?;
    let direction = unit(direction(direction_id, entities, index)?)?;
    let magnitude = nth_number(&vector.parameter, 2)?;
    magnitude
        .is_finite()
        .then_some((origin, direction, magnitude))
}

fn vertex_point(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let record = simple_record(&entities[*index.get(&id)?])?;
    if record.name != "VERTEX_POINT" {
        return None;
    }
    cartesian_point(nth_ref(&record.parameter, 1)?, entities, index)
}

fn point_line_distance(point: [f64; 3], origin: [f64; 3], direction: [f64; 3]) -> f64 {
    let delta = sub(point, origin);
    let axial = dot(direction, delta);
    norm(sub(delta, mul(direction, axial)))
}

fn axis3(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<([f64; 3], [f64; 3], [f64; 3])> {
    let record = simple_record(&entities[*index.get(&id)?])?;
    if record.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let origin = cartesian_point(nth_ref(&record.parameter, 1)?, entities, index)?;
    let z = direction(nth_ref(&record.parameter, 2)?, entities, index)?;
    let x = direction(nth_ref(&record.parameter, 3)?, entities, index)?;
    Some((origin, z, x))
}

fn nth_ref(parameter: &Parameter, idx: usize) -> Option<u64> {
    let Parameter::List(params) = parameter else {
        return None;
    };
    entity_ref_value(params.get(idx)?)
}

fn nth_number(parameter: &Parameter, idx: usize) -> Option<f64> {
    let Parameter::List(params) = parameter else {
        return None;
    };
    number(params.get(idx)?)
}

fn q(v: f64, tol: f64) -> i64 {
    (v / tol).round() as i64
}

fn unit(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = dot(v, v).sqrt();
    if !n.is_finite() || n <= f64::EPSILON {
        return None;
    }
    Some([v[0] / n, v[1] / n, v[2] / n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruststep::ast::{Name, Record};

    fn r(id: u64) -> Parameter {
        Parameter::Ref(Name::Entity(id))
    }

    fn simple(id: u64, name: &str, params: Vec<Parameter>) -> EntityInstance {
        EntityInstance::Simple {
            id,
            record: Record {
                name: name.to_string(),
                parameter: Parameter::List(params),
            },
        }
    }

    fn point(id: u64, xyz: [f64; 3]) -> EntityInstance {
        simple(
            id,
            "CARTESIAN_POINT",
            vec![
                Parameter::String(String::new()),
                Parameter::List(xyz.into_iter().map(Parameter::Real).collect()),
            ],
        )
    }

    fn direction_entity(id: u64, xyz: [f64; 3]) -> EntityInstance {
        simple(
            id,
            "DIRECTION",
            vec![
                Parameter::String(String::new()),
                Parameter::List(xyz.into_iter().map(Parameter::Real).collect()),
            ],
        )
    }

    fn vector(id: u64, direction: u64, magnitude: f64) -> EntityInstance {
        simple(
            id,
            "VECTOR",
            vec![
                Parameter::String(String::new()),
                r(direction),
                Parameter::Real(magnitude),
            ],
        )
    }

    fn line(id: u64, origin: u64, vector: u64) -> EntityInstance {
        simple(
            id,
            "LINE",
            vec![Parameter::String(String::new()), r(origin), r(vector)],
        )
    }

    fn vertex(id: u64, point: u64) -> EntityInstance {
        simple(
            id,
            "VERTEX_POINT",
            vec![Parameter::String(String::new()), r(point)],
        )
    }

    fn edge(id: u64, a: u64, b: u64, line: u64) -> EntityInstance {
        simple(
            id,
            "EDGE_CURVE",
            vec![
                Parameter::String(String::new()),
                r(a),
                r(b),
                r(line),
                Parameter::Enumeration("T".to_string()),
            ],
        )
    }

    fn two_collinear_lines(
        duplicate_origin_x: f64,
        duplicate_edge_length: f64,
    ) -> Vec<EntityInstance> {
        vec![
            point(1, [0.0, 0.0, 0.0]),
            point(2, [duplicate_origin_x, 0.0, 0.0]),
            direction_entity(3, [1.0, 0.0, 0.0]),
            direction_entity(4, [1.0, 0.0, 0.0]),
            vector(5, 3, 1.0),
            vector(6, 4, 1.0),
            line(7, 1, 5),
            line(8, 2, 6),
            point(9, [0.0, 0.0, 0.0]),
            point(10, [0.01, 0.0, 0.0]),
            vertex(11, 9),
            vertex(12, 10),
            edge(13, 11, 12, 7),
            point(14, [duplicate_origin_x, 0.0, 0.0]),
            point(15, [duplicate_origin_x + duplicate_edge_length, 0.0, 0.0]),
            vertex(16, 14),
            vertex(17, 15),
            edge(18, 16, 17, 8),
        ]
    }

    #[test]
    fn merges_collinear_lines_when_parameter_origin_stays_local_to_edge() {
        let mut entities = two_collinear_lines(0.5, 0.01);
        let stats = intern_geometric_supports(&mut entities);
        assert_eq!(stats.lines_merged, 1);
        assert!(!build_index(&entities).contains_key(&8));
    }

    #[test]
    fn rejects_collinear_line_alias_with_distant_parameter_origin() {
        let mut entities = two_collinear_lines(0.7, 0.01);
        let stats = intern_geometric_supports(&mut entities);
        assert_eq!(stats.lines_merged, 0);
        assert!(build_index(&entities).contains_key(&8));
    }

    #[test]
    fn rejects_line_alias_when_edge_endpoint_is_off_canonical_line() {
        let mut entities = two_collinear_lines(0.2, 0.01);
        let index = build_index(&entities);
        let point_idx = index[&14];
        let EntityInstance::Simple { record, .. } = &mut entities[point_idx] else {
            panic!("expected simple point");
        };
        let Parameter::List(params) = &mut record.parameter else {
            panic!("expected point parameters");
        };
        let Parameter::List(coords) = &mut params[1] else {
            panic!("expected point coordinates");
        };
        coords[1] = Parameter::Real(2.0e-5);

        let stats = intern_geometric_supports(&mut entities);
        assert_eq!(stats.lines_merged, 0);
        assert!(build_index(&entities).contains_key(&8));
    }
}
