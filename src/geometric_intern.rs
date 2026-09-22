use crate::instances::{
    build_index, cartesian_point, entity_id, entity_ref_map, entity_ref_value, inbound_map, number,
    simple_record,
};
use ruststep::ast::{EntityInstance, Name, Parameter};
use std::collections::{HashMap, HashSet};

const POS_TOL_MM: f64 = 1.0e-5;
const SCALAR_TOL_MM: f64 = 1.0e-5;
// Direction is dimensionless. Keep this much tighter than position: an angular
// error can accumulate over a long surface even when its origin matches.
const DIR_TOL: f64 = 1.0e-10;

#[derive(Debug, Default, Clone)]
pub(crate) struct GeometricInternStats {
    pub supports_merged: usize,
    pub entities_removed: usize,
    pub planes_merged: usize,
    pub lines_merged: usize,
    pub cylinders_merged: usize,
}

/// Merge support geometry by geometric locus, not exporter-local placement
/// frames.  This is deliberately narrower than ordinary value interning:
///
/// * PLANE / CYLINDRICAL_SURFACE must be referenced only as ADVANCED_FACE
///   support geometry.
/// * LINE must be referenced only by EDGE_CURVE.
///
/// That restriction keeps parameter-space consumers (PCURVE, TRIMMED_CURVE,
/// etc.) out of the pass.  In those safe roles the topology already supplies
/// the trimming endpoints/loops, so changing an arbitrary local origin or
/// in-plane X axis does not change the represented 3-D locus.
pub(crate) fn intern_geometric_supports(
    entities: &mut Vec<EntityInstance>,
) -> GeometricInternStats {
    let mut stats = GeometricInternStats::default();
    if entities.is_empty() {
        return stats;
    }

    let index = build_index(entities);
    let refs = entity_ref_map(entities);
    let inbound = inbound_map(&refs);

    let mut seen: HashMap<String, u64> = HashMap::new();
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
        for &child in refs.get(&id).into_iter().flatten() {
            if index.contains_key(&child) && candidate.insert(child) {
                stack.push(child);
            }
        }
    }

    for entity in entities.iter_mut() {
        rewrite_refs(entity, &alias);
    }

    let rewritten_refs = entity_ref_map(entities);
    let rewritten_inbound = inbound_map(&rewritten_refs);
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
) -> Option<String> {
    let record = simple_record(&entities[*index.get(&id)?])?;
    match record.name.as_str() {
        "PLANE" => {
            let axis = nth_ref(&record.parameter, 1)?;
            let (origin, z, _x) = axis3(axis, entities, index)?;
            let n = unit(z)?;
            let d = dot(n, origin);
            Some(format!(
                "PLANE:{},{},{}:{}",
                q(n[0], DIR_TOL),
                q(n[1], DIR_TOL),
                q(n[2], DIR_TOL),
                q(d, POS_TOL_MM)
            ))
        }
        "LINE" => {
            let point_id = nth_ref(&record.parameter, 1)?;
            let vector_id = nth_ref(&record.parameter, 2)?;
            let p = cartesian_point(point_id, entities, index)?;
            let vrec = simple_record(&entities[*index.get(&vector_id)?])?;
            if vrec.name != "VECTOR" {
                return None;
            }
            let dir_id = nth_ref(&vrec.parameter, 1)?;
            let d = unit(direction(dir_id, entities, index)?)?;
            // Closest point on the infinite line to the global origin.
            let c = sub(p, mul(d, dot(d, p)));
            Some(format!(
                "LINE:{},{},{}:{},{},{}",
                q(d[0], DIR_TOL),
                q(d[1], DIR_TOL),
                q(d[2], DIR_TOL),
                q(c[0], POS_TOL_MM),
                q(c[1], POS_TOL_MM),
                q(c[2], POS_TOL_MM)
            ))
        }
        "CYLINDRICAL_SURFACE" => {
            let axis = nth_ref(&record.parameter, 1)?;
            let radius = nth_number(&record.parameter, 2)?;
            let (origin, z, _x) = axis3(axis, entities, index)?;
            let d = unit(z)?;
            let c = sub(origin, mul(d, dot(d, origin)));
            Some(format!(
                "CYL:{},{},{}:{},{},{}:{}",
                q(d[0], DIR_TOL),
                q(d[1], DIR_TOL),
                q(d[2], DIR_TOL),
                q(c[0], POS_TOL_MM),
                q(c[1], POS_TOL_MM),
                q(c[2], POS_TOL_MM),
                q(radius, SCALAR_TOL_MM)
            ))
        }
        _ => None,
    }
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

fn direction(
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

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mul(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn unit(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = dot(v, v).sqrt();
    if !n.is_finite() || n <= f64::EPSILON {
        return None;
    }
    Some([v[0] / n, v[1] / n, v[2] / n])
}

fn rewrite_refs(entity: &mut EntityInstance, alias: &HashMap<u64, u64>) {
    match entity {
        EntityInstance::Simple { record, .. } => rewrite_param(&mut record.parameter, alias),
        EntityInstance::Complex { subsuper, .. } => {
            for record in &mut subsuper.0 {
                rewrite_param(&mut record.parameter, alias);
            }
        }
    }
}

fn rewrite_param(param: &mut Parameter, alias: &HashMap<u64, u64>) {
    match param {
        Parameter::Ref(Name::Entity(id)) => {
            if let Some(&new) = alias.get(id) {
                *id = new;
            }
        }
        Parameter::List(items) => {
            for item in items {
                rewrite_param(item, alias);
            }
        }
        Parameter::Typed { parameter, .. } => rewrite_param(parameter, alias),
        _ => {}
    }
}
