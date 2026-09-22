use crate::instances::{
    build_index, cartesian_point, entity_id, entity_ref, entity_ref_map, entity_ref_value,
    inbound_map, push_simple,
};
use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use std::collections::{HashMap, HashSet};

const GEOMETRY_TOLERANCE_MM: f64 = 1.0e-5;
// Translation values are part of the actual replica transform. They must not
// be coalesced at the looser geometry-equivalence tolerance: doing so can move
// edge supports enough to break B-rep vertex/edge consistency.
const TRANSFORM_TOLERANCE_MM: f64 = 1.0e-12;

#[derive(Debug, Default, Clone)]
pub(crate) struct CurveReplicaStats {
    pub families: usize,
    pub replicas: usize,
    pub direct_aliases: usize,
    pub transforms: usize,
    pub entities_removed: usize,
    pub max_residual_mm: f64,
}

#[derive(Debug, Clone)]
struct CurveInfo {
    id: u64,
    name: Parameter,
    points: Vec<u64>,
    xyz: Vec<[f64; 3]>,
    key: String,
}

/// Factor 3-D B_SPLINE_CURVE_WITH_KNOTS entities that differ only by one
/// translation.  Degree/knots/multiplicities/flags stay exact, so the curve
/// parameterization is unchanged.  Pole geometry is compared at 1e-5 mm.
pub(crate) fn instance_translated_bspline_curves(
    entities: &mut Vec<EntityInstance>,
) -> CurveReplicaStats {
    let mut stats = CurveReplicaStats::default();
    if entities.is_empty() {
        return stats;
    }

    let index = build_index(entities);
    let mut groups: HashMap<String, Vec<CurveInfo>> = HashMap::new();

    for entity in entities.iter() {
        let Some(info) = parse_curve(entity, entities, &index) else {
            continue;
        };
        groups.entry(info.key.clone()).or_default().push(info);
    }

    let mut next_id = entities.iter().map(entity_id).max().unwrap_or(0) + 1;
    let mut replacements: HashMap<u64, EntityInstance> = HashMap::new();
    let mut aliases: HashMap<u64, u64> = HashMap::new();
    let mut old_points = HashSet::new();
    let mut transform_cache: HashMap<[i64; 3], u64> = HashMap::new();

    for mut group in groups.into_values() {
        if group.len() < 2 {
            continue;
        }
        group.sort_by_key(|c| c.id);
        let canonical = group[0].clone();
        let mut accepted = 1usize;

        for target in group.into_iter().skip(1) {
            if target.xyz.len() != canonical.xyz.len() {
                continue;
            }
            let delta = sub(target.xyz[0], canonical.xyz[0]);
            let mut residual = 0.0f64;
            for (source, target_point) in canonical.xyz.iter().zip(&target.xyz) {
                residual = residual.max(distance(add(*source, delta), *target_point));
            }
            if residual > GEOMETRY_TOLERANCE_MM {
                continue;
            }
            stats.max_residual_mm = stats.max_residual_mm.max(residual);
            accepted += 1;
            old_points.extend(target.points.iter().copied());

            if norm(delta) <= TRANSFORM_TOLERANCE_MM {
                aliases.insert(target.id, canonical.id);
                stats.direct_aliases += 1;
                continue;
            }

            // CURVE_REPLICA applies CARTESIAN_TRANSFORMATION_OPERATOR_3D
            // local_origin as the parent -> replica translation.  Keep the
            // actual delta (not its inverse); this matches the validated
            // prototype and OCCT's imported curve geometry.
            let tq = [
                quant_transform(delta[0]),
                quant_transform(delta[1]),
                quant_transform(delta[2]),
            ];
            let transform = if let Some(&id) = transform_cache.get(&tq) {
                id
            } else {
                let point = push_point(entities, &mut next_id, delta);
                let id = push_simple(
                    entities,
                    &mut next_id,
                    "CARTESIAN_TRANSFORMATION_OPERATOR_3D",
                    vec![
                        target.name.clone(),
                        target.name.clone(),
                        target.name.clone(),
                        Parameter::NotProvided,
                        Parameter::NotProvided,
                        entity_ref(point),
                        Parameter::NotProvided,
                        Parameter::NotProvided,
                    ],
                );
                transform_cache.insert(tq, id);
                id
            };

            replacements.insert(
                target.id,
                EntityInstance::Simple {
                    id: target.id,
                    record: Record {
                        name: "CURVE_REPLICA".to_string(),
                        parameter: Parameter::List(vec![
                            target.name,
                            entity_ref(canonical.id),
                            entity_ref(transform),
                        ]),
                    },
                },
            );
            stats.replicas += 1;
        }

        if accepted >= 2 {
            stats.families += 1;
        }
    }

    if replacements.is_empty() && aliases.is_empty() {
        return stats;
    }
    stats.transforms = transform_cache.len();

    // Replace replica entities in-place first.
    for entity in entities.iter_mut() {
        let id = entity_id(entity);
        if let Some(new) = replacements.remove(&id) {
            *entity = new;
        }
    }

    // Near-zero translations are cheaper as direct reference aliases.
    if !aliases.is_empty() {
        for entity in entities.iter_mut() {
            rewrite_refs(entity, &aliases);
        }
    }

    // Candidate GC: aliased curve roots plus pole points detached by replicas.
    let mut candidate: HashSet<u64> = old_points;
    candidate.extend(aliases.keys().copied());

    let refs = entity_ref_map(entities);
    let inbound = inbound_map(&refs);
    let mut delete: HashSet<u64> = aliases.keys().copied().collect();

    loop {
        let mut changed = false;
        for &id in &candidate {
            if delete.contains(&id) {
                continue;
            }
            let all_dead = inbound
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

fn parse_curve(
    entity: &EntityInstance,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<CurveInfo> {
    let EntityInstance::Simple { id, record } = entity else {
        return None;
    };
    if record.name != "B_SPLINE_CURVE_WITH_KNOTS" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    if params.len() != 9 {
        return None;
    }
    let Parameter::List(point_params) = params.get(2)? else {
        return None;
    };
    let points: Option<Vec<u64>> = point_params.iter().map(entity_ref_value).collect();
    let points = points?;
    if points.len() < 2 {
        return None;
    }
    let xyz: Option<Vec<[f64; 3]>> = points
        .iter()
        .map(|&p| cartesian_point(p, entities, index))
        .collect();
    let xyz = xyz?;
    let p0 = xyz[0];

    // Geometry class = exact non-pole parameters + relative pole positions
    // quantized to the global 1e-5 mm equivalence floor.
    let mut key = String::new();
    for (idx, param) in params.iter().enumerate() {
        if idx == 0 || idx == 2 {
            continue; // names do not affect geometry; poles handled below.
        }
        write_param_key(param, &mut key);
        key.push('|');
    }
    for p in &xyz {
        let r = sub(*p, p0);
        key.push_str(&format!("{},{},{};", quant(r[0]), quant(r[1]), quant(r[2])));
    }

    Some(CurveInfo {
        id: *id,
        name: params[0].clone(),
        points,
        xyz,
        key,
    })
}

fn write_param_key(param: &Parameter, out: &mut String) {
    match param {
        Parameter::Typed { keyword, parameter } => {
            out.push_str(keyword);
            out.push('(');
            write_param_key(parameter, out);
            out.push(')');
        }
        Parameter::Integer(v) => out.push_str(&v.to_string()),
        Parameter::Real(v) => out.push_str(&format!("{v:.17e}")),
        Parameter::String(v) => {
            out.push('\'');
            out.push_str(v);
            out.push('\'');
        }
        Parameter::Enumeration(v) => {
            out.push('.');
            out.push_str(v);
            out.push('.');
        }
        Parameter::List(items) => {
            out.push('(');
            for item in items {
                write_param_key(item, out);
                out.push(',');
            }
            out.push(')');
        }
        Parameter::Ref(Name::Entity(id)) => out.push_str(&format!("#{id}")),
        Parameter::Ref(Name::Value(id)) => out.push_str(&format!("@{id}")),
        Parameter::Ref(Name::ConstantEntity(v)) => out.push_str(&format!("#{v}")),
        Parameter::Ref(Name::ConstantValue(v)) => out.push_str(&format!("@{v}")),
        Parameter::NotProvided => out.push('$'),
        Parameter::Omitted => out.push('*'),
    }
}

fn push_point(entities: &mut Vec<EntityInstance>, next_id: &mut u64, p: [f64; 3]) -> u64 {
    push_simple(
        entities,
        next_id,
        "CARTESIAN_POINT",
        vec![
            Parameter::String(String::new()),
            Parameter::List(p.into_iter().map(Parameter::Real).collect()),
        ],
    )
}

fn quant(v: f64) -> i64 {
    (v / GEOMETRY_TOLERANCE_MM).round() as i64
}

fn quant_transform(v: f64) -> i64 {
    (v / TRANSFORM_TOLERANCE_MM).round() as i64
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn point(id: u64, xyz: [f64; 3]) -> EntityInstance {
        EntityInstance::Simple {
            id,
            record: Record {
                name: "CARTESIAN_POINT".to_string(),
                parameter: Parameter::List(vec![
                    Parameter::String(String::new()),
                    Parameter::List(xyz.into_iter().map(Parameter::Real).collect()),
                ]),
            },
        }
    }

    fn curve(id: u64, points: [u64; 2]) -> EntityInstance {
        EntityInstance::Simple {
            id,
            record: Record {
                name: "B_SPLINE_CURVE_WITH_KNOTS".to_string(),
                parameter: Parameter::List(vec![
                    Parameter::String(String::new()),
                    Parameter::Integer(1),
                    Parameter::List(points.into_iter().map(entity_ref).collect()),
                    Parameter::Enumeration("UNSPECIFIED".to_string()),
                    Parameter::Enumeration("F".to_string()),
                    Parameter::Enumeration("F".to_string()),
                    Parameter::List(vec![Parameter::Integer(2), Parameter::Integer(2)]),
                    Parameter::List(vec![Parameter::Real(0.0), Parameter::Real(1.0)]),
                    Parameter::Enumeration("UNSPECIFIED".to_string()),
                ]),
            },
        }
    }

    fn replica_origin(entities: &[EntityInstance], curve_id: u64) -> [f64; 3] {
        let index = build_index(entities);
        let record = crate::instances::simple_record(&entities[index[&curve_id]]).unwrap();
        assert_eq!(record.name, "CURVE_REPLICA");
        let Parameter::List(params) = &record.parameter else {
            panic!("replica parameters are not a list");
        };
        let transform_id = entity_ref_value(&params[2]).unwrap();
        let transform = crate::instances::simple_record(&entities[index[&transform_id]]).unwrap();
        assert_eq!(transform.name, "CARTESIAN_TRANSFORMATION_OPERATOR_3D");
        let Parameter::List(tparams) = &transform.parameter else {
            panic!("transform parameters are not a list");
        };
        let origin_id = entity_ref_value(&tparams[5]).unwrap();
        cartesian_point(origin_id, entities, &index).unwrap()
    }

    #[test]
    fn replica_transform_uses_parent_to_target_translation() {
        let mut entities = vec![
            point(1, [0.0, 0.0, 0.0]),
            point(2, [1.0, 0.0, 0.0]),
            point(3, [3.0, -2.0, 1.0]),
            point(4, [4.0, -2.0, 1.0]),
            curve(10, [1, 2]),
            curve(20, [3, 4]),
        ];
        let stats = instance_translated_bspline_curves(&mut entities);
        assert_eq!(stats.replicas, 1);
        assert_eq!(stats.direct_aliases, 0);
        assert_eq!(replica_origin(&entities, 20), [3.0, -2.0, 1.0]);
    }

    #[test]
    fn sub_geometry_tolerance_translation_is_not_discarded() {
        let delta = 1.0e-8;
        let mut entities = vec![
            point(1, [0.0, 0.0, 0.0]),
            point(2, [1.0, 0.0, 0.0]),
            point(3, [delta, 0.0, 0.0]),
            point(4, [1.0 + delta, 0.0, 0.0]),
            curve(10, [1, 2]),
            curve(20, [3, 4]),
        ];
        let stats = instance_translated_bspline_curves(&mut entities);
        assert_eq!(stats.direct_aliases, 0);
        assert_eq!(stats.replicas, 1);
        let got = replica_origin(&entities, 20);
        assert!((got[0] - delta).abs() <= 1.0e-15, "{got:?}");
    }
}
