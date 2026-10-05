use crate::math3::{add, distance, norm, sub};
use crate::step_entities::{cartesian_point, entity_ref, push_simple};
use crate::step_graph::{ReferenceGraph, build_index, entity_id, entity_ref_value, simple_record};
use crate::step_identity::{IndexBucket, hash_parameter, parameters_equivalent};
use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

const GEOMETRY_TOLERANCE_MM: f64 = 1.0e-5;
// Translation values are part of the actual replica transform. They must not
// be coalesced at the looser geometry-equivalence tolerance: doing so can move
// edge supports enough to break B-rep vertex/edge consistency.
const TRANSFORM_TOLERANCE_MM: f64 = 1.0e-12;

#[derive(Debug, Default, Clone)]
pub struct CurveReplicaStats {
    pub families: usize,
    pub replicas: usize,
    pub direct_aliases: usize,
    pub transforms: usize,
    pub entities_removed: usize,
    pub max_residual_mm: f64,
}

#[derive(Debug, Clone, Copy)]
struct CurvePole {
    id: u64,
    xyz: [f64; 3],
}

#[derive(Debug, Clone)]
struct CurveInfo {
    id: u64,
    name: Parameter,
    poles: Vec<CurvePole>,
}

/// Factor 3-D `B_SPLINE_CURVE_WITH_KNOTS` entities that differ only by one
/// translation.  Degree/knots/multiplicities/flags stay exact, so the curve
/// parameterization is unchanged.  Pole geometry is compared at 1e-5 mm.
pub fn instance_translated_bspline_curves(entities: &mut Vec<EntityInstance>) -> CurveReplicaStats {
    let mut stats = CurveReplicaStats::default();
    if entities.is_empty() {
        return stats;
    }

    let index = build_index(entities);
    let mut groups = Vec::<Vec<CurveInfo>>::new();
    let mut groups_by_hash = HashMap::<u64, IndexBucket>::new();

    for entity in entities.iter() {
        let Some((key_hash, info)) = parse_curve(entity, entities, &index) else {
            continue;
        };
        let matching_group = groups_by_hash.get(&key_hash).and_then(|candidates| {
            candidates.find(|group_index| {
                curve_keys_equal(&info, &groups[group_index][0], entities, &index)
            })
        });
        if let Some(group_index) = matching_group {
            groups[group_index].push(info);
        } else {
            let group_index = groups.len();
            match groups_by_hash.entry(key_hash) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(IndexBucket::one(group_index));
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    entry.get_mut().push(group_index);
                }
            }
            groups.push(vec![info]);
        }
    }

    let mut next_id = entities.iter().map(entity_id).max().unwrap_or(0) + 1;
    let mut replacements: HashMap<u64, EntityInstance> = HashMap::new();
    let mut aliases: HashMap<u64, u64> = HashMap::new();
    let mut old_points = HashSet::new();
    let mut transform_cache: HashMap<[i64; 3], u64> = HashMap::new();

    for group in &mut groups {
        group.sort_by_key(|curve| curve.id);
    }
    groups.sort_by_key(|group| group.first().map_or(u64::MAX, |curve| curve.id));

    for group in groups {
        if group.len() < 2 {
            continue;
        }
        let canonical = group[0].clone();
        let mut accepted = 1usize;

        for target in group.into_iter().skip(1) {
            if target.poles.len() != canonical.poles.len() {
                continue;
            }
            let delta = sub(target.poles[0].xyz, canonical.poles[0].xyz);
            let mut residual = 0.0f64;
            for (source, target_point) in canonical.poles.iter().zip(&target.poles) {
                residual = residual.max(distance(add(source.xyz, delta), target_point.xyz));
            }
            if residual > GEOMETRY_TOLERANCE_MM {
                continue;
            }
            stats.max_residual_mm = stats.max_residual_mm.max(residual);
            accepted += 1;
            old_points.extend(target.poles.iter().map(|pole| pole.id));

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

    let references = ReferenceGraph::new(entities);
    let inbound = references.inbound();
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
) -> Option<(u64, CurveInfo)> {
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
    let poles: Option<Vec<CurvePole>> = point_params
        .iter()
        .map(|param| {
            let id = entity_ref_value(param)?;
            let xyz = cartesian_point(id, entities, index)?;
            Some(CurvePole { id, xyz })
        })
        .collect();
    let poles = poles?;
    if poles.len() < 2 {
        return None;
    }
    // Geometry class = exact non-pole parameters + relative pole positions
    // quantized to the global 1e-5 mm equivalence floor. Hash structurally and
    // verify equality inside each hash bucket, avoiding a serialized String key.
    let key_hash = curve_key_hash(params, &poles);

    Some((
        key_hash,
        CurveInfo {
            id: *id,
            name: params[0].clone(),
            poles,
        },
    ))
}

fn curve_key_hash(params: &[Parameter], poles: &[CurvePole]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for (index, param) in params.iter().enumerate() {
        if index == 0 || index == 2 {
            continue;
        }
        index.hash(&mut hasher);
        hash_parameter(param, &mut hasher);
    }
    poles.len().hash(&mut hasher);
    let origin = poles[0].xyz;
    for pole in poles {
        let relative = sub(pole.xyz, origin);
        [quant(relative[0]), quant(relative[1]), quant(relative[2])].hash(&mut hasher);
    }
    hasher.finish()
}

fn curve_keys_equal(
    left: &CurveInfo,
    right: &CurveInfo,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> bool {
    if left.poles.len() != right.poles.len() {
        return false;
    }
    let Some(left_params) = curve_params(left.id, entities, index) else {
        return false;
    };
    let Some(right_params) = curve_params(right.id, entities, index) else {
        return false;
    };
    if left_params.len() != right_params.len()
        || left_params
            .iter()
            .zip(right_params)
            .enumerate()
            .any(|(index, (left, right))| {
                index != 0 && index != 2 && !parameters_equivalent(left, right)
            })
    {
        return false;
    }

    let left_origin = left.poles[0].xyz;
    let right_origin = right.poles[0].xyz;
    left.poles.iter().zip(&right.poles).all(|(left, right)| {
        let left = sub(left.xyz, left_origin);
        let right = sub(right.xyz, right_origin);
        [quant(left[0]), quant(left[1]), quant(left[2])]
            == [quant(right[0]), quant(right[1]), quant(right[2])]
    })
}

fn curve_params<'a>(
    id: u64,
    entities: &'a [EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<&'a [Parameter]> {
    let record = simple_record(&entities[*index.get(&id)?])?;
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    Some(params)
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
    use crate::step_graph::simple_record;

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
        curve_with_degree(id, points, 1)
    }

    fn curve_with_degree(id: u64, points: [u64; 2], degree: i64) -> EntityInstance {
        EntityInstance::Simple {
            id,
            record: Record {
                name: "B_SPLINE_CURVE_WITH_KNOTS".to_string(),
                parameter: Parameter::List(vec![
                    Parameter::String(String::new()),
                    Parameter::Integer(degree),
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

    fn replica_origin(entities: &[EntityInstance], curve_id: u64) -> anyhow::Result<[f64; 3]> {
        let index = build_index(entities);
        let curve_index = index
            .get(&curve_id)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("missing curve replica #{curve_id}"))?;
        let record = simple_record(&entities[curve_index])
            .ok_or_else(|| anyhow::anyhow!("curve replica #{curve_id} is not a simple record"))?;
        assert_eq!(record.name, "CURVE_REPLICA");
        let Parameter::List(params) = &record.parameter else {
            anyhow::bail!("replica parameters are not a list");
        };
        let transform_id = entity_ref_value(&params[2])
            .ok_or_else(|| anyhow::anyhow!("replica has no transformation reference"))?;
        let transform_index = index
            .get(&transform_id)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("missing transform #{transform_id}"))?;
        let transform = simple_record(&entities[transform_index])
            .ok_or_else(|| anyhow::anyhow!("transform #{transform_id} is not a simple record"))?;
        assert_eq!(transform.name, "CARTESIAN_TRANSFORMATION_OPERATOR_3D");
        let Parameter::List(tparams) = &transform.parameter else {
            anyhow::bail!("transform parameters are not a list");
        };
        let origin_id = entity_ref_value(&tparams[5])
            .ok_or_else(|| anyhow::anyhow!("transform has no origin reference"))?;
        cartesian_point(origin_id, entities, &index)
            .ok_or_else(|| anyhow::anyhow!("missing transform origin point #{origin_id}"))
    }

    #[test]
    fn replica_transform_uses_parent_to_target_translation() -> anyhow::Result<()> {
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
        assert_eq!(replica_origin(&entities, 20)?, [3.0, -2.0, 1.0]);
        Ok(())
    }

    #[test]
    fn translated_curves_with_different_parameters_do_not_share_a_family() {
        let mut entities = vec![
            point(1, [0.0, 0.0, 0.0]),
            point(2, [1.0, 0.0, 0.0]),
            point(3, [3.0, -2.0, 1.0]),
            point(4, [4.0, -2.0, 1.0]),
            curve_with_degree(10, [1, 2], 1),
            curve_with_degree(20, [3, 4], 2),
        ];
        let stats = instance_translated_bspline_curves(&mut entities);
        assert_eq!(stats.replicas, 0);
        assert_eq!(stats.direct_aliases, 0);
        assert_eq!(stats.families, 0);
    }

    #[test]
    fn sub_geometry_tolerance_translation_is_not_discarded() -> anyhow::Result<()> {
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
        let got = replica_origin(&entities, 20)?;
        assert!((got[0] - delta).abs() <= 1.0e-15, "{got:?}");
        Ok(())
    }
}
