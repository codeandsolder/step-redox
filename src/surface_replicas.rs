use crate::instances::{
    cartesian_point, closure_from, entity_ref, entity_ref_map, inbound_map, number, push_simple,
};
use crate::math3::{distance, norm, sub};
use crate::step_graph::{build_index, entity_id, entity_ref_value, simple_record};
use ruststep::ast::{EntityInstance, Parameter, Record};
use std::collections::{HashMap, HashSet};

const TRANSFORM_TOLERANCE_MM: f64 = 1.0e-12;

#[derive(Debug, Default, Clone)]
pub struct SurfaceReplicaStats {
    pub families: usize,
    pub replicas: usize,
    pub plane_replicas: usize,
    pub cylinder_replicas: usize,
    pub transforms: usize,
    pub entities_removed: usize,
    pub max_transform_residual_mm: f64,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
enum SurfaceKind {
    Plane,
    Cylinder,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct SurfaceKey {
    kind: SurfaceKind,
    axis: [u64; 3],
    ref_direction: [u64; 3],
    scalar: Option<u64>,
}

#[derive(Debug, Clone)]
struct SurfaceInfo {
    id: u64,
    name: Parameter,
    placement_id: u64,
    origin: [f64; 3],
    key: SurfaceKey,
}

/// Factor translated analytic supports while leaving the fused B-rep topology
/// untouched. The first occurrence remains explicit; later occurrences become
/// SURFACE_REPLICA entities under translation-only operators.
///
/// Initial scope is intentionally narrow: PLANE/CYLINDRICAL_SURFACE only,
/// explicit AXIS2_PLACEMENT_3D axis/ref-direction, identical parameterization,
/// and face-only consumers. Anything ambiguous fails closed.
pub fn instance_translated_analytic_surfaces(
    entities: &mut Vec<EntityInstance>,
) -> SurfaceReplicaStats {
    let mut stats = SurfaceReplicaStats::default();
    if entities.is_empty() {
        return stats;
    }

    let index = build_index(entities);
    let refs = entity_ref_map(entities);
    let inbound = inbound_map(&refs);
    let mut groups = HashMap::<SurfaceKey, Vec<SurfaceInfo>>::new();

    for entity in entities.iter() {
        let Some(info) = parse_surface(entity, entities, &index, &inbound) else {
            continue;
        };
        groups.entry(info.key.clone()).or_default().push(info);
    }

    let mut next_id = entities.iter().map(entity_id).max().unwrap_or(0) + 1;
    let mut replacements = HashMap::<u64, EntityInstance>::new();
    let mut gc_candidates = HashSet::<u64>::new();
    let mut transform_cache = HashMap::<[i64; 3], (u64, [f64; 3])>::new();

    let mut groups = groups.into_values().collect::<Vec<_>>();
    for group in &mut groups {
        group.sort_by_key(|surface| surface.id);
    }
    groups.sort_by_key(|group| group.first().map_or(u64::MAX, |surface| surface.id));

    for group in groups {
        if group.len() < 2 {
            continue;
        }
        let canonical = group[0].clone();
        let mut accepted = 1usize;

        for target in group.into_iter().skip(1) {
            let delta = sub(target.origin, canonical.origin);
            if !delta.iter().all(|value| value.is_finite()) || norm(delta) <= TRANSFORM_TOLERANCE_MM
            {
                continue;
            }

            let tq = [
                quant_transform(delta[0]),
                quant_transform(delta[1]),
                quant_transform(delta[2]),
            ];
            let transform = if let Some(&(id, actual_delta)) = transform_cache.get(&tq) {
                let residual = distance(actual_delta, delta);
                if residual > TRANSFORM_TOLERANCE_MM {
                    continue;
                }
                stats.max_transform_residual_mm = stats.max_transform_residual_mm.max(residual);
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
                transform_cache.insert(tq, (id, delta));
                id
            };

            gc_candidates.extend(closure_from(target.placement_id, entities, &index));

            replacements.insert(
                target.id,
                EntityInstance::Simple {
                    id: target.id,
                    record: Record {
                        name: "SURFACE_REPLICA".to_string(),
                        parameter: Parameter::List(vec![
                            target.name,
                            entity_ref(canonical.id),
                            entity_ref(transform),
                        ]),
                    },
                },
            );
            accepted += 1;
            stats.replicas += 1;
            match target.key.kind {
                SurfaceKind::Plane => stats.plane_replicas += 1,
                SurfaceKind::Cylinder => stats.cylinder_replicas += 1,
            }
        }

        if accepted >= 2 {
            stats.families += 1;
        }
    }

    if replacements.is_empty() {
        return stats;
    }
    stats.transforms = transform_cache.len();

    for entity in entities.iter_mut() {
        let id = entity_id(entity);
        if let Some(new) = replacements.remove(&id) {
            *entity = new;
        }
    }

    let refs = entity_ref_map(entities);
    let inbound = inbound_map(&refs);
    let mut delete = HashSet::<u64>::new();
    loop {
        let mut changed = false;
        for &id in &gc_candidates {
            if delete.contains(&id) {
                continue;
            }
            let detached = inbound
                .get(&id)
                .is_none_or(|parents| parents.iter().all(|parent| delete.contains(parent)));
            if detached {
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

fn parse_surface(
    entity: &EntityInstance,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    inbound: &HashMap<u64, HashSet<u64>>,
) -> Option<SurfaceInfo> {
    let EntityInstance::Simple { id, record } = entity else {
        return None;
    };

    let kind = match record.name.as_str() {
        "PLANE" => SurfaceKind::Plane,
        "CYLINDRICAL_SURFACE" => SurfaceKind::Cylinder,
        _ => return None,
    };

    let parents = inbound.get(id)?;
    if parents.is_empty()
        || parents.iter().any(|parent| {
            let Some(&parent_index) = index.get(parent) else {
                return true;
            };
            let Some(parent_record) = simple_record(&entities[parent_index]) else {
                return true;
            };
            !matches!(
                parent_record.name.as_str(),
                "ADVANCED_FACE" | "FACE_SURFACE"
            )
        })
    {
        return None;
    }

    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let name = params.first()?.clone();
    let placement_id = entity_ref_value(params.get(1)?)?;
    let placement = simple_record(&entities[*index.get(&placement_id)?])?;
    if placement.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let Parameter::List(place_params) = &placement.parameter else {
        return None;
    };
    let origin_id = entity_ref_value(place_params.get(1)?)?;
    let axis_id = entity_ref_value(place_params.get(2)?)?;
    let ref_id = entity_ref_value(place_params.get(3)?)?;
    let origin = cartesian_point(origin_id, entities, index)?;
    let axis = direction(axis_id, entities, index)?;
    let ref_direction = direction(ref_id, entities, index)?;

    let scalar = match kind {
        SurfaceKind::Plane => None,
        SurfaceKind::Cylinder => Some(canonical_bits(number(params.get(2)?)?)),
    };

    Some(SurfaceInfo {
        id: *id,
        name,
        placement_id,
        origin,
        key: SurfaceKey {
            kind,
            axis: axis.map(canonical_bits),
            ref_direction: ref_direction.map(canonical_bits),
            scalar,
        },
    })
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
    let Parameter::List(values) = params.get(1)? else {
        return None;
    };
    if values.len() != 3 {
        return None;
    }
    let out = [
        number(&values[0])?,
        number(&values[1])?,
        number(&values[2])?,
    ];
    out.iter().all(|value| value.is_finite()).then_some(out)
}

fn canonical_bits(value: f64) -> u64 {
    if value == 0.0 { 0 } else { value.to_bits() }
}

fn push_point(entities: &mut Vec<EntityInstance>, next_id: &mut u64, point: [f64; 3]) -> u64 {
    push_simple(
        entities,
        next_id,
        "CARTESIAN_POINT",
        vec![
            Parameter::String(String::new()),
            Parameter::List(point.into_iter().map(Parameter::Real).collect()),
        ],
    )
}

fn quant_transform(value: f64) -> i64 {
    (value / TRANSFORM_TOLERANCE_MM).round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruststep::ast::Name;

    fn reference(id: u64) -> Parameter {
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

    fn placement(id: u64, point: u64, axis: u64, ref_direction: u64) -> EntityInstance {
        simple(
            id,
            "AXIS2_PLACEMENT_3D",
            vec![
                Parameter::String(String::new()),
                reference(point),
                reference(axis),
                reference(ref_direction),
            ],
        )
    }

    fn cylinder(id: u64, placement: u64, radius: f64) -> EntityInstance {
        simple(
            id,
            "CYLINDRICAL_SURFACE",
            vec![
                Parameter::String(String::new()),
                reference(placement),
                Parameter::Real(radius),
            ],
        )
    }

    fn face(id: u64, support: u64) -> EntityInstance {
        simple(
            id,
            "ADVANCED_FACE",
            vec![
                Parameter::String(String::new()),
                Parameter::List(Vec::new()),
                reference(support),
                Parameter::Enumeration("T".to_string()),
            ],
        )
    }

    #[test]
    fn translated_cylinders_become_surface_replicas() {
        let mut entities = vec![
            direction_entity(1, [0.0, 0.0, 1.0]),
            direction_entity(2, [1.0, 0.0, 0.0]),
            point(3, [0.0, 0.0, 0.0]),
            point(4, [2.54, 0.0, 0.0]),
            placement(5, 3, 1, 2),
            placement(6, 4, 1, 2),
            cylinder(7, 5, 0.7),
            cylinder(8, 6, 0.7),
            face(9, 7),
            face(10, 8),
        ];
        let stats = instance_translated_analytic_surfaces(&mut entities);
        assert_eq!(stats.families, 1);
        assert_eq!(stats.replicas, 1);
        assert_eq!(stats.cylinder_replicas, 1);
        assert_eq!(stats.transforms, 1);
        let index = build_index(&entities);
        let replica = simple_record(&entities[index[&8]]).expect("replica record");
        assert_eq!(replica.name, "SURFACE_REPLICA");
        let Parameter::List(params) = &replica.parameter else {
            panic!("replica params");
        };
        assert_eq!(entity_ref_value(&params[1]), Some(7));
    }

    #[test]
    fn differing_parameterization_is_not_grouped() {
        let mut entities = vec![
            direction_entity(1, [0.0, 0.0, 1.0]),
            direction_entity(2, [1.0, 0.0, 0.0]),
            direction_entity(11, [0.0, 1.0, 0.0]),
            point(3, [0.0, 0.0, 0.0]),
            point(4, [2.54, 0.0, 0.0]),
            placement(5, 3, 1, 2),
            placement(6, 4, 1, 11),
            cylinder(7, 5, 0.7),
            cylinder(8, 6, 0.7),
            face(9, 7),
            face(10, 8),
        ];
        let stats = instance_translated_analytic_surfaces(&mut entities);
        assert_eq!(stats.replicas, 0);
    }

    #[test]
    fn non_face_consumer_fails_closed() {
        let mut entities = vec![
            direction_entity(1, [0.0, 0.0, 1.0]),
            direction_entity(2, [1.0, 0.0, 0.0]),
            point(3, [0.0, 0.0, 0.0]),
            point(4, [2.54, 0.0, 0.0]),
            placement(5, 3, 1, 2),
            placement(6, 4, 1, 2),
            cylinder(7, 5, 0.7),
            cylinder(8, 6, 0.7),
            face(9, 7),
            face(10, 8),
            simple(
                11,
                "PCURVE",
                vec![Parameter::String(String::new()), reference(8), reference(7)],
            ),
        ];
        let stats = instance_translated_analytic_surfaces(&mut entities);
        assert_eq!(stats.replicas, 0);
    }
}
