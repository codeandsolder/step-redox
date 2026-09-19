use crate::instances::{entity_ref, entity_ref_value, simple_record, simple_record_mut};
use ruststep::ast::{EntityInstance, Parameter};
use std::collections::{HashMap, HashSet};

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
    if record.name != "ADVANCED_FACE" {
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
    let record = simple_record(&entities[*index.get(&bound_id)?])?;
    if record.name != "FACE_BOUND" && record.name != "FACE_OUTER_BOUND" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let loop_id = entity_ref_value(params.get(1)?)?;
    let orientation = enumeration_value(params.get(2)?)?;

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

    let mut edge_curves = HashSet::new();
    for oriented in oriented_edges {
        let oriented_id = entity_ref_value(oriented)?;
        let oriented_record = simple_record(&entities[*index.get(&oriented_id)?])?;
        if oriented_record.name != "ORIENTED_EDGE" {
            return None;
        }
        let Parameter::List(oriented_params) = &oriented_record.parameter else {
            return None;
        };
        edge_curves.insert(entity_ref_value(oriented_params.get(3)?)?);
    }
    Some((loop_id, edge_curves, orientation))
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
