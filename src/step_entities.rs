use crate::step_graph::{entity_ref_value, simple_record, simple_record_mut, visit_entity_refs};
use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use std::collections::{HashMap, HashSet};

pub(super) fn cartesian_point(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let &idx = index.get(&id)?;
    let record = simple_record(&entities[idx])?;
    if record.name != "CARTESIAN_POINT" {
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

pub(super) fn number(param: &Parameter) -> Option<f64> {
    match param {
        Parameter::Real(value) => Some(*value),
        Parameter::Integer(value) => crate::numeric::exact_i64_to_f64(*value),
        _ => None,
    }
}

pub(super) fn nth_entity_ref(parameter: &Parameter, idx: usize) -> Option<u64> {
    let Parameter::List(params) = parameter else {
        return None;
    };
    entity_ref_value(params.get(idx)?)
}

pub(super) fn closure_from(
    root: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> HashSet<u64> {
    let mut seen = HashSet::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(&idx) = index.get(&id) else {
            continue;
        };
        visit_entity_refs(&entities[idx], &mut |child| {
            if index.contains_key(&child) && !seen.contains(&child) {
                stack.push(child);
            }
        });
    }
    seen
}

pub(super) fn representation_items_and_context(entity: &EntityInstance) -> Option<(Vec<u64>, u64)> {
    let record = simple_record(entity)?;
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    if params.len() != 3 {
        return None;
    }
    let Parameter::List(items) = &params[1] else {
        return None;
    };
    let item_ids: Option<Vec<u64>> = items.iter().map(entity_ref_value).collect();
    Some((item_ids?, entity_ref_value(&params[2])?))
}

pub(super) fn patch_presentation_lists(
    entities: &mut [EntityInstance],
    remove: &HashSet<u64>,
    add: &[u64],
) {
    for entity in entities {
        let Some(record) = simple_record_mut(entity) else {
            continue;
        };
        let target_index = match record.name.as_str() {
            "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION" => 1,
            "PRESENTATION_LAYER_ASSIGNMENT" => 2,
            _ => continue,
        };
        let Parameter::List(params) = &mut record.parameter else {
            continue;
        };
        let Some(Parameter::List(items)) = params.get_mut(target_index) else {
            continue;
        };
        let had_removed = items
            .iter()
            .filter_map(entity_ref_value)
            .any(|id| remove.contains(&id));
        if !had_removed {
            continue;
        }
        items.retain(|item| entity_ref_value(item).is_none_or(|id| !remove.contains(&id)));
        items.extend(add.iter().copied().map(entity_ref));
    }
}

pub(super) fn push_point(
    entities: &mut Vec<EntityInstance>,
    next_id: &mut u64,
    point: [f64; 3],
) -> u64 {
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

pub(super) fn push_simple(
    entities: &mut Vec<EntityInstance>,
    next_id: &mut u64,
    name: &str,
    params: Vec<Parameter>,
) -> u64 {
    let id = *next_id;
    *next_id += 1;
    entities.push(EntityInstance::Simple {
        id,
        record: Record {
            name: name.to_string(),
            parameter: Parameter::List(params),
        },
    });
    id
}

pub(super) const fn entity_ref(id: u64) -> Parameter {
    Parameter::Ref(Name::Entity(id))
}
