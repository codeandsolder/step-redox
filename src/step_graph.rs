use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use std::collections::HashMap;

pub(crate) fn build_index(entities: &[EntityInstance]) -> HashMap<u64, usize> {
    entities
        .iter()
        .enumerate()
        .map(|(index, entity)| (entity_id(entity), index))
        .collect()
}

pub(crate) const fn simple_record(entity: &EntityInstance) -> Option<&Record> {
    match entity {
        EntityInstance::Simple { record, .. } => Some(record),
        EntityInstance::Complex { .. } => None,
    }
}

pub(crate) const fn simple_record_mut(entity: &mut EntityInstance) -> Option<&mut Record> {
    match entity {
        EntityInstance::Simple { record, .. } => Some(record),
        EntityInstance::Complex { .. } => None,
    }
}

pub(crate) const fn entity_id(entity: &EntityInstance) -> u64 {
    match entity {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id,
    }
}

pub(crate) const fn entity_ref_value(parameter: &Parameter) -> Option<u64> {
    match parameter {
        Parameter::Ref(Name::Entity(id)) => Some(*id),
        _ => None,
    }
}
