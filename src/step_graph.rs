use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use std::collections::HashMap;

pub fn build_index(entities: &[EntityInstance]) -> HashMap<u64, usize> {
    entities
        .iter()
        .enumerate()
        .map(|(index, entity)| (entity_id(entity), index))
        .collect()
}

pub const fn simple_record(entity: &EntityInstance) -> Option<&Record> {
    match entity {
        EntityInstance::Simple { record, .. } => Some(record),
        EntityInstance::Complex { .. } => None,
    }
}

pub const fn simple_record_mut(entity: &mut EntityInstance) -> Option<&mut Record> {
    match entity {
        EntityInstance::Simple { record, .. } => Some(record),
        EntityInstance::Complex { .. } => None,
    }
}

pub const fn entity_id(entity: &EntityInstance) -> u64 {
    match entity {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id,
    }
}

pub const fn entity_ref_value(parameter: &Parameter) -> Option<u64> {
    match parameter {
        Parameter::Ref(Name::Entity(id)) => Some(*id),
        _ => None,
    }
}

pub fn visit_entity_refs(entity: &EntityInstance, f: &mut impl FnMut(u64)) {
    match entity {
        EntityInstance::Simple { record, .. } => visit_param_refs(&record.parameter, f),
        EntityInstance::Complex { subsuper, .. } => {
            for record in &subsuper.0 {
                visit_param_refs(&record.parameter, f);
            }
        }
    }
}

fn visit_param_refs(param: &Parameter, f: &mut impl FnMut(u64)) {
    match param {
        Parameter::Ref(Name::Entity(id)) => f(*id),
        Parameter::List(items) => {
            for item in items {
                visit_param_refs(item, f);
            }
        }
        Parameter::Typed { parameter, .. } => visit_param_refs(parameter, f),
        _ => {}
    }
}

/// Immutable STEP entity-reference snapshot.
///
/// `forward` preserves reference occurrence order. `inbound` represents unique
/// parent entities, so repeated references from one parent do not masquerade as
/// multiple parents. Build both directions in one entity traversal when a pass
/// needs reachability or garbage-collection information.
#[derive(Debug, Default)]
pub(super) struct ReferenceGraph {
    forward: HashMap<u64, Vec<u64>>,
    inbound: HashMap<u64, std::collections::HashSet<u64>>,
}

impl ReferenceGraph {
    pub(super) fn new(entities: &[EntityInstance]) -> Self {
        let mut forward = HashMap::with_capacity(entities.len());
        let mut inbound = HashMap::with_capacity(entities.len());
        for entity in entities {
            let parent = entity_id(entity);
            let mut children = Vec::new();
            visit_entity_refs(entity, &mut |child| {
                children.push(child);
                inbound
                    .entry(child)
                    .or_insert_with(std::collections::HashSet::new)
                    .insert(parent);
            });
            forward.insert(parent, children);
        }
        Self { forward, inbound }
    }

    pub(super) fn refs(&self, id: u64) -> Option<&[u64]> {
        self.forward.get(&id).map(Vec::as_slice)
    }

    pub(super) const fn forward(&self) -> &HashMap<u64, Vec<u64>> {
        &self.forward
    }

    pub(super) const fn inbound(&self) -> &HashMap<u64, std::collections::HashSet<u64>> {
        &self.inbound
    }
}
