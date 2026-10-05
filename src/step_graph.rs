use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use std::collections::{HashMap, HashSet};

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
    inbound: HashMap<u64, Vec<u64>>,
}

impl ReferenceGraph {
    pub(super) fn new(entities: &[EntityInstance]) -> Self {
        let mut forward = HashMap::with_capacity(entities.len());
        let mut inbound: HashMap<u64, Vec<u64>> = HashMap::with_capacity(entities.len());
        for entity in entities {
            let parent = entity_id(entity);
            let mut children = Vec::new();
            visit_entity_refs(entity, &mut |child| {
                children.push(child);
                let parents = inbound.entry(child).or_default();
                // All references from one parent are visited contiguously, so
                // the last element is enough to suppress repeated references
                // from that same entity without allocating a HashSet per child.
                if parents.last().copied() != Some(parent) {
                    parents.push(parent);
                }
            });
            if !children.is_empty() {
                forward.insert(parent, children);
            }
        }
        Self { forward, inbound }
    }

    pub(super) fn refs(&self, id: u64) -> &[u64] {
        self.forward.get(&id).map_or(&[], Vec::as_slice)
    }

    pub(super) const fn forward(&self) -> &HashMap<u64, Vec<u64>> {
        &self.forward
    }

    pub(super) const fn inbound(&self) -> &HashMap<u64, Vec<u64>> {
        &self.inbound
    }

    pub(super) fn descendant_closure(
        &self,
        present: &HashMap<u64, usize>,
        roots: &HashSet<u64>,
    ) -> HashSet<u64> {
        let mut descendants = HashSet::with_capacity(roots.len());
        let mut stack = roots.iter().copied().collect::<Vec<_>>();
        while let Some(id) = stack.pop() {
            if !present.contains_key(&id) || !descendants.insert(id) {
                continue;
            }
            stack.extend(
                self.refs(id)
                    .iter()
                    .copied()
                    .filter(|child| present.contains_key(child) && !descendants.contains(child)),
            );
        }
        descendants
    }

    fn propagate_detached(
        &self,
        candidates: &HashSet<u64>,
        root_candidates: Option<&HashSet<u64>>,
        mut delete: HashSet<u64>,
    ) -> HashSet<u64> {
        loop {
            let mut changed = false;
            for &id in candidates {
                if delete.contains(&id) {
                    continue;
                }
                let parents = self.inbound.get(&id);
                let all_dead = parents
                    .is_none_or(|parents| parents.iter().all(|parent| delete.contains(parent)));
                if !all_dead {
                    continue;
                }
                let child_of_dead = parents.is_some_and(|parents| {
                    !parents.is_empty() && parents.iter().all(|parent| delete.contains(parent))
                });
                let detached_root = root_candidates.is_some_and(|roots| roots.contains(&id));
                if detached_root || child_of_dead || parents.is_none() {
                    delete.insert(id);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        debug_assert!(self.forward.iter().all(|(id, children)| {
            delete.contains(id) || children.iter().all(|child| !delete.contains(child))
        }));
        delete
    }

    /// Collect descendants after callers have already decided that `delete` roots
    /// are being removed. Shared descendants survive while any live parent remains.
    pub(super) fn detached_descendant_closure(
        &self,
        present: &HashMap<u64, usize>,
        roots: &HashSet<u64>,
        delete: HashSet<u64>,
    ) -> HashSet<u64> {
        let candidates = self.descendant_closure(present, roots);
        self.propagate_detached(&candidates, None, delete)
    }

    /// Collect roots and descendants only when they are genuinely detached from
    /// the surviving graph. This is the safe variant for speculative cleanup.
    pub(super) fn prunable_descendant_closure(
        &self,
        present: &HashMap<u64, usize>,
        roots: &HashSet<u64>,
    ) -> HashSet<u64> {
        let candidates = self.descendant_closure(present, roots);
        self.propagate_detached(&candidates, Some(roots), HashSet::new())
    }
}
