use crate::step_graph::visit_entity_refs;
use crate::step_identity::{
    hash_parameter_with_alias, parameters_equivalent_with_alias, resolve_alias,
};
use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

#[derive(Default)]
pub(super) struct ConsolidateStats {
    pub(super) total: usize,
    pub(super) by_type: BTreeMap<String, usize>,
}

pub(super) fn consolidate_presentation(entities: &mut Vec<EntityInstance>) -> ConsolidateStats {
    use std::collections::{HashMap, HashSet};

    let mut refcounts: HashMap<u64, usize> = HashMap::new();
    for entity in entities.iter() {
        visit_entity_refs(entity, &mut |id| *refcounts.entry(id).or_insert(0) += 1);
    }

    struct Merge {
        into: usize,
        from: usize,
        item_param_index: usize,
        ty: &'static str,
    }

    let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut merges = Vec::new();

    for (idx, entity) in entities.iter().enumerate() {
        let EntityInstance::Simple { id, record } = entity else {
            continue;
        };
        if refcounts.get(id).copied().unwrap_or(0) != 0 {
            continue;
        }
        let Parameter::List(params) = &record.parameter else {
            continue;
        };

        let (ty, key_hash, item_param_index, key_param_indices): (
            &'static str,
            u64,
            usize,
            [usize; 2],
        ) = match record.name.as_str() {
            "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION" if params.len() == 3 => {
                if !matches!(&params[1], Parameter::List(_)) {
                    continue;
                }
                (
                    "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION",
                    presentation_key_hash(0, &params[0], &params[2]),
                    1,
                    [0, 2],
                )
            }
            "PRESENTATION_LAYER_ASSIGNMENT" if params.len() == 3 => {
                if !matches!(&params[2], Parameter::List(_)) {
                    continue;
                }
                (
                    "PRESENTATION_LAYER_ASSIGNMENT",
                    presentation_key_hash(1, &params[0], &params[1]),
                    2,
                    [0, 1],
                )
            }
            _ => continue,
        };

        let matching = groups.get(&key_hash).and_then(|candidates| {
            candidates.iter().copied().find(|&candidate_idx| {
                presentation_keys_equal(&entities[candidate_idx], entity, key_param_indices)
            })
        });
        if let Some(into) = matching {
            merges.push(Merge {
                into,
                from: idx,
                item_param_index,
                ty,
            });
        } else {
            groups.entry(key_hash).or_default().push(idx);
        }
    }

    let mut additions: HashMap<usize, Vec<Parameter>> = HashMap::new();
    let mut remove = HashSet::new();
    let mut stats = ConsolidateStats::default();
    for merge in merges {
        let EntityInstance::Simple { record, .. } = &mut entities[merge.from] else {
            unreachable!();
        };
        let Parameter::List(params) = &mut record.parameter else {
            unreachable!();
        };
        let Parameter::List(items) = &mut params[merge.item_param_index] else {
            unreachable!();
        };
        additions.entry(merge.into).or_default().append(items);
        remove.insert(merge.from);
        stats.total += 1;
        *stats.by_type.entry(merge.ty.to_string()).or_insert(0) += 1;
    }

    for (idx, items) in additions {
        let EntityInstance::Simple { record, .. } = &mut entities[idx] else {
            unreachable!();
        };
        let Parameter::List(params) = &mut record.parameter else {
            unreachable!();
        };
        let target_idx = if record.name == "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION"
        {
            1
        } else {
            2
        };
        let Parameter::List(existing) = &mut params[target_idx] else {
            unreachable!();
        };
        existing.extend(items);
    }

    let mut i = 0usize;
    entities.retain(|_| {
        let keep = !remove.contains(&i);
        i += 1;
        keep
    });
    stats
}

fn presentation_key_hash(kind: u8, first: &Parameter, second: &Parameter) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    kind.hash(&mut hasher);
    crate::step_identity::hash_parameter(first, &mut hasher);
    crate::step_identity::hash_parameter(second, &mut hasher);
    hasher.finish()
}

fn presentation_keys_equal(
    left: &EntityInstance,
    right: &EntityInstance,
    key_param_indices: [usize; 2],
) -> bool {
    let (EntityInstance::Simple { record: left, .. }, EntityInstance::Simple { record: right, .. }) =
        (left, right)
    else {
        return false;
    };
    if left.name != right.name {
        return false;
    }
    let (Parameter::List(left), Parameter::List(right)) = (&left.parameter, &right.parameter)
    else {
        return false;
    };
    key_param_indices
        .into_iter()
        .all(|index| crate::step_identity::parameters_equivalent(&left[index], &right[index]))
}

#[derive(Default)]
pub(super) struct InternStats {
    pub(super) total: usize,
    pub(super) by_type: BTreeMap<String, usize>,
}

pub(super) fn intern_section(entities: &mut Vec<EntityInstance>) -> InternStats {
    // Store redirects only. An identity map for a million-entity STEP file is
    // a surprisingly expensive way of spelling "most things survive".
    let mut alias: HashMap<u64, u64> = HashMap::new();

    // Cache structural hashes and invalidate only internable parents whose direct
    // STEP references acquire a new canonical alias. The old implementation
    // rebuilt every internable key on every fixed-point round even though a key
    // can change only when one of its referenced IDs changes.
    let internable = entities.iter().map(is_internable).collect::<Vec<_>>();
    let mut parents_by_ref = HashMap::<u64, Vec<usize>>::new();
    for (index, entity) in entities.iter().enumerate() {
        if !internable[index] {
            continue;
        }
        visit_entity_refs(entity, &mut |child| {
            parents_by_ref.entry(child).or_default().push(index);
        });
    }
    for parents in parents_by_ref.values_mut() {
        parents.sort_unstable();
        parents.dedup();
    }

    let mut hashes = entities
        .iter()
        .zip(&internable)
        .map(|(entity, &enabled)| enabled.then(|| entity_structural_hash(entity, &alias)))
        .collect::<Vec<_>>();
    let mut dirty = Vec::<usize>::new();
    let mut dirty_flags = vec![false; entities.len()];

    // Value DAGs in the EasyEDA/SolidWorks corpus settle in a handful of
    // rounds (units -> uncertainty/context, colour -> style chains, geometry
    // primitives -> placements/surfaces). Keep the defensive hard cap, but do
    // expensive key construction only for nodes affected by the prior round.
    for _ in 0..16 {
        for index in dirty.drain(..) {
            dirty_flags[index] = false;
            let id = entity_id(&entities[index]);
            if internable[index] && !alias.contains_key(&id) {
                hashes[index] = Some(entity_structural_hash(&entities[index], &alias));
            }
        }

        let mut seen = HashMap::<u64, HashCanonicals>::new();
        let mut pending = Vec::<(u64, u64)>::new();
        for (index, entity) in entities.iter().enumerate() {
            if !internable[index] {
                continue;
            }
            let id = entity_id(entity);
            if alias.contains_key(&id) {
                continue;
            }
            let hash = hashes[index].expect("internable entity must have a cached structural hash");
            match seen.entry(hash) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(HashCanonicals::One(index));
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    if let Some(canonical_index) = entry.get().matching(entity, entities, &alias) {
                        let canonical =
                            resolve_alias(&alias, entity_id(&entities[canonical_index]));
                        if canonical != id {
                            pending.push((id, canonical));
                        }
                    } else {
                        entry.get_mut().push(index);
                    }
                }
            }
        }

        if pending.is_empty() {
            break;
        }

        let mut changed_aliases = Vec::with_capacity(pending.len());
        for (id, canonical) in pending {
            alias.insert(id, canonical);
            changed_aliases.push(id);
        }
        changed_aliases.extend(compress_aliases(&mut alias));
        changed_aliases.sort_unstable();
        changed_aliases.dedup();

        for id in changed_aliases {
            for &parent_index in parents_by_ref.get(&id).into_iter().flatten() {
                let parent_id = entity_id(&entities[parent_index]);
                if internable[parent_index]
                    && !alias.contains_key(&parent_id)
                    && !dirty_flags[parent_index]
                {
                    dirty_flags[parent_index] = true;
                    dirty.push(parent_index);
                }
            }
        }

        // If no surviving internable node references any newly aliased ID, all
        // cached keys are still current and another grouping round is redundant.
        if dirty.is_empty() {
            break;
        }
    }
    let _ = compress_aliases(&mut alias);

    let mut stats = InternStats::default();
    let original = std::mem::take(entities);
    entities.reserve(original.len().saturating_sub(alias.len()));
    for mut entity in original {
        let id = entity_id(&entity);
        let root = resolve_alias(&alias, id);
        if root != id {
            stats.total += 1;
            *stats.by_type.entry(entity_type_label(&entity)).or_insert(0) += 1;
            continue;
        }
        rewrite_entity_refs(&mut entity, &alias);
        entities.push(entity);
    }
    stats
}

fn compress_aliases(alias: &mut HashMap<u64, u64>) -> Vec<u64> {
    let keys = alias.keys().copied().collect::<Vec<_>>();
    let mut changed = Vec::new();
    for id in keys {
        let old = alias[&id];
        let root = resolve_alias(alias, id);
        if root != old {
            alias.insert(id, root);
            changed.push(id);
        }
    }
    changed
}

#[derive(Debug)]
enum HashCanonicals {
    One(usize),
    Many(Vec<usize>),
}

impl HashCanonicals {
    fn matching(
        &self,
        entity: &EntityInstance,
        entities: &[EntityInstance],
        alias: &HashMap<u64, u64>,
    ) -> Option<usize> {
        let matches = |index: usize| entities_equivalent(entity, &entities[index], alias);
        match self {
            Self::One(index) => matches(*index).then_some(*index),
            Self::Many(indices) => indices.iter().copied().find(|&index| matches(index)),
        }
    }

    fn push(&mut self, index: usize) {
        match self {
            Self::One(first) => *self = Self::Many(vec![*first, index]),
            Self::Many(indices) => indices.push(index),
        }
    }
}

fn entity_structural_hash(entity: &EntityInstance, alias: &HashMap<u64, u64>) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    match entity {
        EntityInstance::Simple { record, .. } => {
            0u8.hash(&mut hasher);
            hash_record(record, alias, &mut hasher);
        }
        EntityInstance::Complex { subsuper, .. } => {
            1u8.hash(&mut hasher);
            subsuper.0.len().hash(&mut hasher);
            for record in &subsuper.0 {
                hash_record(record, alias, &mut hasher);
            }
        }
    }
    hasher.finish()
}

fn hash_record(record: &Record, alias: &HashMap<u64, u64>, hasher: &mut impl Hasher) {
    record.name.hash(hasher);
    hash_parameter_with_alias(&record.parameter, alias, hasher);
}

fn entities_equivalent(
    left: &EntityInstance,
    right: &EntityInstance,
    alias: &HashMap<u64, u64>,
) -> bool {
    match (left, right) {
        (
            EntityInstance::Simple { record: left, .. },
            EntityInstance::Simple { record: right, .. },
        ) => records_equivalent(left, right, alias),
        (
            EntityInstance::Complex { subsuper: left, .. },
            EntityInstance::Complex {
                subsuper: right, ..
            },
        ) => {
            left.0.len() == right.0.len()
                && left
                    .0
                    .iter()
                    .zip(&right.0)
                    .all(|(left, right)| records_equivalent(left, right, alias))
        }
        _ => false,
    }
}

fn records_equivalent(left: &Record, right: &Record, alias: &HashMap<u64, u64>) -> bool {
    left.name == right.name
        && parameters_equivalent_with_alias(&left.parameter, &right.parameter, alias)
}

pub(super) fn dense_renumber(entities: &mut [EntityInstance]) {
    let id_map: HashMap<u64, u64> = entities
        .iter()
        .enumerate()
        .map(|(idx, e)| (entity_id(e), idx as u64 + 1))
        .collect();

    for entity in entities.iter_mut() {
        let old = entity_id(entity);
        let new = id_map[&old];
        set_entity_id(entity, new);
        rewrite_entity_refs(entity, &id_map);
    }
}

const fn entity_id(entity: &EntityInstance) -> u64 {
    match entity {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id,
    }
}

const fn set_entity_id(entity: &mut EntityInstance, new_id: u64) {
    match entity {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id = new_id,
    }
}

fn entity_type_label(entity: &EntityInstance) -> String {
    match entity {
        EntityInstance::Simple { record, .. } => record.name.clone(),
        EntityInstance::Complex { subsuper, .. } => subsuper
            .0
            .iter()
            .map(|r| r.name.as_str())
            .collect::<Vec<_>>()
            .join("+"),
    }
}

fn rewrite_entity_refs(entity: &mut EntityInstance, map: &HashMap<u64, u64>) {
    match entity {
        EntityInstance::Simple { record, .. } => rewrite_param_refs(&mut record.parameter, map),
        EntityInstance::Complex { subsuper, .. } => {
            for record in &mut subsuper.0 {
                rewrite_param_refs(&mut record.parameter, map);
            }
        }
    }
}

fn rewrite_param_refs(param: &mut Parameter, map: &HashMap<u64, u64>) {
    match param {
        Parameter::Ref(Name::Entity(id)) => {
            if let Some(&new) = map.get(id) {
                *id = new;
            }
        }
        Parameter::List(items) => {
            for item in items {
                rewrite_param_refs(item, map);
            }
        }
        Parameter::Typed { parameter, .. } => rewrite_param_refs(parameter, map),
        _ => {}
    }
}

fn is_internable(entity: &EntityInstance) -> bool {
    match entity {
        EntityInstance::Simple { record, .. } => internable_record(&record.name),
        EntityInstance::Complex { subsuper, .. } => {
            !subsuper.0.is_empty() && subsuper.0.iter().all(|r| internable_record(&r.name))
        }
    }
}

// Deliberately excludes topological identity objects: VERTEX_POINT, EDGE_CURVE,
// ORIENTED_EDGE, EDGE_LOOP, FACE_*, ADVANCED_FACE, shells, and solids.
//
// Sharing these value/geometry-support objects does not merge topology. It only
// makes multiple topological objects point at the same equal geometry/style/unit
// value, which is the redundancy SolidWorks explodes in these EasyEDA files.
fn internable_record(name: &str) -> bool {
    matches!(
        name,
        // Geometry values / support geometry
        "CARTESIAN_POINT"
            | "DIRECTION"
            | "VECTOR"
            | "AXIS1_PLACEMENT"
            | "AXIS2_PLACEMENT_2D"
            | "AXIS2_PLACEMENT_3D"
            | "LINE"
            | "CIRCLE"
            | "ELLIPSE"
            | "PLANE"
            | "CYLINDRICAL_SURFACE"
            | "CONICAL_SURFACE"
            | "SPHERICAL_SURFACE"
            | "TOROIDAL_SURFACE"
            | "SURFACE_OF_LINEAR_EXTRUSION"
            | "SURFACE_OF_REVOLUTION"
            | "B_SPLINE_CURVE"
            | "B_SPLINE_CURVE_WITH_KNOTS"
            | "RATIONAL_B_SPLINE_CURVE"
            | "B_SPLINE_SURFACE"
            | "B_SPLINE_SURFACE_WITH_KNOTS"
            | "RATIONAL_B_SPLINE_SURFACE"
            // Presentation/style values
            | "COLOUR_RGB"
            | "DRAUGHTING_PRE_DEFINED_COLOUR"
            | "DRAUGHTING_PRE_DEFINED_CURVE_FONT"
            | "CURVE_STYLE"
            | "POINT_STYLE"
            | "FILL_AREA_STYLE_COLOUR"
            | "FILL_AREA_STYLE"
            | "SURFACE_STYLE_FILL_AREA"
            | "SURFACE_SIDE_STYLE"
            | "SURFACE_STYLE_USAGE"
            | "PRESENTATION_STYLE_ASSIGNMENT"
            // Units / contexts / uncertainty values
            | "NAMED_UNIT"
            | "SI_UNIT"
            | "LENGTH_UNIT"
            | "PLANE_ANGLE_UNIT"
            | "SOLID_ANGLE_UNIT"
            | "CONVERSION_BASED_UNIT"
            | "MEASURE_WITH_UNIT"
            | "UNCERTAINTY_MEASURE_WITH_UNIT"
            | "REPRESENTATION_CONTEXT"
            | "GEOMETRIC_REPRESENTATION_CONTEXT"
            | "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT"
            | "GLOBAL_UNIT_ASSIGNED_CONTEXT"
    )
}

const PLACEHOLDER_NAME_TYPES: &[&str] = &[
    "ADVANCED_FACE",
    "AXIS2_PLACEMENT_3D",
    "B_SPLINE_CURVE_WITH_KNOTS",
    "B_SPLINE_SURFACE_WITH_KNOTS",
    "CARTESIAN_POINT",
    "CIRCLE",
    "CLOSED_SHELL",
    "CONICAL_SURFACE",
    "CYLINDRICAL_SURFACE",
    "DIRECTION",
    "EDGE_CURVE",
    "EDGE_LOOP",
    "FACE_BOUND",
    "FACE_OUTER_BOUND",
    "LINE",
    "MANIFOLD_SOLID_BREP",
    "ORIENTED_EDGE",
    "PLANE",
    "SPHERICAL_SURFACE",
    "STYLED_ITEM",
    "TOROIDAL_SURFACE",
    "VECTOR",
    "VERTEX_POINT",
];

pub(super) fn minify_placeholder_names(entities: &mut [EntityInstance]) -> usize {
    let mut changed = 0usize;
    for entity in entities {
        match entity {
            EntityInstance::Simple { record, .. } => {
                changed += minify_placeholder_record_name(record);
            }
            EntityInstance::Complex { subsuper, .. } => {
                for record in &mut subsuper.0 {
                    changed += minify_placeholder_record_name(record);
                }
            }
        }
    }
    changed
}

fn minify_placeholder_record_name(record: &mut Record) -> usize {
    if !PLACEHOLDER_NAME_TYPES.contains(&record.name.as_str()) {
        return 0;
    }
    let Parameter::List(params) = &mut record.parameter else {
        return 0;
    };
    let Some(Parameter::String(name)) = params.first_mut() else {
        return 0;
    };
    if name != "NONE" {
        return 0;
    }
    name.clear();
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(id: u64, name: &str, description: &str, item: u64) -> EntityInstance {
        EntityInstance::Simple {
            id,
            record: Record {
                name: "PRESENTATION_LAYER_ASSIGNMENT".to_string(),
                parameter: Parameter::List(vec![
                    Parameter::String(name.to_string()),
                    Parameter::String(description.to_string()),
                    Parameter::List(vec![Parameter::Ref(Name::Entity(item))]),
                ]),
            },
        }
    }

    #[test]
    fn presentation_consolidation_merges_structurally_equal_roots() {
        let mut entities = vec![
            layer(10, "top", "copper", 100),
            layer(20, "top", "copper", 200),
        ];
        let stats = consolidate_presentation(&mut entities);
        assert_eq!(stats.total, 1);
        assert_eq!(entities.len(), 1);
        let EntityInstance::Simple { record, .. } = &entities[0] else {
            unreachable!();
        };
        let Parameter::List(params) = &record.parameter else {
            unreachable!();
        };
        let Parameter::List(items) = &params[2] else {
            unreachable!();
        };
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn presentation_consolidation_keeps_distinct_structural_keys() {
        let mut entities = vec![
            layer(10, "top", "copper", 100),
            layer(20, "bottom", "copper", 200),
        ];
        let stats = consolidate_presentation(&mut entities);
        assert_eq!(stats.total, 0);
        assert_eq!(entities.len(), 2);
    }
}
