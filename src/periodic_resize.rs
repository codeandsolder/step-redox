use crate::math3::normalize as normalize3;
use crate::step_entities::number as numeric_value;
use crate::step_graph::{
    ReferenceGraph, entity_id, entity_ref_value, simple_record, visit_entity_refs,
};
use anyhow::{Result, anyhow, bail};
use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use std::collections::{HashMap, HashSet};

use crate::instances::StyleRef;

mod graph_editor;

mod chain_weld;
#[cfg(test)]
use chain_weld::chain_curve_key;
mod body;
pub use body::{
    PeriodicBodyResizeStats, expand_periodic_body_positive, shrink_periodic_body_positive,
};

mod chain;
pub use chain::{
    PeriodicChainResizeStats, expand_periodic_chain_positive, shrink_periodic_chain_positive,
};

#[derive(Debug, Clone, Copy)]
struct SsRow {
    center: f64,
    span: f64,
    edge: u64,
}

fn insert_stretch_edge(
    stretch_edges: &mut HashMap<u64, HashSet<u64>>,
    face: u64,
    edge: u64,
) -> Result<()> {
    stretch_edges
        .get_mut(&face)
        .ok_or_else(|| anyhow!("missing target edge set for stretch face #{face}"))?
        .insert(edge);
    Ok(())
}

const COORD_TOL_MM: f64 = 1.0e-7;

fn collect_style_container_parents(
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    styles_by_target: &HashMap<u64, Vec<StyleRef>>,
    relevant_targets: &HashSet<u64>,
) -> Result<HashMap<u64, Vec<u64>>> {
    let relevant_styles = relevant_targets
        .iter()
        .flat_map(|target| styles_by_target.get(target).into_iter().flatten())
        .map(|style| style.id)
        .collect::<HashSet<_>>();
    let mut inbound = HashMap::<u64, Vec<u64>>::with_capacity(relevant_styles.len());
    for entity in entities {
        let parent = entity_id(entity);
        visit_entity_refs(entity, &mut |child| {
            if relevant_styles.contains(&child) {
                inbound.entry(child).or_default().push(parent);
            }
        });
    }
    let mut out = HashMap::<u64, Vec<u64>>::new();
    for style in relevant_targets
        .iter()
        .flat_map(|target| styles_by_target.get(target).into_iter().flatten())
    {
        let mut parents = inbound.get(&style.id).cloned().unwrap_or_default();
        parents.sort_unstable();
        parents.dedup();
        if parents.is_empty() {
            bail!("STYLED_ITEM #{} has no presentation container", style.id);
        }
        for &parent in &parents {
            let Some(&idx) = index.get(&parent) else {
                bail!(
                    "STYLED_ITEM #{} references missing parent #{parent}",
                    style.id
                );
            };
            let Some(record) = simple_record(&entities[idx]) else {
                bail!(
                    "STYLED_ITEM #{} has complex presentation parent #{parent}",
                    style.id
                );
            };
            if !matches!(
                record.name.as_str(),
                "PRESENTATION_LAYER_ASSIGNMENT"
                    | "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION"
            ) {
                bail!(
                    "STYLED_ITEM #{} has unsupported presentation parent {} #{parent}",
                    style.id,
                    record.name
                );
            }
        }
        out.insert(style.id, parents);
    }
    Ok(out)
}

fn require_face_only_direct_styles(
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    touched: &HashSet<u64>,
    styles_by_target: &HashMap<u64, Vec<StyleRef>>,
) -> Result<()> {
    for &target in touched {
        if !styles_by_target.contains_key(&target) {
            continue;
        }
        let Some(&idx) = index.get(&target) else {
            bail!("styled target #{target} is missing");
        };
        let Some(record) = simple_record(&entities[idx]) else {
            bail!("styled moved target #{target} is complex");
        };
        if record.name != "ADVANCED_FACE" {
            bail!(
                "periodic-chain presentation surgery only supports directly styled ADVANCED_FACE targets, got {} #{target}",
                record.name
            );
        }
    }
    Ok(())
}

fn list_params(record: &Record) -> Option<&[Parameter]> {
    match &record.parameter {
        Parameter::List(params) => Some(params),
        _ => None,
    }
}

const fn entity_ref(id: u64) -> Parameter {
    Parameter::Ref(Name::Entity(id))
}

fn entity_ref_list(param: &Parameter) -> Option<Vec<u64>> {
    let Parameter::List(items) = param else {
        return None;
    };
    items.iter().map(entity_ref_value).collect()
}

fn quantize_coord(point: [f64; 3]) -> [i64; 3] {
    [
        (point[0] / COORD_TOL_MM).round() as i64,
        (point[1] / COORD_TOL_MM).round() as i64,
        (point[2] / COORD_TOL_MM).round() as i64,
    ]
}

fn normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    normalize3(v, 1.0e-15)
}

#[cfg(test)]
mod tests;

/// Remove detached topological vertex roots left after later support/value
/// interning. A bare `VERTEX_POINT` with no inbound STEP reference cannot
/// participate in any represented B-rep; descendants are collected only when
/// every surviving parent is collected with it.
pub(crate) fn prune_detached_vertex_points(entities: &mut Vec<EntityInstance>) -> usize {
    if entities.is_empty() {
        return 0;
    }
    let references = ReferenceGraph::new(entities);
    let refs = references.forward();
    let inbound = references.inbound();

    let seeds = entities
        .iter()
        .filter_map(|entity| {
            let id = entity_id(entity);
            simple_record(entity)
                .is_some_and(|record| record.name == "VERTEX_POINT")
                .then_some(id)
                .filter(|id| inbound.get(id).is_none_or(Vec::is_empty))
        })
        .collect::<HashSet<_>>();
    if seeds.is_empty() {
        return 0;
    }

    let mut candidate = HashSet::<u64>::new();
    let mut stack = seeds.iter().copied().collect::<Vec<_>>();
    while let Some(id) = stack.pop() {
        if !candidate.insert(id) {
            continue;
        }
        stack.extend(refs.get(&id).into_iter().flatten().copied());
    }

    let mut delete = HashSet::<u64>::new();
    loop {
        let mut changed = false;
        for &id in &candidate {
            if delete.contains(&id) {
                continue;
            }
            let all_dead = inbound
                .get(&id)
                .is_none_or(|parents| parents.iter().all(|parent| delete.contains(parent)));
            if all_dead {
                delete.insert(id);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let removed = delete.len();
    entities.retain(|entity| !delete.contains(&entity_id(entity)));
    removed
}
