use crate::shape_identity::{
    ShapeKey, normalized_points, rotate_xy, solid_identity, solid_topology_signature,
};
use crate::step_entities::{
    entity_ref, patch_presentation_lists, push_point, push_simple, representation_items_and_context,
};
pub(crate) use crate::step_graph::visit_entity_refs;
use crate::step_graph::{
    ReferenceGraph, build_index, entity_id, entity_ref_value, simple_record, simple_record_mut,
};
use ruststep::ast::{EntityInstance, Parameter, Record, SubSuperRecord};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default, Clone)]
pub struct InstanceStats {
    pub groups: usize,
    pub solids_replaced: usize,
    pub entities_removed: usize,
    pub styles_replaced: usize,
}

#[derive(Debug, Clone)]
struct SolidInfo {
    root: u64,
    closure: HashSet<u64>,
    center: [f64; 3],
    vertex_points: Vec<[f64; 3]>,
    key: ShapeKey,
    face_style: Vec<u64>,
}

pub fn instance_z90_solids(entities: &mut Vec<EntityInstance>) -> InstanceStats {
    let mut stats = InstanceStats::default();
    if entities.is_empty() {
        return stats;
    }

    let initial_index = build_index(entities);
    let styles_by_target = collect_styles_by_target(entities);

    // Only representations present before this pass are candidates. New source
    // representations created below must not be recursively reconsidered.
    let representation_ids: Vec<u64> = entities
        .iter()
        .filter_map(|entity| match entity {
            EntityInstance::Simple { id, record }
                if record.name == "ADVANCED_BREP_SHAPE_REPRESENTATION" =>
            {
                Some(*id)
            }
            _ => None,
        })
        .collect();

    let mut next_id = entities.iter().map(entity_id).max().unwrap_or(0) + 1;
    let mut geometry_candidates = HashSet::new();
    let mut duplicate_roots = HashSet::new();
    let mut old_styles_to_remove = HashSet::new();
    let mut new_style_ids = Vec::new();
    let mut shape_rep_replacements = HashMap::new();

    for representation_id in representation_ids {
        let Some(&rep_idx) = initial_index.get(&representation_id) else {
            continue;
        };
        let Some((item_ids, context_id)) = representation_items_and_context(&entities[rep_idx])
        else {
            continue;
        };

        let solid_ids: Vec<u64> = item_ids
            .iter()
            .copied()
            .filter(|id| {
                initial_index
                    .get(id)
                    .and_then(|&idx| simple_record(&entities[idx]))
                    .is_some_and(|record| record.name == "MANIFOLD_SOLID_BREP")
            })
            .collect();

        if solid_ids.len() < 3 {
            continue;
        }

        let mut infos = Vec::new();
        for root in solid_ids.iter().copied() {
            if let Some(info) = analyze_solid(root, entities, &initial_index, &styles_by_target) {
                infos.push(info);
            }
        }

        if std::env::var_os("STEP_REDOX_DEBUG_INSTANCES").is_some() {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            fn h<T: Hash>(value: &T) -> u64 {
                let mut hasher = DefaultHasher::new();
                value.hash(&mut hasher);
                hasher.finish()
            }

            eprintln!(
                "instance grouping representation={representation_id} solids={} analyzed={}",
                solid_ids.len(),
                infos.len()
            );
            for info in &infos {
                eprintln!(
                    "instance key root={} center={:?} basic=({},{},{},{}) points={:016x} edge_geom={:016x} face_geom={:016x} topo={:016x} style={:016x}",
                    info.root,
                    info.center,
                    info.key.vertices,
                    info.key.edges,
                    info.key.oriented_edges,
                    info.key.faces,
                    h(&info.key.points),
                    h(&info.key.edge_geometry),
                    h(&info.key.face_geometry),
                    h(&info.key.topology),
                    h(&info.face_style),
                );
            }
        }

        // Group owned analysis records directly. The old HashMap path cloned each
        // potentially large ShapeKey and style vector just to form a grouping key.
        infos.sort_by(|a, b| {
            a.key
                .cmp(&b.key)
                .then_with(|| a.face_style.cmp(&b.face_style))
                .then_with(|| a.root.cmp(&b.root))
        });
        let mut groups: Vec<Vec<SolidInfo>> = Vec::new();
        for info in infos {
            if let Some(group) = groups.last_mut()
                && group[0].key == info.key
                && group[0].face_style == info.face_style
            {
                group.push(info);
            } else {
                groups.push(vec![info]);
            }
        }
        // Preserve the previous deterministic emission order.
        groups.sort_by_key(|group| group.first().map_or(u64::MAX, |info| info.root));

        for group in groups {
            if group.len() < 2 || group[0].face_style.is_empty() {
                continue;
            }

            let canonical = &group[0];
            let face_style = &canonical.face_style;

            // Do not infer the actual instance transform from the canonical-key
            // rotation. Symmetric envelopes can have several equivalent
            // canonical rotations. Prove each source -> target transform
            // directly against its transformed vertex and B-rep signatures.
            let mut mapped_group = vec![(canonical, 0u8)];
            for target in group.iter().skip(1) {
                if let Some(quarter) =
                    unique_relative_quarter(canonical, target, entities, &initial_index)
                {
                    mapped_group.push((target, quarter));
                }
            }
            if mapped_group.len() < 2 {
                continue;
            }

            let z_dir = push_simple(
                entities,
                &mut next_id,
                "DIRECTION",
                vec![
                    Parameter::String(String::new()),
                    Parameter::List(vec![
                        Parameter::Real(0.0),
                        Parameter::Real(0.0),
                        Parameter::Real(1.0),
                    ]),
                ],
            );
            let mut x_dirs = [0u64; 4];
            for (quarter, xy) in [
                (0, (1.0, 0.0)),
                (1, (0.0, 1.0)),
                (2, (-1.0, 0.0)),
                (3, (0.0, -1.0)),
            ] {
                x_dirs[quarter] = push_simple(
                    entities,
                    &mut next_id,
                    "DIRECTION",
                    vec![
                        Parameter::String(String::new()),
                        Parameter::List(vec![
                            Parameter::Real(xy.0),
                            Parameter::Real(xy.1),
                            Parameter::Real(0.0),
                        ]),
                    ],
                );
            }

            // Use the representation coordinate-system origin as the map
            // origin. Then each mapping target is the direct rigid transform
            // p' = R*p + d, avoiding Axis2Placement origin-offset semantics.
            let origin_point = push_point(entities, &mut next_id, [0.0, 0.0, 0.0]);
            let origin_axis = push_simple(
                entities,
                &mut next_id,
                "AXIS2_PLACEMENT_3D",
                vec![
                    Parameter::String(String::new()),
                    entity_ref(origin_point),
                    entity_ref(z_dir),
                    entity_ref(x_dirs[0]),
                ],
            );
            let source_rep = push_simple(
                entities,
                &mut next_id,
                "ADVANCED_BREP_SHAPE_REPRESENTATION",
                vec![
                    Parameter::String("step-redox instance source".to_string()),
                    Parameter::List(vec![entity_ref(canonical.root), entity_ref(origin_axis)]),
                    entity_ref(context_id),
                ],
            );
            let rep_map = push_simple(
                entities,
                &mut next_id,
                "REPRESENTATION_MAP",
                vec![entity_ref(origin_axis), entity_ref(source_rep)],
            );

            let mut replacements = HashMap::new();
            let mut group_new_styles = Vec::new();

            for (target, relative_quarter) in &mapped_group {
                // Keep the canonical solid as the explicit parent-representation
                // occurrence and use it as the representation-map geometry.
                // Only noncanonical occurrences need mapped wrappers.
                if target.root == canonical.root {
                    continue;
                }
                let relative_quarter = *relative_quarter;
                let translation =
                    rigid_translation(canonical.center, target.center, relative_quarter);
                if std::env::var_os("STEP_REDOX_DEBUG_INSTANCES").is_some() {
                    eprintln!(
                        "emit instance source={} target={} quarter={relative_quarter} translation={translation:?}",
                        canonical.root, target.root
                    );
                }
                let point = push_point(entities, &mut next_id, translation);
                let axis = push_simple(
                    entities,
                    &mut next_id,
                    "AXIS2_PLACEMENT_3D",
                    vec![
                        Parameter::String(String::new()),
                        entity_ref(point),
                        entity_ref(z_dir),
                        entity_ref(x_dirs[relative_quarter as usize]),
                    ],
                );
                let mapped = push_simple(
                    entities,
                    &mut next_id,
                    "MAPPED_ITEM",
                    vec![
                        Parameter::String(String::new()),
                        entity_ref(rep_map),
                        entity_ref(axis),
                    ],
                );
                let styled = push_simple(
                    entities,
                    &mut next_id,
                    "STYLED_ITEM",
                    vec![
                        Parameter::String("NONE".to_string()),
                        Parameter::List(face_style.iter().copied().map(entity_ref).collect()),
                        entity_ref(mapped),
                    ],
                );
                replacements.insert(target.root, mapped);
                shape_rep_replacements.insert(target.root, mapped);
                group_new_styles.push(styled);
            }

            replace_representation_items(&mut entities[rep_idx], &replacements);

            // The canonical solid remains alive through source_rep. Only the
            // other expanded copies are deletion roots.
            for (target, _) in mapped_group.iter().skip(1) {
                duplicate_roots.insert(target.root);
                geometry_candidates.extend(target.closure.iter().copied());
            }

            // Remove old style roots targeting geometry in duplicate closures.
            for (&target, styles) in &styles_by_target {
                if geometry_candidates.contains(&target) {
                    old_styles_to_remove.extend(styles.iter().map(|style| style.id));
                }
            }

            new_style_ids.extend(group_new_styles);
            stats.groups += 1;
            stats.solids_replaced += mapped_group.len();
        }
    }

    if stats.groups == 0 {
        return stats;
    }

    // Some exporters emit auxiliary SHAPE_REPRESENTATION records for
    // per-solid validation/property data in addition to the primary B-rep
    // representation. If those keep pointing at an expanded duplicate root,
    // the duplicate stays reachable and OCCT imports it as an extra solid.
    // Retarget only shape-representation item lists; arbitrary references are
    // deliberately left untouched.
    retarget_shape_representation_items(entities, &shape_rep_replacements);

    // Presentation lists are roots in the SolidWorks files. Remove deleted
    // styled items from them and register the new mapped-item styles.
    patch_presentation_lists(entities, &old_styles_to_remove, &new_style_ids);

    // Reachability-limited garbage collection. Only geometry that was beneath
    // a duplicate solid (plus its old STYLED_ITEM roots) is eligible.
    let mut candidate = geometry_candidates;
    candidate.extend(old_styles_to_remove.iter().copied());

    let references = ReferenceGraph::new(entities);
    let inbound = references.inbound();
    let mut delete = HashSet::new();

    // Fixed point: an eligible entity can disappear only when every remaining
    // referrer is itself already disappearing. This automatically preserves
    // support values shared with the canonical solid/body/other structures.
    loop {
        let mut changed = false;
        for &id in &candidate {
            if delete.contains(&id) {
                continue;
            }
            let all_dead = inbound
                .get(&id)
                .is_none_or(|parents| parents.iter().all(|parent| delete.contains(parent)));
            if !all_dead {
                continue;
            }

            // Geometry candidates that are not beneath an actual duplicate
            // root can only be shared values; do not seed them accidentally.
            let seed = duplicate_roots.contains(&id) || old_styles_to_remove.contains(&id);
            let child_of_dead = inbound.get(&id).is_some_and(|parents| {
                !parents.is_empty() && parents.iter().all(|p| delete.contains(p))
            });
            if seed || child_of_dead {
                delete.insert(id);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    stats.styles_replaced = old_styles_to_remove
        .iter()
        .filter(|id| delete.contains(id))
        .count();
    stats.entities_removed = delete.len();
    entities.retain(|entity| !delete.contains(&entity_id(entity)));
    stats
}

fn has_z90_candidate_representation(entities: &[EntityInstance]) -> bool {
    let solid_ids = entities
        .iter()
        .filter_map(|entity| {
            let record = simple_record(entity)?;
            (record.name == "MANIFOLD_SOLID_BREP").then_some(entity_id(entity))
        })
        .collect::<HashSet<_>>();
    if solid_ids.len() < 3 {
        return false;
    }

    entities.iter().any(|entity| {
        let Some(record) = simple_record(entity) else {
            return false;
        };
        if record.name != "ADVANCED_BREP_SHAPE_REPRESENTATION" {
            return false;
        }
        representation_items_and_context(entity).is_some_and(|(items, _)| {
            items
                .into_iter()
                .filter(|id| solid_ids.contains(id))
                .take(3)
                .count()
                == 3
        })
    })
}

pub fn instance_z90_solids_assembly(entities: &mut Vec<EntityInstance>) -> InstanceStats {
    if !has_z90_candidate_representation(entities) {
        return InstanceStats::default();
    }

    // Reuse the mature geometric proof + guarded GC from the MAPPED_ITEM pass,
    // then replace only its representation layer with the assembly structure
    // emitted by OpenCascade itself. Keep a rollback copy because assembly
    // conversion intentionally supports only simple, unambiguous product trees.
    let original = entities.clone();
    let stats = instance_z90_solids(entities);
    if stats.groups == 0 {
        return stats;
    }
    if convert_z90_mapped_items_to_assembly(entities) {
        stats
    } else {
        *entities = original;
        InstanceStats::default()
    }
}

struct AssemblyPlan {
    tops: Vec<AssemblyTopPlan>,
    mapped_style_ids: HashSet<u64>,
    mapped_ids: HashSet<u64>,
    used_maps: HashSet<u64>,
}

struct AssemblyTopPlan {
    top_rep: u64,
    top_rep_index: usize,
    context_id: u64,
    residual_items: Vec<u64>,
    root_axes: Vec<u64>,
    product: AssemblyProductContext,
    families: Vec<AssemblyFamilyPlan>,
}

#[derive(Clone, Copy)]
struct AssemblyProductContext {
    root_sdr_index: usize,
    parent_pd: u64,
    pd_context: u64,
    product_context: u64,
}

struct AssemblyFamilyPlan {
    map: u64,
    source_rep: u64,
    source_origin: u64,
    canonical_solid: u64,
    inherited_style: Vec<u64>,
    canonical_style_indices: Vec<usize>,
    occurrences: Vec<(u64, u64)>,
}

impl AssemblyPlan {
    fn discover(entities: &[EntityInstance]) -> Option<Self> {
        let index = build_index(entities);
        let references = ReferenceGraph::new(entities);
        let styles_by_target = collect_styles_by_target(entities);

        let source_reps = discover_instance_source_representations(entities);
        if source_reps.is_empty() {
            return None;
        }
        let (map_to_source, map_to_origin) =
            discover_instance_representation_maps(entities, &source_reps);
        if map_to_source.is_empty() {
            return None;
        }
        let mapped_info = discover_instance_mapped_items(entities, &map_to_source);
        if mapped_info.is_empty() {
            return None;
        }
        let mapped_ids = mapped_info.keys().copied().collect::<HashSet<_>>();
        let top_reps = discover_instance_top_representations(entities, &source_reps, &mapped_ids);
        if top_reps.is_empty()
            || !mapped_items_have_unique_top_owner(entities, &index, &top_reps, &mapped_ids)
        {
            return None;
        }

        let mapped_style_ids = mapped_ids
            .iter()
            .flat_map(|mapped| {
                styles_by_target
                    .get(mapped)
                    .into_iter()
                    .flatten()
                    .map(|style| style.id)
            })
            .collect::<HashSet<_>>();

        let inbound = references.inbound();
        let mut tops = Vec::with_capacity(top_reps.len());
        let mut used_maps = HashSet::new();
        for top_rep in top_reps {
            let top = discover_assembly_top_plan(
                top_rep,
                entities,
                &index,
                inbound,
                &styles_by_target,
                &mapped_info,
                &mapped_ids,
                &map_to_source,
                &map_to_origin,
            )?;
            used_maps.extend(top.families.iter().map(|family| family.map));
            tops.push(top);
        }

        Some(Self {
            tops,
            mapped_style_ids,
            mapped_ids,
            used_maps,
        })
    }
}

fn discover_instance_source_representations(entities: &[EntityInstance]) -> HashSet<u64> {
    entities
        .iter()
        .filter_map(|entity| {
            let id = entity_id(entity);
            let record = simple_record(entity)?;
            if record.name != "ADVANCED_BREP_SHAPE_REPRESENTATION" {
                return None;
            }
            let Parameter::List(params) = &record.parameter else {
                return None;
            };
            matches!(
                params.first(),
                Some(Parameter::String(name)) if name == "step-redox instance source"
            )
            .then_some(id)
        })
        .collect()
}

fn discover_instance_representation_maps(
    entities: &[EntityInstance],
    source_reps: &HashSet<u64>,
) -> (HashMap<u64, u64>, HashMap<u64, u64>) {
    let mut map_to_source = HashMap::new();
    let mut map_to_origin = HashMap::new();
    for entity in entities {
        let id = entity_id(entity);
        let Some(record) = simple_record(entity) else {
            continue;
        };
        if record.name != "REPRESENTATION_MAP" {
            continue;
        }
        let Parameter::List(params) = &record.parameter else {
            continue;
        };
        let [origin_param, source_param] = params.as_slice() else {
            continue;
        };
        let Some(origin) = entity_ref_value(origin_param) else {
            continue;
        };
        let Some(source) = entity_ref_value(source_param) else {
            continue;
        };
        if source_reps.contains(&source) {
            map_to_source.insert(id, source);
            map_to_origin.insert(id, origin);
        }
    }
    (map_to_source, map_to_origin)
}

fn discover_instance_mapped_items(
    entities: &[EntityInstance],
    map_to_source: &HashMap<u64, u64>,
) -> HashMap<u64, (u64, u64)> {
    let mut mapped_info = HashMap::new();
    for entity in entities {
        let id = entity_id(entity);
        let Some(record) = simple_record(entity) else {
            continue;
        };
        if record.name != "MAPPED_ITEM" {
            continue;
        }
        let Parameter::List(params) = &record.parameter else {
            continue;
        };
        let [_, map_param, axis_param] = params.as_slice() else {
            continue;
        };
        let Some(map) = entity_ref_value(map_param) else {
            continue;
        };
        let Some(axis) = entity_ref_value(axis_param) else {
            continue;
        };
        if map_to_source.contains_key(&map) {
            mapped_info.insert(id, (map, axis));
        }
    }
    mapped_info
}

fn discover_instance_top_representations(
    entities: &[EntityInstance],
    source_reps: &HashSet<u64>,
    mapped_ids: &HashSet<u64>,
) -> Vec<u64> {
    entities
        .iter()
        .filter_map(|entity| {
            let id = entity_id(entity);
            if source_reps.contains(&id) {
                return None;
            }
            let record = simple_record(entity)?;
            if record.name != "ADVANCED_BREP_SHAPE_REPRESENTATION" {
                return None;
            }
            let (items, _) = representation_items_and_context(entity)?;
            items
                .iter()
                .any(|item| mapped_ids.contains(item))
                .then_some(id)
        })
        .collect()
}

fn mapped_items_have_unique_top_owner(
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    top_reps: &[u64],
    mapped_ids: &HashSet<u64>,
) -> bool {
    let mut ownership = HashMap::<u64, usize>::new();
    for &rep in top_reps {
        let Some(&entity_index) = index.get(&rep) else {
            return false;
        };
        let Some((items, _)) = representation_items_and_context(&entities[entity_index]) else {
            return false;
        };
        for item in items {
            if mapped_ids.contains(&item) {
                *ownership.entry(item).or_insert(0) += 1;
            }
        }
    }
    mapped_ids
        .iter()
        .all(|id| ownership.get(id).copied() == Some(1))
}

#[expect(
    clippy::too_many_arguments,
    reason = "assembly planning needs the immutable indexes discovered once for the whole conversion"
)]
fn discover_assembly_top_plan(
    top_rep: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    inbound: &HashMap<u64, Vec<u64>>,
    styles_by_target: &HashMap<u64, Vec<StyleRef>>,
    mapped_info: &HashMap<u64, (u64, u64)>,
    mapped_ids: &HashSet<u64>,
    map_to_source: &HashMap<u64, u64>,
    map_to_origin: &HashMap<u64, u64>,
) -> Option<AssemblyTopPlan> {
    let &top_rep_index = index.get(&top_rep)?;
    let (top_items, context_id) = representation_items_and_context(&entities[top_rep_index])?;
    let local_mapped = top_items
        .iter()
        .copied()
        .filter(|item| mapped_ids.contains(item))
        .collect::<Vec<_>>();
    if local_mapped.is_empty() {
        return None;
    }

    let product = discover_assembly_product_context(top_rep, entities, index, inbound)?;
    let residual_items = top_items
        .iter()
        .copied()
        .filter(|item| !mapped_ids.contains(item))
        .collect::<Vec<_>>();

    let mut root_axes = Vec::new();
    let mut seen_axes = HashSet::new();
    let mut local_by_map = HashMap::<u64, Vec<(u64, u64)>>::new();
    for mapped in local_mapped {
        let &(map, axis) = mapped_info.get(&mapped)?;
        if seen_axes.insert(axis) {
            root_axes.push(axis);
        }
        local_by_map.entry(map).or_default().push((mapped, axis));
    }

    let mut local_by_map = local_by_map.into_iter().collect::<Vec<_>>();
    local_by_map.sort_by_key(|(map, _)| *map);
    let mut families = Vec::with_capacity(local_by_map.len());
    for (map, mut occurrences) in local_by_map {
        occurrences.sort_by_key(|(mapped, _)| *mapped);
        families.push(discover_assembly_family_plan(
            map,
            occurrences,
            entities,
            index,
            styles_by_target,
            map_to_source,
            map_to_origin,
        )?);
    }

    Some(AssemblyTopPlan {
        top_rep,
        top_rep_index,
        context_id,
        residual_items,
        root_axes,
        product,
        families,
    })
}

fn discover_assembly_product_context(
    top_rep: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    inbound: &HashMap<u64, Vec<u64>>,
) -> Option<AssemblyProductContext> {
    let sdr_candidates = inbound
        .get(&top_rep)
        .into_iter()
        .flatten()
        .copied()
        .filter(|id| {
            index
                .get(id)
                .and_then(|idx| simple_record(&entities[*idx]))
                .is_some_and(|record| record.name == "SHAPE_DEFINITION_REPRESENTATION")
        })
        .collect::<Vec<_>>();
    let [root_sdr] = sdr_candidates.as_slice() else {
        return None;
    };
    let &root_sdr_index = index.get(root_sdr)?;

    let root_sdr_record = simple_record(&entities[root_sdr_index])?;
    let Parameter::List(root_sdr_params) = &root_sdr_record.parameter else {
        return None;
    };
    if !root_sdr_params
        .iter()
        .any(|param| entity_ref_value(param) == Some(top_rep))
    {
        return None;
    }

    let root_pds = referenced_of_type(*root_sdr, entities, index, &["PRODUCT_DEFINITION_SHAPE"])?;
    let parent_pd = referenced_of_type(root_pds, entities, index, &["PRODUCT_DEFINITION"])?;
    let pd_context = referenced_of_type(
        parent_pd,
        entities,
        index,
        &["PRODUCT_DEFINITION_CONTEXT", "DESIGN_CONTEXT"],
    )?;
    let formation = referenced_of_type(
        parent_pd,
        entities,
        index,
        &[
            "PRODUCT_DEFINITION_FORMATION",
            "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE",
        ],
    )?;
    let parent_product = referenced_of_type(formation, entities, index, &["PRODUCT"])?;
    let product_context = referenced_of_type(
        parent_product,
        entities,
        index,
        &["PRODUCT_CONTEXT", "MECHANICAL_CONTEXT"],
    )?;

    Some(AssemblyProductContext {
        root_sdr_index,
        parent_pd,
        pd_context,
        product_context,
    })
}

fn discover_assembly_family_plan(
    map: u64,
    occurrences: Vec<(u64, u64)>,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    styles_by_target: &HashMap<u64, Vec<StyleRef>>,
    map_to_source: &HashMap<u64, u64>,
    map_to_origin: &HashMap<u64, u64>,
) -> Option<AssemblyFamilyPlan> {
    let source_rep = *map_to_source.get(&map)?;
    let source_origin = *map_to_origin.get(&map)?;
    let &source_index = index.get(&source_rep)?;
    let (source_items, _) = representation_items_and_context(&entities[source_index])?;
    let source_solids = source_items
        .iter()
        .copied()
        .filter(|id| {
            index
                .get(id)
                .and_then(|idx| simple_record(&entities[*idx]))
                .is_some_and(|record| record.name == "MANIFOLD_SOLID_BREP")
        })
        .collect::<Vec<_>>();
    let [canonical_solid] = source_solids.as_slice() else {
        return None;
    };

    let first_mapped = occurrences.first()?.0;
    let mapped_styles = styles_by_target.get(&first_mapped)?;
    if mapped_styles.len() != 1 || mapped_styles[0].assignments.is_empty() {
        return None;
    }
    let inherited_style = mapped_styles[0].assignments.clone();
    if occurrences.iter().any(|(mapped, _)| {
        styles_by_target
            .get(mapped)
            .is_none_or(|styles| styles.len() != 1 || styles[0].assignments != inherited_style)
    }) {
        return None;
    }

    let mut canonical_style_indices = Vec::new();
    if let Some(root_styles) = styles_by_target.get(canonical_solid) {
        canonical_style_indices.reserve(root_styles.len());
        for style in root_styles {
            let &style_index = index.get(&style.id)?;
            let record = simple_record(&entities[style_index])?;
            let Parameter::List(params) = &record.parameter else {
                return None;
            };
            if record.name != "STYLED_ITEM" || params.len() != 3 {
                return None;
            }
            canonical_style_indices.push(style_index);
        }
    }

    Some(AssemblyFamilyPlan {
        map,
        source_rep,
        source_origin,
        canonical_solid: *canonical_solid,
        inherited_style,
        canonical_style_indices,
        occurrences,
    })
}

struct AssemblyEmitter<'a> {
    entities: &'a mut Vec<EntityInstance>,
    next_id: u64,
    new_style_ids: Vec<u64>,
}

#[derive(Clone, Copy)]
struct AssemblyEmissionContext {
    parent_pd: u64,
    pd_context: u64,
    product_context: u64,
    root_rep: u64,
}

impl<'a> AssemblyEmitter<'a> {
    fn new(entities: &'a mut Vec<EntityInstance>) -> Self {
        let next_id = entities.iter().map(entity_id).max().unwrap_or(0) + 1;
        Self {
            entities,
            next_id,
            new_style_ids: Vec::new(),
        }
    }

    fn emit_top(&mut self, rep_ordinal: usize, top: AssemblyTopPlan) -> bool {
        let AssemblyTopPlan {
            top_rep,
            top_rep_index,
            context_id,
            mut residual_items,
            root_axes,
            product,
            families,
        } = top;
        let AssemblyProductContext {
            root_sdr_index,
            parent_pd,
            pd_context,
            product_context,
        } = product;

        let (root_origin, residual_origin) = self.emit_top_origins();
        let has_residual_items = !residual_items.is_empty();
        residual_items.push(residual_origin);
        set_representation_items(&mut self.entities[top_rep_index], &residual_items);

        let mut root_items =
            Vec::with_capacity(1 + usize::from(has_residual_items) + root_axes.len());
        root_items.push(root_origin);
        if has_residual_items {
            root_items.push(residual_origin);
        }
        root_items.extend(root_axes);
        let root_rep = push_simple(
            self.entities,
            &mut self.next_id,
            "SHAPE_REPRESENTATION",
            vec![
                Parameter::String(format!("step-redox assembly {rep_ordinal}")),
                Parameter::List(root_items.into_iter().map(entity_ref).collect()),
                entity_ref(context_id),
            ],
        );
        if !replace_direct_ref_in_simple(&mut self.entities[root_sdr_index], top_rep, root_rep) {
            return false;
        }

        let context = AssemblyEmissionContext {
            parent_pd,
            pd_context,
            product_context,
            root_rep,
        };
        if has_residual_items {
            self.emit_residual_child(rep_ordinal, top_rep, residual_origin, root_origin, context);
        }

        for (family_ordinal, family) in families.iter().enumerate() {
            if !self.emit_family(rep_ordinal, family_ordinal, context, family) {
                return false;
            }
        }
        true
    }

    fn emit_top_origins(&mut self) -> (u64, u64) {
        let z_dir = push_simple(
            self.entities,
            &mut self.next_id,
            "DIRECTION",
            vec![
                Parameter::String(String::new()),
                Parameter::List(vec![
                    Parameter::Real(0.0),
                    Parameter::Real(0.0),
                    Parameter::Real(1.0),
                ]),
            ],
        );
        let x_dir = push_simple(
            self.entities,
            &mut self.next_id,
            "DIRECTION",
            vec![
                Parameter::String(String::new()),
                Parameter::List(vec![
                    Parameter::Real(1.0),
                    Parameter::Real(0.0),
                    Parameter::Real(0.0),
                ]),
            ],
        );
        let root_point = push_point(self.entities, &mut self.next_id, [0.0, 0.0, 0.0]);
        let root_origin = push_simple(
            self.entities,
            &mut self.next_id,
            "AXIS2_PLACEMENT_3D",
            vec![
                Parameter::String(String::new()),
                entity_ref(root_point),
                entity_ref(z_dir),
                entity_ref(x_dir),
            ],
        );
        let residual_point = push_point(self.entities, &mut self.next_id, [0.0, 0.0, 0.0]);
        let residual_origin = push_simple(
            self.entities,
            &mut self.next_id,
            "AXIS2_PLACEMENT_3D",
            vec![
                Parameter::String(String::new()),
                entity_ref(residual_point),
                entity_ref(z_dir),
                entity_ref(x_dir),
            ],
        );
        (root_origin, residual_origin)
    }

    fn emit_residual_child(
        &mut self,
        rep_ordinal: usize,
        top_rep: u64,
        residual_origin: u64,
        root_origin: u64,
        context: AssemblyEmissionContext,
    ) {
        let residual_pd = push_child_product(
            self.entities,
            &mut self.next_id,
            &format!("step-redox residual {rep_ordinal}"),
            context.product_context,
            context.pd_context,
            top_rep,
        );
        push_assembly_occurrence(
            self.entities,
            &mut self.next_id,
            context.parent_pd,
            residual_pd,
            top_rep,
            context.root_rep,
            residual_origin,
            root_origin,
            &format!("residual-{rep_ordinal}"),
        );
    }

    fn emit_family(
        &mut self,
        rep_ordinal: usize,
        family_ordinal: usize,
        context: AssemblyEmissionContext,
        family: &AssemblyFamilyPlan,
    ) -> bool {
        for style_index in family.canonical_style_indices.iter().copied() {
            if !set_styled_item_assignments(
                &mut self.entities[style_index],
                &family.inherited_style,
            ) {
                return false;
            }
        }
        if family.canonical_style_indices.is_empty() {
            let styled = push_simple(
                self.entities,
                &mut self.next_id,
                "STYLED_ITEM",
                vec![
                    Parameter::String("NONE".to_string()),
                    Parameter::List(
                        family
                            .inherited_style
                            .iter()
                            .copied()
                            .map(entity_ref)
                            .collect(),
                    ),
                    entity_ref(family.canonical_solid),
                ],
            );
            self.new_style_ids.push(styled);
        }

        let child_pd = push_child_product(
            self.entities,
            &mut self.next_id,
            &format!("step-redox repeated solid {rep_ordinal}-{family_ordinal}"),
            context.product_context,
            context.pd_context,
            family.source_rep,
        );
        for (occurrence_ordinal, (_, axis)) in family.occurrences.iter().enumerate() {
            push_assembly_occurrence(
                self.entities,
                &mut self.next_id,
                context.parent_pd,
                child_pd,
                family.source_rep,
                context.root_rep,
                family.source_origin,
                *axis,
                &format!("instance-{rep_ordinal}-{family_ordinal}-{occurrence_ordinal}"),
            );
        }
        true
    }
}

fn convert_z90_mapped_items_to_assembly(entities: &mut Vec<EntityInstance>) -> bool {
    let Some(plan) = AssemblyPlan::discover(entities) else {
        return false;
    };
    let AssemblyPlan {
        tops,
        mapped_style_ids,
        mapped_ids,
        used_maps,
    } = plan;

    let new_style_ids = {
        let mut emitter = AssemblyEmitter::new(entities);
        for (rep_ordinal, top) in tops.into_iter().enumerate() {
            if !emitter.emit_top(rep_ordinal, top) {
                return false;
            }
        }
        emitter.new_style_ids
    };

    patch_presentation_lists(entities, &mapped_style_ids, &new_style_ids);

    let mut delete = mapped_style_ids;
    delete.extend(mapped_ids);
    delete.extend(used_maps);
    entities.retain(|entity| !delete.contains(&entity_id(entity)));
    true
}

fn referenced_of_type(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    names: &[&str],
) -> Option<u64> {
    let entity = entities.get(*index.get(&id)?)?;
    let mut found = Vec::new();
    visit_entity_refs(entity, &mut |child| {
        if let Some(record) = index
            .get(&child)
            .and_then(|idx| simple_record(&entities[*idx]))
            && names.iter().any(|name| *name == record.name)
        {
            found.push(child);
        }
    });
    found.sort_unstable();
    found.dedup();
    if found.len() == 1 {
        found.first().copied()
    } else {
        None
    }
}

fn set_representation_items(entity: &mut EntityInstance, items: &[u64]) {
    let Some(record) = simple_record_mut(entity) else {
        return;
    };
    let Parameter::List(params) = &mut record.parameter else {
        return;
    };
    if let Some(slot) = params.get_mut(1) {
        *slot = Parameter::List(items.iter().copied().map(entity_ref).collect());
    }
}

fn replace_direct_ref_in_simple(entity: &mut EntityInstance, old: u64, new: u64) -> bool {
    let Some(record) = simple_record_mut(entity) else {
        return false;
    };
    let Parameter::List(params) = &mut record.parameter else {
        return false;
    };
    let mut changed = false;
    for param in params {
        if entity_ref_value(param) == Some(old) {
            *param = entity_ref(new);
            changed = true;
        }
    }
    changed
}

fn set_styled_item_assignments(entity: &mut EntityInstance, assignments: &[u64]) -> bool {
    let Some(record) = simple_record_mut(entity) else {
        return false;
    };
    if record.name != "STYLED_ITEM" {
        return false;
    }
    let Parameter::List(params) = &mut record.parameter else {
        return false;
    };
    if params.len() != 3 {
        return false;
    }
    params[1] = Parameter::List(assignments.iter().copied().map(entity_ref).collect());
    true
}

fn push_child_product(
    entities: &mut Vec<EntityInstance>,
    next_id: &mut u64,
    name: &str,
    product_context: u64,
    pd_context: u64,
    representation: u64,
) -> u64 {
    let product = push_simple(
        entities,
        next_id,
        "PRODUCT",
        vec![
            Parameter::String(name.to_string()),
            Parameter::String(name.to_string()),
            Parameter::String(String::new()),
            Parameter::List(vec![entity_ref(product_context)]),
        ],
    );
    let formation = push_simple(
        entities,
        next_id,
        "PRODUCT_DEFINITION_FORMATION",
        vec![
            Parameter::String(String::new()),
            Parameter::String(String::new()),
            entity_ref(product),
        ],
    );
    let pd = push_simple(
        entities,
        next_id,
        "PRODUCT_DEFINITION",
        vec![
            Parameter::String("design".to_string()),
            Parameter::String(String::new()),
            entity_ref(formation),
            entity_ref(pd_context),
        ],
    );
    let pds = push_simple(
        entities,
        next_id,
        "PRODUCT_DEFINITION_SHAPE",
        vec![
            Parameter::String(String::new()),
            Parameter::String(String::new()),
            entity_ref(pd),
        ],
    );
    push_simple(
        entities,
        next_id,
        "SHAPE_DEFINITION_REPRESENTATION",
        vec![entity_ref(pds), entity_ref(representation)],
    );
    pd
}

#[expect(
    clippy::too_many_arguments,
    reason = "STEP assembly emission naturally carries the linked entity identifiers as separate arguments"
)]
fn push_assembly_occurrence(
    entities: &mut Vec<EntityInstance>,
    next_id: &mut u64,
    parent_pd: u64,
    child_pd: u64,
    child_rep: u64,
    root_rep: u64,
    source_axis: u64,
    target_axis: u64,
    name: &str,
) {
    let transform = push_simple(
        entities,
        next_id,
        "ITEM_DEFINED_TRANSFORMATION",
        vec![
            Parameter::String(String::new()),
            Parameter::String(String::new()),
            entity_ref(source_axis),
            entity_ref(target_axis),
        ],
    );

    let relationship_id = *next_id;
    *next_id += 1;
    entities.push(EntityInstance::Complex {
        id: relationship_id,
        subsuper: SubSuperRecord(vec![
            Record {
                name: "REPRESENTATION_RELATIONSHIP".to_string(),
                parameter: Parameter::List(vec![
                    Parameter::String(String::new()),
                    Parameter::String(String::new()),
                    entity_ref(child_rep),
                    entity_ref(root_rep),
                ]),
            },
            Record {
                name: "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION".to_string(),
                parameter: Parameter::List(vec![entity_ref(transform)]),
            },
            Record {
                name: "SHAPE_REPRESENTATION_RELATIONSHIP".to_string(),
                parameter: Parameter::List(Vec::new()),
            },
        ]),
    });

    let nauo = push_simple(
        entities,
        next_id,
        "NEXT_ASSEMBLY_USAGE_OCCURRENCE",
        vec![
            Parameter::String(name.to_string()),
            Parameter::String(name.to_string()),
            Parameter::String(String::new()),
            entity_ref(parent_pd),
            entity_ref(child_pd),
            Parameter::NotProvided,
        ],
    );
    let placement = push_simple(
        entities,
        next_id,
        "PRODUCT_DEFINITION_SHAPE",
        vec![
            Parameter::String("Placement".to_string()),
            Parameter::String("Placement of an item".to_string()),
            entity_ref(nauo),
        ],
    );
    push_simple(
        entities,
        next_id,
        "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION",
        vec![entity_ref(relationship_id), entity_ref(placement)],
    );
}

#[derive(Debug, Clone)]
pub struct StyleRef {
    pub(crate) id: u64,
    pub(crate) assignments: Vec<u64>,
}

pub fn collect_styles_by_target(entities: &[EntityInstance]) -> HashMap<u64, Vec<StyleRef>> {
    let mut out: HashMap<u64, Vec<StyleRef>> = HashMap::new();
    for entity in entities {
        let EntityInstance::Simple { id, record } = entity else {
            continue;
        };
        if record.name != "STYLED_ITEM" {
            continue;
        }
        let Parameter::List(params) = &record.parameter else {
            continue;
        };
        if params.len() != 3 {
            continue;
        }
        let Parameter::List(styles) = &params[1] else {
            continue;
        };
        let Some(target) = entity_ref_value(&params[2]) else {
            continue;
        };
        let assignments: Vec<u64> = styles.iter().filter_map(entity_ref_value).collect();
        if assignments.len() != styles.len() || assignments.is_empty() {
            continue;
        }
        out.entry(target).or_default().push(StyleRef {
            id: *id,
            assignments,
        });
    }
    out
}

fn analyze_solid(
    root: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    styles_by_target: &HashMap<u64, Vec<StyleRef>>,
) -> Option<SolidInfo> {
    let identity = solid_identity(root, entities, index)?;
    if identity.vertex_points.len() < 4 {
        return None;
    }

    // Geometric identity is policy-free. Instancing permission adds the
    // presentation invariant that every semantic shell face has one identical
    // explicit style assignment.
    let mut face_style: Option<Vec<u64>> = None;
    for &face in &identity.face_ids {
        let styles = styles_by_target.get(&face)?;
        if styles.len() != 1 {
            return None;
        }
        let assignments = &styles[0].assignments;
        if let Some(existing) = &face_style {
            if existing != assignments {
                return None;
            }
        } else {
            face_style = Some(assignments.clone());
        }
    }

    Some(SolidInfo {
        root: identity.root,
        closure: identity.closure,
        center: identity.center,
        vertex_points: identity.vertex_points,
        key: identity.key,
        face_style: face_style?,
    })
}

fn rigid_translation(source_center: [f64; 3], target_center: [f64; 3], quarter: u8) -> [f64; 3] {
    let (rcx, rcy) = rotate_xy(source_center[0], source_center[1], quarter);
    [
        target_center[0] - rcx,
        target_center[1] - rcy,
        target_center[2] - source_center[2],
    ]
}

fn unique_relative_quarter(
    source: &SolidInfo,
    target: &SolidInfo,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<u8> {
    let target_points = normalized_points(&target.vertex_points, target.center, 0);
    let target_topology = solid_topology_signature(target.root, entities, index, target.center, 0)?;

    let mut matches = Vec::new();
    for quarter in 0..4u8 {
        if normalized_points(&source.vertex_points, source.center, quarter) != target_points {
            continue;
        }
        let source_topology =
            solid_topology_signature(source.root, entities, index, source.center, quarter)?;
        if source_topology == target_topology {
            matches.push(quarter);
        }
    }

    if std::env::var_os("STEP_REDOX_DEBUG_INSTANCES").is_some() {
        eprintln!(
            "instance transform source={} target={} candidates={matches:?} source_center={:?} target_center={:?}",
            source.root, target.root, source.center, target.center
        );
    }

    if matches.len() == 1 {
        Some(matches[0])
    } else {
        None
    }
}

fn replace_representation_items(entity: &mut EntityInstance, replacements: &HashMap<u64, u64>) {
    let Some(record) = simple_record_mut(entity) else {
        return;
    };
    let Parameter::List(params) = &mut record.parameter else {
        return;
    };
    let Some(Parameter::List(items)) = params.get_mut(1) else {
        return;
    };

    for item in items.iter_mut() {
        if let Some(old) = entity_ref_value(item)
            && let Some(&new) = replacements.get(&old)
        {
            *item = entity_ref(new);
        }
    }
}

fn retarget_shape_representation_items(
    entities: &mut [EntityInstance],
    replacements: &HashMap<u64, u64>,
) {
    for entity in entities {
        let is_shape_representation = simple_record(entity)
            .is_some_and(|record| record.name.ends_with("SHAPE_REPRESENTATION"));
        if is_shape_representation {
            replace_representation_items(entity, replacements);
        }
    }
}

#[cfg(test)]
mod tests;
