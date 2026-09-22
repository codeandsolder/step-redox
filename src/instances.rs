use ruststep::ast::{EntityInstance, Name, Parameter, Record, SubSuperRecord};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default, Clone)]
pub(crate) struct InstanceStats {
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

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct ShapeKey {
    vertices: usize,
    edges: usize,
    oriented_edges: usize,
    faces: usize,
    points: Vec<[i64; 3]>,
    edge_geometry: Vec<(String, Vec<i64>)>,
    face_geometry: Vec<(String, Vec<i64>)>,
    topology: String,
}

pub(crate) fn instance_z90_solids(entities: &mut Vec<EntityInstance>) -> InstanceStats {
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

    for representation_id in representation_ids {
        let Some(rep_idx) = current_index_of(entities, representation_id) else {
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
                "instance grouping representation={} solids={} analyzed={}",
                representation_id,
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

        let mut groups: HashMap<(ShapeKey, Vec<u64>), Vec<SolidInfo>> = HashMap::new();
        for info in infos {
            groups
                .entry((info.key.clone(), info.face_style.clone()))
                .or_default()
                .push(info);
        }

        for ((_shape_key, face_style), mut group) in groups {
            if group.len() < 2 || face_style.is_empty() {
                continue;
            }

            group.sort_by_key(|info| info.root);
            let canonical = group[0].clone();

            // Do not infer the actual instance transform from the canonical-key
            // rotation. Symmetric envelopes can have several equivalent
            // canonical rotations. Prove each source -> target transform
            // directly against its transformed vertex and B-rep signatures.
            let mut mapped_group = vec![(canonical.clone(), 0u8)];
            for target in group.iter().skip(1) {
                if let Some(quarter) =
                    unique_relative_quarter(&canonical, target, entities, &initial_index)
                {
                    mapped_group.push((target.clone(), quarter));
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
                let relative_quarter = *relative_quarter;
                let axis = if target.root == canonical.root {
                    origin_axis
                } else {
                    let translation =
                        rigid_translation(canonical.center, target.center, relative_quarter);
                    if std::env::var_os("STEP_REDOX_DEBUG_INSTANCES").is_some() {
                        eprintln!(
                            "emit instance source={} target={} quarter={} translation={:?}",
                            canonical.root, target.root, relative_quarter, translation
                        );
                    }
                    let point = push_point(entities, &mut next_id, translation);
                    push_simple(
                        entities,
                        &mut next_id,
                        "AXIS2_PLACEMENT_3D",
                        vec![
                            Parameter::String(String::new()),
                            entity_ref(point),
                            entity_ref(z_dir),
                            entity_ref(x_dirs[relative_quarter as usize]),
                        ],
                    )
                };
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

    // Presentation lists are roots in the SolidWorks files. Remove deleted
    // styled items from them and register the new mapped-item styles.
    patch_presentation_lists(entities, &old_styles_to_remove, &new_style_ids);

    // Reachability-limited garbage collection. Only geometry that was beneath
    // a duplicate solid (plus its old STYLED_ITEM roots) is eligible.
    let mut candidate = geometry_candidates;
    candidate.extend(old_styles_to_remove.iter().copied());

    let refs = entity_ref_map(entities);
    let inbound = inbound_map(&refs);
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


pub(crate) fn instance_z90_solids_assembly(
    entities: &mut Vec<EntityInstance>,
) -> InstanceStats {
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

fn convert_z90_mapped_items_to_assembly(entities: &mut Vec<EntityInstance>) -> bool {
    let index = build_index(entities);
    let refs = entity_ref_map(entities);
    let inbound = inbound_map(&refs);
    let styles_by_target = collect_styles_by_target(entities);

    let source_reps: HashSet<u64> = entities
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
        .collect();
    if source_reps.is_empty() {
        return false;
    }

    let mut map_to_source = HashMap::new();
    let mut map_to_origin = HashMap::new();
    for entity in entities.iter() {
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
        if params.len() != 2 {
            continue;
        }
        let Some(origin) = entity_ref_value(&params[0]) else {
            continue;
        };
        let Some(rep) = entity_ref_value(&params[1]) else {
            continue;
        };
        if source_reps.contains(&rep) {
            map_to_source.insert(id, rep);
            map_to_origin.insert(id, origin);
        }
    }
    if map_to_source.is_empty() {
        return false;
    }

    let mut mapped_info: HashMap<u64, (u64, u64)> = HashMap::new();
    for entity in entities.iter() {
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
        if params.len() != 3 {
            continue;
        }
        let Some(map) = entity_ref_value(&params[1]) else {
            continue;
        };
        let Some(axis) = entity_ref_value(&params[2]) else {
            continue;
        };
        if map_to_source.contains_key(&map) {
            mapped_info.insert(id, (map, axis));
        }
    }
    if mapped_info.is_empty() {
        return false;
    }
    let mapped_ids: HashSet<u64> = mapped_info.keys().copied().collect();

    let top_reps: Vec<u64> = entities
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
            items.iter().any(|item| mapped_ids.contains(item)).then_some(id)
        })
        .collect();
    if top_reps.is_empty() {
        return false;
    }

    // Every generated mapped item must be owned by exactly one top shape rep.
    let mut ownership = HashMap::<u64, usize>::new();
    for &rep in &top_reps {
        let Some(&idx) = index.get(&rep) else {
            return false;
        };
        let Some((items, _)) = representation_items_and_context(&entities[idx]) else {
            return false;
        };
        for item in items {
            if mapped_ids.contains(&item) {
                *ownership.entry(item).or_insert(0) += 1;
            }
        }
    }
    if mapped_ids
        .iter()
        .any(|id| ownership.get(id).copied() != Some(1))
    {
        return false;
    }

    let mapped_style_ids: HashSet<u64> = mapped_ids
        .iter()
        .flat_map(|mapped| {
            styles_by_target
                .get(mapped)
                .into_iter()
                .flatten()
                .map(|style| style.id)
        })
        .collect();

    let mut next_id = entities.iter().map(entity_id).max().unwrap_or(0) + 1;
    let mut new_style_ids = Vec::new();
    let mut used_maps = HashSet::new();

    for (rep_ordinal, &top_rep) in top_reps.iter().enumerate() {
        let Some(&top_idx) = index.get(&top_rep) else {
            return false;
        };
        let Some((top_items, context_id)) =
            representation_items_and_context(&entities[top_idx])
        else {
            return false;
        };
        let local_mapped: Vec<u64> = top_items
            .iter()
            .copied()
            .filter(|item| mapped_ids.contains(item))
            .collect();
        if local_mapped.is_empty() {
            continue;
        }

        let sdr_candidates: Vec<u64> = inbound
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
            .collect();
        if sdr_candidates.len() != 1 {
            return false;
        }
        let root_sdr = sdr_candidates[0];

        let Some(root_pds) = referenced_of_type(
            root_sdr,
            entities,
            &index,
            &["PRODUCT_DEFINITION_SHAPE"],
        ) else {
            return false;
        };
        let Some(parent_pd) =
            referenced_of_type(root_pds, entities, &index, &["PRODUCT_DEFINITION"])
        else {
            return false;
        };
        let Some(pd_context) = referenced_of_type(
            parent_pd,
            entities,
            &index,
            &["PRODUCT_DEFINITION_CONTEXT"],
        ) else {
            return false;
        };
        let Some(formation) = referenced_of_type(
            parent_pd,
            entities,
            &index,
            &[
                "PRODUCT_DEFINITION_FORMATION",
                "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE",
            ],
        ) else {
            return false;
        };
        let Some(parent_product) =
            referenced_of_type(formation, entities, &index, &["PRODUCT"])
        else {
            return false;
        };
        let Some(product_context) =
            referenced_of_type(parent_product, entities, &index, &["PRODUCT_CONTEXT"])
        else {
            return false;
        };

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
        let x_dir = push_simple(
            entities,
            &mut next_id,
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
        let root_point = push_point(entities, &mut next_id, [0.0, 0.0, 0.0]);
        let root_origin = push_simple(
            entities,
            &mut next_id,
            "AXIS2_PLACEMENT_3D",
            vec![
                Parameter::String(String::new()),
                entity_ref(root_point),
                entity_ref(z_dir),
                entity_ref(x_dir),
            ],
        );
        let residual_point = push_point(entities, &mut next_id, [0.0, 0.0, 0.0]);
        let residual_origin = push_simple(
            entities,
            &mut next_id,
            "AXIS2_PLACEMENT_3D",
            vec![
                Parameter::String(String::new()),
                entity_ref(residual_point),
                entity_ref(z_dir),
                entity_ref(x_dir),
            ],
        );

        let residual_items: Vec<u64> = top_items
            .iter()
            .copied()
            .filter(|item| !mapped_ids.contains(item))
            .collect();
        let mut patched_residual_items = residual_items.clone();
        patched_residual_items.push(residual_origin);
        let Some(top_representation) = entities.get_mut(top_idx) else { return false; };
        set_representation_items(top_representation, &patched_residual_items);

        let mut root_items = vec![root_origin];
        if !residual_items.is_empty() {
            root_items.push(residual_origin);
        }
        for mapped in &local_mapped {
            let Some((_, axis)) = mapped_info.get(mapped) else {
                return false;
            };
            if !root_items.contains(axis) {
                root_items.push(*axis);
            }
        }
        let root_rep = push_simple(
            entities,
            &mut next_id,
            "SHAPE_REPRESENTATION",
            vec![
                Parameter::String(format!("step-redox assembly {rep_ordinal}")),
                Parameter::List(root_items.iter().copied().map(entity_ref).collect()),
                entity_ref(context_id),
            ],
        );
        let Some(&root_sdr_index) = index.get(&root_sdr) else { return false; };
        let Some(root_sdr_entity) = entities.get_mut(root_sdr_index) else { return false; };
        if !replace_direct_ref_in_simple(root_sdr_entity, top_rep, root_rep) {
            return false;
        }

        if !residual_items.is_empty() {
            let residual_pd = push_child_product(
                entities,
                &mut next_id,
                &format!("step-redox residual {rep_ordinal}"),
                product_context,
                pd_context,
                top_rep,
            );
            push_assembly_occurrence(
                entities,
                &mut next_id,
                parent_pd,
                residual_pd,
                top_rep,
                root_rep,
                residual_origin,
                root_origin,
                &format!("residual-{rep_ordinal}"),
            );
        }

        let mut local_by_map: HashMap<u64, Vec<(u64, u64)>> = HashMap::new();
        for mapped in &local_mapped {
            let Some(&(map, axis)) = mapped_info.get(mapped) else {
                return false;
            };
            local_by_map.entry(map).or_default().push((*mapped, axis));
        }

        for (family_ordinal, (map, mut occurrences)) in
            local_by_map.into_iter().enumerate()
        {
            occurrences.sort_by_key(|(mapped, _)| *mapped);
            used_maps.insert(map);
            let Some(&source_rep) = map_to_source.get(&map) else {
                return false;
            };
            let Some(&source_origin) = map_to_origin.get(&map) else {
                return false;
            };
            let Some(&source_idx) = index.get(&source_rep) else {
                return false;
            };
            let Some((source_items, _)) =
                representation_items_and_context(&entities[source_idx])
            else {
                return false;
            };
            let source_solids: Vec<u64> = source_items
                .iter()
                .copied()
                .filter(|id| {
                    index
                        .get(id)
                        .and_then(|idx| simple_record(&entities[*idx]))
                        .is_some_and(|record| record.name == "MANIFOLD_SOLID_BREP")
                })
                .collect();
            if source_solids.len() != 1 {
                return false;
            }
            let canonical_solid = source_solids[0];

            let first_mapped = occurrences[0].0;
            let Some(mapped_styles) = styles_by_target.get(&first_mapped) else {
                return false;
            };
            if mapped_styles.len() != 1 || mapped_styles[0].assignments.is_empty() {
                return false;
            }
            let inherited_style = mapped_styles[0].assignments.clone();
            for (mapped, _) in &occurrences {
                let Some(styles) = styles_by_target.get(mapped) else {
                    return false;
                };
                if styles.len() != 1 || styles[0].assignments != inherited_style {
                    return false;
                }
            }

            if let Some(root_styles) = styles_by_target.get(&canonical_solid) {
                for style in root_styles {
                    let Some(&style_idx) = index.get(&style.id) else {
                        return false;
                    };
                    if !set_styled_item_assignments(
                        &mut entities[style_idx],
                        &inherited_style,
                    ) {
                        return false;
                    }
                }
            } else {
                let styled = push_simple(
                    entities,
                    &mut next_id,
                    "STYLED_ITEM",
                    vec![
                        Parameter::String("NONE".to_string()),
                        Parameter::List(
                            inherited_style.iter().copied().map(entity_ref).collect(),
                        ),
                        entity_ref(canonical_solid),
                    ],
                );
                new_style_ids.push(styled);
            }

            let child_pd = push_child_product(
                entities,
                &mut next_id,
                &format!("step-redox repeated solid {rep_ordinal}-{family_ordinal}"),
                product_context,
                pd_context,
                source_rep,
            );

            for (occurrence_ordinal, (_, axis)) in occurrences.iter().enumerate() {
                push_assembly_occurrence(
                    entities,
                    &mut next_id,
                    parent_pd,
                    child_pd,
                    source_rep,
                    root_rep,
                    source_origin,
                    *axis,
                    &format!(
                        "instance-{rep_ordinal}-{family_ordinal}-{occurrence_ordinal}"
                    ),
                );
            }
        }
    }

    // Remove the intermediate occurrence styles from presentation roots and
    // register any new inherited child styles.
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
    (found.len() == 1).then_some(found[0])
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

fn replace_direct_ref_in_simple(
    entity: &mut EntityInstance,
    old: u64,
    new: u64,
) -> bool {
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

fn set_styled_item_assignments(
    entity: &mut EntityInstance,
    assignments: &[u64],
) -> bool {
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

#[allow(clippy::too_many_arguments)]
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
pub(crate) struct StyleRef {
    pub(crate) id: u64,
    pub(crate) assignments: Vec<u64>,
}

pub(crate) fn collect_styles_by_target(entities: &[EntityInstance]) -> HashMap<u64, Vec<StyleRef>> {
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
    let closure = closure_from(root, entities, index);
    let mut faces = Vec::new();
    let mut vertex_points = Vec::new();
    let mut edge_count = 0usize;
    let mut oriented_edge_count = 0usize;
    let mut face_geometry = Vec::new();
    let mut edge_geometry = Vec::new();

    for &id in &closure {
        let &idx = index.get(&id)?;
        let Some(record) = simple_record(&entities[idx]) else {
            continue;
        };
        match record.name.as_str() {
            "ADVANCED_FACE" => {
                faces.push(id);
                let geometry = nth_entity_ref(&record.parameter, 2)?;
                face_geometry.push(geometry_signature(geometry, entities, index)?);
            }
            "EDGE_CURVE" => {
                edge_count += 1;
                let geometry = nth_entity_ref(&record.parameter, 3)?;
                edge_geometry.push(geometry_signature(geometry, entities, index)?);
            }
            "ORIENTED_EDGE" => oriented_edge_count += 1,
            "VERTEX_POINT" => {
                let point = nth_entity_ref(&record.parameter, 1)?;
                vertex_points.push(cartesian_point(point, entities, index)?);
            }
            _ => {}
        }
    }

    if faces.is_empty() || vertex_points.len() < 4 || edge_count == 0 {
        return None;
    }

    // Require every face to carry exactly the same explicit style. This keeps
    // appearance preservation simple and rejects mixed-colour solids.
    let mut face_style: Option<Vec<u64>> = None;
    for &face in &faces {
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

    let center = centroid(&vertex_points);
    let (points, topology, _canonical_rotation) =
        canonical_z90_solid_signature(root, &vertex_points, entities, index, center)?;

    face_geometry.sort();
    edge_geometry.sort();
    let vertex_count = vertex_points.len();

    Some(SolidInfo {
        root,
        closure,
        center,
        vertex_points,
        key: ShapeKey {
            vertices: vertex_count,
            edges: edge_count,
            oriented_edges: oriented_edge_count,
            faces: faces.len(),
            points,
            edge_geometry,
            face_geometry,
            topology,
        },
        face_style: face_style?,
    })
}

fn solid_topology_signature(
    root: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let root_record = simple_record(&entities[*index.get(&root)?])?;
    if root_record.name != "MANIFOLD_SOLID_BREP" {
        return None;
    }
    let shell_id = nth_entity_ref(&root_record.parameter, 1)?;
    let shell_record = simple_record(&entities[*index.get(&shell_id)?])?;
    if shell_record.name != "CLOSED_SHELL" {
        return None;
    }
    let Parameter::List(shell_params) = &shell_record.parameter else {
        return None;
    };
    let Parameter::List(face_refs) = shell_params.get(1)? else {
        return None;
    };

    let mut faces = Vec::with_capacity(face_refs.len());
    for face_ref in face_refs {
        let face_id = entity_ref_value(face_ref)?;
        faces.push(face_topology_signature(
            face_id, entities, index, center, quarter,
        )?);
    }
    faces.sort();
    Some(format!("SHELL[{}]", faces.join("|")))
}

pub(crate) fn face_topology_signature(
    face_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let record = simple_record(&entities[*index.get(&face_id)?])?;
    if record.name != "ADVANCED_FACE" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let Parameter::List(bound_refs) = params.get(1)? else {
        return None;
    };
    let surface_id = entity_ref_value(params.get(2)?)?;
    let same_sense = parameter_literal_signature(params.get(3)?)?;

    let mut bounds = Vec::with_capacity(bound_refs.len());
    for bound_ref in bound_refs {
        bounds.push(bound_topology_signature(
            entity_ref_value(bound_ref)?,
            entities,
            index,
            center,
            quarter,
        )?);
    }
    bounds.sort();

    let surface = support_entity_signature(
        surface_id,
        entities,
        index,
        center,
        quarter,
        &mut HashSet::new(),
        0,
    )?;

    Some(format!(
        "FACE({same_sense};{};{})",
        surface,
        bounds.join("&")
    ))
}

fn bound_topology_signature(
    bound_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let record = simple_record(&entities[*index.get(&bound_id)?])?;
    if record.name != "FACE_OUTER_BOUND" && record.name != "FACE_BOUND" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let loop_id = entity_ref_value(params.get(1)?)?;
    let orientation = parameter_literal_signature(params.get(2)?)?;
    let loop_sig = edge_loop_signature(loop_id, entities, index, center, quarter)?;
    Some(format!("{}({orientation};{loop_sig})", record.name))
}

fn edge_loop_signature(
    loop_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let record = simple_record(&entities[*index.get(&loop_id)?])?;
    if record.name != "EDGE_LOOP" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let Parameter::List(edge_refs) = params.get(1)? else {
        return None;
    };

    let mut uses = Vec::with_capacity(edge_refs.len());
    for edge_ref in edge_refs {
        uses.push(oriented_edge_signature(
            entity_ref_value(edge_ref)?,
            entities,
            index,
            center,
            quarter,
        )?);
    }
    Some(format!("LOOP[{}]", canonical_cycle(&uses).join(">")))
}

fn oriented_edge_signature(
    oriented_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let record = simple_record(&entities[*index.get(&oriented_id)?])?;
    if record.name != "ORIENTED_EDGE" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let edge_id = entity_ref_value(params.get(3)?)?;
    let orientation = parameter_literal_signature(params.get(4)?)?;

    let edge_record = simple_record(&entities[*index.get(&edge_id)?])?;
    if edge_record.name != "EDGE_CURVE" {
        return None;
    }
    let Parameter::List(edge_params) = &edge_record.parameter else {
        return None;
    };
    let start_id = entity_ref_value(edge_params.get(1)?)?;
    let end_id = entity_ref_value(edge_params.get(2)?)?;
    let curve_id = entity_ref_value(edge_params.get(3)?)?;
    let same_sense = parameter_literal_signature(edge_params.get(4)?)?;

    let mut start = vertex_signature(start_id, entities, index, center, quarter)?;
    let mut end = vertex_signature(end_id, entities, index, center, quarter)?;
    if orientation == ".F." {
        std::mem::swap(&mut start, &mut end);
    }

    let curve = support_entity_signature(
        curve_id,
        entities,
        index,
        center,
        quarter,
        &mut HashSet::new(),
        0,
    )?;

    // ORIENTED_EDGE.orientation and EDGE_CURVE.same_sense are two
    // serialization choices describing one semantic relation: whether this
    // edge use traverses the underlying curve in its parameter direction.
    // Equivalent exporters may flip both while swapping the stored edge
    // endpoints. Signature the combined meaning, not the two raw flags.
    let curve_forward = orientation == same_sense;
    Some(format!(
        "OE(CF{};{start}->{end};{curve})",
        if curve_forward { "T" } else { "F" }
    ))
}

fn vertex_signature(
    vertex_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let record = simple_record(&entities[*index.get(&vertex_id)?])?;
    if record.name != "VERTEX_POINT" {
        return None;
    }
    let point_id = nth_entity_ref(&record.parameter, 1)?;
    let point = cartesian_point(point_id, entities, index)?;
    let q = transform_point(point, center, quarter);
    Some(format!("P({},{},{})", q[0], q[1], q[2]))
}

fn support_entity_signature(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
    visiting: &mut HashSet<u64>,
    depth: usize,
) -> Option<String> {
    if depth > 48 || !visiting.insert(id) {
        return None;
    }
    let entity = &entities[*index.get(&id)?];

    let result = match entity {
        EntityInstance::Simple { record, .. } => match record.name.as_str() {
            "CARTESIAN_POINT" => {
                let p = cartesian_point(id, entities, index)?;
                let q = transform_point(p, center, quarter);
                Some(format!("POINT({},{},{})", q[0], q[1], q[2]))
            }
            "DIRECTION" => {
                let d = direction_components(record)?;
                let q = transform_direction(d, quarter);
                Some(format!("DIR({},{},{})", q[0], q[1], q[2]))
            }
            "LINE" => line_support_signature(record, entities, index, center, quarter),
            "PLANE" => plane_support_signature(record, entities, index, center, quarter),
            "CYLINDRICAL_SURFACE" => {
                cylindrical_surface_signature(record, entities, index, center, quarter)
            }
            _ if is_topology_type(&record.name) => None,
            _ => support_record_signature(
                record,
                entities,
                index,
                center,
                quarter,
                visiting,
                depth + 1,
            ),
        },
        EntityInstance::Complex { subsuper, .. } => {
            let mut parts = Vec::with_capacity(subsuper.0.len());
            for record in &subsuper.0 {
                if is_topology_type(&record.name) {
                    return None;
                }
                parts.push(support_record_signature(
                    record,
                    entities,
                    index,
                    center,
                    quarter,
                    visiting,
                    depth + 1,
                )?);
            }
            Some(format!("COMPLEX[{}]", parts.join("|")))
        }
    };
    visiting.remove(&id);
    result
}

fn line_support_signature(
    record: &Record,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let point_id = entity_ref_value(params.get(1)?)?;
    let vector_id = entity_ref_value(params.get(2)?)?;
    let point = cartesian_point(point_id, entities, index)?;

    let vector = simple_record(&entities[*index.get(&vector_id)?])?;
    if vector.name != "VECTOR" {
        return None;
    }
    let Parameter::List(vector_params) = &vector.parameter else {
        return None;
    };
    let direction_id = entity_ref_value(vector_params.get(1)?)?;
    let magnitude = number(vector_params.get(2)?)?;
    let direction_record = simple_record(&entities[*index.get(&direction_id)?])?;
    let direction = direction_components(direction_record)?;

    let offset = canonical_axis_offset(point, direction, center, quarter)?;
    let q_dir = transform_direction(direction, quarter);
    Some(format!(
        "LINE_LOCUS(P({},{},{});DIR({},{},{});MAG{})",
        offset[0],
        offset[1],
        offset[2],
        q_dir[0],
        q_dir[1],
        q_dir[2],
        (magnitude * 1.0e9).round() as i64,
    ))
}

fn cylindrical_surface_signature(
    record: &Record,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let placement_id = entity_ref_value(params.get(1)?)?;
    let radius = number(params.get(2)?)?;
    let placement = simple_record(&entities[*index.get(&placement_id)?])?;
    if placement.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let Parameter::List(place_params) = &placement.parameter else {
        return None;
    };
    let point_id = entity_ref_value(place_params.get(1)?)?;
    let axis_id = entity_ref_value(place_params.get(2)?)?;
    let ref_direction_id = entity_ref_value(place_params.get(3)?)?;
    let point = cartesian_point(point_id, entities, index)?;
    let axis_record = simple_record(&entities[*index.get(&axis_id)?])?;
    let ref_record = simple_record(&entities[*index.get(&ref_direction_id)?])?;
    let axis = direction_components(axis_record)?;
    let ref_direction = direction_components(ref_record)?;

    // Sliding the placement origin along the cylinder axis changes only the
    // parameter-space V origin, not the 3-D cylindrical locus.  Keep axis and
    // reference-direction orientation strict, but compare the perpendicular
    // axis-line offset instead of the exporter's arbitrary point on that line.
    let offset = canonical_axis_offset(point, axis, center, quarter)?;
    let q_axis = transform_direction(axis, quarter);
    let q_ref = transform_direction(ref_direction, quarter);
    Some(format!(
        "CYLINDER_LOCUS(P({},{},{});AXIS({},{},{});REF({},{},{});R{})",
        offset[0],
        offset[1],
        offset[2],
        q_axis[0],
        q_axis[1],
        q_axis[2],
        q_ref[0],
        q_ref[1],
        q_ref[2],
        (radius * 1.0e9).round() as i64,
    ))
}

fn canonical_axis_offset(
    point: [f64; 3],
    direction: [f64; 3],
    center: [f64; 3],
    quarter: u8,
) -> Option<[i64; 3]> {
    let norm2 =
        direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
    if !norm2.is_finite() || norm2 <= 1.0e-24 {
        return None;
    }
    let rel = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2],
    ];
    let along =
        (rel[0] * direction[0] + rel[1] * direction[1] + rel[2] * direction[2]) / norm2;
    let perpendicular = [
        rel[0] - along * direction[0],
        rel[1] - along * direction[1],
        rel[2] - along * direction[2],
    ];
    let (x, y) = rotate_xy(perpendicular[0], perpendicular[1], quarter);
    Some([
        (x * 1.0e5).round() as i64,
        (y * 1.0e5).round() as i64,
        (perpendicular[2] * 1.0e5).round() as i64,
    ])
}

fn plane_support_signature(
    record: &Record,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
) -> Option<String> {
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let placement_id = entity_ref_value(params.get(1)?)?;
    let placement = simple_record(&entities[*index.get(&placement_id)?])?;
    if placement.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let Parameter::List(place_params) = &placement.parameter else {
        return None;
    };
    let point_id = entity_ref_value(place_params.get(1)?)?;
    let axis_id = entity_ref_value(place_params.get(2)?)?;
    let point = cartesian_point(point_id, entities, index)?;
    let axis_record = simple_record(&entities[*index.get(&axis_id)?])?;
    let axis = direction_components(axis_record)?;

    // A plane is invariant to sliding AXIS2_PLACEMENT_3D's origin within
    // itself, and to rotating ref_direction around the normal. Signature the
    // actual oriented geometric locus: normal + signed perpendicular offset.
    let q_axis = transform_direction(axis, quarter);
    let rel = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2],
    ];
    let (rx, ry) = rotate_xy(rel[0], rel[1], quarter);
    let q_rel = [rx, ry, rel[2]];
    let (anx, any) = rotate_xy(axis[0], axis[1], quarter);
    let q_axis_f = [anx, any, axis[2]];
    let offset = (q_rel[0] * q_axis_f[0] + q_rel[1] * q_axis_f[1] + q_rel[2] * q_axis_f[2]) * 1.0e9;
    Some(format!(
        "PLANE(OFFSET{};DIR({},{},{}))",
        offset.round() as i64,
        q_axis[0],
        q_axis[1],
        q_axis[2],
    ))
}

fn support_record_signature(
    record: &Record,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
    visiting: &mut HashSet<u64>,
    depth: usize,
) -> Option<String> {
    let ignored_self_intersect = match record.name.as_str() {
        // self_intersect is exporter metadata, not part of the mathematical
        // B-spline definition. Keep closed-curve/surface flags strict.
        "B_SPLINE_CURVE" => Some(4usize),
        "B_SPLINE_CURVE_WITH_KNOTS" => Some(5usize),
        "B_SPLINE_SURFACE" => Some(6usize),
        "B_SPLINE_SURFACE_WITH_KNOTS" => Some(7usize),
        _ => None,
    };

    let params = if let (Some(ignore), Parameter::List(items)) =
        (ignored_self_intersect, &record.parameter)
    {
        if ignore < items.len() {
            let mut parts = Vec::with_capacity(items.len());
            for (idx, item) in items.iter().enumerate() {
                if idx == ignore {
                    parts.push("SELF_INTERSECT_IGNORED".to_string());
                } else {
                    parts.push(support_param_signature(
                        item,
                        entities,
                        index,
                        center,
                        quarter,
                        visiting,
                        depth + 1,
                    )?);
                }
            }
            format!("({})", parts.join(","))
        } else {
            // Complex STEP entities split inherited B-spline fields across
            // subrecords, so a WITH_KNOTS record may not carry this field.
            support_param_signature(
                &record.parameter,
                entities,
                index,
                center,
                quarter,
                visiting,
                depth + 1,
            )?
        }
    } else {
        support_param_signature(
            &record.parameter,
            entities,
            index,
            center,
            quarter,
            visiting,
            depth + 1,
        )?
    };
    Some(format!("{}{}", record.name, params))
}

fn support_param_signature(
    parameter: &Parameter,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
    quarter: u8,
    visiting: &mut HashSet<u64>,
    depth: usize,
) -> Option<String> {
    match parameter {
        Parameter::Ref(Name::Entity(id)) => {
            support_entity_signature(*id, entities, index, center, quarter, visiting, depth)
        }
        Parameter::Ref(Name::Value(id)) => Some(format!("@{id}")),
        Parameter::Ref(Name::ConstantEntity(value)) => Some(format!("#{value}")),
        Parameter::Ref(Name::ConstantValue(value)) => Some(format!("@{value}")),
        Parameter::Real(value) => Some(format!("R{}", (value * 1.0e9).round() as i64)),
        Parameter::Integer(value) => Some(format!("I{value}")),
        Parameter::String(value) => Some(format!("S{:?}", value)),
        Parameter::Enumeration(value) => Some(format!(".{value}.")),
        Parameter::List(items) => {
            let mut parts = Vec::with_capacity(items.len());
            for item in items {
                parts.push(support_param_signature(
                    item,
                    entities,
                    index,
                    center,
                    quarter,
                    visiting,
                    depth + 1,
                )?);
            }
            Some(format!("({})", parts.join(",")))
        }
        Parameter::Typed { keyword, parameter } => Some(format!(
            "{}({})",
            keyword,
            support_param_signature(
                parameter,
                entities,
                index,
                center,
                quarter,
                visiting,
                depth + 1,
            )?
        )),
        Parameter::NotProvided => Some("$".to_string()),
        Parameter::Omitted => Some("*".to_string()),
    }
}

fn is_topology_type(name: &str) -> bool {
    matches!(
        name,
        "MANIFOLD_SOLID_BREP"
            | "CLOSED_SHELL"
            | "OPEN_SHELL"
            | "ADVANCED_FACE"
            | "FACE_SURFACE"
            | "FACE_OUTER_BOUND"
            | "FACE_BOUND"
            | "EDGE_LOOP"
            | "ORIENTED_EDGE"
            | "EDGE_CURVE"
            | "VERTEX_POINT"
    )
}

fn direction_components(record: &Record) -> Option<[f64; 3]> {
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

fn rigid_translation(source_center: [f64; 3], target_center: [f64; 3], quarter: u8) -> [f64; 3] {
    let (rcx, rcy) = rotate_xy(source_center[0], source_center[1], quarter);
    [
        target_center[0] - rcx,
        target_center[1] - rcy,
        target_center[2] - source_center[2],
    ]
}

fn transform_point(point: [f64; 3], center: [f64; 3], quarter: u8) -> [i64; 3] {
    let x = point[0] - center[0];
    let y = point[1] - center[1];
    let z = point[2] - center[2];
    let (rx, ry) = rotate_xy(x, y, quarter);
    [
        (rx * 1.0e5).round() as i64,
        (ry * 1.0e5).round() as i64,
        (z * 1.0e5).round() as i64,
    ]
}

fn transform_direction(direction: [f64; 3], quarter: u8) -> [i64; 3] {
    let (x, y) = rotate_xy(direction[0], direction[1], quarter);
    [
        (x * 1.0e9).round() as i64,
        (y * 1.0e9).round() as i64,
        (direction[2] * 1.0e9).round() as i64,
    ]
}

fn rotate_xy(x: f64, y: f64, quarter: u8) -> (f64, f64) {
    match quarter % 4 {
        0 => (x, y),
        1 => (-y, x),
        2 => (-x, -y),
        3 => (y, -x),
        _ => unreachable!(),
    }
}

fn parameter_literal_signature(parameter: &Parameter) -> Option<String> {
    match parameter {
        Parameter::Enumeration(value) => Some(format!(".{value}.")),
        Parameter::Integer(value) => Some(value.to_string()),
        // Scalar geometry (radii, lengths, knot literals in geometric
        // signatures) uses the same 1e-5 mm equivalence floor as points.
        Parameter::Real(value) => Some(format!("{}", (value * 1.0e5).round() as i64)),
        Parameter::Omitted => Some("*".to_string()),
        Parameter::NotProvided => Some("$".to_string()),
        _ => None,
    }
}

fn canonical_cycle(items: &[String]) -> Vec<String> {
    if items.is_empty() {
        return Vec::new();
    }
    let mut best = items.to_vec();
    for shift in 1..items.len() {
        let candidate: Vec<String> = (0..items.len())
            .map(|i| items[(i + shift) % items.len()].clone())
            .collect();
        if candidate < best {
            best = candidate;
        }
    }
    best
}

fn geometry_signature(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<(String, Vec<i64>)> {
    let &idx = index.get(&id)?;
    match &entities[idx] {
        EntityInstance::Simple { record, .. } => {
            let mut scalars = Vec::new();
            collect_nonref_scalars(&record.parameter, &mut scalars);
            Some((record.name.clone(), scalars))
        }
        EntityInstance::Complex { subsuper, .. } => {
            // Complex rational B-spline curves/surfaces are normal geometry,
            // not a reason to reject an otherwise instanceable solid.  Keep
            // the same shallow safeguard as for simple supports: entity class
            // plus non-reference scalar parameters.  The full recursive
            // support geometry is proved independently by topology_signature.
            let mut scalars = Vec::new();
            let mut names = String::from("COMPLEX[");
            for (idx, record) in subsuper.0.iter().enumerate() {
                if idx != 0 {
                    names.push('+');
                }
                names.push_str(&record.name);
                collect_nonref_scalars(&record.parameter, &mut scalars);
            }
            names.push(']');
            Some((names, scalars))
        }
    }
}

fn collect_nonref_scalars(param: &Parameter, out: &mut Vec<i64>) {
    match param {
        Parameter::Real(value) => out.push((value * 1.0e9).round() as i64),
        Parameter::Integer(value) => out.push(*value),
        Parameter::List(items) => {
            for item in items {
                collect_nonref_scalars(item, out);
            }
        }
        Parameter::Typed { parameter, .. } => collect_nonref_scalars(parameter, out),
        Parameter::Ref(_)
        | Parameter::String(_)
        | Parameter::Enumeration(_)
        | Parameter::NotProvided
        | Parameter::Omitted => {}
    }
}

fn canonical_z90_solid_signature(
    root: u64,
    points: &[[f64; 3]],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    center: [f64; 3],
) -> Option<(Vec<[i64; 3]>, String, u8)> {
    let mut best: Option<(Vec<[i64; 3]>, String, u8)> = None;
    for quarter in 0..4u8 {
        let point_key = normalized_points(points, center, quarter);
        let topology = solid_topology_signature(root, entities, index, center, quarter)?;
        let candidate = (point_key, topology, quarter);
        if best
            .as_ref()
            .is_none_or(|current| (&candidate.0, &candidate.1) < (&current.0, &current.1))
        {
            best = Some(candidate);
        }
    }
    best
}

fn normalized_points(points: &[[f64; 3]], center: [f64; 3], quarter: u8) -> Vec<[i64; 3]> {
    let mut candidate: Vec<[i64; 3]> = points
        .iter()
        .map(|point| transform_point(*point, center, quarter))
        .collect();
    candidate.sort_unstable();
    candidate
}

#[cfg(test)]
fn canonical_z90_points(points: &[[f64; 3]], center: [f64; 3]) -> (Vec<[i64; 3]>, u8) {
    let mut best: Option<Vec<[i64; 3]>> = None;
    let mut best_rotation = 0u8;
    for quarter in 0..4u8 {
        let candidate = normalized_points(points, center, quarter);
        if best.as_ref().is_none_or(|current| candidate < *current) {
            best = Some(candidate);
            best_rotation = quarter;
        }
    }
    (best.unwrap_or_default(), best_rotation)
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
            "instance transform source={} target={} candidates={:?} source_center={:?} target_center={:?}",
            source.root, target.root, matches, source.center, target.center
        );
    }

    if matches.len() == 1 {
        Some(matches[0])
    } else {
        None
    }
}

fn centroid(points: &[[f64; 3]]) -> [f64; 3] {
    let mut center = [0.0; 3];
    for point in points {
        center[0] += point[0];
        center[1] += point[1];
        center[2] += point[2];
    }
    let n = points.len() as f64;
    [center[0] / n, center[1] / n, center[2] / n]
}

pub(crate) fn cartesian_point(
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

pub(crate) fn number(param: &Parameter) -> Option<f64> {
    match param {
        Parameter::Real(value) => Some(*value),
        Parameter::Integer(value) => Some(*value as f64),
        _ => None,
    }
}

pub(crate) fn nth_entity_ref(parameter: &Parameter, idx: usize) -> Option<u64> {
    let Parameter::List(params) = parameter else {
        return None;
    };
    entity_ref_value(params.get(idx)?)
}

pub(crate) fn entity_ref_value(param: &Parameter) -> Option<u64> {
    match param {
        Parameter::Ref(Name::Entity(id)) => Some(*id),
        _ => None,
    }
}

pub(crate) fn closure_from(
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

pub(crate) fn representation_items_and_context(entity: &EntityInstance) -> Option<(Vec<u64>, u64)> {
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

pub(crate) fn patch_presentation_lists(
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

pub(crate) fn entity_ref_map(entities: &[EntityInstance]) -> HashMap<u64, Vec<u64>> {
    entities
        .iter()
        .map(|entity| {
            let mut refs = Vec::new();
            visit_entity_refs(entity, &mut |id| refs.push(id));
            (entity_id(entity), refs)
        })
        .collect()
}

pub(crate) fn inbound_map(refs: &HashMap<u64, Vec<u64>>) -> HashMap<u64, HashSet<u64>> {
    let mut inbound: HashMap<u64, HashSet<u64>> = HashMap::new();
    for (&parent, children) in refs {
        for &child in children {
            inbound.entry(child).or_default().insert(parent);
        }
    }
    inbound
}

pub(crate) fn build_index(entities: &[EntityInstance]) -> HashMap<u64, usize> {
    entities
        .iter()
        .enumerate()
        .map(|(idx, entity)| (entity_id(entity), idx))
        .collect()
}

pub(crate) fn current_index_of(entities: &[EntityInstance], id: u64) -> Option<usize> {
    entities.iter().position(|entity| entity_id(entity) == id)
}

pub(crate) fn simple_record(entity: &EntityInstance) -> Option<&Record> {
    match entity {
        EntityInstance::Simple { record, .. } => Some(record),
        EntityInstance::Complex { .. } => None,
    }
}

pub(crate) fn simple_record_mut(entity: &mut EntityInstance) -> Option<&mut Record> {
    match entity {
        EntityInstance::Simple { record, .. } => Some(record),
        EntityInstance::Complex { .. } => None,
    }
}

pub(crate) fn push_point(
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

pub(crate) fn push_simple(
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

pub(crate) fn entity_ref(id: u64) -> Parameter {
    Parameter::Ref(Name::Entity(id))
}

pub(crate) fn entity_id(entity: &EntityInstance) -> u64 {
    match entity {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id,
    }
}

pub(crate) fn visit_entity_refs(entity: &EntityInstance, f: &mut impl FnMut(u64)) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z90_signature_ignores_quarter_turn_and_translation() {
        let a = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 0.0, 3.0],
        ];
        let b: Vec<[f64; 3]> = a
            .iter()
            .map(|p| [10.0 - p[1], -4.0 + p[0], 2.0 + p[2]])
            .collect();
        let (ka, _) = canonical_z90_points(&a, centroid(&a));
        let (kb, _) = canonical_z90_points(&b, centroid(&b));
        assert_eq!(ka, kb);
    }

    #[test]
    fn rigid_translation_maps_source_center_after_rotation() {
        let source = [2.857_319_490_003_104_3, -0.645, 0.588_924_731_498_555_2];
        let target = [-2.857_319_490_003_104_7, 0.625, 0.588_924_731_498_555_1];
        let d = rigid_translation(source, target, 2);
        let (rx, ry) = rotate_xy(source[0], source[1], 2);
        let mapped = [rx + d[0], ry + d[1], source[2] + d[2]];
        for axis in 0..3 {
            assert!((mapped[axis] - target[axis]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn z90_signature_rejects_shape_change() {
        let a = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let b = vec![[0.0, 0.0, 0.0], [2.1, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let (ka, _) = canonical_z90_points(&a, centroid(&a));
        let (kb, _) = canonical_z90_points(&b, centroid(&b));
        assert_ne!(ka, kb);
    }

    #[test]
    fn axis_offset_ignores_slide_along_axis() {
        let center = [0.0, 0.0, 0.0];
        let axis = [0.0, 2.0, 0.0];
        let a = canonical_axis_offset([1.25, -7.0, 3.5], axis, center, 0).unwrap();
        let b = canonical_axis_offset([1.25, 42.0, 3.5], axis, center, 0).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, [125_000, 0, 350_000]);
    }

    #[test]
    fn axis_offset_rotates_with_solid_quarter_turn() {
        let center = [0.0, 0.0, 0.0];
        let axis = [0.0, 0.0, 1.0];
        let a = canonical_axis_offset([2.0, 1.0, 9.0], axis, center, 1).unwrap();
        assert_eq!(a, [-100_000, 200_000, 0]);
    }
}
