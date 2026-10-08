use super::*;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum StyleKey {
    Face(Vec<u64>),
    Container,
    None,
}

#[derive(Debug)]
struct MappedFeatureMember {
    candidate: BoundaryFeatureCandidate,
    style_key: StyleKey,
    old_style_ids: Vec<u64>,
}

#[derive(Debug)]
struct MappedFeatureFamilyPlan {
    members: Vec<MappedFeatureMember>,
}

#[derive(Debug)]
struct MappedFeatureHostPlan {
    context: ShellContext,
    host_face_id: u64,
    frame: PlaneFrame,
    container_styles: Vec<StyleRef>,
    families: Vec<MappedFeatureFamilyPlan>,
}

#[derive(Debug, Default)]
struct BoundaryFeatureInstancePlan {
    index: HashMap<u64, usize>,
    next_id: u64,
    hosts: Vec<MappedFeatureHostPlan>,
}

pub fn instance_boundary_features(
    entities: &mut Vec<EntityInstance>,
) -> Result<PlanarFeatureStats> {
    let plan = plan_additive_mapped_features(entities)?;
    apply_mapped_feature_plan(entities, plan)
}

fn plan_additive_mapped_features(
    entities: &[EntityInstance],
) -> Result<BoundaryFeatureInstancePlan> {
    let index = build_index(entities);
    if entities.is_empty() {
        return Ok(BoundaryFeatureInstancePlan {
            index,
            ..BoundaryFeatureInstancePlan::default()
        });
    }

    let styles_by_target = collect_styles_by_target(entities);
    let (_, groups) = grouped_boundary_features_with_index(entities, &index);
    let mut hosts = Vec::<MappedFeatureHostPlan>::new();
    let mut host_positions = HashMap::<(u64, u64, u64, u64, u64), usize>::new();
    let mut claimed_faces = HashSet::<u64>::new();
    let mut claimed_bounds = HashSet::<u64>::new();

    for group in groups {
        let first = &group[0];
        if group.len() < MIN_GROUP
            || first.polarity != FeaturePolarity::Additive
            || first.interfaces.len() != 1
        {
            continue;
        }

        let context = first.context.clone();
        let host_face_id = first.interfaces[0].host_face_id;
        let Some(frame) = plane_frame(host_face_id, entities, &index) else {
            bail!("proven feature host face #{host_face_id} is no longer a readable plane");
        };
        if ref_list_param(context.shell_id, 1, entities, &index).is_none() {
            bail!(
                "feature shell #{} has no editable face list",
                context.shell_id
            );
        }
        if ref_list_param(host_face_id, 1, entities, &index).is_none() {
            bail!("feature host face #{host_face_id} has no editable bound list");
        }
        let Some(&representation_index) = index.get(&context.representation_id) else {
            bail!(
                "feature representation #{} is missing",
                context.representation_id
            );
        };
        if representation_items_and_context(&entities[representation_index]).is_none() {
            bail!(
                "feature representation #{} has no editable item list",
                context.representation_id
            );
        }

        let container_styles = styles_by_target
            .get(&context.container_id)
            .cloned()
            .unwrap_or_default();
        let mut styled = Vec::<MappedFeatureMember>::new();
        for candidate in group {
            let Some((style_key, old_style_ids)) =
                feature_style(&candidate.face_ids, &styles_by_target, &container_styles)
            else {
                continue;
            };
            styled.push(MappedFeatureMember {
                candidate,
                style_key,
                old_style_ids,
            });
        }
        styled.sort_by(|a, b| {
            a.candidate.interfaces[0]
                .bound_orientation
                .cmp(&b.candidate.interfaces[0].bound_orientation)
                .then_with(|| a.style_key.cmp(&b.style_key))
                .then_with(|| a.candidate.face_ids.cmp(&b.candidate.face_ids))
        });

        let mut families = Vec::<MappedFeatureFamilyPlan>::new();
        for member in styled {
            if let Some(family) = families.last_mut()
                && family.members[0].candidate.interfaces[0].bound_orientation
                    == member.candidate.interfaces[0].bound_orientation
                && family.members[0].style_key == member.style_key
            {
                family.members.push(member);
            } else {
                families.push(MappedFeatureFamilyPlan {
                    members: vec![member],
                });
            }
        }
        families.retain(|family| family.members.len() >= MIN_GROUP);
        if families.is_empty() {
            continue;
        }

        for family in &families {
            for member in &family.members {
                if member
                    .candidate
                    .face_ids
                    .iter()
                    .any(|face| !claimed_faces.insert(*face))
                {
                    bail!("boundary feature analysis produced overlapping accepted face patches");
                }
                let bound_id = member.candidate.interfaces[0].bound_id;
                if !claimed_bounds.insert(bound_id) {
                    bail!("boundary feature analysis reused interface bound #{bound_id}");
                }
            }
        }

        let key = (
            context.representation_id,
            context.context_id,
            context.container_id,
            context.shell_id,
            host_face_id,
        );
        if let Some(&position) = host_positions.get(&key) {
            hosts[position].families.extend(families);
        } else {
            let position = hosts.len();
            host_positions.insert(key, position);
            hosts.push(MappedFeatureHostPlan {
                context,
                host_face_id,
                frame,
                container_styles,
                families,
            });
        }
    }

    Ok(BoundaryFeatureInstancePlan {
        index,
        next_id: entities
            .iter()
            .map(entity_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1),
        hosts,
    })
}

fn apply_mapped_feature_plan(
    entities: &mut Vec<EntityInstance>,
    plan: BoundaryFeatureInstancePlan,
) -> Result<PlanarFeatureStats> {
    let BoundaryFeatureInstancePlan {
        index,
        mut next_id,
        hosts,
    } = plan;
    let mut stats = PlanarFeatureStats::default();
    if hosts.is_empty() {
        return Ok(stats);
    }

    let mut candidate_roots = HashSet::new();
    let mut delete_seed = HashSet::new();

    for host in hosts {
        let Some(&representation_index) = index.get(&host.context.representation_id) else {
            bail!("planned feature representation disappeared before apply");
        };
        let Some(&shell_index) = index.get(&host.context.shell_id) else {
            bail!("planned feature shell disappeared before apply");
        };
        let Some(&host_index) = index.get(&host.host_face_id) else {
            bail!("planned feature host disappeared before apply");
        };

        let z_dir = push_direction(entities, &mut next_id, [0.0, 0.0, 1.0]);
        let x_dirs = [
            push_direction(entities, &mut next_id, [1.0, 0.0, 0.0]),
            push_direction(entities, &mut next_id, [0.0, 1.0, 0.0]),
            push_direction(entities, &mut next_id, [-1.0, 0.0, 0.0]),
            push_direction(entities, &mut next_id, [0.0, -1.0, 0.0]),
        ];
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

        let mut mapped_ids = Vec::new();
        let mut remove_faces = HashSet::new();
        let mut remove_bounds = HashSet::new();

        for family in host.families {
            let canonical = &family.members[0];
            let canonical_interface = &canonical.candidate.interfaces[0];
            let cap_bound = push_simple(
                entities,
                &mut next_id,
                "FACE_OUTER_BOUND",
                vec![
                    Parameter::String("NONE".to_string()),
                    entity_ref(canonical_interface.loop_id),
                    Parameter::Enumeration(toggle_tf(&canonical_interface.bound_orientation)),
                ],
            );
            let cap_face = push_simple(
                entities,
                &mut next_id,
                "ADVANCED_FACE",
                vec![
                    Parameter::String("NONE".to_string()),
                    Parameter::List(vec![entity_ref(cap_bound)]),
                    entity_ref(host.frame.surface_id),
                    Parameter::Enumeration(toggle_tf(&host.frame.sense)),
                ],
            );
            let mut closed_faces = canonical
                .candidate
                .face_ids
                .iter()
                .copied()
                .map(entity_ref)
                .collect::<Vec<_>>();
            closed_faces.push(entity_ref(cap_face));
            let feature_shell = push_simple(
                entities,
                &mut next_id,
                "CLOSED_SHELL",
                vec![
                    Parameter::String("NONE".to_string()),
                    Parameter::List(closed_faces),
                ],
            );
            let feature_solid = push_simple(
                entities,
                &mut next_id,
                "MANIFOLD_SOLID_BREP",
                vec![
                    Parameter::String("step-redox canonical boundary feature".to_string()),
                    entity_ref(feature_shell),
                ],
            );
            let source_rep = push_simple(
                entities,
                &mut next_id,
                "ADVANCED_BREP_SHAPE_REPRESENTATION",
                vec![
                    Parameter::String("step-redox boundary feature source".to_string()),
                    Parameter::List(vec![entity_ref(feature_solid), entity_ref(origin_axis)]),
                    entity_ref(host.context.context_id),
                ],
            );
            let rep_map = push_simple(
                entities,
                &mut next_id,
                "REPRESENTATION_MAP",
                vec![entity_ref(origin_axis), entity_ref(source_rep)],
            );

            let mut new_style_ids = Vec::new();
            let mut old_style_ids = HashSet::new();
            for member in &family.members {
                let quarter = (canonical.candidate.normalized_quarter + 4
                    - member.candidate.normalized_quarter)
                    % 4;
                let translation =
                    rigid_translation(canonical.candidate.center, member.candidate.center, quarter);
                let target_axis =
                    if quarter == 0 && translation.iter().all(|value| value.abs() <= 1.0e-12) {
                        origin_axis
                    } else {
                        let point = push_point(entities, &mut next_id, translation);
                        push_simple(
                            entities,
                            &mut next_id,
                            "AXIS2_PLACEMENT_3D",
                            vec![
                                Parameter::String(String::new()),
                                entity_ref(point),
                                entity_ref(z_dir),
                                entity_ref(x_dirs[quarter as usize]),
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
                        entity_ref(target_axis),
                    ],
                );
                mapped_ids.push(mapped);

                match &member.style_key {
                    StyleKey::Face(assignments) => {
                        let styled = push_simple(
                            entities,
                            &mut next_id,
                            "STYLED_ITEM",
                            vec![
                                Parameter::String("NONE".to_string()),
                                Parameter::List(
                                    assignments.iter().copied().map(entity_ref).collect(),
                                ),
                                entity_ref(mapped),
                            ],
                        );
                        new_style_ids.push(styled);
                        old_style_ids.extend(member.old_style_ids.iter().copied());
                    }
                    StyleKey::Container => {
                        for style in &host.container_styles {
                            let styled = push_simple(
                                entities,
                                &mut next_id,
                                "STYLED_ITEM",
                                vec![
                                    Parameter::String("NONE".to_string()),
                                    Parameter::List(
                                        style.assignments.iter().copied().map(entity_ref).collect(),
                                    ),
                                    entity_ref(mapped),
                                ],
                            );
                            new_style_ids.push(styled);
                        }
                    }
                    StyleKey::None => {}
                }

                for &face in &member.candidate.face_ids {
                    remove_faces.insert(face);
                    candidate_roots.insert(face);
                }
                let bound_id = member.candidate.interfaces[0].bound_id;
                remove_bounds.insert(bound_id);
                candidate_roots.insert(bound_id);
                delete_seed.insert(bound_id);
            }
            for member in family.members.iter().skip(1) {
                delete_seed.extend(member.candidate.face_ids.iter().copied());
            }

            if !old_style_ids.is_empty() {
                for &style in &old_style_ids {
                    candidate_roots.insert(style);
                    delete_seed.insert(style);
                }
                patch_presentation_lists(entities, &old_style_ids, &new_style_ids);
            } else if matches!(&canonical.style_key, StyleKey::Container)
                && !new_style_ids.is_empty()
            {
                let anchors = host
                    .container_styles
                    .iter()
                    .map(|style| style.id)
                    .collect::<HashSet<_>>();
                append_presentation_items_with_anchors(entities, &anchors, &new_style_ids);
            }

            stats.families += 1;
            stats.instances += family.members.len();
        }

        if !remove_refs_from_list_param(&mut entities[shell_index], 1, &remove_faces) {
            bail!("planned feature shell edit became invalid during apply");
        }
        if !remove_refs_from_list_param(&mut entities[host_index], 1, &remove_bounds) {
            bail!("planned feature host edit became invalid during apply");
        }
        if !append_refs_to_list_param(&mut entities[representation_index], 1, &mapped_ids) {
            bail!("planned feature representation edit became invalid during apply");
        }
        stats.arrays += 1;
    }

    let delete = collect_detached_feature_entities(entities, &candidate_roots, delete_seed);
    let index_after = build_index(entities);
    stats.styles_replaced = candidate_roots
        .iter()
        .filter(|id| {
            index_after
                .get(id)
                .and_then(|idx| simple_record(&entities[*idx]))
                .is_some_and(|record| record.name == "STYLED_ITEM")
                && delete.contains(id)
        })
        .count();
    stats.entities_removed = delete.len();
    entities.retain(|entity| !delete.contains(&entity_id(entity)));
    Ok(stats)
}

fn collect_detached_feature_entities(
    entities: &[EntityInstance],
    candidate_roots: &HashSet<u64>,
    delete_seed: HashSet<u64>,
) -> HashSet<u64> {
    let index = build_index(entities);
    ReferenceGraph::new(entities).detached_descendant_closure(&index, candidate_roots, delete_seed)
}

fn feature_style(
    face_ids: &[u64],
    styles_by_target: &HashMap<u64, Vec<StyleRef>>,
    container_styles: &[StyleRef],
) -> Option<(StyleKey, Vec<u64>)> {
    let mut any_face_style = false;
    let mut assignments: Option<Vec<u64>> = None;
    let mut style_ids = Vec::new();

    for face in face_ids {
        match styles_by_target.get(face) {
            None => {}
            Some(styles) if styles.is_empty() => {}
            Some(styles) if styles.len() == 1 && !styles[0].assignments.is_empty() => {
                any_face_style = true;
                if let Some(expected) = &assignments {
                    if *expected != styles[0].assignments {
                        return None;
                    }
                } else {
                    assignments = Some(styles[0].assignments.clone());
                }
                style_ids.push(styles[0].id);
            }
            Some(_) => return None,
        }
    }

    if any_face_style {
        if style_ids.len() != face_ids.len() {
            return None;
        }
        Some((StyleKey::Face(assignments?), style_ids))
    } else if !container_styles.is_empty() {
        Some((StyleKey::Container, Vec::new()))
    } else {
        Some((StyleKey::None, Vec::new()))
    }
}

fn push_direction(entities: &mut Vec<EntityInstance>, next_id: &mut u64, value: [f64; 3]) -> u64 {
    push_simple(
        entities,
        next_id,
        "DIRECTION",
        vec![
            Parameter::String(String::new()),
            Parameter::List(value.into_iter().map(Parameter::Real).collect()),
        ],
    )
}

fn append_presentation_items_with_anchors(
    entities: &mut [EntityInstance],
    anchors: &HashSet<u64>,
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
        let has_anchor = items
            .iter()
            .filter_map(entity_ref_value)
            .any(|id| anchors.contains(&id));
        if !has_anchor {
            continue;
        }
        let existing: HashSet<u64> = items.iter().filter_map(entity_ref_value).collect();
        items.extend(
            add.iter()
                .copied()
                .filter(|id| !existing.contains(id))
                .map(entity_ref),
        );
    }
}
