use crate::brep::{
    append_refs_to_list_param, bound_loop_edges, face_edge_curves, face_sense, face_surface,
    manifold_shell, ref_list_param, remove_refs_from_list_param, toggle_tf,
};
use crate::instances::{
    StyleRef, build_index, cartesian_point, collect_styles_by_target, entity_id, entity_ref,
    entity_ref_map, entity_ref_value, face_topology_signature, inbound_map,
    patch_presentation_lists, push_point, push_simple, representation_items_and_context,
    simple_record, simple_record_mut, visit_entity_refs,
};
use ruststep::ast::{EntityInstance, Parameter};
use std::collections::{HashMap, HashSet, VecDeque};

const MIN_GROUP: usize = 8;
const MAX_FEATURE_FACES: usize = 128;
const SIDE_TOLERANCE: f64 = 1.0e-8;

#[derive(Debug, Default, Clone)]
pub(crate) struct PlanarFeatureStats {
    pub arrays: usize,
    pub families: usize,
    pub instances: usize,
    pub entities_removed: usize,
    pub styles_replaced: usize,
}

#[derive(Debug, Clone)]
struct ShellContext {
    representation_id: u64,
    context_id: u64,
    container_id: u64,
    shell_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum StyleKey {
    Face(Vec<u64>),
    Container,
    None,
}

#[derive(Debug, Clone)]
struct Feature {
    faces: Vec<u64>,
    interface_bound: u64,
    interface_loop: u64,
    bound_orientation: String,
    center: [f64; 3],
    normalized_quarter: u8,
    signature: String,
    style_key: StyleKey,
    old_style_ids: Vec<u64>,
}

#[derive(Debug, Clone)]
struct PlaneFrame {
    surface_id: u64,
    sense: String,
    origin: [f64; 3],
    outward: [f64; 3],
}

pub(crate) fn instance_planar_positive_features(
    entities: &mut Vec<EntityInstance>,
) -> PlanarFeatureStats {
    let mut stats = PlanarFeatureStats::default();
    if entities.is_empty() {
        return stats;
    }

    let index = build_index(entities);
    let styles_by_target = collect_styles_by_target(entities);
    let contexts = collect_shell_contexts(entities, &index);
    if contexts.is_empty() {
        return stats;
    }

    let mut next_id = entities.iter().map(entity_id).max().unwrap_or(0) + 1;
    let mut candidate_roots = HashSet::new();
    let mut delete_seed = HashSet::new();
    let mut claimed_faces = HashSet::new();
    let mut claimed_bounds = HashSet::new();

    for context in contexts {
        let Some(shell_faces) = ref_list_param(context.shell_id, 1, entities, &index) else {
            continue;
        };
        if shell_faces.len() < MIN_GROUP + 2 {
            continue;
        }

        let mut face_edges = HashMap::<u64, HashSet<u64>>::new();
        let mut edge_faces = HashMap::<u64, Vec<u64>>::new();
        for &face in &shell_faces {
            let Some(edges) = face_edge_curves(face, entities, &index) else {
                continue;
            };
            for &edge in &edges {
                edge_faces.entry(edge).or_default().push(face);
            }
            face_edges.insert(face, edges);
        }

        let mut adjacency = HashMap::<u64, Vec<u64>>::new();
        for &face in &shell_faces {
            adjacency.entry(face).or_default();
        }
        for attached in edge_faces.values() {
            if attached.len() < 2 {
                continue;
            }
            for &face in attached {
                let out = adjacency.entry(face).or_default();
                out.extend(attached.iter().copied().filter(|other| *other != face));
            }
        }
        for neighbors in adjacency.values_mut() {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        let container_styles = styles_by_target
            .get(&context.container_id)
            .cloned()
            .unwrap_or_default();

        let host_faces: Vec<u64> = shell_faces
            .iter()
            .copied()
            .filter(|face| {
                face_surface(*face, entities, &index)
                    .and_then(|surface| index.get(&surface).copied())
                    .and_then(|idx| simple_record(&entities[idx]))
                    .is_some_and(|record| record.name == "PLANE")
                    && ref_list_param(*face, 1, entities, &index)
                        .is_some_and(|bounds| bounds.len() > MIN_GROUP)
            })
            .collect();

        for host_face in host_faces {
            let Some(frame) = plane_frame(host_face, entities, &index) else {
                continue;
            };
            let Some(host_bounds) = ref_list_param(host_face, 1, entities, &index) else {
                continue;
            };
            let bound_lookup = host_bound_lookup(&host_bounds, entities, &index);
            let components = face_components_without_host(&shell_faces, host_face, &adjacency);
            let mut features = Vec::new();

            for component in components {
                if component.is_empty()
                    || component.len() > MAX_FEATURE_FACES
                    || component.iter().any(|face| claimed_faces.contains(face))
                {
                    continue;
                }

                if !component_is_two_manifold_with_host(
                    &component,
                    host_face,
                    &face_edges,
                    &edge_faces,
                ) {
                    continue;
                }
                let interface_edges =
                    component_interface_edges(&component, host_face, &face_edges, &edge_faces);
                if interface_edges.is_empty() {
                    continue;
                }
                let Some(matches) = bound_lookup.get(&edge_set_key(&interface_edges)) else {
                    continue;
                };
                if matches.len() != 1 {
                    continue;
                }
                let (interface_bound, interface_loop, bound_orientation) = matches[0].clone();
                if claimed_bounds.contains(&interface_bound) {
                    continue;
                }

                let Some(vertices) =
                    component_vertex_points(&component, &face_edges, entities, &index)
                else {
                    continue;
                };
                if !positive_side(&vertices, &frame) {
                    continue;
                }
                let Some(center) = bbox_center(&vertices) else {
                    continue;
                };
                let Some((signature, normalized_quarter)) =
                    component_signature(&component, center, entities, &index)
                else {
                    continue;
                };
                let Some((style_key, old_style_ids)) =
                    feature_style(&component, &styles_by_target, &container_styles)
                else {
                    continue;
                };

                let mut faces: Vec<u64> = component.into_iter().collect();
                faces.sort_unstable();
                features.push(Feature {
                    faces,
                    interface_bound,
                    interface_loop,
                    bound_orientation,
                    center,
                    normalized_quarter,
                    signature,
                    style_key,
                    old_style_ids,
                });
            }

            if features.len() < MIN_GROUP {
                continue;
            }

            let mut groups = HashMap::<(String, String, StyleKey), Vec<Feature>>::new();
            for feature in features {
                groups
                    .entry((
                        feature.signature.clone(),
                        feature.bound_orientation.clone(),
                        feature.style_key.clone(),
                    ))
                    .or_default()
                    .push(feature);
            }

            let mut accepted: Vec<Vec<Feature>> = groups
                .into_values()
                .filter(|group| group.len() >= MIN_GROUP)
                .collect();
            if accepted.is_empty() {
                continue;
            }
            accepted.sort_by_key(|group| {
                group
                    .iter()
                    .map(|feature| feature.faces[0])
                    .min()
                    .unwrap_or(u64::MAX)
            });

            let Some(&rep_idx) = index.get(&context.representation_id) else {
                continue;
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

            let mut host_mapped_ids = Vec::new();
            let mut host_remove_faces = HashSet::new();
            let mut host_remove_bounds = HashSet::new();
            let mut host_family_count = 0usize;

            for group in &mut accepted {
                group.sort_by_key(|feature| feature.faces.clone());
                let canonical = group[0].clone();

                let disk_bound = push_simple(
                    entities,
                    &mut next_id,
                    "FACE_OUTER_BOUND",
                    vec![
                        Parameter::String("NONE".to_string()),
                        entity_ref(canonical.interface_loop),
                        Parameter::Enumeration(toggle_tf(&canonical.bound_orientation)),
                    ],
                );
                let disk_face = push_simple(
                    entities,
                    &mut next_id,
                    "ADVANCED_FACE",
                    vec![
                        Parameter::String("NONE".to_string()),
                        Parameter::List(vec![entity_ref(disk_bound)]),
                        entity_ref(frame.surface_id),
                        Parameter::Enumeration(toggle_tf(&frame.sense)),
                    ],
                );

                let mut closed_faces: Vec<Parameter> =
                    canonical.faces.iter().copied().map(entity_ref).collect();
                closed_faces.push(entity_ref(disk_face));
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
                        Parameter::String(
                            "step-redox canonical planar positive feature".to_string(),
                        ),
                        entity_ref(feature_shell),
                    ],
                );
                let source_rep = push_simple(
                    entities,
                    &mut next_id,
                    "ADVANCED_BREP_SHAPE_REPRESENTATION",
                    vec![
                        Parameter::String("step-redox planar positive feature source".to_string()),
                        Parameter::List(vec![entity_ref(feature_solid), entity_ref(origin_axis)]),
                        entity_ref(context.context_id),
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

                for feature in group.iter() {
                    let quarter =
                        (canonical.normalized_quarter + 4 - feature.normalized_quarter) % 4;
                    let translation = rigid_translation(canonical.center, feature.center, quarter);
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
                    host_mapped_ids.push(mapped);

                    match &feature.style_key {
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
                            old_style_ids.extend(feature.old_style_ids.iter().copied());
                        }
                        StyleKey::Container => {
                            for style in &container_styles {
                                let styled = push_simple(
                                    entities,
                                    &mut next_id,
                                    "STYLED_ITEM",
                                    vec![
                                        Parameter::String("NONE".to_string()),
                                        Parameter::List(
                                            style
                                                .assignments
                                                .iter()
                                                .copied()
                                                .map(entity_ref)
                                                .collect(),
                                        ),
                                        entity_ref(mapped),
                                    ],
                                );
                                new_style_ids.push(styled);
                            }
                        }
                        StyleKey::None => {}
                    }

                    for &face in &feature.faces {
                        host_remove_faces.insert(face);
                        candidate_roots.insert(face);
                    }
                    host_remove_bounds.insert(feature.interface_bound);
                    candidate_roots.insert(feature.interface_bound);
                    delete_seed.insert(feature.interface_bound);
                    claimed_faces.extend(feature.faces.iter().copied());
                    claimed_bounds.insert(feature.interface_bound);
                }

                for feature in group.iter().skip(1) {
                    delete_seed.extend(feature.faces.iter().copied());
                }

                if !old_style_ids.is_empty() {
                    for &style in &old_style_ids {
                        candidate_roots.insert(style);
                        delete_seed.insert(style);
                    }
                    patch_presentation_lists(entities, &old_style_ids, &new_style_ids);
                } else if matches!(canonical.style_key, StyleKey::Container)
                    && !new_style_ids.is_empty()
                {
                    let anchors: HashSet<u64> =
                        container_styles.iter().map(|style| style.id).collect();
                    append_presentation_items_with_anchors(entities, &anchors, &new_style_ids);
                }

                host_family_count += 1;
                stats.instances += group.len();
            }

            if host_remove_faces.is_empty() {
                continue;
            }

            let Some(&shell_idx) = index.get(&context.shell_id) else {
                continue;
            };
            let Some(&host_idx) = index.get(&host_face) else {
                continue;
            };
            if !remove_refs_from_list_param(&mut entities[shell_idx], 1, &host_remove_faces) {
                continue;
            }
            if !remove_refs_from_list_param(&mut entities[host_idx], 1, &host_remove_bounds) {
                continue;
            }
            if !append_refs_to_list_param(&mut entities[rep_idx], 1, &host_mapped_ids) {
                continue;
            }

            stats.arrays += 1;
            stats.families += host_family_count;
        }
    }

    if stats.arrays == 0 {
        return stats;
    }

    let index_after = build_index(entities);
    let mut candidate = HashSet::new();
    let mut stack: Vec<u64> = candidate_roots.iter().copied().collect();
    while let Some(id) = stack.pop() {
        if !candidate.insert(id) {
            continue;
        }
        let Some(&idx) = index_after.get(&id) else {
            continue;
        };
        visit_entity_refs(&entities[idx], &mut |child| {
            if index_after.contains_key(&child) && !candidate.contains(&child) {
                stack.push(child);
            }
        });
    }

    let refs = entity_ref_map(entities);
    let inbound = inbound_map(&refs);
    let mut delete = delete_seed;

    loop {
        let mut changed = false;
        for &id in &candidate {
            if delete.contains(&id) {
                continue;
            }
            let parents = inbound.get(&id);
            let all_dead =
                parents.is_none_or(|parents| parents.iter().all(|parent| delete.contains(parent)));
            let child_of_dead = parents.is_some_and(|parents| {
                !parents.is_empty() && parents.iter().all(|parent| delete.contains(parent))
            });
            if all_dead && (child_of_dead || !inbound.contains_key(&id)) {
                delete.insert(id);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    for (id, children) in &refs {
        if delete.contains(id) {
            continue;
        }
        debug_assert!(children.iter().all(|child| !delete.contains(child)));
    }

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
    stats
}

fn collect_shell_contexts(
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Vec<ShellContext> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();

    for entity in entities {
        let Some(record) = simple_record(entity) else {
            continue;
        };
        if record.name != "ADVANCED_BREP_SHAPE_REPRESENTATION"
            && record.name != "MANIFOLD_SURFACE_SHAPE_REPRESENTATION"
        {
            continue;
        }
        let representation_id = entity_id(entity);
        let Some((items, context_id)) = representation_items_and_context(entity) else {
            continue;
        };

        for item in items {
            let Some(&item_idx) = index.get(&item) else {
                continue;
            };
            let Some(item_record) = simple_record(&entities[item_idx]) else {
                continue;
            };
            match item_record.name.as_str() {
                "MANIFOLD_SOLID_BREP" => {
                    let Some(shell_id) = manifold_shell(item, entities, index) else {
                        continue;
                    };
                    if is_closed_shell(shell_id, entities, index)
                        && seen.insert((representation_id, item, shell_id))
                    {
                        out.push(ShellContext {
                            representation_id,
                            context_id,
                            container_id: item,
                            shell_id,
                        });
                    }
                }
                "SHELL_BASED_SURFACE_MODEL" => {
                    let Some(shells) = ref_list_param(item, 1, entities, index) else {
                        continue;
                    };
                    for shell_id in shells {
                        if is_closed_shell(shell_id, entities, index)
                            && seen.insert((representation_id, item, shell_id))
                        {
                            out.push(ShellContext {
                                representation_id,
                                context_id,
                                container_id: item,
                                shell_id,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
    }
    out
}

fn is_closed_shell(
    shell_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> bool {
    index
        .get(&shell_id)
        .and_then(|idx| simple_record(&entities[*idx]))
        .is_some_and(|record| record.name == "CLOSED_SHELL")
}

fn host_bound_lookup(
    bounds: &[u64],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> HashMap<Vec<u64>, Vec<(u64, u64, String)>> {
    let mut out = HashMap::<Vec<u64>, Vec<(u64, u64, String)>>::new();
    for &bound in bounds {
        let Some(record) = index
            .get(&bound)
            .and_then(|idx| simple_record(&entities[*idx]))
        else {
            continue;
        };
        if record.name != "FACE_BOUND" {
            continue;
        }
        let Some((loop_id, edges, orientation)) = bound_loop_edges(bound, entities, index) else {
            continue;
        };
        out.entry(edge_set_key(&edges))
            .or_default()
            .push((bound, loop_id, orientation));
    }
    out
}

fn edge_set_key(edges: &HashSet<u64>) -> Vec<u64> {
    let mut key: Vec<u64> = edges.iter().copied().collect();
    key.sort_unstable();
    key
}

fn face_components_without_host(
    faces: &[u64],
    host: u64,
    adjacency: &HashMap<u64, Vec<u64>>,
) -> Vec<HashSet<u64>> {
    let mut remaining: HashSet<u64> = faces.iter().copied().filter(|face| *face != host).collect();
    let mut out = Vec::new();

    while let Some(&seed) = remaining.iter().next() {
        remaining.remove(&seed);
        let mut component = HashSet::from([seed]);
        let mut queue = VecDeque::from([seed]);
        while let Some(face) = queue.pop_front() {
            for &neighbor in adjacency.get(&face).into_iter().flatten() {
                if neighbor == host || !remaining.remove(&neighbor) {
                    continue;
                }
                component.insert(neighbor);
                queue.push_back(neighbor);
            }
        }
        out.push(component);
    }
    out
}

fn component_is_two_manifold_with_host(
    component: &HashSet<u64>,
    host: u64,
    face_edges: &HashMap<u64, HashSet<u64>>,
    edge_faces: &HashMap<u64, Vec<u64>>,
) -> bool {
    let mut edges = HashSet::new();
    for face in component {
        let Some(face_edges) = face_edges.get(face) else {
            return false;
        };
        edges.extend(face_edges.iter().copied());
    }

    edges.into_iter().all(|edge| {
        let Some(attached) = edge_faces.get(&edge) else {
            return false;
        };
        if attached.len() != 2 {
            return false;
        }
        attached
            .iter()
            .all(|face| *face == host || component.contains(face))
            && attached.iter().filter(|face| **face == host).count() <= 1
    })
}

fn component_interface_edges(
    component: &HashSet<u64>,
    host: u64,
    face_edges: &HashMap<u64, HashSet<u64>>,
    edge_faces: &HashMap<u64, Vec<u64>>,
) -> HashSet<u64> {
    let mut out = HashSet::new();
    for face in component {
        for edge in face_edges.get(face).into_iter().flatten() {
            if edge_faces
                .get(edge)
                .is_some_and(|faces| faces.contains(&host))
            {
                out.insert(*edge);
            }
        }
    }
    out
}

fn component_vertex_points(
    component: &HashSet<u64>,
    face_edges: &HashMap<u64, HashSet<u64>>,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<[f64; 3]>> {
    let mut edge_ids = HashSet::new();
    for face in component {
        edge_ids.extend(face_edges.get(face)?.iter().copied());
    }

    let mut vertex_ids = HashSet::new();
    for edge in edge_ids {
        let record = simple_record(&entities[*index.get(&edge)?])?;
        if record.name != "EDGE_CURVE" {
            return None;
        }
        let Parameter::List(params) = &record.parameter else {
            return None;
        };
        vertex_ids.insert(entity_ref_value(params.get(1)?)?);
        vertex_ids.insert(entity_ref_value(params.get(2)?)?);
    }

    let mut out = Vec::with_capacity(vertex_ids.len());
    for vertex in vertex_ids {
        let record = simple_record(&entities[*index.get(&vertex)?])?;
        if record.name != "VERTEX_POINT" {
            return None;
        }
        let Parameter::List(params) = &record.parameter else {
            return None;
        };
        let point_id = entity_ref_value(params.get(1)?)?;
        out.push(cartesian_point(point_id, entities, index)?);
    }
    Some(out)
}

fn bbox_center(points: &[[f64; 3]]) -> Option<[f64; 3]> {
    let first = *points.first()?;
    let mut min = first;
    let mut max = first;
    for point in points.iter().skip(1) {
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    Some([
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ])
}

fn plane_frame(
    host_face: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<PlaneFrame> {
    let surface_id = face_surface(host_face, entities, index)?;
    let surface = simple_record(&entities[*index.get(&surface_id)?])?;
    if surface.name != "PLANE" {
        return None;
    }
    let Parameter::List(surface_params) = &surface.parameter else {
        return None;
    };
    let axis_id = entity_ref_value(surface_params.get(1)?)?;
    let axis = simple_record(&entities[*index.get(&axis_id)?])?;
    if axis.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let Parameter::List(axis_params) = &axis.parameter else {
        return None;
    };
    let origin_id = entity_ref_value(axis_params.get(1)?)?;
    let direction_id = entity_ref_value(axis_params.get(2)?)?;
    let origin = cartesian_point(origin_id, entities, index)?;
    let direction = direction_components(direction_id, entities, index)?;
    let length =
        (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
            .sqrt();
    if !length.is_finite() || length <= 0.0 {
        return None;
    }
    let sense = face_sense(host_face, entities, index)?;
    let sign = match sense.as_str() {
        "T" => 1.0,
        "F" => -1.0,
        _ => return None,
    };
    Some(PlaneFrame {
        surface_id,
        sense,
        origin,
        outward: [
            sign * direction[0] / length,
            sign * direction[1] / length,
            sign * direction[2] / length,
        ],
    })
}

fn direction_components(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let record = simple_record(&entities[*index.get(&id)?])?;
    if record.name != "DIRECTION" {
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
        numeric(&coords[0])?,
        numeric(&coords[1])?,
        numeric(&coords[2])?,
    ])
}

fn numeric(parameter: &Parameter) -> Option<f64> {
    match parameter {
        Parameter::Real(value) => Some(*value),
        Parameter::Integer(value) => Some(*value as f64),
        _ => None,
    }
}

fn positive_side(points: &[[f64; 3]], frame: &PlaneFrame) -> bool {
    let mut max_projection = f64::NEG_INFINITY;
    for point in points {
        let delta = [
            point[0] - frame.origin[0],
            point[1] - frame.origin[1],
            point[2] - frame.origin[2],
        ];
        let projection =
            delta[0] * frame.outward[0] + delta[1] * frame.outward[1] + delta[2] * frame.outward[2];
        if !projection.is_finite() || projection < -SIDE_TOLERANCE {
            return false;
        }
        max_projection = max_projection.max(projection);
    }
    max_projection > SIDE_TOLERANCE
}

fn component_signature(
    component: &HashSet<u64>,
    center: [f64; 3],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<(String, u8)> {
    let mut best: Option<(String, u8)> = None;
    for quarter in 0..4u8 {
        let mut faces = Vec::with_capacity(component.len());
        for &face in component {
            faces.push(face_topology_signature(
                face, entities, index, center, quarter,
            )?);
        }
        faces.sort();
        let signature = format!("F{}[{}]", component.len(), faces.join("|"));
        if best
            .as_ref()
            .is_none_or(|(current, _)| signature < *current)
        {
            best = Some((signature, quarter));
        }
    }
    best
}

fn feature_style(
    component: &HashSet<u64>,
    styles_by_target: &HashMap<u64, Vec<StyleRef>>,
    container_styles: &[StyleRef],
) -> Option<(StyleKey, Vec<u64>)> {
    let mut any_face_style = false;
    let mut assignments: Option<Vec<u64>> = None;
    let mut style_ids = Vec::new();

    for face in component {
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
        if style_ids.len() != component.len() {
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

fn rigid_translation(source_center: [f64; 3], target_center: [f64; 3], quarter: u8) -> [f64; 3] {
    let (x, y) = rotate_xy(source_center[0], source_center[1], quarter);
    [
        target_center[0] - x,
        target_center[1] - y,
        target_center[2] - source_center[2],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarter_turn_translation_maps_center() {
        let source = [2.0, 3.0, 4.0];
        let target = [10.0, -7.0, 8.0];
        for quarter in 0..4 {
            let translation = rigid_translation(source, target, quarter);
            let (x, y) = rotate_xy(source[0], source[1], quarter);
            assert!((x + translation[0] - target[0]).abs() < 1.0e-12);
            assert!((y + translation[1] - target[1]).abs() < 1.0e-12);
            assert!((source[2] + translation[2] - target[2]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn positive_side_rejects_recess() {
        let frame = PlaneFrame {
            surface_id: 1,
            sense: "T".to_string(),
            origin: [0.0, 0.0, 0.0],
            outward: [0.0, 0.0, 1.0],
        };
        assert!(positive_side(&[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], &frame));
        assert!(!positive_side(&[[0.0, 0.0, 0.0], [0.0, 0.0, -2.0]], &frame));
    }

    #[test]
    fn feature_edges_must_be_exactly_two_manifold_with_host() {
        let component = HashSet::from([1_u64, 2_u64]);
        let face_edges = HashMap::from([
            (1_u64, HashSet::from([10_u64, 11_u64])),
            (2_u64, HashSet::from([10_u64, 12_u64])),
        ]);
        let mut edge_faces = HashMap::from([
            (10_u64, vec![1_u64, 2_u64]),
            (11_u64, vec![1_u64, 9_u64]),
            (12_u64, vec![2_u64, 9_u64]),
        ]);

        assert!(component_is_two_manifold_with_host(
            &component,
            9,
            &face_edges,
            &edge_faces,
        ));

        edge_faces.get_mut(&11).unwrap().push(99);
        assert!(!component_is_two_manifold_with_host(
            &component,
            9,
            &face_edges,
            &edge_faces,
        ));
    }
}
