use crate::instances::{
    build_index, cartesian_point, collect_styles_by_target, entity_id, entity_ref, entity_ref_map,
    entity_ref_value, face_topology_signature, inbound_map, number, patch_presentation_lists,
    push_point, push_simple, representation_items_and_context, simple_record, simple_record_mut,
    visit_entity_refs,
};
use ruststep::ast::{EntityInstance, Parameter};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default, Clone)]
pub(crate) struct SphericalCapStats {
    pub arrays: usize,
    pub instances: usize,
    pub entities_removed: usize,
    pub styles_replaced: usize,
}

#[derive(Debug, Clone)]
struct CapFeature {
    faces: [u64; 2],
    interface_bound: u64,
    interface_loop: u64,
    center: [f64; 3],
    topology_key: String,
    style_assignments: Vec<u64>,
    bound_orientation: String,
}

pub(crate) fn instance_planar_spherical_caps(
    entities: &mut Vec<EntityInstance>,
) -> SphericalCapStats {
    let mut stats = SphericalCapStats::default();
    if entities.is_empty() {
        return stats;
    }

    let index = build_index(entities);
    let styles_by_target = collect_styles_by_target(entities);
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
    let mut candidate_roots = HashSet::new();
    let mut delete_seed = HashSet::new();
    let mut old_styles_to_remove = HashSet::new();
    let mut new_style_ids = Vec::new();

    for representation_id in representation_ids {
        let Some(&rep_idx) = index.get(&representation_id) else {
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
                index
                    .get(id)
                    .and_then(|&idx| simple_record(&entities[idx]))
                    .is_some_and(|record| record.name == "MANIFOLD_SOLID_BREP")
            })
            .collect();

        for solid_id in solid_ids {
            let Some(shell_id) = manifold_shell(solid_id, entities, &index) else {
                continue;
            };
            let Some(shell_faces) = ref_list_param(shell_id, 1, entities, &index) else {
                continue;
            };
            if shell_faces.len() < 16 {
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

            let sphere_faces: Vec<u64> = shell_faces
                .iter()
                .copied()
                .filter(|face| {
                    face_surface(*face, entities, &index)
                        .and_then(|surface| index.get(&surface).copied())
                        .and_then(|idx| simple_record(&entities[idx]))
                        .is_some_and(|record| record.name == "SPHERICAL_SURFACE")
                })
                .collect();

            if sphere_faces.len() < 16 {
                continue;
            }

            let mut seen_spheres = HashSet::new();
            let mut by_plane = HashMap::<u64, Vec<CapFeature>>::new();

            for face in sphere_faces {
                if seen_spheres.contains(&face) {
                    continue;
                }
                let Some(edges) = face_edges.get(&face) else {
                    continue;
                };
                if edges.len() != 3 {
                    continue;
                }

                let mut sphere_neighbors = HashSet::new();
                for edge in edges {
                    for &neighbor in edge_faces.get(edge).into_iter().flatten() {
                        if neighbor == face {
                            continue;
                        }
                        if face_surface(neighbor, entities, &index)
                            .and_then(|surface| index.get(&surface).copied())
                            .and_then(|idx| simple_record(&entities[idx]))
                            .is_some_and(|record| record.name == "SPHERICAL_SURFACE")
                        {
                            sphere_neighbors.insert(neighbor);
                        }
                    }
                }
                if sphere_neighbors.len() != 1 {
                    continue;
                }
                let sibling = *sphere_neighbors.iter().next().unwrap();
                if seen_spheres.contains(&sibling) {
                    continue;
                }
                let Some(sibling_edges) = face_edges.get(&sibling) else {
                    continue;
                };
                if sibling_edges.len() != 3 {
                    continue;
                }

                let shared: HashSet<u64> = edges.intersection(sibling_edges).copied().collect();
                if shared.len() != 2 {
                    continue;
                }
                let union: HashSet<u64> = edges.union(sibling_edges).copied().collect();
                let interface: HashSet<u64> = union.difference(&shared).copied().collect();
                if interface.len() != 2 {
                    continue;
                }

                let pair_faces = HashSet::from([face, sibling]);
                let mut plane_neighbors = HashSet::new();
                for edge in &interface {
                    for &neighbor in edge_faces.get(edge).into_iter().flatten() {
                        if pair_faces.contains(&neighbor) {
                            continue;
                        }
                        if face_surface(neighbor, entities, &index)
                            .and_then(|surface| index.get(&surface).copied())
                            .and_then(|idx| simple_record(&entities[idx]))
                            .is_some_and(|record| record.name == "PLANE")
                        {
                            plane_neighbors.insert(neighbor);
                        }
                    }
                }
                if plane_neighbors.len() != 1 {
                    continue;
                }
                let plane_face = *plane_neighbors.iter().next().unwrap();

                let Some((interface_bound, interface_loop, bound_orientation)) =
                    matching_plane_bound(plane_face, &interface, entities, &index)
                else {
                    continue;
                };

                let Some(center_a) = spherical_face_center(face, entities, &index) else {
                    continue;
                };
                let Some(center_b) = spherical_face_center(sibling, entities, &index) else {
                    continue;
                };
                if !same_point(center_a, center_b) {
                    continue;
                }

                let mut topology = Vec::new();
                let Some(sig_a) = face_topology_signature(face, entities, &index, center_a, 0)
                else {
                    continue;
                };
                let Some(sig_b) = face_topology_signature(sibling, entities, &index, center_a, 0)
                else {
                    continue;
                };
                topology.push(sig_a);
                topology.push(sig_b);
                topology.sort();
                let topology_key = topology.join("||");

                let Some(style_a) = single_face_style(face, &styles_by_target) else {
                    continue;
                };
                let Some(style_b) = single_face_style(sibling, &styles_by_target) else {
                    continue;
                };
                if style_a != style_b {
                    continue;
                }

                let mut faces = [face, sibling];
                faces.sort_unstable();
                by_plane.entry(plane_face).or_default().push(CapFeature {
                    faces,
                    interface_bound,
                    interface_loop,
                    center: center_a,
                    topology_key,
                    style_assignments: style_a,
                    bound_orientation,
                });
                seen_spheres.insert(face);
                seen_spheres.insert(sibling);
            }

            for (plane_face, mut features) in by_plane {
                if features.len() < 8 {
                    continue;
                }
                features.sort_by_key(|feature| feature.faces);

                let first_key = features[0].topology_key.clone();
                let first_style = features[0].style_assignments.clone();
                let first_bound_orientation = features[0].bound_orientation.clone();
                if features.iter().any(|feature| {
                    feature.topology_key != first_key
                        || feature.style_assignments != first_style
                        || feature.bound_orientation != first_bound_orientation
                }) {
                    continue;
                }

                let bound_ids: HashSet<u64> = features
                    .iter()
                    .map(|feature| feature.interface_bound)
                    .collect();
                let sphere_ids: HashSet<u64> =
                    features.iter().flat_map(|feature| feature.faces).collect();
                if bound_ids.len() != features.len() || sphere_ids.len() != features.len() * 2 {
                    continue;
                }

                let Some(plane_surface) = face_surface(plane_face, entities, &index) else {
                    continue;
                };
                let Some(&plane_surface_idx) = index.get(&plane_surface) else {
                    continue;
                };
                if simple_record(&entities[plane_surface_idx])
                    .is_none_or(|record| record.name != "PLANE")
                {
                    continue;
                }
                let Some(plane_sense) = face_sense(plane_face, entities, &index) else {
                    continue;
                };

                // Require all matched bounds to still be direct bounds of this plane.
                let Some(plane_bounds) = ref_list_param(plane_face, 1, entities, &index) else {
                    continue;
                };
                if !bound_ids.iter().all(|bound| plane_bounds.contains(bound)) {
                    continue;
                }

                let canonical = features[0].clone();

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
                let origin_point = push_point(entities, &mut next_id, [0.0, 0.0, 0.0]);
                let origin_axis = push_simple(
                    entities,
                    &mut next_id,
                    "AXIS2_PLACEMENT_3D",
                    vec![
                        Parameter::String(String::new()),
                        entity_ref(origin_point),
                        entity_ref(z_dir),
                        entity_ref(x_dir),
                    ],
                );

                // Convert the canonical inner-hole loop into the outer boundary
                // of a closure disk. The disk faces the opposite direction from
                // the substrate plane.
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
                        entity_ref(plane_surface),
                        Parameter::Enumeration(toggle_tf(&plane_sense)),
                    ],
                );
                let ball_shell = push_simple(
                    entities,
                    &mut next_id,
                    "CLOSED_SHELL",
                    vec![
                        Parameter::String("NONE".to_string()),
                        Parameter::List(vec![
                            entity_ref(canonical.faces[0]),
                            entity_ref(canonical.faces[1]),
                            entity_ref(disk_face),
                        ]),
                    ],
                );
                let ball_solid = push_simple(
                    entities,
                    &mut next_id,
                    "MANIFOLD_SOLID_BREP",
                    vec![
                        Parameter::String("step-redox canonical spherical cap".to_string()),
                        entity_ref(ball_shell),
                    ],
                );
                let source_rep = push_simple(
                    entities,
                    &mut next_id,
                    "ADVANCED_BREP_SHAPE_REPRESENTATION",
                    vec![
                        Parameter::String("step-redox spherical-cap source".to_string()),
                        Parameter::List(vec![entity_ref(ball_solid), entity_ref(origin_axis)]),
                        entity_ref(context_id),
                    ],
                );
                let rep_map = push_simple(
                    entities,
                    &mut next_id,
                    "REPRESENTATION_MAP",
                    vec![entity_ref(origin_axis), entity_ref(source_rep)],
                );

                let mut mapped_ids = Vec::with_capacity(features.len());
                let mut group_style_ids = Vec::with_capacity(features.len());
                for (feature_index, feature) in features.iter().enumerate() {
                    let axis = if feature_index == 0 {
                        origin_axis
                    } else {
                        let translation = [
                            feature.center[0] - canonical.center[0],
                            feature.center[1] - canonical.center[1],
                            feature.center[2] - canonical.center[2],
                        ];
                        let point = push_point(entities, &mut next_id, translation);
                        push_simple(
                            entities,
                            &mut next_id,
                            "AXIS2_PLACEMENT_3D",
                            vec![
                                Parameter::String(String::new()),
                                entity_ref(point),
                                entity_ref(z_dir),
                                entity_ref(x_dir),
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
                            Parameter::List(first_style.iter().copied().map(entity_ref).collect()),
                            entity_ref(mapped),
                        ],
                    );
                    mapped_ids.push(mapped);
                    group_style_ids.push(styled);
                }

                // Remove spherical faces from the fused substrate shell and fill
                // the corresponding holes in its planar face.
                if !remove_refs_from_list_param(
                    &mut entities[*index.get(&shell_id).unwrap()],
                    1,
                    &sphere_ids,
                ) {
                    continue;
                }
                if !remove_refs_from_list_param(
                    &mut entities[*index.get(&plane_face).unwrap()],
                    1,
                    &bound_ids,
                ) {
                    continue;
                }
                if !append_refs_to_list_param(&mut entities[rep_idx], 1, &mapped_ids) {
                    continue;
                }

                for feature in &features {
                    for face in feature.faces {
                        candidate_roots.insert(face);
                        for style in styles_by_target.get(&face).into_iter().flatten() {
                            old_styles_to_remove.insert(style.id);
                            candidate_roots.insert(style.id);
                            delete_seed.insert(style.id);
                        }
                    }
                    candidate_roots.insert(feature.interface_bound);
                    delete_seed.insert(feature.interface_bound);
                }
                // The canonical two spherical faces stay live through source_rep.
                for feature in features.iter().skip(1) {
                    for face in feature.faces {
                        delete_seed.insert(face);
                    }
                }

                new_style_ids.extend(group_style_ids);
                stats.arrays += 1;
                stats.instances += features.len();
            }
        }
    }

    if stats.arrays == 0 {
        return stats;
    }

    patch_presentation_lists(entities, &old_styles_to_remove, &new_style_ids);

    // Reachability-limited GC over descendants of the replaced feature roots.
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

    // No live entity may point at a deleted entity.
    for (id, children) in &refs {
        if delete.contains(id) {
            continue;
        }
        debug_assert!(children.iter().all(|child| !delete.contains(child)));
    }

    stats.styles_replaced = old_styles_to_remove
        .iter()
        .filter(|id| delete.contains(id))
        .count();
    stats.entities_removed = delete.len();
    entities.retain(|entity| !delete.contains(&entity_id(entity)));
    stats
}

fn manifold_shell(
    solid_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<u64> {
    let record = simple_record(&entities[*index.get(&solid_id)?])?;
    if record.name != "MANIFOLD_SOLID_BREP" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    entity_ref_value(params.get(1)?)
}

fn face_surface(
    face_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<u64> {
    let record = simple_record(&entities[*index.get(&face_id)?])?;
    if record.name != "ADVANCED_FACE" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    entity_ref_value(params.get(2)?)
}

fn face_sense(
    face_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<String> {
    let record = simple_record(&entities[*index.get(&face_id)?])?;
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    enumeration_value(params.get(3)?)
}

fn face_edge_curves(
    face_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<HashSet<u64>> {
    let record = simple_record(&entities[*index.get(&face_id)?])?;
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let Parameter::List(bounds) = params.get(1)? else {
        return None;
    };

    let mut edges = HashSet::new();
    for bound in bounds {
        let bound_id = entity_ref_value(bound)?;
        let (_, bound_edges, _) = bound_loop_edges(bound_id, entities, index)?;
        edges.extend(bound_edges);
    }
    Some(edges)
}

fn bound_loop_edges(
    bound_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<(u64, HashSet<u64>, String)> {
    let record = simple_record(&entities[*index.get(&bound_id)?])?;
    if record.name != "FACE_BOUND" && record.name != "FACE_OUTER_BOUND" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let loop_id = entity_ref_value(params.get(1)?)?;
    let orientation = enumeration_value(params.get(2)?)?;

    let loop_record = simple_record(&entities[*index.get(&loop_id)?])?;
    if loop_record.name != "EDGE_LOOP" {
        return None;
    }
    let Parameter::List(loop_params) = &loop_record.parameter else {
        return None;
    };
    let Parameter::List(oriented_edges) = loop_params.get(1)? else {
        return None;
    };

    let mut edge_curves = HashSet::new();
    for oriented in oriented_edges {
        let oriented_id = entity_ref_value(oriented)?;
        let oriented_record = simple_record(&entities[*index.get(&oriented_id)?])?;
        if oriented_record.name != "ORIENTED_EDGE" {
            return None;
        }
        let Parameter::List(oriented_params) = &oriented_record.parameter else {
            return None;
        };
        edge_curves.insert(entity_ref_value(oriented_params.get(3)?)?);
    }
    Some((loop_id, edge_curves, orientation))
}

fn matching_plane_bound(
    plane_face: u64,
    interface_edges: &HashSet<u64>,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<(u64, u64, String)> {
    let bounds = ref_list_param(plane_face, 1, entities, index)?;
    let mut matched = None;
    for bound in bounds {
        let record = simple_record(&entities[*index.get(&bound)?])?;
        if record.name != "FACE_BOUND" {
            continue;
        }
        let Some((loop_id, edges, orientation)) = bound_loop_edges(bound, entities, index) else {
            continue;
        };
        if edges == *interface_edges {
            if matched.is_some() {
                return None;
            }
            matched = Some((bound, loop_id, orientation));
        }
    }
    matched
}

fn spherical_face_center(
    face_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let surface = face_surface(face_id, entities, index)?;
    let surface_record = simple_record(&entities[*index.get(&surface)?])?;
    if surface_record.name != "SPHERICAL_SURFACE" {
        return None;
    }
    let Parameter::List(surface_params) = &surface_record.parameter else {
        return None;
    };
    // Require a finite positive radius.
    if number(surface_params.get(2)?)? <= 0.0 {
        return None;
    }
    let axis_id = entity_ref_value(surface_params.get(1)?)?;
    let axis_record = simple_record(&entities[*index.get(&axis_id)?])?;
    if axis_record.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let Parameter::List(axis_params) = &axis_record.parameter else {
        return None;
    };
    let point_id = entity_ref_value(axis_params.get(1)?)?;
    cartesian_point(point_id, entities, index)
}

fn single_face_style(
    face_id: u64,
    styles_by_target: &HashMap<u64, Vec<crate::instances::StyleRef>>,
) -> Option<Vec<u64>> {
    let styles = styles_by_target.get(&face_id)?;
    if styles.len() != 1 || styles[0].assignments.is_empty() {
        return None;
    }
    Some(styles[0].assignments.clone())
}

fn ref_list_param(
    id: u64,
    param_index: usize,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<u64>> {
    let record = simple_record(&entities[*index.get(&id)?])?;
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let Parameter::List(items) = params.get(param_index)? else {
        return None;
    };
    items.iter().map(entity_ref_value).collect()
}

fn remove_refs_from_list_param(
    entity: &mut EntityInstance,
    param_index: usize,
    remove: &HashSet<u64>,
) -> bool {
    let Some(record) = simple_record_mut(entity) else {
        return false;
    };
    let Parameter::List(params) = &mut record.parameter else {
        return false;
    };
    let Some(Parameter::List(items)) = params.get_mut(param_index) else {
        return false;
    };
    let before = items.len();
    items.retain(|item| entity_ref_value(item).is_none_or(|id| !remove.contains(&id)));
    items.len() < before
}

fn append_refs_to_list_param(
    entity: &mut EntityInstance,
    param_index: usize,
    append: &[u64],
) -> bool {
    let Some(record) = simple_record_mut(entity) else {
        return false;
    };
    let Parameter::List(params) = &mut record.parameter else {
        return false;
    };
    let Some(Parameter::List(items)) = params.get_mut(param_index) else {
        return false;
    };
    items.extend(append.iter().copied().map(entity_ref));
    true
}

fn enumeration_value(parameter: &Parameter) -> Option<String> {
    match parameter {
        Parameter::Enumeration(value) => Some(value.clone()),
        _ => None,
    }
}

fn toggle_tf(value: &str) -> String {
    match value {
        "T" => "F".to_string(),
        "F" => "T".to_string(),
        _ => value.to_string(),
    }
}

fn same_point(a: [f64; 3], b: [f64; 3]) -> bool {
    (0..3).all(|axis| (a[axis] - b[axis]).abs() <= 1.0e-12)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggles_step_boolean_enumerations() {
        assert_eq!(toggle_tf("T"), "F");
        assert_eq!(toggle_tf("F"), "T");
    }

    #[test]
    fn point_comparison_uses_tight_geometry_tolerance() {
        assert!(same_point([1.0, 2.0, 3.0], [1.0, 2.0, 3.0 + 5.0e-13]));
        assert!(!same_point([1.0, 2.0, 3.0], [1.0, 2.0, 3.0 + 2.0e-12]));
    }
}
