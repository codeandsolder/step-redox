use crate::brep::{
    append_refs_to_list_param, face_edge_curves, face_sense, face_surface, manifold_shell,
    matching_plane_bound, ref_list_param, remove_refs_from_list_param, toggle_tf,
};
use crate::instances::collect_styles_by_target;
use crate::shape_identity::face_topology_signature;
use crate::step_entities::{
    cartesian_point, entity_ref, number, patch_presentation_lists, push_point, push_simple,
    representation_items_and_context,
};
use crate::step_graph::{ReferenceGraph, build_index, entity_id, entity_ref_value, simple_record};
use ruststep::ast::{EntityInstance, Parameter};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Default, Clone)]
pub struct SphericalCapStats {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaceSurfaceKind {
    Plane,
    Sphere,
    Other,
}

#[derive(Debug, Clone, Copy)]
struct CapSourceFrame {
    z_dir: u64,
    x_dir: u64,
    origin_axis: u64,
    rep_map: u64,
}

#[derive(Default)]
struct CapRewriteState {
    candidate_roots: HashSet<u64>,
    delete_seed: HashSet<u64>,
    old_styles_to_remove: HashSet<u64>,
    new_style_ids: Vec<u64>,
    stats: SphericalCapStats,
}

pub fn instance_planar_spherical_caps(entities: &mut Vec<EntityInstance>) -> SphericalCapStats {
    if entities.is_empty() {
        return SphericalCapStats::default();
    }

    let index = build_index(entities);
    let styles_by_target = collect_styles_by_target(entities);
    let representation_ids = entities
        .iter()
        .filter_map(|entity| match entity {
            EntityInstance::Simple { id, record }
                if record.name == "ADVANCED_BREP_SHAPE_REPRESENTATION" =>
            {
                Some(*id)
            }
            _ => None,
        })
        .collect::<Vec<_>>();

    let mut next_id = entities.iter().map(entity_id).max().unwrap_or(0) + 1;
    let mut state = CapRewriteState::default();

    for representation_id in representation_ids {
        let Some(&rep_idx) = index.get(&representation_id) else {
            continue;
        };
        let Some((item_ids, context_id)) = representation_items_and_context(&entities[rep_idx])
        else {
            continue;
        };

        let solid_ids = item_ids
            .into_iter()
            .filter(|id| is_manifold_solid(*id, entities, &index))
            .collect::<Vec<_>>();
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

            let families = discover_cap_families(&shell_faces, entities, &index, &styles_by_target);
            for (plane_face, mut features) in families {
                if prepare_cap_family(
                    plane_face,
                    &mut features,
                    shell_id,
                    rep_idx,
                    context_id,
                    entities,
                    &index,
                    &styles_by_target,
                    &mut next_id,
                    &mut state,
                ) {
                    state.stats.arrays += 1;
                    state.stats.instances += features.len();
                }
            }
        }
    }

    if state.stats.arrays == 0 {
        return state.stats;
    }

    patch_presentation_lists(entities, &state.old_styles_to_remove, &state.new_style_ids);
    collect_replaced_cap_geometry(entities, &mut state);
    state.stats
}

fn is_manifold_solid(id: u64, entities: &[EntityInstance], index: &HashMap<u64, usize>) -> bool {
    index
        .get(&id)
        .and_then(|&idx| simple_record(&entities[idx]))
        .is_some_and(|record| record.name == "MANIFOLD_SOLID_BREP")
}

fn shell_face_surface_kinds(
    shell_faces: &[u64],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> HashMap<u64, FaceSurfaceKind> {
    shell_faces
        .iter()
        .copied()
        .map(|face| {
            let kind = face_surface(face, entities, index)
                .and_then(|surface| index.get(&surface).copied())
                .and_then(|idx| simple_record(&entities[idx]))
                .map_or(FaceSurfaceKind::Other, |record| {
                    match record.name.as_str() {
                        "PLANE" => FaceSurfaceKind::Plane,
                        "SPHERICAL_SURFACE" => FaceSurfaceKind::Sphere,
                        _ => FaceSurfaceKind::Other,
                    }
                });
            (face, kind)
        })
        .collect()
}

fn discover_cap_families(
    shell_faces: &[u64],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    styles_by_target: &HashMap<u64, Vec<crate::instances::StyleRef>>,
) -> HashMap<u64, Vec<CapFeature>> {
    let surface_kinds = shell_face_surface_kinds(shell_faces, entities, index);
    let mut face_edges = HashMap::<u64, HashSet<u64>>::with_capacity(shell_faces.len());
    let mut edge_faces = HashMap::<u64, Vec<u64>>::new();
    for &face in shell_faces {
        let Some(edges) = face_edge_curves(face, entities, index) else {
            continue;
        };
        for &edge in &edges {
            edge_faces.entry(edge).or_default().push(face);
        }
        face_edges.insert(face, edges);
    }

    let sphere_faces = shell_faces
        .iter()
        .copied()
        .filter(|face| surface_kinds.get(face) == Some(&FaceSurfaceKind::Sphere))
        .collect::<Vec<_>>();
    if sphere_faces.len() < 16 {
        return HashMap::new();
    }

    let mut seen_spheres = HashSet::new();
    let mut by_plane = HashMap::<u64, Vec<CapFeature>>::new();
    for face in sphere_faces {
        if seen_spheres.contains(&face) {
            continue;
        }
        let Some(edges) = face_edges.get(&face).filter(|edges| edges.len() == 3) else {
            continue;
        };

        let sphere_neighbors = neighbors_of_kind(
            face,
            edges,
            &edge_faces,
            &surface_kinds,
            FaceSurfaceKind::Sphere,
        );
        let [sibling] = sphere_neighbors.as_slice() else {
            continue;
        };
        let sibling = *sibling;
        if seen_spheres.contains(&sibling) {
            continue;
        }
        let Some(sibling_edges) = face_edges.get(&sibling).filter(|edges| edges.len() == 3) else {
            continue;
        };

        let shared = edges
            .intersection(sibling_edges)
            .copied()
            .collect::<HashSet<_>>();
        if shared.len() != 2 {
            continue;
        }
        let interface = edges
            .union(sibling_edges)
            .filter(|edge| !shared.contains(edge))
            .copied()
            .collect::<HashSet<_>>();
        if interface.len() != 2 {
            continue;
        }

        let mut plane_neighbors = HashSet::new();
        for edge in &interface {
            for &neighbor in edge_faces.get(edge).into_iter().flatten() {
                if neighbor != face
                    && neighbor != sibling
                    && surface_kinds.get(&neighbor) == Some(&FaceSurfaceKind::Plane)
                {
                    plane_neighbors.insert(neighbor);
                }
            }
        }
        let Some(&plane_face) = plane_neighbors
            .iter()
            .next()
            .filter(|_| plane_neighbors.len() == 1)
        else {
            continue;
        };

        let Some((interface_bound, interface_loop, bound_orientation)) =
            matching_plane_bound(plane_face, &interface, entities, index)
        else {
            continue;
        };
        let Some(center_a) = spherical_face_center(face, entities, index) else {
            continue;
        };
        let Some(center_b) = spherical_face_center(sibling, entities, index) else {
            continue;
        };
        if !same_point(center_a, center_b) {
            continue;
        }

        let Some(mut topology) = [face, sibling]
            .into_iter()
            .map(|id| face_topology_signature(id, entities, index, center_a, 0))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        topology.sort();

        let Some(style_a) = single_face_style(face, styles_by_target) else {
            continue;
        };
        let Some(style_b) = single_face_style(sibling, styles_by_target) else {
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
            topology_key: topology.join("||"),
            style_assignments: style_a,
            bound_orientation,
        });
        seen_spheres.extend([face, sibling]);
    }
    by_plane
}

fn neighbors_of_kind(
    face: u64,
    edges: &HashSet<u64>,
    edge_faces: &HashMap<u64, Vec<u64>>,
    surface_kinds: &HashMap<u64, FaceSurfaceKind>,
    kind: FaceSurfaceKind,
) -> Vec<u64> {
    let mut neighbors = HashSet::new();
    for edge in edges {
        for &neighbor in edge_faces.get(edge).into_iter().flatten() {
            if neighbor != face && surface_kinds.get(&neighbor) == Some(&kind) {
                neighbors.insert(neighbor);
            }
        }
    }
    let mut neighbors = neighbors.into_iter().collect::<Vec<_>>();
    neighbors.sort_unstable();
    neighbors
}

#[expect(
    clippy::too_many_arguments,
    reason = "feature-family emission needs the proven source context plus shared rewrite state"
)]
fn prepare_cap_family(
    plane_face: u64,
    features: &mut [CapFeature],
    shell_id: u64,
    rep_idx: usize,
    context_id: u64,
    entities: &mut Vec<EntityInstance>,
    index: &HashMap<u64, usize>,
    styles_by_target: &HashMap<u64, Vec<crate::instances::StyleRef>>,
    next_id: &mut u64,
    state: &mut CapRewriteState,
) -> bool {
    if features.len() < 8 {
        return false;
    }
    features.sort_by_key(|feature| feature.faces);
    let canonical = &features[0];
    if features.iter().any(|feature| {
        feature.topology_key != canonical.topology_key
            || feature.style_assignments != canonical.style_assignments
            || feature.bound_orientation != canonical.bound_orientation
    }) {
        return false;
    }

    let bound_ids = features
        .iter()
        .map(|feature| feature.interface_bound)
        .collect::<HashSet<_>>();
    let sphere_ids = features
        .iter()
        .flat_map(|feature| feature.faces)
        .collect::<HashSet<_>>();
    if bound_ids.len() != features.len() || sphere_ids.len() != features.len() * 2 {
        return false;
    }

    let Some(plane_surface) = face_surface(plane_face, entities, index) else {
        return false;
    };
    let Some(&plane_surface_idx) = index.get(&plane_surface) else {
        return false;
    };
    if simple_record(&entities[plane_surface_idx]).is_none_or(|record| record.name != "PLANE") {
        return false;
    }
    let Some(plane_sense) = face_sense(plane_face, entities, index) else {
        return false;
    };
    let Some(plane_bounds) = ref_list_param(plane_face, 1, entities, index) else {
        return false;
    };
    if !bound_ids.iter().all(|bound| plane_bounds.contains(bound)) {
        return false;
    }

    // Resolve every mutable target before appending anything. Once this proof
    // passes, emission cannot leave half-created cap infrastructure behind on a
    // missing source entity.
    let Some(&shell_index) = index.get(&shell_id) else {
        return false;
    };
    let Some(&plane_face_index) = index.get(&plane_face) else {
        return false;
    };

    let source = emit_cap_source(
        canonical,
        plane_surface,
        &plane_sense,
        context_id,
        entities,
        next_id,
    );
    let (mapped_ids, group_style_ids) =
        emit_cap_instances(features, canonical, source, entities, next_id);

    if !remove_refs_from_list_param(&mut entities[shell_index], 1, &sphere_ids)
        || !remove_refs_from_list_param(&mut entities[plane_face_index], 1, &bound_ids)
        || !append_refs_to_list_param(&mut entities[rep_idx], 1, &mapped_ids)
    {
        // All list shapes were proven above. Reaching this branch would mean a
        // mutation helper no longer matches the corresponding read helper.
        debug_assert!(false, "validated spherical-cap list rewrite failed");
        return false;
    }

    register_replaced_cap_roots(features, styles_by_target, state);
    state.new_style_ids.extend(group_style_ids);
    true
}

fn emit_cap_source(
    canonical: &CapFeature,
    plane_surface: u64,
    plane_sense: &str,
    context_id: u64,
    entities: &mut Vec<EntityInstance>,
    next_id: &mut u64,
) -> CapSourceFrame {
    let z_dir = push_simple(
        entities,
        next_id,
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
        next_id,
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
    let origin_point = push_point(entities, next_id, [0.0, 0.0, 0.0]);
    let origin_axis = push_simple(
        entities,
        next_id,
        "AXIS2_PLACEMENT_3D",
        vec![
            Parameter::String(String::new()),
            entity_ref(origin_point),
            entity_ref(z_dir),
            entity_ref(x_dir),
        ],
    );

    let disk_bound = push_simple(
        entities,
        next_id,
        "FACE_OUTER_BOUND",
        vec![
            Parameter::String("NONE".to_string()),
            entity_ref(canonical.interface_loop),
            Parameter::Enumeration(toggle_tf(&canonical.bound_orientation)),
        ],
    );
    let disk_face = push_simple(
        entities,
        next_id,
        "ADVANCED_FACE",
        vec![
            Parameter::String("NONE".to_string()),
            Parameter::List(vec![entity_ref(disk_bound)]),
            entity_ref(plane_surface),
            Parameter::Enumeration(toggle_tf(plane_sense)),
        ],
    );
    let ball_shell = push_simple(
        entities,
        next_id,
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
        next_id,
        "MANIFOLD_SOLID_BREP",
        vec![
            Parameter::String("step-redox canonical spherical cap".to_string()),
            entity_ref(ball_shell),
        ],
    );
    let source_rep = push_simple(
        entities,
        next_id,
        "ADVANCED_BREP_SHAPE_REPRESENTATION",
        vec![
            Parameter::String("step-redox spherical-cap source".to_string()),
            Parameter::List(vec![entity_ref(ball_solid), entity_ref(origin_axis)]),
            entity_ref(context_id),
        ],
    );
    let rep_map = push_simple(
        entities,
        next_id,
        "REPRESENTATION_MAP",
        vec![entity_ref(origin_axis), entity_ref(source_rep)],
    );
    CapSourceFrame {
        z_dir,
        x_dir,
        origin_axis,
        rep_map,
    }
}

fn emit_cap_instances(
    features: &[CapFeature],
    canonical: &CapFeature,
    source: CapSourceFrame,
    entities: &mut Vec<EntityInstance>,
    next_id: &mut u64,
) -> (Vec<u64>, Vec<u64>) {
    let mut mapped_ids = Vec::with_capacity(features.len());
    let mut style_ids = Vec::with_capacity(features.len());
    for (feature_index, feature) in features.iter().enumerate() {
        let axis = if feature_index == 0 {
            source.origin_axis
        } else {
            let translation = [
                feature.center[0] - canonical.center[0],
                feature.center[1] - canonical.center[1],
                feature.center[2] - canonical.center[2],
            ];
            let point = push_point(entities, next_id, translation);
            push_simple(
                entities,
                next_id,
                "AXIS2_PLACEMENT_3D",
                vec![
                    Parameter::String(String::new()),
                    entity_ref(point),
                    entity_ref(source.z_dir),
                    entity_ref(source.x_dir),
                ],
            )
        };
        let mapped = push_simple(
            entities,
            next_id,
            "MAPPED_ITEM",
            vec![
                Parameter::String(String::new()),
                entity_ref(source.rep_map),
                entity_ref(axis),
            ],
        );
        let styled = push_simple(
            entities,
            next_id,
            "STYLED_ITEM",
            vec![
                Parameter::String("NONE".to_string()),
                Parameter::List(
                    canonical
                        .style_assignments
                        .iter()
                        .copied()
                        .map(entity_ref)
                        .collect(),
                ),
                entity_ref(mapped),
            ],
        );
        mapped_ids.push(mapped);
        style_ids.push(styled);
    }
    (mapped_ids, style_ids)
}

fn register_replaced_cap_roots(
    features: &[CapFeature],
    styles_by_target: &HashMap<u64, Vec<crate::instances::StyleRef>>,
    state: &mut CapRewriteState,
) {
    for feature in features {
        for face in feature.faces {
            state.candidate_roots.insert(face);
            for style in styles_by_target.get(&face).into_iter().flatten() {
                state.old_styles_to_remove.insert(style.id);
                state.candidate_roots.insert(style.id);
                state.delete_seed.insert(style.id);
            }
        }
        state.candidate_roots.insert(feature.interface_bound);
        state.delete_seed.insert(feature.interface_bound);
    }
    // Canonical spherical faces stay live through the source representation.
    for feature in features.iter().skip(1) {
        state.delete_seed.extend(feature.faces);
    }
}

fn collect_replaced_cap_geometry(entities: &mut Vec<EntityInstance>, state: &mut CapRewriteState) {
    let index = build_index(entities);
    let delete = ReferenceGraph::new(entities).detached_descendant_closure(
        &index,
        &state.candidate_roots,
        std::mem::take(&mut state.delete_seed),
    );

    state.stats.styles_replaced = state
        .old_styles_to_remove
        .iter()
        .filter(|id| delete.contains(id))
        .count();
    state.stats.entities_removed = delete.len();
    entities.retain(|entity| !delete.contains(&entity_id(entity)));
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

fn same_point(a: [f64; 3], b: [f64; 3]) -> bool {
    (0..3).all(|axis| (a[axis] - b[axis]).abs() <= 1.0e-12)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_comparison_uses_tight_geometry_tolerance() {
        assert!(same_point([1.0, 2.0, 3.0], [1.0, 2.0, 3.0 + 5.0e-13]));
        assert!(!same_point([1.0, 2.0, 3.0], [1.0, 2.0, 3.0 + 2.0e-12]));
    }
}
