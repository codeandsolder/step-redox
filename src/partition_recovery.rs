use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

const POINT_Q: f64 = 1.0e-8;
const NORMAL_Q: f64 = 1.0e-10;
const OPPOSITE_DOT: f64 = -1.0 + 1.0e-8;

#[derive(Debug, Default, Clone)]
pub(crate) struct PartitionRecoveryStats {
    pub components: usize,
    pub solids_merged: usize,
    pub interfaces_removed: usize,
    pub styles_retargeted: usize,
    pub entities_removed: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum EdgeKey {
    Line {
        ends: [[i64; 3]; 2],
    },
    Circle {
        center: [i64; 3],
        axis: [i64; 3],
        radius: i64,
        ends: [[i64; 3]; 2],
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct BoundKey {
    kind: u8,
    edges: Vec<EdgeKey>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct FaceKey {
    plane_axis: [i64; 3],
    plane_d: i64,
    bounds: Vec<BoundKey>,
}

#[derive(Debug, Clone)]
struct FaceCandidate {
    solid: u64,
    face: u64,
    outward: [f64; 3],
}

#[derive(Debug, Clone)]
struct Interface {
    a_solid: u64,
    b_solid: u64,
    a_face: u64,
    b_face: u64,
}

#[derive(Debug, Clone)]
struct ComponentPlan {
    solids: Vec<u64>,
    interfaces: Vec<Interface>,
    retained_faces: Vec<u64>,
    rep: u64,
    style_assignment: Vec<u64>,
    old_styles: Vec<u64>,
}

#[derive(Debug, Default)]
struct UnionFind {
    parent: HashMap<u64, u64>,
}

impl UnionFind {
    fn find(&mut self, id: u64) -> u64 {
        let p = *self.parent.entry(id).or_insert(id);
        if p == id {
            return id;
        }
        let root = self.find(p);
        self.parent.insert(id, root);
        root
    }

    fn union(&mut self, a: u64, b: u64) {
        let mut a = self.find(a);
        let mut b = self.find(b);
        if a == b {
            return;
        }
        if b < a {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent.insert(b, a);
    }

    fn mapped(&mut self, id: u64) -> u64 {
        if self.parent.contains_key(&id) {
            self.find(id)
        } else {
            id
        }
    }
}

pub(crate) fn recover_partitioned_bodies(
    entities: &mut Vec<EntityInstance>,
) -> PartitionRecoveryStats {
    let original = entities.clone();
    match recover_inner(entities) {
        Some(stats) => stats,
        None => {
            *entities = original;
            PartitionRecoveryStats::default()
        }
    }
}

fn recover_inner(entities: &mut Vec<EntityInstance>) -> Option<PartitionRecoveryStats> {
    if entities.is_empty() {
        return Some(PartitionRecoveryStats::default());
    }

    let index = build_index(entities);
    let refs_before = entity_ref_map(entities);
    let styles_by_target = styles_by_target(entities);

    let mut solid_shell = HashMap::new();
    for entity in entities.iter() {
        let id = entity_id(entity);
        let Some(record) = simple_record(entity) else {
            continue;
        };
        if record.name != "MANIFOLD_SOLID_BREP" {
            continue;
        }
        let shell = direct_refs_of_type(entity, entities, &index, "CLOSED_SHELL");
        if shell.len() == 1 {
            solid_shell.insert(id, shell[0]);
        }
    }
    if solid_shell.len() < 2 {
        return Some(PartitionRecoveryStats::default());
    }

    // Partition interfaces must be presentation-neutral planar faces. A direct
    // face style is evidence that the face may be authored/visible semantics.
    let mut buckets: BTreeMap<FaceKey, Vec<FaceCandidate>> = BTreeMap::new();
    for (&solid, &shell) in &solid_shell {
        let &shell_idx = index.get(&shell)?;
        for face in direct_refs_of_type(&entities[shell_idx], entities, &index, "ADVANCED_FACE") {
            if styles_by_target.contains_key(&face) {
                continue;
            }
            let Some((key, outward)) = face_signature(face, entities, &index) else {
                continue;
            };
            buckets.entry(key).or_default().push(FaceCandidate {
                solid,
                face,
                outward,
            });
        }
    }

    let mut interfaces = Vec::new();
    for candidates in buckets.values() {
        // Ambiguous coincident stacks are skipped deliberately.
        if candidates.len() != 2 {
            continue;
        }
        let a = &candidates[0];
        let b = &candidates[1];
        if a.solid == b.solid || dot(a.outward, b.outward) > OPPOSITE_DOT {
            continue;
        }
        interfaces.push(Interface {
            a_solid: a.solid,
            b_solid: b.solid,
            a_face: a.face,
            b_face: b.face,
        });
    }
    if interfaces.is_empty() {
        return Some(PartitionRecoveryStats::default());
    }

    let mut adjacency: HashMap<u64, BTreeSet<u64>> = HashMap::new();
    for interface in &interfaces {
        adjacency
            .entry(interface.a_solid)
            .or_default()
            .insert(interface.b_solid);
        adjacency
            .entry(interface.b_solid)
            .or_default()
            .insert(interface.a_solid);
    }

    let mut seen = HashSet::new();
    let mut raw_components = Vec::<Vec<u64>>::new();
    let mut roots: Vec<u64> = adjacency.keys().copied().collect();
    roots.sort_unstable();
    for root in roots {
        if seen.contains(&root) {
            continue;
        }
        let mut stack = vec![root];
        let mut component = BTreeSet::new();
        while let Some(id) = stack.pop() {
            if !component.insert(id) {
                continue;
            }
            seen.insert(id);
            if let Some(neighbors) = adjacency.get(&id) {
                stack.extend(neighbors.iter().copied());
            }
        }
        if component.len() > 1 {
            raw_components.push(component.into_iter().collect());
        }
    }

    let rep_items = representation_items(entities);
    let mut plans = Vec::<ComponentPlan>::new();
    for solids in raw_components {
        let solid_set: HashSet<u64> = solids.iter().copied().collect();
        let component_interfaces: Vec<Interface> = interfaces
            .iter()
            .filter(|it| solid_set.contains(&it.a_solid) && solid_set.contains(&it.b_solid))
            .cloned()
            .collect();
        if component_interfaces.is_empty() {
            continue;
        }

        let mut assignment: Option<Vec<u64>> = None;
        let mut old_styles = Vec::new();
        let mut style_ok = true;
        for solid in &solids {
            let Some(styles) = styles_by_target.get(solid) else {
                style_ok = false;
                break;
            };
            if styles.len() != 1 {
                style_ok = false;
                break;
            }
            let (style_id, current) = &styles[0];
            match &assignment {
                None => assignment = Some(current.clone()),
                Some(expected) if expected == current => {}
                Some(_) => {
                    style_ok = false;
                    break;
                }
            }
            old_styles.push(*style_id);
        }
        if !style_ok {
            continue;
        }
        let Some(style_assignment) = assignment else {
            continue;
        };

        let owners: Vec<u64> = rep_items
            .iter()
            .filter_map(|(&rep, items)| solids.iter().all(|s| items.contains(s)).then_some(rep))
            .collect();
        if owners.len() != 1 {
            continue;
        }

        let drop_faces: HashSet<u64> = component_interfaces
            .iter()
            .flat_map(|it| [it.a_face, it.b_face])
            .collect();
        let mut retained_faces = Vec::new();
        let mut retained_seen = HashSet::new();
        let mut topology_ok = true;
        for solid in &solids {
            let Some(&shell) = solid_shell.get(solid) else {
                topology_ok = false;
                break;
            };
            let Some(&sh_idx) = index.get(&shell) else {
                topology_ok = false;
                break;
            };
            for face in direct_refs_of_type(&entities[sh_idx], entities, &index, "ADVANCED_FACE") {
                if !drop_faces.contains(&face) && retained_seen.insert(face) {
                    retained_faces.push(face);
                }
            }
        }
        if !topology_ok || retained_faces.is_empty() {
            continue;
        }

        plans.push(ComponentPlan {
            solids,
            interfaces: component_interfaces,
            retained_faces,
            rep: owners[0],
            style_assignment,
            old_styles,
        });
    }
    if plans.is_empty() {
        return Some(PartitionRecoveryStats::default());
    }

    // Union only topology proven equivalent by matched interface faces.
    let mut vertex_uf = UnionFind::default();
    let mut edge_uf = UnionFind::default();
    for plan in &plans {
        for interface in &plan.interfaces {
            let a_edges = keyed_face_edges(interface.a_face, entities, &index)?;
            let b_edges = keyed_face_edges(interface.b_face, entities, &index)?;
            if a_edges.len() != b_edges.len() {
                return None;
            }

            for (key, &(_, a_edge)) in &a_edges {
                let &(_, b_edge) = b_edges.get(key)?;
                edge_uf.union(a_edge, b_edge);

                let [av0, av1] = edge_vertices(a_edge, entities, &index)?;
                let [bv0, bv1] = edge_vertices(b_edge, entities, &index)?;
                let mut a_by_point = BTreeMap::new();
                let mut b_by_point = BTreeMap::new();
                for v in [av0, av1] {
                    a_by_point.insert(point_key(vertex_point(v, entities, &index)?)?, v);
                }
                for v in [bv0, bv1] {
                    b_by_point.insert(point_key(vertex_point(v, entities, &index)?)?, v);
                }
                if a_by_point.keys().collect::<Vec<_>>() != b_by_point.keys().collect::<Vec<_>>() {
                    return None;
                }
                for (key, av) in a_by_point {
                    vertex_uf.union(av, *b_by_point.get(&key)?);
                }
            }
        }
    }

    // Only live exterior loops need seam edge substitution. Rewriting the
    // soon-to-be-deleted interface loops would orphan their duplicate edges
    // before reachability GC and needlessly preserve dead support geometry.
    let mut retained_oes = HashSet::new();
    for plan in &plans {
        for &face in &plan.retained_faces {
            retained_oes.extend(face_oriented_edges(face, entities, &index)?);
        }
    }

    // Work out ORIENTED_EDGE replacements against the old edge endpoints.
    let mut oe_updates = HashMap::<u64, (u64, bool)>::new();
    for entity in entities.iter() {
        let id = entity_id(entity);
        if !retained_oes.contains(&id) {
            continue;
        }
        let Some(record) = simple_record(entity) else {
            continue;
        };
        if record.name != "ORIENTED_EDGE" {
            continue;
        }
        let edge = oriented_edge_element(record)?;
        let canonical = edge_uf.mapped(edge);
        if canonical == edge {
            continue;
        }

        let [v0, v1] = edge_vertices(edge, entities, &index)?;
        let forward = oriented_edge_orientation(record)?;
        let (start, end) = if forward { (v0, v1) } else { (v1, v0) };
        let want = (vertex_uf.mapped(start), vertex_uf.mapped(end));

        let [c0, c1] = edge_vertices(canonical, entities, &index)?;
        let got = (vertex_uf.mapped(c0), vertex_uf.mapped(c1));
        let orientation = if want == got {
            true
        } else if want == (got.1, got.0) {
            false
        } else {
            return None;
        };
        oe_updates.insert(id, (canonical, orientation));
    }

    for entity in entities.iter_mut() {
        let id = entity_id(entity);
        let Some(record) = simple_record_mut(entity) else {
            continue;
        };
        match record.name.as_str() {
            "EDGE_CURVE" => rewrite_edge_vertices(record, &mut vertex_uf)?,
            "ORIENTED_EDGE" => {
                if let Some(&(edge, orientation)) = oe_updates.get(&id) {
                    rewrite_oriented_edge(record, edge, orientation)?;
                }
            }
            _ => {}
        }
    }

    let mut next_id = entities.iter().map(entity_id).max().unwrap_or(0) + 1;
    let mut old_style_to_new = HashMap::new();
    let mut old_solids = HashSet::new();
    let mut old_styles = HashSet::new();
    let mut rep_replacements: HashMap<u64, HashMap<u64, u64>> = HashMap::new();
    let mut stats = PartitionRecoveryStats::default();

    for plan in &plans {
        let shell = push_simple(
            entities,
            &mut next_id,
            "CLOSED_SHELL",
            vec![
                Parameter::String("step-redox welded partition".to_string()),
                Parameter::List(
                    plan.retained_faces
                        .iter()
                        .copied()
                        .map(entity_ref)
                        .collect(),
                ),
            ],
        );
        let solid = push_simple(
            entities,
            &mut next_id,
            "MANIFOLD_SOLID_BREP",
            vec![
                Parameter::String("step-redox welded partition".to_string()),
                entity_ref(shell),
            ],
        );
        let style = push_simple(
            entities,
            &mut next_id,
            "STYLED_ITEM",
            vec![
                Parameter::String(String::new()),
                Parameter::List(
                    plan.style_assignment
                        .iter()
                        .copied()
                        .map(entity_ref)
                        .collect(),
                ),
                entity_ref(solid),
            ],
        );

        let map = rep_replacements.entry(plan.rep).or_default();
        for old in &plan.solids {
            map.insert(*old, solid);
            old_solids.insert(*old);
        }
        for old_style in &plan.old_styles {
            old_style_to_new.insert(*old_style, style);
            old_styles.insert(*old_style);
        }
        stats.components += 1;
        stats.solids_merged += plan.solids.len();
        stats.interfaces_removed += plan.interfaces.len();
    }

    let current_index = build_index(entities);
    for (rep, mapping) in &rep_replacements {
        let &idx = current_index.get(rep)?;
        replace_representation_items(&mut entities[idx], mapping)?;
    }

    for entity in entities.iter_mut() {
        let Some(record) = simple_record_mut(entity) else {
            continue;
        };
        if matches!(
            record.name.as_str(),
            "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION"
                | "PRESENTATION_LAYER_ASSIGNMENT"
        ) {
            replace_and_dedup_direct_ref_lists(&mut record.parameter, &old_style_to_new);
        }
    }

    // Reachability-limited GC: only descendants of obsolete solids/styles are
    // eligible. Retained faces are protected by the newly added shell refs.
    let mut candidate = old_styles.clone();
    candidate.extend(old_solids.iter().copied());
    let mut stack: Vec<u64> = old_solids.iter().copied().collect();
    while let Some(id) = stack.pop() {
        if let Some(children) = refs_before.get(&id) {
            for &child in children {
                if candidate.insert(child) {
                    stack.push(child);
                }
            }
        }
    }
    let mut seeds = old_solids;
    seeds.extend(old_styles.iter().copied());

    let refs_after = entity_ref_map(entities);
    let inbound_after = inbound_map(&refs_after);
    let mut delete = HashSet::new();
    loop {
        let mut changed = false;
        for &id in &candidate {
            if delete.contains(&id) {
                continue;
            }
            let parents = inbound_after.get(&id).cloned().unwrap_or_default();
            let detached = parents.is_empty();
            let child_of_dead = !parents.is_empty() && parents.iter().all(|p| delete.contains(p));
            if seeds.contains(&id) {
                if detached || child_of_dead {
                    delete.insert(id);
                    changed = true;
                }
            } else if detached || child_of_dead {
                // Candidate is constrained to old-solid descendants. Becoming
                // unreferenced after seam welding therefore proves it dead.
                delete.insert(id);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    stats.styles_retargeted = old_styles.len();
    stats.entities_removed = delete.len();
    entities.retain(|entity| !delete.contains(&entity_id(entity)));
    Some(stats)
}

fn face_signature(
    face: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<(FaceKey, [f64; 3])> {
    let record = simple_record(entities.get(*index.get(&face)?)?)?;
    if record.name != "ADVANCED_FACE" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    if params.len() < 4 {
        return None;
    }

    let bounds = entity_ref_list(params.get(1)?)?;
    let plane = entity_ref_value(params.get(2)?)?;
    let same_sense = logical_bool(params.get(3)?)?;
    let (plane_axis, plane_d, raw_normal) = plane_key(plane, entities, index)?;
    let outward = if same_sense {
        raw_normal
    } else {
        raw_normal.map(|x| -x)
    };

    let mut bound_keys = Vec::new();
    for bound in bounds {
        bound_keys.push(bound_key(bound, entities, index)?);
    }
    bound_keys.sort();

    Some((
        FaceKey {
            plane_axis,
            plane_d,
            bounds: bound_keys,
        },
        outward,
    ))
}

fn bound_key(
    bound: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<BoundKey> {
    let entity = entities.get(*index.get(&bound)?)?;
    let record = simple_record(entity)?;
    let kind = match record.name.as_str() {
        "FACE_OUTER_BOUND" => 0,
        "FACE_BOUND" => 1,
        _ => return None,
    };
    let loops = direct_refs_of_type(entity, entities, index, "EDGE_LOOP");
    if loops.len() != 1 {
        return None;
    }
    let loop_record = simple_record(entities.get(*index.get(&loops[0])?)?)?;
    let Parameter::List(params) = &loop_record.parameter else {
        return None;
    };
    let oes = entity_ref_list(params.get(1)?)?;

    let mut edges = Vec::new();
    for oe in oes {
        let oe_record = simple_record(entities.get(*index.get(&oe)?)?)?;
        let edge = oriented_edge_element(oe_record)?;
        edges.push(edge_key(edge, entities, index)?);
    }
    edges.sort();
    Some(BoundKey { kind, edges })
}

fn face_oriented_edges(
    face: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<u64>> {
    let face_record = simple_record(entities.get(*index.get(&face)?)?)?;
    let Parameter::List(params) = &face_record.parameter else {
        return None;
    };
    let bounds = entity_ref_list(params.get(1)?)?;
    let mut out = Vec::new();
    for bound in bounds {
        let bound_entity = entities.get(*index.get(&bound)?)?;
        let loops = direct_refs_of_type(bound_entity, entities, index, "EDGE_LOOP");
        if loops.len() != 1 {
            return None;
        }
        let loop_record = simple_record(entities.get(*index.get(&loops[0])?)?)?;
        let Parameter::List(loop_params) = &loop_record.parameter else {
            return None;
        };
        out.extend(entity_ref_list(loop_params.get(1)?)?);
    }
    Some(out)
}

fn keyed_face_edges(
    face: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<BTreeMap<EdgeKey, (u64, u64)>> {
    let face_record = simple_record(entities.get(*index.get(&face)?)?)?;
    let Parameter::List(params) = &face_record.parameter else {
        return None;
    };
    let bounds = entity_ref_list(params.get(1)?)?;
    let mut out = BTreeMap::new();

    for bound in bounds {
        let bound_entity = entities.get(*index.get(&bound)?)?;
        let loops = direct_refs_of_type(bound_entity, entities, index, "EDGE_LOOP");
        if loops.len() != 1 {
            return None;
        }
        let loop_record = simple_record(entities.get(*index.get(&loops[0])?)?)?;
        let Parameter::List(loop_params) = &loop_record.parameter else {
            return None;
        };
        let oes = entity_ref_list(loop_params.get(1)?)?;
        for oe in oes {
            let oe_record = simple_record(entities.get(*index.get(&oe)?)?)?;
            let edge = oriented_edge_element(oe_record)?;
            let key = edge_key(edge, entities, index)?;
            // Repeated identical edge geometry inside one face would make the
            // seam correspondence ambiguous; skip rather than guess.
            if out.insert(key, (oe, edge)).is_some() {
                return None;
            }
        }
    }
    Some(out)
}

fn edge_key(
    edge: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<EdgeKey> {
    let [v0, v1] = edge_vertices(edge, entities, index)?;
    let mut ends = [
        point_key(vertex_point(v0, entities, index)?)?,
        point_key(vertex_point(v1, entities, index)?)?,
    ];
    if ends[1] < ends[0] {
        ends.swap(0, 1);
    }

    let edge_record = simple_record(entities.get(*index.get(&edge)?)?)?;
    let Parameter::List(params) = &edge_record.parameter else {
        return None;
    };
    let geom = entity_ref_value(params.get(3)?)?;
    let geom_record = simple_record(entities.get(*index.get(&geom)?)?)?;

    match geom_record.name.as_str() {
        "LINE" => Some(EdgeKey::Line { ends }),
        "CIRCLE" => {
            let Parameter::List(circle_params) = &geom_record.parameter else {
                return None;
            };
            let placement = entity_ref_value(circle_params.get(1)?)?;
            let radius = numeric_value(circle_params.get(2)?)?;
            let (center, axis) = axis_location_and_axis(placement, entities, index)?;
            Some(EdgeKey::Circle {
                center: point_key(center)?,
                axis: direction_key(canonical_axis(axis)?)?,
                radius: quantize(radius, POINT_Q)?,
                ends,
            })
        }
        _ => None,
    }
}

fn plane_key(
    plane: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<([i64; 3], i64, [f64; 3])> {
    let record = simple_record(entities.get(*index.get(&plane)?)?)?;
    if record.name != "PLANE" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let placement = entity_ref_value(params.get(1)?)?;
    let (point, normal) = axis_location_and_axis(placement, entities, index)?;
    let raw = normalize(normal)?;
    let canonical = canonical_axis(raw)?;
    Some((
        direction_key(canonical)?,
        quantize(dot(canonical, point), POINT_Q)?,
        raw,
    ))
}

fn axis_location_and_axis(
    placement: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<([f64; 3], [f64; 3])> {
    let record = simple_record(entities.get(*index.get(&placement)?)?)?;
    if record.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let point = point_coords(entity_ref_value(params.get(1)?)?, entities, index)?;
    let axis = direction_coords(entity_ref_value(params.get(2)?)?, entities, index)?;
    Some((point, axis))
}

fn vertex_point(
    vertex: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let record = simple_record(entities.get(*index.get(&vertex)?)?)?;
    if record.name != "VERTEX_POINT" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    point_coords(entity_ref_value(params.get(1)?)?, entities, index)
}

fn point_coords(
    point: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let record = simple_record(entities.get(*index.get(&point)?)?)?;
    if record.name != "CARTESIAN_POINT" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let xyz = numeric_list(params.get(1)?)?;
    (xyz.len() == 3).then(|| [xyz[0], xyz[1], xyz[2]])
}

fn direction_coords(
    direction: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let record = simple_record(entities.get(*index.get(&direction)?)?)?;
    if record.name != "DIRECTION" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let xyz = numeric_list(params.get(1)?)?;
    (xyz.len() == 3).then(|| [xyz[0], xyz[1], xyz[2]])
}

fn edge_vertices(
    edge: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[u64; 2]> {
    let record = simple_record(entities.get(*index.get(&edge)?)?)?;
    if record.name != "EDGE_CURVE" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    Some([
        entity_ref_value(params.get(1)?)?,
        entity_ref_value(params.get(2)?)?,
    ])
}

fn oriented_edge_element(record: &Record) -> Option<u64> {
    if record.name != "ORIENTED_EDGE" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    entity_ref_value(params.get(3)?)
}

fn oriented_edge_orientation(record: &Record) -> Option<bool> {
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    logical_bool(params.get(4)?)
}

fn rewrite_edge_vertices(record: &mut Record, uf: &mut UnionFind) -> Option<()> {
    let Parameter::List(params) = &mut record.parameter else {
        return None;
    };
    if params.len() < 5 {
        return None;
    }
    let v0 = entity_ref_value(&params[1])?;
    let v1 = entity_ref_value(&params[2])?;
    params[1] = entity_ref(uf.mapped(v0));
    params[2] = entity_ref(uf.mapped(v1));
    Some(())
}

fn rewrite_oriented_edge(record: &mut Record, edge: u64, orientation: bool) -> Option<()> {
    let Parameter::List(params) = &mut record.parameter else {
        return None;
    };
    if params.len() < 5 {
        return None;
    }
    params[3] = entity_ref(edge);
    params[4] = Parameter::Enumeration(if orientation { "T" } else { "F" }.to_string());
    Some(())
}

fn representation_items(entities: &[EntityInstance]) -> HashMap<u64, Vec<u64>> {
    let mut out = HashMap::new();
    for entity in entities {
        let id = entity_id(entity);
        let Some(record) = simple_record(entity) else {
            continue;
        };
        if !record.name.contains("SHAPE_REPRESENTATION") {
            continue;
        }
        let Parameter::List(params) = &record.parameter else {
            continue;
        };
        let Some(items) = params.get(1).and_then(entity_ref_list) else {
            continue;
        };
        out.insert(id, items);
    }
    out
}

fn replace_representation_items(
    entity: &mut EntityInstance,
    mapping: &HashMap<u64, u64>,
) -> Option<()> {
    let record = simple_record_mut(entity)?;
    let Parameter::List(params) = &mut record.parameter else {
        return None;
    };
    let Parameter::List(items) = params.get_mut(1)? else {
        return None;
    };
    let mut out = Vec::new();
    let mut emitted = HashSet::new();
    for item in items.iter() {
        let old = entity_ref_value(item)?;
        if let Some(&new) = mapping.get(&old) {
            if emitted.insert(new) {
                out.push(entity_ref(new));
            }
        } else {
            out.push(entity_ref(old));
        }
    }
    *items = out;
    Some(())
}

fn replace_and_dedup_direct_ref_lists(param: &mut Parameter, mapping: &HashMap<u64, u64>) {
    match param {
        Parameter::List(items) => {
            let all_refs = !items.is_empty() && items.iter().all(|p| entity_ref_value(p).is_some());
            if all_refs {
                let mut seen = HashSet::new();
                let mut out = Vec::new();
                for old in items.iter().filter_map(entity_ref_value) {
                    let new = mapping.get(&old).copied().unwrap_or(old);
                    if seen.insert(new) {
                        out.push(entity_ref(new));
                    }
                }
                *items = out;
            } else {
                for item in items {
                    replace_and_dedup_direct_ref_lists(item, mapping);
                }
            }
        }
        Parameter::Typed { parameter, .. } => {
            replace_and_dedup_direct_ref_lists(parameter, mapping);
        }
        _ => {}
    }
}

fn styles_by_target(entities: &[EntityInstance]) -> HashMap<u64, Vec<(u64, Vec<u64>)>> {
    let mut out: HashMap<u64, Vec<(u64, Vec<u64>)>> = HashMap::new();
    for entity in entities {
        let id = entity_id(entity);
        let Some(record) = simple_record(entity) else {
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
        let Some(target) = entity_ref_value(&params[2]) else {
            continue;
        };
        let Some(assignments) = entity_ref_list(&params[1]) else {
            continue;
        };
        out.entry(target).or_default().push((id, assignments));
    }
    out
}

fn direct_refs_of_type(
    entity: &EntityInstance,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    wanted: &str,
) -> Vec<u64> {
    let mut refs = Vec::new();
    visit_entity_refs(entity, &mut |id| {
        if index
            .get(&id)
            .and_then(|idx| simple_record(&entities[*idx]))
            .is_some_and(|record| record.name == wanted)
        {
            refs.push(id);
        }
    });
    refs
}

fn build_index(entities: &[EntityInstance]) -> HashMap<u64, usize> {
    entities
        .iter()
        .enumerate()
        .map(|(idx, entity)| (entity_id(entity), idx))
        .collect()
}

fn entity_ref_map(entities: &[EntityInstance]) -> HashMap<u64, Vec<u64>> {
    let mut out = HashMap::new();
    for entity in entities {
        let id = entity_id(entity);
        let mut refs = Vec::new();
        visit_entity_refs(entity, &mut |child| refs.push(child));
        out.insert(id, refs);
    }
    out
}

fn inbound_map(refs: &HashMap<u64, Vec<u64>>) -> HashMap<u64, Vec<u64>> {
    let mut out: HashMap<u64, Vec<u64>> = HashMap::new();
    for (&parent, children) in refs {
        for &child in children {
            out.entry(child).or_default().push(parent);
        }
    }
    out
}

fn simple_record(entity: &EntityInstance) -> Option<&Record> {
    match entity {
        EntityInstance::Simple { record, .. } => Some(record),
        EntityInstance::Complex { .. } => None,
    }
}

fn simple_record_mut(entity: &mut EntityInstance) -> Option<&mut Record> {
    match entity {
        EntityInstance::Simple { record, .. } => Some(record),
        EntityInstance::Complex { .. } => None,
    }
}

fn entity_id(entity: &EntityInstance) -> u64 {
    match entity {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id,
    }
}

fn entity_ref(id: u64) -> Parameter {
    Parameter::Ref(Name::Entity(id))
}

fn entity_ref_value(param: &Parameter) -> Option<u64> {
    match param {
        Parameter::Ref(Name::Entity(id)) => Some(*id),
        _ => None,
    }
}

fn entity_ref_list(param: &Parameter) -> Option<Vec<u64>> {
    let Parameter::List(items) = param else {
        return None;
    };
    items.iter().map(entity_ref_value).collect()
}

fn numeric_value(param: &Parameter) -> Option<f64> {
    match param {
        Parameter::Integer(v) => Some(*v as f64),
        Parameter::Real(v) => Some(*v),
        _ => None,
    }
}

fn numeric_list(param: &Parameter) -> Option<Vec<f64>> {
    let Parameter::List(items) = param else {
        return None;
    };
    items.iter().map(numeric_value).collect()
}

fn logical_bool(param: &Parameter) -> Option<bool> {
    match param {
        Parameter::Enumeration(v) if v == "T" => Some(true),
        Parameter::Enumeration(v) if v == "F" => Some(false),
        _ => None,
    }
}

fn point_key(p: [f64; 3]) -> Option<[i64; 3]> {
    Some([
        quantize(p[0], POINT_Q)?,
        quantize(p[1], POINT_Q)?,
        quantize(p[2], POINT_Q)?,
    ])
}

fn direction_key(v: [f64; 3]) -> Option<[i64; 3]> {
    Some([
        quantize(v[0], NORMAL_Q)?,
        quantize(v[1], NORMAL_Q)?,
        quantize(v[2], NORMAL_Q)?,
    ])
}

fn quantize(v: f64, q: f64) -> Option<i64> {
    if !v.is_finite() || !q.is_finite() || q <= 0.0 {
        return None;
    }
    let x = (v / q).round();
    if x < i64::MIN as f64 || x > i64::MAX as f64 {
        return None;
    }
    Some(x as i64)
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = dot(v, v).sqrt();
    if !n.is_finite() || n <= 1.0e-15 {
        return None;
    }
    Some([v[0] / n, v[1] / n, v[2] / n])
}

fn canonical_axis(v: [f64; 3]) -> Option<[f64; 3]> {
    let mut v = normalize(v)?;
    for x in v {
        if x.abs() > 1.0e-12 {
            if x < 0.0 {
                v = v.map(|q| -q);
            }
            break;
        }
    }
    Some(v)
}

fn visit_entity_refs(entity: &EntityInstance, f: &mut impl FnMut(u64)) {
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

fn push_simple(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_axis_ignores_sign() {
        assert_eq!(
            direction_key(canonical_axis([1.0, 0.0, 0.0]).unwrap()),
            direction_key(canonical_axis([-1.0, 0.0, 0.0]).unwrap())
        );
    }

    #[test]
    fn union_find_uses_stable_smallest_root() {
        let mut uf = UnionFind::default();
        uf.union(20, 10);
        uf.union(30, 20);
        assert_eq!(uf.find(30), 10);
        assert_eq!(uf.find(20), 10);
    }
}
