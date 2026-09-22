use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use std::collections::{BTreeSet, HashMap, HashSet};

const SURFACE_TYPES: &[&str] = &[
    "PLANE",
    "CYLINDRICAL_SURFACE",
    "TOROIDAL_SURFACE",
    "CONICAL_SURFACE",
    "SPHERICAL_SURFACE",
    "B_SPLINE_SURFACE_WITH_KNOTS",
    "SURFACE_OF_LINEAR_EXTRUSION",
    "SURFACE_OF_REVOLUTION",
];

#[derive(Debug, Default, Clone)]
pub(crate) struct FaceCoalesceStats {
    pub groups: usize,
    pub faces_merged: usize,
    pub faces_removed: usize,
    pub internal_edges_removed: usize,
    pub styles_removed: usize,
    pub entities_removed: usize,
}

#[derive(Debug, Clone)]
struct FaceInfo {
    support: u64,
    sense: bool,
    oes: Vec<(u64, u64)>,
}

#[derive(Debug, Clone)]
struct MergePlan {
    faces: BTreeSet<u64>,
    canonical: u64,
    shell: u64,
    support: u64,
    sense: bool,
    surviving_oes: Vec<u64>,
    internal_edges: HashSet<u64>,
    style_delete: HashSet<u64>,
}

pub(crate) fn coalesce_same_support_faces(
    entities: &mut Vec<EntityInstance>,
) -> FaceCoalesceStats {
    let original = entities.clone();
    match coalesce_inner(entities) {
        Some(stats) => stats,
        None => {
            *entities = original;
            FaceCoalesceStats::default()
        }
    }
}

fn coalesce_inner(entities: &mut Vec<EntityInstance>) -> Option<FaceCoalesceStats> {
    if entities.is_empty() {
        return Some(FaceCoalesceStats::default());
    }

    let index = build_index(entities);
    let refs_before = entity_ref_map(entities);
    let mut face_info = HashMap::<u64, FaceInfo>::new();
    let mut edge_faces: HashMap<u64, BTreeSet<u64>> = HashMap::new();

    for entity in entities.iter() {
        let face = entity_id(entity);
        let Some(record) = simple_record(entity) else { continue; };
        if record.name != "ADVANCED_FACE" { continue; }
        let Some(info) = parse_face_info(face, entities, &index) else { continue; };
        for &(_, edge) in &info.oes {
            edge_faces.entry(edge).or_default().insert(face);
        }
        face_info.insert(face, info);
    }

    // Adjacency exists only across a manifold edge whose two faces use the
    // exact same support entity and the same face sense.
    let mut adjacency: HashMap<u64, BTreeSet<u64>> = HashMap::new();
    for users in edge_faces.values() {
        if users.len() != 2 { continue; }
        let mut it = users.iter();
        let (Some(&a), Some(&b)) = (it.next(), it.next()) else { continue; };
        let (Some(ai), Some(bi)) = (face_info.get(&a), face_info.get(&b)) else { continue; };
        if ai.support != bi.support || ai.sense != bi.sense { continue; }
        adjacency.entry(a).or_default().insert(b);
        adjacency.entry(b).or_default().insert(a);
    }

    if adjacency.is_empty() {
        return Some(FaceCoalesceStats::default());
    }

    let styles = styles_by_target(entities);
    let face_shells = face_shell_owners(entities, &index);

    let mut roots: Vec<u64> = adjacency.keys().copied().collect();
    roots.sort_unstable();
    let mut seen = HashSet::new();
    let mut plans = Vec::<MergePlan>::new();

    for root in roots {
        if seen.contains(&root) { continue; }
        let mut stack = vec![root];
        let mut component = BTreeSet::new();
        while let Some(face) = stack.pop() {
            if !component.insert(face) { continue; }
            seen.insert(face);
            if let Some(neighbors) = adjacency.get(&face) {
                stack.extend(neighbors.iter().copied());
            }
        }
        if component.len() < 2 { continue; }

        let canonical = *component.iter().next()?;
        let canonical_info = face_info.get(&canonical)?;
        let support = canonical_info.support;
        let sense = canonical_info.sense;

        // All faces must belong to exactly the same one shell.
        let Some(first_owners) = face_shells.get(&canonical) else { continue; };
        if first_owners.len() != 1 { continue; }
        if component.iter().any(|f| face_shells.get(f) != Some(first_owners)) {
            continue;
        }
        let shell = *first_owners.iter().next()?;

        // Direct face styles are either absent on every face, or exactly one
        // identical assignment list on every face. Mixed inheritance is not
        // merged because it can change effective appearance.
        let mut style_signature: Option<Option<Vec<u64>>> = None;
        let mut style_delete = HashSet::new();
        let mut style_ok = true;
        for &face in &component {
            let current = match styles.get(&face) {
                None => None,
                Some(v) if v.len() == 1 => Some(v[0].1.clone()),
                Some(_) => { style_ok = false; break; }
            };
            match &style_signature {
                None => style_signature = Some(current.clone()),
                Some(expected) if expected == &current => {}
                Some(_) => { style_ok = false; break; }
            }
            if face != canonical && let Some(v) = styles.get(&face) {
                for (style, _) in v { style_delete.insert(*style); }
            }
        }
        if !style_ok { continue; }

        // Any edge touched by >=2 group faces is an internal seam only if all
        // of its users are inside the group.
        let mut internal_edges = HashSet::new();
        let mut ownership_ok = true;
        for (&edge, users) in &edge_faces {
            let hit = users.iter().filter(|f| component.contains(f)).count();
            if hit >= 2 {
                if users.iter().any(|f| !component.contains(f)) {
                    ownership_ok = false;
                    break;
                }
                internal_edges.insert(edge);
            }
        }
        if !ownership_ok || internal_edges.is_empty() { continue; }

        let mut surviving = Vec::<(u64, u64)>::new();
        for face in &component {
            let Some(info) = face_info.get(face) else { ownership_ok = false; break; };
            for &(oe, edge) in &info.oes {
                if !internal_edges.contains(&edge) {
                    surviving.push((oe, edge));
                }
            }
        }
        if !ownership_ok || surviving.is_empty() { continue; }

        // A group boundary may use each topological edge exactly once.
        let mut edge_counts = HashMap::<u64, usize>::new();
        for &(_, edge) in &surviving {
            *edge_counts.entry(edge).or_default() += 1;
        }
        if edge_counts.values().any(|&n| n != 1) { continue; }

        // Trace by exact VERTEX_POINT identity. The first production version
        // deliberately accepts one loop only; holes/multiple loops require
        // proper nesting classification rather than an edge-count heuristic.
        let Some(loop_oes) = trace_single_loop(&surviving, entities, &index) else { continue; };

        plans.push(MergePlan {
            faces: component,
            canonical,
            shell,
            support,
            sense,
            surviving_oes: loop_oes,
            internal_edges,
            style_delete,
        });
    }

    if plans.is_empty() {
        return Some(FaceCoalesceStats::default());
    }

    let mut next_id = entities.iter().map(entity_id).max().unwrap_or(0) + 1;
    let mut shell_maps: HashMap<u64, HashMap<u64, u64>> = HashMap::new();
    let mut style_delete = HashSet::new();
    let mut candidate = HashSet::new();
    let mut delete_faces = HashSet::new();
    let mut stats = FaceCoalesceStats::default();

    // Candidate GC is restricted to old descendants of every face we rewrite
    // or delete. Shared supports/boundary topology remain protected by live
    // inbound references after the rewrites.
    for plan in &plans {
        for &face in &plan.faces {
            let mut stack = vec![face];
            while let Some(id) = stack.pop() {
                if !candidate.insert(id) { continue; }
                if let Some(children) = refs_before.get(&id) {
                    stack.extend(children.iter().copied());
                }
            }
        }

        let loop_id = push_simple(entities, &mut next_id, "EDGE_LOOP", vec![
            Parameter::String(String::new()),
            Parameter::List(plan.surviving_oes.iter().copied().map(entity_ref).collect()),
        ]);
        let bound_id = push_simple(entities, &mut next_id, "FACE_OUTER_BOUND", vec![
            Parameter::String(String::new()),
            entity_ref(loop_id),
            Parameter::Enumeration("T".to_string()),
        ]);

        let &canonical_idx = index.get(&plan.canonical)?;
        let canonical_name = {
            let record = simple_record(&entities[canonical_idx])?;
            let Parameter::List(params) = &record.parameter else { return None; };
            params.first()?.clone()
        };
        {
            let record = simple_record_mut(&mut entities[canonical_idx])?;
            record.parameter = Parameter::List(vec![
                canonical_name,
                Parameter::List(vec![entity_ref(bound_id)]),
                entity_ref(plan.support),
                Parameter::Enumeration(if plan.sense { "T" } else { "F" }.to_string()),
            ]);
        }

        let map = shell_maps.entry(plan.shell).or_default();
        for &face in &plan.faces {
            map.insert(face, plan.canonical);
            if face != plan.canonical { delete_faces.insert(face); }
        }
        style_delete.extend(plan.style_delete.iter().copied());

        stats.groups += 1;
        stats.faces_merged += plan.faces.len();
        stats.faces_removed += plan.faces.len() - 1;
        stats.internal_edges_removed += plan.internal_edges.len();
    }

    // Rewrite shell face aggregates, keeping the canonical face once at the
    // first member's position.
    let current_index = build_index(entities);
    for (shell, mapping) in &shell_maps {
        let &idx = current_index.get(shell)?;
        rewrite_shell_faces(&mut entities[idx], mapping)?;
    }

    // Remove duplicate direct styles from presentation root aggregates.
    if !style_delete.is_empty() {
        for entity in entities.iter_mut() {
            let Some(record) = simple_record_mut(entity) else { continue; };
            if matches!(
                record.name.as_str(),
                "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION"
                    | "PRESENTATION_LAYER_ASSIGNMENT"
            ) {
                remove_refs_from_direct_lists(&mut record.parameter, &style_delete);
            }
        }
    }

    candidate.extend(style_delete.iter().copied());
    // Recompute references after canonical-face/shell/presentation rewrites.
    let refs_after = entity_ref_map(entities);
    let inbound_after = inbound_map(&refs_after);
    let mut delete = HashSet::new();
    loop {
        let mut changed = false;
        for &id in &candidate {
            if delete.contains(&id) || plans.iter().any(|p| p.canonical == id) {
                continue;
            }
            let parents = inbound_after.get(&id).cloned().unwrap_or_default();
            let detached = parents.is_empty();
            let child_of_dead = !parents.is_empty() && parents.iter().all(|p| delete.contains(p));
            if detached || child_of_dead {
                delete.insert(id);
                changed = true;
            }
        }
        if !changed { break; }
    }

    stats.styles_removed = style_delete.len();
    stats.entities_removed = delete.len();
    entities.retain(|entity| !delete.contains(&entity_id(entity)));
    Some(stats)
}


fn parse_face_info(
    face: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<FaceInfo> {
    let record = simple_record(entities.get(*index.get(&face)?)?)?;
    if record.name != "ADVANCED_FACE" { return None; }
    let Parameter::List(params) = &record.parameter else { return None; };
    if params.len() < 4 { return None; }

    let support = entity_ref_value(params.get(2)?)?;
    let support_record = simple_record(entities.get(*index.get(&support)?)?)?;
    if !SURFACE_TYPES.contains(&support_record.name.as_str()) { return None; }
    let sense = logical_bool(params.get(3)?)?;

    let bounds = entity_ref_list(params.get(1)?)?;
    let mut oes = Vec::new();
    for bound in bounds {
        let bound_record = simple_record(entities.get(*index.get(&bound)?)?)?;
        if !matches!(bound_record.name.as_str(), "FACE_BOUND" | "FACE_OUTER_BOUND") {
            return None;
        }
        let Parameter::List(bound_params) = &bound_record.parameter else { return None; };
        // Reusing ORIENTED_EDGEs under a new .T. outer bound is only trivially
        // semantics-preserving when every source bound already has .T. sense.
        if !logical_bool(bound_params.get(2)?)? { return None; }
        let loop_id = entity_ref_value(bound_params.get(1)?)?;
        let loop_record = simple_record(entities.get(*index.get(&loop_id)?)?)?;
        if loop_record.name != "EDGE_LOOP" { return None; }
        let Parameter::List(loop_params) = &loop_record.parameter else { return None; };
        let oe_ids = entity_ref_list(loop_params.get(1)?)?;
        for oe in oe_ids {
            let oe_record = simple_record(entities.get(*index.get(&oe)?)?)?;
            let edge = oriented_edge_element(oe_record)?;
            oes.push((oe, edge));
        }
    }

    Some(FaceInfo { support, sense, oes })
}

fn trace_single_loop(
    surviving: &[(u64, u64)],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<u64>> {
    let mut start_map: HashMap<u64, u64> = HashMap::new();
    let mut oe_info: HashMap<u64, (u64, u64)> = HashMap::new();

    for &(oe, edge) in surviving {
        let (start, end) = oriented_vertices(oe, edge, entities, index)?;
        if start_map.insert(start, oe).is_some() {
            return None;
        }
        if oe_info.insert(oe, (start, end)).is_some() {
            return None;
        }
    }

    let first = *oe_info.keys().min()?;
    let loop_start = oe_info.get(&first)?.0;
    let mut current = first;
    let mut unused: HashSet<u64> = oe_info.keys().copied().collect();
    let mut sequence = Vec::with_capacity(unused.len());

    loop {
        if !unused.remove(&current) { return None; }
        sequence.push(current);
        let end = oe_info.get(&current)?.1;
        if end == loop_start {
            break;
        }
        current = *start_map.get(&end)?;
        if sequence.len() > oe_info.len() {
            return None;
        }
    }

    // Any remainder would be a second loop/hole; intentionally reject it.
    if !unused.is_empty() { return None; }
    Some(sequence)
}

fn oriented_vertices(
    oe: u64,
    edge: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<(u64, u64)> {
    let [v0, v1] = edge_vertices(edge, entities, index)?;
    let record = simple_record(entities.get(*index.get(&oe)?)?)?;
    let forward = oriented_edge_orientation(record)?;
    Some(if forward { (v0, v1) } else { (v1, v0) })
}

fn edge_vertices(
    edge: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[u64; 2]> {
    let record = simple_record(entities.get(*index.get(&edge)?)?)?;
    if record.name != "EDGE_CURVE" { return None; }
    let Parameter::List(params) = &record.parameter else { return None; };
    Some([
        entity_ref_value(params.get(1)?)?,
        entity_ref_value(params.get(2)?)?,
    ])
}

fn oriented_edge_element(record: &Record) -> Option<u64> {
    if record.name != "ORIENTED_EDGE" { return None; }
    let Parameter::List(params) = &record.parameter else { return None; };
    entity_ref_value(params.get(3)?)
}

fn oriented_edge_orientation(record: &Record) -> Option<bool> {
    let Parameter::List(params) = &record.parameter else { return None; };
    logical_bool(params.get(4)?)
}

fn styles_by_target(
    entities: &[EntityInstance],
) -> HashMap<u64, Vec<(u64, Vec<u64>)>> {
    let mut out: HashMap<u64, Vec<(u64, Vec<u64>)>> = HashMap::new();
    for entity in entities {
        let id = entity_id(entity);
        let Some(record) = simple_record(entity) else { continue; };
        if record.name != "STYLED_ITEM" { continue; }
        let Parameter::List(params) = &record.parameter else { continue; };
        if params.len() != 3 { continue; }
        let Some(target) = entity_ref_value(&params[2]) else { continue; };
        let Some(assignments) = entity_ref_list(&params[1]) else { continue; };
        out.entry(target).or_default().push((id, assignments));
    }
    out
}

fn face_shell_owners(
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> HashMap<u64, BTreeSet<u64>> {
    let mut out: HashMap<u64, BTreeSet<u64>> = HashMap::new();
    for entity in entities {
        let shell = entity_id(entity);
        let Some(record) = simple_record(entity) else { continue; };
        if !matches!(record.name.as_str(), "OPEN_SHELL" | "CLOSED_SHELL") { continue; }
        let Parameter::List(params) = &record.parameter else { continue; };
        let Some(faces) = params.get(1).and_then(entity_ref_list) else { continue; };
        for face in faces {
            if index.get(&face)
                .and_then(|idx| simple_record(&entities[*idx]))
                .is_some_and(|r| r.name == "ADVANCED_FACE")
            {
                out.entry(face).or_default().insert(shell);
            }
        }
    }
    out
}

fn rewrite_shell_faces(
    shell: &mut EntityInstance,
    mapping: &HashMap<u64, u64>,
) -> Option<()> {
    let record = simple_record_mut(shell)?;
    if !matches!(record.name.as_str(), "OPEN_SHELL" | "CLOSED_SHELL") { return None; }
    let Parameter::List(params) = &mut record.parameter else { return None; };
    let Parameter::List(faces) = params.get_mut(1)? else { return None; };

    let canonical_values: HashSet<u64> = mapping.values().copied().collect();
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(faces.len());
    for face_param in faces.iter() {
        let old = entity_ref_value(face_param)?;
        let new = mapping.get(&old).copied().unwrap_or(old);
        if canonical_values.contains(&new) {
            if seen.insert(new) { out.push(entity_ref(new)); }
        } else {
            out.push(entity_ref(new));
        }
    }
    *faces = out;
    if let Some(Parameter::String(name)) = params.get_mut(0)
        && (name.is_empty() || name == "NONE")
    {
        *name = "step-redox merged same-support faces".to_string();
    }
    Some(())
}

fn remove_refs_from_direct_lists(param: &mut Parameter, drop: &HashSet<u64>) {
    match param {
        Parameter::List(items) => {
            let all_refs = !items.is_empty() && items.iter().all(|p| entity_ref_value(p).is_some());
            if all_refs {
                items.retain(|p| entity_ref_value(p).is_none_or(|id| !drop.contains(&id)));
            } else {
                for item in items {
                    remove_refs_from_direct_lists(item, drop);
                }
            }
        }
        Parameter::Typed { parameter, .. } => remove_refs_from_direct_lists(parameter, drop),
        _ => {}
    }
}

fn build_index(entities: &[EntityInstance]) -> HashMap<u64, usize> {
    entities.iter().enumerate().map(|(idx, entity)| (entity_id(entity), idx)).collect()
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
    let Parameter::List(items) = param else { return None; };
    items.iter().map(entity_ref_value).collect()
}

fn logical_bool(param: &Parameter) -> Option<bool> {
    match param {
        Parameter::Enumeration(v) if v == "T" => Some(true),
        Parameter::Enumeration(v) if v == "F" => Some(false),
        _ => None,
    }
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
            for item in items { visit_param_refs(item, f); }
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

    fn simple(id: u64, name: &str, params: Vec<Parameter>) -> EntityInstance {
        EntityInstance::Simple {
            id,
            record: Record {
                name: name.to_string(),
                parameter: Parameter::List(params),
            },
        }
    }

    fn edge(id: u64, a: u64, b: u64) -> EntityInstance {
        simple(
            id,
            "EDGE_CURVE",
            vec![
                Parameter::String(String::new()),
                entity_ref(a),
                entity_ref(b),
                entity_ref(999),
                Parameter::Enumeration("T".to_string()),
            ],
        )
    }

    fn oe(id: u64, edge: u64) -> EntityInstance {
        simple(
            id,
            "ORIENTED_EDGE",
            vec![
                Parameter::String(String::new()),
                Parameter::Omitted,
                Parameter::Omitted,
                entity_ref(edge),
                Parameter::Enumeration("T".to_string()),
            ],
        )
    }

    #[test]
    fn traces_one_exact_topological_loop() {
        let entities = vec![
            edge(10, 1, 2),
            edge(11, 2, 3),
            edge(12, 3, 1),
            oe(20, 10),
            oe(21, 11),
            oe(22, 12),
        ];
        let index = build_index(&entities);
        let loop_oes =
            trace_single_loop(&[(20, 10), (21, 11), (22, 12)], &entities, &index).unwrap();
        assert_eq!(loop_oes.len(), 3);
        assert_eq!(loop_oes.iter().copied().collect::<HashSet<_>>(), HashSet::from([20, 21, 22]));
    }

    #[test]
    fn rejects_two_disjoint_boundary_loops() {
        let entities = vec![
            edge(10, 1, 2),
            edge(11, 2, 3),
            edge(12, 3, 1),
            edge(13, 4, 5),
            edge(14, 5, 6),
            edge(15, 6, 4),
            oe(20, 10),
            oe(21, 11),
            oe(22, 12),
            oe(23, 13),
            oe(24, 14),
            oe(25, 15),
        ];
        let index = build_index(&entities);
        assert!(trace_single_loop(
            &[(20, 10), (21, 11), (22, 12), (23, 13), (24, 14), (25, 15)],
            &entities,
            &index,
        )
        .is_none());
    }
}
