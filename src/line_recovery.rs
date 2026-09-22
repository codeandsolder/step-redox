use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use std::collections::{BTreeMap, HashMap, HashSet};

const COLLINEAR_TOL: f64 = 1.0e-12;
const MONOTONIC_TOL: f64 = 1.0e-12;
const DIRECTION_QUANTUM: f64 = 1.0e-12;
const DIRECTION_ENDPOINT_TOL: f64 = 2.0e-13;
const DEGENERATE_CHORD2: f64 = 1.0e-24;

#[derive(Debug, Default, Clone)]
pub(crate) struct LineRecoveryStats {
    pub curves_recovered: usize,
    pub direction_groups: usize,
    pub orphan_points_removed: usize,
}

#[derive(Debug, Clone)]
struct Candidate {
    id: u64,
    name: String,
    poles: Vec<u64>,
    points: Vec<[f64; 3]>,
    unit: [f64; 3],
    direction_key: [i64; 3],
}

#[derive(Debug)]
struct DirectionGroup {
    unit: [f64; 3],
    candidate_ids: Vec<u64>,
}

pub(crate) fn recover_straight_bspline_lines(
    entities: &mut Vec<EntityInstance>,
) -> LineRecoveryStats {
    let mut stats = LineRecoveryStats::default();
    if entities.is_empty() {
        return stats;
    }

    let index = build_index(entities);
    let spline_ids: HashSet<u64> = entities
        .iter()
        .filter_map(|entity| match entity {
            EntityInstance::Simple { id, record } if record.name == "B_SPLINE_CURVE_WITH_KNOTS" => {
                Some(*id)
            }
            _ => None,
        })
        .collect();
    let inbound = inbound_map_for_targets(entities, &spline_ids);

    let mut candidates = Vec::new();
    for entity in entities.iter() {
        let EntityInstance::Simple { id, record } = entity else {
            continue;
        };
        if record.name != "B_SPLINE_CURVE_WITH_KNOTS" {
            continue;
        }

        // Keep this pass deliberately narrow: the original curve must be used
        // exactly once, as the geometric support of one EDGE_CURVE. That
        // avoids changing parameter-space semantics for trimmed curves,
        // pcurves, annotations, or other secondary references.
        let Some(parents) = inbound.get(id) else {
            continue;
        };
        if parents.len() != 1 || !is_edge_curve_geometry(parents[0], *id, entities, &index) {
            continue;
        }
        let edge_id = parents[0];

        let Some(candidate) = analyze_candidate(*id, edge_id, record, entities, &index) else {
            continue;
        };
        candidates.push(candidate);
    }

    if candidates.is_empty() {
        return stats;
    }
    candidates.sort_by_key(|candidate| candidate.id);

    // Quantization is only a coarse bucket. Membership in a shared direction
    // group is subsequently proven against the representative direction with
    // a much tighter endpoint residual, so a quantization collision cannot
    // silently perturb an edge.
    let mut buckets: BTreeMap<[i64; 3], Vec<Candidate>> = BTreeMap::new();
    for candidate in candidates {
        buckets
            .entry(candidate.direction_key)
            .or_default()
            .push(candidate);
    }

    let mut by_id: HashMap<u64, Candidate> = HashMap::new();
    let mut groups = Vec::new();
    for (_, mut bucket) in buckets {
        bucket.sort_by_key(|candidate| candidate.id);
        let mut bucket_groups: Vec<DirectionGroup> = Vec::new();

        for candidate in bucket {
            let mut target = None;
            for (idx, group) in bucket_groups.iter().enumerate() {
                if direction_group_accepts(group.unit, &candidate) {
                    target = Some(idx);
                    break;
                }
            }
            let id = candidate.id;
            if let Some(idx) = target {
                bucket_groups[idx].candidate_ids.push(id);
            } else {
                bucket_groups.push(DirectionGroup {
                    unit: candidate.unit,
                    candidate_ids: vec![id],
                });
            }
            by_id.insert(id, candidate);
        }
        groups.extend(bucket_groups);
    }

    let mut next_id = entities.iter().map(entity_id).max().unwrap_or(0) + 1;
    let mut vector_for_curve = HashMap::new();

    for group in &groups {
        let direction = push_simple(
            entities,
            &mut next_id,
            "DIRECTION",
            vec![
                Parameter::String(String::new()),
                Parameter::List(group.unit.iter().copied().map(Parameter::Real).collect()),
            ],
        );
        let vector = push_simple(
            entities,
            &mut next_id,
            "VECTOR",
            vec![
                Parameter::String(String::new()),
                entity_ref(direction),
                Parameter::Real(1.0),
            ],
        );
        for id in &group.candidate_ids {
            vector_for_curve.insert(*id, vector);
        }
    }

    let mut old_poles = HashSet::new();
    for entity in entities.iter_mut() {
        let EntityInstance::Simple { id, record } = entity else {
            continue;
        };
        let Some(candidate) = by_id.get(id) else {
            continue;
        };
        let Some(&vector) = vector_for_curve.get(id) else {
            continue;
        };

        old_poles.extend(candidate.poles.iter().copied());
        record.name = "LINE".to_string();
        record.parameter = Parameter::List(vec![
            Parameter::String(candidate.name.clone()),
            entity_ref(candidate.poles[0]),
            entity_ref(vector),
        ]);
        stats.curves_recovered += 1;
    }
    stats.direction_groups = groups.len();

    // GC is intentionally limited to former control points. The curve entity
    // itself keeps its ID, and all unrelated geometry/topology remains outside
    // this pass's deletion domain.
    let live_inbound = inbound_counts_for_targets(entities, &old_poles);
    let point_ids: HashSet<u64> = old_poles
        .into_iter()
        .filter(|id| live_inbound.get(id).copied().unwrap_or(0) == 0)
        .filter(|id| {
            index
                .get(id)
                .and_then(|&idx| simple_record(&entities[idx]))
                .is_some_and(|record| record.name == "CARTESIAN_POINT")
        })
        .collect();

    if !point_ids.is_empty() {
        stats.orphan_points_removed = point_ids.len();
        entities.retain(|entity| !point_ids.contains(&entity_id(entity)));
    }

    stats
}

fn analyze_candidate(
    id: u64,
    edge_id: u64,
    record: &Record,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Candidate> {
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    if params.len() != 9 {
        return None;
    }

    let name = match &params[0] {
        Parameter::String(value) => value.clone(),
        _ => return None,
    };
    let degree = integer_value(&params[1])?;
    if degree < 1 {
        return None;
    }
    let degree = usize::try_from(degree).ok()?;

    let poles = entity_ref_list(&params[2])?;
    if poles.len() < degree + 1 {
        return None;
    }

    // Closed/self-intersecting curves are outside the proof used here.
    if !is_false_logical(&params[4]) || !is_false_logical(&params[5]) {
        return None;
    }

    let multiplicities = integer_list(&params[6])?;
    let knots = numeric_list(&params[7])?;
    if multiplicities.len() != knots.len() || knots.len() < 2 {
        return None;
    }
    if multiplicities.first().copied()? != (degree + 1) as i64
        || multiplicities.last().copied()? != (degree + 1) as i64
    {
        return None;
    }
    if multiplicities.iter().any(|value| *value <= 0)
        || multiplicities
            .iter()
            .skip(1)
            .take(multiplicities.len().saturating_sub(2))
            .any(|value| *value > degree as i64)
    {
        return None;
    }
    let expected_sum = poles.len().checked_add(degree)?.checked_add(1)?;
    let actual_sum: usize = multiplicities
        .iter()
        .map(|value| usize::try_from(*value).ok())
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .sum();
    if actual_sum != expected_sum {
        return None;
    }
    if knots.iter().any(|value| !value.is_finite())
        || knots.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return None;
    }

    let points = poles
        .iter()
        .map(|id| point_coords(*id, entities, index))
        .collect::<Option<Vec<_>>>()?;
    if points.iter().flatten().any(|value| !value.is_finite()) {
        return None;
    }

    let first = points[0];
    let last = *points.last()?;
    let chord = sub(last, first);
    let chord2 = dot(chord, chord);
    if chord2 < DEGENERATE_CHORD2 {
        return None;
    }

    for point in points.iter().skip(1).take(points.len().saturating_sub(2)) {
        if point_line_distance(*point, first, last) > COLLINEAR_TOL {
            return None;
        }
    }

    let mut previous = None;
    for point in &points {
        let t = dot(sub(*point, first), chord) / chord2;
        if !t.is_finite() || !(-MONOTONIC_TOL..=1.0 + MONOTONIC_TOL).contains(&t) {
            return None;
        }
        if previous.is_some_and(|prev| t + MONOTONIC_TOL < prev) {
            return None;
        }
        previous = Some(t);
    }

    // The B-spline support and EDGE_CURVE topology are allowed to disagree
    // within model tolerances in real exporter output. A LINE rewrite is only
    // safe when the topological edge vertices are the actual clamped spline
    // endpoints (in either orientation). This rejects rare exporter records
    // whose control polygon looks perfectly straight but whose EDGE_CURVE is
    // trimming a different portion of that support curve.
    if !edge_vertices_match_endpoints(edge_id, first, last, entities, index) {
        return None;
    }

    let chord_len = chord2.sqrt();
    let unit = [
        chord[0] / chord_len,
        chord[1] / chord_len,
        chord[2] / chord_len,
    ];
    let direction_key = unit.map(|value| (value / DIRECTION_QUANTUM).round() as i64);

    Some(Candidate {
        id,
        name,
        poles,
        points,
        unit,
        direction_key,
    })
}

fn direction_group_accepts(representative: [f64; 3], candidate: &Candidate) -> bool {
    if dot(representative, candidate.unit) <= 0.0 {
        return false;
    }

    let (Some(&first), Some(&last)) = (candidate.points.first(), candidate.points.last()) else {
        return false;
    };
    let chord = sub(last, first);
    let projection = dot(chord, representative);
    let projected = add(first, scale(representative, projection));
    norm(sub(last, projected)) <= DIRECTION_ENDPOINT_TOL
}

fn is_edge_curve_geometry(
    parent: u64,
    curve: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> bool {
    let Some(&idx) = index.get(&parent) else {
        return false;
    };
    let Some(record) = simple_record(&entities[idx]) else {
        return false;
    };
    if record.name != "EDGE_CURVE" {
        return false;
    }
    let Parameter::List(params) = &record.parameter else {
        return false;
    };
    params.len() == 5 && entity_ref_value(&params[3]) == Some(curve)
}

fn edge_vertices_match_endpoints(
    edge_id: u64,
    first: [f64; 3],
    last: [f64; 3],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> bool {
    let Some(&idx) = index.get(&edge_id) else {
        return false;
    };
    let Some(record) = simple_record(&entities[idx]) else {
        return false;
    };
    let Parameter::List(params) = &record.parameter else {
        return false;
    };
    if record.name != "EDGE_CURVE" || params.len() != 5 {
        return false;
    }
    let (Some(start), Some(end)) = (entity_ref_value(&params[1]), entity_ref_value(&params[2]))
    else {
        return false;
    };
    let (Some(start), Some(end)) = (
        vertex_coords(start, entities, index),
        vertex_coords(end, entities, index),
    ) else {
        return false;
    };

    let direct = norm(sub(start, first)) <= COLLINEAR_TOL && norm(sub(end, last)) <= COLLINEAR_TOL;
    let reverse = norm(sub(start, last)) <= COLLINEAR_TOL && norm(sub(end, first)) <= COLLINEAR_TOL;
    direct || reverse
}

fn vertex_coords(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let record = simple_record(entities.get(*index.get(&id)?)?)?;
    if record.name != "VERTEX_POINT" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    point_coords(entity_ref_value(params.get(1)?)?, entities, index)
}

fn build_index(entities: &[EntityInstance]) -> HashMap<u64, usize> {
    entities
        .iter()
        .enumerate()
        .map(|(idx, entity)| (entity_id(entity), idx))
        .collect()
}

fn inbound_map_for_targets(
    entities: &[EntityInstance],
    targets: &HashSet<u64>,
) -> HashMap<u64, Vec<u64>> {
    let mut out: HashMap<u64, Vec<u64>> = HashMap::new();
    for entity in entities {
        let parent = entity_id(entity);
        visit_entity_refs(entity, &mut |child| {
            if targets.contains(&child) {
                out.entry(child).or_default().push(parent);
            }
        });
    }
    out
}

fn inbound_counts_for_targets(
    entities: &[EntityInstance],
    targets: &HashSet<u64>,
) -> HashMap<u64, usize> {
    let mut out = HashMap::new();
    for entity in entities {
        visit_entity_refs(entity, &mut |child| {
            if targets.contains(&child) {
                *out.entry(child).or_insert(0) += 1;
            }
        });
    }
    out
}

fn point_coords(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let record = simple_record(entities.get(*index.get(&id)?)?)?;
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
        numeric_value(&coords[0])?,
        numeric_value(&coords[1])?,
        numeric_value(&coords[2])?,
    ])
}

fn simple_record(entity: &EntityInstance) -> Option<&Record> {
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
    let refs = items
        .iter()
        .map(entity_ref_value)
        .collect::<Option<Vec<_>>>()?;
    Some(refs)
}

fn integer_value(param: &Parameter) -> Option<i64> {
    match param {
        Parameter::Integer(value) => Some(*value),
        _ => None,
    }
}

fn integer_list(param: &Parameter) -> Option<Vec<i64>> {
    let Parameter::List(items) = param else {
        return None;
    };
    items.iter().map(integer_value).collect()
}

fn numeric_value(param: &Parameter) -> Option<f64> {
    match param {
        Parameter::Integer(value) => Some(*value as f64),
        Parameter::Real(value) => Some(*value),
        _ => None,
    }
}

fn numeric_list(param: &Parameter) -> Option<Vec<f64>> {
    let Parameter::List(items) = param else {
        return None;
    };
    items.iter().map(numeric_value).collect()
}

fn is_false_logical(param: &Parameter) -> bool {
    matches!(param, Parameter::Enumeration(value) if value == "F")
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

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(v: [f64; 3], factor: f64) -> [f64; 3] {
    [v[0] * factor, v[1] * factor, v[2] * factor]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

fn point_line_distance(point: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let direction = sub(b, a);
    let length2 = dot(direction, direction);
    if length2 == 0.0 {
        return norm(sub(point, a));
    }
    let u = dot(sub(point, a), direction) / length2;
    let projection = add(a, scale(direction, u));
    norm(sub(point, projection))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(id: u64, xyz: [f64; 3]) -> EntityInstance {
        EntityInstance::Simple {
            id,
            record: Record {
                name: "CARTESIAN_POINT".to_string(),
                parameter: Parameter::List(vec![
                    Parameter::String(String::new()),
                    Parameter::List(xyz.into_iter().map(Parameter::Real).collect()),
                ]),
            },
        }
    }

    fn vertex(id: u64, point: u64) -> EntityInstance {
        EntityInstance::Simple {
            id,
            record: Record {
                name: "VERTEX_POINT".to_string(),
                parameter: Parameter::List(vec![
                    Parameter::String(String::new()),
                    entity_ref(point),
                ]),
            },
        }
    }

    fn spline(id: u64, poles: &[u64]) -> EntityInstance {
        EntityInstance::Simple {
            id,
            record: Record {
                name: "B_SPLINE_CURVE_WITH_KNOTS".to_string(),
                parameter: Parameter::List(vec![
                    Parameter::String(String::new()),
                    Parameter::Integer(3),
                    Parameter::List(poles.iter().copied().map(entity_ref).collect()),
                    Parameter::Enumeration("UNSPECIFIED".to_string()),
                    Parameter::Enumeration("F".to_string()),
                    Parameter::Enumeration("F".to_string()),
                    Parameter::List(vec![Parameter::Integer(4), Parameter::Integer(4)]),
                    Parameter::List(vec![Parameter::Real(0.0), Parameter::Real(1.0)]),
                    Parameter::Enumeration("UNSPECIFIED".to_string()),
                ]),
            },
        }
    }

    fn edge(id: u64, a: u64, b: u64, curve: u64) -> EntityInstance {
        EntityInstance::Simple {
            id,
            record: Record {
                name: "EDGE_CURVE".to_string(),
                parameter: Parameter::List(vec![
                    Parameter::String(String::new()),
                    entity_ref(a),
                    entity_ref(b),
                    entity_ref(curve),
                    Parameter::Enumeration("T".to_string()),
                ]),
            },
        }
    }

    #[test]
    fn recovers_strict_straight_clamped_spline_and_only_orphan_poles() {
        let mut entities = vec![
            point(1, [0.0, 0.0, 0.0]),
            point(2, [1.0, 0.0, 0.0]),
            point(3, [2.0, 0.0, 0.0]),
            point(4, [3.0, 0.0, 0.0]),
            vertex(5, 1),
            vertex(6, 4),
            spline(7, &[1, 2, 3, 4]),
            edge(8, 5, 6, 7),
        ];

        let stats = recover_straight_bspline_lines(&mut entities);
        assert_eq!(stats.curves_recovered, 1);
        assert_eq!(stats.direction_groups, 1);
        assert_eq!(stats.orphan_points_removed, 2);

        let index = build_index(&entities);
        let record = simple_record(&entities[index[&7]]).unwrap();
        assert_eq!(record.name, "LINE");
        assert!(index.contains_key(&1));
        assert!(index.contains_key(&4));
        assert!(!index.contains_key(&2));
        assert!(!index.contains_key(&3));
    }

    #[test]
    fn does_not_recover_curved_or_multiply_referenced_splines() {
        let mut curved = vec![
            point(1, [0.0, 0.0, 0.0]),
            point(2, [1.0, 0.1, 0.0]),
            point(3, [2.0, 0.0, 0.0]),
            point(4, [3.0, 0.0, 0.0]),
            vertex(5, 1),
            vertex(6, 4),
            spline(7, &[1, 2, 3, 4]),
            edge(8, 5, 6, 7),
        ];
        assert_eq!(
            recover_straight_bspline_lines(&mut curved).curves_recovered,
            0
        );

        let mut shared = vec![
            point(1, [0.0, 0.0, 0.0]),
            point(2, [1.0, 0.0, 0.0]),
            point(3, [2.0, 0.0, 0.0]),
            point(4, [3.0, 0.0, 0.0]),
            vertex(5, 1),
            vertex(6, 4),
            spline(7, &[1, 2, 3, 4]),
            edge(8, 5, 6, 7),
            edge(9, 5, 6, 7),
        ];
        assert_eq!(
            recover_straight_bspline_lines(&mut shared).curves_recovered,
            0
        );
    }

    #[test]
    fn rejects_straight_support_when_edge_vertices_trim_different_endpoints() {
        let mut entities = vec![
            point(1, [0.0, 0.0, 0.0]),
            point(2, [1.0, 0.0, 0.0]),
            point(3, [2.0, 0.0, 0.0]),
            point(4, [3.0, 0.0, 0.0]),
            point(9, [0.25, 0.0, 0.0]),
            point(10, [2.75, 0.0, 0.0]),
            vertex(5, 9),
            vertex(6, 10),
            spline(7, &[1, 2, 3, 4]),
            edge(8, 5, 6, 7),
        ];

        let stats = recover_straight_bspline_lines(&mut entities);
        assert_eq!(stats.curves_recovered, 0);
        let index = build_index(&entities);
        let record = simple_record(&entities[index[&7]]).unwrap();
        assert_eq!(record.name, "B_SPLINE_CURVE_WITH_KNOTS");
    }

    #[test]
    fn shares_proven_parallel_direction_support() {
        let mut entities = vec![
            point(1, [0.0, 0.0, 0.0]),
            point(2, [1.0, 0.0, 0.0]),
            point(3, [2.0, 0.0, 0.0]),
            point(4, [3.0, 0.0, 0.0]),
            point(11, [0.0, 2.0, 0.0]),
            point(12, [1.0, 2.0, 0.0]),
            point(13, [2.0, 2.0, 0.0]),
            point(14, [3.0, 2.0, 0.0]),
            vertex(5, 1),
            vertex(6, 4),
            vertex(15, 11),
            vertex(16, 14),
            spline(7, &[1, 2, 3, 4]),
            spline(17, &[11, 12, 13, 14]),
            edge(8, 5, 6, 7),
            edge(18, 15, 16, 17),
        ];

        let stats = recover_straight_bspline_lines(&mut entities);
        assert_eq!(stats.curves_recovered, 2);
        assert_eq!(stats.direction_groups, 1);
        assert_eq!(
            entities
                .iter()
                .filter(|entity| simple_record(entity).is_some_and(|r| r.name == "VECTOR"))
                .count(),
            1
        );
    }
}
