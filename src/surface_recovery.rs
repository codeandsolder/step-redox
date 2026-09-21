use crate::instances::{
    build_index, cartesian_point, entity_id, entity_ref, entity_ref_value, number, push_simple,
    simple_record, visit_entity_refs,
};
use ruststep::ast::{EntityInstance, Parameter, Record, SubSuperRecord};
use std::collections::{HashMap, HashSet};

// Geometry below 10 nm is exporter noise for the ECAD/display corpus.  Do not
// let sub-1e-5 mm differences prevent semantic recovery; parameter/weight
// checks remain stricter where they affect the actual STEP parameterization.
const GEOMETRY_TOLERANCE: f64 = 1.0e-5;
const WEIGHT_TOLERANCE: f64 = 1.0e-12;
const PARAMETER_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Default, Clone)]
pub(crate) struct VExtrusionStats {
    pub surfaces_recovered: usize,
    pub rational_surfaces_recovered: usize,
    pub profile_curves_created: usize,
    pub orphan_points_removed: usize,
}

#[derive(Debug, Clone)]
struct ParsedSurface {
    name: Parameter,
    u_degree: usize,
    v_degree: usize,
    control_points: Vec<Vec<u64>>,
    u_closed: Parameter,
    self_intersect: Parameter,
    u_multiplicities: Parameter,
    u_knots: Parameter,
    knot_spec: Parameter,
    weights: Option<Vec<Vec<Parameter>>>,
}

#[derive(Debug, Clone)]
struct Candidate {
    surface_id: u64,
    name: Parameter,
    u_degree: usize,
    profile_points: Vec<u64>,
    profile_weights: Option<Vec<Parameter>>,
    u_closed: Parameter,
    self_intersect: Parameter,
    u_multiplicities: Parameter,
    u_knots: Parameter,
    knot_spec: Parameter,
    extrusion: [f64; 3],
    old_points: Vec<u64>,
}

pub(crate) fn recover_v_extrusion_surfaces(entities: &mut Vec<EntityInstance>) -> VExtrusionStats {
    let mut stats = VExtrusionStats::default();
    if entities.is_empty() {
        return stats;
    }

    let index = build_index(entities);
    let candidate_ids: HashSet<u64> = entities
        .iter()
        .filter_map(|entity| match entity {
            EntityInstance::Complex { id, subsuper } => subsuper
                .0
                .iter()
                .any(|record| record.name == "RATIONAL_B_SPLINE_SURFACE")
                .then_some(*id),
            EntityInstance::Simple { id, record } => {
                (record.name == "B_SPLINE_SURFACE_WITH_KNOTS").then_some(*id)
            }
        })
        .collect();
    if candidate_ids.is_empty() {
        return stats;
    }

    let parents = targeted_inbound_parents(entities, &candidate_ids);
    let mut candidates = Vec::new();

    for &surface_id in &candidate_ids {
        if !surface_is_face_support_only(surface_id, entities, &index, &parents) {
            continue;
        }
        let Some(parsed) = parse_surface(surface_id, entities, &index) else {
            continue;
        };
        let Some(candidate) = detect_v_extrusion(surface_id, &parsed, entities, &index) else {
            continue;
        };
        candidates.push(candidate);
    }
    candidates.sort_by_key(|candidate| candidate.surface_id);
    if candidates.is_empty() {
        return stats;
    }

    let mut next_id = entities.iter().map(entity_id).max().unwrap_or(0) + 1;
    let mut old_points = HashSet::new();

    for candidate in candidates {
        let length = norm(candidate.extrusion);
        if !length.is_finite() || length <= GEOMETRY_TOLERANCE {
            continue;
        }
        let direction = [
            candidate.extrusion[0] / length,
            candidate.extrusion[1] / length,
            candidate.extrusion[2] / length,
        ];

        let profile_curve_id = next_id;
        next_id += 1;
        let profile_curve = if let Some(weights) = &candidate.profile_weights {
            stats.rational_surfaces_recovered += 1;
            rational_bspline_curve(
                profile_curve_id,
                candidate.u_degree,
                &candidate.profile_points,
                weights,
                candidate.u_closed.clone(),
                candidate.self_intersect.clone(),
                candidate.u_multiplicities.clone(),
                candidate.u_knots.clone(),
                candidate.knot_spec.clone(),
                candidate.name.clone(),
            )
        } else {
            bspline_curve_with_knots(
                profile_curve_id,
                candidate.u_degree,
                &candidate.profile_points,
                candidate.u_closed.clone(),
                candidate.self_intersect.clone(),
                candidate.u_multiplicities.clone(),
                candidate.u_knots.clone(),
                candidate.knot_spec.clone(),
                candidate.name.clone(),
            )
        };
        entities.push(profile_curve);

        let direction_id = push_simple(
            entities,
            &mut next_id,
            "DIRECTION",
            vec![
                Parameter::String(String::new()),
                Parameter::List(direction.into_iter().map(Parameter::Real).collect()),
            ],
        );
        let vector_id = push_simple(
            entities,
            &mut next_id,
            "VECTOR",
            vec![
                Parameter::String(String::new()),
                entity_ref(direction_id),
                Parameter::Real(length),
            ],
        );

        let Some(surface_idx) = entities
            .iter()
            .position(|entity| entity_id(entity) == candidate.surface_id)
        else {
            continue;
        };
        entities[surface_idx] = EntityInstance::Simple {
            id: candidate.surface_id,
            record: Record {
                name: "SURFACE_OF_LINEAR_EXTRUSION".to_string(),
                parameter: Parameter::List(vec![
                    candidate.name,
                    entity_ref(profile_curve_id),
                    entity_ref(vector_id),
                ]),
            },
        };

        old_points.extend(candidate.old_points);
        stats.surfaces_recovered += 1;
        stats.profile_curves_created += 1;
    }

    if stats.surfaces_recovered == 0 {
        return stats;
    }

    let live_counts = targeted_inbound_counts(entities, &old_points);
    let remove: HashSet<u64> = old_points
        .into_iter()
        .filter(|id| live_counts.get(id).copied().unwrap_or(0) == 0)
        .collect();
    stats.orphan_points_removed = remove.len();
    entities.retain(|entity| !remove.contains(&entity_id(entity)));
    stats
}

fn parse_surface(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<ParsedSurface> {
    match &entities[*index.get(&id)?] {
        EntityInstance::Simple { record, .. } if record.name == "B_SPLINE_SURFACE_WITH_KNOTS" => {
            parse_simple_bspline_surface(record)
        }
        EntityInstance::Complex { subsuper, .. } => parse_complex_rational_surface(subsuper),
        _ => None,
    }
}

fn parse_simple_bspline_surface(record: &Record) -> Option<ParsedSurface> {
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    // Flattened B_SPLINE_SURFACE_WITH_KNOTS:
    // name, u_degree, v_degree, poles, form, u_closed, v_closed,
    // self_intersect, u_mults, v_mults, u_knots, v_knots, knot_spec.
    if params.len() != 13 {
        return None;
    }
    let u_degree = positive_degree(params.get(1)?)?;
    let v_degree = positive_degree(params.get(2)?)?;
    let control_points = parse_control_points(params.get(3)?)?;
    if !is_false(params.get(6)?) || !canonical_unit_linear_v(params.get(9)?, params.get(11)?)? {
        return None;
    }

    Some(ParsedSurface {
        name: params.first()?.clone(),
        u_degree,
        v_degree,
        control_points,
        u_closed: params.get(5)?.clone(),
        self_intersect: params.get(7)?.clone(),
        u_multiplicities: params.get(8)?.clone(),
        u_knots: params.get(10)?.clone(),
        knot_spec: params.get(12)?.clone(),
        weights: None,
    })
}

fn parse_complex_rational_surface(subsuper: &SubSuperRecord) -> Option<ParsedSurface> {
    let bspline = record_by_name(subsuper, "B_SPLINE_SURFACE")?;
    let knots = record_by_name(subsuper, "B_SPLINE_SURFACE_WITH_KNOTS")?;
    let rational = record_by_name(subsuper, "RATIONAL_B_SPLINE_SURFACE")?;
    let repr = record_by_name(subsuper, "REPRESENTATION_ITEM")?;

    let Parameter::List(bspline_params) = &bspline.parameter else {
        return None;
    };
    if bspline_params.len() != 7 {
        return None;
    }
    let u_degree = positive_degree(bspline_params.first()?)?;
    let v_degree = positive_degree(bspline_params.get(1)?)?;
    let control_points = parse_control_points(bspline_params.get(2)?)?;

    let Parameter::List(knot_params) = &knots.parameter else {
        return None;
    };
    if knot_params.len() != 5 {
        return None;
    }

    // Keep the transform parameterization-preserving for existing p-curves.
    // With degree 1, two V poles, clamped multiplicities (2,2), and knots
    // exactly spanning [0,1], the STEP B-spline surface is
    // S(u,v) = C(u) + v * D, exactly the native extrusion parameterization.
    // A different knot interval would describe the same locus but a different
    // V parameter mapping and would require rewriting every trimming p-curve.
    if !is_false(bspline_params.get(5)?)
        || !canonical_unit_linear_v(knot_params.get(1)?, knot_params.get(3)?)?
    {
        return None;
    }

    let Parameter::List(rational_params) = &rational.parameter else {
        return None;
    };
    let Parameter::List(weight_rows) = rational_params.first()? else {
        return None;
    };
    let mut weights = Vec::with_capacity(weight_rows.len());
    for row in weight_rows {
        let Parameter::List(items) = row else {
            return None;
        };
        weights.push(items.clone());
    }
    if weights.len() != control_points.len()
        || weights
            .iter()
            .zip(control_points.iter())
            .any(|(weights, points)| weights.len() != points.len())
    {
        return None;
    }

    let Parameter::List(repr_params) = &repr.parameter else {
        return None;
    };
    let name = repr_params
        .first()
        .cloned()
        .unwrap_or(Parameter::String(String::new()));

    Some(ParsedSurface {
        name,
        u_degree,
        v_degree,
        control_points,
        u_closed: bspline_params.get(4)?.clone(),
        self_intersect: bspline_params.get(6)?.clone(),
        u_multiplicities: knot_params.first()?.clone(),
        u_knots: knot_params.get(2)?.clone(),
        knot_spec: knot_params.get(4)?.clone(),
        weights: Some(weights),
    })
}

fn parse_control_points(parameter: &Parameter) -> Option<Vec<Vec<u64>>> {
    let Parameter::List(rows) = parameter else {
        return None;
    };
    let mut control_points = Vec::with_capacity(rows.len());
    for row in rows {
        let Parameter::List(items) = row else {
            return None;
        };
        let ids: Option<Vec<u64>> = items.iter().map(entity_ref_value).collect();
        control_points.push(ids?);
    }
    if control_points.is_empty()
        || control_points[0].is_empty()
        || control_points
            .iter()
            .any(|row| row.len() != control_points[0].len())
    {
        return None;
    }
    Some(control_points)
}

fn positive_degree(parameter: &Parameter) -> Option<usize> {
    let degree = integer(parameter)?;
    (degree > 0).then_some(degree as usize)
}

fn is_false(parameter: &Parameter) -> bool {
    matches!(parameter, Parameter::Enumeration(value) if value == "F")
}

fn detect_v_extrusion(
    surface_id: u64,
    surface: &ParsedSurface,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Candidate> {
    // Deliberately only recover the parameter-order-preserving case:
    // S(u,v) = profile(u) + v*D. U-sweep recovery would swap the
    // surface parameters and was observed to change how OCCT rebuilds
    // trims/pcurves even when the surface locus itself is exact.
    if surface.v_degree != 1
        || surface.control_points.len() < 2
        || surface.control_points.iter().any(|row| row.len() != 2)
    {
        return None;
    }

    let mut shared_delta = None;
    let mut profile_points = Vec::with_capacity(surface.control_points.len());
    let mut profile_weights = surface
        .weights
        .as_ref()
        .map(|_| Vec::with_capacity(surface.control_points.len()));

    for (row_idx, point_row) in surface.control_points.iter().enumerate() {
        let p0 = cartesian_point(point_row[0], entities, index)?;
        let p1 = cartesian_point(point_row[1], entities, index)?;
        let delta = sub(p1, p0);
        if norm(delta) <= GEOMETRY_TOLERANCE {
            return None;
        }
        if let Some(expected) = shared_delta {
            if distance(delta, expected) > GEOMETRY_TOLERANCE {
                return None;
            }
        } else {
            shared_delta = Some(delta);
        }

        if let Some(weights) = &surface.weights {
            let weight_row = weights.get(row_idx)?;
            let w0 = number(weight_row.first()?)?;
            let w1 = number(weight_row.get(1)?)?;
            if !w0.is_finite() || !w1.is_finite() || (w1 - w0).abs() > WEIGHT_TOLERANCE {
                return None;
            }
            profile_weights.as_mut()?.push(weight_row[0].clone());
        }
        profile_points.push(point_row[0]);
    }

    let old_points = surface.control_points.iter().flatten().copied().collect();
    Some(Candidate {
        surface_id,
        name: surface.name.clone(),
        u_degree: surface.u_degree,
        profile_points,
        profile_weights,
        u_closed: surface.u_closed.clone(),
        self_intersect: surface.self_intersect.clone(),
        u_multiplicities: surface.u_multiplicities.clone(),
        u_knots: surface.u_knots.clone(),
        knot_spec: surface.knot_spec.clone(),
        extrusion: shared_delta?,
        old_points,
    })
}

#[allow(clippy::too_many_arguments)]
fn bspline_curve_with_knots(
    id: u64,
    degree: usize,
    points: &[u64],
    closed: Parameter,
    self_intersect: Parameter,
    multiplicities: Parameter,
    knots: Parameter,
    knot_spec: Parameter,
    name: Parameter,
) -> EntityInstance {
    EntityInstance::Simple {
        id,
        record: Record {
            name: "B_SPLINE_CURVE_WITH_KNOTS".to_string(),
            parameter: Parameter::List(vec![
                name,
                Parameter::Integer(degree as i64),
                Parameter::List(points.iter().copied().map(entity_ref).collect()),
                Parameter::Enumeration("UNSPECIFIED".to_string()),
                closed,
                self_intersect,
                multiplicities,
                knots,
                knot_spec,
            ]),
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn rational_bspline_curve(
    id: u64,
    degree: usize,
    points: &[u64],
    weights: &[Parameter],
    closed: Parameter,
    self_intersect: Parameter,
    multiplicities: Parameter,
    knots: Parameter,
    knot_spec: Parameter,
    name: Parameter,
) -> EntityInstance {
    EntityInstance::Complex {
        id,
        subsuper: SubSuperRecord(vec![
            empty_record("BOUNDED_CURVE"),
            Record {
                name: "B_SPLINE_CURVE".to_string(),
                parameter: Parameter::List(vec![
                    Parameter::Integer(degree as i64),
                    Parameter::List(points.iter().copied().map(entity_ref).collect()),
                    Parameter::Enumeration("UNSPECIFIED".to_string()),
                    closed,
                    self_intersect,
                ]),
            },
            Record {
                name: "B_SPLINE_CURVE_WITH_KNOTS".to_string(),
                parameter: Parameter::List(vec![multiplicities, knots, knot_spec]),
            },
            empty_record("CURVE"),
            empty_record("GEOMETRIC_REPRESENTATION_ITEM"),
            Record {
                name: "RATIONAL_B_SPLINE_CURVE".to_string(),
                parameter: Parameter::List(vec![Parameter::List(weights.to_vec())]),
            },
            Record {
                name: "REPRESENTATION_ITEM".to_string(),
                parameter: Parameter::List(vec![name]),
            },
        ]),
    }
}

fn empty_record(name: &str) -> Record {
    Record {
        name: name.to_string(),
        parameter: Parameter::List(Vec::new()),
    }
}

fn record_by_name<'a>(subsuper: &'a SubSuperRecord, name: &str) -> Option<&'a Record> {
    subsuper.0.iter().find(|record| record.name == name)
}

fn surface_is_face_support_only(
    surface_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    parents: &HashMap<u64, Vec<u64>>,
) -> bool {
    let Some(parent_ids) = parents.get(&surface_id) else {
        return false;
    };
    !parent_ids.is_empty()
        && parent_ids.iter().all(|parent_id| {
            let Some(&parent_idx) = index.get(parent_id) else {
                return false;
            };
            let Some(record) = simple_record(&entities[parent_idx]) else {
                return false;
            };
            if record.name != "ADVANCED_FACE" {
                return false;
            }
            let Parameter::List(params) = &record.parameter else {
                return false;
            };
            params.get(2).and_then(entity_ref_value) == Some(surface_id)
        })
}

fn targeted_inbound_parents(
    entities: &[EntityInstance],
    targets: &HashSet<u64>,
) -> HashMap<u64, Vec<u64>> {
    let mut out = HashMap::<u64, Vec<u64>>::new();
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

fn targeted_inbound_counts(
    entities: &[EntityInstance],
    targets: &HashSet<u64>,
) -> HashMap<u64, usize> {
    let mut out = HashMap::<u64, usize>::new();
    for entity in entities {
        visit_entity_refs(entity, &mut |child| {
            if targets.contains(&child) {
                *out.entry(child).or_default() += 1;
            }
        });
    }
    out
}

fn canonical_unit_linear_v(multiplicities: &Parameter, knots: &Parameter) -> Option<bool> {
    let Parameter::List(multiplicities) = multiplicities else {
        return Some(false);
    };
    let Parameter::List(knots) = knots else {
        return Some(false);
    };
    if multiplicities.len() != 2 || knots.len() != 2 {
        return Some(false);
    }
    if integer(&multiplicities[0])? != 2 || integer(&multiplicities[1])? != 2 {
        return Some(false);
    }
    let v0 = number(&knots[0])?;
    let v1 = number(&knots[1])?;
    Some(
        v0.is_finite()
            && v1.is_finite()
            && v0.abs() <= PARAMETER_TOLERANCE
            && (v1 - 1.0).abs() <= PARAMETER_TOLERANCE,
    )
}

fn integer(parameter: &Parameter) -> Option<i64> {
    match parameter {
        Parameter::Integer(value) => Some(*value),
        _ => None,
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
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

    fn rational_surface(id: u64, rows: [[u64; 2]; 3], weights: [[f64; 2]; 3]) -> EntityInstance {
        EntityInstance::Complex {
            id,
            subsuper: SubSuperRecord(vec![
                empty_record("BOUNDED_SURFACE"),
                Record {
                    name: "B_SPLINE_SURFACE".to_string(),
                    parameter: Parameter::List(vec![
                        Parameter::Integer(2),
                        Parameter::Integer(1),
                        Parameter::List(
                            rows.into_iter()
                                .map(|row| {
                                    Parameter::List(row.into_iter().map(entity_ref).collect())
                                })
                                .collect(),
                        ),
                        Parameter::Enumeration("UNSPECIFIED".to_string()),
                        Parameter::Enumeration("F".to_string()),
                        Parameter::Enumeration("F".to_string()),
                        Parameter::Enumeration("F".to_string()),
                    ]),
                },
                Record {
                    name: "B_SPLINE_SURFACE_WITH_KNOTS".to_string(),
                    parameter: Parameter::List(vec![
                        Parameter::List(vec![Parameter::Integer(3), Parameter::Integer(3)]),
                        Parameter::List(vec![Parameter::Integer(2), Parameter::Integer(2)]),
                        Parameter::List(vec![Parameter::Real(0.0), Parameter::Real(1.0)]),
                        Parameter::List(vec![Parameter::Real(0.0), Parameter::Real(1.0)]),
                        Parameter::Enumeration("UNSPECIFIED".to_string()),
                    ]),
                },
                empty_record("GEOMETRIC_REPRESENTATION_ITEM"),
                Record {
                    name: "RATIONAL_B_SPLINE_SURFACE".to_string(),
                    parameter: Parameter::List(vec![Parameter::List(
                        weights
                            .into_iter()
                            .map(|row| {
                                Parameter::List(row.into_iter().map(Parameter::Real).collect())
                            })
                            .collect(),
                    )]),
                },
                Record {
                    name: "REPRESENTATION_ITEM".to_string(),
                    parameter: Parameter::List(vec![Parameter::String(String::new())]),
                },
                empty_record("SURFACE"),
            ]),
        }
    }

    fn bspline_surface(id: u64, rows: [[u64; 2]; 3]) -> EntityInstance {
        EntityInstance::Simple {
            id,
            record: Record {
                name: "B_SPLINE_SURFACE_WITH_KNOTS".to_string(),
                parameter: Parameter::List(vec![
                    Parameter::String(String::new()),
                    Parameter::Integer(2),
                    Parameter::Integer(1),
                    Parameter::List(
                        rows.into_iter()
                            .map(|row| Parameter::List(row.into_iter().map(entity_ref).collect()))
                            .collect(),
                    ),
                    Parameter::Enumeration("UNSPECIFIED".to_string()),
                    Parameter::Enumeration("F".to_string()),
                    Parameter::Enumeration("F".to_string()),
                    Parameter::Enumeration("F".to_string()),
                    Parameter::List(vec![Parameter::Integer(3), Parameter::Integer(3)]),
                    Parameter::List(vec![Parameter::Integer(2), Parameter::Integer(2)]),
                    Parameter::List(vec![Parameter::Real(0.0), Parameter::Real(1.0)]),
                    Parameter::List(vec![Parameter::Real(0.0), Parameter::Real(1.0)]),
                    Parameter::Enumeration("UNSPECIFIED".to_string()),
                ]),
            },
        }
    }

    fn face(id: u64, surface: u64) -> EntityInstance {
        EntityInstance::Simple {
            id,
            record: Record {
                name: "ADVANCED_FACE".to_string(),
                parameter: Parameter::List(vec![
                    Parameter::String(String::new()),
                    Parameter::List(Vec::new()),
                    entity_ref(surface),
                    Parameter::Enumeration("T".to_string()),
                ]),
            },
        }
    }

    #[test]
    fn recovers_parameter_order_preserving_non_rational_v_extrusion() {
        let mut entities = vec![
            point(1, [0.0, 0.0, 0.0]),
            point(2, [0.0, 2.0, 0.0]),
            point(3, [1.0, 0.0, 0.0]),
            point(4, [1.0, 2.0, 0.0]),
            point(5, [2.0, 0.0, 0.0]),
            point(6, [2.0, 2.0, 0.0]),
            bspline_surface(10, [[1, 2], [3, 4], [5, 6]]),
            face(11, 10),
        ];
        let stats = recover_v_extrusion_surfaces(&mut entities);
        assert_eq!(stats.surfaces_recovered, 1);
        assert_eq!(stats.rational_surfaces_recovered, 0);
        assert_eq!(stats.profile_curves_created, 1);

        let index = build_index(&entities);
        let surface = simple_record(&entities[index[&10]]).unwrap();
        assert_eq!(surface.name, "SURFACE_OF_LINEAR_EXTRUSION");
        assert!(entities.iter().any(|entity| matches!(
            entity,
            EntityInstance::Simple { record, .. }
                if record.name == "B_SPLINE_CURVE_WITH_KNOTS"
        )));
    }

    #[test]
    fn recovers_parameter_order_preserving_rational_v_extrusion() {
        let mut entities = vec![
            point(1, [0.0, 0.0, 0.0]),
            point(2, [0.0, 2.0, 0.0]),
            point(3, [1.0, 0.0, 0.0]),
            point(4, [1.0, 2.0, 0.0]),
            point(5, [2.0, 0.0, 0.0]),
            point(6, [2.0, 2.0, 0.0]),
            rational_surface(
                10,
                [[1, 2], [3, 4], [5, 6]],
                [[1.0, 1.0], [0.8, 0.8], [1.0, 1.0]],
            ),
            face(11, 10),
        ];
        let stats = recover_v_extrusion_surfaces(&mut entities);
        assert_eq!(stats.surfaces_recovered, 1);
        assert_eq!(stats.rational_surfaces_recovered, 1);
        assert_eq!(stats.profile_curves_created, 1);
        let index = build_index(&entities);
        let surface = simple_record(&entities[index[&10]]).unwrap();
        assert_eq!(surface.name, "SURFACE_OF_LINEAR_EXTRUSION");
        assert!(entities.iter().any(|entity| matches!(
            entity,
            EntityInstance::Complex { subsuper, .. }
                if subsuper.0.iter().any(|r| r.name == "RATIONAL_B_SPLINE_CURVE")
        )));
    }

    #[test]
    fn rejects_weight_change_across_sweep() {
        let mut entities = vec![
            point(1, [0.0, 0.0, 0.0]),
            point(2, [0.0, 2.0, 0.0]),
            point(3, [1.0, 0.0, 0.0]),
            point(4, [1.0, 2.0, 0.0]),
            point(5, [2.0, 0.0, 0.0]),
            point(6, [2.0, 2.0, 0.0]),
            rational_surface(
                10,
                [[1, 2], [3, 4], [5, 6]],
                [[1.0, 1.0], [0.8, 0.7], [1.0, 1.0]],
            ),
            face(11, 10),
        ];
        assert_eq!(
            recover_v_extrusion_surfaces(&mut entities).surfaces_recovered,
            0
        );
    }

    #[test]
    fn rejects_non_unit_v_parameterization() {
        let mut surface = rational_surface(
            10,
            [[1, 2], [3, 4], [5, 6]],
            [[1.0, 1.0], [0.8, 0.8], [1.0, 1.0]],
        );
        let EntityInstance::Complex { subsuper, .. } = &mut surface else {
            unreachable!();
        };
        let Parameter::List(knot_params) = &mut subsuper.0[2].parameter else {
            unreachable!();
        };
        knot_params[3] = Parameter::List(vec![Parameter::Real(2.0), Parameter::Real(4.0)]);

        let mut entities = vec![
            point(1, [0.0, 0.0, 0.0]),
            point(2, [0.0, 2.0, 0.0]),
            point(3, [1.0, 0.0, 0.0]),
            point(4, [1.0, 2.0, 0.0]),
            point(5, [2.0, 0.0, 0.0]),
            point(6, [2.0, 2.0, 0.0]),
            surface,
            face(11, 10),
        ];
        assert_eq!(
            recover_v_extrusion_surfaces(&mut entities).surfaces_recovered,
            0
        );
    }

    #[test]
    fn rejects_v_closed_surface() {
        let mut surface = rational_surface(
            10,
            [[1, 2], [3, 4], [5, 6]],
            [[1.0, 1.0], [0.8, 0.8], [1.0, 1.0]],
        );
        let EntityInstance::Complex { subsuper, .. } = &mut surface else {
            unreachable!();
        };
        let Parameter::List(bspline_params) = &mut subsuper.0[1].parameter else {
            unreachable!();
        };
        bspline_params[5] = Parameter::Enumeration("T".to_string());

        let mut entities = vec![
            point(1, [0.0, 0.0, 0.0]),
            point(2, [0.0, 2.0, 0.0]),
            point(3, [1.0, 0.0, 0.0]),
            point(4, [1.0, 2.0, 0.0]),
            point(5, [2.0, 0.0, 0.0]),
            point(6, [2.0, 2.0, 0.0]),
            surface,
            face(11, 10),
        ];
        assert_eq!(
            recover_v_extrusion_surfaces(&mut entities).surfaces_recovered,
            0
        );
    }
}
