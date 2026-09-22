use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use std::collections::HashMap;

#[derive(Debug, Default, Clone)]
pub(crate) struct BezierRecoveryStats {
    pub curves_recovered: usize,
}

pub(crate) fn recover_exact_bezier_curves(entities: &mut [EntityInstance]) -> BezierRecoveryStats {
    let mut stats = BezierRecoveryStats::default();
    if entities.is_empty() {
        return stats;
    }

    let types = entity_types(entities);
    let inbound = inbound_map(entities);

    let candidates: Vec<u64> = entities
        .iter()
        .filter_map(|entity| {
            let EntityInstance::Simple { id, record } = entity else {
                return None;
            };
            (record.name == "B_SPLINE_CURVE_WITH_KNOTS"
                && is_single_span_bezier(record)
                && parameter_usage_is_safe(*id, &types, &inbound))
            .then_some(*id)
        })
        .collect();

    if candidates.is_empty() {
        return stats;
    }

    let wanted: std::collections::HashSet<u64> = candidates.into_iter().collect();
    for entity in entities.iter_mut() {
        let EntityInstance::Simple { id, record } = entity else {
            continue;
        };
        if !wanted.contains(id) {
            continue;
        }
        let Parameter::List(params) = &record.parameter else {
            continue;
        };
        // BEZIER_CURVE inherits exactly the six B_SPLINE_CURVE attributes:
        // name, degree, control points, form, closed, self_intersect.
        let base = params[..6].to_vec();
        record.name = "BEZIER_CURVE".to_string();
        record.parameter = Parameter::List(base);
        stats.curves_recovered += 1;
    }

    stats
}

fn is_single_span_bezier(record: &Record) -> bool {
    let Parameter::List(params) = &record.parameter else {
        return false;
    };
    if params.len() != 9 {
        return false;
    }

    let Some(degree) = integer_value(&params[1]).and_then(|x| usize::try_from(x).ok()) else {
        return false;
    };
    if degree < 1 {
        return false;
    }
    let Some(poles) = entity_ref_list(&params[2]) else {
        return false;
    };
    if poles.len() != degree + 1 {
        return false;
    }

    let Some(mult) = integer_list(&params[6]) else {
        return false;
    };
    if mult != [(degree + 1) as i64, (degree + 1) as i64] {
        return false;
    }

    let Some(knots) = numeric_list(&params[7]) else {
        return false;
    };
    if knots.len() != 2 || !knots.iter().all(|x| x.is_finite()) || knots[0] == knots[1] {
        return false;
    }

    true
}

fn parameter_usage_is_safe(
    curve: u64,
    types: &HashMap<u64, String>,
    inbound: &HashMap<u64, Vec<u64>>,
) -> bool {
    let Some(parents) = inbound.get(&curve) else {
        return false;
    };
    if parents.is_empty() {
        return false;
    }

    parents
        .iter()
        .all(|parent| match types.get(parent).map(String::as_str) {
            // EDGE_CURVE trims by topological endpoint vertices, not by stored curve
            // parameter values, so changing [u0,u1] to Bezier's implicit [0,1] is safe.
            Some("EDGE_CURVE") => true,

            // Reparameterizing the basis curve also reparameterizes U on this swept
            // surface. That is safe only when no consumer refers to surface
            // parameters (PCURVE, RECTANGULAR_TRIMMED_SURFACE, etc.).
            Some("SURFACE_OF_LINEAR_EXTRUSION") => {
                surface_usage_is_topological_only(*parent, types, inbound)
            }

            _ => false,
        })
}

fn surface_usage_is_topological_only(
    surface: u64,
    types: &HashMap<u64, String>,
    inbound: &HashMap<u64, Vec<u64>>,
) -> bool {
    let Some(parents) = inbound.get(&surface) else {
        return false;
    };
    !parents.is_empty()
        && parents
            .iter()
            .all(|parent| types.get(parent).map(String::as_str) == Some("ADVANCED_FACE"))
}

fn entity_types(entities: &[EntityInstance]) -> HashMap<u64, String> {
    entities
        .iter()
        .map(|entity| match entity {
            EntityInstance::Simple { id, record } => (*id, record.name.clone()),
            EntityInstance::Complex { id, .. } => (*id, "COMPLEX".to_string()),
        })
        .collect()
}

fn inbound_map(entities: &[EntityInstance]) -> HashMap<u64, Vec<u64>> {
    let mut out: HashMap<u64, Vec<u64>> = HashMap::new();
    for entity in entities {
        let parent = entity_id(entity);
        visit_entity_refs(entity, &mut |child| {
            out.entry(child).or_default().push(parent);
        });
    }
    out
}

fn entity_id(entity: &EntityInstance) -> u64 {
    match entity {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id,
    }
}

fn entity_ref_list(param: &Parameter) -> Option<Vec<u64>> {
    let Parameter::List(items) = param else {
        return None;
    };
    items.iter().map(entity_ref_value).collect()
}

fn entity_ref_value(param: &Parameter) -> Option<u64> {
    match param {
        Parameter::Ref(Name::Entity(id)) => Some(*id),
        _ => None,
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn r(id: u64) -> Parameter {
        Parameter::Ref(Name::Entity(id))
    }

    fn simple(id: u64, name: &str, params: Vec<Parameter>) -> EntityInstance {
        EntityInstance::Simple {
            id,
            record: Record {
                name: name.to_string(),
                parameter: Parameter::List(params),
            },
        }
    }

    fn spline(id: u64, mult: Vec<i64>, knots: Vec<f64>) -> EntityInstance {
        simple(
            id,
            "B_SPLINE_CURVE_WITH_KNOTS",
            vec![
                Parameter::String(String::new()),
                Parameter::Integer(3),
                Parameter::List(vec![r(10), r(11), r(12), r(13)]),
                Parameter::Enumeration("UNSPECIFIED".to_string()),
                Parameter::Enumeration("F".to_string()),
                Parameter::Enumeration("F".to_string()),
                Parameter::List(mult.into_iter().map(Parameter::Integer).collect()),
                Parameter::List(knots.into_iter().map(Parameter::Real).collect()),
                Parameter::Enumeration("UNSPECIFIED".to_string()),
            ],
        )
    }

    #[test]
    fn recovers_exact_clamped_single_span() {
        let mut e = vec![
            spline(1, vec![4, 4], vec![3.0, 4.0]),
            simple(
                2,
                "EDGE_CURVE",
                vec![
                    Parameter::String(String::new()),
                    r(20),
                    r(21),
                    r(1),
                    Parameter::Enumeration("T".to_string()),
                ],
            ),
        ];
        let s = recover_exact_bezier_curves(&mut e);
        assert_eq!(s.curves_recovered, 1);
        let EntityInstance::Simple { record, .. } = &e[0] else {
            panic!();
        };
        assert_eq!(record.name, "BEZIER_CURVE");
        let Parameter::List(p) = &record.parameter else {
            panic!();
        };
        assert_eq!(p.len(), 6);
    }

    #[test]
    fn rejects_multispan_curve() {
        let mut e = vec![
            spline(1, vec![4, 1, 4], vec![0.0, 0.5, 1.0]),
            simple(
                2,
                "EDGE_CURVE",
                vec![
                    Parameter::String(String::new()),
                    r(20),
                    r(21),
                    r(1),
                    Parameter::Enumeration("T".to_string()),
                ],
            ),
        ];
        assert_eq!(recover_exact_bezier_curves(&mut e).curves_recovered, 0);
    }

    #[test]
    fn rejects_parameter_trimmed_curve_usage() {
        let mut e = vec![
            spline(1, vec![4, 4], vec![0.0, 1.0]),
            simple(
                2,
                "TRIMMED_CURVE",
                vec![Parameter::String(String::new()), r(1)],
            ),
        ];
        assert_eq!(recover_exact_bezier_curves(&mut e).curves_recovered, 0);
    }

    #[test]
    fn accepts_topological_only_extrusion_usage() {
        let mut e = vec![
            spline(1, vec![4, 4], vec![0.0, 1.0]),
            simple(
                2,
                "SURFACE_OF_LINEAR_EXTRUSION",
                vec![Parameter::String(String::new()), r(1), r(30)],
            ),
            simple(
                3,
                "ADVANCED_FACE",
                vec![
                    Parameter::String(String::new()),
                    Parameter::List(Vec::new()),
                    r(2),
                    Parameter::Enumeration("T".to_string()),
                ],
            ),
        ];
        assert_eq!(recover_exact_bezier_curves(&mut e).curves_recovered, 1);
    }

    #[test]
    fn rejects_parameterized_surface_consumer() {
        let mut e = vec![
            spline(1, vec![4, 4], vec![0.0, 1.0]),
            simple(
                2,
                "SURFACE_OF_LINEAR_EXTRUSION",
                vec![Parameter::String(String::new()), r(1), r(30)],
            ),
            simple(
                3,
                "RECTANGULAR_TRIMMED_SURFACE",
                vec![Parameter::String(String::new()), r(2)],
            ),
        ];
        assert_eq!(recover_exact_bezier_curves(&mut e).curves_recovered, 0);
    }
}
