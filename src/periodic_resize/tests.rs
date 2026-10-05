use super::*;

#[test]
fn face_style_clone_and_remove_updates_presentation_containers() -> anyhow::Result<()> {
    let simple = |id, name: &str, params: Vec<Parameter>| EntityInstance::Simple {
        id,
        record: Record {
            name: name.to_string(),
            parameter: Parameter::List(params),
        },
    };
    let mut entities = vec![
        simple(
            1,
            "ADVANCED_FACE",
            vec![
                Parameter::String(String::new()),
                Parameter::List(Vec::new()),
                entity_ref(99),
                Parameter::Enumeration("T".to_string()),
            ],
        ),
        simple(
            2,
            "ADVANCED_FACE",
            vec![
                Parameter::String(String::new()),
                Parameter::List(Vec::new()),
                entity_ref(99),
                Parameter::Enumeration("T".to_string()),
            ],
        ),
        simple(
            10,
            "STYLED_ITEM",
            vec![
                Parameter::String(String::new()),
                Parameter::List(vec![entity_ref(20)]),
                entity_ref(1),
            ],
        ),
        simple(
            30,
            "PRESENTATION_LAYER_ASSIGNMENT",
            vec![
                Parameter::String(String::new()),
                Parameter::String(String::new()),
                Parameter::List(vec![entity_ref(10)]),
            ],
        ),
        simple(
            31,
            "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION",
            vec![
                Parameter::String(String::new()),
                Parameter::List(vec![entity_ref(10)]),
                Parameter::NotProvided,
            ],
        ),
    ];

    let styles = collect_styles_by_target(&entities);
    let relevant = HashSet::from([1]);
    let index = build_index(&entities);
    let parents = collect_style_container_parents(&entities, &index, &styles, &relevant)?;
    require_face_only_direct_styles(&entities, &index, &relevant, &styles)?;

    {
        let mut graph = GraphEditor::new(&mut entities);
        assert_eq!(
            graph.clone_face_styles(&styles, &parents, &HashMap::from([(1, 2)]))?,
            1
        );
        assert_eq!(
            graph.remove_face_styles(&styles, &parents, &HashSet::from([1]))?,
            1
        );
    }

    let styles_after = collect_styles_by_target(&entities);
    assert!(!styles_after.contains_key(&1));
    assert_eq!(styles_after.get(&2).map(Vec::len), Some(1));
    let cloned_style = styles_after[&2][0].id;

    let index = build_index(&entities);
    assert!(!index.contains_key(&10));
    for parent in [30, 31] {
        let parent_index = *index
            .get(&parent)
            .ok_or_else(|| anyhow!("missing presentation parent #{parent}"))?;
        let record = simple_record(&entities[parent_index])
            .ok_or_else(|| anyhow!("presentation parent #{parent} is complex"))?;
        let params = list_params(record)
            .ok_or_else(|| anyhow!("presentation parent #{parent} params invalid"))?;
        let members = match record.name.as_str() {
            "PRESENTATION_LAYER_ASSIGNMENT" => &params[2],
            "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION" => &params[1],
            other => bail!("unexpected presentation parent type {other}"),
        };
        let Parameter::List(members) = members else {
            bail!("presentation member aggregate is not a list");
        };
        assert_eq!(members, &[entity_ref(cloned_style)]);
    }
    Ok(())
}

#[test]
fn clone_descendants_preserves_replica_relative_translation() -> anyhow::Result<()> {
    let simple = |id, name: &str, params: Vec<Parameter>| EntityInstance::Simple {
        id,
        record: Record {
            name: name.to_string(),
            parameter: Parameter::List(params),
        },
    };
    let point = |id, xyz: [f64; 3]| {
        simple(
            id,
            "CARTESIAN_POINT",
            vec![
                Parameter::String(String::new()),
                Parameter::List(xyz.into_iter().map(Parameter::Real).collect()),
            ],
        )
    };
    let mut entities = vec![
        point(1, [0.0, 0.0, 0.0]),
        point(2, [2.54, 0.0, 0.0]),
        simple(3, "CURVE_SOURCE", vec![entity_ref(1)]),
        simple(
            4,
            "CARTESIAN_TRANSFORMATION_OPERATOR_3D",
            vec![
                Parameter::String(String::new()),
                Parameter::String(String::new()),
                Parameter::String(String::new()),
                Parameter::NotProvided,
                Parameter::NotProvided,
                entity_ref(2),
                Parameter::NotProvided,
                Parameter::NotProvided,
            ],
        ),
        simple(
            5,
            "CURVE_REPLICA",
            vec![
                Parameter::String(String::new()),
                entity_ref(3),
                entity_ref(4),
            ],
        ),
    ];

    let mut graph = GraphEditor::new(&mut entities);
    let mapping = graph.clone_descendants(&HashSet::from([5]), [10.0, 0.0, 0.0])?;
    assert_eq!(graph.cartesian_point(mapping[&1])?, [10.0, 0.0, 0.0]);
    assert_eq!(graph.cartesian_point(mapping[&2])?, [2.54, 0.0, 0.0]);
    Ok(())
}

#[test]
fn quantized_coordinate_is_stable_under_small_noise() {
    assert_eq!(
        quantize_coord([1.0, 2.0, 3.0]),
        quantize_coord([1.0 + 1.0e-9, 2.0 - 1.0e-9, 3.0])
    );
}

#[test]
fn rational_single_span_seam_key_ignores_parameter_interval_noise() -> anyhow::Result<()> {
    let simple = |id, name: &str, params: Vec<Parameter>| EntityInstance::Simple {
        id,
        record: Record {
            name: name.to_string(),
            parameter: Parameter::List(params),
        },
    };
    let point = |id, xyz: [f64; 3]| {
        simple(
            id,
            "CARTESIAN_POINT",
            vec![
                Parameter::String(String::new()),
                Parameter::List(xyz.into_iter().map(Parameter::Real).collect()),
            ],
        )
    };
    let curve = |id: u64, knot0: f64, knot1: f64, inner_weight: f64| EntityInstance::Complex {
        id,
        subsuper: ruststep::ast::SubSuperRecord(vec![
            Record {
                name: "BOUNDED_CURVE".to_string(),
                parameter: Parameter::List(Vec::new()),
            },
            Record {
                name: "B_SPLINE_CURVE".to_string(),
                parameter: Parameter::List(vec![
                    Parameter::Integer(3),
                    Parameter::List(vec![
                        entity_ref(1),
                        entity_ref(2),
                        entity_ref(3),
                        entity_ref(4),
                    ]),
                    Parameter::Enumeration("UNSPECIFIED".to_string()),
                    Parameter::Enumeration("F".to_string()),
                    Parameter::Enumeration("F".to_string()),
                ]),
            },
            Record {
                name: "B_SPLINE_CURVE_WITH_KNOTS".to_string(),
                parameter: Parameter::List(vec![
                    Parameter::List(vec![Parameter::Integer(4), Parameter::Integer(4)]),
                    Parameter::List(vec![Parameter::Real(knot0), Parameter::Real(knot1)]),
                    Parameter::Enumeration("UNSPECIFIED".to_string()),
                ]),
            },
            Record {
                name: "CURVE".to_string(),
                parameter: Parameter::List(Vec::new()),
            },
            Record {
                name: "GEOMETRIC_REPRESENTATION_ITEM".to_string(),
                parameter: Parameter::List(Vec::new()),
            },
            Record {
                name: "RATIONAL_B_SPLINE_CURVE".to_string(),
                parameter: Parameter::List(vec![Parameter::List(vec![
                    Parameter::Real(1.0),
                    Parameter::Real(inner_weight),
                    Parameter::Real(inner_weight),
                    Parameter::Real(1.0),
                ])]),
            },
            Record {
                name: "REPRESENTATION_ITEM".to_string(),
                parameter: Parameter::List(vec![Parameter::String(String::new())]),
            },
        ]),
    };
    let edge = |id, curve| {
        simple(
            id,
            "EDGE_CURVE",
            vec![
                Parameter::String(String::new()),
                entity_ref(1),
                entity_ref(4),
                entity_ref(curve),
                Parameter::Enumeration("T".to_string()),
            ],
        )
    };

    let mut entities = vec![
        point(1, [0.0, 0.0, 0.0]),
        point(2, [0.0, 0.2, 0.0]),
        point(3, [0.0, 0.4, 0.1]),
        point(4, [0.0, 0.5, 0.2]),
        curve(10, 6.28318530714962, 7.8539816339151, 0.804737854131298),
        curve(11, 6.28318530715069, 7.8539816339173, 0.8047378541310309),
        curve(12, 6.28318530714962, 7.8539816339151, 0.8047378543),
        edge(20, 10),
        edge(21, 11),
        edge(22, 12),
    ];

    let graph = GraphEditor::new(&mut entities);
    assert_eq!(chain_curve_key(&graph, 20)?, chain_curve_key(&graph, 21)?);
    assert_ne!(chain_curve_key(&graph, 20)?, chain_curve_key(&graph, 22)?);
    Ok(())
}

#[test]
fn prunes_only_detached_vertex_closure() {
    let simple = |id, name: &str, params: Vec<Parameter>| EntityInstance::Simple {
        id,
        record: Record {
            name: name.to_string(),
            parameter: Parameter::List(params),
        },
    };
    let mut entities = vec![
        simple(
            1,
            "CARTESIAN_POINT",
            vec![
                Parameter::String(String::new()),
                Parameter::List(vec![
                    Parameter::Real(0.0),
                    Parameter::Real(0.0),
                    Parameter::Real(0.0),
                ]),
            ],
        ),
        simple(
            2,
            "VERTEX_POINT",
            vec![Parameter::String(String::new()), entity_ref(1)],
        ),
        simple(
            3,
            "CARTESIAN_POINT",
            vec![
                Parameter::String(String::new()),
                Parameter::List(vec![
                    Parameter::Real(1.0),
                    Parameter::Real(0.0),
                    Parameter::Real(0.0),
                ]),
            ],
        ),
        simple(
            4,
            "VERTEX_POINT",
            vec![Parameter::String(String::new()), entity_ref(3)],
        ),
    ];
    // A surviving non-topological parent keeps vertex #4 and its point.
    entities.push(simple(5, "KEEP", vec![entity_ref(4)]));

    assert_eq!(prune_detached_vertex_points(&mut entities), 2);
    let ids = entities.iter().map(entity_id).collect::<HashSet<_>>();
    assert_eq!(ids, HashSet::from([3, 4, 5]));
}
