use super::*;

#[test]
fn auxiliary_shape_representations_are_retargeted_conservatively() {
    let mut entities = vec![
        EntityInstance::Simple {
            id: 1,
            record: Record {
                name: "SHAPE_REPRESENTATION".to_string(),
                parameter: Parameter::List(vec![
                    Parameter::String(String::new()),
                    Parameter::List(vec![entity_ref(10)]),
                    entity_ref(20),
                ]),
            },
        },
        EntityInstance::Simple {
            id: 2,
            record: Record {
                name: "REPRESENTATION".to_string(),
                parameter: Parameter::List(vec![
                    Parameter::String("volume".to_string()),
                    Parameter::List(vec![entity_ref(10)]),
                    entity_ref(20),
                ]),
            },
        },
    ];
    let replacements = HashMap::from([(10, 30)]);
    retarget_shape_representation_items(&mut entities, &replacements);

    assert_eq!(
        representation_items_and_context(&entities[0]),
        Some((vec![30], 20))
    );
    assert_eq!(
        representation_items_and_context(&entities[1]),
        Some((vec![10], 20))
    );
}

#[test]
fn referenced_of_type_accepts_ap214_context_subtypes() {
    let entities = vec![
        EntityInstance::Simple {
            id: 1,
            record: Record {
                name: "PRODUCT_DEFINITION".to_string(),
                parameter: Parameter::List(vec![entity_ref(2)]),
            },
        },
        EntityInstance::Simple {
            id: 2,
            record: Record {
                name: "DESIGN_CONTEXT".to_string(),
                parameter: Parameter::List(Vec::new()),
            },
        },
        EntityInstance::Simple {
            id: 3,
            record: Record {
                name: "PRODUCT".to_string(),
                parameter: Parameter::List(vec![entity_ref(4)]),
            },
        },
        EntityInstance::Simple {
            id: 4,
            record: Record {
                name: "MECHANICAL_CONTEXT".to_string(),
                parameter: Parameter::List(Vec::new()),
            },
        },
    ];
    let index = build_index(&entities);
    assert_eq!(
        referenced_of_type(
            1,
            &entities,
            &index,
            &["PRODUCT_DEFINITION_CONTEXT", "DESIGN_CONTEXT"]
        ),
        Some(2)
    );
    assert_eq!(
        referenced_of_type(
            3,
            &entities,
            &index,
            &["PRODUCT_CONTEXT", "MECHANICAL_CONTEXT"]
        ),
        Some(4)
    );
}

#[test]
fn referenced_of_type_returns_none_without_eager_indexing() {
    let entities = vec![
        EntityInstance::Simple {
            id: 1,
            record: Record {
                name: "PRODUCT_DEFINITION_SHAPE".to_string(),
                parameter: Parameter::List(vec![
                    Parameter::String(String::new()),
                    Parameter::String(String::new()),
                    entity_ref(2),
                ]),
            },
        },
        EntityInstance::Simple {
            id: 2,
            record: Record {
                name: "DIRECTION".to_string(),
                parameter: Parameter::List(Vec::new()),
            },
        },
    ];
    let index = build_index(&entities);
    assert_eq!(
        referenced_of_type(1, &entities, &index, &["PRODUCT_DEFINITION"]),
        None
    );
}

#[test]
fn z90_signature_ignores_quarter_turn_and_translation() {
    let a = vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 0.0, 3.0],
    ];
    let b: Vec<[f64; 3]> = a
        .iter()
        .map(|p| [10.0 - p[1], -4.0 + p[0], 2.0 + p[2]])
        .collect();
    let (ka, _) = canonical_z90_points(&a, centroid(&a));
    let (kb, _) = canonical_z90_points(&b, centroid(&b));
    assert_eq!(ka, kb);
}

#[test]
fn rigid_translation_maps_source_center_after_rotation() {
    let source = [2.857_319_490_003_104_3, -0.645, 0.588_924_731_498_555_2];
    let target = [-2.857_319_490_003_104_7, 0.625, 0.588_924_731_498_555_1];
    let d = rigid_translation(source, target, 2);
    let (rx, ry) = rotate_xy(source[0], source[1], 2);
    let mapped = [rx + d[0], ry + d[1], source[2] + d[2]];
    for axis in 0..3 {
        assert!((mapped[axis] - target[axis]).abs() < 1.0e-12);
    }
}

#[test]
fn centroid_is_independent_of_input_order() {
    let a = vec![
        [
            7.187_335_566_918_112,
            -10.551_553_818_311_925,
            0.409_865_672_927_023_36,
        ],
        [
            4.527_269_766_918_11,
            -10.553_611_218_311_952,
            0.409_865_672_927_023_36,
        ],
        [1.0e6, -1.0e6, 1.0e-9],
        [-1.0e6, 1.0e6, -1.0e-9],
    ];
    let mut b = a.clone();
    b.reverse();
    assert_eq!(centroid(&a), centroid(&b));
}

#[test]
fn shape_key_ignores_edge_wrapper_serialization() -> anyhow::Result<()> {
    fn key(loop_entities: &str) -> anyhow::Result<ShapeKey> {
        let text = format!(
            "ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('x'),'1');
FILE_NAME('a','b',(''),(''),'x','y','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(1.,0.,0.));
#3=DIRECTION('',(1.,0.,0.));
#4=VECTOR('',#3,1.);
#5=LINE('',#1,#4);
#6=VERTEX_POINT('',#1);
#7=VERTEX_POINT('',#2);
#8=EDGE_CURVE('',#6,#7,#5,.T.);
{loop_entities}
#11=FACE_OUTER_BOUND('',#10,.T.);
#12=DIRECTION('',(0.,0.,1.));
#13=AXIS2_PLACEMENT_3D('',#1,#12,#3);
#14=PLANE('',#13);
#15=ADVANCED_FACE('',(#11),#14,.T.);
#16=CLOSED_SHELL('',(#15));
#17=MANIFOLD_SOLID_BREP('',#16);
ENDSEC;
END-ISO-10303-21;
"
        );
        let exchange = ruststep::parser::parse(&text)?;
        let entities = &exchange.data[0].entities;
        let index = build_index(entities);
        Ok(solid_shape_key(17, entities, &index)
            .ok_or_else(|| anyhow::anyhow!("fixture solid has no shape key"))?
            .0)
    }

    let direct = key("#10=EDGE_LOOP('',(#8));")?;
    let nested = key("#9=ORIENTED_EDGE('',*,*,#8,.F.);
#19=ORIENTED_EDGE('',*,*,#9,.F.);
#10=EDGE_LOOP('',(#19));")?;
    assert_eq!(direct, nested);
    assert_eq!(direct.oriented_edges, 1);
    Ok(())
}

#[test]
fn shape_key_uses_position_tolerance_for_plane_locus() -> anyhow::Result<()> {
    fn key(plane_z: f64) -> anyhow::Result<ShapeKey> {
        let text = format!(
            "ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('x'),'1');
FILE_NAME('a','b',(''),(''),'x','y','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(1.,0.,0.));
#3=DIRECTION('',(1.,0.,0.));
#4=VECTOR('',#3,1.);
#5=LINE('',#1,#4);
#6=VERTEX_POINT('',#1);
#7=VERTEX_POINT('',#2);
#8=EDGE_CURVE('',#6,#7,#5,.T.);
#10=EDGE_LOOP('',(#8));
#11=FACE_OUTER_BOUND('',#10,.T.);
#12=DIRECTION('',(0.,0.,1.));
#18=CARTESIAN_POINT('',(0.,0.,{plane_z:.12}));
#13=AXIS2_PLACEMENT_3D('',#18,#12,#3);
#14=PLANE('',#13);
#15=ADVANCED_FACE('',(#11),#14,.T.);
#16=CLOSED_SHELL('',(#15));
#17=MANIFOLD_SOLID_BREP('',#16);
ENDSEC;
END-ISO-10303-21;
"
        );
        let exchange = ruststep::parser::parse(&text)?;
        let entities = &exchange.data[0].entities;
        let index = build_index(entities);
        Ok(solid_shape_key(17, entities, &index)
            .ok_or_else(|| anyhow::anyhow!("fixture solid has no shape key"))?
            .0)
    }

    let exact = key(0.0)?;
    let sub_tolerance_noise = key(1.0e-7)?;
    let distinct_locus = key(2.0e-5)?;
    assert_eq!(exact, sub_tolerance_noise);
    assert_ne!(exact, distinct_locus);
    Ok(())
}

#[test]
fn shell_face_list_ignores_non_face_members() -> anyhow::Result<()> {
    fn parse(data: &str) -> anyhow::Result<(Vec<EntityInstance>, HashMap<u64, usize>)> {
        let text = format!(
            "ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('x'),'1');
FILE_NAME('a','b',(''),(''),'x','y','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
{data}
ENDSEC;
END-ISO-10303-21;
"
        );
        let exchange = ruststep::parser::parse(&text)?;
        let entities = exchange.data[0].entities.clone();
        let index = build_index(&entities);
        Ok((entities, index))
    }

    let common = "
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(1.,0.,0.));
#3=DIRECTION('',(1.,0.,0.));
#4=VECTOR('',#3,1.);
#5=LINE('',#1,#4);
#6=VERTEX_POINT('',#1);
#7=VERTEX_POINT('',#2);
#8=EDGE_CURVE('',#6,#7,#5,.T.);
#9=ORIENTED_EDGE('',*,*,#8,.T.);
#10=EDGE_LOOP('',(#9));
#11=FACE_OUTER_BOUND('',#10,.T.);
#12=DIRECTION('',(0.,0.,1.));
#13=AXIS2_PLACEMENT_3D('',#1,#12,#3);
#14=PLANE('',#13);
#15=ADVANCED_FACE('',(#11),#14,.T.);
";

    let (entities, index) = parse(&format!(
        "{common}
#16=CLOSED_SHELL('',(#15,#9));
#17=MANIFOLD_SOLID_BREP('',#16);"
    ))?;
    assert_eq!(
        manifold_solid_face_ids(17, &entities, &index),
        Some(vec![15])
    );

    let (entities, index) = parse(&format!(
        "{common}
#16=CLOSED_SHELL('',(#15,#18));
#17=MANIFOLD_SOLID_BREP('',#16);
#18=DIRECTION('',(0.,1.,0.));"
    ))?;
    assert_eq!(
        manifold_solid_face_ids(17, &entities, &index),
        Some(vec![15])
    );
    let raw = closure_from(17, &entities, &index);
    let semantic = semantic_solid_closure(17, &entities, &index)
        .ok_or_else(|| anyhow::anyhow!("fixture solid has no semantic closure"))?;
    assert!(raw.contains(&18));
    assert!(!semantic.contains(&18));
    assert!(semantic.contains(&15));
    Ok(())
}

#[test]
fn z90_signature_rejects_shape_change() {
    let a = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let b = vec![[0.0, 0.0, 0.0], [2.1, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let (ka, _) = canonical_z90_points(&a, centroid(&a));
    let (kb, _) = canonical_z90_points(&b, centroid(&b));
    assert_ne!(ka, kb);
}

#[test]
fn axis_offset_ignores_slide_along_axis() -> anyhow::Result<()> {
    let center = [0.0, 0.0, 0.0];
    let axis = [0.0, 2.0, 0.0];
    let a = canonical_axis_offset([1.25, -7.0, 3.5], axis, center, 0)
        .ok_or_else(|| anyhow::anyhow!("axis offset unavailable"))?;
    let b = canonical_axis_offset([1.25, 42.0, 3.5], axis, center, 0)
        .ok_or_else(|| anyhow::anyhow!("axis offset unavailable"))?;
    assert_eq!(a, b);
    assert_eq!(a, [125_000, 0, 350_000]);
    Ok(())
}

#[test]
fn axis_offset_rotates_with_solid_quarter_turn() -> anyhow::Result<()> {
    let center = [0.0, 0.0, 0.0];
    let axis = [0.0, 0.0, 1.0];
    let a = canonical_axis_offset([2.0, 1.0, 9.0], axis, center, 1)
        .ok_or_else(|| anyhow::anyhow!("axis offset unavailable"))?;
    assert_eq!(a, [-100_000, 200_000, 0]);
    Ok(())
}
#[test]
fn cylindrical_locus_ignores_parameter_seam_direction() -> anyhow::Result<()> {
    fn signature(ref_direction: [f64; 3], radius: f64) -> anyhow::Result<String> {
        let entities = vec![
            EntityInstance::Simple {
                id: 1,
                record: Record {
                    name: "CARTESIAN_POINT".to_string(),
                    parameter: Parameter::List(vec![
                        Parameter::String(String::new()),
                        Parameter::List(vec![
                            Parameter::Real(1.0),
                            Parameter::Real(2.0),
                            Parameter::Real(3.0),
                        ]),
                    ]),
                },
            },
            EntityInstance::Simple {
                id: 2,
                record: Record {
                    name: "DIRECTION".to_string(),
                    parameter: Parameter::List(vec![
                        Parameter::String(String::new()),
                        Parameter::List(vec![
                            Parameter::Real(0.0),
                            Parameter::Real(0.0),
                            Parameter::Real(1.0),
                        ]),
                    ]),
                },
            },
            EntityInstance::Simple {
                id: 3,
                record: Record {
                    name: "DIRECTION".to_string(),
                    parameter: Parameter::List(vec![
                        Parameter::String(String::new()),
                        Parameter::List(ref_direction.into_iter().map(Parameter::Real).collect()),
                    ]),
                },
            },
            EntityInstance::Simple {
                id: 4,
                record: Record {
                    name: "AXIS2_PLACEMENT_3D".to_string(),
                    parameter: Parameter::List(vec![
                        Parameter::String(String::new()),
                        entity_ref(1),
                        entity_ref(2),
                        entity_ref(3),
                    ]),
                },
            },
            EntityInstance::Simple {
                id: 5,
                record: Record {
                    name: "CYLINDRICAL_SURFACE".to_string(),
                    parameter: Parameter::List(vec![
                        Parameter::String(String::new()),
                        entity_ref(4),
                        Parameter::Real(radius),
                    ]),
                },
            },
        ];
        let index = build_index(&entities);
        cylindrical_surface_signature(
            simple_record(&entities[index[&5]])
                .ok_or_else(|| anyhow::anyhow!("cylinder is complex"))?,
            &entities,
            &index,
            [0.0; 3],
            0,
        )
        .ok_or_else(|| anyhow::anyhow!("cylinder signature unavailable"))
    }

    assert_eq!(
        signature([1.0, 0.0, 0.0], 2.5)?,
        signature([0.0, 1.0, 0.0], 2.5)?
    );
    assert_ne!(
        signature([1.0, 0.0, 0.0], 2.5)?,
        signature([1.0, 0.0, 0.0], 2.6)?
    );
    Ok(())
}

#[test]
fn circle_locus_ignores_parameter_seam_direction() -> anyhow::Result<()> {
    fn signature(ref_direction: [f64; 3]) -> anyhow::Result<String> {
        let entities = vec![
            EntityInstance::Simple {
                id: 1,
                record: Record {
                    name: "CARTESIAN_POINT".to_string(),
                    parameter: Parameter::List(vec![
                        Parameter::String(String::new()),
                        Parameter::List(vec![
                            Parameter::Real(1.0),
                            Parameter::Real(2.0),
                            Parameter::Real(3.0),
                        ]),
                    ]),
                },
            },
            EntityInstance::Simple {
                id: 2,
                record: Record {
                    name: "DIRECTION".to_string(),
                    parameter: Parameter::List(vec![
                        Parameter::String(String::new()),
                        Parameter::List(vec![
                            Parameter::Real(0.0),
                            Parameter::Real(0.0),
                            Parameter::Real(1.0),
                        ]),
                    ]),
                },
            },
            EntityInstance::Simple {
                id: 3,
                record: Record {
                    name: "DIRECTION".to_string(),
                    parameter: Parameter::List(vec![
                        Parameter::String(String::new()),
                        Parameter::List(ref_direction.into_iter().map(Parameter::Real).collect()),
                    ]),
                },
            },
            EntityInstance::Simple {
                id: 4,
                record: Record {
                    name: "AXIS2_PLACEMENT_3D".to_string(),
                    parameter: Parameter::List(vec![
                        Parameter::String(String::new()),
                        entity_ref(1),
                        entity_ref(2),
                        entity_ref(3),
                    ]),
                },
            },
            EntityInstance::Simple {
                id: 5,
                record: Record {
                    name: "CIRCLE".to_string(),
                    parameter: Parameter::List(vec![
                        Parameter::String(String::new()),
                        entity_ref(4),
                        Parameter::Real(2.5),
                    ]),
                },
            },
        ];
        let index = build_index(&entities);
        circle_support_signature(
            simple_record(&entities[index[&5]])
                .ok_or_else(|| anyhow::anyhow!("circle is complex"))?,
            &entities,
            &index,
            [0.0; 3],
            0,
        )
        .ok_or_else(|| anyhow::anyhow!("circle signature unavailable"))
    }

    assert_eq!(signature([1.0, 0.0, 0.0])?, signature([0.0, 1.0, 0.0])?);
    Ok(())
}
#[test]
fn curve_replica_signature_matches_explicit_translated_curve() -> anyhow::Result<()> {
    let text = "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('x'),'1');\nFILE_NAME('a','b',(''),(''),'x','y','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n#1=CARTESIAN_POINT('',(0.,0.,0.));\n#2=CARTESIAN_POINT('',(1.,0.,0.));\n#3=CARTESIAN_POINT('',(3.,0.,0.));\n#4=CARTESIAN_POINT('',(4.,0.,0.));\n#5=CARTESIAN_POINT('',(3.,0.,0.));\n#10=B_SPLINE_CURVE_WITH_KNOTS('',1,(#1,#2),.UNSPECIFIED.,.F.,.F.,(2,2),(0.,1.),.UNSPECIFIED.);\n#20=B_SPLINE_CURVE_WITH_KNOTS('',1,(#3,#4),.UNSPECIFIED.,.F.,.F.,(2,2),(0.,1.),.UNSPECIFIED.);\n#30=CARTESIAN_TRANSFORMATION_OPERATOR_3D('','','',$,$,#5,$,$);\n#40=CURVE_REPLICA('',#10,#30);\nENDSEC;\nEND-ISO-10303-21;\n";
    let exchange = ruststep::parser::parse(text)?;
    let entities = &exchange.data[0].entities;
    let index = build_index(entities);
    let center = [3.5, 0.0, 0.0];
    let explicit =
        support_entity_signature(20, entities, &index, center, 0, &mut HashSet::new(), 0)
            .ok_or_else(|| anyhow::anyhow!("explicit curve signature unavailable"))?;
    let replica = support_entity_signature(40, entities, &index, center, 0, &mut HashSet::new(), 0)
        .ok_or_else(|| anyhow::anyhow!("replica curve signature unavailable"))?;
    assert_eq!(explicit, replica);
    Ok(())
}
