use super::*;

#[test]
fn reads_representation_length_uncertainty_with_hard_cap() -> anyhow::Result<()> {
    fn tolerance_from(text: &str) -> anyhow::Result<Option<f64>> {
        let exchange = ruststep::parser::parse(text)?;
        let entities = &exchange
            .data
            .first()
            .ok_or_else(|| anyhow::anyhow!("parsed exchange has no DATA section"))?
            .entities;
        let index = build_index(entities);
        let solid_id = entities
            .iter()
            .find(|entity| {
                simple_record(entity).is_some_and(|record| record.name == "MANIFOLD_SOLID_BREP")
            })
            .map(entity_id)
            .ok_or_else(|| anyhow::anyhow!("fixture has no MANIFOLD_SOLID_BREP"))?;
        Ok(source_tolerance_by_representation_item(entities, &index)
            .get(&solid_id)
            .copied())
    }

    let source = std::str::from_utf8(include_bytes!(
        "../../validation/fixtures/native_conical_frustum.step"
    ))?;
    assert_eq!(tolerance_from(source)?, Some(GEOM_TOL_MM));

    let widened = source.replacen("LENGTH_MEASURE(1.E-07)", "LENGTH_MEASURE(5.E-06)", 1);
    assert!(
        (tolerance_from(&widened)?
            .ok_or_else(|| anyhow::anyhow!("widened uncertainty missing"))?
            - 5.0e-6)
            .abs()
            <= 1.0e-15
    );

    let excessive = source.replacen("LENGTH_MEASURE(1.E-07)", "LENGTH_MEASURE(2.E-05)", 1);
    assert_eq!(tolerance_from(&excessive)?, None);

    let centimetres = source.replacen(
        "LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.)",
        "LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.CENTI.,.METRE.)",
        1,
    );
    assert!(
        (tolerance_from(&centimetres)?
            .ok_or_else(|| anyhow::anyhow!("centimetre uncertainty missing"))?
            - 1.0e-6)
            .abs()
            <= 1.0e-15
    );

    let ambiguous = source.replacen(
        "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#117))",
        "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#117,#117))",
        1,
    );
    assert_eq!(tolerance_from(&ambiguous)?, None);
    Ok(())
}

fn test_circle_edge(
    edge_id: u64,
    start_mm: [f64; 3],
    end_mm: [f64; 3],
    center_mm: [f64; 3],
    normal: [f64; 3],
    radius_mm: f64,
) -> brep::OrientedEdgeUse {
    brep::OrientedEdgeUse {
        oriented_edge_id: edge_id + 10_000,
        edge_id,
        curve_id: edge_id + 20_000,
        curve_same_sense: true,
        parameter_forward: true,
        start_vertex: edge_id + 30_000,
        end_vertex: edge_id + 40_000,
        start_mm,
        end_mm,
        support: CurveSupport::Circle(brep::CircleSupport {
            center_mm,
            normal,
            x_direction: [1.0, 0.0, 0.0],
            radius_mm,
        }),
    }
}

fn test_line_edge(edge_id: u64, start_mm: [f64; 3], end_mm: [f64; 3]) -> brep::OrientedEdgeUse {
    let delta = sub(end_mm, start_mm);
    let length = norm(delta);
    assert!(length > 1.0e-12, "test edge must have nonzero length");
    let direction = mul(delta, 1.0 / length);
    brep::OrientedEdgeUse {
        oriented_edge_id: edge_id + 10_000,
        edge_id,
        curve_id: edge_id + 20_000,
        curve_same_sense: true,
        parameter_forward: true,
        start_vertex: edge_id + 30_000,
        end_vertex: edge_id + 40_000,
        start_mm,
        end_mm,
        support: CurveSupport::Line(brep::LineSupport {
            origin_mm: start_mm,
            direction,
        }),
    }
}

fn test_face(surface: SurfaceSupport, edges: Vec<brep::OrientedEdgeUse>) -> FaceInfo {
    FaceInfo {
        surface,
        loops: vec![brep::FaceLoop {
            bound_id: 1,
            loop_id: 2,
            outer: true,
            orientation: true,
            edges,
        }],
    }
}

#[test]
fn plane_profile_accepts_bounded_noisy_circle_vertices() -> anyhow::Result<()> {
    let make_plane = |noise: f64| {
        let positive = [1.0, 0.0, noise];
        let negative = [-1.0, 0.0, 0.0];
        test_face(
            SurfaceSupport::Plane(brep::PlaneSupport {
                origin_mm: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                max_residual_mm: 0.0,
            }),
            vec![
                test_circle_edge(
                    800,
                    positive,
                    negative,
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    1.0,
                ),
                test_circle_edge(
                    801,
                    negative,
                    positive,
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    1.0,
                ),
            ],
        )
    };

    let noisy = make_plane(0.5 * REVOLUTION_SOURCE_SUPPORT_TOL_MM);
    let faces = vec![noisy.clone()];
    let edge_faces = edge_face_map(&faces);
    let entities = Vec::new();
    let index = HashMap::new();
    let context = TopologyContext {
        faces: &faces,
        edge_faces: &edge_faces,
        entities: &entities,
        index: &index,
    };
    let (segment, residual) = plane_profile_segment(
        0,
        &noisy,
        match noisy.surface {
            SurfaceSupport::Plane(plane) => plane,
            _ => unreachable!(),
        },
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        &context,
        REVOLUTION_SOURCE_SUPPORT_TOL_MM,
    )
    .ok_or_else(|| anyhow::anyhow!("expected bounded profile segment"))?;
    assert_eq!(segment.a, [0.0, 0.0]);
    assert_eq!(segment.b, [1.0, 0.0]);
    assert!((residual - 0.5 * REVOLUTION_SOURCE_SUPPORT_TOL_MM).abs() <= 1.0e-12);

    let too_noisy = make_plane(2.0 * REVOLUTION_SOURCE_SUPPORT_TOL_MM);
    let faces = vec![too_noisy.clone()];
    let edge_faces = edge_face_map(&faces);
    let context = TopologyContext {
        faces: &faces,
        edge_faces: &edge_faces,
        entities: &entities,
        index: &index,
    };
    assert!(
        plane_profile_segment(
            0,
            &too_noisy,
            match too_noisy.surface {
                SurfaceSupport::Plane(plane) => plane,
                _ => unreachable!(),
            },
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            &context,
            REVOLUTION_SOURCE_SUPPORT_TOL_MM,
        )
        .is_none()
    );
    Ok(())
}

#[test]
fn plane_profile_detects_axis_crossing_inside_radial_edge() -> anyhow::Result<()> {
    let make_face = |line_start: [f64; 3], line_end: [f64; 3]| {
        test_face(
            SurfaceSupport::Plane(brep::PlaneSupport {
                origin_mm: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                max_residual_mm: 0.0,
            }),
            vec![
                test_circle_edge(
                    820,
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    1.0,
                ),
                test_line_edge(821, line_start, line_end),
            ],
        )
    };

    let crossing = make_face([0.0, -0.25, 0.0], [0.0, 0.25, 0.0]);
    let faces = vec![crossing.clone()];
    let edge_faces = edge_face_map(&faces);
    let entities = Vec::new();
    let index = HashMap::new();
    let context = TopologyContext {
        faces: &faces,
        edge_faces: &edge_faces,
        entities: &entities,
        index: &index,
    };
    let (segment, _) = plane_profile_segment(
        0,
        &crossing,
        match crossing.surface {
            SurfaceSupport::Plane(plane) => plane,
            _ => unreachable!(),
        },
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        &context,
        REVOLUTION_SOURCE_SUPPORT_TOL_MM,
    )
    .ok_or_else(|| anyhow::anyhow!("axis-crossing radial edge should prove a disk"))?;
    assert_eq!(segment.a, [0.0, 0.0]);
    assert_eq!(segment.b, [1.0, 0.0]);

    let one_sided = make_face([0.0, 0.25, 0.0], [0.0, 0.5, 0.0]);
    let faces = vec![one_sided.clone()];
    let edge_faces = edge_face_map(&faces);
    let context = TopologyContext {
        faces: &faces,
        edge_faces: &edge_faces,
        entities: &entities,
        index: &index,
    };
    assert!(
        plane_profile_segment(
            0,
            &one_sided,
            match one_sided.surface {
                SurfaceSupport::Plane(plane) => plane,
                _ => unreachable!(),
            },
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            &context,
            REVOLUTION_SOURCE_SUPPORT_TOL_MM,
        )
        .is_none()
    );
    Ok(())
}

#[test]
fn cylinder_profile_accepts_bounded_noisy_trim_vertices() -> anyhow::Result<()> {
    let make_cylinder = |noise: f64| {
        let bottom_pos = [1.0 + noise, 0.0, 0.0];
        let bottom_neg = [-1.0, 0.0, 0.0];
        let top_pos = [1.0 + noise, 0.0, 1.0];
        let top_neg = [-1.0, 0.0, 1.0];
        test_face(
            SurfaceSupport::Cylinder(brep::CylinderSupport {
                axis_origin_mm: [0.0, 0.0, 0.0],
                axis: [0.0, 0.0, 1.0],
                x_direction: [1.0, 0.0, 0.0],
                radius_mm: 1.0,
            }),
            vec![
                test_circle_edge(
                    900,
                    bottom_pos,
                    bottom_neg,
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    1.0,
                ),
                test_line_edge(901, bottom_neg, top_neg),
                test_circle_edge(902, top_neg, top_pos, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 1.0),
                test_line_edge(903, top_pos, bottom_pos),
            ],
        )
    };

    let empty_edge_faces = HashMap::new();
    let empty_entities = Vec::new();
    let empty_index = HashMap::new();

    let noisy = make_cylinder(0.5 * REVOLUTION_SOURCE_SUPPORT_TOL_MM);
    let faces = vec![noisy.clone()];
    let context = TopologyContext {
        faces: &faces,
        edge_faces: &empty_edge_faces,
        entities: &empty_entities,
        index: &empty_index,
    };
    let (segment, residual) = cylinder_profile_segment(
        0,
        &noisy,
        match noisy.surface {
            SurfaceSupport::Cylinder(cylinder) => cylinder,
            _ => unreachable!(),
        },
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        &context,
        REVOLUTION_SOURCE_SUPPORT_TOL_MM,
    )
    .ok_or_else(|| anyhow::anyhow!("expected bounded profile segment"))?;
    assert_eq!(segment.a, [1.0, 0.0]);
    assert_eq!(segment.b, [1.0, 1.0]);
    assert!((residual - 0.5 * REVOLUTION_SOURCE_SUPPORT_TOL_MM).abs() <= 1.0e-12);

    let too_noisy = make_cylinder(2.0 * REVOLUTION_SOURCE_SUPPORT_TOL_MM);
    let faces = vec![too_noisy.clone()];
    let context = TopologyContext {
        faces: &faces,
        edge_faces: &empty_edge_faces,
        entities: &empty_entities,
        index: &empty_index,
    };
    assert!(
        cylinder_profile_segment(
            0,
            &too_noisy,
            match too_noisy.surface {
                SurfaceSupport::Cylinder(cylinder) => cylinder,
                _ => unreachable!(),
            },
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            &context,
            REVOLUTION_SOURCE_SUPPORT_TOL_MM,
        )
        .is_none()
    );
    Ok(())
}

#[test]
fn cone_profile_accepts_proven_apex_and_bounded_source_noise() -> anyhow::Result<()> {
    let apex = [0.0, 0.0, 0.0];
    let base_pos = [1.0, 0.0, 1.0];
    let base_neg = [-1.0, 0.0, 1.0];
    let mut apex_line_a = test_line_edge(1000, apex, base_pos);
    let mut apex_line_b = test_line_edge(1002, base_neg, apex);
    apex_line_a.start_vertex = 42_424;
    apex_line_b.end_vertex = 42_424;
    let apex_face = test_face(
        SurfaceSupport::Cone(brep::ConeSupport {
            reference_origin_mm: [0.0, 0.0, 1.0],
            axis: [0.0, 0.0, 1.0],
            x_direction: [1.0, 0.0, 0.0],
            reference_radius_mm: 1.0,
            semi_angle_rad: std::f64::consts::FRAC_PI_4,
        }),
        vec![
            apex_line_a,
            test_circle_edge(
                1001,
                base_pos,
                base_neg,
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                1.0,
            ),
            apex_line_b,
        ],
    );
    let empty_edge_faces = HashMap::new();
    let empty_entities = Vec::new();
    let empty_index = HashMap::new();
    let faces = vec![apex_face.clone()];
    let context = TopologyContext {
        faces: &faces,
        edge_faces: &empty_edge_faces,
        entities: &empty_entities,
        index: &empty_index,
    };
    let (segment, residual) = cone_profile_segment(
        0,
        &apex_face,
        match apex_face.surface {
            SurfaceSupport::Cone(cone) => cone,
            _ => unreachable!(),
        },
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        &context,
        REVOLUTION_SOURCE_SUPPORT_TOL_MM,
    )
    .ok_or_else(|| anyhow::anyhow!("expected bounded profile segment"))?;
    assert_eq!(segment.a, [0.0, 0.0]);
    assert_eq!(segment.b, [1.0, 1.0]);
    assert!(residual <= 1.0e-12);

    let make_frustum = |reference_radius_mm: f64| {
        let bottom_pos = [1.0, 0.0, 0.0];
        let bottom_neg = [-1.0, 0.0, 0.0];
        let top_pos = [2.0, 0.0, 1.0];
        let top_neg = [-2.0, 0.0, 1.0];
        test_face(
            SurfaceSupport::Cone(brep::ConeSupport {
                reference_origin_mm: [0.0, 0.0, 0.0],
                axis: [0.0, 0.0, 1.0],
                x_direction: [1.0, 0.0, 0.0],
                reference_radius_mm,
                semi_angle_rad: std::f64::consts::FRAC_PI_4,
            }),
            vec![
                test_circle_edge(
                    1010,
                    bottom_pos,
                    bottom_neg,
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    1.0,
                ),
                test_line_edge(1011, bottom_neg, top_neg),
                test_circle_edge(
                    1012,
                    top_neg,
                    top_pos,
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    2.0,
                ),
                test_line_edge(1013, top_pos, bottom_pos),
            ],
        )
    };

    let noisy = make_frustum(1.0 + 0.5 * REVOLUTION_SOURCE_SUPPORT_TOL_MM);
    let faces = vec![noisy.clone()];
    let context = TopologyContext {
        faces: &faces,
        edge_faces: &empty_edge_faces,
        entities: &empty_entities,
        index: &empty_index,
    };
    let (segment, residual) = cone_profile_segment(
        0,
        &noisy,
        match noisy.surface {
            SurfaceSupport::Cone(cone) => cone,
            _ => unreachable!(),
        },
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        &context,
        REVOLUTION_SOURCE_SUPPORT_TOL_MM,
    )
    .ok_or_else(|| anyhow::anyhow!("expected bounded profile segment"))?;
    assert_eq!(segment.a, [1.0, 0.0]);
    assert_eq!(segment.b, [2.0, 1.0]);
    assert!((residual - 0.5 * REVOLUTION_SOURCE_SUPPORT_TOL_MM).abs() <= 1.0e-12);

    let too_noisy = make_frustum(1.0 + 2.0 * REVOLUTION_SOURCE_SUPPORT_TOL_MM);
    let faces = vec![too_noisy.clone()];
    let context = TopologyContext {
        faces: &faces,
        edge_faces: &empty_edge_faces,
        entities: &empty_entities,
        index: &empty_index,
    };
    assert!(
        cone_profile_segment(
            0,
            &too_noisy,
            match too_noisy.surface {
                SurfaceSupport::Cone(cone) => cone,
                _ => unreachable!(),
            },
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            &context,
            REVOLUTION_SOURCE_SUPPORT_TOL_MM,
        )
        .is_none()
    );

    let declared_tolerance_mm = 1.0e-6;
    let declared_noisy = make_frustum(1.0 + 0.5 * declared_tolerance_mm);
    let faces = vec![declared_noisy.clone()];
    let context = TopologyContext {
        faces: &faces,
        edge_faces: &empty_edge_faces,
        entities: &empty_entities,
        index: &empty_index,
    };
    assert!(
        cone_profile_segment(
            0,
            &declared_noisy,
            match declared_noisy.surface {
                SurfaceSupport::Cone(cone) => cone,
                _ => unreachable!(),
            },
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            &context,
            REVOLUTION_SOURCE_SUPPORT_TOL_MM,
        )
        .is_none()
    );
    assert!(
        cone_profile_segment(
            0,
            &declared_noisy,
            match declared_noisy.surface {
                SurfaceSupport::Cone(cone) => cone,
                _ => unreachable!(),
            },
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            &context,
            declared_tolerance_mm,
        )
        .is_some()
    );
    Ok(())
}

#[test]
fn cone_profile_uses_neighbor_constraint_plus_repeated_trim_circle() -> anyhow::Result<()> {
    let bottom_pos = [1.5, 0.0, 0.5000005];
    let bottom_neg = [-1.5, 0.0, 0.5000005];
    let top_pos = [2.0000004, 0.0, 1.0000004];
    let top_neg = [-2.0000004, 0.0, 1.0000004];

    let bottom = test_circle_edge(
        1100,
        bottom_pos,
        bottom_neg,
        [0.0, 0.0, 0.5000005],
        [0.0, 0.0, 1.0],
        1.5,
    );
    let top = test_circle_edge(
        1102,
        top_neg,
        top_pos,
        [0.0, 0.0, 1.0000004],
        [0.0, 0.0, 1.0],
        2.0000004,
    );
    let cone_face = test_face(
        SurfaceSupport::Cone(brep::ConeSupport {
            reference_origin_mm: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 1.0],
            x_direction: [1.0, 0.0, 0.0],
            reference_radius_mm: 1.0,
            semi_angle_rad: std::f64::consts::FRAC_PI_4,
        }),
        vec![
            bottom.clone(),
            test_line_edge(1101, bottom_neg, top_neg),
            top.clone(),
            test_line_edge(1103, top_pos, bottom_pos),
        ],
    );
    let cylinder_face = test_face(
        SurfaceSupport::Cylinder(brep::CylinderSupport {
            axis_origin_mm: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 1.0],
            x_direction: [1.0, 0.0, 0.0],
            radius_mm: 1.5,
        }),
        vec![bottom],
    );
    let plane_face = test_face(
        SurfaceSupport::Plane(brep::PlaneSupport {
            origin_mm: [0.0, 0.0, 1.0],
            normal: [0.0, 0.0, 1.0],
            max_residual_mm: 0.0,
        }),
        vec![top],
    );
    let faces = vec![cone_face.clone(), cylinder_face, plane_face];
    let edge_faces = edge_face_map(&faces);
    let entities = Vec::new();
    let index = HashMap::new();
    let context = TopologyContext {
        faces: &faces,
        edge_faces: &edge_faces,
        entities: &entities,
        index: &index,
    };

    assert!(
        cone_profile_segment(
            0,
            &cone_face,
            match cone_face.surface {
                SurfaceSupport::Cone(cone) => cone,
                _ => unreachable!(),
            },
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            &context,
            REVOLUTION_SOURCE_SUPPORT_TOL_MM,
        )
        .is_none()
    );

    let (segment, residual) = cone_profile_segment(
        0,
        &cone_face,
        match cone_face.surface {
            SurfaceSupport::Cone(cone) => cone,
            _ => unreachable!(),
        },
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        &context,
        1.0e-5,
    )
    .ok_or_else(|| anyhow::anyhow!("expected recovered test geometry"))?;

    // The cylinder constrains radius while preserving the repeated trim-circle
    // axial coordinate; the plane constrains axial position while preserving
    // the repeated trim-circle radius. The cone support is evidence, not the
    // sole source of truth when its own parameters disagree within STEP uncertainty.
    assert_eq!(segment.a, [1.5, 0.5000005]);
    assert_eq!(segment.b, [2.0000004, 1.0]);
    assert!(residual > REVOLUTION_SOURCE_SUPPORT_TOL_MM);
    assert!(residual < 1.0e-5);
    Ok(())
}

fn split_hemisphere_faces(cylinder_radius_mm: f64) -> Vec<FaceInfo> {
    let sphere = brep::SphereSupport {
        center_mm: [0.0, 0.0, 0.0],
        axis: [0.0, 0.0, 1.0],
        x_direction: [1.0, 0.0, 0.0],
        radius_mm: 1.0,
    };
    let cylinder = brep::CylinderSupport {
        axis_origin_mm: [0.0, 0.0, 0.0],
        axis: [0.0, 0.0, 1.0],
        x_direction: [1.0, 0.0, 0.0],
        radius_mm: cylinder_radius_mm,
    };
    let plane = brep::PlaneSupport {
        origin_mm: [0.0, 0.0, 2.0],
        normal: [0.0, 0.0, 1.0],
        max_residual_mm: 0.0,
    };

    let equator_a = test_circle_edge(
        10,
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        1.0,
    );
    let equator_b = test_circle_edge(
        11,
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        1.0,
    );
    let sphere_seam_a = test_circle_edge(
        12,
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        1.0,
    );
    let sphere_seam_b = test_circle_edge(
        13,
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        1.0,
    );
    let disk_a = test_circle_edge(
        14,
        [1.0, 0.0, 2.0],
        [-1.0, 0.0, 2.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 1.0],
        1.0,
    );
    let disk_b = test_circle_edge(
        15,
        [-1.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 1.0],
        1.0,
    );
    let cylinder_seam_a = test_line_edge(16, [1.0, 0.0, 0.0], [1.0, 0.0, 2.0]);
    let cylinder_seam_b = test_line_edge(17, [-1.0, 0.0, 0.0], [-1.0, 0.0, 2.0]);

    vec![
        test_face(
            SurfaceSupport::Sphere(sphere),
            vec![
                equator_a.clone(),
                sphere_seam_a.clone(),
                sphere_seam_b.clone(),
            ],
        ),
        test_face(
            SurfaceSupport::Sphere(sphere),
            vec![equator_b.clone(), sphere_seam_a, sphere_seam_b],
        ),
        test_face(
            SurfaceSupport::Cylinder(cylinder),
            vec![
                equator_a,
                disk_a.clone(),
                cylinder_seam_a.clone(),
                cylinder_seam_b.clone(),
            ],
        ),
        test_face(
            SurfaceSupport::Cylinder(cylinder),
            vec![equator_b, disk_b.clone(), cylinder_seam_a, cylinder_seam_b],
        ),
        test_face(SurfaceSupport::Plane(plane), vec![disk_a, disk_b]),
    ]
}

fn edge_face_map(faces: &[FaceInfo]) -> HashMap<u64, Vec<usize>> {
    let mut edge_faces = HashMap::<u64, Vec<usize>>::new();
    for (face_index, face) in faces.iter().enumerate() {
        for edge in face.loops.iter().flat_map(|loop_| &loop_.edges) {
            edge_faces.entry(edge.edge_id).or_default().push(face_index);
        }
    }
    edge_faces
}

#[test]
fn recovers_split_hemispherical_end_topology() -> anyhow::Result<()> {
    let faces = split_hemisphere_faces(1.0);
    let edge_faces = edge_face_map(&faces);
    let entities = Vec::new();
    let index = HashMap::new();
    let context = TopologyContext {
        faces: &faces,
        edge_faces: &edge_faces,
        entities: &entities,
        index: &index,
    };
    let recovered = detect_hemispherical_end(99, &[1, 2, 3, 4, 5], &faces, &context)
        .ok_or_else(|| anyhow::anyhow!("expected hemispherical recovery"))?;

    assert_eq!(recovered.profile_curves.len(), 4);
    assert!(recovered.max_residual_mm <= 1.0e-12);
    assert_eq!(recovered.axis_direction, [0.0, 0.0, 1.0]);

    assert!(matches!(
        recovered.profile_curves.as_slice(),
        [
            RecoveredProfileCurve::Line {
                start_mm: [0.0, 2.0],
                end_mm: [1.0, 2.0],
                ..
            },
            RecoveredProfileCurve::Line {
                start_mm: [1.0, 2.0],
                end_mm: [1.0, 0.0],
                ..
            },
            RecoveredProfileCurve::CircleArc {
                center_mm: [0.0, 0.0],
                radius_mm: 1.0,
                start_angle_rad: 0.0,
                end_angle_rad,
                ..
            },
            RecoveredProfileCurve::Line {
                start_mm: [0.0, -1.0],
                end_mm: [0.0, 2.0],
                ..
            }
        ] if (*end_angle_rad + std::f64::consts::FRAC_PI_2).abs() <= 1.0e-12
    ));

    #[cfg(feature = "cad-kernel-monstertruck")]
    {
        use crate::cad_kernel::CadKernel;
        let fragment = crate::cad_recovery::recover_solid_revolution_fragment(&recovered)?;
        let kernel = crate::cad_kernel::monstertruck::MonstertruckKernel;
        let rebuilt = kernel.evaluate(&fragment.model, fragment.root)?;
        assert!(kernel.summarize(&rebuilt).geometrically_consistent);
        ruststep::parser::parse(&kernel.to_step(&rebuilt)?)?;
    }

    let tampered = split_hemisphere_faces(1.01);
    let edge_faces = edge_face_map(&tampered);
    let context = TopologyContext {
        faces: &tampered,
        edge_faces: &edge_faces,
        entities: &entities,
        index: &index,
    };
    assert!(detect_hemispherical_end(99, &[1, 2, 3, 4, 5], &tampered, &context).is_none());
    Ok(())
}

#[test]
fn mixed_curved_graph_recovers_split_sphere_arc() -> anyhow::Result<()> {
    let faces = split_hemisphere_faces(1.0);
    let edge_faces = edge_face_map(&faces);
    let entities = Vec::new();
    let index = HashMap::new();
    let context = TopologyContext {
        faces: &faces,
        edge_faces: &edge_faces,
        entities: &entities,
        index: &index,
    };
    let recovered = detect_mixed_curved_revolution(99, &[1, 2, 3, 4, 5], &faces, &context)
        .ok_or_else(|| anyhow::anyhow!("expected mixed-curved recovery"))?;
    assert_eq!(recovered.profile_curves.len(), 4);
    assert!(recovered.profile_curves.iter().any(|curve| matches!(
        curve,
        RecoveredProfileCurve::CircleArc {
            center_mm: [0.0, 0.0],
            radius_mm: 1.0,
            start_angle_rad,
            end_angle_rad,
            ..
        } if (start_angle_rad.abs() <= 1.0e-12
            && (*end_angle_rad + std::f64::consts::FRAC_PI_2).abs() <= 1.0e-12)
            || ((*start_angle_rad + std::f64::consts::FRAC_PI_2).abs() <= 1.0e-12
                && end_angle_rad.abs() <= 1.0e-12)
    )));
    assert!(recovered.max_residual_mm <= 1.0e-12);

    let tampered = split_hemisphere_faces(1.01);
    let edge_faces = edge_face_map(&tampered);
    let context = TopologyContext {
        faces: &tampered,
        edge_faces: &edge_faces,
        entities: &entities,
        index: &index,
    };
    assert!(detect_mixed_curved_revolution(99, &[1, 2, 3, 4, 5], &tampered, &context).is_none());
    Ok(())
}

fn quarter_fillet_profile() -> Vec<RecoveredProfileCurve> {
    vec![
        RecoveredProfileCurve::Line {
            source_edge_ids: Vec::new(),
            start_mm: [0.0, 1.0],
            end_mm: [1.0, 1.0],
        },
        RecoveredProfileCurve::Line {
            source_edge_ids: Vec::new(),
            start_mm: [1.0, 1.0],
            end_mm: [1.0, 1.4],
        },
        RecoveredProfileCurve::CircleArc {
            source_edge_ids: Vec::new(),
            center_mm: [0.9, 1.4],
            radius_mm: 0.1,
            start_angle_rad: 0.0,
            end_angle_rad: std::f64::consts::FRAC_PI_2,
        },
        RecoveredProfileCurve::Line {
            source_edge_ids: Vec::new(),
            start_mm: [0.9, 1.5],
            end_mm: [0.0, 1.5],
        },
        RecoveredProfileCurve::Line {
            source_edge_ids: Vec::new(),
            start_mm: [0.0, 1.5],
            end_mm: [0.0, 1.0],
        },
    ]
}

#[test]
fn orders_mixed_line_arc_profile_and_rejects_line_arc_crossing() -> anyhow::Result<()> {
    let profile = quarter_fillet_profile();
    let scrambled = vec![
        profile[2].reversed(),
        profile[4].clone(),
        profile[1].clone(),
        profile[3].reversed(),
        profile[0].clone(),
    ];
    let ordered = closed_profile_from_curves(scrambled)
        .ok_or_else(|| anyhow::anyhow!("expected closed profile"))?;
    assert_eq!(ordered.len(), 5);
    assert!(matches!(
        &ordered[2],
        RecoveredProfileCurve::CircleArc {
            center_mm: [0.9, 1.4],
            radius_mm: 0.1,
            start_angle_rad,
            end_angle_rad,
            ..
        } if start_angle_rad.abs() <= 1.0e-12
            && (*end_angle_rad - std::f64::consts::FRAC_PI_2).abs() <= 1.0e-12
    ));

    let crossing = RecoveredProfileCurve::Line {
        source_edge_ids: Vec::new(),
        start_mm: [0.95, 1.2],
        end_mm: [0.95, 1.6],
    };
    assert!(line_arc_has_extra_intersection(
        &crossing,
        &profile[2],
        false
    ));
    Ok(())
}

#[test]
fn orders_two_arc_profile_and_rejects_arc_crossings() -> anyhow::Result<()> {
    let profile = vec![
        RecoveredProfileCurve::Line {
            source_edge_ids: Vec::new(),
            start_mm: [0.0, 0.0],
            end_mm: [0.9, 0.0],
        },
        RecoveredProfileCurve::CircleArc {
            source_edge_ids: Vec::new(),
            center_mm: [0.9, 0.1],
            radius_mm: 0.1,
            start_angle_rad: -std::f64::consts::FRAC_PI_2,
            end_angle_rad: 0.0,
        },
        RecoveredProfileCurve::Line {
            source_edge_ids: Vec::new(),
            start_mm: [1.0, 0.1],
            end_mm: [1.0, 0.9],
        },
        RecoveredProfileCurve::CircleArc {
            source_edge_ids: Vec::new(),
            center_mm: [0.9, 0.9],
            radius_mm: 0.1,
            start_angle_rad: 0.0,
            end_angle_rad: std::f64::consts::FRAC_PI_2,
        },
        RecoveredProfileCurve::Line {
            source_edge_ids: Vec::new(),
            start_mm: [0.9, 1.0],
            end_mm: [0.0, 1.0],
        },
        RecoveredProfileCurve::Line {
            source_edge_ids: Vec::new(),
            start_mm: [0.0, 1.0],
            end_mm: [0.0, 0.0],
        },
    ];
    let scrambled = vec![
        profile[3].reversed(),
        profile[0].clone(),
        profile[5].clone(),
        profile[2].reversed(),
        profile[4].clone(),
        profile[1].clone(),
    ];
    let ordered = closed_profile_from_curves(scrambled)
        .ok_or_else(|| anyhow::anyhow!("expected closed profile"))?;
    assert_eq!(ordered.len(), 6);
    assert_eq!(
        ordered
            .iter()
            .filter(|curve| matches!(curve, RecoveredProfileCurve::CircleArc { .. }))
            .count(),
        2
    );
    assert!(!mixed_profile_self_intersects(&ordered));

    let first = RecoveredProfileCurve::CircleArc {
        source_edge_ids: Vec::new(),
        center_mm: [0.0, 0.0],
        radius_mm: 1.0,
        start_angle_rad: 0.0,
        end_angle_rad: std::f64::consts::PI,
    };
    let crossing = RecoveredProfileCurve::CircleArc {
        source_edge_ids: Vec::new(),
        center_mm: [1.0, 0.0],
        radius_mm: 1.0,
        start_angle_rad: std::f64::consts::FRAC_PI_2,
        end_angle_rad: 3.0 * std::f64::consts::FRAC_PI_2,
    };
    assert!(arc_pair_has_extra_intersection(&first, &crossing, false));

    let tangent_a = RecoveredProfileCurve::CircleArc {
        source_edge_ids: Vec::new(),
        center_mm: [0.0, 0.0],
        radius_mm: 1.0,
        start_angle_rad: 0.0,
        end_angle_rad: std::f64::consts::FRAC_PI_2,
    };
    let tangent_b = RecoveredProfileCurve::CircleArc {
        source_edge_ids: Vec::new(),
        center_mm: [0.0, 2.0],
        radius_mm: 1.0,
        start_angle_rad: -std::f64::consts::FRAC_PI_2,
        end_angle_rad: 0.0,
    };
    assert!(!arc_pair_has_extra_intersection(
        &tangent_a, &tangent_b, true
    ));

    let coincident_adjacent = RecoveredProfileCurve::CircleArc {
        source_edge_ids: Vec::new(),
        center_mm: [0.0, 0.0],
        radius_mm: 1.0,
        start_angle_rad: std::f64::consts::FRAC_PI_2,
        end_angle_rad: std::f64::consts::PI,
    };
    assert!(!arc_pair_has_extra_intersection(
        &tangent_a,
        &coincident_adjacent,
        true
    ));

    let coincident_overlap = RecoveredProfileCurve::CircleArc {
        source_edge_ids: Vec::new(),
        center_mm: [0.0, 0.0],
        radius_mm: 1.0,
        start_angle_rad: std::f64::consts::FRAC_PI_4,
        end_angle_rad: std::f64::consts::PI,
    };
    assert!(arc_pair_has_extra_intersection(
        &tangent_a,
        &coincident_overlap,
        true
    ));
    Ok(())
}

fn split_quarter_torus_faces(
    torus_major_radius_mm: f64,
    torus_minor_radius_mm: f64,
) -> Vec<FaceInfo> {
    let torus = brep::TorusSupport {
        center_mm: [0.0, 0.0, 1.4],
        axis: [0.0, 0.0, 1.0],
        x_direction: [1.0, 0.0, 0.0],
        major_radius_mm: torus_major_radius_mm,
        minor_radius_mm: torus_minor_radius_mm,
    };
    let cylinder = brep::CylinderSupport {
        axis_origin_mm: [0.0, 0.0, 0.0],
        axis: [0.0, 0.0, 1.0],
        x_direction: [1.0, 0.0, 0.0],
        radius_mm: torus_major_radius_mm + torus_minor_radius_mm,
    };
    let bottom_plane = brep::PlaneSupport {
        origin_mm: [0.0, 0.0, 1.0],
        normal: [0.0, 0.0, 1.0],
        max_residual_mm: 0.0,
    };
    let top_plane = brep::PlaneSupport {
        origin_mm: [0.0, 0.0, 1.4 + torus_minor_radius_mm],
        normal: [0.0, 0.0, 1.0],
        max_residual_mm: 0.0,
    };

    let outer_radius = torus_major_radius_mm + torus_minor_radius_mm;
    let top_z = 1.4 + torus_minor_radius_mm;
    let top_a = test_circle_edge(
        110,
        [torus_major_radius_mm, 0.0, top_z],
        [-torus_major_radius_mm, 0.0, top_z],
        [0.0, 0.0, top_z],
        [0.0, 0.0, 1.0],
        torus_major_radius_mm,
    );
    let top_b = test_circle_edge(
        111,
        [-torus_major_radius_mm, 0.0, top_z],
        [torus_major_radius_mm, 0.0, top_z],
        [0.0, 0.0, top_z],
        [0.0, 0.0, 1.0],
        torus_major_radius_mm,
    );
    let side_a = test_circle_edge(
        112,
        [outer_radius, 0.0, 1.4],
        [-outer_radius, 0.0, 1.4],
        [0.0, 0.0, 1.4],
        [0.0, 0.0, 1.0],
        outer_radius,
    );
    let side_b = test_circle_edge(
        113,
        [-outer_radius, 0.0, 1.4],
        [outer_radius, 0.0, 1.4],
        [0.0, 0.0, 1.4],
        [0.0, 0.0, 1.0],
        outer_radius,
    );
    let bottom_a = test_circle_edge(
        114,
        [outer_radius, 0.0, 1.0],
        [-outer_radius, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        outer_radius,
    );
    let bottom_b = test_circle_edge(
        115,
        [-outer_radius, 0.0, 1.0],
        [outer_radius, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        outer_radius,
    );
    let torus_seam_positive = test_circle_edge(
        116,
        [torus_major_radius_mm, 0.0, top_z],
        [outer_radius, 0.0, 1.4],
        [torus_major_radius_mm, 0.0, 1.4],
        [0.0, 1.0, 0.0],
        torus_minor_radius_mm,
    );
    let torus_seam_negative = test_circle_edge(
        117,
        [-torus_major_radius_mm, 0.0, top_z],
        [-outer_radius, 0.0, 1.4],
        [-torus_major_radius_mm, 0.0, 1.4],
        [0.0, -1.0, 0.0],
        torus_minor_radius_mm,
    );
    let cylinder_seam_positive =
        test_line_edge(118, [outer_radius, 0.0, 1.0], [outer_radius, 0.0, 1.4]);
    let cylinder_seam_negative =
        test_line_edge(119, [-outer_radius, 0.0, 1.0], [-outer_radius, 0.0, 1.4]);

    vec![
        test_face(
            SurfaceSupport::Torus(torus),
            vec![
                top_a.clone(),
                torus_seam_positive.clone(),
                side_a.clone(),
                torus_seam_negative.clone(),
            ],
        ),
        test_face(
            SurfaceSupport::Torus(torus),
            vec![
                top_b.clone(),
                torus_seam_positive,
                side_b.clone(),
                torus_seam_negative,
            ],
        ),
        test_face(
            SurfaceSupport::Cylinder(cylinder),
            vec![
                bottom_a.clone(),
                cylinder_seam_positive.clone(),
                side_a,
                cylinder_seam_negative.clone(),
            ],
        ),
        test_face(
            SurfaceSupport::Cylinder(cylinder),
            vec![
                bottom_b.clone(),
                cylinder_seam_positive,
                side_b,
                cylinder_seam_negative,
            ],
        ),
        test_face(
            SurfaceSupport::Plane(bottom_plane),
            vec![bottom_a, bottom_b],
        ),
        test_face(SurfaceSupport::Plane(top_plane), vec![top_a, top_b]),
    ]
}

#[test]
fn recovers_split_quarter_torus_fillet_topology() -> anyhow::Result<()> {
    let faces = split_quarter_torus_faces(0.9, 0.1);
    let edge_faces = edge_face_map(&faces);
    let entities = Vec::new();
    let index = HashMap::new();
    let context = TopologyContext {
        faces: &faces,
        edge_faces: &edge_faces,
        entities: &entities,
        index: &index,
    };
    let recovered = detect_mixed_curved_revolution(99, &[1, 2, 3, 4, 5, 6], &faces, &context)
        .ok_or_else(|| anyhow::anyhow!("expected torus-fillet recovery"))?;
    assert_eq!(recovered.profile_curves.len(), 5);
    assert!(recovered.profile_curves.iter().any(|curve| matches!(
        curve,
        RecoveredProfileCurve::CircleArc {
            center_mm: [0.9, 1.4],
            radius_mm: 0.1,
            start_angle_rad,
            end_angle_rad,
            ..
        } if start_angle_rad.abs() <= 1.0e-12
            && (*end_angle_rad - std::f64::consts::FRAC_PI_2).abs() <= 1.0e-12
    )));

    let spindle = split_quarter_torus_faces(0.05, 0.1);
    let edge_faces = edge_face_map(&spindle);
    let context = TopologyContext {
        faces: &spindle,
        edge_faces: &edge_faces,
        entities: &entities,
        index: &index,
    };
    let recovered = detect_mixed_curved_revolution(99, &[1, 2, 3, 4, 5, 6], &spindle, &context)
        .ok_or_else(|| anyhow::anyhow!("expected spindle-torus recovery"))?;
    assert!(recovered.profile_curves.iter().any(|curve| matches!(
        curve,
        RecoveredProfileCurve::CircleArc {
            center_mm: [0.05, 1.4],
            radius_mm: 0.1,
            start_angle_rad,
            end_angle_rad,
            ..
        } if start_angle_rad.abs() <= 1.0e-12
            && (*end_angle_rad - std::f64::consts::FRAC_PI_2).abs() <= 1.0e-12
    )));

    assert!(circle_arc_min_radius(0.05, 0.1, 0.0, std::f64::consts::FRAC_PI_2) >= 0.0);
    assert!(circle_arc_min_radius(0.05, 0.1, 0.0, std::f64::consts::PI) < -GEOM_TOL_MM);

    let mut tampered = split_quarter_torus_faces(0.9, 0.1);
    for face in tampered.iter_mut().take(2) {
        let SurfaceSupport::Torus(mut torus) = face.surface else {
            anyhow::bail!("expected torus test face");
        };
        torus.minor_radius_mm = 0.11;
        face.surface = SurfaceSupport::Torus(torus);
    }
    let edge_faces = edge_face_map(&tampered);
    let context = TopologyContext {
        faces: &tampered,
        edge_faces: &edge_faces,
        entities: &entities,
        index: &index,
    };
    assert!(detect_mixed_curved_revolution(99, &[1, 2, 3, 4, 5, 6], &tampered, &context).is_none());
    Ok(())
}

#[test]
fn solid_surface_signature_reports_native_spherical_cap() -> anyhow::Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/native_spherical_cap.step");
    let signatures = crate::detect_solid_surface_signatures_bytes(bytes)?;
    assert_eq!(signatures.len(), 1);
    assert_eq!(signatures[0].face_count, 2);
    assert_eq!(signatures[0].support_counts.get("sphere"), Some(&1));
    assert_eq!(signatures[0].support_counts.get("plane"), Some(&1));
    assert!(signatures[0].closed_two_manifold);
    Ok(())
}

#[test]
fn shell_connectivity_rejects_disconnected_two_manifolds() {
    let connected = HashMap::from([(1, vec![0, 1]), (2, vec![1, 2]), (3, vec![2, 0])]);
    assert!(shell_faces_connected(3, &connected));

    let disconnected = HashMap::from([
        (1, vec![0, 1]),
        (2, vec![0, 1]),
        (3, vec![2, 3]),
        (4, vec![2, 3]),
    ]);
    assert!(!shell_faces_connected(4, &disconnected));
}

#[test]
fn closes_axis_touching_step_profile() -> anyhow::Result<()> {
    let profile = closed_profile_from_segments(vec![
        Segment2 {
            a: [0.0, 0.0],
            b: [2.0, 0.0],
        },
        Segment2 {
            a: [2.0, 0.0],
            b: [2.0, 3.0],
        },
        Segment2 {
            a: [0.0, 3.0],
            b: [2.0, 3.0],
        },
    ])
    .ok_or_else(|| anyhow::anyhow!("expected recovered test geometry"))?;
    assert_eq!(profile.len(), 4);
    assert!(profile.iter().any(|point| *point == [0.0, 0.0]));
    assert!(profile.iter().any(|point| *point == [0.0, 3.0]));
    assert!(signed_area(&profile) > 0.0);
    Ok(())
}

#[test]
fn closes_sloped_frustum_profile() -> anyhow::Result<()> {
    let profile = closed_profile_from_segments(vec![
        Segment2 {
            a: [0.0, -1.0],
            b: [2.0, -1.0],
        },
        Segment2 {
            a: [2.0, -1.0],
            b: [1.0, 1.0],
        },
        Segment2 {
            a: [0.0, 1.0],
            b: [1.0, 1.0],
        },
    ])
    .ok_or_else(|| anyhow::anyhow!("expected recovered test geometry"))?;
    assert_eq!(profile.len(), 4);
    assert!(profile.iter().any(|point| *point == [2.0, -1.0]));
    assert!(profile.iter().any(|point| *point == [1.0, 1.0]));
    assert!(signed_area(&profile) > 0.0);
    Ok(())
}

#[test]
fn rejects_crossing_sloped_profiles() {
    assert!(segments_intersect(
        [1.0, 0.0],
        [3.0, 2.0],
        [3.0, 0.0],
        [1.0, 2.0]
    ));
    assert!(!segments_intersect(
        [1.0, 0.0],
        [2.0, 1.0],
        [3.0, 0.0],
        [4.0, 1.0]
    ));
}

#[test]
fn recovers_native_conical_frustum_fixture() -> anyhow::Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/native_conical_frustum.step");
    let recovered = crate::detect_solid_revolutions_bytes(bytes)?;
    assert_eq!(recovered.len(), 1);
    assert_eq!(
        recovered[0]
            .polygon_points()
            .ok_or_else(|| anyhow::anyhow!("expected polygon profile"))?,
        vec![[0.0, 0.0], [2.0, 0.0], [1.0, 2.0], [0.0, 2.0]]
    );
    assert!(recovered[0].max_residual_mm < 1.0e-9);
    assert_eq!(
        recovered[0].source_tolerance_mm,
        REVOLUTION_SOURCE_SUPPORT_TOL_MM
    );

    let declared_uncertainty = String::from_utf8_lossy(bytes).replacen(
        "LENGTH_MEASURE(1.E-07)",
        "LENGTH_MEASURE(1.E-05)",
        1,
    );
    let recovered_with_declared_uncertainty =
        crate::detect_solid_revolutions_bytes(declared_uncertainty.as_bytes())?;
    assert_eq!(recovered_with_declared_uncertainty.len(), 1);
    assert_eq!(
        recovered_with_declared_uncertainty[0].source_tolerance_mm,
        MAX_REVOLUTION_SOURCE_UNCERTAINTY_MM
    );

    let oversized_uncertainty = String::from_utf8_lossy(bytes).replacen(
        "LENGTH_MEASURE(1.E-07)",
        "LENGTH_MEASURE(1.E-04)",
        1,
    );
    let recovered_with_oversized_uncertainty =
        crate::detect_solid_revolutions_bytes(oversized_uncertainty.as_bytes())?;
    assert_eq!(recovered_with_oversized_uncertainty.len(), 1);
    assert_eq!(
        recovered_with_oversized_uncertainty[0].source_tolerance_mm,
        REVOLUTION_SOURCE_SUPPORT_TOL_MM
    );

    let tampered = String::from_utf8_lossy(bytes).replacen(
        "CONICAL_SURFACE('',#32,2.,0.463647609001)",
        "CONICAL_SURFACE('',#32,2.25,0.463647609001)",
        1,
    );
    assert!(crate::detect_solid_revolutions_bytes(tampered.as_bytes())?.is_empty());

    #[cfg(feature = "cad-kernel-monstertruck")]
    {
        use crate::cad_kernel::CadKernel;
        let fragment = crate::cad_recovery::recover_solid_revolution_fragment(&recovered[0])?;
        let kernel = crate::cad_kernel::monstertruck::MonstertruckKernel;
        let rebuilt = kernel.evaluate(&fragment.model, fragment.root)?;
        assert!(kernel.summarize(&rebuilt).geometrically_consistent);
        ruststep::parser::parse(&kernel.to_step(&rebuilt)?)?;
    }
    Ok(())
}

#[test]
fn recovers_native_hollow_conical_frustum_fixture() -> anyhow::Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/native_hollow_conical_frustum.step");
    let recovered = crate::detect_solid_revolutions_bytes(bytes)?;
    assert_eq!(recovered.len(), 1);
    assert_eq!(
        recovered[0]
            .polygon_points()
            .ok_or_else(|| anyhow::anyhow!("expected polygon profile"))?,
        vec![[0.5, 2.0], [1.0, 0.0], [3.0, 0.0], [2.0, 2.0]]
    );
    assert!(recovered[0].max_residual_mm < 1.0e-9);

    #[cfg(feature = "cad-kernel-monstertruck")]
    {
        use crate::cad_kernel::CadKernel;
        let fragment = crate::cad_recovery::recover_solid_revolution_fragment(&recovered[0])?;
        let kernel = crate::cad_kernel::monstertruck::MonstertruckKernel;
        let rebuilt = kernel.evaluate(&fragment.model, fragment.root)?;
        assert!(kernel.summarize(&rebuilt).geometrically_consistent);
        ruststep::parser::parse(&kernel.to_step(&rebuilt)?)?;
    }
    Ok(())
}

#[test]
fn recovers_native_line_surface_of_revolution_fixture() -> anyhow::Result<()> {
    let bytes =
        include_bytes!("../../validation/fixtures/native_line_surface_of_revolution_frustum.step");
    let recovered = crate::detect_solid_revolutions_bytes(bytes)?;
    assert_eq!(recovered.len(), 1);
    assert_eq!(
        recovered[0]
            .polygon_points()
            .ok_or_else(|| anyhow::anyhow!("expected polygon profile"))?,
        vec![[0.0, 0.0], [2.0, 0.0], [1.0, 2.0], [0.0, 2.0]]
    );
    assert!(recovered[0].max_residual_mm < 1.0e-9);

    let skewed = String::from_utf8_lossy(bytes).replacen(
        "CARTESIAN_POINT('',(2.,-4.898587196589E-16,0.))",
        "CARTESIAN_POINT('',(2.,0.25,0.))",
        1,
    );
    assert!(crate::detect_solid_revolutions_bytes(skewed.as_bytes())?.is_empty());

    #[cfg(feature = "cad-kernel-monstertruck")]
    {
        use crate::cad_kernel::CadKernel;
        let fragment = crate::cad_recovery::recover_solid_revolution_fragment(&recovered[0])?;
        let kernel = crate::cad_kernel::monstertruck::MonstertruckKernel;
        let rebuilt = kernel.evaluate(&fragment.model, fragment.root)?;
        assert!(kernel.summarize(&rebuilt).geometrically_consistent);
    }
    Ok(())
}

#[test]
fn recovers_native_spherical_cap_fixture() -> anyhow::Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/native_spherical_cap.step");
    let recovered = crate::detect_solid_revolutions_bytes(bytes)?;
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].face_ids.len(), 2);
    assert_eq!(recovered[0].profile_curves.len(), 3);

    let RecoveredProfileCurve::Line {
        start_mm: plane_axis,
        end_mm: cap_edge,
        ..
    } = &recovered[0].profile_curves[0]
    else {
        anyhow::bail!("expected planar radial segment");
    };
    assert!(plane_axis[0].abs() < 1.0e-12);
    assert!((plane_axis[1] - 0.073).abs() < 1.0e-12);
    assert!((cap_edge[0] - 0.107568582774).abs() < 1.0e-12);
    assert!((cap_edge[1] - 0.073).abs() < 1.0e-12);

    let RecoveredProfileCurve::CircleArc {
        center_mm,
        radius_mm,
        start_angle_rad,
        end_angle_rad,
        ..
    } = &recovered[0].profile_curves[1]
    else {
        anyhow::bail!("expected spherical meridian arc");
    };
    assert!(center_mm[0].abs() < 1.0e-12);
    assert!(center_mm[1].abs() < 1.0e-12);
    assert!((*radius_mm - 0.13).abs() < 1.0e-12);
    assert!((*start_angle_rad - 0.596243908486).abs() < 1.0e-12);
    assert!((*end_angle_rad + std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);

    let RecoveredProfileCurve::Line {
        start_mm: pole,
        end_mm: close,
        ..
    } = &recovered[0].profile_curves[2]
    else {
        anyhow::bail!("expected axis closure");
    };
    assert!(pole[0].abs() < 1.0e-12);
    assert!((pole[1] + 0.13).abs() < 1.0e-12);
    assert_eq!(*close, *plane_axis);
    assert!(recovered[0].max_residual_mm < 1.0e-9);

    let malformed = String::from_utf8_lossy(bytes).replacen(
        "CIRCLE('',#26,0.107568582774)",
        "CIRCLE('',#26,0.117568582774)",
        1,
    );
    assert!(crate::detect_solid_revolutions_bytes(malformed.as_bytes())?.is_empty());

    #[cfg(feature = "cad-kernel-monstertruck")]
    {
        use crate::cad_kernel::CadKernel;
        let fragment = crate::cad_recovery::recover_solid_revolution_fragment(&recovered[0])?;
        let kernel = crate::cad_kernel::monstertruck::MonstertruckKernel;
        let rebuilt = kernel.evaluate(&fragment.model, fragment.root)?;
        assert!(kernel.summarize(&rebuilt).geometrically_consistent);
        ruststep::parser::parse(&kernel.to_step(&rebuilt)?)?;
    }
    Ok(())
}

#[test]
fn recovers_positive_spherical_cap_fixture() -> anyhow::Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/native_spherical_cap_positive.step");
    let recovered = crate::detect_solid_revolutions_bytes(bytes)?;
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].profile_curves.len(), 3);
    let RecoveredProfileCurve::CircleArc { end_angle_rad, .. } = &recovered[0].profile_curves[1]
    else {
        anyhow::bail!("expected spherical meridian arc");
    };
    assert!((*end_angle_rad - std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);
    let RecoveredProfileCurve::Line { start_mm: pole, .. } = &recovered[0].profile_curves[2] else {
        anyhow::bail!("expected axis closure");
    };
    assert!((pole[1] - 0.13).abs() < 1.0e-12);
    assert!(recovered[0].max_residual_mm < 1.0e-10);

    #[cfg(feature = "cad-kernel-monstertruck")]
    {
        use crate::cad_kernel::CadKernel;
        let fragment = crate::cad_recovery::recover_solid_revolution_fragment(&recovered[0])?;
        let kernel = crate::cad_kernel::monstertruck::MonstertruckKernel;
        let rebuilt = kernel.evaluate(&fragment.model, fragment.root)?;
        assert!(kernel.summarize(&rebuilt).geometrically_consistent);
        ruststep::parser::parse(&kernel.to_step(&rebuilt)?)?;
    }
    Ok(())
}

#[test]
fn recovers_native_ring_torus_fixture() -> anyhow::Result<()> {
    let bytes = include_bytes!("../../validation/fixtures/native_torus.step");
    let recovered = crate::detect_solid_revolutions_bytes(bytes)?;
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].face_ids.len(), 1);
    assert_eq!(recovered[0].profile_curves.len(), 1);
    let RecoveredProfileCurve::CircleArc {
        center_mm,
        radius_mm,
        start_angle_rad,
        end_angle_rad,
        ..
    } = &recovered[0].profile_curves[0]
    else {
        anyhow::bail!("expected full-circle torus meridian");
    };
    assert!((center_mm[0] - 1.2).abs() < 1.0e-12);
    assert!(center_mm[1].abs() < 1.0e-12);
    assert!((*radius_mm - 0.09).abs() < 1.0e-12);
    assert_eq!(*start_angle_rad, 0.0);
    assert_eq!(*end_angle_rad, std::f64::consts::TAU);

    let malformed =
        String::from_utf8_lossy(bytes).replacen("CIRCLE('',#26,1.29)", "CIRCLE('',#26,1.31)", 1);
    assert!(crate::detect_solid_revolutions_bytes(malformed.as_bytes())?.is_empty());

    #[cfg(feature = "cad-kernel-monstertruck")]
    {
        use crate::cad_kernel::CadKernel;
        let fragment = crate::cad_recovery::recover_solid_revolution_fragment(&recovered[0])?;
        let kernel = crate::cad_kernel::monstertruck::MonstertruckKernel;
        let rebuilt = kernel.evaluate(&fragment.model, fragment.root)?;
        assert!(kernel.summarize(&rebuilt).geometrically_consistent);
        ruststep::parser::parse(&kernel.to_step(&rebuilt)?)?;
    }
    Ok(())
}

#[test]
fn closes_hollow_step_profile_and_deduplicates_patches() -> anyhow::Result<()> {
    let mut segments = Vec::new();
    for _ in 0..4 {
        push_unique_segment(
            &mut segments,
            Segment2 {
                a: [1.0, 0.0],
                b: [3.0, 0.0],
            },
        );
        push_unique_segment(
            &mut segments,
            Segment2 {
                a: [3.0, 0.0],
                b: [3.0, 2.0],
            },
        );
        push_unique_segment(
            &mut segments,
            Segment2 {
                a: [1.0, 2.0],
                b: [3.0, 2.0],
            },
        );
        push_unique_segment(
            &mut segments,
            Segment2 {
                a: [1.0, 0.0],
                b: [1.0, 2.0],
            },
        );
    }
    assert_eq!(segments.len(), 4);
    let profile = closed_profile_from_segments(segments)
        .ok_or_else(|| anyhow::anyhow!("expected closed profile"))?;
    assert_eq!(profile.len(), 4);
    assert!(profile.iter().all(|point| point[0] >= 1.0));
    Ok(())
}

#[test]
fn rejects_branching_or_self_intersecting_profiles() {
    assert!(
        closed_profile_from_segments(vec![
            Segment2 {
                a: [1.0, 0.0],
                b: [3.0, 0.0]
            },
            Segment2 {
                a: [3.0, 0.0],
                b: [3.0, 2.0]
            },
            Segment2 {
                a: [3.0, 2.0],
                b: [1.0, 2.0]
            },
            Segment2 {
                a: [1.0, 2.0],
                b: [1.0, 0.0]
            },
            Segment2 {
                a: [2.0, 0.0],
                b: [2.0, 2.0]
            },
        ])
        .is_none()
    );
}
