use crate::cad_ir::{CadModel, NodeId};
use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelSummary {
    pub geometrically_consistent: bool,
    pub shells: usize,
    pub faces: usize,
}

/// Narrow execution boundary for constructive CAD.
///
/// The associated evaluated value is intentionally backend-defined and opaque to
/// recovery code. The only stable outputs are a summary and serialized exchange
/// geometry that can be validated independently.
pub trait CadKernel {
    type Evaluated;

    fn evaluate(&self, model: &CadModel, root: NodeId) -> Result<Self::Evaluated>;
    fn summarize(&self, evaluated: &Self::Evaluated) -> KernelSummary;
    fn to_step(&self, evaluated: &Self::Evaluated) -> Result<String>;
}

#[cfg(feature = "cad-kernel-monstertruck")]
pub mod monstertruck {
    use super::{CadKernel, KernelSummary};
    use crate::cad_ir::{Axis3, CadModel, CadNode, Curve2d, NodeId, Profile2d, RigidTransform};
    use anyhow::{Result, bail};
    use monstertruck_io::step::save::{self, CompleteStepDisplay};
    use monstertruck_modeling::*;

    #[derive(Debug, Default, Clone, Copy)]
    pub struct MonstertruckKernel;

    /// Opaque wrapper: Monstertruck topology does not escape the backend module.
    pub struct EvaluatedShape {
        solid: Solid,
    }

    fn profile_wires(profile: &Profile2d) -> Result<Vec<Wire>> {
        if profile.loops.is_empty() {
            bail!("profile must contain at least one loop");
        }
        profile
            .loops
            .iter()
            .map(|loop_| profile_loop_wire(&loop_.curves))
            .collect::<Result<Vec<_>>>()
    }

    fn extrude_profile_z(profile: &Profile2d, length_mm: f64) -> Result<Solid> {
        if !length_mm.is_finite() || length_mm == 0.0 {
            bail!("extrusion length must be finite and nonzero");
        }
        Ok(profile::solid_from_planar_profile(
            profile_wires(profile)?,
            Vector3::new(0.0, 0.0, length_mm),
        )?)
    }

    fn revolve_profile(profile: &Profile2d, axis: Axis3, angle_rad: f64) -> Result<Solid> {
        revolve_wires(profile_wires(profile)?, axis, angle_rad)
    }

    fn revolve_profile_transformed(
        profile: &Profile2d,
        axis: Axis3,
        angle_rad: f64,
        transform: &RigidTransform,
    ) -> Result<Solid> {
        let matrix = rigid_matrix(transform);
        let wires = profile_wires(profile)?
            .into_iter()
            .map(|wire| builder::transformed(&wire, matrix))
            .collect::<Vec<_>>();
        revolve_wires(wires, transform_axis(axis, transform), angle_rad)
    }

    fn revolve_wires(mut wires: Vec<Wire>, axis: Axis3, angle_rad: f64) -> Result<Solid> {
        if !angle_rad.is_finite() || angle_rad == 0.0 {
            bail!("revolution angle must be finite and nonzero");
        }
        if angle_rad.abs() > std::f64::consts::TAU + 1.0e-10 {
            bail!("revolution angle must not exceed one full turn");
        }
        if axis.origin_mm.iter().any(|value| !value.is_finite())
            || axis.direction.iter().any(|value| !value.is_finite())
        {
            bail!("revolution axis must be finite");
        }

        let direction = Vector3::new(axis.direction[0], axis.direction[1], axis.direction[2]);
        let magnitude = direction.magnitude();
        if !magnitude.is_finite() || magnitude <= 1.0e-12 {
            bail!("revolution axis direction must be nonzero");
        }
        let direction = direction / magnitude;
        let origin = Point3::new(axis.origin_mm[0], axis.origin_mm[1], axis.origin_mm[2]);
        let closed = (angle_rad.abs() - std::f64::consts::TAU).abs() <= 1.0e-10;
        let sweep = if closed {
            builder::SweepAngle::Closed
        } else {
            builder::SweepAngle::Partial(Rad(angle_rad))
        };

        let touches_axis = wires.iter().any(|wire| {
            wire.vertex_iter()
                .any(|vertex| point_axis_distance(vertex.point(), origin, direction) <= 1.0e-9)
        });

        if touches_axis {
            if !closed {
                bail!("partial revolution with a profile touching the axis is not yet supported");
            }
            if wires.len() != 1 {
                bail!("axis-touching revolution with profile holes is not yet supported");
            }

            let mut wire = wires
                .pop()
                .ok_or_else(|| anyhow::anyhow!("missing profile wire"))?;
            let axis_edges = wire
                .iter()
                .enumerate()
                .filter_map(|(index, edge)| {
                    (point_axis_distance(edge.front().point(), origin, direction) <= 1.0e-9
                        && point_axis_distance(edge.back().point(), origin, direction) <= 1.0e-9)
                        .then_some(index)
                })
                .collect::<Vec<_>>();
            let [axis_edge_index] = axis_edges.as_slice() else {
                bail!("axis-touching closed profile needs exactly one line segment on the axis");
            };

            wire.rotate_left(*axis_edge_index);
            wire.pop_front()
                .ok_or_else(|| anyhow::anyhow!("missing axis profile edge"))?;
            let Some(front) = wire.front_vertex() else {
                bail!("axis-touching revolution has no generatrix");
            };
            let Some(back) = wire.back_vertex() else {
                bail!("axis-touching revolution has no generatrix");
            };
            if point_axis_distance(front.point(), origin, direction) > 1.0e-9
                || point_axis_distance(back.point(), origin, direction) > 1.0e-9
            {
                bail!("revolution generatrix endpoints must lie on the axis");
            }

            let shell =
                builder::revolve_wire(&wire, origin, direction, builder::SweepAngle::Closed, 4);
            return Ok(Solid::try_new(vec![shell])?);
        }

        let face = profile::attach_plane_normalized(wires)?;
        Ok(builder::revolve(
            &face,
            origin,
            direction,
            sweep,
            if closed { 4 } else { 1 },
        ))
    }

    fn rigid_matrix(transform: &RigidTransform) -> Matrix4 {
        let m = transform.matrix;
        Matrix4::from_cols(
            Vector4::new(m[0][0], m[1][0], m[2][0], m[3][0]),
            Vector4::new(m[0][1], m[1][1], m[2][1], m[3][1]),
            Vector4::new(m[0][2], m[1][2], m[2][2], m[3][2]),
            Vector4::new(m[0][3], m[1][3], m[2][3], m[3][3]),
        )
    }

    fn transform_axis(axis: Axis3, transform: &RigidTransform) -> Axis3 {
        let m = transform.matrix;
        let [x, y, z] = axis.origin_mm;
        let [dx, dy, dz] = axis.direction;
        Axis3 {
            origin_mm: [
                m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
                m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
                m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3],
            ],
            direction: [
                m[0][0] * dx + m[0][1] * dy + m[0][2] * dz,
                m[1][0] * dx + m[1][1] * dy + m[1][2] * dz,
                m[2][0] * dx + m[2][1] * dy + m[2][2] * dz,
            ],
        }
    }

    fn point_axis_distance(point: Point3, origin: Point3, direction: Vector3) -> f64 {
        (point - origin).cross(direction).magnitude()
    }

    fn profile_loop_wire(curves: &[Curve2d]) -> Result<Wire> {
        if curves.is_empty() {
            bail!("profile loop must contain at least one curve");
        }
        if curves.len() == 1
            && let Curve2d::CircleArc {
                center_mm,
                radius_mm,
                start_angle_rad,
                end_angle_rad,
            } = &curves[0]
            && ((end_angle_rad - start_angle_rad).abs() - std::f64::consts::TAU).abs() <= 1.0e-9
        {
            let start = Point3::new(
                center_mm[0] + radius_mm * start_angle_rad.cos(),
                center_mm[1] + radius_mm * start_angle_rad.sin(),
                0.0,
            );
            return Ok(primitive::circle(
                start,
                Point3::new(center_mm[0], center_mm[1], 0.0),
                Vector3::new(0.0, 0.0, 1.0),
                2,
            ));
        }
        profile_wire(curves)
    }

    fn profile_wire(curves: &[Curve2d]) -> Result<Wire> {
        let starts = curves
            .iter()
            .map(curve_start_point)
            .collect::<Result<Vec<_>>>()?;
        let vertices = starts
            .iter()
            .map(|[x, y]| builder::vertex(Point3::new(*x, *y, 0.0)))
            .collect::<Vec<_>>();
        let mut edges = Vec::with_capacity(curves.len());

        for (index, curve) in curves.iter().enumerate() {
            let next = (index + 1) % curves.len();
            let expected_end = curve_end_point(curve)?;
            if point2_distance(expected_end, starts[next]) > 1.0e-8 {
                bail!("profile curve endpoints are not topologically continuous");
            }
            match curve {
                Curve2d::Line { .. } => {
                    edges.push(builder::line(&vertices[index], &vertices[next]));
                }
                Curve2d::CircleArc {
                    center_mm,
                    radius_mm,
                    start_angle_rad,
                    end_angle_rad,
                } => {
                    let sweep = end_angle_rad - start_angle_rad;
                    if !sweep.is_finite() || sweep.abs() <= 1.0e-12 {
                        bail!("circle arc sweep must be finite and nonzero");
                    }
                    if sweep.abs() >= std::f64::consts::TAU - 1.0e-9 {
                        bail!("full circles must be represented by one profile curve");
                    }
                    let midpoint = (start_angle_rad + end_angle_rad) * 0.5;
                    let transit = Point3::new(
                        center_mm[0] + radius_mm * midpoint.cos(),
                        center_mm[1] + radius_mm * midpoint.sin(),
                        0.0,
                    );
                    edges.push(builder::circle_arc(
                        &vertices[index],
                        &vertices[next],
                        transit,
                    ));
                }
                Curve2d::Bezier { control_points_mm } => {
                    if control_points_mm.len() < 2 {
                        bail!("Bezier profile curve needs at least two control points");
                    }
                    let inter_points = control_points_mm[1..control_points_mm.len() - 1]
                        .iter()
                        .map(|[x, y]| Point3::new(*x, *y, 0.0))
                        .collect::<Vec<_>>();
                    edges.push(builder::bezier(
                        &vertices[index],
                        &vertices[next],
                        inter_points,
                    ));
                }
                Curve2d::BSpline {
                    degree,
                    control_points_mm,
                    knots,
                    weights,
                } => {
                    let Some(min_control_points) = degree.checked_add(1) else {
                        bail!("B-spline profile degree overflows dimensions");
                    };
                    let Some(expected_knots) =
                        control_points_mm.len().checked_add(min_control_points)
                    else {
                        bail!("B-spline profile knot count overflows dimensions");
                    };
                    let finite_points = control_points_mm
                        .iter()
                        .flatten()
                        .all(|value| value.is_finite());
                    let endpoint_clamped =
                        min_control_points.checked_mul(2).is_some_and(|min_knots| {
                            knots.len() >= min_knots
                                && knots[..min_control_points]
                                    .iter()
                                    .all(|knot| *knot == knots[0])
                                && knots[knots.len() - min_control_points..]
                                    .iter()
                                    .all(|knot| *knot == knots[knots.len() - 1])
                        });
                    if *degree == 0
                        || control_points_mm.len() < min_control_points
                        || knots.len() != expected_knots
                        || !finite_points
                        || !knots.iter().all(|knot| knot.is_finite())
                        || knots.windows(2).any(|pair| pair[0] > pair[1])
                        || knots.first() == knots.last()
                        || !endpoint_clamped
                    {
                        bail!("invalid B-spline profile geometry");
                    }
                    if let Some(weights) = weights
                        && (weights.len() != control_points_mm.len()
                            || weights
                                .iter()
                                .any(|weight| !weight.is_finite() || *weight <= 0.0))
                    {
                        bail!("invalid B-spline profile weights");
                    }
                    let control_points = control_points_mm
                        .iter()
                        .map(|[x, y]| Point3::new(*x, *y, 0.0))
                        .collect::<Vec<_>>();
                    let curve = BsplineCurve::new(KnotVector::from(knots.clone()), control_points);
                    let geometry = if let Some(weights) = weights {
                        Curve::NurbsCurve(NurbsCurve::try_from_bspline_and_weights(
                            curve,
                            weights.clone(),
                        )?)
                    } else {
                        Curve::BsplineCurve(curve)
                    };
                    edges.push(Edge::new(&vertices[index], &vertices[next], geometry));
                }
            }
        }

        Ok(edges.into())
    }

    fn curve_start_point(curve: &Curve2d) -> Result<[f64; 2]> {
        match curve {
            Curve2d::Line { start_mm, .. } => Ok(*start_mm),
            Curve2d::CircleArc {
                center_mm,
                radius_mm,
                start_angle_rad,
                ..
            } => Ok([
                center_mm[0] + radius_mm * start_angle_rad.cos(),
                center_mm[1] + radius_mm * start_angle_rad.sin(),
            ]),
            Curve2d::Bezier { control_points_mm }
            | Curve2d::BSpline {
                control_points_mm, ..
            } => control_points_mm
                .first()
                .copied()
                .ok_or_else(|| anyhow::anyhow!("spline profile has no control points")),
        }
    }

    fn curve_end_point(curve: &Curve2d) -> Result<[f64; 2]> {
        match curve {
            Curve2d::Line { end_mm, .. } => Ok(*end_mm),
            Curve2d::CircleArc {
                center_mm,
                radius_mm,
                end_angle_rad,
                ..
            } => Ok([
                center_mm[0] + radius_mm * end_angle_rad.cos(),
                center_mm[1] + radius_mm * end_angle_rad.sin(),
            ]),
            Curve2d::Bezier { control_points_mm }
            | Curve2d::BSpline {
                control_points_mm, ..
            } => control_points_mm
                .last()
                .copied()
                .ok_or_else(|| anyhow::anyhow!("spline profile has no control points")),
        }
    }

    fn point2_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
    }

    fn evaluate_node(model: &CadModel, root: NodeId) -> Result<Solid> {
        match model.node(root)? {
            CadNode::Extrude { profile, vector_mm } => {
                if vector_mm[0] != 0.0 || vector_mm[1] != 0.0 {
                    bail!("Monstertruck backend currently supports local Z extrusion only");
                }
                extrude_profile_z(profile, vector_mm[2])
            }
            CadNode::Revolve {
                profile,
                axis,
                angle_rad,
            } => revolve_profile(profile, *axis, *angle_rad),
            CadNode::Transform { transform, child } => {
                if let CadNode::Revolve {
                    profile,
                    axis,
                    angle_rad,
                } = model.node(*child)?
                {
                    return revolve_profile_transformed(profile, *axis, *angle_rad, transform);
                }
                let child = evaluate_node(model, *child)?;
                Ok(builder::transformed(&child, rigid_matrix(transform)))
            }
            unsupported => {
                bail!("Monstertruck backend does not yet evaluate node {root:?}: {unsupported:?}")
            }
        }
    }

    impl CadKernel for MonstertruckKernel {
        type Evaluated = EvaluatedShape;

        fn evaluate(&self, model: &CadModel, root: NodeId) -> Result<Self::Evaluated> {
            model.validate()?;
            Ok(EvaluatedShape {
                solid: evaluate_node(model, root)?,
            })
        }

        fn summarize(&self, evaluated: &Self::Evaluated) -> KernelSummary {
            let boundaries = evaluated.solid.boundaries();
            KernelSummary {
                geometrically_consistent: evaluated.solid.is_geometric_consistent(),
                shells: boundaries.len(),
                faces: boundaries.iter().map(|shell| shell.len()).sum(),
            }
        }

        fn to_step(&self, evaluated: &Self::Evaluated) -> Result<String> {
            let compressed = evaluated.solid.compress();
            Ok(CompleteStepDisplay::new(
                save::StepModel::from(&compressed),
                save::StepHeaderDescriptor {
                    organization_system: "step-redox Monstertruck backend".into(),
                    ..Default::default()
                },
            )
            .to_string())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::cad_ir::{Axis3, CadNode, Curve2d, Profile2d, ProfileLoop};

        fn box_model() -> Result<(CadModel, NodeId)> {
            let mut model = CadModel::new();
            let profile =
                Profile2d::polygon(vec![[0.0, 0.0], [10.0, 0.0], [10.0, 6.0], [0.0, 6.0]])?;
            let root = model.add_node(CadNode::Extrude {
                profile,
                vector_mm: [0.0, 0.0, 2.0],
            });
            model.add_root(root)?;
            Ok((model, root))
        }

        fn curved_profile_model(curve: Curve2d) -> Result<(CadModel, NodeId)> {
            let mut model = CadModel::new();
            let profile = Profile2d {
                loops: vec![ProfileLoop {
                    curves: vec![
                        curve,
                        Curve2d::Line {
                            start_mm: [2.0, 0.0],
                            end_mm: [0.0, 0.0],
                        },
                    ],
                }],
            };
            let root = model.add_node(CadNode::Extrude {
                profile,
                vector_mm: [0.0, 0.0, 2.0],
            });
            model.add_root(root)?;
            Ok((model, root))
        }

        fn assert_curved_profile_evaluates(
            curve: Curve2d,
            expected_step_fragment: &str,
        ) -> Result<()> {
            let (model, root) = curved_profile_model(curve)?;
            let kernel = MonstertruckKernel;
            let evaluated = kernel.evaluate(&model, root)?;
            let summary = kernel.summarize(&evaluated);
            assert!(summary.geometrically_consistent);
            assert_eq!(summary.shells, 1);
            assert_eq!(summary.faces, 4);

            let step = kernel.to_step(&evaluated)?;
            assert!(step.contains(expected_step_fragment));
            ruststep::parser::parse(&step)?;
            Ok(())
        }

        #[test]
        fn evaluates_full_profile_revolution() -> Result<()> {
            let mut model = CadModel::new();
            let profile =
                Profile2d::polygon(vec![[1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [1.0, 1.0]])?;
            let root = model.add_node(CadNode::Revolve {
                profile,
                axis: Axis3 {
                    origin_mm: [0.0, 0.0, 0.0],
                    direction: [0.0, 1.0, 0.0],
                },
                angle_rad: std::f64::consts::TAU,
            });
            model.add_root(root)?;

            let kernel = MonstertruckKernel;
            let evaluated = kernel.evaluate(&model, root)?;
            let summary = kernel.summarize(&evaluated);
            assert!(summary.geometrically_consistent);
            assert_eq!(summary.shells, 1);
            let step = kernel.to_step(&evaluated)?;
            ruststep::parser::parse(&step)?;
            let recovered = crate::detect_solid_revolutions_bytes(step.as_bytes())?;
            assert_eq!(recovered.len(), 1);
            let fragment = crate::cad_recovery::recover_solid_revolution_fragment(&recovered[0])?;
            let rebuilt = kernel.evaluate(&fragment.model, fragment.root)?;
            assert!(kernel.summarize(&rebuilt).geometrically_consistent);
            ruststep::parser::parse(&kernel.to_step(&rebuilt)?)?;
            Ok(())
        }

        #[test]
        fn evaluates_partial_profile_revolution() -> Result<()> {
            let mut model = CadModel::new();
            let profile = Profile2d::polygon(vec![[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]])?;
            let root = model.add_node(CadNode::Revolve {
                profile,
                axis: Axis3 {
                    origin_mm: [0.0, 0.0, 0.0],
                    direction: [0.0, 1.0, 0.0],
                },
                angle_rad: std::f64::consts::FRAC_PI_2,
            });
            model.add_root(root)?;

            let kernel = MonstertruckKernel;
            let evaluated = kernel.evaluate(&model, root)?;
            let summary = kernel.summarize(&evaluated);
            assert!(summary.geometrically_consistent);
            assert_eq!(summary.shells, 1);
            let step = kernel.to_step(&evaluated)?;
            ruststep::parser::parse(&step)?;
            Ok(())
        }

        #[test]
        fn evaluates_on_axis_profile_revolution() -> Result<()> {
            let mut model = CadModel::new();
            let profile =
                Profile2d::polygon(vec![[0.0, -1.0], [2.0, -1.0], [2.0, 1.0], [0.0, 1.0]])?;
            let root = model.add_node(CadNode::Revolve {
                profile,
                axis: Axis3 {
                    origin_mm: [0.0, 0.0, 0.0],
                    direction: [0.0, 1.0, 0.0],
                },
                angle_rad: std::f64::consts::TAU,
            });
            model.add_root(root)?;

            let kernel = MonstertruckKernel;
            let evaluated = kernel.evaluate(&model, root)?;
            let summary = kernel.summarize(&evaluated);
            assert!(summary.geometrically_consistent);
            assert_eq!(summary.shells, 1);
            let step = kernel.to_step(&evaluated)?;
            ruststep::parser::parse(&step)?;
            let recovered = crate::detect_solid_revolutions_bytes(step.as_bytes())?;
            assert_eq!(recovered.len(), 1);
            let fragment = crate::cad_recovery::recover_solid_revolution_fragment(&recovered[0])?;
            let rebuilt = kernel.evaluate(&fragment.model, fragment.root)?;
            assert!(kernel.summarize(&rebuilt).geometrically_consistent);
            ruststep::parser::parse(&kernel.to_step(&rebuilt)?)?;
            Ok(())
        }

        #[test]
        fn evaluates_transformed_profile_revolution_without_post_transform_breakage() -> Result<()>
        {
            let mut model = CadModel::new();
            let profile = Profile2d::polygon(vec![[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]])?;
            let body = model.add_node(CadNode::Revolve {
                profile,
                axis: Axis3 {
                    origin_mm: [0.0, 0.0, 0.0],
                    direction: [0.0, 1.0, 0.0],
                },
                angle_rad: std::f64::consts::TAU,
            });
            let root = model.add_node(CadNode::Transform {
                transform: RigidTransform {
                    matrix: [
                        [0.0, 0.0, 1.0, 10.0],
                        [1.0, 0.0, 0.0, 20.0],
                        [0.0, 1.0, 0.0, 30.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
                child: body,
            });
            model.add_root(root)?;

            let kernel = MonstertruckKernel;
            let evaluated = kernel.evaluate(&model, root)?;
            let summary = kernel.summarize(&evaluated);
            assert!(summary.geometrically_consistent);
            assert_eq!(summary.shells, 1);
            ruststep::parser::parse(&kernel.to_step(&evaluated)?)?;
            Ok(())
        }

        #[test]
        fn evaluates_quadratic_bezier_profile_extrusion() -> Result<()> {
            assert_curved_profile_evaluates(
                Curve2d::Bezier {
                    control_points_mm: vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
                },
                "B_SPLINE_CURVE",
            )
        }

        #[test]
        fn evaluates_quadratic_bspline_profile_extrusion() -> Result<()> {
            assert_curved_profile_evaluates(
                Curve2d::BSpline {
                    degree: 2,
                    control_points_mm: vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
                    knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    weights: None,
                },
                "B_SPLINE_CURVE",
            )
        }

        #[test]
        fn evaluates_rational_bspline_profile_extrusion() -> Result<()> {
            assert_curved_profile_evaluates(
                Curve2d::BSpline {
                    degree: 2,
                    control_points_mm: vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
                    knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    weights: Some(vec![1.0, 0.75, 1.0]),
                },
                "RATIONAL_B_SPLINE_CURVE",
            )
        }

        #[test]
        fn roundtrips_holed_extrusion_through_recovery() -> Result<()> {
            let mut model = CadModel::new();
            let profile = Profile2d {
                loops: vec![
                    ProfileLoop {
                        curves: vec![
                            Curve2d::Line {
                                start_mm: [0.0, 0.0],
                                end_mm: [8.0, 0.0],
                            },
                            Curve2d::Line {
                                start_mm: [8.0, 0.0],
                                end_mm: [8.0, 6.0],
                            },
                            Curve2d::Line {
                                start_mm: [8.0, 6.0],
                                end_mm: [0.0, 6.0],
                            },
                            Curve2d::Line {
                                start_mm: [0.0, 6.0],
                                end_mm: [0.0, 0.0],
                            },
                        ],
                    },
                    ProfileLoop {
                        curves: vec![Curve2d::CircleArc {
                            center_mm: [4.0, 3.0],
                            radius_mm: 1.25,
                            start_angle_rad: 0.0,
                            end_angle_rad: -std::f64::consts::TAU,
                        }],
                    },
                ],
            };
            let root = model.add_node(CadNode::Extrude {
                profile,
                vector_mm: [0.0, 0.0, 2.5],
            });
            model.add_root(root)?;

            let kernel = MonstertruckKernel;
            let evaluated = kernel.evaluate(&model, root)?;
            let summary = kernel.summarize(&evaluated);
            assert!(summary.geometrically_consistent);
            assert_eq!(summary.shells, 1);
            assert_eq!(summary.faces, 8);

            let step = kernel.to_step(&evaluated)?;
            ruststep::parser::parse(&step)?;
            let recovered = crate::detect_solid_extrusions_bytes(step.as_bytes())?;
            assert_eq!(recovered.len(), 1);
            assert_eq!(recovered[0].inner_profile_loops.len(), 1);
            assert_eq!(recovered[0].inner_profile_loops[0].len(), 2);
            assert!(
                recovered[0].inner_profile_loops[0]
                    .iter()
                    .all(|curve| matches!(
                        curve,
                        crate::solid_extrusions::RecoveredProfileCurve::BSpline {
                            weights: Some(_),
                            ..
                        }
                    ))
            );

            let fragment = crate::cad_recovery::recover_solid_extrusion_fragment(&recovered[0])?;
            let CadNode::Transform { child, .. } = fragment.model.node(fragment.root)? else {
                panic!("expected transform root");
            };
            let CadNode::Extrude { profile, .. } = fragment.model.node(*child)? else {
                panic!("expected extrusion child");
            };
            assert_eq!(profile.loops.len(), 2);

            let rebuilt = kernel.evaluate(&fragment.model, fragment.root)?;
            let rebuilt_summary = kernel.summarize(&rebuilt);
            assert!(rebuilt_summary.geometrically_consistent);
            assert_eq!(rebuilt_summary.faces, 8);
            ruststep::parser::parse(&kernel.to_step(&rebuilt)?)?;
            Ok(())
        }

        #[test]
        fn roundtrips_rational_bspline_extrusion_through_recovery() -> Result<()> {
            let (model, root) = curved_profile_model(Curve2d::BSpline {
                degree: 2,
                control_points_mm: vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
                knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                weights: Some(vec![1.0, 0.75, 1.0]),
            })?;
            let kernel = MonstertruckKernel;
            let evaluated = kernel.evaluate(&model, root)?;
            let step = kernel.to_step(&evaluated)?;
            let recovered = crate::detect_solid_extrusions_bytes(step.as_bytes())?;
            assert_eq!(recovered.len(), 1);
            assert!(recovered[0].profile_curves.iter().any(|curve| matches!(
                curve,
                crate::solid_extrusions::RecoveredProfileCurve::BSpline {
                    weights: Some(_),
                    ..
                }
            )));

            let fragment = crate::cad_recovery::recover_solid_extrusion_fragment(&recovered[0])?;
            let rebuilt = kernel.evaluate(&fragment.model, fragment.root)?;
            assert!(kernel.summarize(&rebuilt).geometrically_consistent);
            ruststep::parser::parse(&kernel.to_step(&rebuilt)?)?;
            Ok(())
        }

        #[test]
        fn malformed_bspline_profile_fails_closed() -> Result<()> {
            let mut model = CadModel::new();
            let profile = Profile2d {
                loops: vec![ProfileLoop {
                    curves: vec![
                        Curve2d::Line {
                            start_mm: [0.0, 0.0],
                            end_mm: [2.0, 0.0],
                        },
                        Curve2d::BSpline {
                            degree: usize::MAX,
                            control_points_mm: vec![[2.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
                            knots: Vec::new(),
                            weights: None,
                        },
                    ],
                }],
            };
            let root = model.add_node(CadNode::Extrude {
                profile,
                vector_mm: [0.0, 0.0, 1.0],
            });
            model.add_root(root)?;
            let kernel = MonstertruckKernel;
            assert!(kernel.evaluate(&model, root).is_err());
            Ok(())
        }

        #[test]
        fn evaluates_semicircular_profile_extrusion() -> Result<()> {
            let mut model = CadModel::new();
            let profile = Profile2d {
                loops: vec![ProfileLoop {
                    curves: vec![
                        Curve2d::Line {
                            start_mm: [-1.0, 0.0],
                            end_mm: [1.0, 0.0],
                        },
                        Curve2d::CircleArc {
                            center_mm: [0.0, 0.0],
                            radius_mm: 1.0,
                            start_angle_rad: 0.0,
                            end_angle_rad: std::f64::consts::PI,
                        },
                    ],
                }],
            };
            let root = model.add_node(CadNode::Extrude {
                profile,
                vector_mm: [0.0, 0.0, 2.0],
            });
            model.add_root(root)?;

            let kernel = MonstertruckKernel;
            let evaluated = kernel.evaluate(&model, root)?;
            let summary = kernel.summarize(&evaluated);
            assert!(summary.geometrically_consistent);
            assert_eq!(summary.shells, 1);
            let step = kernel.to_step(&evaluated)?;
            ruststep::parser::parse(&step)?;
            Ok(())
        }

        #[test]
        fn evaluates_full_circle_profile_extrusion() -> Result<()> {
            let mut model = CadModel::new();
            let profile = Profile2d {
                loops: vec![ProfileLoop {
                    curves: vec![Curve2d::CircleArc {
                        center_mm: [0.0, 0.0],
                        radius_mm: 2.0,
                        start_angle_rad: 0.0,
                        end_angle_rad: std::f64::consts::TAU,
                    }],
                }],
            };
            let root = model.add_node(CadNode::Extrude {
                profile,
                vector_mm: [0.0, 0.0, 5.0],
            });
            model.add_root(root)?;

            let kernel = MonstertruckKernel;
            let evaluated = kernel.evaluate(&model, root)?;
            let summary = kernel.summarize(&evaluated);
            assert!(summary.geometrically_consistent);
            assert_eq!(summary.shells, 1);
            let step = kernel.to_step(&evaluated)?;
            assert!(step.starts_with("ISO-10303-21;"));
            ruststep::parser::parse(&step)?;
            Ok(())
        }

        #[test]
        fn refuses_unimplemented_pattern_execution() -> Result<()> {
            use crate::cad_ir::PatternSpec;

            let (mut model, body) = box_model()?;
            model.roots.clear();
            let patterned = model.add_node(CadNode::Pattern {
                pattern: PatternSpec::Linear {
                    count: 4,
                    step_mm: [2.54, 0.0, 0.0],
                },
                child: body,
            });
            model.add_root(patterned)?;

            let error = match MonstertruckKernel.evaluate(&model, patterned) {
                Ok(_) => panic!("unimplemented pattern unexpectedly evaluated"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("does not yet evaluate"));
            Ok(())
        }

        #[test]
        fn evaluates_box_without_exposing_monstertruck_topology() -> Result<()> {
            let (model, root) = box_model()?;
            let kernel = MonstertruckKernel;
            let evaluated = kernel.evaluate(&model, root)?;
            let summary = kernel.summarize(&evaluated);
            assert!(summary.geometrically_consistent);
            assert_eq!(summary.shells, 1);
            assert_eq!(summary.faces, 6);

            let step = kernel.to_step(&evaluated)?;
            assert!(step.starts_with("ISO-10303-21;"));
            ruststep::parser::parse(&step)?;
            Ok(())
        }
    }
}
