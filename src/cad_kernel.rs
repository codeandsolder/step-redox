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
    use crate::cad_ir::{CadModel, CadNode, Curve2d, NodeId, Profile2d};
    use anyhow::{Result, bail};
    use monstertruck_modeling::*;
    use monstertruck_io::step::save::{self, CompleteStepDisplay};

    #[derive(Debug, Default, Clone, Copy)]
    pub struct MonstertruckKernel;

    /// Opaque wrapper: Monstertruck topology does not escape the backend module.
    pub struct EvaluatedShape {
        solid: Solid,
    }

    fn extrude_profile_z(profile: &Profile2d, length_mm: f64) -> Result<Solid> {
        if !length_mm.is_finite() || length_mm == 0.0 {
            bail!("extrusion length must be finite and nonzero");
        }
        if profile.loops.len() != 1 {
            bail!("Monstertruck backend currently supports one profile loop");
        }
        let curves = &profile.loops[0].curves;
        if curves.is_empty() {
            bail!("profile loop must contain at least one curve");
        }

        let wire: Wire = if curves.len() == 1 {
            match &curves[0] {
                Curve2d::CircleArc {
                    center_mm,
                    radius_mm,
                    start_angle_rad,
                    end_angle_rad,
                } if ((end_angle_rad - start_angle_rad).abs() - std::f64::consts::TAU).abs()
                    <= 1.0e-9 =>
                {
                    let start = Point3::new(
                        center_mm[0] + radius_mm * start_angle_rad.cos(),
                        center_mm[1] + radius_mm * start_angle_rad.sin(),
                        0.0,
                    );
                    primitive::circle(
                        start,
                        Point3::new(center_mm[0], center_mm[1], 0.0),
                        Vector3::new(0.0, 0.0, 1.0),
                        2,
                    )
                }
                _ => profile_wire(curves)?,
            }
        } else {
            profile_wire(curves)?
        };

        let face: Face = builder::try_attach_plane(vec![wire])?;
        Ok(builder::extrude(&face, Vector3::new(0.0, 0.0, length_mm)))
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
                unsupported => {
                    bail!("Monstertruck backend does not yet evaluate profile curve {unsupported:?}")
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
            unsupported => bail!("unsupported profile curve {unsupported:?}"),
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
            unsupported => bail!("unsupported profile curve {unsupported:?}"),
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
            CadNode::Transform { transform, child } => {
                let child = evaluate_node(model, *child)?;
                let m = transform.matrix;
                let matrix = Matrix4::from_cols(
                    Vector4::new(m[0][0], m[1][0], m[2][0], m[3][0]),
                    Vector4::new(m[0][1], m[1][1], m[2][1], m[3][1]),
                    Vector4::new(m[0][2], m[1][2], m[2][2], m[3][2]),
                    Vector4::new(m[0][3], m[1][3], m[2][3], m[3][3]),
                );
                Ok(builder::transformed(&child, matrix))
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
        use crate::cad_ir::{CadNode, Profile2d};

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

        #[test]
        fn evaluates_semicircular_profile_extrusion() -> Result<()> {
            use crate::cad_ir::{Curve2d, ProfileLoop};

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
            use crate::cad_ir::{Curve2d, ProfileLoop};

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
