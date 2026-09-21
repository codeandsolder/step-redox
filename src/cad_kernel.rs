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

#[cfg(feature = "cad-kernel-truck")]
pub mod truck {
    use super::{CadKernel, KernelSummary};
    use crate::cad_ir::{CadModel, CadNode, NodeId};
    use anyhow::{Result, bail};
    use truck_modeling::*;
    use truck_stepio::out::{self, StepDesign};

    #[derive(Debug, Default, Clone, Copy)]
    pub struct TruckKernel;

    /// Opaque wrapper: Truck topology does not escape the backend module.
    pub struct EvaluatedShape {
        solid: Solid,
    }

    fn extrude_polygon_z(points_mm: &[[f64; 2]], length_mm: f64) -> Result<Solid> {
        if points_mm.len() < 3 {
            bail!("polygon needs at least three points");
        }
        if !length_mm.is_finite() || length_mm == 0.0 {
            bail!("extrusion length must be finite and nonzero");
        }

        let vertices = points_mm
            .iter()
            .map(|[x, y]| builder::vertex(Point3::new(*x, *y, 0.0)))
            .collect::<Vec<_>>();
        let mut edges = Vec::with_capacity(vertices.len());
        for index in 0..vertices.len() {
            let next = (index + 1) % vertices.len();
            edges.push(builder::line(&vertices[index], &vertices[next]));
        }

        let wire: Wire = edges.into();
        let face: Face = builder::try_attach_plane(vec![wire])?;
        Ok(builder::tsweep(&face, Vector3::new(0.0, 0.0, length_mm)))
    }

    fn evaluate_node(model: &CadModel, root: NodeId) -> Result<Solid> {
        match model.node(root)? {
            CadNode::Extrude { profile, vector_mm } => {
                if vector_mm[0] != 0.0 || vector_mm[1] != 0.0 {
                    bail!("Truck backend currently supports local Z extrusion only");
                }
                let points = profile.single_polygon_points().ok_or_else(|| {
                    anyhow::anyhow!("Truck backend currently needs one line polygon")
                })?;
                extrude_polygon_z(&points, vector_mm[2])
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
                bail!("Truck backend does not yet evaluate node {root:?}: {unsupported:?}")
            }
        }
    }

    impl CadKernel for TruckKernel {
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
            let design = StepDesign::from_model(out::StepModel::from(&compressed));
            Ok(out::StepDisplay::new(
                out::StepHeaderDescriptor {
                    organization_system: "step-redox Truck backend".into(),
                    ..Default::default()
                },
                design,
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

            let error = match TruckKernel.evaluate(&model, patterned) {
                Ok(_) => panic!("unimplemented pattern unexpectedly evaluated"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("does not yet evaluate"));
            Ok(())
        }

        #[test]
        fn evaluates_box_without_exposing_truck_topology() -> Result<()> {
            let (model, root) = box_model()?;
            let kernel = TruckKernel;
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
