use anyhow::{Result, bail};
use serde::Serialize;
use std::collections::BTreeMap;

/// A step-redox-owned constructive node ID. It is intentionally unrelated to
/// STEP entity numbers or geometry-kernel handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct NodeId(pub usize);

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CadModel {
    pub nodes: Vec<CadNode>,
    pub roots: Vec<NodeId>,
    pub provenance: BTreeMap<NodeId, Provenance>,
}

impl CadModel {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            roots: Vec::new(),
            provenance: BTreeMap::new(),
        }
    }

    pub fn add_node(&mut self, node: CadNode) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    pub fn add_root(&mut self, id: NodeId) -> Result<()> {
        self.require_node(id)?;
        self.roots.push(id);
        Ok(())
    }

    pub fn set_provenance(&mut self, id: NodeId, provenance: Provenance) -> Result<()> {
        self.require_node(id)?;
        self.provenance.insert(id, provenance);
        Ok(())
    }

    pub fn node(&self, id: NodeId) -> Result<&CadNode> {
        self.nodes
            .get(id.0)
            .ok_or_else(|| anyhow::anyhow!("invalid CAD node id {}", id.0))
    }

    pub fn validate(&self) -> Result<()> {
        for &root in &self.roots {
            self.require_node(root)?;
        }
        for (index, node) in self.nodes.iter().enumerate() {
            let id = NodeId(index);
            for child in node.children() {
                self.require_node(child)?;
            }
            if let Some(provenance) = self.provenance.get(&id)
                && provenance.proof == ProofStatus::WithinTolerance
                && provenance.max_residual_mm.is_none()
            {
                bail!("node {} is WithinTolerance without max_residual_mm", id.0);
            }
        }

        let mut state = vec![0u8; self.nodes.len()];
        for index in 0..self.nodes.len() {
            self.validate_acyclic(NodeId(index), &mut state)?;
        }
        Ok(())
    }

    fn validate_acyclic(&self, id: NodeId, state: &mut [u8]) -> Result<()> {
        match state[id.0] {
            2 => return Ok(()),
            1 => bail!("cycle in CAD IR at node {}", id.0),
            _ => {}
        }
        state[id.0] = 1;
        for child in self.node(id)?.children() {
            self.validate_acyclic(child, state)?;
        }
        state[id.0] = 2;
        Ok(())
    }

    fn require_node(&self, id: NodeId) -> Result<()> {
        if id.0 >= self.nodes.len() {
            bail!(
                "CAD node {} is out of range for {} nodes",
                id.0,
                self.nodes.len()
            );
        }
        Ok(())
    }

    pub fn complexity_score(&self, root: NodeId) -> Result<u64> {
        self.require_node(root)?;
        self.validate()?;
        let mut seen = vec![false; self.nodes.len()];
        self.complexity_score_unique(root, &mut seen)
    }

    fn complexity_score_unique(&self, id: NodeId, seen: &mut [bool]) -> Result<u64> {
        if seen[id.0] {
            return Ok(0);
        }
        seen[id.0] = true;
        let node = self.node(id)?;
        let mut score = node.local_complexity();
        for child in node.children() {
            score += self.complexity_score_unique(child, seen)?;
        }
        Ok(score)
    }
}

impl Default for CadModel {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum CadNode {
    Primitive(Primitive),
    Extrude {
        profile: Profile2d,
        vector_mm: [f64; 3],
    },
    Revolve {
        profile: Profile2d,
        axis: Axis3,
        angle_rad: f64,
    },
    Sweep {
        profile: Profile2d,
        path: Path3d,
        frame: SweepFrame,
    },
    Boolean {
        op: BooleanOp,
        children: Vec<NodeId>,
    },
    Transform {
        transform: RigidTransform,
        child: NodeId,
    },
    Pattern {
        pattern: PatternSpec,
        child: NodeId,
    },
    Assembly {
        children: Vec<NodeId>,
    },
    BrepFallback(BrepFallback),
}

impl CadNode {
    pub fn children(&self) -> Vec<NodeId> {
        match self {
            Self::Boolean { children, .. } | Self::Assembly { children } => children.clone(),
            Self::Transform { child, .. } | Self::Pattern { child, .. } => vec![*child],
            _ => Vec::new(),
        }
    }

    fn local_complexity(&self) -> u64 {
        match self {
            Self::Primitive(_) => 1,
            Self::Extrude { profile, .. } | Self::Revolve { profile, .. } => {
                1 + profile.complexity()
            }
            Self::Sweep { profile, path, .. } => 2 + profile.complexity() + path.complexity(),
            Self::Boolean { children, .. } => 1 + children.len() as u64,
            Self::Transform { .. } => 2,
            Self::Pattern { pattern, .. } => 2 + pattern.complexity(),
            Self::Assembly { children } => 1 + children.len() as u64,
            Self::BrepFallback(fallback) => fallback.complexity(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Primitive {
    Box { size_mm: [f64; 3] },
    Cylinder { radius_mm: f64, height_mm: f64 },
    Sphere { radius_mm: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Profile2d {
    pub loops: Vec<ProfileLoop>,
}

impl Profile2d {
    pub fn polygon(points_mm: Vec<[f64; 2]>) -> Result<Self> {
        if points_mm.len() < 3 {
            bail!("polygon profile needs at least three points");
        }
        let mut curves = Vec::with_capacity(points_mm.len());
        for index in 0..points_mm.len() {
            curves.push(Curve2d::Line {
                start_mm: points_mm[index],
                end_mm: points_mm[(index + 1) % points_mm.len()],
            });
        }
        Ok(Self {
            loops: vec![ProfileLoop { curves }],
        })
    }

    fn complexity(&self) -> u64 {
        self.loops
            .iter()
            .map(|loop_| 1 + loop_.curves.iter().map(Curve2d::complexity).sum::<u64>())
            .sum()
    }

    pub fn single_polygon_points(&self) -> Option<Vec<[f64; 2]>> {
        if self.loops.len() != 1 {
            return None;
        }
        let curves = &self.loops[0].curves;
        if curves.len() < 3 {
            return None;
        }

        let mut points = Vec::with_capacity(curves.len());
        let mut previous_end = None;
        for curve in curves {
            let Curve2d::Line { start_mm, end_mm } = curve else {
                return None;
            };
            if previous_end.is_some_and(|end| end != *start_mm) {
                return None;
            }
            points.push(*start_mm);
            previous_end = Some(*end_mm);
        }
        if previous_end != points.first().copied() {
            return None;
        }
        Some(points)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProfileLoop {
    pub curves: Vec<Curve2d>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Curve2d {
    Line {
        start_mm: [f64; 2],
        end_mm: [f64; 2],
    },
    CircleArc {
        center_mm: [f64; 2],
        radius_mm: f64,
        start_angle_rad: f64,
        end_angle_rad: f64,
    },
    Bezier {
        control_points_mm: Vec<[f64; 2]>,
    },
    BSpline {
        degree: usize,
        control_points_mm: Vec<[f64; 2]>,
        knots: Vec<f64>,
        weights: Option<Vec<f64>>,
    },
}

impl Curve2d {
    fn complexity(&self) -> u64 {
        match self {
            Self::Line { .. } => 1,
            Self::CircleArc { .. } => 2,
            Self::Bezier { control_points_mm } => 2 + control_points_mm.len() as u64,
            Self::BSpline {
                control_points_mm,
                knots,
                weights,
                ..
            } => {
                4 + control_points_mm.len() as u64
                    + knots.len() as u64
                    + weights.as_ref().map_or(0, |values| values.len() as u64)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Path3d {
    Polyline {
        points_mm: Vec<[f64; 3]>,
    },
    BSpline {
        degree: usize,
        control_points_mm: Vec<[f64; 3]>,
        knots: Vec<f64>,
        weights: Option<Vec<f64>>,
    },
}

impl Path3d {
    fn complexity(&self) -> u64 {
        match self {
            Self::Polyline { points_mm } => 1 + points_mm.len() as u64,
            Self::BSpline {
                control_points_mm,
                knots,
                weights,
                ..
            } => {
                4 + control_points_mm.len() as u64
                    + knots.len() as u64
                    + weights.as_ref().map_or(0, |values| values.len() as u64)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Axis3 {
    pub origin_mm: [f64; 3],
    pub direction: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SweepFrame {
    Fixed,
    ParallelTransport,
    Frenet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BooleanOp {
    Union,
    Intersection,
    Difference,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct RigidTransform {
    pub matrix: [[f64; 4]; 4],
}

impl RigidTransform {
    pub fn identity() -> Self {
        Self {
            matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn translation_mm(xyz: [f64; 3]) -> Self {
        let mut result = Self::identity();
        result.matrix[0][3] = xyz[0];
        result.matrix[1][3] = xyz[1];
        result.matrix[2][3] = xyz[2];
        result
    }

    fn pure_translation(&self) -> Option<[f64; 3]> {
        let expected = Self::identity().matrix;
        for (row_index, row) in self.matrix.iter().enumerate() {
            for (column_index, value) in row.iter().enumerate() {
                if column_index == 3 && row_index < 3 {
                    continue;
                }
                if *value != expected[row_index][column_index] {
                    return None;
                }
            }
        }
        Some([self.matrix[0][3], self.matrix[1][3], self.matrix[2][3]])
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum PatternSpec {
    Linear {
        count: usize,
        step_mm: [f64; 3],
    },
    Grid {
        counts: [usize; 3],
        step_vectors_mm: [[f64; 3]; 3],
        occupancy: Option<Vec<[usize; 3]>>,
    },
    Polar {
        count: usize,
        axis: Axis3,
        total_angle_rad: f64,
        rotate_instances: bool,
    },
}

impl PatternSpec {
    fn complexity(&self) -> u64 {
        match self {
            Self::Linear { .. } => 2,
            Self::Grid { occupancy, .. } => {
                4 + occupancy.as_ref().map_or(0, |sites| sites.len() as u64)
            }
            Self::Polar { .. } => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BrepFallback {
    pub source_entity_ids: Vec<u64>,
    pub estimated_faces: usize,
    pub estimated_edges: usize,
    pub estimated_control_points: usize,
}

impl BrepFallback {
    fn complexity(&self) -> u64 {
        100 + self.estimated_faces as u64 * 8
            + self.estimated_edges as u64 * 3
            + self.estimated_control_points as u64
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Provenance {
    pub source_entity_ids: Vec<u64>,
    pub proof: ProofStatus,
    pub max_residual_mm: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ProofStatus {
    Exact,
    WithinTolerance,
    HeuristicCandidate,
}

/// Emit only the KCL subset for which step-redox has an exact lowering.
/// Unsupported nodes fail explicitly rather than being approximated.
pub fn emit_kcl(model: &CadModel) -> Result<String> {
    model.validate()?;
    let mut emitter = KclEmitter::new(model);
    for &root in &model.roots {
        emitter.emit_node(root)?;
    }
    Ok(emitter.output)
}

struct KclEmitter<'a> {
    model: &'a CadModel,
    emitted: Vec<bool>,
    output: String,
}

impl<'a> KclEmitter<'a> {
    fn new(model: &'a CadModel) -> Self {
        Self {
            model,
            emitted: vec![false; model.nodes.len()],
            output: String::new(),
        }
    }

    fn emit_node(&mut self, id: NodeId) -> Result<()> {
        if self.emitted[id.0] {
            return Ok(());
        }
        for child in self.model.node(id)?.children() {
            self.emit_node(child)?;
        }

        match self.model.node(id)? {
            CadNode::Extrude { profile, vector_mm } => {
                if vector_mm[0] != 0.0 || vector_mm[1] != 0.0 {
                    bail!("KCL emitter only supports Z extrusion today");
                }
                let points = profile.single_polygon_points().ok_or_else(|| {
                    anyhow::anyhow!("KCL emitter currently needs one line polygon")
                })?;
                emit_polygon_sketch(&mut self.output, id, &points)?;
                self.output.push_str(&format!(
                    "n{} = extrude(p{}, length = {})\n\n",
                    id.0,
                    id.0,
                    scalar(vector_mm[2])
                ));
            }
            CadNode::Transform { transform, child } => {
                let xyz = transform.pure_translation().ok_or_else(|| {
                    anyhow::anyhow!("KCL emitter currently supports translation only")
                })?;
                self.output.push_str(&format!(
                    "n{} = n{} |> translate(xyz = [{}, {}, {}], global = true)\n\n",
                    id.0,
                    child.0,
                    scalar(xyz[0]),
                    scalar(xyz[1]),
                    scalar(xyz[2])
                ));
            }
            CadNode::Pattern {
                pattern: PatternSpec::Linear { count, step_mm },
                child,
            } => {
                if *count == 0 {
                    bail!("linear pattern count must be at least one");
                }
                let distance =
                    (step_mm[0] * step_mm[0] + step_mm[1] * step_mm[1] + step_mm[2] * step_mm[2])
                        .sqrt();
                if !distance.is_finite() || distance == 0.0 {
                    bail!("linear pattern step must be finite and nonzero");
                }
                let axis = [
                    step_mm[0] / distance,
                    step_mm[1] / distance,
                    step_mm[2] / distance,
                ];
                self.output.push_str(&format!(
                    "n{} = n{} |> patternLinear3d(instances = {}, distance = {}, axis = [{}, {}, {}])\n\n",
                    id.0,
                    child.0,
                    count,
                    scalar(distance),
                    scalar(axis[0]),
                    scalar(axis[1]),
                    scalar(axis[2])
                ));
            }
            unsupported => bail!("KCL emitter does not support node {id:?}: {unsupported:?}"),
        }

        self.emitted[id.0] = true;
        Ok(())
    }
}

fn emit_polygon_sketch(output: &mut String, id: NodeId, points: &[[f64; 2]]) -> Result<()> {
    if points.len() < 3 {
        bail!("polygon needs at least three points");
    }
    let first = points[0];
    output.push_str(&format!(
        "s{} = startSketchOn(XY)\np{} = startProfile(s{}, at = [{}, {}])\n",
        id.0,
        id.0,
        id.0,
        scalar(first[0]),
        scalar(first[1])
    ));
    for pair in points.windows(2) {
        output.push_str(&format!(
            "  |> line(end = [{}, {}])\n",
            scalar(pair[1][0] - pair[0][0]),
            scalar(pair[1][1] - pair[0][1])
        ));
    }
    output.push_str("  |> close()\n");
    Ok(())
}

fn scalar(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    let mut text = format!("{value:.12}");
    while text.contains('.') && text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extrude_pattern_emits_compact_kcl_from_one_dag() -> Result<()> {
        let mut model = CadModel::new();
        let profile = Profile2d::polygon(vec![[0.0, 0.0], [10.0, 0.0], [10.0, 6.0], [0.0, 6.0]])?;
        let body = model.add_node(CadNode::Extrude {
            profile,
            vector_mm: [0.0, 0.0, 2.0],
        });
        let array = model.add_node(CadNode::Pattern {
            pattern: PatternSpec::Linear {
                count: 20,
                step_mm: [2.54, 0.0, 0.0],
            },
            child: body,
        });
        model.add_root(array)?;
        model.set_provenance(
            body,
            Provenance {
                source_entity_ids: vec![10, 11, 12],
                proof: ProofStatus::Exact,
                max_residual_mm: Some(0.0),
            },
        )?;

        let kcl = emit_kcl(&model)?;
        assert!(kcl.contains("extrude(p0, length = 2)"));
        assert!(kcl.contains("patternLinear3d(instances = 20, distance = 2.54, axis = [1, 0, 0])"));
        assert!(model.complexity_score(array)? < 20);
        Ok(())
    }

    #[test]
    fn complexity_counts_shared_dag_nodes_once() -> Result<()> {
        let mut model = CadModel::new();
        let primitive = model.add_node(CadNode::Primitive(Primitive::Box {
            size_mm: [1.0, 1.0, 1.0],
        }));
        let root = model.add_node(CadNode::Assembly {
            children: vec![primitive, primitive],
        });
        model.add_root(root)?;
        assert_eq!(model.complexity_score(root)?, 4);
        Ok(())
    }

    #[test]
    fn fallback_is_deliberately_expensive() -> Result<()> {
        let mut model = CadModel::new();
        let fallback = model.add_node(CadNode::BrepFallback(BrepFallback {
            source_entity_ids: vec![1, 2, 3],
            estimated_faces: 100,
            estimated_edges: 300,
            estimated_control_points: 1_000,
        }));
        model.add_root(fallback)?;
        assert!(model.complexity_score(fallback)? > 2_000);
        Ok(())
    }

    #[test]
    fn model_rejects_cycles() {
        let model = CadModel {
            nodes: vec![
                CadNode::Transform {
                    transform: RigidTransform::identity(),
                    child: NodeId(1),
                },
                CadNode::Transform {
                    transform: RigidTransform::identity(),
                    child: NodeId(0),
                },
            ],
            roots: vec![NodeId(0)],
            provenance: BTreeMap::new(),
        };
        assert!(model.validate().is_err());
    }
}
