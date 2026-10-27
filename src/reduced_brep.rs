use crate::brep::{self, CurveSupport, FaceLoop, OrientedEdgeUse, SurfaceSupport};
use crate::edge_treatments::{EdgeTreatmentEvidence, EdgeTreatmentVirtualEdge};
use crate::math3::{
    add, cross, distance, mul, normalize as normalize3, point_to_unit_line_distance, sub,
};
use crate::step_graph::entity_id;
use ruststep::ast::EntityInstance;
use std::collections::{BTreeMap, HashMap, HashSet};

const GEOM_TOL_MM: f64 = 1.0e-7;
const DIR_TOL: f64 = 1.0e-10;

/// Geometry/topology view used by inverse-feature reducers.
///
/// This is intentionally independent of any recovered CAD primitive. Reducers
/// simplify this representation; primitive recognizers are consumers of the
/// simplified result rather than owners of the simplification rules.
#[derive(Debug, Clone)]
pub(crate) struct ReducedFace {
    pub(crate) id: u64,
    pub(crate) surface: SurfaceSupport,
    pub(crate) surface_forward: bool,
    pub(crate) loops: Vec<FaceLoop>,
}

impl ReducedFace {
    fn from_step(
        face_id: u64,
        entities: &[EntityInstance],
        index: &HashMap<u64, usize>,
    ) -> Option<Self> {
        let surface_id = brep::face_surface(face_id, entities, index)?;
        let surface = brep::surface_support(surface_id, entities, index);
        let loops = brep::face_loops(face_id, entities, index)?;
        let surface_forward = match brep::face_sense(face_id, entities, index)?.as_str() {
            "T" => true,
            "F" => false,
            _ => return None,
        };
        Some(Self {
            id: face_id,
            surface,
            surface_forward,
            loops,
        })
    }

    pub(crate) fn loop_edges(&self) -> impl Iterator<Item = &OrientedEdgeUse> {
        self.loops.iter().flat_map(|loop_| loop_.edges.iter())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ReducedSolid {
    solid_id: u64,
    faces: BTreeMap<u64, ReducedFace>,
    next_virtual_id: u64,
}

#[derive(Debug, Clone, Copy)]
struct VirtualSegmentIds {
    start_vertex: u64,
    end_vertex: u64,
    curve: u64,
    edge: u64,
    oriented: [u64; 2],
}

#[derive(Debug, Clone, Copy)]
struct EndpointEdit {
    edge_id: u64,
    old_vertex: u64,
    new_vertex: u64,
    new_point: [f64; 3],
}

impl ReducedSolid {
    pub(crate) fn from_step(
        solid_id: u64,
        entities: &[EntityInstance],
        index: &HashMap<u64, usize>,
    ) -> Option<Self> {
        let face_ids = brep::solid_face_ids(solid_id, entities, index)?;
        let faces = face_ids
            .into_iter()
            .map(|face_id| Some((face_id, ReducedFace::from_step(face_id, entities, index)?)))
            .collect::<Option<BTreeMap<_, _>>>()?;
        let next_virtual_id = entities
            .iter()
            .map(entity_id)
            .max()
            .unwrap_or(0)
            .checked_add(1)?;
        Some(Self {
            solid_id,
            faces,
            next_virtual_id,
        })
    }

    pub(crate) fn solid_id(&self) -> u64 {
        self.solid_id
    }

    pub(crate) fn faces(&self) -> impl Iterator<Item = &ReducedFace> {
        self.faces.values()
    }

    /// Undo one proven edge treatment as a topology operation.
    ///
    /// The evidence determines *why* the treatment may be removed. The actual
    /// reduction is generic: delete the treatment face, extend both carrier
    /// boundaries to their inferred intersection, and make that virtual edge a
    /// shared topological edge of the carriers. The operation is transactional.
    pub(crate) fn remove_edge_treatment(&mut self, evidence: &EdgeTreatmentEvidence) -> Option<()> {
        if evidence.solid_id != self.solid_id || !evidence.carrier_pair_unique {
            return None;
        }
        if !self.faces.contains_key(&evidence.treatment_face_id)
            || evidence
                .carriers
                .iter()
                .any(|carrier| !self.faces.contains_key(&carrier.face_id))
        {
            return None;
        }

        let mut candidate = self.clone();
        match evidence.virtual_edge {
            EdgeTreatmentVirtualEdge::Segment { start_mm, end_mm } => {
                candidate.remove_segment_treatment_inner(evidence, start_mm, end_mm)?;
            }
            EdgeTreatmentVirtualEdge::Circle {
                center_mm,
                normal,
                radius_mm,
            } => {
                candidate.remove_circle_treatment_inner(evidence, center_mm, normal, radius_mm)?;
            }
        }
        candidate.validate_closed_topology()?;
        *self = candidate;
        Some(())
    }

    fn remove_segment_treatment_inner(
        &mut self,
        evidence: &EdgeTreatmentEvidence,
        start_mm: [f64; 3],
        end_mm: [f64; 3],
    ) -> Option<()> {
        let ids = self.allocate_virtual_segment_ids()?;
        let mut edits = Vec::new();
        for (carrier_index, carrier) in evidence.carriers.iter().enumerate() {
            let face = self.faces.get_mut(&carrier.face_id)?;
            edits.extend(splice_face_virtual_segment(
                face,
                &carrier.shared_edge_ids,
                start_mm,
                end_mm,
                ids,
                ids.oriented[carrier_index],
            )?);
        }
        self.propagate_endpoint_edits(&edits)?;
        self.faces.remove(&evidence.treatment_face_id)?;
        Some(())
    }

    fn remove_circle_treatment_inner(
        &mut self,
        evidence: &EdgeTreatmentEvidence,
        center_mm: [f64; 3],
        normal: [f64; 3],
        radius_mm: f64,
    ) -> Option<()> {
        if !radius_mm.is_finite() || radius_mm <= GEOM_TOL_MM {
            return None;
        }
        let normal = normalize3(normal, DIR_TOL)?;
        let x_direction = deterministic_perpendicular(normal)?;
        let ids = self.allocate_virtual_segment_ids()?;
        for (carrier_index, carrier) in evidence.carriers.iter().enumerate() {
            let face = self.faces.get_mut(&carrier.face_id)?;
            splice_face_virtual_circle(
                face,
                &carrier.shared_edge_ids,
                center_mm,
                normal,
                x_direction,
                radius_mm,
                ids,
                ids.oriented[carrier_index],
            )?;
        }
        self.faces.remove(&evidence.treatment_face_id)?;
        Some(())
    }

    fn allocate_virtual_segment_ids(&mut self) -> Option<VirtualSegmentIds> {
        let first = self.next_virtual_id;
        self.next_virtual_id = self.next_virtual_id.checked_add(6)?;
        Some(VirtualSegmentIds {
            start_vertex: first,
            end_vertex: first + 1,
            curve: first + 2,
            edge: first + 3,
            oriented: [first + 4, first + 5],
        })
    }

    fn propagate_endpoint_edits(&mut self, edits: &[EndpointEdit]) -> Option<()> {
        for edit in edits {
            let mut matched = false;
            for face in self.faces.values_mut() {
                for edge in face
                    .loops
                    .iter_mut()
                    .flat_map(|loop_| loop_.edges.iter_mut())
                {
                    if edge.edge_id != edit.edge_id {
                        continue;
                    }
                    if edge.start_vertex == edit.old_vertex {
                        edge.start_vertex = edit.new_vertex;
                        edge.start_mm = edit.new_point;
                        matched = true;
                    }
                    if edge.end_vertex == edit.old_vertex {
                        edge.end_vertex = edit.new_vertex;
                        edge.end_mm = edit.new_point;
                        matched = true;
                    }
                }
            }
            if !matched {
                return None;
            }
        }
        Some(())
    }

    fn validate_closed_topology(&self) -> Option<()> {
        let mut uses = HashMap::<u64, Vec<(u64, u64)>>::new();
        for face in self.faces.values() {
            for loop_ in &face.loops {
                if loop_.edges.is_empty() {
                    return None;
                }
                for (index, edge) in loop_.edges.iter().enumerate() {
                    let next = &loop_.edges[(index + 1) % loop_.edges.len()];
                    if edge.end_vertex != next.start_vertex
                        || distance(edge.end_mm, next.start_mm) > GEOM_TOL_MM
                    {
                        return None;
                    }
                    uses.entry(edge.edge_id)
                        .or_default()
                        .push((edge.start_vertex, edge.end_vertex));
                }
            }
        }
        uses.values()
            .all(|edge_uses| {
                edge_uses.len() == 2 && unordered_pair(edge_uses[0]) == unordered_pair(edge_uses[1])
            })
            .then_some(())
    }
}

fn deterministic_perpendicular(normal: [f64; 3]) -> Option<[f64; 3]> {
    let seed = if normal[0].abs() <= normal[1].abs() && normal[0].abs() <= normal[2].abs() {
        [1.0, 0.0, 0.0]
    } else if normal[1].abs() <= normal[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    normalize3(cross(normal, seed), DIR_TOL)
}

fn splice_face_virtual_circle(
    face: &mut ReducedFace,
    shared_edge_ids: &[u64],
    center_mm: [f64; 3],
    normal: [f64; 3],
    x_direction: [f64; 3],
    radius_mm: f64,
    ids: VirtualSegmentIds,
    oriented_edge_id: u64,
) -> Option<()> {
    if shared_edge_ids.is_empty() {
        return None;
    }
    let shared = shared_edge_ids.iter().copied().collect::<HashSet<_>>();
    if shared.len() != shared_edge_ids.len() {
        return None;
    }
    let matches = face
        .loops
        .iter()
        .enumerate()
        .filter_map(|(loop_index, loop_)| {
            let count = loop_
                .edges
                .iter()
                .filter(|edge| shared.contains(&edge.edge_id))
                .count();
            (count != 0).then_some((loop_index, count, loop_.edges.len()))
        })
        .collect::<Vec<_>>();
    let [(loop_index, count, edge_count)] = matches.as_slice() else {
        return None;
    };
    if *count != shared.len() || *count != *edge_count {
        return None;
    }

    let seam = add(center_mm, mul(x_direction, radius_mm));
    let source_forward = face.loops[*loop_index]
        .edges
        .first()
        .map_or(true, |edge| edge.parameter_forward);
    face.loops[*loop_index].edges = vec![OrientedEdgeUse {
        oriented_edge_id,
        edge_id: ids.edge,
        curve_id: ids.curve,
        curve_same_sense: true,
        parameter_forward: source_forward,
        start_vertex: ids.start_vertex,
        end_vertex: ids.start_vertex,
        start_mm: seam,
        end_mm: seam,
        support: CurveSupport::Circle(brep::CircleSupport {
            center_mm,
            normal,
            x_direction,
            radius_mm,
        }),
    }];
    Some(())
}

fn unordered_pair(pair: (u64, u64)) -> (u64, u64) {
    if pair.0 <= pair.1 {
        pair
    } else {
        (pair.1, pair.0)
    }
}

fn splice_face_virtual_segment(
    face: &mut ReducedFace,
    shared_edge_ids: &[u64],
    virtual_start: [f64; 3],
    virtual_end: [f64; 3],
    ids: VirtualSegmentIds,
    oriented_edge_id: u64,
) -> Option<Vec<EndpointEdit>> {
    if shared_edge_ids.is_empty() {
        return None;
    }
    let shared = shared_edge_ids.iter().copied().collect::<HashSet<_>>();
    if shared.len() != shared_edge_ids.len() {
        return None;
    }
    let mut matches = face
        .loops
        .iter()
        .enumerate()
        .filter_map(|(loop_index, loop_)| {
            let count = loop_
                .edges
                .iter()
                .filter(|edge| shared.contains(&edge.edge_id))
                .count();
            (count != 0).then_some((loop_index, count))
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 || matches[0].1 != shared.len() {
        return None;
    }
    let loop_index = matches.pop()?.0;
    let loop_ = &mut face.loops[loop_index];
    let edge_count = loop_.edges.len();
    if shared.len() >= edge_count.saturating_sub(1) {
        return None;
    }

    let starts = (0..edge_count)
        .filter(|&index| {
            shared.contains(&loop_.edges[index].edge_id)
                && !shared.contains(&loop_.edges[(index + edge_count - 1) % edge_count].edge_id)
        })
        .collect::<Vec<_>>();
    let [run_start] = starts.as_slice() else {
        return None;
    };
    let mut run_len = 0;
    while run_len < edge_count
        && shared.contains(&loop_.edges[(run_start + run_len) % edge_count].edge_id)
    {
        run_len += 1;
    }
    if run_len != shared.len() {
        return None;
    }
    let previous_index = (run_start + edge_count - 1) % edge_count;
    let next_index = (run_start + run_len) % edge_count;
    let previous = loop_.edges[previous_index].clone();
    let next = loop_.edges[next_index].clone();
    let previous_old_vertex = previous.end_vertex;
    let next_old_vertex = next.start_vertex;
    let previous_edge_id = previous.edge_id;
    let next_edge_id = next.edge_id;
    let CurveSupport::Line(previous_line) = previous.support else {
        return None;
    };
    let CurveSupport::Line(next_line) = next.support else {
        return None;
    };
    let previous_direction = normalize3(previous_line.direction, DIR_TOL)?;
    let next_direction = normalize3(next_line.direction, DIR_TOL)?;
    let (healed_start, healed_end) = [(virtual_start, virtual_end), (virtual_end, virtual_start)]
        .into_iter()
        .find(|(previous_point, next_point)| {
            point_to_unit_line_distance(
                *previous_point,
                previous_line.origin_mm,
                previous_direction,
            ) <= GEOM_TOL_MM
                && point_to_unit_line_distance(*next_point, next_line.origin_mm, next_direction)
                    <= GEOM_TOL_MM
        })?;

    let forward = distance(healed_start, virtual_start) <= GEOM_TOL_MM
        && distance(healed_end, virtual_end) <= GEOM_TOL_MM;
    let (start_vertex, end_vertex) = if forward {
        (ids.start_vertex, ids.end_vertex)
    } else {
        (ids.end_vertex, ids.start_vertex)
    };
    let direction = normalize3(sub(virtual_end, virtual_start), DIR_TOL)?;
    let replacement = OrientedEdgeUse {
        oriented_edge_id,
        edge_id: ids.edge,
        curve_id: ids.curve,
        curve_same_sense: true,
        parameter_forward: forward,
        start_vertex,
        end_vertex,
        start_mm: healed_start,
        end_mm: healed_end,
        support: CurveSupport::Line(brep::LineSupport {
            origin_mm: virtual_start,
            direction,
        }),
    };

    let mut rebuilt = Vec::with_capacity(edge_count - run_len + 1);
    let mut previous = previous;
    previous.end_vertex = start_vertex;
    previous.end_mm = healed_start;
    rebuilt.push(previous);
    rebuilt.push(replacement);
    let mut index = next_index;
    while index != previous_index {
        let mut edge = loop_.edges[index].clone();
        if index == next_index {
            edge.start_vertex = end_vertex;
            edge.start_mm = healed_end;
        }
        rebuilt.push(edge);
        index = (index + 1) % edge_count;
    }
    if rebuilt.iter().enumerate().any(|(index, edge)| {
        let next = &rebuilt[(index + 1) % rebuilt.len()];
        edge.end_vertex != next.start_vertex || distance(edge.end_mm, next.start_mm) > GEOM_TOL_MM
    }) {
        return None;
    }
    loop_.edges = rebuilt;

    Some(vec![
        EndpointEdit {
            edge_id: previous_edge_id,
            old_vertex: previous_old_vertex,
            new_vertex: start_vertex,
            new_point: healed_start,
        },
        EndpointEdit {
            edge_id: next_edge_id,
            old_vertex: next_old_vertex,
            new_vertex: end_vertex,
            new_point: healed_end,
        },
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brep::{LineSupport, PlaneSupport};

    fn line_edge(
        id: u64,
        start_vertex: u64,
        end_vertex: u64,
        start: [f64; 3],
        end: [f64; 3],
    ) -> OrientedEdgeUse {
        OrientedEdgeUse {
            oriented_edge_id: id + 100,
            edge_id: id,
            curve_id: id + 200,
            curve_same_sense: true,
            parameter_forward: true,
            start_vertex,
            end_vertex,
            start_mm: start,
            end_mm: end,
            support: CurveSupport::Line(LineSupport {
                origin_mm: start,
                direction: sub(end, start),
            }),
        }
    }

    fn trimmed_face() -> ReducedFace {
        let edges = vec![
            line_edge(10, 1, 2, [0.0, 0.25, 0.0], [0.4, 0.25, 0.0]),
            line_edge(11, 2, 3, [0.4, 0.25, 0.0], [0.6, 0.25, 0.0]),
            line_edge(12, 3, 4, [0.6, 0.25, 0.0], [1.0, 0.25, 0.0]),
            line_edge(13, 4, 5, [1.0, 0.25, 0.0], [1.0, 1.0, 0.0]),
            line_edge(14, 5, 6, [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
            line_edge(15, 6, 1, [0.0, 1.0, 0.0], [0.0, 0.25, 0.0]),
        ];
        ReducedFace {
            id: 1,
            surface: SurfaceSupport::Plane(PlaneSupport {
                origin_mm: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                max_residual_mm: 0.0,
            }),
            surface_forward: true,
            loops: vec![FaceLoop {
                bound_id: 20,
                loop_id: 21,
                outer: true,
                orientation: true,
                edges,
            }],
        }
    }

    fn virtual_ids() -> VirtualSegmentIds {
        VirtualSegmentIds {
            start_vertex: 1000,
            end_vertex: 1001,
            curve: 1002,
            edge: 1003,
            oriented: [1004, 1005],
        }
    }

    fn circular_carrier_face(face_id: u64, edge_id: u64, vertex_id: u64) -> ReducedFace {
        let seam = [1.0, 0.0, 0.0];
        ReducedFace {
            id: face_id,
            surface: SurfaceSupport::Plane(PlaneSupport {
                origin_mm: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                max_residual_mm: 0.0,
            }),
            surface_forward: true,
            loops: vec![FaceLoop {
                bound_id: face_id + 100,
                loop_id: face_id + 200,
                outer: true,
                orientation: true,
                edges: vec![OrientedEdgeUse {
                    oriented_edge_id: edge_id + 1000,
                    edge_id,
                    curve_id: edge_id + 2000,
                    curve_same_sense: true,
                    parameter_forward: true,
                    start_vertex: vertex_id,
                    end_vertex: vertex_id,
                    start_mm: seam,
                    end_mm: seam,
                    support: CurveSupport::Circle(brep::CircleSupport {
                        center_mm: [0.0, 0.0, 0.0],
                        normal: [0.0, 0.0, 1.0],
                        x_direction: [1.0, 0.0, 0.0],
                        radius_mm: 1.0,
                    }),
                }],
            }],
        }
    }

    #[test]
    fn virtual_segment_splice_collapses_contiguous_edge_run() -> anyhow::Result<()> {
        let mut face = trimmed_face();
        let edits = splice_face_virtual_segment(
            &mut face,
            &[10, 11, 12],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            virtual_ids(),
            1004,
        )
        .ok_or_else(|| anyhow::anyhow!("contiguous treatment boundary should reduce"))?;

        let edges = &face.loops[0].edges;
        assert_eq!(edges.len(), 4);
        assert_eq!(edges[1].edge_id, 1003);
        assert_eq!(edges[1].start_mm, [0.0, 0.0, 0.0]);
        assert_eq!(edges[1].end_mm, [1.0, 0.0, 0.0]);
        assert_eq!(edges[0].end_mm, [0.0, 0.0, 0.0]);
        assert_eq!(edges[2].start_mm, [1.0, 0.0, 0.0]);
        assert_eq!(edits.len(), 2);
        Ok(())
    }

    #[test]
    fn circular_treatment_reduction_reconnects_carriers_with_one_shared_edge() -> anyhow::Result<()>
    {
        let first = circular_carrier_face(1, 10, 1);
        let second = circular_carrier_face(2, 20, 2);
        let treatment = ReducedFace {
            id: 99,
            surface: SurfaceSupport::Other { entity_id: 9999 },
            surface_forward: true,
            loops: Vec::new(),
        };
        let mut solid = ReducedSolid {
            solid_id: 100,
            faces: BTreeMap::from([(1, first), (2, second), (99, treatment)]),
            next_virtual_id: 1000,
        };
        let evidence = EdgeTreatmentEvidence {
            solid_id: 100,
            treatment_face_id: 99,
            kind: crate::edge_treatments::EdgeTreatmentKind::ToroidalFillet,
            strength: crate::edge_treatments::EdgeTreatmentEvidenceStrength::AnalyticTangency,
            carrier_pair_unique: true,
            treatment_support: "torus".to_owned(),
            carriers: [
                crate::edge_treatments::EdgeTreatmentCarrier {
                    face_id: 1,
                    support: "plane".to_owned(),
                    shared_edge_ids: vec![10],
                },
                crate::edge_treatments::EdgeTreatmentCarrier {
                    face_id: 2,
                    support: "cylinder".to_owned(),
                    shared_edge_ids: vec![20],
                },
            ],
            inferred_radius_mm: Some(0.1),
            inferred_setbacks_mm: None,
            inferred_angle_rad: None,
            carrier_angle_rad: std::f64::consts::FRAC_PI_2,
            virtual_edge: EdgeTreatmentVirtualEdge::Circle {
                center_mm: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                radius_mm: 1.0,
            },
            max_boundary_residual_mm: 0.0,
        };

        solid
            .remove_edge_treatment(&evidence)
            .ok_or_else(|| anyhow::anyhow!("circular inverse treatment should reduce"))?;
        assert!(!solid.faces.contains_key(&99));
        let first_edge = &solid.faces[&1].loops[0].edges[0];
        let second_edge = &solid.faces[&2].loops[0].edges[0];
        assert_eq!(first_edge.edge_id, second_edge.edge_id);
        assert_eq!(first_edge.start_vertex, first_edge.end_vertex);
        assert_eq!(first_edge.start_vertex, second_edge.start_vertex);
        assert!(matches!(first_edge.support, CurveSupport::Circle(_)));
        Ok(())
    }

    #[test]
    fn virtual_segment_splice_rejects_disconnected_shared_edges() {
        let mut face = trimmed_face();
        assert!(
            splice_face_virtual_segment(
                &mut face,
                &[10, 12],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                virtual_ids(),
                1004,
            )
            .is_none()
        );
    }
}
