//! Pure normalization of proven STEP edge-treatment evidence into kernel topology patches.
//!
//! This module is deliberately read-only with respect to [`ReducedSolid`].  It uses the
//! parser-side B-rep only to understand adjacency and bounded source geometry, then emits a
//! backend-neutral patch addressed entirely by the kernel `StableId` journal.  Mutation belongs to
//! [`crate::kernel_topology::SourceTopologyEditor`].

use crate::brep::{CurveSupport, OrientedEdgeUse};
use crate::edge_treatments::{EdgeTreatmentEvidence, EdgeTreatmentVirtualEdge};
use crate::kernel_source_load::{KernelTopologyId, KernelTopologyProvenance};
use crate::kernel_topology::{
    KernelFaceSplice, KernelNewEdge, KernelPatchCurve, KernelPatchEdgeRef, KernelPatchEdgeUse,
    KernelPatchVertexRef, KernelRetainedEdge, KernelTopologyPatch,
};
use crate::math3::{cross, distance, dot, normalize, point_to_unit_line_distance, sub};
use crate::reduced_brep::{ReducedFace, ReducedSolid};
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, BTreeSet};

const GEOM_TOL_MM: f64 = 1.0e-7;
const DIR_TOL: f64 = 1.0e-10;

#[derive(Debug, Clone)]
struct EndpointEdit {
    carrier_face_id: u64,
    edge: OrientedEdgeUse,
    old_vertex: u64,
    new_vertex: usize,
    new_point: [f64; 3],
}

#[derive(Debug)]
struct SegmentCarrierPlan {
    splice: KernelFaceSplice,
    endpoint_edits: [EndpointEdit; 2],
}

/// Normalize one proven inverse edge treatment without mutating parser-side topology.
pub fn plan_edge_treatment_patch(
    solid: &ReducedSolid,
    evidence: &EdgeTreatmentEvidence,
    data_section: usize,
    provenance: &KernelTopologyProvenance,
) -> Result<KernelTopologyPatch> {
    ensure!(
        evidence.solid_id == solid.solid_id(),
        "edge-treatment evidence belongs to solid #{}, not #{}",
        evidence.solid_id,
        solid.solid_id()
    );
    ensure!(
        evidence.carrier_pair_unique,
        "edge-treatment face #{} has no unique carrier pair",
        evidence.treatment_face_id
    );

    match evidence.virtual_edge {
        EdgeTreatmentVirtualEdge::Segment { start_mm, end_mm } => {
            plan_segment_patch(solid, evidence, data_section, provenance, start_mm, end_mm)
        }
        EdgeTreatmentVirtualEdge::Circle {
            center_mm,
            normal,
            radius_mm,
        } => plan_circle_patch(
            solid,
            evidence,
            data_section,
            provenance,
            center_mm,
            normal,
            radius_mm,
        ),
    }
}

fn plan_segment_patch(
    solid: &ReducedSolid,
    evidence: &EdgeTreatmentEvidence,
    data_section: usize,
    provenance: &KernelTopologyProvenance,
    virtual_start: [f64; 3],
    virtual_end: [f64; 3],
) -> Result<KernelTopologyPatch> {
    ensure!(
        distance(virtual_start, virtual_end) > GEOM_TOL_MM,
        "inverse segment for treatment face #{} is degenerate",
        evidence.treatment_face_id
    );
    let treatment = face(solid, evidence.treatment_face_id)?;
    let treatment_edges = unique_face_edges(treatment)?;
    let treatment_face = topology_face(provenance, data_section, evidence.treatment_face_id)?;
    let (carrier_plans, interface_edges) = collect_segment_carrier_plans(
        solid,
        evidence,
        data_section,
        provenance,
        virtual_start,
        virtual_end,
    )?;
    ensure!(
        interface_edges.is_subset(&treatment_edges),
        "carrier interface contains an edge absent from treatment face #{}",
        evidence.treatment_face_id
    );
    let endpoint_edits = carrier_plans
        .iter()
        .flat_map(|plan| plan.endpoint_edits.iter().cloned())
        .collect::<Vec<_>>();
    let rewrite_edges =
        normalize_retained_edge_rewrites(&endpoint_edits, data_section, provenance)?;
    let terminal_edges = treatment_edges
        .difference(&interface_edges)
        .copied()
        .collect::<BTreeSet<_>>();
    ensure!(
        !terminal_edges.is_empty(),
        "segment treatment face #{} has no terminal boundary",
        evidence.treatment_face_id
    );
    let mut face_splices = carrier_plans
        .into_iter()
        .map(|plan| plan.splice)
        .collect::<Vec<_>>();
    face_splices.extend(plan_terminal_splices(
        solid,
        evidence,
        data_section,
        provenance,
        &terminal_edges,
    )?);

    Ok(KernelTopologyPatch {
        remove_faces: vec![treatment_face],
        new_vertices: vec![virtual_start, virtual_end],
        new_edges: vec![KernelNewEdge {
            vertices: (KernelPatchVertexRef::New(0), KernelPatchVertexRef::New(1)),
            curve: KernelPatchCurve::Line {
                start_mm: virtual_start,
                end_mm: virtual_end,
            },
        }],
        rewrite_edges,
        face_splices,
        require_unused_edges: topology_edges(
            provenance,
            data_section,
            treatment_edges.iter().copied(),
        )?,
    })
}

fn collect_segment_carrier_plans(
    solid: &ReducedSolid,
    evidence: &EdgeTreatmentEvidence,
    data_section: usize,
    provenance: &KernelTopologyProvenance,
    virtual_start: [f64; 3],
    virtual_end: [f64; 3],
) -> Result<(Vec<SegmentCarrierPlan>, BTreeSet<u64>)> {
    let mut plans = Vec::with_capacity(2);
    let mut interface_edges = BTreeSet::new();
    for carrier in &evidence.carriers {
        for &edge_id in &carrier.shared_edge_ids {
            ensure!(
                interface_edges.insert(edge_id),
                "treatment face #{} assigns source edge #{edge_id} to both carriers",
                evidence.treatment_face_id
            );
        }
        plans.push(plan_segment_carrier(
            face(solid, carrier.face_id)?,
            &carrier.shared_edge_ids,
            data_section,
            provenance,
            virtual_start,
            virtual_end,
        )?);
    }
    Ok((plans, interface_edges))
}

fn plan_terminal_splices(
    solid: &ReducedSolid,
    evidence: &EdgeTreatmentEvidence,
    data_section: usize,
    provenance: &KernelTopologyProvenance,
    terminal_edges: &BTreeSet<u64>,
) -> Result<Vec<KernelFaceSplice>> {
    let mut terminal_by_face = BTreeMap::<u64, BTreeSet<u64>>::new();
    for &terminal_edge in terminal_edges {
        let users = edge_face_users(solid, terminal_edge);
        ensure!(
            users.len() == 2 && users.contains(&evidence.treatment_face_id),
            "terminal treatment edge #{terminal_edge} is not manifold around face #{}",
            evidence.treatment_face_id
        );
        let terminal_face = users
            .iter()
            .copied()
            .find(|&face_id| face_id != evidence.treatment_face_id)
            .context("terminal treatment edge has no surviving face")?;
        ensure!(
            !evidence
                .carriers
                .iter()
                .any(|carrier| carrier.face_id == terminal_face),
            "terminal treatment edge #{terminal_edge} folds back into a carrier face"
        );
        terminal_by_face
            .entry(terminal_face)
            .or_default()
            .insert(terminal_edge);
    }
    terminal_by_face
        .into_iter()
        .map(|(face_id, edges)| {
            let terminal_face = face(solid, face_id)?;
            ensure!(
                edges_form_one_cyclic_run(terminal_face, &edges),
                "terminal edges {edges:?} are not one contiguous run on face #{face_id}"
            );
            Ok(KernelFaceSplice {
                face: topology_face(provenance, data_section, face_id)?,
                remove_edges: topology_edges(provenance, data_section, edges.iter().copied())?,
                replacement: Vec::new(),
            })
        })
        .collect()
}

fn plan_segment_carrier(
    face: &ReducedFace,
    shared_edge_ids: &[u64],
    data_section: usize,
    provenance: &KernelTopologyProvenance,
    virtual_start: [f64; 3],
    virtual_end: [f64; 3],
) -> Result<SegmentCarrierPlan> {
    ensure!(
        !shared_edge_ids.is_empty(),
        "carrier has no shared treatment edges"
    );
    let shared = shared_edge_ids.iter().copied().collect::<BTreeSet<_>>();
    ensure!(
        shared.len() == shared_edge_ids.len(),
        "carrier face #{} repeats a shared treatment edge",
        face.id
    );
    let (previous, next) = carrier_run_neighbors(face, &shared)?;
    let CurveSupport::Line(previous_line) = previous.support else {
        bail!(
            "carrier face #{} previous retained edge is not linear",
            face.id
        );
    };
    let CurveSupport::Line(next_line) = next.support else {
        bail!("carrier face #{} next retained edge is not linear", face.id);
    };
    let previous_direction = normalize(previous_line.direction, DIR_TOL)
        .context("previous retained edge direction is degenerate")?;
    let next_direction = normalize(next_line.direction, DIR_TOL)
        .context("next retained edge direction is degenerate")?;
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
        })
        .with_context(|| {
            format!(
                "proven inverse segment does not meet retained neighbors on carrier face #{}",
                face.id
            )
        })?;
    let forward = distance(healed_start, virtual_start) <= GEOM_TOL_MM;
    let (start_vertex, end_vertex) = if forward { (0, 1) } else { (1, 0) };

    Ok(SegmentCarrierPlan {
        splice: KernelFaceSplice {
            face: topology_face(provenance, data_section, face.id)?,
            remove_edges: topology_edges(provenance, data_section, shared.iter().copied())?,
            replacement: vec![KernelPatchEdgeUse {
                edge: KernelPatchEdgeRef::New(0),
                orientation: forward,
            }],
        },
        endpoint_edits: [
            EndpointEdit {
                carrier_face_id: face.id,
                old_vertex: previous.end_vertex,
                edge: previous,
                new_vertex: start_vertex,
                new_point: healed_start,
            },
            EndpointEdit {
                carrier_face_id: face.id,
                old_vertex: next.start_vertex,
                edge: next,
                new_vertex: end_vertex,
                new_point: healed_end,
            },
        ],
    })
}

fn carrier_run_neighbors(
    face: &ReducedFace,
    shared: &BTreeSet<u64>,
) -> Result<(OrientedEdgeUse, OrientedEdgeUse)> {
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
            (count != 0).then_some((loop_index, count))
        })
        .collect::<Vec<_>>();
    let [(loop_index, count)] = matches.as_slice() else {
        bail!(
            "carrier face #{} does not have one treatment boundary run",
            face.id
        );
    };
    ensure!(
        *count == shared.len(),
        "carrier face #{} does not contain every shared treatment edge",
        face.id
    );
    let loop_ = &face.loops[*loop_index];
    let edge_count = loop_.edges.len();
    ensure!(
        shared.len() < edge_count.saturating_sub(1),
        "carrier face #{} leaves no retained neighbors around treatment run",
        face.id
    );
    let starts = (0..edge_count)
        .filter(|&index| {
            shared.contains(&loop_.edges[index].edge_id)
                && !shared.contains(&loop_.edges[(index + edge_count - 1) % edge_count].edge_id)
        })
        .collect::<Vec<_>>();
    let [run_start] = starts.as_slice() else {
        bail!(
            "carrier face #{} treatment edges are not one cyclic run",
            face.id
        );
    };
    let mut run_len = 0usize;
    while run_len < edge_count
        && shared.contains(&loop_.edges[(run_start + run_len) % edge_count].edge_id)
    {
        run_len += 1;
    }
    ensure!(
        run_len == shared.len(),
        "carrier face #{} treatment run is disconnected",
        face.id
    );
    Ok((
        loop_.edges[(run_start + edge_count - 1) % edge_count].clone(),
        loop_.edges[(run_start + run_len) % edge_count].clone(),
    ))
}

fn normalize_retained_edge_rewrites(
    edits: &[EndpointEdit],
    data_section: usize,
    provenance: &KernelTopologyProvenance,
) -> Result<Vec<KernelRetainedEdge>> {
    let mut by_edge = BTreeMap::<u64, Vec<&EndpointEdit>>::new();
    for edit in edits {
        by_edge.entry(edit.edge.edge_id).or_default().push(edit);
    }
    let mut out = Vec::with_capacity(by_edge.len());
    for (edge_id, edits) in by_edge {
        let representative = edits
            .first()
            .context("retained edge rewrite lost its representative use")?;
        let (vertices, mut points) = global_edge_endpoints(&representative.edge);
        let CurveSupport::Line(_) = representative.edge.support else {
            bail!("retained edge #{edge_id} is not linear");
        };
        let mut changed = [false; 2];
        let mut new_refs = [None, None];
        for edit in edits {
            let endpoint = if vertices[0] == edit.old_vertex {
                0
            } else if vertices[1] == edit.old_vertex {
                1
            } else {
                bail!(
                    "retained edge #{edge_id} on carrier face #{} does not contain edited vertex #{}",
                    edit.carrier_face_id,
                    edit.old_vertex
                );
            };
            if changed[endpoint] {
                ensure!(
                    new_refs[endpoint] == Some(edit.new_vertex)
                        && distance(points[endpoint], edit.new_point) <= GEOM_TOL_MM,
                    "retained edge #{edge_id} receives conflicting endpoint edits"
                );
                continue;
            }
            changed[endpoint] = true;
            new_refs[endpoint] = Some(edit.new_vertex);
            points[endpoint] = edit.new_point;
        }
        let refs = [0usize, 1usize].map(|endpoint| {
            new_refs[endpoint].map_or_else(
                || {
                    topology_vertex(provenance, data_section, vertices[endpoint])
                        .map(KernelPatchVertexRef::Existing)
                },
                |new_index| Ok(KernelPatchVertexRef::New(new_index)),
            )
        });
        let [front, back] = refs;
        ensure!(
            distance(points[0], points[1]) > GEOM_TOL_MM,
            "retained edge #{edge_id} collapses after inverse treatment"
        );
        out.push(KernelRetainedEdge {
            edge: topology_edge(provenance, data_section, edge_id)?,
            vertices: (front?, back?),
            curve: KernelPatchCurve::Line {
                start_mm: points[0],
                end_mm: points[1],
            },
        });
    }
    Ok(out)
}

const fn global_edge_endpoints(edge: &OrientedEdgeUse) -> ([u64; 2], [[f64; 3]; 2]) {
    if edge.parameter_forward == edge.curve_same_sense {
        (
            [edge.start_vertex, edge.end_vertex],
            [edge.start_mm, edge.end_mm],
        )
    } else {
        (
            [edge.end_vertex, edge.start_vertex],
            [edge.end_mm, edge.start_mm],
        )
    }
}

fn plan_circle_patch(
    solid: &ReducedSolid,
    evidence: &EdgeTreatmentEvidence,
    data_section: usize,
    provenance: &KernelTopologyProvenance,
    center_mm: [f64; 3],
    normal: [f64; 3],
    radius_mm: f64,
) -> Result<KernelTopologyPatch> {
    ensure!(
        radius_mm.is_finite() && radius_mm > GEOM_TOL_MM,
        "inverse circle for treatment face #{} has invalid radius",
        evidence.treatment_face_id
    );
    let normal = normalize(normal, DIR_TOL).context("inverse circle normal is degenerate")?;
    let x_direction = deterministic_perpendicular(normal)
        .context("inverse circle cannot construct a deterministic radial axis")?;
    let treatment_edges = unique_face_edges(face(solid, evidence.treatment_face_id)?)?;
    let (face_splices, interface_edges) =
        plan_circle_carriers(solid, evidence, data_section, provenance, center_mm, normal)?;
    ensure!(
        interface_edges == treatment_edges,
        "circular treatment face #{} has terminal edges; partial-circle surgery is not normalized yet",
        evidence.treatment_face_id
    );
    let (new_vertices, new_edges) =
        circle_patch_geometry(center_mm, normal, x_direction, radius_mm);

    Ok(KernelTopologyPatch {
        remove_faces: vec![topology_face(
            provenance,
            data_section,
            evidence.treatment_face_id,
        )?],
        new_vertices,
        new_edges,
        rewrite_edges: Vec::new(),
        face_splices,
        require_unused_edges: topology_edges(
            provenance,
            data_section,
            treatment_edges.iter().copied(),
        )?,
    })
}

fn plan_circle_carriers(
    solid: &ReducedSolid,
    evidence: &EdgeTreatmentEvidence,
    data_section: usize,
    provenance: &KernelTopologyProvenance,
    center_mm: [f64; 3],
    normal: [f64; 3],
) -> Result<(Vec<KernelFaceSplice>, BTreeSet<u64>)> {
    let mut interface_edges = BTreeSet::new();
    let mut face_splices = Vec::with_capacity(2);
    for carrier in &evidence.carriers {
        let carrier_face = face(solid, carrier.face_id)?;
        let shared = carrier
            .shared_edge_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        ensure!(
            !shared.is_empty() && shared.len() == carrier.shared_edge_ids.len(),
            "circular carrier face #{} has invalid shared-edge set",
            carrier.face_id
        );
        for &edge_id in &shared {
            ensure!(
                interface_edges.insert(edge_id),
                "circular treatment face #{} assigns edge #{edge_id} to both carriers",
                evidence.treatment_face_id
            );
        }
        let loop_ = complete_carrier_loop(carrier_face, &shared)?;
        let forward = circular_loop_forward(loop_, center_mm, normal)?;
        face_splices.push(KernelFaceSplice {
            face: topology_face(provenance, data_section, carrier.face_id)?,
            remove_edges: topology_edges(provenance, data_section, shared.iter().copied())?,
            replacement: circle_replacement_uses(forward),
        });
    }
    Ok((face_splices, interface_edges))
}

fn complete_carrier_loop<'a>(
    face: &'a ReducedFace,
    shared: &BTreeSet<u64>,
) -> Result<&'a crate::brep::FaceLoop> {
    let matching = face
        .loops
        .iter()
        .filter(|loop_| {
            loop_.edges.len() == shared.len()
                && loop_
                    .edges
                    .iter()
                    .map(|edge| edge.edge_id)
                    .collect::<BTreeSet<_>>()
                    == *shared
        })
        .collect::<Vec<_>>();
    let [loop_] = matching.as_slice() else {
        bail!(
            "circular carrier face #{} does not expose the treatment as one complete loop",
            face.id
        );
    };
    Ok(loop_)
}

fn circle_replacement_uses(forward: bool) -> Vec<KernelPatchEdgeUse> {
    if forward {
        (0..4)
            .map(|edge| KernelPatchEdgeUse {
                edge: KernelPatchEdgeRef::New(edge),
                orientation: true,
            })
            .collect()
    } else {
        (0..4)
            .rev()
            .map(|edge| KernelPatchEdgeUse {
                edge: KernelPatchEdgeRef::New(edge),
                orientation: false,
            })
            .collect()
    }
}

fn circle_patch_geometry(
    center_mm: [f64; 3],
    normal: [f64; 3],
    x_direction: [f64; 3],
    radius_mm: f64,
) -> (Vec<[f64; 3]>, Vec<KernelNewEdge>) {
    let y_direction = cross(normal, x_direction);
    let quarter = std::f64::consts::FRAC_PI_2;
    let angles = [0.0, quarter, 2.0 * quarter, 3.0 * quarter];
    let vertices = angles
        .map(|angle| {
            let radial = [
                x_direction[0].mul_add(angle.cos(), y_direction[0] * angle.sin()),
                x_direction[1].mul_add(angle.cos(), y_direction[1] * angle.sin()),
                x_direction[2].mul_add(angle.cos(), y_direction[2] * angle.sin()),
            ];
            [
                radial[0].mul_add(radius_mm, center_mm[0]),
                radial[1].mul_add(radius_mm, center_mm[1]),
                radial[2].mul_add(radius_mm, center_mm[2]),
            ]
        })
        .to_vec();
    let ranges = [
        (0.0, quarter),
        (quarter, 2.0 * quarter),
        (2.0 * quarter, 3.0 * quarter),
        (3.0 * quarter, 4.0 * quarter),
    ];
    let edges = ranges
        .into_iter()
        .enumerate()
        .map(|(edge, (start_angle_rad, end_angle_rad))| KernelNewEdge {
            vertices: (
                KernelPatchVertexRef::New(edge),
                KernelPatchVertexRef::New((edge + 1) % 4),
            ),
            curve: KernelPatchCurve::CircleArc {
                center_mm,
                normal,
                x_direction,
                radius_mm,
                start_angle_rad,
                end_angle_rad,
            },
        })
        .collect();
    (vertices, edges)
}

fn circular_loop_forward(
    loop_: &crate::brep::FaceLoop,
    center_mm: [f64; 3],
    normal: [f64; 3],
) -> Result<bool> {
    for edge in &loop_.edges {
        let start = sub(edge.start_mm, center_mm);
        let end = sub(edge.end_mm, center_mm);
        let turn = dot(normal, cross(start, end));
        if turn.abs() > GEOM_TOL_MM * GEOM_TOL_MM {
            return Ok(turn > 0.0);
        }
        if let CurveSupport::Circle(circle) = edge.support {
            let alignment = dot(circle.normal, normal);
            if alignment.abs() > 1.0 - DIR_TOL {
                let parameter_forward = if edge.parameter_forward { 1.0 } else { -1.0 };
                return Ok(parameter_forward * alignment > 0.0);
            }
        }
    }
    bail!("circular carrier loop has no resolvable traversal direction")
}

fn deterministic_perpendicular(normal: [f64; 3]) -> Option<[f64; 3]> {
    let seed = if normal[0].abs() <= normal[1].abs() && normal[0].abs() <= normal[2].abs() {
        [1.0, 0.0, 0.0]
    } else if normal[1].abs() <= normal[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    normalize(cross(normal, seed), DIR_TOL)
}

fn face(solid: &ReducedSolid, face_id: u64) -> Result<&ReducedFace> {
    solid
        .faces()
        .find(|face| face.id == face_id)
        .with_context(|| format!("reduced solid #{} has no face #{face_id}", solid.solid_id()))
}

fn unique_face_edges(face: &ReducedFace) -> Result<BTreeSet<u64>> {
    let mut edges = BTreeSet::new();
    for edge in face.loop_edges() {
        ensure!(
            edges.insert(edge.edge_id),
            "face #{} uses source edge #{} more than once",
            face.id,
            edge.edge_id
        );
    }
    Ok(edges)
}

fn edge_face_users(solid: &ReducedSolid, edge_id: u64) -> BTreeSet<u64> {
    solid
        .faces()
        .filter(|face| face.loop_edges().any(|edge| edge.edge_id == edge_id))
        .map(|face| face.id)
        .collect()
}

fn edges_form_one_cyclic_run(face: &ReducedFace, edges: &BTreeSet<u64>) -> bool {
    let matches = face
        .loops
        .iter()
        .filter_map(|loop_| {
            let count = loop_
                .edges
                .iter()
                .filter(|edge| edges.contains(&edge.edge_id))
                .count();
            (count != 0).then_some((loop_, count))
        })
        .collect::<Vec<_>>();
    let [(loop_, count)] = matches.as_slice() else {
        return false;
    };
    if *count != edges.len() || edges.len() >= loop_.edges.len() {
        return false;
    }
    let edge_count = loop_.edges.len();
    let starts = (0..edge_count)
        .filter(|&index| {
            edges.contains(&loop_.edges[index].edge_id)
                && !edges.contains(&loop_.edges[(index + edge_count - 1) % edge_count].edge_id)
        })
        .count();
    starts == 1
}

fn topology_vertex(
    provenance: &KernelTopologyProvenance,
    data_section: usize,
    source_id: u64,
) -> Result<KernelTopologyId> {
    provenance
        .vertex_for_source_id(data_section, source_id)
        .with_context(|| format!("source vertex #{source_id} has no unique kernel StableId"))
}

fn topology_edge(
    provenance: &KernelTopologyProvenance,
    data_section: usize,
    source_id: u64,
) -> Result<KernelTopologyId> {
    provenance
        .edge_for_source_id(data_section, source_id)
        .with_context(|| format!("source edge #{source_id} has no unique kernel StableId"))
}

fn topology_edges(
    provenance: &KernelTopologyProvenance,
    data_section: usize,
    source_ids: impl IntoIterator<Item = u64>,
) -> Result<Vec<KernelTopologyId>> {
    source_ids
        .into_iter()
        .map(|source_id| topology_edge(provenance, data_section, source_id))
        .collect()
}

fn topology_face(
    provenance: &KernelTopologyProvenance,
    data_section: usize,
    source_id: u64,
) -> Result<KernelTopologyId> {
    provenance
        .face_for_source_id(data_section, source_id)
        .with_context(|| format!("source face #{source_id} has no unique kernel StableId"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SourceEntity;
    use crate::brep::{FaceLoop, LineSupport, PlaneSupport, SurfaceSupport};
    use crate::edge_treatments::{
        EdgeTreatmentCarrier, EdgeTreatmentEvidenceStrength, EdgeTreatmentKind,
    };
    use crate::kernel_source_load::{KernelFaceTopologyProvenance, KernelTopologyEntityProvenance};

    const DATA_SECTION: usize = 0;
    const SOLID_ID: u64 = 500;

    fn source(id: u64) -> SourceEntity {
        SourceEntity {
            data_section: DATA_SECTION,
            ordinal: usize::try_from(id).unwrap_or(0),
            id,
        }
    }

    fn kid(raw: u64) -> KernelTopologyId {
        KernelTopologyId::from_raw(raw)
    }

    fn edge(
        id: u64,
        vertices: (u64, u64),
        points: ([f64; 3], [f64; 3]),
        forward: bool,
    ) -> OrientedEdgeUse {
        let (start_vertex, end_vertex, start_mm, end_mm) = if forward {
            (vertices.0, vertices.1, points.0, points.1)
        } else {
            (vertices.1, vertices.0, points.1, points.0)
        };
        OrientedEdgeUse {
            oriented_edge_id: id + 10_000,
            edge_id: id,
            curve_id: id + 20_000,
            curve_same_sense: true,
            parameter_forward: forward,
            start_vertex,
            end_vertex,
            start_mm,
            end_mm,
            support: CurveSupport::Line(LineSupport {
                origin_mm: points.0,
                direction: sub(points.1, points.0),
            }),
        }
    }

    fn plane_face(id: u64, edges: Vec<OrientedEdgeUse>) -> ReducedFace {
        ReducedFace {
            id,
            surface: SurfaceSupport::Plane(PlaneSupport {
                origin_mm: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                max_residual_mm: 0.0,
            }),
            surface_forward: true,
            loops: vec![FaceLoop {
                bound_id: id + 30_000,
                loop_id: id + 40_000,
                outer: true,
                orientation: true,
                edges,
            }],
        }
    }

    const CHAMFER_POINTS: [[f64; 3]; 10] = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.5, 0.0],
        [1.5, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [2.0, 1.5, 1.0],
        [1.5, 2.0, 1.0],
        [0.0, 2.0, 1.0],
    ];
    const CHAMFER_VERTEX_IDS: [u64; 10] = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110];
    const CHAMFER_EDGE_IDS: [u64; 15] = [
        201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
    ];
    const CHAMFER_FACE_IDS: [u64; 7] = [301, 302, 303, 304, 305, 306, 307];
    const CHAMFER_EDGE_VERTICES: [(usize, usize); 15] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 5),
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),
    ];

    fn chamfer_faces() -> Vec<ReducedFace> {
        let edge_use = |index: usize, forward: bool| {
            let (front, back) = CHAMFER_EDGE_VERTICES[index];
            edge(
                CHAMFER_EDGE_IDS[index],
                (CHAMFER_VERTEX_IDS[front], CHAMFER_VERTEX_IDS[back]),
                (CHAMFER_POINTS[front], CHAMFER_POINTS[back]),
                forward,
            )
        };
        let uses = |entries: &[(usize, bool)]| {
            entries
                .iter()
                .map(|&(index, forward)| edge_use(index, forward))
                .collect()
        };
        vec![
            plane_face(
                CHAMFER_FACE_IDS[0],
                uses(&[(0, true), (1, true), (2, true), (3, true), (4, true)]),
            ),
            plane_face(
                CHAMFER_FACE_IDS[1],
                uses(&[(9, false), (8, false), (7, false), (6, false), (5, false)]),
            ),
            plane_face(
                CHAMFER_FACE_IDS[2],
                uses(&[(0, false), (10, true), (5, true), (11, false)]),
            ),
            plane_face(
                CHAMFER_FACE_IDS[3],
                uses(&[(1, false), (11, true), (6, true), (12, false)]),
            ),
            plane_face(
                CHAMFER_FACE_IDS[4],
                uses(&[(2, false), (12, true), (7, true), (13, false)]),
            ),
            plane_face(
                CHAMFER_FACE_IDS[5],
                uses(&[(3, false), (13, true), (8, true), (14, false)]),
            ),
            plane_face(
                CHAMFER_FACE_IDS[6],
                uses(&[(4, false), (14, true), (9, true), (10, false)]),
            ),
        ]
    }

    fn chamfer_evidence() -> EdgeTreatmentEvidence {
        EdgeTreatmentEvidence {
            solid_id: SOLID_ID,
            treatment_face_id: CHAMFER_FACE_IDS[4],
            kind: EdgeTreatmentKind::PlanarChamfer,
            strength: EdgeTreatmentEvidenceStrength::AnalyticTangency,
            carrier_pair_unique: true,
            treatment_support: "plane".to_owned(),
            carriers: [
                EdgeTreatmentCarrier {
                    face_id: CHAMFER_FACE_IDS[3],
                    support: "plane".to_owned(),
                    shared_edge_ids: vec![CHAMFER_EDGE_IDS[12]],
                },
                EdgeTreatmentCarrier {
                    face_id: CHAMFER_FACE_IDS[5],
                    support: "plane".to_owned(),
                    shared_edge_ids: vec![CHAMFER_EDGE_IDS[13]],
                },
            ],
            inferred_radius_mm: None,
            inferred_setbacks_mm: Some([0.5, 0.5]),
            inferred_angle_rad: Some(std::f64::consts::FRAC_PI_4),
            carrier_angle_rad: std::f64::consts::FRAC_PI_2,
            virtual_edge: EdgeTreatmentVirtualEdge::Segment {
                start_mm: [2.0, 2.0, 0.0],
                end_mm: [2.0, 2.0, 1.0],
            },
            max_boundary_residual_mm: 0.0,
        }
    }

    fn topology_provenance(
        vertex_ids: &[u64],
        edge_ids: &[u64],
        face_ids: &[u64],
    ) -> KernelTopologyProvenance {
        KernelTopologyProvenance {
            vertices: vertex_ids
                .iter()
                .enumerate()
                .map(|(index, &id)| KernelTopologyEntityProvenance {
                    source: source(id),
                    topology_id: kid(1_000 + u64::try_from(index).unwrap_or(0)),
                })
                .collect(),
            edges: edge_ids
                .iter()
                .enumerate()
                .map(|(index, &id)| KernelTopologyEntityProvenance {
                    source: source(id),
                    topology_id: kid(2_000 + u64::try_from(index).unwrap_or(0)),
                })
                .collect(),
            faces: face_ids
                .iter()
                .enumerate()
                .map(|(index, &id)| KernelFaceTopologyProvenance {
                    source: None,
                    shell_member: Some(source(id)),
                    topology_id: kid(3_000 + u64::try_from(index).unwrap_or(0)),
                })
                .collect(),
        }
    }

    fn chamfer_fixture() -> (
        ReducedSolid,
        EdgeTreatmentEvidence,
        KernelTopologyProvenance,
        [u64; 10],
        [u64; 15],
        [u64; 7],
    ) {
        (
            ReducedSolid::from_test_faces(SOLID_ID, chamfer_faces()),
            chamfer_evidence(),
            topology_provenance(&CHAMFER_VERTEX_IDS, &CHAMFER_EDGE_IDS, &CHAMFER_FACE_IDS),
            CHAMFER_VERTEX_IDS,
            CHAMFER_EDGE_IDS,
            CHAMFER_FACE_IDS,
        )
    }

    #[test]
    fn chamfer_planner_emits_complete_identity_preserving_patch() -> Result<()> {
        let (solid, evidence, provenance, vertex_ids, edge_ids, face_ids) = chamfer_fixture();
        let patch = plan_edge_treatment_patch(&solid, &evidence, DATA_SECTION, &provenance)?;

        assert_eq!(patch.remove_faces, vec![kid(3_004)]);
        assert_eq!(patch.new_vertices, vec![[2.0, 2.0, 0.0], [2.0, 2.0, 1.0]]);
        assert_eq!(patch.new_edges.len(), 1);
        assert_eq!(
            patch.new_edges[0].vertices,
            (KernelPatchVertexRef::New(0), KernelPatchVertexRef::New(1))
        );
        assert!(matches!(
            patch.new_edges[0].curve,
            KernelPatchCurve::Line {
                start_mm: [2.0, 2.0, 0.0],
                end_mm: [2.0, 2.0, 1.0]
            }
        ));

        let rewritten = patch
            .rewrite_edges
            .iter()
            .map(|edge| edge.edge)
            .collect::<Vec<_>>();
        assert_eq!(
            rewritten,
            vec![kid(2_001), kid(2_003), kid(2_006), kid(2_008)]
        );
        let expected_unused = [edge_ids[2], edge_ids[7], edge_ids[12], edge_ids[13]]
            .into_iter()
            .map(|id| {
                provenance
                    .edge_for_source_id(DATA_SECTION, id)
                    .context("fixture edge provenance")
            })
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(patch.require_unused_edges, expected_unused);

        let by_face = patch
            .face_splices
            .iter()
            .map(|splice| (splice.face, splice))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(by_face.len(), 4);
        for terminal in [face_ids[0], face_ids[1]] {
            let face_id = provenance
                .face_for_source_id(DATA_SECTION, terminal)
                .context("fixture terminal face provenance")?;
            assert_eq!(by_face[&face_id].replacement, Vec::new());
        }
        for carrier in [face_ids[3], face_ids[5]] {
            let face_id = provenance
                .face_for_source_id(DATA_SECTION, carrier)
                .context("fixture carrier face provenance")?;
            assert_eq!(by_face[&face_id].replacement.len(), 1);
            assert_eq!(
                by_face[&face_id].replacement[0].edge,
                KernelPatchEdgeRef::New(0)
            );
        }

        // The untouched endpoint of every rewritten source edge still refers to its original
        // imported vertex identity; only the tangency endpoint becomes a new derived vertex.
        let expected_untouched = [vertex_ids[1], vertex_ids[4], vertex_ids[6], vertex_ids[9]]
            .into_iter()
            .map(|id| {
                provenance
                    .vertex_for_source_id(DATA_SECTION, id)
                    .context("fixture vertex provenance")
            })
            .collect::<Result<BTreeSet<_>>>()?;
        let actual_untouched = patch
            .rewrite_edges
            .iter()
            .flat_map(|edge| <[KernelPatchVertexRef; 2]>::from(edge.vertices))
            .filter_map(|reference| match reference {
                KernelPatchVertexRef::Existing(id) => Some(id),
                KernelPatchVertexRef::New(_) => None,
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(actual_untouched, expected_untouched);
        Ok(())
    }

    fn circular_edge(id: u64, vertex_id: u64, forward: bool) -> OrientedEdgeUse {
        let seam = [1.0, 0.0, 0.0];
        OrientedEdgeUse {
            oriented_edge_id: id + 10_000,
            edge_id: id,
            curve_id: id + 20_000,
            curve_same_sense: true,
            parameter_forward: forward,
            start_vertex: vertex_id,
            end_vertex: vertex_id,
            start_mm: seam,
            end_mm: seam,
            support: CurveSupport::Circle(crate::brep::CircleSupport {
                center_mm: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                x_direction: [1.0, 0.0, 0.0],
                radius_mm: 1.0,
            }),
        }
    }

    fn circular_face(id: u64, edge: OrientedEdgeUse) -> ReducedFace {
        plane_face(id, vec![edge])
    }

    fn circular_fixture() -> (
        ReducedSolid,
        EdgeTreatmentEvidence,
        KernelTopologyProvenance,
    ) {
        let first_face = 401;
        let second_face = 402;
        let treatment_face = 403;
        let first_edge = 501;
        let second_edge = 502;
        let treatment = ReducedFace {
            id: treatment_face,
            surface: SurfaceSupport::Other { entity_id: 9999 },
            surface_forward: true,
            loops: vec![
                FaceLoop {
                    bound_id: 701,
                    loop_id: 702,
                    outer: true,
                    orientation: true,
                    edges: vec![circular_edge(first_edge, 601, false)],
                },
                FaceLoop {
                    bound_id: 703,
                    loop_id: 704,
                    outer: false,
                    orientation: true,
                    edges: vec![circular_edge(second_edge, 602, true)],
                },
            ],
        };
        let solid = ReducedSolid::from_test_faces(
            SOLID_ID,
            vec![
                circular_face(first_face, circular_edge(first_edge, 601, true)),
                circular_face(second_face, circular_edge(second_edge, 602, false)),
                treatment,
            ],
        );
        let evidence = EdgeTreatmentEvidence {
            solid_id: SOLID_ID,
            treatment_face_id: treatment_face,
            kind: EdgeTreatmentKind::ToroidalFillet,
            strength: EdgeTreatmentEvidenceStrength::AnalyticTangency,
            carrier_pair_unique: true,
            treatment_support: "torus".to_owned(),
            carriers: [
                EdgeTreatmentCarrier {
                    face_id: first_face,
                    support: "plane".to_owned(),
                    shared_edge_ids: vec![first_edge],
                },
                EdgeTreatmentCarrier {
                    face_id: second_face,
                    support: "cylinder".to_owned(),
                    shared_edge_ids: vec![second_edge],
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
        let provenance = KernelTopologyProvenance {
            vertices: Vec::new(),
            edges: [first_edge, second_edge]
                .into_iter()
                .enumerate()
                .map(|(index, id)| KernelTopologyEntityProvenance {
                    source: source(id),
                    topology_id: kid(5_000 + u64::try_from(index).unwrap_or(0)),
                })
                .collect(),
            faces: [first_face, second_face, treatment_face]
                .into_iter()
                .enumerate()
                .map(|(index, id)| KernelFaceTopologyProvenance {
                    source: None,
                    shell_member: Some(source(id)),
                    topology_id: kid(6_000 + u64::try_from(index).unwrap_or(0)),
                })
                .collect(),
        };
        (solid, evidence, provenance)
    }

    #[test]
    fn circular_planner_replaces_illegal_single_edge_loops_with_shared_four_arc_cycle() -> Result<()>
    {
        let (solid, evidence, provenance) = circular_fixture();
        let patch = plan_edge_treatment_patch(&solid, &evidence, DATA_SECTION, &provenance)?;
        assert_eq!(patch.new_vertices.len(), 4);
        assert_eq!(patch.new_edges.len(), 4);
        assert_eq!(patch.rewrite_edges, Vec::new());
        assert_eq!(patch.face_splices.len(), 2);
        assert_eq!(
            patch.face_splices[0]
                .replacement
                .iter()
                .map(|edge| (edge.edge, edge.orientation))
                .collect::<Vec<_>>(),
            vec![
                (KernelPatchEdgeRef::New(0), true),
                (KernelPatchEdgeRef::New(1), true),
                (KernelPatchEdgeRef::New(2), true),
                (KernelPatchEdgeRef::New(3), true),
            ]
        );
        assert_eq!(
            patch.face_splices[1]
                .replacement
                .iter()
                .map(|edge| (edge.edge, edge.orientation))
                .collect::<Vec<_>>(),
            vec![
                (KernelPatchEdgeRef::New(3), false),
                (KernelPatchEdgeRef::New(2), false),
                (KernelPatchEdgeRef::New(1), false),
                (KernelPatchEdgeRef::New(0), false),
            ]
        );
        assert!(
            patch
                .new_edges
                .iter()
                .all(|edge| edge.vertices.0 != edge.vertices.1)
        );
        Ok(())
    }

    #[test]
    fn planner_rejects_incomplete_terminal_topology() -> Result<()> {
        let (mut solid, evidence, provenance, _, edge_ids, face_ids) = chamfer_fixture();
        let faces = solid
            .faces()
            .filter(|face| face.id != face_ids[0])
            .cloned()
            .collect();
        solid = ReducedSolid::from_test_faces(SOLID_ID, faces);
        let error = plan_edge_treatment_patch(&solid, &evidence, DATA_SECTION, &provenance)
            .err()
            .context("missing terminal face unexpectedly planned successfully")?;
        assert!(error.to_string().contains(&format!("#{}", edge_ids[2])));
        Ok(())
    }
}
