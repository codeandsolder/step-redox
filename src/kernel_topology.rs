//! Backend-neutral normalized topology edits for source-solid recovery.
//!
//! Geometry proof and STEP-source interpretation live above this layer.  The
//! backend adapter receives only explicit topology identities, bounded replacement
//! curves, and boundary splices; it must either apply the complete transaction or
//! leave the imported source solid unchanged.

use crate::SourceEntity;
use crate::kernel_source_load::KernelTopologyId;
use anyhow::{Context, Result, bail};
use std::collections::{BTreeMap, BTreeSet};

/// Apply normalized topology edits to one source solid without exposing backend-native topology.
pub trait SourceTopologyEditor {
    /// Re-import `source` from `input`, apply every patch in order transactionally,
    /// and serialize the resulting single solid as a fresh STEP exchange.
    ///
    /// # Errors
    /// Returns an error if the source solid cannot be imported, any normalized
    /// patch is invalid for the current topology, exact trim regeneration fails,
    /// or the healed solid cannot be serialized.
    fn edit_source_topology(
        &self,
        input: &[u8],
        source: SourceEntity,
        patches: &[KernelTopologyPatch],
    ) -> Result<String>;
}

/// Vertex reference used by a normalized backend-neutral topology patch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelPatchVertexRef {
    /// Reuse one existing kernel topology vertex.
    Existing(KernelTopologyId),
    /// Reference an entry in [`KernelTopologyPatch::new_vertices`].
    New(usize),
}

/// Edge reference used by a face-boundary splice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelPatchEdgeRef {
    /// Reuse one existing kernel topology edge.
    Existing(KernelTopologyId),
    /// Reference an entry in [`KernelTopologyPatch::new_edges`].
    New(usize),
}

/// Exact bounded curve supplied by the semantic planner.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelPatchCurve {
    /// Bounded line segment.
    Line {
        start_mm: [f64; 3],
        end_mm: [f64; 3],
    },
    /// Bounded circular arc in a deterministic orthonormal frame.
    CircleArc {
        center_mm: [f64; 3],
        normal: [f64; 3],
        x_direction: [f64; 3],
        radius_mm: f64,
        start_angle_rad: f64,
        end_angle_rad: f64,
    },
}

/// One new shared edge in a normalized topology patch.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelNewEdge {
    pub vertices: (KernelPatchVertexRef, KernelPatchVertexRef),
    pub curve: KernelPatchCurve,
}

/// Complete replacement for one retained shared edge while preserving its identity.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelRetainedEdge {
    pub edge: KernelTopologyId,
    pub vertices: (KernelPatchVertexRef, KernelPatchVertexRef),
    pub curve: KernelPatchCurve,
}

/// One oriented edge use inserted into a rebuilt face boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelPatchEdgeUse {
    pub edge: KernelPatchEdgeRef,
    pub orientation: bool,
}

/// Replace one contiguous cyclic run in one face boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelFaceSplice {
    pub face: KernelTopologyId,
    pub remove_edges: Vec<KernelTopologyId>,
    pub replacement: Vec<KernelPatchEdgeUse>,
}

/// Replace the complete boundary-wire set of one retained face.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelFaceBoundaryReplacement {
    pub face: KernelTopologyId,
    pub boundaries: Vec<Vec<KernelPatchEdgeUse>>,
}

/// Fully normalized topology edit for one imported source-solid boundary shell.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelTopologyPatch {
    pub remove_faces: Vec<KernelTopologyId>,
    pub new_vertices: Vec<[f64; 3]>,
    pub new_edges: Vec<KernelNewEdge>,
    pub rewrite_edges: Vec<KernelRetainedEdge>,
    pub face_splices: Vec<KernelFaceSplice>,
    pub face_boundary_replacements: Vec<KernelFaceBoundaryReplacement>,
    pub require_unused_edges: Vec<KernelTopologyId>,
}

const PATCH_POINT_TOL_MM: f64 = 1.0e-7;

/// Normalize one proven source edge treatment into a backend-neutral topology patch.
///
/// This is a read-only planning operation. It parses only the source DATA section named by
/// `source`, verifies that the requested source solid identity is still exact, and translates the
/// already-proven treatment evidence through the source-to-kernel `StableId` journal. No backend
/// topology is mutated here.
///
/// # Errors
/// Returns an error if the source identity cannot be resolved, the parser-side topology does not
/// match the evidence, a required source entity has no unique kernel topology identity, or the
/// treatment cannot be normalized into a complete fail-closed patch.
pub fn plan_source_edge_treatment_patch(
    input: &[u8],
    source: SourceEntity,
    evidence: &crate::edge_treatments::EdgeTreatmentEvidence,
    provenance: &crate::kernel_source_load::KernelTopologyProvenance,
) -> Result<KernelTopologyPatch> {
    plan_source_edge_treatment_patches(input, source, std::slice::from_ref(evidence), provenance)
}

/// Normalize every proven treatment on one source solid into one atomic topology transaction.
///
/// New vertices shared by independently proven local treatments are deduplicated geometrically.
/// Compatible rewrites of opposite endpoints of the same retained line edge are merged so adjacent
/// treatments cannot undo one another when the transaction is applied.
///
/// # Errors
/// Returns an error when the source identity or parser-side B-rep cannot be resolved, any local
/// treatment cannot be normalized, or two local plans make incompatible claims about retained
/// topology.
pub fn plan_source_edge_treatment_patches(
    input: &[u8],
    source: SourceEntity,
    evidence: &[crate::edge_treatments::EdgeTreatmentEvidence],
    provenance: &crate::kernel_source_load::KernelTopologyProvenance,
) -> Result<KernelTopologyPatch> {
    if evidence.is_empty() {
        bail!("edge-treatment patch batch cannot be empty");
    }
    let solid = source_reduced_solid(input, source)?;
    let mut ordered = evidence.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|item| item.treatment_face_id);
    let patches = ordered
        .into_iter()
        .map(|item| {
            crate::inverse_edge_treatment::plan_edge_treatment_patch(
                &solid,
                item,
                source.data_section,
                provenance,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    merge_topology_patches(patches)
}

/// Normalize representation-only coplanar partitions into one authoritative kernel topology edit.
///
/// The parser-side reduced B-rep is used only as a read-only proof view. Each proven component
/// retains its lowest source face, replaces that face's complete boundary decomposition with the
/// traced merged outer wire, removes the redundant coplanar faces, and requires every internal seam
/// edge to become unused before canonical compaction.
///
/// # Errors
/// Returns an error when source identity cannot be resolved or any proven source face/edge lacks a
/// unique retained kernel topology identity.
pub fn plan_source_coplanar_normalization_patch(
    input: &[u8],
    source: SourceEntity,
    provenance: &crate::kernel_source_load::KernelTopologyProvenance,
) -> Result<Option<KernelTopologyPatch>> {
    let solid = source_reduced_solid(input, source)?;
    plan_coplanar_normalization_patch(&solid, source.data_section, provenance)
}

/// Translate one already-proven boundary-feature peel into an authoritative kernel topology edit.
///
/// The feature proof stays in `planar_features`; this layer only translates the pure peel plan
/// through retained source topology identity. Host faces are rebuilt from every surviving source
/// `FACE_BOUND`, using the same bound-orientation × face-sense composition as the Monstertruck STEP
/// importer. Interface edges are retired only when their complete source-face usage is proven to be
/// contained in the retained host plus feature faces removed by this transaction.
///
/// # Errors
/// Returns an error when source identity changes, a selected face/edge lacks unique kernel identity,
/// a requested interface bound is not present on its host, a host would lose every boundary, or an
/// interface edge has any surviving source-face user outside the planned peel.
pub fn plan_source_boundary_feature_peel_patch(
    input: &[u8],
    source: SourceEntity,
    report: &crate::planar_features::BoundaryFeatureDiagnostics,
    selected_indices: &[usize],
    provenance: &crate::kernel_source_load::KernelTopologyProvenance,
) -> Result<KernelTopologyPatch> {
    let exchange = crate::step_io::ParsedExchange::parse(input)?.exchange;
    let section = exchange
        .data
        .get(source.data_section)
        .ok_or_else(|| anyhow::anyhow!("missing DATA section {}", source.data_section))?;
    let index = crate::step_graph::build_index(&section.entities);
    let actual_source =
        crate::step_document::StepSection::new(source.data_section, &section.entities, &index)
            .entity(source.id)
            .ok_or_else(|| anyhow::anyhow!("source solid #{} is missing", source.id))?
            .source();
    if actual_source != source {
        bail!("source solid identity changed: requested {source:?}, found {actual_source:?}");
    }

    let peel = crate::planar_features::plan_boundary_feature_family_peel(
        &section.entities,
        &index,
        report,
        selected_indices,
    )?;
    boundary_feature_peel_patch(source, &section.entities, &index, &peel, provenance)
}

#[expect(
    clippy::too_many_lines,
    reason = "the boundary-peel preflight keeps all identity, shell, host-bound and edge-use checks sequential and fail-closed"
)]
fn boundary_feature_peel_patch(
    source: SourceEntity,
    entities: &[ruststep::ast::EntityInstance],
    index: &std::collections::HashMap<u64, usize>,
    peel: &crate::planar_features::BoundaryFeaturePeelPlan,
    provenance: &crate::kernel_source_load::KernelTopologyProvenance,
) -> Result<KernelTopologyPatch> {
    let source_shell = crate::brep::manifold_shell(source.id, entities, index)
        .with_context(|| format!("source solid #{} has no readable shell", source.id))?;
    if peel
        .shell_face_removals
        .iter()
        .any(|removal| removal.shell_id != source_shell)
    {
        bail!("boundary peel contains removals from a different source shell");
    }
    let source_faces = crate::brep::solid_face_ids(source.id, entities, index)
        .with_context(|| {
            format!(
                "source solid #{} has no readable face membership",
                source.id
            )
        })?
        .into_iter()
        .collect::<BTreeSet<_>>();
    let removed_faces = peel
        .shell_face_removals
        .iter()
        .flat_map(|removal| removal.face_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    if !removed_faces.is_subset(&source_faces) {
        bail!(
            "boundary peel removes faces outside source solid #{}",
            source.id
        );
    }

    let edge_faces = source_edge_face_users(&source_faces, entities, index)?;
    let mut patch = empty_topology_patch();
    patch.remove_faces = removed_faces
        .iter()
        .copied()
        .map(|face_id| topology_face(provenance, source.data_section, face_id))
        .collect::<Result<Vec<_>>>()?;

    let mut required_unused = BTreeSet::new();
    for edit in &peel.host_bound_removals {
        if !source_faces.contains(&edit.host_face_id) {
            bail!(
                "boundary peel host face #{} is outside source solid #{}",
                edit.host_face_id,
                source.id
            );
        }
        if removed_faces.contains(&edit.host_face_id) {
            bail!(
                "boundary peel cannot remove and retain host face #{}",
                edit.host_face_id
            );
        }
        let loops =
            crate::brep::face_loops(edit.host_face_id, entities, index).with_context(|| {
                format!(
                    "feature host face #{} has unreadable bounds",
                    edit.host_face_id
                )
            })?;
        let remove_bounds = edit.bound_ids.iter().copied().collect::<BTreeSet<_>>();
        let actual_bounds = loops
            .iter()
            .map(|loop_| loop_.bound_id)
            .collect::<BTreeSet<_>>();
        if !remove_bounds.is_subset(&actual_bounds) {
            bail!(
                "feature host face #{} does not own every planned interface bound",
                edit.host_face_id
            );
        }
        let face_forward =
            match crate::brep::face_sense(edit.host_face_id, entities, index).as_deref() {
                Some("T") => true,
                Some("F") => false,
                _ => bail!(
                    "feature host face #{} has invalid face sense",
                    edit.host_face_id
                ),
            };
        let boundaries = loops
            .iter()
            .filter(|loop_| !remove_bounds.contains(&loop_.bound_id))
            .map(|loop_| {
                source_loop_kernel_boundary(loop_, face_forward, source.data_section, provenance)
            })
            .collect::<Result<Vec<_>>>()?;
        if boundaries.is_empty() {
            bail!(
                "boundary peel would leave host face #{} without a boundary",
                edit.host_face_id
            );
        }
        patch
            .face_boundary_replacements
            .push(KernelFaceBoundaryReplacement {
                face: topology_face(provenance, source.data_section, edit.host_face_id)?,
                boundaries,
            });

        for loop_ in loops
            .iter()
            .filter(|loop_| remove_bounds.contains(&loop_.bound_id))
        {
            if loop_.outer {
                bail!(
                    "boundary peel cannot retire outer bound #{}",
                    loop_.bound_id
                );
            }
            for edge in &loop_.edges {
                let Some(users) = edge_faces.get(&edge.edge_id) else {
                    bail!("interface edge #{} has no source-face users", edge.edge_id);
                };
                if !users.contains(&edit.host_face_id)
                    || !users.iter().any(|face| removed_faces.contains(face))
                    || users
                        .iter()
                        .any(|face| *face != edit.host_face_id && !removed_faces.contains(face))
                {
                    bail!(
                        "interface edge #{} has users outside retained host #{} and removed feature faces",
                        edge.edge_id,
                        edit.host_face_id
                    );
                }
                required_unused.insert(topology_edge(
                    provenance,
                    source.data_section,
                    edge.edge_id,
                )?);
            }
        }
    }
    patch.require_unused_edges = required_unused.into_iter().collect();
    Ok(patch)
}

fn source_loop_kernel_boundary(
    loop_: &crate::brep::FaceLoop,
    face_forward: bool,
    data_section: usize,
    provenance: &crate::kernel_source_load::KernelTopologyProvenance,
) -> Result<Vec<KernelPatchEdgeUse>> {
    let absolute_orientation = loop_.orientation == face_forward;
    let mut boundary = loop_
        .edges
        .iter()
        .map(|edge| {
            let effective_forward = edge.parameter_forward == edge.curve_same_sense;
            Ok(KernelPatchEdgeUse {
                edge: KernelPatchEdgeRef::Existing(topology_edge(
                    provenance,
                    data_section,
                    edge.edge_id,
                )?),
                orientation: effective_forward == absolute_orientation,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if !absolute_orientation {
        boundary.reverse();
    }
    Ok(boundary)
}

fn source_edge_face_users(
    faces: &BTreeSet<u64>,
    entities: &[ruststep::ast::EntityInstance],
    index: &std::collections::HashMap<u64, usize>,
) -> Result<BTreeMap<u64, BTreeSet<u64>>> {
    let mut users = BTreeMap::<u64, BTreeSet<u64>>::new();
    for &face_id in faces {
        let loops = crate::brep::face_loops(face_id, entities, index)
            .with_context(|| format!("source face #{face_id} has unreadable bounds"))?;
        for edge in loops.iter().flat_map(|loop_| &loop_.edges) {
            users.entry(edge.edge_id).or_default().insert(face_id);
        }
    }
    Ok(users)
}

const fn empty_topology_patch() -> KernelTopologyPatch {
    KernelTopologyPatch {
        remove_faces: Vec::new(),
        new_vertices: Vec::new(),
        new_edges: Vec::new(),
        rewrite_edges: Vec::new(),
        face_splices: Vec::new(),
        face_boundary_replacements: Vec::new(),
        require_unused_edges: Vec::new(),
    }
}

fn plan_coplanar_normalization_patch(
    solid: &crate::reduced_brep::ReducedSolid,
    data_section: usize,
    provenance: &crate::kernel_source_load::KernelTopologyProvenance,
) -> Result<Option<KernelTopologyPatch>> {
    let plans = solid.plan_coplanar_face_merges();
    if plans.is_empty() {
        return Ok(None);
    }

    let mut patch = empty_topology_patch();
    for plan in plans {
        patch.remove_faces.extend(
            plan.remove_face_ids
                .into_iter()
                .map(|face_id| topology_face(provenance, data_section, face_id))
                .collect::<Result<Vec<_>>>()?,
        );
        let boundary = plan
            .boundary
            .iter()
            .map(|edge| {
                Ok(KernelPatchEdgeUse {
                    edge: KernelPatchEdgeRef::Existing(topology_edge(
                        provenance,
                        data_section,
                        edge.edge_id,
                    )?),
                    orientation: edge.parameter_forward == edge.curve_same_sense,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        patch
            .face_boundary_replacements
            .push(KernelFaceBoundaryReplacement {
                face: topology_face(provenance, data_section, plan.canonical_face_id)?,
                boundaries: vec![boundary],
            });
        patch.require_unused_edges.extend(
            plan.internal_edge_ids
                .into_iter()
                .map(|edge_id| topology_edge(provenance, data_section, edge_id))
                .collect::<Result<Vec<_>>>()?,
        );
    }
    Ok(Some(patch))
}

fn topology_edge(
    provenance: &crate::kernel_source_load::KernelTopologyProvenance,
    data_section: usize,
    source_id: u64,
) -> Result<KernelTopologyId> {
    provenance
        .edge_for_source_id(data_section, source_id)
        .with_context(|| format!("source edge #{source_id} has no unique kernel StableId"))
}

fn topology_face(
    provenance: &crate::kernel_source_load::KernelTopologyProvenance,
    data_section: usize,
    source_id: u64,
) -> Result<KernelTopologyId> {
    provenance
        .face_for_source_id(data_section, source_id)
        .with_context(|| format!("source face #{source_id} has no unique kernel StableId"))
}

fn source_reduced_solid(
    input: &[u8],
    source: SourceEntity,
) -> Result<crate::reduced_brep::ReducedSolid> {
    let exchange = crate::step_io::ParsedExchange::parse(input)?.exchange;
    let section = exchange
        .data
        .get(source.data_section)
        .ok_or_else(|| anyhow::anyhow!("missing DATA section {}", source.data_section))?;
    let index = crate::step_graph::build_index(&section.entities);
    let actual_source =
        crate::step_document::StepSection::new(source.data_section, &section.entities, &index)
            .entity(source.id)
            .ok_or_else(|| anyhow::anyhow!("source solid #{} is missing", source.id))?
            .source();
    if actual_source != source {
        bail!("source solid identity changed: requested {source:?}, found {actual_source:?}");
    }
    crate::reduced_brep::ReducedSolid::from_step(source.id, &section.entities, &index)
        .with_context(|| format!("source solid #{} has no complete reduced B-rep", source.id))
}

fn merge_topology_patches(patches: Vec<KernelTopologyPatch>) -> Result<KernelTopologyPatch> {
    let mut merged = KernelTopologyPatch {
        remove_faces: Vec::new(),
        new_vertices: Vec::new(),
        new_edges: Vec::new(),
        rewrite_edges: Vec::new(),
        face_splices: Vec::new(),
        face_boundary_replacements: Vec::new(),
        require_unused_edges: Vec::new(),
    };
    let mut removed_faces = BTreeSet::new();
    let mut required_unused = BTreeSet::new();
    let mut rewrites = BTreeMap::<KernelTopologyId, KernelRetainedEdge>::new();

    for patch in patches {
        let vertex_map = merge_new_vertices(&mut merged.new_vertices, &patch.new_vertices);
        let edge_offset = merged.new_edges.len();
        merged
            .new_edges
            .extend(patch.new_edges.into_iter().map(|mut edge| {
                edge.vertices.0 = remap_vertex_ref(edge.vertices.0, &vertex_map);
                edge.vertices.1 = remap_vertex_ref(edge.vertices.1, &vertex_map);
                edge
            }));
        for face in patch.remove_faces {
            if removed_faces.insert(face) {
                merged.remove_faces.push(face);
            }
        }
        for mut rewrite in patch.rewrite_edges {
            rewrite.vertices.0 = remap_vertex_ref(rewrite.vertices.0, &vertex_map);
            rewrite.vertices.1 = remap_vertex_ref(rewrite.vertices.1, &vertex_map);
            match rewrites.entry(rewrite.edge) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(rewrite);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let combined = merge_retained_edge(entry.get(), &rewrite)?;
                    entry.insert(combined);
                }
            }
        }
        merged
            .face_splices
            .extend(patch.face_splices.into_iter().map(|mut splice| {
                for edge_use in &mut splice.replacement {
                    if let KernelPatchEdgeRef::New(index) = edge_use.edge {
                        edge_use.edge = KernelPatchEdgeRef::New(edge_offset + index);
                    }
                }
                splice
            }));
        merged
            .face_boundary_replacements
            .extend(
                patch
                    .face_boundary_replacements
                    .into_iter()
                    .map(|mut replacement| {
                        for wire in &mut replacement.boundaries {
                            for edge_use in wire {
                                if let KernelPatchEdgeRef::New(index) = edge_use.edge {
                                    edge_use.edge = KernelPatchEdgeRef::New(edge_offset + index);
                                }
                            }
                        }
                        replacement
                    }),
            );
        for edge in patch.require_unused_edges {
            if required_unused.insert(edge) {
                merged.require_unused_edges.push(edge);
            }
        }
    }
    merged.rewrite_edges = rewrites.into_values().collect();
    Ok(merged)
}

fn merge_new_vertices(global: &mut Vec<[f64; 3]>, local: &[[f64; 3]]) -> Vec<usize> {
    local
        .iter()
        .map(|&point| {
            global
                .iter()
                .position(|&candidate| points_close(candidate, point))
                .unwrap_or_else(|| {
                    let index = global.len();
                    global.push(point);
                    index
                })
        })
        .collect()
}

const fn remap_vertex_ref(
    reference: KernelPatchVertexRef,
    vertex_map: &[usize],
) -> KernelPatchVertexRef {
    match reference {
        KernelPatchVertexRef::Existing(id) => KernelPatchVertexRef::Existing(id),
        KernelPatchVertexRef::New(index) => KernelPatchVertexRef::New(vertex_map[index]),
    }
}

fn merge_retained_edge(
    first: &KernelRetainedEdge,
    second: &KernelRetainedEdge,
) -> Result<KernelRetainedEdge> {
    let first_points = line_points(&first.curve).with_context(|| {
        format!(
            "retained edge {} receives non-line overlapping rewrites",
            first.edge.raw()
        )
    })?;
    let second_points = line_points(&second.curve).with_context(|| {
        format!(
            "retained edge {} receives non-line overlapping rewrites",
            first.edge.raw()
        )
    })?;
    let first_refs: [KernelPatchVertexRef; 2] = first.vertices.into();
    let second_refs: [KernelPatchVertexRef; 2] = second.vertices.into();
    let mut refs = first_refs;
    let mut points = first_points;
    for endpoint in 0..2 {
        match merge_endpoint(first_refs[endpoint], second_refs[endpoint]) {
            Some(reference) => {
                if reference != first_refs[endpoint] {
                    points[endpoint] = second_points[endpoint];
                } else if reference == second_refs[endpoint]
                    && !points_close(first_points[endpoint], second_points[endpoint])
                {
                    bail!(
                        "retained edge {} receives incompatible endpoint geometry",
                        first.edge.raw()
                    );
                }
                refs[endpoint] = reference;
            }
            None => bail!(
                "retained edge {} receives incompatible endpoint topology",
                first.edge.raw()
            ),
        }
    }
    if points_close(points[0], points[1]) {
        bail!(
            "retained edge {} collapses while merging treatment plans",
            first.edge.raw()
        );
    }
    Ok(KernelRetainedEdge {
        edge: first.edge,
        vertices: refs.into(),
        curve: KernelPatchCurve::Line {
            start_mm: points[0],
            end_mm: points[1],
        },
    })
}

fn merge_endpoint(
    first: KernelPatchVertexRef,
    second: KernelPatchVertexRef,
) -> Option<KernelPatchVertexRef> {
    if first == second {
        return Some(first);
    }
    match (first, second) {
        (KernelPatchVertexRef::Existing(_), KernelPatchVertexRef::New(_)) => Some(second),
        (KernelPatchVertexRef::New(_), KernelPatchVertexRef::Existing(_)) => Some(first),
        _ => None,
    }
}

const fn line_points(curve: &KernelPatchCurve) -> Option<[[f64; 3]; 2]> {
    match *curve {
        KernelPatchCurve::Line { start_mm, end_mm } => Some([start_mm, end_mm]),
        KernelPatchCurve::CircleArc { .. } => None,
    }
}

fn points_close(first: [f64; 3], second: [f64; 3]) -> bool {
    first
        .into_iter()
        .zip(second)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        <= PATCH_POINT_TOL_MM * PATCH_POINT_TOL_MM
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn id(raw: u64) -> KernelTopologyId {
        KernelTopologyId::from_raw(raw)
    }

    fn empty_patch() -> KernelTopologyPatch {
        KernelTopologyPatch {
            remove_faces: Vec::new(),
            new_vertices: Vec::new(),
            new_edges: Vec::new(),
            rewrite_edges: Vec::new(),
            face_splices: Vec::new(),
            face_boundary_replacements: Vec::new(),
            require_unused_edges: Vec::new(),
        }
    }

    #[test]
    fn source_loop_translation_matches_step_import_orientation_composition() -> Result<()> {
        let provenance = crate::kernel_source_load::KernelTopologyProvenance {
            vertices: Vec::new(),
            edges: vec![
                crate::kernel_source_load::KernelTopologyEntityProvenance {
                    source: source_entity(11),
                    topology_id: id(111),
                },
                crate::kernel_source_load::KernelTopologyEntityProvenance {
                    source: source_entity(12),
                    topology_id: id(112),
                },
            ],
            faces: Vec::new(),
        };
        let loop_ = crate::brep::FaceLoop {
            bound_id: 100,
            loop_id: 101,
            outer: true,
            orientation: true,
            edges: vec![
                planar_edge(11, (1, 2), ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]), true),
                planar_edge(12, (2, 3), ([1.0, 0.0, 0.0], [1.0, 1.0, 0.0]), false),
            ],
        };

        // FACE_BOUND=.T. on a same_sense=.F. face means absolute orientation is false:
        // use orientations are composed with false and the entire wire order reverses.
        let boundary = source_loop_kernel_boundary(&loop_, false, 0, &provenance)?;
        assert_eq!(
            boundary,
            vec![
                KernelPatchEdgeUse {
                    edge: KernelPatchEdgeRef::Existing(id(112)),
                    orientation: true,
                },
                KernelPatchEdgeUse {
                    edge: KernelPatchEdgeRef::Existing(id(111)),
                    orientation: false,
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn merges_opposite_endpoint_rewrites_of_one_retained_line() -> Result<()> {
        let mut first = empty_patch();
        first.new_vertices.push([1.0, 0.0, 0.0]);
        first.rewrite_edges.push(KernelRetainedEdge {
            edge: id(10),
            vertices: (
                KernelPatchVertexRef::New(0),
                KernelPatchVertexRef::Existing(id(2)),
            ),
            curve: KernelPatchCurve::Line {
                start_mm: [1.0, 0.0, 0.0],
                end_mm: [2.0, 0.0, 0.0],
            },
        });

        let mut second = empty_patch();
        second.new_vertices.push([3.0, 0.0, 0.0]);
        second.rewrite_edges.push(KernelRetainedEdge {
            edge: id(10),
            vertices: (
                KernelPatchVertexRef::Existing(id(1)),
                KernelPatchVertexRef::New(0),
            ),
            curve: KernelPatchCurve::Line {
                start_mm: [0.0, 0.0, 0.0],
                end_mm: [3.0, 0.0, 0.0],
            },
        });

        let merged = merge_topology_patches(vec![first, second])?;
        assert_eq!(merged.new_vertices, vec![[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]);
        assert_eq!(merged.rewrite_edges.len(), 1);
        let rewrite = &merged.rewrite_edges[0];
        assert_eq!(
            rewrite.vertices,
            (KernelPatchVertexRef::New(0), KernelPatchVertexRef::New(1))
        );
        assert_eq!(
            rewrite.curve,
            KernelPatchCurve::Line {
                start_mm: [1.0, 0.0, 0.0],
                end_mm: [3.0, 0.0, 0.0],
            }
        );
        Ok(())
    }

    #[test]
    fn deduplicates_shared_derived_vertices_across_local_plans() -> Result<()> {
        let mut first = empty_patch();
        first.new_vertices.push([1.0, 2.0, 3.0]);
        first.new_edges.push(KernelNewEdge {
            vertices: (
                KernelPatchVertexRef::Existing(id(1)),
                KernelPatchVertexRef::New(0),
            ),
            curve: KernelPatchCurve::Line {
                start_mm: [0.0, 0.0, 0.0],
                end_mm: [1.0, 2.0, 3.0],
            },
        });

        let mut second = empty_patch();
        second
            .new_vertices
            .push([0.25_f64.mul_add(PATCH_POINT_TOL_MM, 1.0), 2.0, 3.0]);
        second.new_edges.push(KernelNewEdge {
            vertices: (
                KernelPatchVertexRef::New(0),
                KernelPatchVertexRef::Existing(id(2)),
            ),
            curve: KernelPatchCurve::Line {
                start_mm: [1.0, 2.0, 3.0],
                end_mm: [4.0, 5.0, 6.0],
            },
        });
        second.face_splices.push(KernelFaceSplice {
            face: id(20),
            remove_edges: vec![id(30)],
            replacement: vec![KernelPatchEdgeUse {
                edge: KernelPatchEdgeRef::New(0),
                orientation: true,
            }],
        });

        let merged = merge_topology_patches(vec![first, second])?;
        assert_eq!(merged.new_vertices.len(), 1);
        assert_eq!(merged.new_edges[1].vertices.0, KernelPatchVertexRef::New(0));
        assert_eq!(
            merged.face_splices[0].replacement[0].edge,
            KernelPatchEdgeRef::New(1)
        );
        Ok(())
    }
    fn source_entity(id: u64) -> SourceEntity {
        SourceEntity {
            data_section: 0,
            ordinal: usize::try_from(id).unwrap_or(0),
            id,
        }
    }

    fn planar_edge(
        id: u64,
        vertices: (u64, u64),
        points: ([f64; 3], [f64; 3]),
        forward: bool,
    ) -> crate::brep::OrientedEdgeUse {
        let (start_vertex, end_vertex, start_mm, end_mm) = if forward {
            (vertices.0, vertices.1, points.0, points.1)
        } else {
            (vertices.1, vertices.0, points.1, points.0)
        };
        crate::brep::OrientedEdgeUse {
            oriented_edge_id: id + 10_000,
            edge_id: id,
            curve_id: id + 20_000,
            curve_same_sense: true,
            parameter_forward: forward,
            start_vertex,
            end_vertex,
            start_mm,
            end_mm,
            support: crate::brep::CurveSupport::Line(crate::brep::LineSupport {
                origin_mm: points.0,
                direction: [
                    points.1[0] - points.0[0],
                    points.1[1] - points.0[1],
                    points.1[2] - points.0[2],
                ],
            }),
        }
    }

    fn planar_face(
        id: u64,
        edges: Vec<crate::brep::OrientedEdgeUse>,
    ) -> crate::reduced_brep::ReducedFace {
        crate::reduced_brep::ReducedFace {
            id,
            surface: crate::brep::SurfaceSupport::Plane(crate::brep::PlaneSupport {
                origin_mm: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                max_residual_mm: 0.0,
            }),
            surface_forward: true,
            loops: vec![crate::brep::FaceLoop {
                bound_id: id + 30_000,
                loop_id: id + 40_000,
                outer: true,
                orientation: true,
                edges,
            }],
        }
    }

    #[test]
    fn coplanar_plan_replaces_complete_canonical_boundary_and_retires_seam() -> Result<()> {
        let p0 = [0.0, 0.0, 0.0];
        let p1 = [1.0, 0.0, 0.0];
        let p2 = [1.0, 1.0, 0.0];
        let p3 = [0.0, 1.0, 0.0];
        let p4 = [2.0, 0.0, 0.0];
        let p5 = [2.0, 1.0, 0.0];
        let first = planar_face(
            10,
            vec![
                planar_edge(100, (1, 2), (p0, p1), true),
                planar_edge(102, (2, 3), (p1, p2), true),
                planar_edge(101, (3, 4), (p2, p3), true),
                planar_edge(103, (4, 1), (p3, p0), true),
            ],
        );
        let second = planar_face(
            11,
            vec![
                planar_edge(104, (2, 5), (p1, p4), true),
                planar_edge(105, (5, 6), (p4, p5), true),
                planar_edge(106, (6, 3), (p5, p2), true),
                planar_edge(102, (2, 3), (p1, p2), false),
            ],
        );
        let solid = crate::reduced_brep::ReducedSolid::from_test_faces(500, vec![first, second]);
        let provenance = crate::kernel_source_load::KernelTopologyProvenance {
            vertices: Vec::new(),
            edges: (100..=106)
                .map(
                    |source_id| crate::kernel_source_load::KernelTopologyEntityProvenance {
                        source: source_entity(source_id),
                        topology_id: id(source_id + 1_000),
                    },
                )
                .collect(),
            faces: [10, 11]
                .into_iter()
                .map(
                    |source_id| crate::kernel_source_load::KernelFaceTopologyProvenance {
                        source: None,
                        shell_member: Some(source_entity(source_id)),
                        topology_id: id(source_id + 2_000),
                    },
                )
                .collect(),
        };

        let patch = plan_coplanar_normalization_patch(&solid, 0, &provenance)?
            .ok_or_else(|| anyhow::anyhow!("coplanar component was not planned"))?;
        assert_eq!(patch.remove_faces, vec![id(2_011)]);
        assert_eq!(patch.require_unused_edges, vec![id(1_102)]);
        assert_eq!(patch.face_boundary_replacements.len(), 1);
        let replacement = &patch.face_boundary_replacements[0];
        assert_eq!(replacement.face, id(2_010));
        let edge_ids = replacement.boundaries[0]
            .iter()
            .map(|edge_use| match edge_use.edge {
                KernelPatchEdgeRef::Existing(id) => id,
                KernelPatchEdgeRef::New(_) => KernelTopologyId::from_raw(0),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            edge_ids,
            vec![
                id(1_100),
                id(1_104),
                id(1_105),
                id(1_106),
                id(1_101),
                id(1_103)
            ]
        );
        assert!(
            replacement.boundaries[0]
                .iter()
                .all(|edge_use| edge_use.orientation)
        );
        Ok(())
    }
}
