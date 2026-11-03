//! Backend-neutral normalized topology edits for source-solid recovery.
//!
//! Geometry proof and STEP-source interpretation live above this layer.  The
//! backend adapter receives only explicit topology identities, bounded replacement
//! curves, and boundary splices; it must either apply the complete transaction or
//! leave the imported source solid unchanged.

use crate::SourceEntity;
use crate::kernel_source_load::KernelTopologyId;
use anyhow::Result;

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

/// Fully normalized topology edit for one imported source-solid boundary shell.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelTopologyPatch {
    pub remove_faces: Vec<KernelTopologyId>,
    pub new_vertices: Vec<[f64; 3]>,
    pub new_edges: Vec<KernelNewEdge>,
    pub rewrite_edges: Vec<KernelRetainedEdge>,
    pub face_splices: Vec<KernelFaceSplice>,
    pub require_unused_edges: Vec<KernelTopologyId>,
}

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
        anyhow::bail!(
            "source solid identity changed: requested {source:?}, found {actual_source:?}"
        );
    }
    let solid = crate::reduced_brep::ReducedSolid::from_step(source.id, &section.entities, &index)
        .ok_or_else(|| {
            anyhow::anyhow!("source solid #{} has no complete reduced B-rep", source.id)
        })?;
    crate::inverse_edge_treatment::plan_edge_treatment_patch(
        &solid,
        evidence,
        source.data_section,
        provenance,
    )
}
