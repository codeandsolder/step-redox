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
            require_unused_edges: Vec::new(),
        }
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
}
