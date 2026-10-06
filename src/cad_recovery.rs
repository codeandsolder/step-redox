use crate::cad_ir::{
    CadModel, CadNode, FusedPeriodicChain, IndexedCount, NodeId, ProofStatus, Provenance,
};
use crate::periodic_chains::PeriodicChainPattern;
use anyhow::{Result, bail};
use serde::Serialize;

mod solid;
pub use solid::{
    recover_radial_slot_revolution_fragment, recover_radial_slot_revolution_fragments,
    recover_solid_extrusion_fragment, recover_solid_extrusion_fragments,
    recover_solid_revolution_fragment, recover_solid_revolution_fragments,
};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CadFragment {
    pub source: CadFragmentSource,
    pub model: CadModel,
    pub root: NodeId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CadFragmentSource {
    SolidExtrusion {
        solid_id: u64,
        cap_face_ids: [u64; 2],
        side_face_ids: Vec<u64>,
    },
    SolidRevolution {
        solid_id: u64,
        face_ids: Vec<u64>,
    },
    RadialSlotRevolution {
        solid_id: u64,
        base_face_ids: Vec<u64>,
        slot_face_ids: [u64; 3],
    },
    PeriodicChain {
        solid_id: u64,
        sites: usize,
    },
}

/// Recover a first-class fused periodic-chain IR fragment from a proven source partition.
///
/// This node is descriptive, not yet executable by Monstertruck/KCL. It references the exact
/// source solid while recording the proven pitch/count grammar; detailed face ownership stays
/// in the detector/debug evidence rather than bloating high-level IR. Backends that do not
/// implement fused periodic chains must reject the node explicitly instead of lowering it as
/// a pattern of independent solids.
///
/// # Errors
/// Returns an error if the chain is not fully proven or its partition is inconsistent.
pub fn recover_periodic_chain_fragment(chain: &PeriodicChainPattern) -> Result<CadFragment> {
    validate_periodic_chain(chain)?;

    let adjacent_site_edges_per_boundary = chain
        .adjacent_site_edge_counts
        .first()
        .copied()
        .unwrap_or(0);

    let mut model = CadModel::new();
    let root = model.add_node(CadNode::PeriodicChain(FusedPeriodicChain {
        source_solid_id: chain.solid_id,
        axis: chain.axis,
        pitch_mm: chain.pitch_mm,
        sites: chain.sites,
        first_site_center_mm: chain.site_centers_mm[0],
        interior_site_face_count: chain.interior_site_face_count,
        site_face_count_overrides: count_overrides(
            &chain.site_face_counts,
            chain.interior_site_face_count,
        ),
        interior_gap_face_count: chain.interior_gap_face_count,
        gap_face_count_overrides: count_overrides(
            &chain.gap_face_counts,
            chain.interior_gap_face_count,
        ),
        stretch_face_count: chain.stretch_face_ids.len(),
        fixed_negative_face_count: chain.fixed_negative_face_ids.len(),
        fixed_positive_face_count: chain.fixed_positive_face_ids.len(),
        adjacent_site_edges_per_boundary,
        repeat_coverage_ratio: chain.repeat_coverage_ratio,
    }));

    model.set_provenance(
        root,
        Provenance {
            source_entity_ids: vec![chain.solid_id],
            // This is a structural partition/grammar proof, not a reconstructed
            // geometry comparison, so there is no geometric residual to invent.
            proof: ProofStatus::StructurallyProven,
            max_residual_mm: None,
        },
    )?;
    model.add_root(root)?;
    model.validate()?;

    Ok(CadFragment {
        source: CadFragmentSource::PeriodicChain {
            solid_id: chain.solid_id,
            sites: chain.sites,
        },
        model,
        root,
    })
}

/// Recover compact CAD-IR fragments for every periodic chain whose detector proof is complete.
///
/// Diagnostic candidates that are not `read_only_proven` remain available through the
/// periodic-chain report, but they are deliberately excluded from canonical CAD IR.
///
/// # Errors
/// Returns the first inconsistency found in a chain already marked as proven.
pub fn recover_periodic_chain_fragments(
    chains: &[PeriodicChainPattern],
) -> Result<Vec<CadFragment>> {
    chains
        .iter()
        .filter(|chain| chain.read_only_proven)
        .map(recover_periodic_chain_fragment)
        .collect()
}

fn count_overrides(counts: &[usize], baseline: usize) -> Vec<IndexedCount> {
    counts
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, count)| (count != baseline).then_some(IndexedCount { index, count }))
        .collect()
}

fn validate_periodic_chain(chain: &PeriodicChainPattern) -> Result<()> {
    if !chain.read_only_proven
        || !chain.complete_partition
        || chain.faces_without_geometry != 0
        || chain.nonmanifold_edges != 0
        || chain.nonlocal_cross_site_edges != 0
        || !chain.fixed_middle_face_ids.is_empty()
    {
        bail!("periodic-chain source proof is not complete and editable");
    }
    if chain.solid_id == 0 || chain.sites < 2 {
        bail!("periodic-chain source identity/site count is invalid");
    }
    if !chain.pitch_mm.is_finite() || chain.pitch_mm <= 0.0 {
        bail!("periodic-chain source pitch is invalid");
    }
    if chain.axis.iter().any(|value| !value.is_finite()) {
        bail!("periodic-chain source axis is non-finite");
    }
    let axis_norm = chain.axis[2]
        .mul_add(
            chain.axis[2],
            chain.axis[1].mul_add(chain.axis[1], chain.axis[0] * chain.axis[0]),
        )
        .sqrt();
    if (axis_norm - 1.0).abs() > 1.0e-9 {
        bail!("periodic-chain source axis is not unit length");
    }
    if chain.site_centers_mm.iter().any(|value| !value.is_finite()) {
        bail!("periodic-chain source site centers are non-finite");
    }
    let pitch_tolerance = 1.0e-5_f64.max(chain.pitch_mm.abs() * 1.0e-9);
    if chain
        .site_centers_mm
        .windows(2)
        .any(|pair| ((pair[1] - pair[0]).abs() - chain.pitch_mm).abs() > pitch_tolerance)
    {
        bail!("periodic-chain source site centers disagree with pitch");
    }
    if chain.site_face_ids.len() != chain.sites
        || chain.site_face_counts.len() != chain.sites
        || chain.site_centers_mm.len() != chain.sites
        || chain.gap_face_ids.len() + 1 != chain.sites
        || chain.gap_face_counts.len() + 1 != chain.sites
        || chain.adjacent_site_edge_counts.len() + 1 != chain.sites
    {
        bail!("periodic-chain proof arrays disagree with site count");
    }
    if chain.interior_site_face_count == 0 || chain.site_face_counts.contains(&0) {
        bail!("periodic-chain proof contains an empty site");
    }
    if chain
        .site_face_ids
        .iter()
        .zip(&chain.site_face_counts)
        .any(|(faces, &count)| faces.len() != count)
        || chain
            .gap_face_ids
            .iter()
            .zip(&chain.gap_face_counts)
            .any(|(faces, &count)| faces.len() != count)
    {
        bail!("periodic-chain face counts disagree with face partitions");
    }
    if chain
        .adjacent_site_edge_counts
        .first()
        .is_some_and(|first| {
            chain
                .adjacent_site_edge_counts
                .iter()
                .any(|count| count != first)
        })
    {
        bail!("periodic-chain adjacent-site seam width is inconsistent");
    }

    let mut owned = std::collections::HashSet::new();
    for face in chain
        .site_face_ids
        .iter()
        .flatten()
        .chain(chain.gap_face_ids.iter().flatten())
        .chain(chain.stretch_face_ids.iter())
        .chain(chain.fixed_negative_face_ids.iter())
        .chain(chain.fixed_positive_face_ids.iter())
    {
        if *face == 0 || !owned.insert(*face) {
            bail!("periodic-chain source face partition is invalid");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn periodic_chain_fixture() -> PeriodicChainPattern {
        PeriodicChainPattern {
            solid_id: 42,
            axis: [1.0, 0.0, 0.0],
            pitch_mm: 2.54,
            sites: 4,
            site_centers_mm: vec![0.0, 2.54, 5.08, 7.62],
            lattice_face_family_votes: 8,
            complete_partition: true,
            read_only_proven: true,
            faces_without_geometry: 0,
            nonmanifold_edges: 0,
            cross_site_edges: 6,
            nonlocal_cross_site_edges: 0,
            adjacent_site_edge_counts: vec![2, 2, 2],
            site_face_counts: vec![2, 2, 2, 3],
            interior_site_face_count: 2,
            gap_face_counts: vec![1, 1, 1],
            interior_gap_face_count: 1,
            site_face_ids: vec![vec![10, 11], vec![20, 21], vec![30, 31], vec![40, 41, 42]],
            gap_face_ids: vec![vec![50], vec![51], vec![52]],
            stretch_face_ids: vec![60],
            fixed_negative_face_ids: vec![70],
            fixed_middle_face_ids: Vec::new(),
            fixed_positive_face_ids: vec![71],
            repeat_coverage_ratio: 0.9,
            edge_category_counts: std::collections::BTreeMap::new(),
            site_adjacency_signatures: vec![std::collections::BTreeMap::new(); 4],
            gap_adjacency_signatures: vec![std::collections::BTreeMap::new(); 3],
        }
    }

    #[test]
    fn proven_fused_periodic_chain_becomes_first_class_ir() -> Result<()> {
        let chain = periodic_chain_fixture();
        let fragment = recover_periodic_chain_fragment(&chain)?;
        assert_eq!(
            fragment.source,
            CadFragmentSource::PeriodicChain {
                solid_id: 42,
                sites: 4,
            }
        );
        let CadNode::PeriodicChain(node) = fragment.model.node(fragment.root)? else {
            bail!("expected fused periodic-chain root");
        };
        assert_eq!(node.sites, 4);
        assert_eq!(node.pitch_mm, 2.54);
        assert_eq!(node.adjacent_site_edges_per_boundary, 2);
        assert_eq!(node.interior_site_face_count, 2);
        assert_eq!(node.interior_gap_face_count, 1);
        assert_eq!(
            node.site_face_count_overrides,
            vec![IndexedCount { index: 3, count: 3 }]
        );
        assert_eq!(node.gap_face_count_overrides, Vec::<IndexedCount>::new());
        assert_eq!(node.stretch_face_count, 1);
        assert_eq!(
            fragment.model.provenance[&fragment.root].proof,
            ProofStatus::StructurallyProven
        );
        assert_eq!(
            fragment.model.provenance[&fragment.root].max_residual_mm,
            None
        );
        assert!(fragment.model.complexity_score(fragment.root)? < 200);
        Ok(())
    }

    #[test]
    fn periodic_chain_recovery_fails_closed_on_incomplete_or_ambiguous_proof() -> Result<()> {
        let mut chain = periodic_chain_fixture();
        chain.read_only_proven = false;
        assert!(recover_periodic_chain_fragment(&chain).is_err());

        let mut chain = periodic_chain_fixture();
        chain.site_face_ids[1][0] = chain.site_face_ids[0][0];
        assert!(recover_periodic_chain_fragment(&chain).is_err());

        let mut chain = periodic_chain_fixture();
        chain.adjacent_site_edge_counts[1] = 3;
        assert!(recover_periodic_chain_fragment(&chain).is_err());

        let mut diagnostic = periodic_chain_fixture();
        diagnostic.read_only_proven = false;
        let fragments = recover_periodic_chain_fragments(&[periodic_chain_fixture(), diagnostic])?;
        assert_eq!(fragments.len(), 1);
        Ok(())
    }
}
