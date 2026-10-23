use crate::cad_ir::{
    BooleanOp, CadModel, CadNode, FusedPeriodicChain, IndexedCount, NodeId, ProofStatus, Provenance,
};
use crate::cad_kernel::CadKernel;
use crate::math3::dot;
use crate::periodic_chains::PeriodicChainPattern;
use crate::step_graph::{build_index, entity_id, simple_record};
use crate::step_io::ParsedExchange;
use anyhow::{Result, bail};
use serde::Serialize;

mod solid;
pub use solid::{
    recover_closed_round_sweep_fragment, recover_closed_round_sweep_fragments,
    recover_open_rectangular_sweep_fragment, recover_open_rectangular_sweep_fragments,
    recover_radial_slot_revolution_fragment, recover_radial_slot_revolution_fragments,
    recover_revolved_round_tail_fragment, recover_revolved_round_tail_fragments,
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
    SolidSweep {
        solid_id: u64,
        face_ids: Vec<u64>,
        closed: bool,
    },
    RevolvedRoundTail {
        solid_id: u64,
        head_face_ids: Vec<u64>,
        tail_face_ids: Vec<u64>,
    },
    PeriodicChain {
        solid_id: u64,
        sites: usize,
    },
    ExtrusionCoreUnion {
        solid_id: u64,
        cap_face_ids: Vec<[u64; 2]>,
    },
    BoundaryFeatureComposite {
        solid_id: u64,
        passes: usize,
        feature_families: usize,
        feature_instances: usize,
    },
}

/// Compose two or more already-recovered CAD fragments into a union model.
///
/// This helper only composes executable CAD models; it deliberately does not create a
/// `CadFragmentSource` claim for the union. Callers must independently prove that the input
/// fragments form the intended source solid before admitting the composite into Complete IR.
///
/// # Errors
/// Returns an error if fewer than two fragments are provided, an input model is invalid, or a
/// fragment root cannot be remapped into the composed model.
pub fn compose_union_model(fragments: &[CadFragment]) -> Result<(CadModel, NodeId)> {
    if fragments.len() < 2 {
        bail!("a CAD union composite needs at least two fragments");
    }

    let mut model = CadModel::new();
    let mut children = Vec::with_capacity(fragments.len());
    for fragment in fragments {
        fragment.model.validate()?;
        let offset = model.append_model(&fragment.model)?;
        let root = NodeId(
            fragment
                .root
                .0
                .checked_add(offset)
                .ok_or_else(|| anyhow::anyhow!("CAD root id overflow while composing union"))?,
        );
        model.node(root)?;
        children.push(root);
    }

    let root = add_bounded_union_tree(&mut model, children);
    model.add_root(root)?;
    model.validate()?;
    Ok((model, root))
}

const MAX_FLAT_UNION_CHILDREN: usize = 8;
const AXIS_GROUP_DOT_TOLERANCE: f64 = 1.0e-10;

fn add_bounded_union_tree(model: &mut CadModel, children: Vec<NodeId>) -> NodeId {
    if children.len() == 1 {
        return children[0];
    }
    if children.len() <= MAX_FLAT_UNION_CHILDREN {
        return model.add_node(CadNode::Boolean {
            op: BooleanOp::Union,
            children,
        });
    }

    let mut level = children;
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            if let [first, second] = pair {
                next.push(model.add_node(CadNode::Boolean {
                    op: BooleanOp::Union,
                    children: vec![*first, *second],
                }));
            } else {
                next.push(pair[0]);
            }
        }
        level = next;
    }
    level[0]
}

fn compose_extrusion_core_union_model(
    fragments: &[CadFragment],
    axes: &[[f64; 3]],
) -> Result<(CadModel, NodeId)> {
    if fragments.len() != axes.len() || fragments.len() < 2 {
        bail!("extrusion-core union composition needs matching fragments and axes");
    }
    if fragments.len() <= MAX_FLAT_UNION_CHILDREN {
        return compose_union_model(fragments);
    }

    let mut model = CadModel::new();
    let mut groups = Vec::<([f64; 3], Vec<NodeId>)>::new();
    for (fragment, &axis) in fragments.iter().zip(axes) {
        fragment.model.validate()?;
        let offset = model.append_model(&fragment.model)?;
        let root = NodeId(
            fragment
                .root
                .0
                .checked_add(offset)
                .ok_or_else(|| anyhow::anyhow!("CAD root id overflow while composing union"))?,
        );
        model.node(root)?;

        if let Some((_, roots)) = groups
            .iter_mut()
            .find(|(group_axis, _)| dot(*group_axis, axis).abs() >= 1.0 - AXIS_GROUP_DOT_TOLERANCE)
        {
            roots.push(root);
        } else {
            groups.push((axis, vec![root]));
        }
    }

    let group_roots = groups
        .into_iter()
        .map(|(_, roots)| add_bounded_union_tree(&mut model, roots))
        .collect::<Vec<_>>();
    let root = add_bounded_union_tree(&mut model, group_roots);
    model.add_root(root)?;
    model.validate()?;
    Ok((model, root))
}
/// Recover and admit a union of sharp extrusion cores only when an independent
/// analytic boundary proof establishes that the evaluated union is the source solid.
///
/// Cap-pair recovery is source-derived and fail-closed. The supplied CAD kernel
/// merely proposes executable geometry: after evaluation and STEP serialization,
/// the result must contain exactly one manifold solid and its complete oriented
/// plane/cylinder boundary signature must equal the source signature. Unsupported
/// carriers, unproven edge loci, disconnected kernel results, or any signature
/// mismatch return `Ok(None)` rather than manufacturing semantic IR.
///
/// # Errors
/// Returns an error for malformed API input, STEP parsing failures, invalid CAD IR,
/// or a backend evaluation/serialization failure.
pub fn recover_proven_extrusion_core_union_fragment<K: CadKernel>(
    source_step: &[u8],
    solid_id: u64,
    cap_face_ids: &[[u64; 2]],
    kernel: &K,
) -> Result<Option<CadFragment>> {
    if solid_id == 0 || cap_face_ids.len() < 2 {
        bail!("extrusion-core union admission needs one solid and at least two cap pairs");
    }

    let mut canonical_pairs = cap_face_ids
        .iter()
        .copied()
        .map(|mut pair| {
            pair.sort_unstable();
            pair
        })
        .collect::<Vec<_>>();
    if canonical_pairs
        .iter()
        .any(|pair| pair[0] == 0 || pair[0] == pair[1])
    {
        bail!("extrusion-core union cap pairs must contain two distinct nonzero face ids");
    }
    canonical_pairs.sort_unstable();
    if canonical_pairs.windows(2).any(|pair| pair[0] == pair[1]) {
        bail!("extrusion-core union cap pairs must be unique");
    }

    let source = ParsedExchange::parse(source_step)?.exchange;
    if source.data.len() != 1 {
        bail!("extrusion-core union admission currently requires exactly one DATA section");
    }
    let entities = &source.data[0].entities;
    let index = build_index(entities);
    let Some(source_face_ids) = crate::brep::solid_face_ids(solid_id, entities, &index) else {
        return Ok(None);
    };

    let mut fragments = Vec::with_capacity(canonical_pairs.len());
    let mut axes = Vec::with_capacity(canonical_pairs.len());
    let mut max_residual_mm = crate::boundary_proof::ANALYTIC_BOUNDARY_PROOF_TOL_MM;
    for &pair in &canonical_pairs {
        let Some(core) = crate::solid_extrusions::recover_extrusion_core_from_cap_faces_with_index(
            solid_id, pair, entities, &index,
        ) else {
            return Ok(None);
        };
        max_residual_mm = max_residual_mm.max(core.max_residual_mm);
        axes.push(core.z_axis);
        fragments.push(recover_solid_extrusion_fragment(&core)?);
    }

    let (mut model, root) = compose_extrusion_core_union_model(&fragments, &axes)?;
    let evaluated = kernel.evaluate(&model, root)?;
    let summary = kernel.summarize(&evaluated);
    if !summary.geometrically_consistent || summary.shells != 1 {
        return Ok(None);
    }

    let rebuilt_step = kernel.to_step(&evaluated)?;
    let rebuilt = ParsedExchange::parse(rebuilt_step.as_bytes())?.exchange;
    if rebuilt.data.len() != 1 {
        return Ok(None);
    }
    let rebuilt_entities = &rebuilt.data[0].entities;
    let rebuilt_solid_ids = rebuilt_entities
        .iter()
        .filter_map(|entity| {
            simple_record(entity)
                .is_some_and(|record| record.name == "MANIFOLD_SOLID_BREP")
                .then_some(entity_id(entity))
        })
        .collect::<Vec<_>>();
    let [rebuilt_solid_id] = rebuilt_solid_ids.as_slice() else {
        return Ok(None);
    };

    let comparison = crate::boundary_proof::compare_analytic_boundaries(
        solid_id,
        entities,
        *rebuilt_solid_id,
        rebuilt_entities,
    );
    if !comparison.equivalent {
        return Ok(None);
    }

    let mut source_entity_ids = Vec::with_capacity(source_face_ids.len() + 1);
    source_entity_ids.push(solid_id);
    source_entity_ids.extend(source_face_ids);
    source_entity_ids.sort_unstable();
    source_entity_ids.dedup();
    model.set_provenance(
        root,
        Provenance {
            source_entity_ids,
            proof: ProofStatus::WithinTolerance,
            max_residual_mm: Some(max_residual_mm),
        },
    )?;
    model.validate()?;

    Ok(Some(CadFragment {
        source: CadFragmentSource::ExtrusionCoreUnion {
            solid_id,
            cap_face_ids: canonical_pairs,
        },
        model,
        root,
    }))
}

/// Detect and admit one exact union of strict extrusion-core candidates.
///
/// The search is intentionally bounded. It first tries every strict source-derived
/// core. If that does not admit and there are at most eight candidates, it tries
/// each single-core omission. Exactly one successful omission is required; this
/// avoids exponential subset search and arbitrary construction choices. Candidate
/// selection is never semantic authority: every attempted set still passes through
/// [`recover_proven_extrusion_core_union_fragment`] and its complete analytic
/// source/rebuild boundary proof.
///
/// # Errors
/// Returns an error for malformed STEP input or when the canonical full-set
/// backend evaluation fails and no exact single-omission alternative succeeds.
/// Failures from exploratory omission attempts do not turn an otherwise clean
/// semantic refusal into an error.
pub fn recover_auto_proven_extrusion_core_union_fragment<K: CadKernel>(
    source_step: &[u8],
    solid_id: u64,
    kernel: &K,
) -> Result<Option<CadFragment>> {
    const MAX_SINGLE_OMISSION_CANDIDATES: usize = 8;

    let source = ParsedExchange::parse(source_step)?.exchange;
    if source.data.len() != 1 {
        bail!("automatic extrusion-core union recovery requires exactly one DATA section");
    }
    let entities = &source.data[0].entities;
    let index = build_index(entities);
    let mut cap_pairs =
        crate::solid_extrusions::detect_extrusion_core_candidates_with_index(entities, &index)
            .into_iter()
            .filter(|candidate| candidate.solid_id == solid_id)
            .map(|candidate| {
                let mut pair = candidate.cap_face_ids;
                pair.sort_unstable();
                pair
            })
            .collect::<Vec<_>>();
    cap_pairs.sort_unstable();
    cap_pairs.dedup();
    if cap_pairs.len() < 2 {
        return Ok(None);
    }

    let remembered_error = match recover_proven_extrusion_core_union_fragment(
        source_step,
        solid_id,
        &cap_pairs,
        kernel,
    ) {
        Ok(Some(fragment)) => return Ok(Some(fragment)),
        Ok(None) => None,
        Err(error) => Some(error),
    };

    if !(3..=MAX_SINGLE_OMISSION_CANDIDATES).contains(&cap_pairs.len()) {
        return remembered_error.map_or(Ok(None), Err);
    }

    let mut admitted = None;
    for omitted in 0..cap_pairs.len() {
        let subset = cap_pairs
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != omitted)
            .map(|(_, pair)| *pair)
            .collect::<Vec<_>>();
        match recover_proven_extrusion_core_union_fragment(source_step, solid_id, &subset, kernel) {
            Ok(Some(fragment)) => {
                if admitted.is_some() {
                    // Multiple exact constructions are semantically valid but this
                    // bounded selector has no source-derived reason to prefer one.
                    return Ok(None);
                }
                admitted = Some(fragment);
            }
            Ok(None) => {}
            Err(_error) => {
                // Omission attempts are optional recall probes. A kernel failure
                // here must not turn a clean full-set semantic refusal into an error.
            }
        }
    }

    admitted.map_or_else(
        || remembered_error.map_or(Ok(None), Err),
        |fragment| Ok(Some(fragment)),
    )
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

    fn primitive_fragment(solid_id: u64) -> Result<CadFragment> {
        let mut model = CadModel::new();
        let root = model.add_node(CadNode::Primitive(crate::cad_ir::Primitive::Box {
            size_mm: [1.0, 1.0, 1.0],
        }));
        model.add_root(root)?;
        Ok(CadFragment {
            source: CadFragmentSource::SolidExtrusion {
                solid_id,
                cap_face_ids: [solid_id + 1, solid_id + 2],
                side_face_ids: Vec::new(),
            },
            model,
            root,
        })
    }

    #[test]
    fn union_model_composes_fragment_roots_without_claiming_semantics() -> Result<()> {
        let fragments = [primitive_fragment(10)?, primitive_fragment(20)?];
        let (model, root) = compose_union_model(&fragments)?;
        let CadNode::Boolean { op, children } = model.node(root)? else {
            bail!("expected union root");
        };
        assert_eq!(*op, BooleanOp::Union);
        assert_eq!(children.as_slice(), &[NodeId(0), NodeId(1)]);
        assert_eq!(model.roots.as_slice(), &[root]);
        Ok(())
    }

    #[test]
    fn union_model_keeps_bounded_fragment_roots_flat() -> Result<()> {
        let fragments = [
            primitive_fragment(10)?,
            primitive_fragment(20)?,
            primitive_fragment(30)?,
            primitive_fragment(40)?,
        ];
        let (model, root) = compose_union_model(&fragments)?;
        assert_eq!(root, NodeId(4));
        let CadNode::Boolean { op, children } = model.node(root)? else {
            bail!("expected flat union node");
        };
        assert_eq!(*op, BooleanOp::Union);
        assert_eq!(
            children.as_slice(),
            &[NodeId(0), NodeId(1), NodeId(2), NodeId(3)]
        );
        Ok(())
    }

    #[test]
    fn union_model_balances_large_fragment_sets() -> Result<()> {
        let fragments = (0..9)
            .map(|index| primitive_fragment(index))
            .collect::<Result<Vec<_>>>()?;
        let (model, root) = compose_union_model(&fragments)?;
        assert_eq!(root, NodeId(16));
        for node_index in 9..=16 {
            let CadNode::Boolean { op, children } = model.node(NodeId(node_index))? else {
                bail!("expected binary union node");
            };
            assert_eq!(*op, BooleanOp::Union);
            assert_eq!(children.len(), 2);
        }
        Ok(())
    }

    #[test]
    fn extrusion_core_union_groups_large_sets_by_axis() -> Result<()> {
        let fragments = (0..9)
            .map(|index| primitive_fragment(index))
            .collect::<Result<Vec<_>>>()?;
        let axes = vec![[0.0, 0.0, 1.0]; 5]
            .into_iter()
            .chain(vec![[1.0, 0.0, 0.0]; 4])
            .collect::<Vec<_>>();
        let (model, root) = compose_extrusion_core_union_model(&fragments, &axes)?;
        assert_eq!(root, NodeId(11));
        let CadNode::Boolean { op, children } = model.node(root)? else {
            bail!("expected grouped union root");
        };
        assert_eq!(*op, BooleanOp::Union);
        assert_eq!(children.as_slice(), &[NodeId(9), NodeId(10)]);
        for (node, expected_len) in [(NodeId(9), 5), (NodeId(10), 4)] {
            let CadNode::Boolean { op, children } = model.node(node)? else {
                bail!("expected axis-group union node");
            };
            assert_eq!(*op, BooleanOp::Union);
            assert_eq!(children.len(), expected_len);
        }
        Ok(())
    }

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
