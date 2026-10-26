use crate::cad_ir::{
    BooleanOp, CadModel, CadNode, EdgeSelector, EdgeTreatmentEdge, EdgeTreatmentProfile, NodeId,
    ProofStatus, Provenance, RigidTransform,
};
use crate::cad_kernel::CadKernel;
use crate::cad_recovery::{
    CadFragment, CadFragmentSource, recover_auto_proven_extrusion_core_union_fragment,
    recover_closed_round_sweep_fragments, recover_open_rectangular_sweep_fragments,
    recover_periodic_chain_fragments, recover_radial_slot_revolution_fragments,
    recover_revolved_round_tail_fragments, recover_solid_extrusion_fragment,
    recover_solid_extrusion_fragments, recover_solid_revolution_fragments,
};
use crate::patterns::InstancePattern;
use crate::planar_features::{BoundaryFeatureInstanceEvidence, FeaturePolarity};
use crate::solid_revolutions::SolidSurfaceSignature;
use crate::{Options, OutputProfile, Stats};
use anyhow::{Result, bail};
use ruststep::ast::{EntityInstance, Exchange};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

pub const COMPLETE_IR_SCHEMA: &str = "step-redox-complete-ir-v8";
pub const KERNEL_COMPLETE_IR_SCHEMA: &str = "step-redox-complete-ir-v9";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SurfaceSignatureSummary {
    pub solid_id: u64,
    pub face_count: usize,
    pub support_counts: BTreeMap<String, usize>,
    pub unique_edge_count: usize,
    pub edge_use_count: usize,
    pub closed_two_manifold: bool,
}

impl From<&SolidSurfaceSignature> for SurfaceSignatureSummary {
    fn from(value: &SolidSurfaceSignature) -> Self {
        Self {
            solid_id: value.solid_id,
            face_count: value.face_count,
            support_counts: value.support_counts.clone(),
            unique_edge_count: value.unique_edge_count,
            edge_use_count: value.edge_use_count,
            closed_two_manifold: value.closed_two_manifold,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SourceSolidSummary {
    pub solid_id: u64,
    pub faces: usize,
    pub edges: usize,
    pub closure_entities: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BodyScanSummary {
    pub solids: Vec<SourceSolidSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelRecoveryFailure {
    pub solid_id: u64,
    pub error: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompleteIr {
    pub schema: &'static str,
    pub idempotent_compact_rewrite: bool,
    pub compact_second_pass_stats: Stats,
    pub surface_signature_summaries: Vec<SurfaceSignatureSummary>,
    pub instance_pattern_diagnostics: Vec<InstancePattern>,
    pub solid_extrusions: Vec<CadFragment>,
    pub solid_revolutions: Vec<CadFragment>,
    pub radial_slot_revolutions: Vec<CadFragment>,
    pub solid_sweeps: Vec<CadFragment>,
    pub periodic_chains: Vec<CadFragment>,
    pub boundary_feature_composites: Vec<CadFragment>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub extrusion_core_unions: Vec<CadFragment>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub kernel_recovery_failures: Vec<KernelRecoveryFailure>,
    pub unrecovered_solids: Vec<SourceSolidSummary>,
    pub body_scan: BodyScanSummary,
    pub body_count: usize,
    pub constructively_recovered_solid_count: usize,
    pub semantically_recovered_solid_count: usize,
    pub unrecovered_solid_count: usize,
}

#[derive(Debug, Clone)]
struct RecoveredBoundaryFeatureFamily {
    pass_index: usize,
    polarity: FeaturePolarity,
    tool: CadFragment,
    source_member: BoundaryFeatureInstanceEvidence,
    members: Vec<BoundaryFeatureInstanceEvidence>,
}

const MAX_BOUNDARY_FEATURE_DECOMPOSITION_PASSES: usize = 32;

/// Build the compact complete semantic IR for a normalized STEP exchange.
///
/// This report deliberately separates compact canonical IR from source diagnostics.
/// Every recovered manifold solid is represented by the cheapest proven whole-solid
/// semantic fragment. Bodies without a proven constructive representation are listed
/// explicitly in `unrecovered_solids` and never enter `cad_ir`. Instance-pattern
/// evidence remains diagnostic until its repeated child can be linked to proven CAD IR.
///
/// # Errors
/// Returns an error if STEP parsing fails, duplicate solid IDs make body ownership
/// ambiguous, or a detector's proven result cannot be converted into valid CAD IR.
pub fn recover_complete_ir_bytes(input: &[u8]) -> Result<CompleteIr> {
    let session = crate::AnalysisSession::from_bytes(input)?;
    session.complete_ir()
}

/// Build Complete IR and opportunistically recover otherwise-unrecovered solids
/// through a supplied CAD kernel plus the independent analytic boundary proof.
///
/// The parser-only [`recover_complete_ir_bytes`] path remains schema v8 and never
/// invokes a CAD backend. This explicit kernel-backed path reports schema v9,
/// records admitted extrusion-core unions separately, and leaves any unsupported
/// or unproven body in `unrecovered_solids`. Backend failures are diagnostic per
/// solid rather than aborting the rest of the report.
///
/// # Errors
/// Returns an error if the parser-only Complete IR pass itself fails.
pub fn recover_complete_ir_with_kernel_bytes<K: CadKernel>(
    input: &[u8],
    kernel: &K,
) -> Result<CompleteIr> {
    let report = recover_complete_ir_bytes(input)?;
    Ok(apply_kernel_recovery(report, |solid_id| {
        recover_auto_proven_extrusion_core_union_fragment(input, solid_id, kernel)
    }))
}

fn apply_kernel_recovery<F>(mut report: CompleteIr, mut recover: F) -> CompleteIr
where
    F: FnMut(u64) -> Result<Option<CadFragment>>,
{
    let candidates = report.unrecovered_solids.clone();
    let mut recovered = Vec::new();
    let mut failures = Vec::new();

    for solid in candidates {
        match recover(solid.solid_id) {
            Ok(Some(fragment)) if source_solid_id(&fragment.source) == solid.solid_id => {
                recovered.push(fragment);
            }
            Ok(Some(fragment)) => failures.push(KernelRecoveryFailure {
                solid_id: solid.solid_id,
                error: format!(
                    "kernel recovery returned solid #{} while recovering #{}",
                    source_solid_id(&fragment.source),
                    solid.solid_id
                ),
            }),
            Ok(None) => {}
            Err(error) => failures.push(KernelRecoveryFailure {
                solid_id: solid.solid_id,
                error: error.to_string(),
            }),
        }
    }

    let recovered_ids = recovered
        .iter()
        .map(|fragment| source_solid_id(&fragment.source))
        .collect::<BTreeSet<_>>();
    report
        .unrecovered_solids
        .retain(|solid| !recovered_ids.contains(&solid.solid_id));
    report.constructively_recovered_solid_count += recovered.len();
    report.semantically_recovered_solid_count += recovered.len();
    report.unrecovered_solid_count = report.unrecovered_solids.len();
    report.extrusion_core_unions = recovered;
    report.kernel_recovery_failures = failures;
    report.schema = KERNEL_COMPLETE_IR_SCHEMA;
    report
}

pub(crate) fn recover_complete_ir_exchange(
    exchange: &Exchange,
    input: &[u8],
) -> Result<CompleteIr> {
    let mut signatures = Vec::new();
    let mut extrusions = Vec::new();
    let mut revolutions = Vec::new();
    let mut radial_slots = Vec::new();
    let mut closed_sweeps = Vec::new();
    let mut revolved_round_tails = Vec::new();
    let mut open_sweeps = Vec::new();
    let mut periodic_chains = Vec::new();
    let mut patterns = Vec::new();
    let mut body_scan = Vec::new();

    for section in &exchange.data {
        let index = crate::step_graph::build_index(&section.entities);
        signatures.extend(
            crate::solid_revolutions::detect_solid_surface_signatures_with_index(
                &section.entities,
                &index,
            ),
        );
        extrusions.extend(crate::solid_extrusions::detect_solid_extrusions_with_index(
            &section.entities,
            &index,
        ));
        revolutions.extend(
            crate::solid_revolutions::detect_solid_revolutions_with_index(
                &section.entities,
                &index,
            ),
        );
        radial_slots.extend(
            crate::solid_revolutions::detect_radial_slot_revolutions_with_index(
                &section.entities,
                &index,
            ),
        );
        closed_sweeps.extend(crate::solid_sweeps::detect_closed_round_sweeps(
            &section.entities,
        ));
        revolved_round_tails.extend(crate::solid_sweeps::detect_revolved_round_tails(
            &section.entities,
        ));
        open_sweeps.extend(crate::solid_sweeps::detect_open_rectangular_sweeps(
            &section.entities,
        ));
        periodic_chains.extend(crate::periodic_chains::detect_periodic_chains_with_index(
            &section.entities,
            &index,
        ));
        patterns.extend(crate::patterns::detect_instance_patterns_with_index(
            &section.entities,
            &index,
            1.0e-7,
            4,
        ));
        body_scan.extend(scan_source_solids(&section.entities, &index));
    }

    body_scan.sort_by_key(|solid| solid.solid_id);
    for pair in body_scan.windows(2) {
        if pair[0].solid_id == pair[1].solid_id {
            bail!(
                "complete IR cannot disambiguate duplicate solid id #{} across DATA sections",
                pair[0].solid_id
            );
        }
    }

    signatures.sort_by_key(|signature| signature.solid_id);
    let surface_signature_summaries = signatures
        .iter()
        .map(SurfaceSignatureSummary::from)
        .collect::<Vec<_>>();

    let instance_pattern_diagnostics = patterns;
    let solid_extrusions = recover_solid_extrusion_fragments(&extrusions)?;
    let solid_revolutions = recover_solid_revolution_fragments(&revolutions)?;
    let radial_slot_revolutions = recover_radial_slot_revolution_fragments(&radial_slots)?;
    let mut solid_sweeps = recover_closed_round_sweep_fragments(&closed_sweeps)?;
    solid_sweeps.extend(recover_revolved_round_tail_fragments(
        &revolved_round_tails,
    )?);
    solid_sweeps.extend(recover_open_rectangular_sweep_fragments(&open_sweeps)?);
    let periodic_chains = recover_periodic_chain_fragments(&periodic_chains)?;

    let base_candidates = solid_extrusions
        .iter()
        .chain(&solid_revolutions)
        .chain(&radial_slot_revolutions)
        .chain(&solid_sweeps)
        .chain(&periodic_chains)
        .collect::<Vec<_>>();
    let base_selected = select_cheapest_whole_solid_fragments(&base_candidates)?;
    let boundary_feature_targets = body_scan
        .iter()
        .filter_map(|solid| {
            base_selected
                .get(&solid.solid_id)
                .is_none_or(|fragment| !is_constructive_source(&fragment.source))
                .then_some(solid.solid_id)
        })
        .collect::<BTreeSet<_>>();
    let mut boundary_feature_composites = Vec::new();
    if !boundary_feature_targets.is_empty() {
        for section in &exchange.data {
            boundary_feature_composites.extend(recover_boundary_feature_composites_for_section(
                &section.entities,
                &boundary_feature_targets,
            )?);
        }
    }

    let pre_treatment_candidates = base_candidates
        .iter()
        .copied()
        .chain(boundary_feature_composites.iter())
        .collect::<Vec<_>>();
    let pre_treatment_selected = select_cheapest_whole_solid_fragments(&pre_treatment_candidates)?;
    let treatment_targets = body_scan
        .iter()
        .filter_map(|solid| {
            pre_treatment_selected
                .get(&solid.solid_id)
                .is_none_or(|fragment| !is_constructive_source(&fragment.source))
                .then_some(solid.solid_id)
        })
        .collect::<BTreeSet<_>>();
    let mut treated_extrusions = Vec::new();
    if !treatment_targets.is_empty() {
        let dominant_extrusions =
            crate::dominant_extrusions::detect_dominant_extrusion_candidates(&signatures);
        for section in &exchange.data {
            let index = crate::step_graph::build_index(&section.entities);
            let section_candidates = dominant_extrusions
                .iter()
                .filter(|candidate| index.contains_key(&candidate.solid_id))
                .cloned()
                .collect::<Vec<_>>();
            treated_extrusions.extend(recover_edge_treated_extrusion_fragments_from_candidates(
                &section.entities,
                &index,
                &section_candidates,
                Some(&treatment_targets),
            )?);
        }
    }

    let whole_solid_candidates = pre_treatment_candidates
        .into_iter()
        .chain(treated_extrusions.iter())
        .collect::<Vec<_>>();
    let selected = select_cheapest_whole_solid_fragments(&whole_solid_candidates)?;
    let recovered_ids = selected.keys().copied().collect::<BTreeSet<_>>();

    let by_id = body_scan
        .iter()
        .map(|solid| (solid.solid_id, solid))
        .collect::<HashMap<_, _>>();
    for solid_id in &recovered_ids {
        if !by_id.contains_key(solid_id) {
            bail!("recovered CAD fragment refers to missing solid #{solid_id}");
        }
    }

    let unrecovered_solids = body_scan
        .iter()
        .filter(|solid| !recovered_ids.contains(&solid.solid_id))
        .cloned()
        .collect::<Vec<_>>();

    let selected_constructive = selected
        .values()
        .filter(|fragment| is_constructive_source(&fragment.source))
        .count();

    let second = crate::clean_bytes(input, &Options::for_profile(OutputProfile::Compact))?;
    let idempotent_compact_rewrite = second.bytes == input;

    Ok(CompleteIr {
        schema: COMPLETE_IR_SCHEMA,
        idempotent_compact_rewrite,
        compact_second_pass_stats: second.stats,
        surface_signature_summaries,
        instance_pattern_diagnostics,
        solid_extrusions: selected_category(&selected, |source| {
            matches!(source, CadFragmentSource::SolidExtrusion { .. })
        }),
        solid_revolutions: selected_category(&selected, |source| {
            matches!(source, CadFragmentSource::SolidRevolution { .. })
        }),
        radial_slot_revolutions: selected_category(&selected, |source| {
            matches!(source, CadFragmentSource::RadialSlotRevolution { .. })
        }),
        solid_sweeps: selected_category(&selected, |source| {
            matches!(
                source,
                CadFragmentSource::SolidSweep { .. } | CadFragmentSource::RevolvedRoundTail { .. }
            )
        }),
        periodic_chains: selected_category(&selected, |source| {
            matches!(source, CadFragmentSource::PeriodicChain { .. })
        }),
        boundary_feature_composites: selected_category(&selected, |source| {
            matches!(source, CadFragmentSource::BoundaryFeatureComposite { .. })
        }),
        extrusion_core_unions: Vec::new(),
        kernel_recovery_failures: Vec::new(),
        unrecovered_solid_count: unrecovered_solids.len(),
        body_count: body_scan.len(),
        constructively_recovered_solid_count: selected_constructive,
        semantically_recovered_solid_count: selected.len(),
        body_scan: BodyScanSummary { solids: body_scan },
        unrecovered_solids,
    })
}

fn recover_boundary_feature_composites_for_section(
    entities: &[EntityInstance],
    target_solid_ids: &BTreeSet<u64>,
) -> Result<Vec<CadFragment>> {
    let initial_index = crate::step_graph::build_index(entities);
    let mut keyed_targets = BTreeMap::<crate::shape_identity::ShapeKey, Vec<_>>::new();
    let mut representative_targets = BTreeSet::new();
    let mut recovery_groups = HashMap::new();
    for &solid_id in target_solid_ids {
        if let Some(identity) =
            crate::shape_identity::solid_identity(solid_id, entities, &initial_index)
        {
            keyed_targets
                .entry(identity.key.clone())
                .or_default()
                .push(identity);
        } else {
            representative_targets.insert(solid_id);
        }
    }
    for (_, mut group) in keyed_targets {
        group.sort_by_key(|identity| identity.root);
        let Some(representative) = group.first().map(|identity| identity.root) else {
            continue;
        };
        representative_targets.insert(representative);
        recovery_groups.insert(representative, group);
    }

    let target_shells = representative_targets
        .iter()
        .filter_map(|&solid_id| {
            crate::brep::manifold_shell(solid_id, entities, &initial_index)
                .map(|shell_id| (shell_id, solid_id))
        })
        .collect::<HashMap<_, _>>();
    if target_shells.is_empty() {
        return Ok(Vec::new());
    }

    let mut working = entities.to_vec();
    let mut recovered = BTreeMap::<u64, Vec<RecoveredBoundaryFeatureFamily>>::new();
    for pass_index in 0..MAX_BOUNDARY_FEATURE_DECOMPOSITION_PASSES {
        if !recover_boundary_feature_pass(&mut working, &target_shells, &mut recovered, pass_index)?
        {
            break;
        }
    }

    let mut host_candidates = recover_constructive_fragments_from_entities(&working)?;
    let working_index = crate::step_graph::build_index(&working);
    let host_target_ids = target_shells.values().copied().collect::<BTreeSet<_>>();
    host_candidates.extend(recover_edge_treated_extrusion_fragments(
        &working,
        &working_index,
        Some(&host_target_ids),
    )?);
    let host_refs = host_candidates.iter().collect::<Vec<_>>();
    let selected_hosts = select_cheapest_whole_solid_fragments(&host_refs)?;
    let mut out = Vec::new();
    for (solid_id, families) in recovered {
        if families.is_empty() {
            continue;
        }
        let Some(host) = selected_hosts.get(&solid_id) else {
            continue;
        };
        if !is_constructive_source(&host.source) {
            continue;
        }
        out.push(compose_boundary_feature_fragment(
            solid_id, host, &families,
        )?);
    }

    let mut expanded = Vec::new();
    for fragment in out {
        let representative = source_solid_id(&fragment.source);
        if let Some(group) = recovery_groups.get(&representative)
            && let Some(source_identity) = group.first()
        {
            for target_identity in group.iter().skip(1) {
                if let Some(instance) = instantiate_boundary_feature_fragment(
                    &fragment,
                    source_identity,
                    target_identity,
                    entities,
                    &initial_index,
                )? {
                    expanded.push(instance);
                }
            }
        }
        expanded.push(fragment);
    }
    Ok(expanded)
}

fn instantiate_boundary_feature_fragment(
    fragment: &CadFragment,
    source_identity: &crate::shape_identity::SolidIdentity,
    target_identity: &crate::shape_identity::SolidIdentity,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Result<Option<CadFragment>> {
    let CadFragmentSource::BoundaryFeatureComposite {
        passes,
        feature_families,
        feature_instances,
        ..
    } = fragment.source
    else {
        return Ok(None);
    };
    let Some(quarter) = crate::shape_identity::relative_z90_quarter(
        source_identity,
        target_identity,
        entities,
        index,
    ) else {
        return Ok(None);
    };
    let translation =
        relative_quarter_translation(source_identity.center, target_identity.center, quarter);
    let mut model = fragment.model.clone();
    model.roots.clear();
    let root = model.add_node(CadNode::Transform {
        transform: quarter_turn_transform(quarter, translation),
        child: fragment.root,
    });
    model.set_provenance(
        root,
        Provenance {
            source_entity_ids: vec![target_identity.root],
            proof: ProofStatus::StructurallyProven,
            max_residual_mm: None,
        },
    )?;
    model.add_root(root)?;
    model.validate()?;
    Ok(Some(CadFragment {
        source: CadFragmentSource::BoundaryFeatureComposite {
            solid_id: target_identity.root,
            passes,
            feature_families,
            feature_instances,
        },
        model,
        root,
    }))
}

fn relative_quarter_translation(
    source_center: [f64; 3],
    target_center: [f64; 3],
    quarter: u8,
) -> [f64; 3] {
    let (x, y) = match quarter % 4 {
        0 => (source_center[0], source_center[1]),
        1 => (-source_center[1], source_center[0]),
        2 => (-source_center[0], -source_center[1]),
        _ => (source_center[1], -source_center[0]),
    };
    [
        target_center[0] - x,
        target_center[1] - y,
        target_center[2] - source_center[2],
    ]
}

fn recover_boundary_feature_pass(
    working: &mut [EntityInstance],
    target_shells: &HashMap<u64, u64>,
    recovered: &mut BTreeMap<u64, Vec<RecoveredBoundaryFeatureFamily>>,
    pass_index: usize,
) -> Result<bool> {
    let target_shell_ids = target_shells.keys().copied().collect::<HashSet<_>>();
    let report =
        crate::planar_features::diagnose_boundary_features_for_shells(working, &target_shell_ids);
    let target_family_indices =
        crate::planar_features::leaf_feature_family_indices(&report.families)
            .into_iter()
            .filter(|&index| target_shells.contains_key(&report.families[index].shell_id))
            .collect::<Vec<_>>();
    if target_family_indices.is_empty() {
        return Ok(false);
    }

    let mut tool_entities = working.to_vec();
    let (tool_report, materialized) =
        crate::planar_features::materialize_boundary_feature_tools_for_analysis_for_shells(
            &mut tool_entities,
            1,
            &target_shell_ids,
        )?;
    if tool_report.families.len() != report.families.len() {
        bail!("boundary-feature tool materialization changed family indexing");
    }
    let tool_candidates = recover_constructive_fragments_from_entities(&tool_entities)?;
    let tool_refs = tool_candidates.iter().collect::<Vec<_>>();
    let selected_tools = select_cheapest_whole_solid_fragments(&tool_refs)?;
    let working_index = crate::step_graph::build_index(working);
    let mut peelable_family_indices = Vec::new();

    for family_index in target_family_indices {
        let family = &report.families[family_index];
        let Some(&source_solid_id) = target_shells.get(&family.shell_id) else {
            continue;
        };
        let Some(source_member) = family.members.first().cloned() else {
            continue;
        };
        let Some(materialization) = materialized
            .iter()
            .find(|tool| tool.family_index == family_index)
        else {
            continue;
        };
        let Some(tool) = selected_tools.get(&materialization.solid_id) else {
            continue;
        };
        if !is_constructive_source(&tool.source) {
            continue;
        }
        let mut tool = (*tool).clone();
        if matches!(tool.source, CadFragmentSource::SolidExtrusion { .. }) {
            let Some(host_outwards) =
                feature_host_outward_directions(&family.host_face_ids, working, &working_index)
            else {
                continue;
            };
            if regularize_boundary_tool_for_boolean(&mut tool, family.polarity, &host_outwards)
                .is_err()
            {
                continue;
            }
        }
        recovered
            .entry(source_solid_id)
            .or_default()
            .push(RecoveredBoundaryFeatureFamily {
                pass_index,
                polarity: family.polarity,
                tool,
                source_member,
                members: family.members.clone(),
            });
        peelable_family_indices.push(family_index);
    }

    if peelable_family_indices.is_empty() {
        return Ok(false);
    }
    let stats = crate::planar_features::peel_boundary_feature_family_indices(
        working,
        &report,
        &peelable_family_indices,
    )?;
    Ok(stats.families != 0)
}

fn recover_constructive_fragments_from_entities(
    entities: &[EntityInstance],
) -> Result<Vec<CadFragment>> {
    let index = crate::step_graph::build_index(entities);
    let extrusions = crate::solid_extrusions::detect_solid_extrusions_with_index(entities, &index);
    let revolutions =
        crate::solid_revolutions::detect_solid_revolutions_with_index(entities, &index);
    let radial_slots =
        crate::solid_revolutions::detect_radial_slot_revolutions_with_index(entities, &index);
    let closed_sweeps = crate::solid_sweeps::detect_closed_round_sweeps(entities);
    let revolved_round_tails = crate::solid_sweeps::detect_revolved_round_tails(entities);
    let open_sweeps = crate::solid_sweeps::detect_open_rectangular_sweeps(entities);

    let mut out = recover_solid_extrusion_fragments(&extrusions)?;
    out.extend(recover_solid_revolution_fragments(&revolutions)?);
    out.extend(recover_radial_slot_revolution_fragments(&radial_slots)?);
    out.extend(recover_closed_round_sweep_fragments(&closed_sweeps)?);
    out.extend(recover_revolved_round_tail_fragments(
        &revolved_round_tails,
    )?);
    out.extend(recover_open_rectangular_sweep_fragments(&open_sweeps)?);
    Ok(out)
}

fn recover_edge_treated_extrusion_fragments(
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    target_solid_ids: Option<&BTreeSet<u64>>,
) -> Result<Vec<CadFragment>> {
    let signatures =
        crate::solid_revolutions::detect_solid_surface_signatures_with_index(entities, index);
    let dominant = crate::dominant_extrusions::detect_dominant_extrusion_candidates(&signatures);
    recover_edge_treated_extrusion_fragments_from_candidates(
        entities,
        index,
        &dominant,
        target_solid_ids,
    )
}

fn recover_edge_treated_extrusion_fragments_from_candidates(
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    dominant: &[crate::dominant_extrusions::DominantExtrusionCandidate],
    target_solid_ids: Option<&BTreeSet<u64>>,
) -> Result<Vec<CadFragment>> {
    let candidate_solid_ids = dominant
        .iter()
        .filter(|candidate| {
            !candidate.residual_faces.is_empty()
                && target_solid_ids.is_none_or(|targets| targets.contains(&candidate.solid_id))
        })
        .map(|candidate| candidate.solid_id)
        .collect::<HashSet<_>>();
    if candidate_solid_ids.is_empty() {
        return Ok(Vec::new());
    }
    let treatments = crate::edge_treatments::detect_edge_treatment_evidence_for_solids_with_index(
        entities,
        index,
        &candidate_solid_ids,
    );
    let mut out = Vec::new();

    for candidate in dominant {
        if candidate.residual_faces.is_empty()
            || target_solid_ids.is_some_and(|targets| !targets.contains(&candidate.solid_id))
        {
            continue;
        }

        let mut selected = Vec::with_capacity(candidate.residual_faces.len());
        let mut usable = true;
        for residual in &candidate.residual_faces {
            let mut matches = treatments.iter().filter_map(|item| {
                let same_solid = item.solid_id == candidate.solid_id;
                let same_face = item.treatment_face_id == residual.face_id;
                (same_solid && same_face)
                    .then(|| replayable_edge_treatment(item))
                    .flatten()
                    .map(|(profile, edge)| (item, profile, edge))
            });
            let Some(replay) = matches.next() else {
                usable = false;
                break;
            };
            if matches.next().is_some() {
                usable = false;
                break;
            }
            selected.push(replay);
        }
        if !usable || selected.len() != candidate.residual_faces.len() {
            continue;
        }

        let residual_face_ids = candidate
            .residual_faces
            .iter()
            .map(|face| face.face_id)
            .collect::<HashSet<_>>();
        let core = crate::solid_extrusions::recover_extrusion_core_from_cap_faces_with_index(
            candidate.solid_id,
            candidate.cap_face_ids,
            entities,
            index,
        )
        .or_else(|| {
            crate::solid_extrusions::recover_treatment_healed_extrusion_core_from_cap_faces_with_index(
                candidate.solid_id,
                candidate.cap_face_ids,
                entities,
                index,
                Some(&residual_face_ids),
            )
            .map(|healed| healed.core)
        });
        let Some(core) = core else {
            continue;
        };

        selected.sort_by_key(|(item, _, _)| item.treatment_face_id);
        let mut fragment = recover_solid_extrusion_fragment(&core)?;
        fragment.model.roots.clear();
        let mut root = fragment.root;
        for (item, profile, edge) in selected {
            root = fragment.model.add_node(CadNode::EdgeTreatment {
                child: root,
                profile,
                edges: vec![edge],
            });
            fragment.model.set_provenance(
                root,
                Provenance {
                    source_entity_ids: vec![candidate.solid_id, item.treatment_face_id],
                    proof: ProofStatus::WithinTolerance,
                    max_residual_mm: Some(item.max_boundary_residual_mm),
                },
            )?;
        }
        fragment.model.add_root(root)?;
        fragment.root = root;
        fragment.model.validate()?;
        out.push(fragment);
    }

    Ok(out)
}

fn replayable_edge_treatment(
    item: &crate::edge_treatments::EdgeTreatmentEvidence,
) -> Option<(EdgeTreatmentProfile, EdgeTreatmentEdge)> {
    if !item.carrier_pair_unique {
        return None;
    }
    let selector = match &item.virtual_edge {
        crate::edge_treatments::EdgeTreatmentVirtualEdge::Segment { start_mm, end_mm } => {
            EdgeSelector::Segment {
                start_mm: *start_mm,
                end_mm: *end_mm,
            }
        }
        crate::edge_treatments::EdgeTreatmentVirtualEdge::Circle { .. } => return None,
    };
    let (profile, radius_mm) = match item.kind {
        crate::edge_treatments::EdgeTreatmentKind::PlanarChamfer => {
            let [first, second] = item.inferred_setbacks_mm?;
            let tolerance = 1.0e-7_f64.max(first.abs().max(second.abs()) * 1.0e-8);
            if (first - second).abs() > tolerance {
                return None;
            }
            (EdgeTreatmentProfile::Chamfer, first.midpoint(second))
        }
        crate::edge_treatments::EdgeTreatmentKind::CylindricalFillet => {
            let radius = item.inferred_radius_mm?;
            if !radius.is_finite() || radius <= 0.0 {
                return None;
            }
            (EdgeTreatmentProfile::Round, radius)
        }
        _ => return None,
    };
    Some((
        profile,
        EdgeTreatmentEdge {
            selector,
            radius_mm,
        },
    ))
}

fn feature_host_outward_directions(
    host_face_ids: &[u64],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<[f64; 3]>> {
    let mut out = Vec::with_capacity(host_face_ids.len());
    for &face_id in host_face_ids {
        let surface_id = crate::brep::face_surface(face_id, entities, index)?;
        let crate::brep::SurfaceSupport::Plane(plane) =
            crate::brep::surface_support(surface_id, entities, index)
        else {
            return None;
        };
        let sign = match crate::brep::face_sense(face_id, entities, index)?.as_str() {
            "T" => 1.0,
            "F" => -1.0,
            _ => return None,
        };
        out.push([
            sign * plane.normal[0],
            sign * plane.normal[1],
            sign * plane.normal[2],
        ]);
    }
    (!out.is_empty()).then_some(out)
}

fn regularize_boundary_tool_for_boolean(
    tool: &mut CadFragment,
    polarity: FeaturePolarity,
    host_outwards: &[[f64; 3]],
) -> Result<()> {
    if !matches!(tool.source, CadFragmentSource::SolidExtrusion { .. }) {
        bail!("boundary-feature Boolean tool is not an extrusion");
    }
    let (transform_index, body_index, z_axis) = match tool.model.node(tool.root)? {
        CadNode::Transform { transform, child } => (
            tool.root.0,
            child.0,
            [
                transform.matrix[0][2],
                transform.matrix[1][2],
                transform.matrix[2][2],
            ],
        ),
        _ => bail!("boundary-feature extrusion tool root is not a rigid transform"),
    };
    let z_norm = z_axis[2]
        .mul_add(
            z_axis[2],
            z_axis[1].mul_add(z_axis[1], z_axis[0] * z_axis[0]),
        )
        .sqrt();
    if !z_norm.is_finite() || (z_norm - 1.0).abs() > 1.0e-9 {
        bail!("boundary-feature extrusion tool axis is not unit length");
    }

    let max_residual = tool
        .model
        .provenance
        .values()
        .filter_map(|provenance| provenance.max_residual_mm)
        .fold(0.0_f64, f64::max);
    let margin_mm = (10.0 * max_residual).max(1.0e-5);
    let mut extend_negative = false;
    let mut extend_positive = false;
    for outward in host_outwards {
        let outward_norm = outward[2]
            .mul_add(
                outward[2],
                outward[1].mul_add(outward[1], outward[0] * outward[0]),
            )
            .sqrt();
        if !outward_norm.is_finite() || outward_norm <= 0.0 {
            bail!("boundary-feature host normal is invalid");
        }
        let alignment = outward[2].mul_add(
            z_axis[2],
            outward[1].mul_add(z_axis[1], outward[0] * z_axis[0]),
        ) / outward_norm;
        if (alignment.abs() - 1.0).abs() > 1.0e-8 {
            bail!("boundary-feature interface is not an extrusion end cap");
        }
        match (polarity, alignment > 0.0) {
            // A subtractive tool occupies source-solid material and opens at the
            // carrier plane. Extend it outward, beyond the source solid.
            (FeaturePolarity::Subtractive, true)
            // An additive tool occupies material outside the source host and is
            // attached at the carrier plane. Extend it inward, into material the
            // host already occupies, so the union has volumetric overlap.
            | (FeaturePolarity::Additive, false) => extend_positive = true,
            (FeaturePolarity::Subtractive, false) | (FeaturePolarity::Additive, true) => {
                extend_negative = true;
            }
        }
    }

    let extension_count = u8::from(extend_negative) + u8::from(extend_positive);
    if extension_count == 0 {
        bail!("boundary-feature Boolean tool has no extendable interface");
    }
    let CadNode::Extrude { vector_mm, .. } = &mut tool.model.nodes[body_index] else {
        bail!("boundary-feature tool transform does not wrap an extrusion");
    };
    if vector_mm[0].abs() > 1.0e-12
        || vector_mm[1].abs() > 1.0e-12
        || !vector_mm[2].is_finite()
        || vector_mm[2] <= 0.0
    {
        bail!("boundary-feature extrusion vector is not canonical +Z");
    }
    vector_mm[2] = margin_mm.mul_add(f64::from(extension_count), vector_mm[2]);

    if extend_negative {
        let CadNode::Transform { transform, .. } = &mut tool.model.nodes[transform_index] else {
            unreachable!("root shape checked above");
        };
        for (axis, axis_component) in z_axis.iter().copied().enumerate() {
            transform.matrix[axis][3] =
                (-margin_mm).mul_add(axis_component, transform.matrix[axis][3]);
        }
    }

    for provenance in tool.model.provenance.values_mut() {
        provenance.proof = ProofStatus::StructurallyProven;
        provenance.max_residual_mm = None;
    }
    tool.model.validate()?;
    Ok(())
}

fn compose_boundary_feature_fragment(
    source_solid_id: u64,
    host: &CadFragment,
    families: &[RecoveredBoundaryFeatureFamily],
) -> Result<CadFragment> {
    let mut model = CadModel::new();
    let host_offset = model.append_model(&host.model)?;
    let mut current = shifted_fragment_root(host.root, host_offset)?;

    for family in families.iter().rev() {
        let tool_offset = model.append_model(&family.tool.model)?;
        let tool_root = shifted_fragment_root(family.tool.root, tool_offset)?;
        let exemplar_sources = feature_member_source_ids(source_solid_id, &family.source_member);
        for node_index in tool_offset..tool_offset + family.tool.model.nodes.len() {
            if let Some(provenance) = model.provenance.get_mut(&NodeId(node_index)) {
                provenance.source_entity_ids.clone_from(&exemplar_sources);
            }
        }

        let mut instance_roots = Vec::with_capacity(family.members.len());
        for member in &family.members {
            let quarter =
                (family.source_member.normalized_quarter + 4 - member.normalized_quarter) % 4;
            let translation = crate::planar_features::rigid_translation(
                family.source_member.center_mm,
                member.center_mm,
                quarter,
            );
            let root = if quarter == 0 && translation.iter().all(|value| value.abs() <= 1.0e-12) {
                tool_root
            } else {
                let transform = model.add_node(CadNode::Transform {
                    transform: quarter_turn_transform(quarter, translation),
                    child: tool_root,
                });
                model.set_provenance(
                    transform,
                    Provenance {
                        source_entity_ids: feature_member_source_ids(source_solid_id, member),
                        proof: ProofStatus::StructurallyProven,
                        max_residual_mm: None,
                    },
                )?;
                transform
            };
            instance_roots.push(root);
        }

        let mut children = Vec::with_capacity(instance_roots.len() + 1);
        children.push(current);
        children.extend(instance_roots);
        let op = match family.polarity {
            FeaturePolarity::Additive => BooleanOp::Union,
            FeaturePolarity::Subtractive => BooleanOp::Difference,
        };
        current = model.add_node(CadNode::Boolean { op, children });
        let mut sources = vec![source_solid_id];
        for member in &family.members {
            sources.extend(feature_member_source_ids(source_solid_id, member));
        }
        sources.sort_unstable();
        sources.dedup();
        model.set_provenance(
            current,
            Provenance {
                source_entity_ids: sources,
                proof: ProofStatus::StructurallyProven,
                max_residual_mm: None,
            },
        )?;
    }

    model.add_root(current)?;
    model.validate()?;
    let passes = families
        .iter()
        .map(|family| family.pass_index)
        .max()
        .map_or(0, |last| last + 1);
    Ok(CadFragment {
        source: CadFragmentSource::BoundaryFeatureComposite {
            solid_id: source_solid_id,
            passes,
            feature_families: families.len(),
            feature_instances: families.iter().map(|family| family.members.len()).sum(),
        },
        model,
        root: current,
    })
}

fn shifted_fragment_root(root: NodeId, offset: usize) -> Result<NodeId> {
    Ok(NodeId(root.0.checked_add(offset).ok_or_else(|| {
        anyhow::anyhow!("CAD root id overflow while composing fragments")
    })?))
}

fn feature_member_source_ids(
    source_solid_id: u64,
    member: &BoundaryFeatureInstanceEvidence,
) -> Vec<u64> {
    let mut out = Vec::with_capacity(member.face_ids.len() + member.interface_bound_ids.len() + 1);
    out.push(source_solid_id);
    out.extend(member.face_ids.iter().copied());
    out.extend(member.interface_bound_ids.iter().copied());
    out.sort_unstable();
    out.dedup();
    out
}

fn quarter_turn_transform(quarter: u8, translation: [f64; 3]) -> RigidTransform {
    let (xx, xy, yx, yy) = match quarter % 4 {
        0 => (1.0, 0.0, 0.0, 1.0),
        1 => (0.0, -1.0, 1.0, 0.0),
        2 => (-1.0, 0.0, 0.0, -1.0),
        3 => (0.0, 1.0, -1.0, 0.0),
        _ => unreachable!(),
    };
    RigidTransform {
        matrix: [
            [xx, xy, 0.0, translation[0]],
            [yx, yy, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
}

fn scan_source_solids(
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Vec<SourceSolidSummary> {
    let mut out = Vec::new();
    for entity in entities {
        let Some(record) = crate::step_graph::simple_record(entity) else {
            continue;
        };
        if record.name != "MANIFOLD_SOLID_BREP" {
            continue;
        }
        let solid_id = crate::step_graph::entity_id(entity);
        let closure = crate::shape_identity::semantic_solid_closure(solid_id, entities, index)
            .unwrap_or_else(|| crate::step_entities::closure_from(solid_id, entities, index));
        let mut faces = 0usize;
        let mut edges = 0usize;
        for id in &closure {
            let Some(&entity_index) = index.get(id) else {
                continue;
            };
            let Some(child) = crate::step_graph::simple_record(&entities[entity_index]) else {
                continue;
            };
            match child.name.as_str() {
                "ADVANCED_FACE" => faces += 1,
                "EDGE_CURVE" => edges += 1,
                _ => {}
            }
        }
        out.push(SourceSolidSummary {
            solid_id,
            faces,
            edges,
            closure_entities: closure.len(),
        });
    }
    out
}

fn source_solid_id(source: &CadFragmentSource) -> u64 {
    match source {
        CadFragmentSource::SolidExtrusion { solid_id, .. }
        | CadFragmentSource::SolidRevolution { solid_id, .. }
        | CadFragmentSource::RadialSlotRevolution { solid_id, .. }
        | CadFragmentSource::SolidSweep { solid_id, .. }
        | CadFragmentSource::RevolvedRoundTail { solid_id, .. }
        | CadFragmentSource::PeriodicChain { solid_id, .. }
        | CadFragmentSource::ExtrusionCoreUnion { solid_id, .. }
        | CadFragmentSource::BoundaryFeatureComposite { solid_id, .. } => *solid_id,
    }
}

const fn is_constructive_source(source: &CadFragmentSource) -> bool {
    !matches!(source, CadFragmentSource::PeriodicChain { .. })
}

fn select_cheapest_whole_solid_fragments<'a>(
    candidates: &[&'a CadFragment],
) -> Result<BTreeMap<u64, &'a CadFragment>> {
    let mut selected = BTreeMap::<u64, (&CadFragment, u64)>::new();
    for &fragment in candidates {
        let solid_id = source_solid_id(&fragment.source);
        let complexity = fragment.model.complexity_score(fragment.root)?;
        match selected.get(&solid_id) {
            Some((existing_fragment, existing_complexity))
                if is_constructive_source(&existing_fragment.source)
                    && !is_constructive_source(&fragment.source) => {}
            Some((existing_fragment, _))
                if !is_constructive_source(&existing_fragment.source)
                    && is_constructive_source(&fragment.source) =>
            {
                selected.insert(solid_id, (fragment, complexity));
            }
            Some((_, existing_complexity)) if *existing_complexity <= complexity => {}
            _ => {
                selected.insert(solid_id, (fragment, complexity));
            }
        }
    }
    Ok(selected
        .into_iter()
        .map(|(solid_id, (fragment, _))| (solid_id, fragment))
        .collect())
}

fn selected_category<F>(selected: &BTreeMap<u64, &CadFragment>, predicate: F) -> Vec<CadFragment>
where
    F: Fn(&CadFragmentSource) -> bool,
{
    selected
        .values()
        .filter(|fragment| predicate(&fragment.source))
        .map(|fragment| (*fragment).clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cad_ir::{CadModel, CadNode, Primitive};

    fn report_with_unrecovered(solid_ids: &[u64]) -> CompleteIr {
        let solids = solid_ids
            .iter()
            .map(|&solid_id| SourceSolidSummary {
                solid_id,
                faces: 0,
                edges: 0,
                closure_entities: 0,
            })
            .collect::<Vec<_>>();
        CompleteIr {
            schema: COMPLETE_IR_SCHEMA,
            idempotent_compact_rewrite: true,
            compact_second_pass_stats: Stats::default(),
            surface_signature_summaries: Vec::new(),
            instance_pattern_diagnostics: Vec::new(),
            solid_extrusions: Vec::new(),
            solid_revolutions: Vec::new(),
            radial_slot_revolutions: Vec::new(),
            solid_sweeps: Vec::new(),
            periodic_chains: Vec::new(),
            boundary_feature_composites: Vec::new(),
            extrusion_core_unions: Vec::new(),
            kernel_recovery_failures: Vec::new(),
            unrecovered_solids: solids.clone(),
            body_scan: BodyScanSummary { solids },
            body_count: solid_ids.len(),
            constructively_recovered_solid_count: 0,
            semantically_recovered_solid_count: 0,
            unrecovered_solid_count: solid_ids.len(),
        }
    }

    fn constructive_fragment(solid_id: u64, boxes: usize) -> CadFragment {
        assert!(boxes > 0);
        let mut model = CadModel::new();
        let children = (0..boxes)
            .map(|_| {
                model.add_node(CadNode::Primitive(Primitive::Box {
                    size_mm: [1.0, 1.0, 1.0],
                }))
            })
            .collect::<Vec<_>>();
        let root = if children.len() == 1 {
            children[0]
        } else {
            model.add_node(CadNode::Assembly { children })
        };
        model.add_root(root).expect("test root");
        CadFragment {
            source: CadFragmentSource::SolidExtrusion {
                solid_id,
                cap_face_ids: [1, 2],
                side_face_ids: Vec::new(),
            },
            model,
            root,
        }
    }

    fn extrusion_core_union_fragment(solid_id: u64) -> Result<CadFragment> {
        let mut model = CadModel::new();
        let root = model.add_node(CadNode::Primitive(Primitive::Box {
            size_mm: [1.0, 1.0, 1.0],
        }));
        model.add_root(root)?;
        Ok(CadFragment {
            source: CadFragmentSource::ExtrusionCoreUnion {
                solid_id,
                cap_face_ids: vec![[1, 2], [3, 4]],
            },
            model,
            root,
        })
    }

    #[test]
    fn kernel_augmentation_skips_recovery_when_nothing_is_unrecovered() {
        let mut calls = 0usize;
        let report = apply_kernel_recovery(report_with_unrecovered(&[]), |_| {
            calls += 1;
            Ok(None)
        });
        assert_eq!(calls, 0);
        assert_eq!(report.schema, KERNEL_COMPLETE_IR_SCHEMA);
        assert_eq!(report.kernel_recovery_failures, Vec::new());
    }

    #[test]
    fn kernel_augmentation_rejects_fragment_for_another_solid() -> Result<()> {
        let mut fragment = Some(extrusion_core_union_fragment(8)?);
        let report = apply_kernel_recovery(report_with_unrecovered(&[7]), |_| Ok(fragment.take()));
        assert_eq!(report.unrecovered_solid_count, 1);
        assert_eq!(report.semantically_recovered_solid_count, 0);
        assert_eq!(report.extrusion_core_unions, Vec::new());
        assert_eq!(report.kernel_recovery_failures.len(), 1);
        assert_eq!(report.kernel_recovery_failures[0].solid_id, 7);
        Ok(())
    }

    #[test]
    fn kernel_augmentation_records_backend_failure_per_solid() {
        let report = apply_kernel_recovery(report_with_unrecovered(&[7]), |solid_id| {
            Err(anyhow::anyhow!("kernel refused solid #{solid_id}"))
        });
        assert_eq!(report.unrecovered_solid_count, 1);
        assert_eq!(report.semantically_recovered_solid_count, 0);
        assert_eq!(report.kernel_recovery_failures.len(), 1);
        assert_eq!(report.kernel_recovery_failures[0].solid_id, 7);
    }

    #[test]
    fn kernel_augmentation_moves_admitted_union_out_of_unrecovered() -> Result<()> {
        let mut fragment = Some(extrusion_core_union_fragment(7)?);
        let report = apply_kernel_recovery(report_with_unrecovered(&[7]), |_| Ok(fragment.take()));
        assert_eq!(report.unrecovered_solid_count, 0);
        assert_eq!(report.constructively_recovered_solid_count, 1);
        assert_eq!(report.semantically_recovered_solid_count, 1);
        assert_eq!(report.extrusion_core_unions.len(), 1);
        assert_eq!(report.kernel_recovery_failures, Vec::new());
        Ok(())
    }

    #[test]
    fn cheapest_whole_solid_candidate_wins() -> Result<()> {
        let expensive = constructive_fragment(7, 4);
        let cheap = constructive_fragment(7, 1);
        let selected = select_cheapest_whole_solid_fragments(&[&expensive, &cheap])?;
        assert_eq!(selected.len(), 1);
        assert!(std::ptr::eq(selected[&7], &cheap));
        Ok(())
    }

    #[test]
    fn constructive_candidate_beats_cheaper_descriptive_chain() -> Result<()> {
        let constructive = constructive_fragment(7, 3);
        let mut model = CadModel::new();
        let root = model.add_node(CadNode::Primitive(Primitive::Box {
            size_mm: [1.0, 1.0, 1.0],
        }));
        model.add_root(root)?;
        let descriptive = CadFragment {
            source: CadFragmentSource::PeriodicChain {
                solid_id: 7,
                sites: 2,
            },
            model,
            root,
        };
        let selected = select_cheapest_whole_solid_fragments(&[&descriptive, &constructive])?;
        assert!(std::ptr::eq(selected[&7], &raw const constructive));
        Ok(())
    }
}
