use crate::SourceEntity;
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
use crate::kernel_source_load::{
    KernelSourceLoadEvidence, KernelTopologyProvenance, SourceStepLoadInspector,
};
use crate::kernel_topology::{
    SourceTopologyEditor, plan_source_boundary_feature_peel_patch,
    plan_source_coplanar_normalization_patch, plan_source_edge_treatment_patches,
};
use crate::patterns::InstancePattern;
use crate::planar_features::{
    BoundaryFeatureInstanceEvidence, BoundaryFeaturePeelPlan, FeaturePolarity,
};
use crate::solid_revolutions::SolidSurfaceSignature;
use crate::{Options, OutputProfile, Stats};
use anyhow::{Result, bail};
use ruststep::ast::{EntityInstance, Exchange};
use serde::Serialize;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

pub const COMPLETE_IR_SCHEMA: &str = "step-redox-complete-ir-v10";
pub const KERNEL_COMPLETE_IR_SCHEMA: &str = "step-redox-complete-ir-v15";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReducedBrepMutationMode {
    ParserOwned,
    KernelOwned,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SurfaceSignatureSummary {
    pub source: SourceEntity,
    pub face_count: usize,
    pub support_counts: BTreeMap<String, usize>,
    pub unique_edge_count: usize,
    pub edge_use_count: usize,
    pub closed_two_manifold: bool,
}

impl SurfaceSignatureSummary {
    fn from_source(source: SourceEntity, value: &SolidSurfaceSignature) -> Self {
        Self {
            source,
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
    pub source: SourceEntity,
    pub faces: usize,
    pub edges: usize,
    pub closure_entities: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BodyScanSummary {
    pub solids: Vec<SourceSolidSummary>,
}

/// One recovered CAD fragment qualified by its document-global source solid.
///
/// Raw STEP entity IDs inside `fragment.source` and node provenance are local to
/// `source_solid.data_section`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SourceCadFragment {
    pub source_solid: SourceEntity,
    pub fragment: CadFragment,
}

/// Instance-pattern diagnostics qualified by the DATA section owning their raw STEP IDs.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SourceInstancePattern {
    pub data_section: usize,
    pub pattern: InstancePattern,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelRecoveryFailure {
    pub source: SourceEntity,
    pub error: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompleteIr {
    pub schema: &'static str,
    pub idempotent_compact_rewrite: bool,
    pub compact_second_pass_stats: Stats,
    pub surface_signature_summaries: Vec<SurfaceSignatureSummary>,
    pub instance_pattern_diagnostics: Vec<SourceInstancePattern>,
    pub solid_extrusions: Vec<SourceCadFragment>,
    pub solid_revolutions: Vec<SourceCadFragment>,
    pub radial_slot_revolutions: Vec<SourceCadFragment>,
    pub solid_sweeps: Vec<SourceCadFragment>,
    pub periodic_chains: Vec<SourceCadFragment>,
    pub boundary_feature_composites: Vec<SourceCadFragment>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub edge_treatment_composites: Vec<SourceCadFragment>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub coplanar_normalized_solids: Vec<SourceCadFragment>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub extrusion_core_unions: Vec<SourceCadFragment>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub kernel_recovery_failures: Vec<KernelRecoveryFailure>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_source_load: Option<KernelSourceLoadEvidence>,
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
    /// False when feature IDs came from a kernel-generated intermediate STEP.
    original_source_ids: bool,
}

#[derive(Debug)]
struct RecoverySection<'a> {
    data_section: usize,
    entities: &'a [EntityInstance],
    index: HashMap<u64, usize>,
}

impl RecoverySection<'_> {
    fn source_entity(&self, id: u64) -> Option<SourceEntity> {
        crate::step_document::StepSection::new(self.data_section, self.entities, &self.index)
            .entity(id)
            .map(crate::step_document::StepEntity::source)
    }

    fn local_target_ids(&self, targets: &BTreeSet<SourceEntity>) -> BTreeSet<u64> {
        targets
            .iter()
            .filter(|source| source.data_section == self.data_section)
            .map(|source| source.id)
            .collect()
    }

    fn qualify_fragment(&self, fragment: CadFragment) -> Result<SourceCadFragment> {
        let solid_id = source_solid_id(&fragment.source);
        let source = self.source_entity(solid_id).ok_or_else(|| {
            anyhow::anyhow!(
                "recovered CAD fragment refers to missing solid #{solid_id} in DATA section {}",
                self.data_section
            )
        })?;
        Ok(SourceCadFragment {
            source_solid: source,
            fragment,
        })
    }

    fn qualify_fragments<I>(&self, fragments: I) -> Result<Vec<SourceCadFragment>>
    where
        I: IntoIterator<Item = CadFragment>,
    {
        fragments
            .into_iter()
            .map(|fragment| self.qualify_fragment(fragment))
            .collect()
    }
}

#[derive(Debug)]
struct LocalRecoveryTargets {
    solid_ids: BTreeSet<u64>,
    shell_to_solid: BTreeMap<u64, u64>,
}

impl LocalRecoveryTargets {
    fn from_solids(
        solid_ids: &BTreeSet<u64>,
        entities: &[EntityInstance],
        index: &HashMap<u64, usize>,
    ) -> Result<Self> {
        let mut shell_to_solid = BTreeMap::new();
        for &solid_id in solid_ids {
            let Some(shell_id) = crate::brep::manifold_shell(solid_id, entities, index) else {
                continue;
            };
            if let Some(previous) = shell_to_solid.insert(shell_id, solid_id)
                && previous != solid_id
            {
                bail!(
                    "local recovery shell #{shell_id} belongs to both solid #{previous} and #{solid_id}"
                );
            }
        }
        Ok(Self {
            solid_ids: solid_ids.clone(),
            shell_to_solid,
        })
    }

    fn is_empty(&self) -> bool {
        self.solid_ids.is_empty()
    }

    fn shell_ids(&self) -> HashSet<u64> {
        self.shell_to_solid.keys().copied().collect()
    }

    fn solid_for_shell(&self, shell_id: u64) -> Option<u64> {
        self.shell_to_solid.get(&shell_id).copied()
    }

    fn retain_unrecovered(&mut self, recovered: &BTreeMap<u64, CadFragment>) {
        self.solid_ids
            .retain(|solid_id| !recovered.contains_key(solid_id));
        self.shell_to_solid
            .retain(|_, solid_id| self.solid_ids.contains(solid_id));
    }
}

#[derive(Debug, Clone)]
enum LocalReductionStep {
    BoundaryFeature(RecoveredBoundaryFeatureFamily),
}

/// Parser-only semantic peel journal. The original Part 21 graph remains immutable;
/// a sequence of proven deletion plans projects a temporary entity view for the
/// existing recognizers, and is never mistaken for a separately owned B-rep.
/// The kernel path does not construct this journal.
#[derive(Debug, Default)]
struct ParserPeelJournal {
    plans: Vec<BoundaryFeaturePeelPlan>,
    replay: BTreeMap<u64, Vec<LocalReductionStep>>,
}

impl ParserPeelJournal {
    fn view<'a>(
        &self,
        source: &'a [EntityInstance],
        index: &HashMap<u64, usize>,
    ) -> Result<Cow<'a, [EntityInstance]>> {
        if self.plans.is_empty() {
            return Ok(Cow::Borrowed(source));
        }
        Ok(Cow::Owned(
            crate::planar_features::materialize_boundary_feature_peel_plans(
                source,
                index,
                &self.plans,
            )?,
        ))
    }

    fn commit(
        &mut self,
        steps: Vec<(u64, LocalReductionStep)>,
        plan: BoundaryFeaturePeelPlan,
        current: &[EntityInstance],
        index: &HashMap<u64, usize>,
    ) -> Result<bool> {
        if steps.is_empty() {
            return Ok(false);
        }
        if steps.len() != plan.stats.families || plan.stats.families == 0 {
            bail!(
                "parser peel journal has {} replay steps but {} proven families",
                steps.len(),
                plan.stats.families
            );
        }
        // Prove that the next projection is actually materializable before
        // committing either topology removals or semantic replay. A failed
        // transaction leaves the journal (and original input) unchanged.
        let _ = crate::planar_features::materialize_boundary_feature_peel_plans(
            current,
            index,
            std::slice::from_ref(&plan),
        )?;
        self.plans.push(plan);
        for (solid_id, step) in steps {
            self.replay.entry(solid_id).or_default().push(step);
        }
        Ok(true)
    }

    fn boundary_features(&self, solid_id: u64) -> Vec<RecoveredBoundaryFeatureFamily> {
        self.replay
            .get(&solid_id)
            .into_iter()
            .flatten()
            .map(|step| match step {
                LocalReductionStep::BoundaryFeature(feature) => feature.clone(),
            })
            .collect()
    }

    fn next_boundary_pass_index(&self) -> usize {
        self.replay
            .values()
            .flatten()
            .map(|step| match step {
                LocalReductionStep::BoundaryFeature(feature) => feature.pass_index,
            })
            .max()
            .map_or(0, |last| last + 1)
    }
}

#[derive(Debug)]
struct PreparedBoundaryReduction {
    steps: Vec<(u64, LocalReductionStep)>,
    family_indices: Vec<usize>,
}

#[derive(Debug)]
struct SelectedWholeSolid {
    fragment: CadFragment,
    complexity: u64,
}

#[derive(Debug)]
struct WholeSolidRecoveryState {
    source_solids: BTreeSet<SourceEntity>,
    selected: BTreeMap<SourceEntity, SelectedWholeSolid>,
    revision: u64,
}

impl WholeSolidRecoveryState {
    fn new(body_scan: &[SourceSolidSummary]) -> Self {
        Self {
            source_solids: body_scan.iter().map(|solid| solid.source).collect(),
            selected: BTreeMap::new(),
            revision: 0,
        }
    }

    fn add_candidates<I>(&mut self, candidates: I) -> Result<usize>
    where
        I: IntoIterator<Item = SourceCadFragment>,
    {
        let mut changes = 0;
        for candidate in candidates {
            let SourceCadFragment {
                source_solid: source,
                fragment,
            } = candidate;
            let local_solid_id = source_solid_id(&fragment.source);
            if local_solid_id != source.id {
                bail!(
                    "recovered CAD fragment for DATA section {} solid #{} carries local solid #{local_solid_id}",
                    source.data_section,
                    source.id,
                );
            }
            if !self.source_solids.contains(&source) {
                bail!(
                    "recovered CAD fragment refers to missing source solid {:?} #{}",
                    source.data_section,
                    source.id
                );
            }
            let complexity = fragment.model.complexity_score(fragment.root)?;
            let replace = self.selected.get(&source).is_none_or(|existing| {
                whole_solid_candidate_is_preferred(
                    &existing.fragment.source,
                    existing.complexity,
                    &fragment.source,
                    complexity,
                )
            });
            if replace {
                self.selected.insert(
                    source,
                    SelectedWholeSolid {
                        fragment,
                        complexity,
                    },
                );
                self.revision += 1;
                changes += 1;
            }
        }
        Ok(changes)
    }

    fn constructively_unresolved_sources(&self) -> BTreeSet<SourceEntity> {
        self.source_solids
            .iter()
            .copied()
            .filter(|source| {
                self.selected
                    .get(source)
                    .is_none_or(|selected| !is_constructive_source(&selected.fragment.source))
            })
            .collect()
    }

    fn semantically_unrecovered_sources(&self) -> BTreeSet<SourceEntity> {
        self.source_solids
            .iter()
            .copied()
            .filter(|source| !self.selected.contains_key(source))
            .collect()
    }

    fn selected_fragments(&self) -> BTreeMap<SourceEntity, &CadFragment> {
        self.selected
            .iter()
            .map(|(&source, selected)| (source, &selected.fragment))
            .collect()
    }
}

#[derive(Debug)]
struct CompleteIrBuildState {
    idempotent_compact_rewrite: bool,
    compact_second_pass_stats: Stats,
    surface_signature_summaries: Vec<SurfaceSignatureSummary>,
    instance_pattern_diagnostics: Vec<SourceInstancePattern>,
    body_scan: Vec<SourceSolidSummary>,
    recovery: WholeSolidRecoveryState,
}

impl CompleteIrBuildState {
    fn finish(
        self,
        schema: &'static str,
        kernel_recovery_failures: Vec<KernelRecoveryFailure>,
        kernel_source_load: Option<KernelSourceLoadEvidence>,
    ) -> CompleteIr {
        let selected = self.recovery.selected_fragments();
        let recovered_ids = selected.keys().copied().collect::<BTreeSet<_>>();
        let unrecovered_solids = self
            .body_scan
            .iter()
            .filter(|solid| !recovered_ids.contains(&solid.source))
            .cloned()
            .collect::<Vec<_>>();
        let constructively_recovered_solid_count = selected
            .values()
            .filter(|fragment| is_constructive_source(&fragment.source))
            .count();

        CompleteIr {
            schema,
            idempotent_compact_rewrite: self.idempotent_compact_rewrite,
            compact_second_pass_stats: self.compact_second_pass_stats,
            surface_signature_summaries: self.surface_signature_summaries,
            instance_pattern_diagnostics: self.instance_pattern_diagnostics,
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
                    CadFragmentSource::SolidSweep { .. }
                        | CadFragmentSource::RevolvedRoundTail { .. }
                )
            }),
            periodic_chains: selected_category(&selected, |source| {
                matches!(source, CadFragmentSource::PeriodicChain { .. })
            }),
            boundary_feature_composites: selected_category(&selected, |source| {
                matches!(source, CadFragmentSource::BoundaryFeatureComposite { .. })
            }),
            edge_treatment_composites: selected_category(&selected, |source| {
                matches!(source, CadFragmentSource::EdgeTreatmentComposite { .. })
            }),
            coplanar_normalized_solids: selected_category(&selected, |source| {
                matches!(source, CadFragmentSource::CoplanarNormalized { .. })
            }),
            extrusion_core_unions: selected_category(&selected, |source| {
                matches!(source, CadFragmentSource::ExtrusionCoreUnion { .. })
            }),
            kernel_recovery_failures,
            kernel_source_load,
            unrecovered_solid_count: unrecovered_solids.len(),
            body_count: self.body_scan.len(),
            constructively_recovered_solid_count,
            semantically_recovered_solid_count: selected.len(),
            body_scan: BodyScanSummary {
                solids: self.body_scan,
            },
            unrecovered_solids,
        }
    }
}

type WholeSolidCandidateProvider = for<'a> fn(
    &[RecoverySection<'a>],
    &BTreeSet<SourceEntity>,
    ReducedBrepMutationMode,
) -> Result<Vec<SourceCadFragment>>;

const WHOLE_SOLID_CANDIDATE_PROVIDERS: [WholeSolidCandidateProvider; 2] = [
    recover_boundary_feature_candidates,
    recover_reduced_brep_candidates,
];
const MAX_LOCAL_REDUCTION_ROUNDS: usize = 32;

/// Build the compact complete semantic IR for a normalized STEP exchange.
///
/// This report deliberately separates compact canonical IR from source diagnostics.
/// Every recovered manifold solid is represented by the cheapest proven whole-solid
/// semantic fragment. Bodies without a proven constructive representation are listed
/// explicitly in `unrecovered_solids` and never enter `cad_ir`. Instance-pattern
/// evidence remains diagnostic until its repeated child can be linked to proven CAD IR.
///
/// # Errors
/// Returns an error if STEP parsing fails or a detector's proven result cannot be converted into valid CAD IR.
pub fn recover_complete_ir_bytes(input: &[u8]) -> Result<CompleteIr> {
    let session = crate::AnalysisSession::from_bytes(input)?;
    session.complete_ir()
}

/// Build Complete IR and opportunistically recover otherwise-unrecovered solids
/// through a supplied CAD kernel plus the independent analytic boundary proof.
///
/// The parser-only [`recover_complete_ir_bytes`] path reports schema v10 and never
/// invokes a CAD backend. This explicit kernel-backed path reports schema v15,
/// includes parser-neutral source-load evidence,
/// records admitted kernel-owned topology-normalized composites and extrusion-core unions separately, and leaves any unsupported
/// or unproven body in `unrecovered_solids`. Backend failures are diagnostic per
/// solid rather than aborting the rest of the report.
///
/// # Errors
/// Returns an error if STEP parsing/Complete IR construction fails or if the selected kernel cannot inspect the original STEP source.
pub fn recover_complete_ir_with_kernel_bytes<
    K: CadKernel + SourceStepLoadInspector + SourceTopologyEditor,
>(
    input: &[u8],
    kernel: &K,
) -> Result<CompleteIr> {
    let exchange = crate::step_io::ParsedExchange::parse(input)?.exchange;
    let mut build =
        recover_complete_ir_build_state(&exchange, input, ReducedBrepMutationMode::KernelOwned)?;
    let source_load = kernel.inspect_source_step(input)?;
    let mut failures = apply_kernel_recovery(&mut build.recovery, |source| {
        recover_kernel_boundary_feature_fragment(input, source, &source_load, kernel)
    });
    failures.extend(apply_kernel_recovery(&mut build.recovery, |source| {
        recover_kernel_topology_normalized_fragment(input, &exchange, source, &source_load, kernel)
    }));
    failures.extend(apply_kernel_recovery(&mut build.recovery, |source| {
        recover_auto_proven_extrusion_core_union_fragment(
            input,
            source.data_section,
            source.id,
            kernel,
        )
    }));
    Ok(build.finish(KERNEL_COMPLETE_IR_SCHEMA, failures, Some(source_load)))
}

/// Peel fully proven boundary-feature leaves on kernel-owned topology. The parser
/// constructs tool/selection evidence, never a mutable recovery shape.
fn recover_kernel_boundary_feature_fragment<
    K: CadKernel + SourceStepLoadInspector + SourceTopologyEditor,
>(
    input: &[u8],
    source: SourceEntity,
    source_load: &KernelSourceLoadEvidence,
    kernel: &K,
) -> Result<Option<CadFragment>> {
    let Some(load) = source_load.solids.iter().find(|item| item.source == source) else {
        return Ok(None);
    };
    if load.failure.is_some() {
        return Ok(None);
    }
    let Some(mut provenance) = load.topology_provenance.clone() else {
        return Ok(None);
    };
    let mut current_step = None::<String>;
    let mut current_source = source;
    let mut replay = Vec::<RecoveredBoundaryFeatureFamily>::new();

    for pass in 0..MAX_LOCAL_REDUCTION_ROUNDS {
        let bytes = current_step.as_deref().map_or(input, str::as_bytes);
        let exchange = crate::step_io::ParsedExchange::parse(bytes)?.exchange;
        let section = exchange
            .data
            .get(current_source.data_section)
            .ok_or_else(|| anyhow::anyhow!("kernel peel lost its current DATA section"))?;
        let index = crate::step_graph::build_index(&section.entities);
        let targets = LocalRecoveryTargets::from_solids(
            &BTreeSet::from([current_source.id]),
            &section.entities,
            &index,
        )?;
        if targets.is_empty() {
            break;
        }

        // Kernel-owned peeling only reads the source for feature evidence.
        // The sole mutable copy is temporary tool materialization, never a
        // competing authoritative B-rep or parser-side replay state.
        let shells = targets.shell_ids();
        let report = crate::planar_features::diagnose_boundary_features_for_shells_with_index(
            &section.entities,
            &index,
            &shells,
        );
        let selected = crate::planar_features::leaf_feature_family_indices(&report.families)
            .into_iter()
            .filter(|&i| {
                targets
                    .solid_for_shell(report.families[i].shell_id)
                    .is_some()
            })
            .collect::<Vec<_>>();
        if selected.is_empty() {
            break;
        }
        let prepared = prepare_boundary_feature_reduction(
            &section.entities,
            &index,
            &targets,
            &shells,
            &report,
            &selected,
            pass,
        )?;
        if prepared.family_indices.is_empty() {
            break;
        }
        let patch = plan_source_boundary_feature_peel_patch(
            bytes,
            current_source,
            &report,
            &prepared.family_indices,
            &provenance,
        )?;
        let healed = kernel.edit_source_topology(bytes, current_source, &[patch])?;
        let Some((healed_source, healed_provenance)) =
            inspect_single_loaded_source(kernel, healed.as_bytes())?
        else {
            return Ok(None);
        };

        // Only after a successful kernel transaction may semantic replay advance.
        for (_, LocalReductionStep::BoundaryFeature(mut feature)) in prepared.steps {
            feature.original_source_ids = pass == 0;
            replay.push(feature);
        }
        current_source = healed_source;
        provenance = healed_provenance;
        current_step = Some(healed);
        if pass + 1 == MAX_LOCAL_REDUCTION_ROUNDS {
            bail!("kernel boundary-feature peeling did not reach a fixed point");
        }
    }
    let Some(final_step) = current_step else {
        return Ok(None);
    };
    let Some(mut host) = recognize_single_healed_host(&final_step)? else {
        return Ok(None);
    };
    host.model.provenance.clear();
    let fragment = compose_boundary_feature_fragment(source.id, &host, &replay)?;
    kernel_replay_is_single_solid(kernel, &fragment).map(|accepted| accepted.then_some(fragment))
}

fn recover_kernel_topology_normalized_fragment<
    K: CadKernel + SourceStepLoadInspector + SourceTopologyEditor,
>(
    input: &[u8],
    exchange: &Exchange,
    source: SourceEntity,
    source_load: &KernelSourceLoadEvidence,
    kernel: &K,
) -> Result<Option<CadFragment>> {
    let Some(load) = source_load
        .solids
        .iter()
        .find(|solid| solid.source == source)
    else {
        return Ok(None);
    };
    if load.failure.is_some() {
        return Ok(None);
    }
    let Some(mut provenance) = load.topology_provenance.clone() else {
        return Ok(None);
    };

    let evidence = replayable_source_treatments(exchange, source)?;
    let mut current_step = None::<String>;
    let mut current_source = source;
    let mut normalized = if evidence.is_empty() {
        false
    } else {
        let patch = plan_source_edge_treatment_patches(input, source, &evidence, &provenance)?;
        let healed = kernel.edit_source_topology(input, source, &[patch])?;
        let Some((healed_source, healed_provenance)) =
            inspect_single_loaded_source(kernel, healed.as_bytes())?
        else {
            return Ok(None);
        };
        current_source = healed_source;
        provenance = healed_provenance;
        current_step = Some(healed);
        true
    };

    let current_bytes = current_step.as_deref().map_or(input, str::as_bytes);
    if let Some(patch) =
        plan_source_coplanar_normalization_patch(current_bytes, current_source, &provenance)?
    {
        current_step =
            Some(kernel.edit_source_topology(current_bytes, current_source, &[patch])?);
        normalized = true;
    }
    if !normalized {
        return Ok(None);
    }
    let final_step = current_step
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("kernel topology normalization lost its edited STEP"))?;
    let Some(mut host) = recognize_single_healed_host(final_step)? else {
        return Ok(None);
    };
    // Regenerated STEP identities are an internal handoff, not source provenance.
    host.model.provenance.clear();

    if evidence.is_empty() {
        host.source = CadFragmentSource::CoplanarNormalized {
            solid_id: source.id,
        };
        return Ok(Some(host));
    }

    let replay = evidence
        .iter()
        .filter_map(|item| {
            replayable_edge_treatment(item).map(|(profile, edge)| (item, profile, edge))
        })
        .collect::<Vec<_>>();
    if replay.len() != evidence.len() {
        return Ok(None);
    }
    let fragment = replay_edge_treatments(host, source.id, replay)?;
    kernel_replay_is_single_solid(kernel, &fragment).map(|accepted| accepted.then_some(fragment))
}

fn inspect_single_loaded_source<K: SourceStepLoadInspector>(
    kernel: &K,
    input: &[u8],
) -> Result<Option<(SourceEntity, KernelTopologyProvenance)>> {
    let load = kernel.inspect_source_step(input)?;
    let [solid] = load.solids.as_slice() else {
        return Ok(None);
    };
    if solid.failure.is_some() {
        return Ok(None);
    }
    let Some(provenance) = solid.topology_provenance.clone() else {
        return Ok(None);
    };
    Ok(Some((solid.source, provenance)))
}

fn replayable_source_treatments(
    exchange: &Exchange,
    source: SourceEntity,
) -> Result<Vec<crate::edge_treatments::EdgeTreatmentEvidence>> {
    let section = exchange
        .data
        .get(source.data_section)
        .ok_or_else(|| anyhow::anyhow!("missing DATA section {}", source.data_section))?;
    let index = crate::step_graph::build_index(&section.entities);
    let target = HashSet::from([source.id]);
    let mut evidence =
        crate::edge_treatments::detect_edge_treatment_evidence_for_solids_with_index(
            &section.entities,
            &index,
            &target,
        )
        .into_iter()
        .filter(|item| replayable_edge_treatment(item).is_some())
        .collect::<Vec<_>>();
    evidence.sort_by_key(|item| item.treatment_face_id);
    evidence.dedup_by_key(|item| item.treatment_face_id);
    Ok(evidence)
}

fn recognize_single_healed_host(healed_step: &str) -> Result<Option<CadFragment>> {
    let healed = crate::step_io::ParsedExchange::parse(healed_step.as_bytes())?.exchange;
    let [section] = healed.data.as_slice() else {
        return Ok(None);
    };
    let index = crate::step_graph::build_index(&section.entities);
    let solid_ids = section
        .entities
        .iter()
        .filter_map(|entity| {
            let record = crate::step_graph::simple_record(entity)?;
            (record.name == "MANIFOLD_SOLID_BREP").then(|| crate::step_graph::entity_id(entity))
        })
        .collect::<Vec<_>>();
    let [solid_id] = solid_ids.as_slice() else {
        return Ok(None);
    };
    let mut candidates =
        recover_constructive_fragments_from_entities_with_index(&section.entities, &index)?;
    // Monstertruck's STEP re-export is an internal handoff, not a requirement that the raw-entity
    // recognizers understand every equivalent exporter decomposition. ReducedSolid is allowed here only
    // as an immutable analytic view of the topology the kernel has already normalized.
    if let Some(reduced) =
        crate::reduced_brep::ReducedSolid::from_step(*solid_id, &section.entities, &index)
    {
        if let Some(extrusion) =
            crate::solid_extrusions::detect_solid_extrusion_from_reduced(&reduced)
        {
            candidates.push(recover_solid_extrusion_fragment(&extrusion)?);
        }
        if let Some(revolution) = crate::solid_revolutions::detect_solid_revolution_from_reduced(
            &reduced,
            &section.entities,
            &index,
        ) {
            candidates.push(crate::cad_recovery::recover_solid_revolution_fragment(
                &revolution,
            )?);
        }
    }
    let refs = candidates.iter().collect::<Vec<_>>();
    let selected = select_cheapest_whole_solid_fragments(&refs)?;
    Ok(selected
        .get(solid_id)
        .filter(|host| is_constructive_source(&host.source))
        .map(|host| (*host).clone()))
}

fn kernel_replay_is_single_solid<K: CadKernel>(kernel: &K, fragment: &CadFragment) -> Result<bool> {
    let evaluated = kernel.evaluate(&fragment.model, fragment.root)?;
    let summary = kernel.summarize(&evaluated);
    if !summary.geometrically_consistent || summary.shells != 1 {
        return Ok(false);
    }
    let rebuilt = kernel.to_step(&evaluated)?;
    let exchange = crate::step_io::ParsedExchange::parse(rebuilt.as_bytes())?.exchange;
    if exchange.data.len() != 1 {
        return Ok(false);
    }
    Ok(exchange.data[0]
        .entities
        .iter()
        .filter(|entity| {
            crate::step_graph::simple_record(entity)
                .is_some_and(|record| record.name == "MANIFOLD_SOLID_BREP")
        })
        .count()
        == 1)
}

fn apply_kernel_recovery<F>(
    recovery: &mut WholeSolidRecoveryState,
    mut recover: F,
) -> Vec<KernelRecoveryFailure>
where
    F: FnMut(SourceEntity) -> Result<Option<CadFragment>>,
{
    let mut failures = Vec::new();
    for source in recovery.semantically_unrecovered_sources() {
        match recover(source) {
            Ok(Some(fragment)) if source_solid_id(&fragment.source) == source.id => {
                if let Err(error) = recovery.add_candidates([SourceCadFragment {
                    source_solid: source,
                    fragment,
                }]) {
                    failures.push(KernelRecoveryFailure {
                        source,
                        error: error.to_string(),
                    });
                }
            }
            Ok(Some(fragment)) => failures.push(KernelRecoveryFailure {
                source,
                error: format!(
                    "kernel recovery returned local solid #{} while recovering DATA section {:?} solid #{}",
                    source_solid_id(&fragment.source),
                    source.data_section,
                    source.id
                ),
            }),
            Ok(None) => {}
            Err(error) => failures.push(KernelRecoveryFailure {
                source,
                error: error.to_string(),
            }),
        }
    }
    failures
}

pub(crate) fn recover_complete_ir_exchange(
    exchange: &Exchange,
    input: &[u8],
) -> Result<CompleteIr> {
    Ok(
        recover_complete_ir_build_state(exchange, input, ReducedBrepMutationMode::ParserOwned)?
            .finish(COMPLETE_IR_SCHEMA, Vec::new(), None),
    )
}

fn recover_complete_ir_build_state(
    exchange: &Exchange,
    input: &[u8],
    mutation_mode: ReducedBrepMutationMode,
) -> Result<CompleteIrBuildState> {
    let mut signatures = Vec::new();
    let mut patterns = Vec::new();
    let mut body_scan = Vec::new();
    let mut direct_candidates = Vec::new();
    let mut recovery_sections = Vec::with_capacity(exchange.data.len());

    for (data_section, section) in exchange.data.iter().enumerate() {
        let index = crate::step_graph::build_index(&section.entities);
        let recovery_section = RecoverySection {
            data_section,
            entities: &section.entities,
            index,
        };

        for signature in crate::solid_revolutions::detect_solid_surface_signatures_with_index(
            recovery_section.entities,
            &recovery_section.index,
        ) {
            let source = recovery_section.source_entity(signature.solid_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "surface signature refers to missing solid #{} in DATA section {data_section}",
                    signature.solid_id
                )
            })?;
            signatures.push(SurfaceSignatureSummary::from_source(source, &signature));
        }

        let mut section_candidates = recover_constructive_fragments_from_entities_with_index(
            recovery_section.entities,
            &recovery_section.index,
        )?;
        let periodic_chains = crate::periodic_chains::detect_periodic_chains_with_index(
            recovery_section.entities,
            &recovery_section.index,
        );
        section_candidates.extend(recover_periodic_chain_fragments(&periodic_chains)?);
        direct_candidates.extend(recovery_section.qualify_fragments(section_candidates)?);

        patterns.extend(
            crate::patterns::detect_instance_patterns_with_index(
                recovery_section.entities,
                &recovery_section.index,
                1.0e-7,
                4,
            )
            .into_iter()
            .map(|pattern| SourceInstancePattern {
                data_section,
                pattern,
            }),
        );
        body_scan.extend(scan_source_solids(&recovery_section));
        recovery_sections.push(recovery_section);
    }

    body_scan.sort_by_key(|solid| solid.source);
    signatures.sort_by_key(|signature| signature.source);

    let surface_signature_summaries = signatures;
    let instance_pattern_diagnostics = patterns;
    let recovery = recover_whole_solids(
        &recovery_sections,
        &body_scan,
        direct_candidates,
        mutation_mode,
    )?;

    let second = crate::clean_bytes(input, &Options::for_profile(OutputProfile::Compact))?;
    Ok(CompleteIrBuildState {
        idempotent_compact_rewrite: second.bytes == input,
        compact_second_pass_stats: second.stats,
        surface_signature_summaries,
        instance_pattern_diagnostics,
        body_scan,
        recovery,
    })
}

fn recover_whole_solids(
    sections: &[RecoverySection<'_>],
    body_scan: &[SourceSolidSummary],
    direct_candidates: Vec<SourceCadFragment>,
    mutation_mode: ReducedBrepMutationMode,
) -> Result<WholeSolidRecoveryState> {
    let mut recovery = WholeSolidRecoveryState::new(body_scan);
    recovery.add_candidates(direct_candidates)?;

    for provider in WHOLE_SOLID_CANDIDATE_PROVIDERS {
        let targets = recovery.constructively_unresolved_sources();
        if targets.is_empty() {
            break;
        }
        recovery.add_candidates(provider(sections, &targets, mutation_mode)?)?;
    }
    Ok(recovery)
}

fn recover_boundary_feature_candidates(
    sections: &[RecoverySection<'_>],
    targets: &BTreeSet<SourceEntity>,
    mutation_mode: ReducedBrepMutationMode,
) -> Result<Vec<SourceCadFragment>> {
    if mutation_mode == ReducedBrepMutationMode::KernelOwned {
        return Ok(Vec::new());
    }
    let mut candidates = Vec::new();
    for section in sections {
        let local_targets = section.local_target_ids(targets);
        if local_targets.is_empty() {
            continue;
        }
        candidates.extend(section.qualify_fragments(
            recover_boundary_feature_composites_for_section(
                section,
                &local_targets,
                mutation_mode,
            )?,
        )?);
    }
    Ok(candidates)
}

fn recover_reduced_brep_candidates(
    sections: &[RecoverySection<'_>],
    targets: &BTreeSet<SourceEntity>,
    mutation_mode: ReducedBrepMutationMode,
) -> Result<Vec<SourceCadFragment>> {
    let mut candidates = Vec::new();
    for section in sections {
        let local_targets = section.local_target_ids(targets);
        if local_targets.is_empty() {
            continue;
        }
        candidates.extend(section.qualify_fragments(recover_reduced_brep_fragments(
            section.entities,
            &section.index,
            Some(&local_targets),
            mutation_mode,
        )?)?);
    }
    Ok(candidates)
}

fn recover_boundary_feature_composites_for_section(
    section: &RecoverySection<'_>,
    target_solid_ids: &BTreeSet<u64>,
    mutation_mode: ReducedBrepMutationMode,
) -> Result<Vec<CadFragment>> {
    let entities = section.entities;
    let initial_index = &section.index;
    let mut keyed_targets = BTreeMap::<crate::shape_identity::ShapeKey, Vec<_>>::new();
    let mut representative_targets = BTreeSet::new();
    let mut recovery_groups = HashMap::new();
    for &solid_id in target_solid_ids {
        if let Some(identity) =
            crate::shape_identity::solid_identity(solid_id, entities, initial_index)
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

    let mut active_targets =
        LocalRecoveryTargets::from_solids(&representative_targets, entities, initial_index)?;
    if active_targets.is_empty() {
        return Ok(Vec::new());
    }

    let mut journal = ParserPeelJournal::default();
    let recovered = recover_local_reductions_to_fixed_point(
        &mut journal,
        entities,
        initial_index,
        &mut active_targets,
        mutation_mode,
    )?;
    let out = recovered.into_values().collect::<Vec<_>>();

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
                    initial_index,
                )? {
                    expanded.push(instance);
                }
            }
        }
        expanded.push(fragment);
    }
    Ok(expanded)
}

fn recover_local_reductions_to_fixed_point(
    journal: &mut ParserPeelJournal,
    source: &[EntityInstance],
    index: &HashMap<u64, usize>,
    active_targets: &mut LocalRecoveryTargets,
    mutation_mode: ReducedBrepMutationMode,
) -> Result<BTreeMap<u64, CadFragment>> {
    let mut recovered = BTreeMap::new();
    for _ in 0..MAX_LOCAL_REDUCTION_ROUNDS {
        let view = journal.view(source, index)?;
        if !recover_boundary_feature_pass(journal, view.as_ref(), index, active_targets)? {
            return Ok(recovered);
        }
        let healed = journal.view(source, index)?;
        for fragment in recover_replayed_boundary_hosts(
            journal,
            healed.as_ref(),
            index,
            &active_targets.solid_ids,
            mutation_mode,
        )? {
            recovered.insert(source_solid_id(&fragment.source), fragment);
        }
        active_targets.retain_unrecovered(&recovered);
        if active_targets.is_empty() {
            return Ok(recovered);
        }
    }
    bail!("local recovery did not reach a fixed point after {MAX_LOCAL_REDUCTION_ROUNDS} rounds")
}

fn recover_replayed_boundary_hosts(
    journal: &ParserPeelJournal,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    target_solid_ids: &BTreeSet<u64>,
    mutation_mode: ReducedBrepMutationMode,
) -> Result<Vec<CadFragment>> {
    if target_solid_ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut host_candidates =
        recover_constructive_fragments_from_entities_with_index(entities, index)?;
    host_candidates.extend(recover_reduced_brep_fragments(
        entities,
        index,
        Some(target_solid_ids),
        mutation_mode,
    )?);
    let host_refs = host_candidates.iter().collect::<Vec<_>>();
    let selected_hosts = select_cheapest_whole_solid_fragments(&host_refs)?;
    let mut out = Vec::new();
    for &solid_id in target_solid_ids {
        let families = journal.boundary_features(solid_id);
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
    Ok(out)
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
    journal: &mut ParserPeelJournal,
    current: &[EntityInstance],
    index: &HashMap<u64, usize>,
    targets: &LocalRecoveryTargets,
) -> Result<bool> {
    let pass_index = journal.next_boundary_pass_index();
    let target_shell_ids = targets.shell_ids();
    let report = crate::planar_features::diagnose_boundary_features_for_shells_with_index(
        current,
        index,
        &target_shell_ids,
    );
    let target_family_indices =
        crate::planar_features::leaf_feature_family_indices(&report.families)
            .into_iter()
            .filter(|&candidate| {
                targets
                    .shell_to_solid
                    .contains_key(&report.families[candidate].shell_id)
            })
            .collect::<Vec<_>>();
    if target_family_indices.is_empty() {
        return Ok(false);
    }

    let prepared = prepare_boundary_feature_reduction(
        current,
        index,
        targets,
        &target_shell_ids,
        &report,
        &target_family_indices,
        pass_index,
    )?;
    if prepared.family_indices.is_empty() {
        return Ok(false);
    }
    let plan = crate::planar_features::plan_boundary_feature_family_peel(
        current,
        index,
        &report,
        &prepared.family_indices,
    )?;
    journal.commit(prepared.steps, plan, current, index)
}

fn prepare_boundary_feature_reduction(
    source_entities: &[EntityInstance],
    source_index: &HashMap<u64, usize>,
    targets: &LocalRecoveryTargets,
    target_shell_ids: &HashSet<u64>,
    report: &crate::planar_features::BoundaryFeatureDiagnostics,
    target_family_indices: &[usize],
    pass_index: usize,
) -> Result<PreparedBoundaryReduction> {
    // This clone is the disposable proof-only tool materialization. The source
    // is immutable, and neither consumer receives a mutable topology owner.
    let mut tool_entities = source_entities.to_vec();
    let (tool_report, materialized) =
        crate::planar_features::materialize_boundary_feature_tools_for_analysis_for_shells(
            &mut tool_entities,
            1,
            target_shell_ids,
        )?;
    if tool_report.families.len() != report.families.len() {
        bail!("boundary-feature tool materialization changed family indexing");
    }
    let tool_candidates = recover_constructive_fragments_from_entities(&tool_entities)?;
    let tool_refs = tool_candidates.iter().collect::<Vec<_>>();
    let selected_tools = select_cheapest_whole_solid_fragments(&tool_refs)?;
    let mut peelable_family_indices = Vec::new();
    let mut pending_steps = Vec::new();

    for &family_index in target_family_indices {
        let family = &report.families[family_index];
        let Some(source_solid_id) = targets.solid_for_shell(family.shell_id) else {
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
        if !regularize_boundary_feature_tool(&mut tool, family, source_entities, source_index) {
            continue;
        }
        pending_steps.push((
            source_solid_id,
            LocalReductionStep::BoundaryFeature(RecoveredBoundaryFeatureFamily {
                pass_index,
                polarity: family.polarity,
                tool,
                source_member,
                members: family.members.clone(),
                original_source_ids: true,
            }),
        ));
        peelable_family_indices.push(family_index);
    }
    Ok(PreparedBoundaryReduction {
        steps: pending_steps,
        family_indices: peelable_family_indices,
    })
}

fn regularize_boundary_feature_tool(
    tool: &mut CadFragment,
    family: &crate::planar_features::BoundaryFeatureFamilyEvidence,
    source_entities: &[EntityInstance],
    source_index: &HashMap<u64, usize>,
) -> bool {
    if !matches!(tool.source, CadFragmentSource::SolidExtrusion { .. }) {
        return true;
    }
    let Some(host_outwards) =
        feature_host_outward_directions(&family.host_face_ids, source_entities, source_index)
    else {
        return false;
    };
    regularize_boundary_tool_for_boolean(tool, family.polarity, &host_outwards).is_ok()
}

fn recover_constructive_fragments_from_entities(
    entities: &[EntityInstance],
) -> Result<Vec<CadFragment>> {
    let index = crate::step_graph::build_index(entities);
    recover_constructive_fragments_from_entities_with_index(entities, &index)
}

fn recover_constructive_fragments_from_entities_with_index(
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Result<Vec<CadFragment>> {
    let extrusions = crate::solid_extrusions::detect_solid_extrusions_with_index(entities, index);
    let revolutions =
        crate::solid_revolutions::detect_solid_revolutions_with_index(entities, index);
    let radial_slots =
        crate::solid_revolutions::detect_radial_slot_revolutions_with_index(entities, index);
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

fn recover_reduced_brep_fragments(
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    target_solid_ids: Option<&BTreeSet<u64>>,
    mutation_mode: ReducedBrepMutationMode,
) -> Result<Vec<CadFragment>> {
    if mutation_mode == ReducedBrepMutationMode::KernelOwned {
        return Ok(Vec::new());
    }
    let candidate_solid_ids = entities
        .iter()
        .filter_map(|entity| {
            let record = crate::step_graph::simple_record(entity)?;
            (record.name == "MANIFOLD_SOLID_BREP").then(|| crate::step_graph::entity_id(entity))
        })
        .filter(|solid_id| target_solid_ids.is_none_or(|targets| targets.contains(solid_id)))
        .collect::<HashSet<_>>();
    if candidate_solid_ids.is_empty() {
        return Ok(Vec::new());
    }

    let evidence = crate::edge_treatments::detect_edge_treatment_evidence_for_solids_with_index(
        entities,
        index,
        &candidate_solid_ids,
    );
    let mut by_solid = BTreeMap::<u64, Vec<_>>::new();
    for item in &evidence {
        if replayable_edge_treatment(item).is_some() {
            by_solid.entry(item.solid_id).or_default().push(item);
        }
    }

    let mut out = Vec::new();
    for solid_id in candidate_solid_ids {
        let Some(mut reduced) =
            crate::reduced_brep::ReducedSolid::from_step(solid_id, entities, index)
        else {
            continue;
        };
        let mut replay = Vec::new();
        if let Some(items) = by_solid.get_mut(&solid_id) {
            items.sort_by_key(|item| item.treatment_face_id);
            for item in items.iter().copied() {
                let Some((profile, edge)) = replayable_edge_treatment(item) else {
                    continue;
                };
                if reduced.remove_edge_treatment(item).is_some() {
                    replay.push((item, profile, edge));
                }
            }
        }
        let coalesced = reduced.coalesce_coplanar_faces();
        if replay.is_empty() && !coalesced {
            continue;
        }

        if let Some(extrusion) =
            crate::solid_extrusions::detect_solid_extrusion_from_reduced(&reduced)
        {
            let fragment = recover_solid_extrusion_fragment(&extrusion)?;
            out.push(replay_edge_treatments(fragment, solid_id, replay.clone())?);
        }
        if let Some(revolution) = crate::solid_revolutions::detect_solid_revolution_from_reduced(
            &reduced, entities, index,
        ) {
            let fragment = crate::cad_recovery::recover_solid_revolution_fragment(&revolution)?;
            out.push(replay_edge_treatments(fragment, solid_id, replay)?);
        }
    }
    Ok(out)
}

fn replay_edge_treatments(
    mut fragment: CadFragment,
    solid_id: u64,
    mut replay: Vec<(
        &crate::edge_treatments::EdgeTreatmentEvidence,
        EdgeTreatmentProfile,
        EdgeTreatmentEdge,
    )>,
) -> Result<CadFragment> {
    replay.sort_by_key(|(item, _, _)| item.treatment_face_id);
    let treatment_face_ids = replay
        .iter()
        .map(|(item, _, _)| item.treatment_face_id)
        .collect::<Vec<_>>();
    fragment.model.roots.clear();
    let mut root = fragment.root;
    for (item, profile, edge) in replay {
        root = fragment.model.add_node(CadNode::EdgeTreatment {
            child: root,
            profile,
            edges: vec![edge],
        });
        fragment.model.set_provenance(
            root,
            Provenance {
                source_entity_ids: vec![solid_id, item.treatment_face_id],
                proof: ProofStatus::WithinTolerance,
                max_residual_mm: Some(item.max_boundary_residual_mm),
            },
        )?;
    }
    fragment.model.add_root(root)?;
    fragment.root = root;
    if !treatment_face_ids.is_empty() {
        fragment.source = CadFragmentSource::EdgeTreatmentComposite {
            solid_id,
            treatment_face_ids,
        };
    }
    fragment.model.validate()?;
    Ok(fragment)
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
        crate::edge_treatments::EdgeTreatmentVirtualEdge::Circle {
            center_mm,
            normal,
            radius_mm,
        } => EdgeSelector::Circle {
            center_mm: *center_mm,
            normal: *normal,
            radius_mm: *radius_mm,
        },
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
        crate::edge_treatments::EdgeTreatmentKind::CylindricalFillet
        | crate::edge_treatments::EdgeTreatmentKind::ToroidalFillet => {
            let radius = item.inferred_radius_mm?;
            if !radius.is_finite() || radius <= 0.0 {
                return None;
            }
            (EdgeTreatmentProfile::Round, radius)
        }
        crate::edge_treatments::EdgeTreatmentKind::ConicalChamfer => return None,
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
        let exemplar_sources = if family.original_source_ids {
            feature_member_source_ids(source_solid_id, &family.source_member)
        } else {
            vec![source_solid_id]
        };
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
                        source_entity_ids: if family.original_source_ids {
                            feature_member_source_ids(source_solid_id, member)
                        } else {
                            vec![source_solid_id]
                        },
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
            if family.original_source_ids {
                sources.extend(feature_member_source_ids(source_solid_id, member));
            }
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

fn scan_source_solids(section: &RecoverySection<'_>) -> Vec<SourceSolidSummary> {
    let mut out = Vec::new();
    for entity in section.entities {
        let Some(record) = crate::step_graph::simple_record(entity) else {
            continue;
        };
        if record.name != "MANIFOLD_SOLID_BREP" {
            continue;
        }
        let solid_id = crate::step_graph::entity_id(entity);
        let Some(source) = section.source_entity(solid_id) else {
            continue;
        };
        let closure = crate::shape_identity::semantic_solid_closure(
            solid_id,
            section.entities,
            &section.index,
        )
        .unwrap_or_else(|| {
            crate::step_entities::closure_from(solid_id, section.entities, &section.index)
        });
        let mut faces = 0usize;
        let mut edges = 0usize;
        for id in &closure {
            let Some(&entity_index) = section.index.get(id) else {
                continue;
            };
            let Some(child) = crate::step_graph::simple_record(&section.entities[entity_index])
            else {
                continue;
            };
            match child.name.as_str() {
                "ADVANCED_FACE" => faces += 1,
                "EDGE_CURVE" => edges += 1,
                _ => {}
            }
        }
        out.push(SourceSolidSummary {
            source,
            faces,
            edges,
            closure_entities: closure.len(),
        });
    }
    out
}

const fn source_solid_id(source: &CadFragmentSource) -> u64 {
    match source {
        CadFragmentSource::SolidExtrusion { solid_id, .. }
        | CadFragmentSource::SolidRevolution { solid_id, .. }
        | CadFragmentSource::RadialSlotRevolution { solid_id, .. }
        | CadFragmentSource::SolidSweep { solid_id, .. }
        | CadFragmentSource::RevolvedRoundTail { solid_id, .. }
        | CadFragmentSource::PeriodicChain { solid_id, .. }
        | CadFragmentSource::ExtrusionCoreUnion { solid_id, .. }
        | CadFragmentSource::BoundaryFeatureComposite { solid_id, .. }
        | CadFragmentSource::EdgeTreatmentComposite { solid_id, .. }
        | CadFragmentSource::CoplanarNormalized { solid_id } => *solid_id,
    }
}

const fn is_constructive_source(source: &CadFragmentSource) -> bool {
    !matches!(source, CadFragmentSource::PeriodicChain { .. })
}

const fn whole_solid_candidate_is_preferred(
    existing_source: &CadFragmentSource,
    existing_complexity: u64,
    candidate_source: &CadFragmentSource,
    candidate_complexity: u64,
) -> bool {
    match (
        is_constructive_source(existing_source),
        is_constructive_source(candidate_source),
    ) {
        (true, false) => false,
        (false, true) => true,
        _ => candidate_complexity < existing_complexity,
    }
}

fn select_cheapest_whole_solid_fragments<'a>(
    candidates: &[&'a CadFragment],
) -> Result<BTreeMap<u64, &'a CadFragment>> {
    let mut selected = BTreeMap::<u64, (&CadFragment, u64)>::new();
    for &fragment in candidates {
        let solid_id = source_solid_id(&fragment.source);
        let complexity = fragment.model.complexity_score(fragment.root)?;
        let replace =
            selected
                .get(&solid_id)
                .is_none_or(|(existing_fragment, existing_complexity)| {
                    whole_solid_candidate_is_preferred(
                        &existing_fragment.source,
                        *existing_complexity,
                        &fragment.source,
                        complexity,
                    )
                });
        if replace {
            selected.insert(solid_id, (fragment, complexity));
        }
    }
    Ok(selected
        .into_iter()
        .map(|(solid_id, (fragment, _))| (solid_id, fragment))
        .collect())
}

fn selected_category<F>(
    selected: &BTreeMap<SourceEntity, &CadFragment>,
    predicate: F,
) -> Vec<SourceCadFragment>
where
    F: Fn(&CadFragmentSource) -> bool,
{
    selected
        .iter()
        .filter(|(_, fragment)| predicate(&fragment.source))
        .map(|(&source, fragment)| SourceCadFragment {
            source_solid: source,
            fragment: (*fragment).clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cad_ir::{CadModel, CadNode, Primitive};

    const fn test_source_in(data_section: usize, solid_id: u64) -> SourceEntity {
        SourceEntity {
            data_section,
            ordinal: 0,
            id: solid_id,
        }
    }

    const fn test_source(solid_id: u64) -> SourceEntity {
        test_source_in(0, solid_id)
    }

    fn qualified_fragment_in(data_section: usize, fragment: CadFragment) -> SourceCadFragment {
        SourceCadFragment {
            source_solid: test_source_in(data_section, source_solid_id(&fragment.source)),
            fragment,
        }
    }

    fn qualified_fragment(fragment: CadFragment) -> SourceCadFragment {
        qualified_fragment_in(0, fragment)
    }

    fn report_with_unrecovered(solid_ids: &[u64]) -> CompleteIr {
        let solids = solid_ids
            .iter()
            .map(|&solid_id| SourceSolidSummary {
                source: test_source(solid_id),
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
            edge_treatment_composites: Vec::new(),
            coplanar_normalized_solids: Vec::new(),
            extrusion_core_unions: Vec::new(),
            kernel_recovery_failures: Vec::new(),
            kernel_source_load: None,
            unrecovered_solids: solids.clone(),
            body_scan: BodyScanSummary { solids },
            body_count: solid_ids.len(),
            constructively_recovered_solid_count: 0,
            semantically_recovered_solid_count: 0,
            unrecovered_solid_count: solid_ids.len(),
        }
    }

    fn build_state_with_unrecovered(solid_ids: &[u64]) -> CompleteIrBuildState {
        let report = report_with_unrecovered(solid_ids);
        let body_scan = report.body_scan.solids;
        let recovery = WholeSolidRecoveryState::new(&body_scan);
        CompleteIrBuildState {
            idempotent_compact_rewrite: report.idempotent_compact_rewrite,
            compact_second_pass_stats: report.compact_second_pass_stats,
            surface_signature_summaries: report.surface_signature_summaries,
            instance_pattern_diagnostics: report.instance_pattern_diagnostics,
            body_scan,
            recovery,
        }
    }

    fn apply_test_kernel_recovery<F>(solid_ids: &[u64], mut recover: F) -> CompleteIr
    where
        F: FnMut(u64) -> Result<Option<CadFragment>>,
    {
        let mut build = build_state_with_unrecovered(solid_ids);
        let failures = apply_kernel_recovery(&mut build.recovery, |source| recover(source.id));
        build.finish(KERNEL_COMPLETE_IR_SCHEMA, failures, None)
    }

    fn constructive_fragment(solid_id: u64, boxes: usize) -> Result<CadFragment> {
        if boxes == 0 {
            bail!("test fragment requires at least one box");
        }
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
        model.add_root(root)?;
        Ok(CadFragment {
            source: CadFragmentSource::SolidExtrusion {
                solid_id,
                cap_face_ids: [1, 2],
                side_face_ids: Vec::new(),
            },
            model,
            root,
        })
    }

    fn test_treatment_evidence(
        solid_id: u64,
        treatment_face_id: u64,
    ) -> crate::edge_treatments::EdgeTreatmentEvidence {
        use crate::edge_treatments::{
            EdgeTreatmentCarrier, EdgeTreatmentEvidence, EdgeTreatmentEvidenceStrength,
            EdgeTreatmentKind, EdgeTreatmentVirtualEdge,
        };
        EdgeTreatmentEvidence {
            solid_id,
            treatment_face_id,
            kind: EdgeTreatmentKind::PlanarChamfer,
            strength: EdgeTreatmentEvidenceStrength::AnalyticTangency,
            carrier_pair_unique: true,
            treatment_support: "plane".to_owned(),
            carriers: [
                EdgeTreatmentCarrier {
                    face_id: treatment_face_id + 1,
                    support: "plane".to_owned(),
                    shared_edge_ids: vec![treatment_face_id + 10],
                },
                EdgeTreatmentCarrier {
                    face_id: treatment_face_id + 2,
                    support: "plane".to_owned(),
                    shared_edge_ids: vec![treatment_face_id + 11],
                },
            ],
            inferred_radius_mm: None,
            inferred_setbacks_mm: Some([0.5, 0.5]),
            inferred_angle_rad: Some(std::f64::consts::FRAC_PI_4),
            carrier_angle_rad: std::f64::consts::FRAC_PI_2,
            virtual_edge: EdgeTreatmentVirtualEdge::Segment {
                start_mm: [0.0, 0.0, 0.0],
                end_mm: [0.0, 0.0, 1.0],
            },
            max_boundary_residual_mm: 1.0e-8,
        }
    }

    #[test]
    fn replayed_edge_treatments_are_first_class_composites() -> Result<()> {
        let solid_id = 42;
        let evidence = test_treatment_evidence(solid_id, 77);
        let fragment = replay_edge_treatments(
            constructive_fragment(solid_id, 1)?,
            solid_id,
            vec![(
                &evidence,
                EdgeTreatmentProfile::Chamfer,
                EdgeTreatmentEdge {
                    selector: EdgeSelector::Segment {
                        start_mm: [0.0, 0.0, 0.0],
                        end_mm: [0.0, 0.0, 1.0],
                    },
                    radius_mm: 0.5,
                },
            )],
        )?;
        assert_eq!(
            fragment.source,
            CadFragmentSource::EdgeTreatmentComposite {
                solid_id,
                treatment_face_ids: vec![77],
            }
        );
        assert!(matches!(
            fragment.model.node(fragment.root)?,
            CadNode::EdgeTreatment { .. }
        ));
        assert_eq!(
            fragment.model.provenance[&fragment.root].source_entity_ids,
            vec![solid_id, 77]
        );
        Ok(())
    }

    fn boundary_reduction_step(solid_id: u64) -> Result<(u64, LocalReductionStep)> {
        let member = BoundaryFeatureInstanceEvidence {
            face_ids: vec![10],
            center_mm: [0.0; 3],
            normalized_quarter: 0,
            interface_bound_ids: vec![11],
        };
        Ok((
            solid_id,
            LocalReductionStep::BoundaryFeature(RecoveredBoundaryFeatureFamily {
                pass_index: 0,
                polarity: FeaturePolarity::Subtractive,
                tool: constructive_fragment(99, 1)?,
                source_member: member.clone(),
                members: vec![member],
                original_source_ids: true,
            }),
        ))
    }

    fn descriptive_fragment(solid_id: u64, boxes: usize) -> Result<CadFragment> {
        if boxes == 0 {
            bail!("test fragment requires at least one box");
        }
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
        model.add_root(root)?;
        Ok(CadFragment {
            source: CadFragmentSource::PeriodicChain { solid_id, sites: 2 },
            model,
            root,
        })
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
    fn parser_only_boss_still_recovers_the_constructive_feature_composite() -> Result<()> {
        let input = include_bytes!("../validation/fixtures/boundary_boss_extrusion.step");
        let report = recover_complete_ir_bytes(input)?;
        assert_eq!(report.schema, COMPLETE_IR_SCHEMA);
        assert_eq!(report.body_count, 1);
        assert_eq!(report.constructively_recovered_solid_count, 1);
        assert_eq!(report.semantically_recovered_solid_count, 1);
        assert_eq!(report.unrecovered_solid_count, 0);
        let [composite] = report.boundary_feature_composites.as_slice() else {
            bail!("parser-only recovery lost its proven boss feature");
        };
        assert!(matches!(
            composite.fragment.source,
            CadFragmentSource::BoundaryFeatureComposite {
                passes: 1,
                feature_families: 1,
                feature_instances: 1,
                ..
            }
        ));
        composite.fragment.model.validate()?;
        Ok(())
    }

    #[test]
    fn parser_only_nested_boss_preserves_two_proven_fixed_point_passes() -> Result<()> {
        let input = include_bytes!("../validation/fixtures/boundary_boss_nested.step");
        let report = recover_complete_ir_bytes(input)?;
        assert_eq!(report.schema, COMPLETE_IR_SCHEMA);
        assert_eq!(report.body_count, 1);
        assert_eq!(report.constructively_recovered_solid_count, 1);
        assert_eq!(report.unrecovered_solid_count, 0);
        let [composite] = report.boundary_feature_composites.as_slice() else {
            bail!("nested boss lost its parser-only boundary composite");
        };
        assert!(matches!(
            composite.fragment.source,
            CadFragmentSource::BoundaryFeatureComposite {
                passes: 2,
                feature_families: 2,
                feature_instances: 2,
                ..
            }
        ));
        composite.fragment.model.validate()?;
        Ok(())
    }

    #[test]
    fn parser_peel_journal_is_atomic_and_keeps_original_step_immutable() -> Result<()> {
        let input = include_bytes!("../validation/fixtures/boundary_boss_extrusion.step");
        let exchange = crate::step_io::ParsedExchange::parse(input)?.exchange;
        let entities = &exchange.data[0].entities;
        let original = entities.clone();
        let index = crate::step_graph::build_index(entities);
        let report = crate::planar_features::diagnose_boundary_features(entities);
        let selected = crate::planar_features::leaf_feature_family_indices(&report.families);
        assert_eq!(selected.len(), 1);
        let plan = crate::planar_features::plan_boundary_feature_family_peel(
            entities, &index, &report, &selected,
        )?;
        assert_eq!(plan.stats.families, 1);
        assert_eq!(plan.stats.faces_removed_from_shells, 5);
        assert_eq!(plan.stats.interface_bounds_healed, 1);

        let mut journal = ParserPeelJournal::default();
        assert!(matches!(journal.view(entities, &index)?, Cow::Borrowed(_)));
        assert!(!journal.commit(Vec::new(), plan.clone(), entities, &index)?);

        let mut invalid = plan.clone();
        invalid.host_bound_removals[0].bound_ids.push(u64::MAX);
        assert!(
            journal
                .commit(vec![boundary_reduction_step(7)?], invalid, entities, &index)
                .is_err()
        );
        assert!(journal.plans.is_empty());
        assert!(journal.replay.is_empty());

        assert!(journal.commit(
            vec![boundary_reduction_step(7)?],
            plan.clone(),
            entities,
            &index
        )?);
        assert_eq!(journal.boundary_features(7).len(), 1);
        assert_eq!(journal.next_boundary_pass_index(), 1);
        let projected = journal.view(entities, &index)?;
        assert!(matches!(projected, Cow::Owned(_)));
        assert_eq!(entities, &original);
        let shell = plan.shell_face_removals[0].shell_id;
        let host = plan.host_bound_removals[0].host_face_id;
        let refs = crate::brep::ref_list_param;
        let original_shell = refs(shell, 1, entities, &index)
            .ok_or_else(|| anyhow::anyhow!("source shell unreadable"))?;
        let projected_shell = refs(shell, 1, projected.as_ref(), &index)
            .ok_or_else(|| anyhow::anyhow!("projected shell unreadable"))?;
        let original_host = refs(host, 1, entities, &index)
            .ok_or_else(|| anyhow::anyhow!("source host unreadable"))?;
        let projected_host = refs(host, 1, projected.as_ref(), &index)
            .ok_or_else(|| anyhow::anyhow!("projected host unreadable"))?;
        assert_eq!(original_shell.len(), projected_shell.len() + 5);
        assert_eq!(original_host.len(), projected_host.len() + 1);
        assert!(
            journal
                .commit(
                    vec![boundary_reduction_step(7)?],
                    plan,
                    projected.as_ref(),
                    &index
                )
                .is_err()
        );
        assert_eq!(journal.plans.len(), 1);
        assert_eq!(journal.boundary_features(7).len(), 1);
        Ok(())
    }

    #[test]
    fn recovery_state_unifies_selection_and_unresolved_tracking() -> Result<()> {
        let report = report_with_unrecovered(&[7, 8]);
        let mut state = WholeSolidRecoveryState::new(&report.body_scan.solids);
        assert_eq!(
            state.constructively_unresolved_sources(),
            BTreeSet::from([test_source(7), test_source(8)])
        );

        assert_eq!(
            state.add_candidates([qualified_fragment(descriptive_fragment(7, 1)?)])?,
            1
        );
        assert_eq!(
            state.constructively_unresolved_sources(),
            BTreeSet::from([test_source(7), test_source(8)])
        );
        assert_eq!(
            state.semantically_unrecovered_sources(),
            BTreeSet::from([test_source(8)])
        );

        assert_eq!(
            state.add_candidates([qualified_fragment(constructive_fragment(7, 4)?)])?,
            1
        );
        assert_eq!(
            state.constructively_unresolved_sources(),
            BTreeSet::from([test_source(8)])
        );
        let revision = state.revision;

        assert_eq!(
            state.add_candidates([qualified_fragment(constructive_fragment(7, 6)?)])?,
            0
        );
        assert_eq!(state.revision, revision);
        assert_eq!(
            state.add_candidates([qualified_fragment(constructive_fragment(7, 1)?)])?,
            1
        );
        assert!(state.revision > revision);
        assert_eq!(
            state.selected_fragments()[&test_source(7)]
                .model
                .complexity_score(state.selected_fragments()[&test_source(7)].root)?,
            1
        );
        Ok(())
    }

    #[test]
    fn recovery_state_keeps_same_local_id_from_distinct_data_sections() -> Result<()> {
        let body_scan = vec![
            SourceSolidSummary {
                source: test_source_in(0, 7),
                faces: 0,
                edges: 0,
                closure_entities: 0,
            },
            SourceSolidSummary {
                source: test_source_in(1, 7),
                faces: 0,
                edges: 0,
                closure_entities: 0,
            },
        ];
        let mut state = WholeSolidRecoveryState::new(&body_scan);
        assert_eq!(
            state.add_candidates([
                qualified_fragment_in(0, constructive_fragment(7, 1)?),
                qualified_fragment_in(1, constructive_fragment(7, 1)?),
            ])?,
            2
        );
        assert_eq!(state.selected_fragments().len(), 2);
        assert!(state.semantically_unrecovered_sources().is_empty());
        assert!(
            state
                .selected_fragments()
                .contains_key(&test_source_in(0, 7))
        );
        assert!(
            state
                .selected_fragments()
                .contains_key(&test_source_in(1, 7))
        );
        Ok(())
    }

    #[test]
    fn recovery_state_rejects_candidate_for_unknown_solid() -> Result<()> {
        let report = report_with_unrecovered(&[7]);
        let mut state = WholeSolidRecoveryState::new(&report.body_scan.solids);
        let result = state.add_candidates([qualified_fragment(constructive_fragment(8, 1)?)]);
        let Err(error) = result else {
            bail!("unknown source solid unexpectedly succeeded");
        };
        assert!(error.to_string().contains("missing source solid"));
        Ok(())
    }

    const DUPLICATE_SOLID_ID_MULTI_DATA: &[u8] = br"ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('duplicate-id regression'),'1');
FILE_NAME('duplicate.step','2026-11-02T00:00:00',(''),(''),'step-redox','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=MANIFOLD_SOLID_BREP('',#2);
#2=CLOSED_SHELL('',());
ENDSEC;
DATA;
#1=MANIFOLD_SOLID_BREP('',#2);
#2=CLOSED_SHELL('',());
ENDSEC;
END-ISO-10303-21;
";

    #[test]
    fn duplicate_solid_ids_in_distinct_data_sections_remain_distinct_sources() -> Result<()> {
        let report = recover_complete_ir_bytes(DUPLICATE_SOLID_ID_MULTI_DATA)?;
        assert_eq!(report.schema, COMPLETE_IR_SCHEMA);
        assert_eq!(report.body_count, 2);
        assert_eq!(report.unrecovered_solid_count, 2);
        let sources = report
            .body_scan
            .solids
            .iter()
            .map(|solid| solid.source)
            .collect::<Vec<_>>();
        assert_eq!(
            sources,
            vec![
                SourceEntity {
                    data_section: 0,
                    ordinal: 0,
                    id: 1,
                },
                SourceEntity {
                    data_section: 1,
                    ordinal: 0,
                    id: 1,
                },
            ]
        );
        assert_eq!(
            report
                .unrecovered_solids
                .iter()
                .map(|solid| solid.source)
                .collect::<Vec<_>>(),
            sources
        );
        Ok(())
    }

    #[cfg(feature = "cad-kernel-monstertruck")]
    #[test]
    fn kernel_complete_ir_handles_duplicate_solid_ids_across_data_sections() -> Result<()> {
        let kernel = crate::cad_kernel::monstertruck::MonstertruckKernel::project_standard();
        let report = recover_complete_ir_with_kernel_bytes(DUPLICATE_SOLID_ID_MULTI_DATA, &kernel)?;
        assert_eq!(report.schema, KERNEL_COMPLETE_IR_SCHEMA);
        assert_eq!(report.body_count, 2);
        assert_eq!(report.unrecovered_solid_count, 2);
        let evidence = report
            .kernel_source_load
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("kernel v14 omitted source-load evidence"))?;
        assert_eq!(evidence.sections.len(), 2);
        assert_eq!(evidence.solids.len(), 2);
        assert_eq!(
            evidence
                .solids
                .iter()
                .map(|solid| (solid.source.data_section, solid.source.id))
                .collect::<Vec<_>>(),
            vec![(0, 1), (1, 1)]
        );
        Ok(())
    }

    #[cfg(feature = "cad-kernel-monstertruck")]
    #[test]
    fn kernel_v15_square_boss_peel_has_complete_source_and_kernel_proofs() -> Result<()> {
        let input = include_bytes!("../validation/fixtures/boundary_boss_extrusion.step");
        let kernel = crate::cad_kernel::monstertruck::MonstertruckKernel::project_standard();
        let source_load = kernel.inspect_source_step(input)?;
        let [loaded] = source_load.solids.as_slice() else {
            bail!("single-boss fixture must import as exactly one source solid");
        };
        let provenance = loaded
            .topology_provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("boss fixture did not retain kernel StableIds"))?;
        let exchange = crate::step_io::ParsedExchange::parse(input)?.exchange;
        let section = &exchange.data[loaded.source.data_section];
        let index = crate::step_graph::build_index(&section.entities);
        let shell = crate::brep::manifold_shell(loaded.source.id, &section.entities, &index)
            .ok_or_else(|| anyhow::anyhow!("boss fixture has no solid shell"))?;
        let report = crate::planar_features::diagnose_boundary_features_for_shells_with_index(
            &section.entities,
            &index,
            &HashSet::from([shell]),
        );
        let selected = crate::planar_features::leaf_feature_family_indices(&report.families);
        assert_eq!(
            report
                .families
                .iter()
                .filter(|family| {
                    family.polarity == FeaturePolarity::Additive && family.faces_per_instance == 5
                })
                .count(),
            1,
            "unexpected boss families: {report:?}"
        );
        let targets = LocalRecoveryTargets::from_solids(
            &BTreeSet::from([loaded.source.id]),
            &section.entities,
            &index,
        )?;
        let prepared = prepare_boundary_feature_reduction(
            &section.entities,
            &index,
            &targets,
            &HashSet::from([shell]),
            &report,
            &selected,
            0,
        )?;
        assert_eq!(
            prepared.family_indices.len(),
            1,
            "boss evidence did not produce exactly one constructive tool: {report:?}"
        );
        let patch = plan_source_boundary_feature_peel_patch(
            input,
            loaded.source,
            &report,
            &prepared.family_indices,
            provenance,
        )?;
        assert_eq!(patch.remove_faces.len(), 5);
        assert_eq!(patch.face_boundary_replacements.len(), 1);
        assert_eq!(patch.require_unused_edges.len(), 4);
        let healed = kernel.edit_source_topology(input, loaded.source, &[patch])?;
        let healed_report = kernel.inspect_source_step(healed.as_bytes())?;
        assert_eq!(healed_report.solids.len(), 1);
        let host = recognize_single_healed_host(&healed)?
            .ok_or_else(|| anyhow::anyhow!("boss peel did not expose an analytic box host"))?;
        assert!(matches!(
            host.source,
            CadFragmentSource::SolidExtrusion { .. }
        ));
        Ok(())
    }

    #[cfg(feature = "cad-kernel-monstertruck")]
    #[test]
    fn kernel_v15_replays_square_boss_through_complete_ir() -> Result<()> {
        let input = include_bytes!("../validation/fixtures/boundary_boss_extrusion.step");
        let kernel = crate::cad_kernel::monstertruck::MonstertruckKernel::project_standard();
        let report = recover_complete_ir_with_kernel_bytes(input, &kernel)?;
        assert_eq!(report.schema, KERNEL_COMPLETE_IR_SCHEMA);
        assert_eq!(
            report.kernel_recovery_failures,
            Vec::<KernelRecoveryFailure>::new()
        );
        assert_eq!(report.unrecovered_solid_count, 0);
        assert_eq!(report.boundary_feature_composites.len(), 1);
        assert!(matches!(
            report.boundary_feature_composites[0].fragment.source,
            CadFragmentSource::BoundaryFeatureComposite {
                passes: 1,
                feature_families: 1,
                feature_instances: 1,
                ..
            }
        ));
        Ok(())
    }

    #[cfg(feature = "cad-kernel-monstertruck")]
    #[test]
    fn kernel_v14_owns_coplanar_normalization_without_parser_topology_mutation() -> Result<()> {
        let input = include_bytes!("../validation/fixtures/coplanar_partitioned_extrusion.step");
        let parser = recover_complete_ir_bytes(input)?;
        assert_eq!(parser.schema, COMPLETE_IR_SCHEMA);
        assert_eq!(parser.semantically_recovered_solid_count, 1);
        assert_eq!(parser.unrecovered_solid_count, 0);
        assert_eq!(parser.solid_extrusions.len(), 1);
        assert_eq!(
            parser.coplanar_normalized_solids,
            Vec::<SourceCadFragment>::new()
        );

        let kernel = crate::cad_kernel::monstertruck::MonstertruckKernel::project_standard();
        let source_load = kernel.inspect_source_step(input)?;
        let [solid] = source_load.solids.as_slice() else {
            bail!("coplanar fixture should load as exactly one source solid");
        };
        assert!(solid.failure.is_none());
        let provenance = solid
            .topology_provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("coplanar fixture omitted topology provenance"))?;
        let patch = plan_source_coplanar_normalization_patch(input, solid.source, provenance)?
            .ok_or_else(|| {
                anyhow::anyhow!("coplanar fixture did not produce a normalization patch")
            })?;
        assert_eq!(patch.remove_faces.len(), 1);
        assert_eq!(patch.face_boundary_replacements.len(), 1);
        assert_eq!(patch.require_unused_edges.len(), 1);
        assert_eq!(patch.new_vertices, Vec::<[f64; 3]>::new());
        assert_eq!(
            patch.new_edges,
            Vec::<crate::kernel_topology::KernelNewEdge>::new()
        );
        assert_eq!(
            patch.rewrite_edges,
            Vec::<crate::kernel_topology::KernelRetainedEdge>::new()
        );
        assert_eq!(
            patch.face_splices,
            Vec::<crate::kernel_topology::KernelFaceSplice>::new()
        );

        let report = recover_complete_ir_with_kernel_bytes(input, &kernel)?;
        assert_eq!(report.schema, KERNEL_COMPLETE_IR_SCHEMA);
        assert_eq!(report.semantically_recovered_solid_count, 1);
        assert_eq!(report.unrecovered_solid_count, 0);
        assert_eq!(report.solid_extrusions, Vec::<SourceCadFragment>::new());
        assert_eq!(report.coplanar_normalized_solids.len(), 1);
        assert_eq!(
            report.kernel_recovery_failures,
            Vec::<KernelRecoveryFailure>::new()
        );
        assert!(matches!(
            report.coplanar_normalized_solids[0].fragment.source,
            CadFragmentSource::CoplanarNormalized { .. }
        ));
        Ok(())
    }

    #[cfg(feature = "cad-kernel-monstertruck")]
    #[test]
    fn kernel_complete_ir_v14_attaches_parser_neutral_topology_provenance() -> Result<()> {
        let input = include_bytes!("../validation/fixtures/production_radial_slot.step");
        let parser_only = recover_complete_ir_bytes(input)?;
        assert_eq!(parser_only.schema, COMPLETE_IR_SCHEMA);
        assert!(parser_only.kernel_source_load.is_none());
        assert!(
            serde_json::to_value(&parser_only)?
                .get("kernel_source_load")
                .is_none()
        );

        let kernel = crate::cad_kernel::monstertruck::MonstertruckKernel::project_standard();
        let kernel_report = recover_complete_ir_with_kernel_bytes(input, &kernel)?;
        assert_eq!(kernel_report.schema, KERNEL_COMPLETE_IR_SCHEMA);
        let evidence = kernel_report
            .kernel_source_load
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("kernel v14 omitted source-load evidence"))?;
        assert_eq!(evidence.sections.len(), 1);
        let solid = evidence
            .solids
            .iter()
            .find(|solid| solid.source.id == 15_385)
            .ok_or_else(|| anyhow::anyhow!("missing source-load evidence for solid #15385"))?;
        assert_eq!(solid.source.data_section, 0);
        assert!(solid.failure.is_none());
        assert_eq!(
            solid.boundaries.first().map(|boundary| boundary.role),
            Some(crate::kernel_source_load::KernelBoundaryRole::Outer)
        );
        let provenance = solid
            .topology_provenance
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("kernel v14 omitted topology provenance"))?;
        assert_ne!(provenance.vertices.len(), 0);
        assert_ne!(provenance.edges.len(), 0);
        assert_ne!(provenance.faces.len(), 0);
        assert!(
            provenance
                .vertices
                .iter()
                .all(|entry| { entry.source.data_section == 0 && entry.topology_id.raw() != 0 })
        );
        assert!(
            provenance
                .edges
                .iter()
                .all(|entry| { entry.source.data_section == 0 && entry.topology_id.raw() != 0 })
        );
        assert!(provenance.faces.iter().all(|entry| {
            entry.topology_id.raw() != 0
                && entry.source.is_none_or(|source| source.data_section == 0)
                && entry
                    .shell_member
                    .is_none_or(|source| source.data_section == 0)
        }));
        assert!(
            serde_json::to_value(&kernel_report)?
                .get("kernel_source_load")
                .is_some()
        );
        Ok(())
    }

    #[test]
    fn kernel_augmentation_skips_recovery_when_nothing_is_unrecovered() {
        let mut calls = 0usize;
        let report = apply_test_kernel_recovery(&[], |_| {
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
        let report = apply_test_kernel_recovery(&[7], |_| Ok(fragment.take()));
        assert_eq!(report.unrecovered_solid_count, 1);
        assert_eq!(report.semantically_recovered_solid_count, 0);
        assert_eq!(report.extrusion_core_unions, Vec::new());
        assert_eq!(report.kernel_recovery_failures.len(), 1);
        assert_eq!(report.kernel_recovery_failures[0].source.id, 7);
        Ok(())
    }

    #[test]
    fn kernel_augmentation_records_backend_failure_per_solid() {
        let report = apply_test_kernel_recovery(&[7], |solid_id| {
            Err(anyhow::anyhow!("kernel refused solid #{solid_id}"))
        });
        assert_eq!(report.unrecovered_solid_count, 1);
        assert_eq!(report.semantically_recovered_solid_count, 0);
        assert_eq!(report.kernel_recovery_failures.len(), 1);
        assert_eq!(report.kernel_recovery_failures[0].source.id, 7);
    }

    #[test]
    fn kernel_augmentation_moves_admitted_union_out_of_unrecovered() -> Result<()> {
        let mut fragment = Some(extrusion_core_union_fragment(7)?);
        let report = apply_test_kernel_recovery(&[7], |_| Ok(fragment.take()));
        assert_eq!(report.unrecovered_solid_count, 0);
        assert_eq!(report.constructively_recovered_solid_count, 1);
        assert_eq!(report.semantically_recovered_solid_count, 1);
        assert_eq!(report.extrusion_core_unions.len(), 1);
        assert_eq!(report.kernel_recovery_failures, Vec::new());
        Ok(())
    }

    #[test]
    fn cheapest_whole_solid_candidate_wins() -> Result<()> {
        let expensive = constructive_fragment(7, 4)?;
        let cheap = constructive_fragment(7, 1)?;
        let selected = select_cheapest_whole_solid_fragments(&[&expensive, &cheap])?;
        assert_eq!(selected.len(), 1);
        assert!(std::ptr::eq(selected[&7], std::ptr::from_ref(&cheap)));
        Ok(())
    }

    #[test]
    fn constructive_candidate_beats_cheaper_descriptive_chain() -> Result<()> {
        let constructive = constructive_fragment(7, 3)?;
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
        assert_eq!(selected[&7], &constructive);
        Ok(())
    }
}
