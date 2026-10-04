use crate::cad_recovery::{
    CadFragment, CadFragmentSource, recover_brep_fallback_fragment,
    recover_instance_pattern_fragments, recover_periodic_chain_fragments,
    recover_radial_slot_revolution_fragments, recover_solid_extrusion_fragments,
    recover_solid_revolution_fragments,
};
use crate::solid_revolutions::SolidSurfaceSignature;
use crate::{Options, OutputProfile, Stats};
use anyhow::{Context, Result, bail};
use ruststep::ast::EntityInstance;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};

pub const COMPLETE_IR_SCHEMA: &str = "step-redox-complete-ir-v3";

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
pub struct SolidBrepSummary {
    pub solid_id: u64,
    pub faces: usize,
    pub edges: usize,
    pub closure_entities: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BodyScanSummary {
    pub solids: Vec<SolidBrepSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompleteIr {
    pub schema: &'static str,
    pub idempotent_compact_rewrite: bool,
    pub compact_second_pass_stats: Stats,
    pub surface_signature_summaries: Vec<SurfaceSignatureSummary>,
    pub instance_patterns: Vec<CadFragment>,
    pub solid_extrusions: Vec<CadFragment>,
    pub solid_revolutions: Vec<CadFragment>,
    pub radial_slot_revolutions: Vec<CadFragment>,
    pub periodic_chains: Vec<CadFragment>,
    pub brep_fallbacks: Vec<CadFragment>,
    pub body_scan: BodyScanSummary,
    pub body_count: usize,
    pub constructively_recovered_solid_count: usize,
    pub semantically_recovered_solid_count: usize,
    pub brep_fallback_solid_count: usize,
}

/// Build the compact complete semantic IR for a normalized STEP exchange.
///
/// This report deliberately separates compact canonical IR from detailed detector
/// diagnostics. Every manifold solid is represented exactly once by the cheapest
/// proven whole-solid semantic fragment, or by an exact source-BREP fallback.
/// Instance-pattern fragments are additional assembly semantics and never suppress
/// the owning solid's fallback.
///
/// # Errors
/// Returns an error if STEP parsing fails, duplicate solid IDs make fallback ownership
/// ambiguous, or a detector's proven result cannot be converted into valid CAD IR.
pub fn recover_complete_ir_bytes(input: &[u8]) -> Result<CompleteIr> {
    let (input_text, _) = crate::decode_input(input)?;
    let exchange =
        ruststep::parser::parse(&input_text).context("parse STEP exchange structure for IR")?;
    if !exchange.anchor.is_empty()
        || !exchange.reference.is_empty()
        || !exchange.signature.is_empty()
    {
        bail!("ANCHOR/REFERENCE/SIGNATURE sections are not yet supported by complete IR");
    }

    let mut signatures = Vec::new();
    let mut extrusions = Vec::new();
    let mut revolutions = Vec::new();
    let mut radial_slots = Vec::new();
    let mut periodic_chains = Vec::new();
    let mut body_scan = Vec::new();

    for section in &exchange.data {
        signatures.extend(crate::solid_revolutions::detect_solid_surface_signatures(
            &section.entities,
        ));
        extrusions.extend(crate::solid_extrusions::detect_solid_extrusions(
            &section.entities,
        ));
        revolutions.extend(crate::solid_revolutions::detect_solid_revolutions(
            &section.entities,
        ));
        radial_slots.extend(crate::solid_revolutions::detect_radial_slot_revolutions(
            &section.entities,
        ));
        periodic_chains.extend(crate::periodic_chains::detect_periodic_chains(
            &section.entities,
        ));
        body_scan.extend(scan_solid_breps(&section.entities));
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

    let (patterns, _, _) = crate::detect_exchange_semantics(&exchange);
    let instance_patterns = recover_instance_pattern_fragments(&patterns)?;
    let solid_extrusions = recover_solid_extrusion_fragments(&extrusions)?;
    let solid_revolutions = recover_solid_revolution_fragments(&revolutions)?;
    let radial_slot_revolutions = recover_radial_slot_revolution_fragments(&radial_slots)?;
    let periodic_chains = recover_periodic_chain_fragments(&periodic_chains)?;

    let whole_solid_candidates = solid_extrusions
        .iter()
        .chain(&solid_revolutions)
        .chain(&radial_slot_revolutions)
        .chain(&periodic_chains)
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

    let mut brep_fallbacks = Vec::new();
    for solid in &body_scan {
        if recovered_ids.contains(&solid.solid_id) {
            continue;
        }
        brep_fallbacks.push(recover_brep_fallback_fragment(
            solid.solid_id,
            solid.faces,
            solid.edges,
            solid.closure_entities,
        )?);
    }

    let selected_constructive = selected
        .values()
        .filter(|fragment| {
            matches!(
                fragment.source,
                CadFragmentSource::SolidExtrusion { .. }
                    | CadFragmentSource::SolidRevolution { .. }
                    | CadFragmentSource::RadialSlotRevolution { .. }
            )
        })
        .count();

    let second = crate::clean_bytes(input, &Options::for_profile(OutputProfile::Compact))?;
    let idempotent_compact_rewrite = second.bytes == input;

    Ok(CompleteIr {
        schema: COMPLETE_IR_SCHEMA,
        idempotent_compact_rewrite,
        compact_second_pass_stats: second.stats,
        surface_signature_summaries,
        instance_patterns,
        solid_extrusions: selected_category(&selected, |source| {
            matches!(source, CadFragmentSource::SolidExtrusion { .. })
        }),
        solid_revolutions: selected_category(&selected, |source| {
            matches!(source, CadFragmentSource::SolidRevolution { .. })
        }),
        radial_slot_revolutions: selected_category(&selected, |source| {
            matches!(source, CadFragmentSource::RadialSlotRevolution { .. })
        }),
        periodic_chains: selected_category(&selected, |source| {
            matches!(source, CadFragmentSource::PeriodicChain { .. })
        }),
        brep_fallback_solid_count: brep_fallbacks.len(),
        body_count: body_scan.len(),
        constructively_recovered_solid_count: selected_constructive,
        semantically_recovered_solid_count: selected.len(),
        body_scan: BodyScanSummary { solids: body_scan },
        brep_fallbacks,
    })
}

fn scan_solid_breps(entities: &[EntityInstance]) -> Vec<SolidBrepSummary> {
    let index = crate::instances::build_index(entities);
    let mut out = Vec::new();
    for entity in entities {
        let Some(record) = crate::instances::simple_record(entity) else {
            continue;
        };
        if record.name != "MANIFOLD_SOLID_BREP" {
            continue;
        }
        let solid_id = crate::instances::entity_id(entity);
        let closure = crate::instances::semantic_solid_closure(solid_id, entities, &index)
            .unwrap_or_else(|| crate::instances::closure_from(solid_id, entities, &index));
        let mut faces = 0usize;
        let mut edges = 0usize;
        for id in &closure {
            let Some(&entity_index) = index.get(id) else {
                continue;
            };
            let Some(child) = crate::instances::simple_record(&entities[entity_index]) else {
                continue;
            };
            match child.name.as_str() {
                "ADVANCED_FACE" => faces += 1,
                "EDGE_CURVE" => edges += 1,
                _ => {}
            }
        }
        out.push(SolidBrepSummary {
            solid_id,
            faces,
            edges,
            closure_entities: closure.len(),
        });
    }
    out
}

fn source_solid_id(source: &CadFragmentSource) -> Option<u64> {
    match source {
        CadFragmentSource::SolidExtrusion { solid_id, .. }
        | CadFragmentSource::SolidRevolution { solid_id, .. }
        | CadFragmentSource::RadialSlotRevolution { solid_id, .. }
        | CadFragmentSource::PeriodicChain { solid_id, .. }
        | CadFragmentSource::BrepFallback { solid_id } => Some(*solid_id),
        CadFragmentSource::InstancePattern { .. } => None,
    }
}

fn select_cheapest_whole_solid_fragments<'a>(
    candidates: &[&'a CadFragment],
) -> Result<BTreeMap<u64, &'a CadFragment>> {
    let mut selected = BTreeMap::<u64, (&CadFragment, u64)>::new();
    for &fragment in candidates {
        let Some(solid_id) = source_solid_id(&fragment.source) else {
            continue;
        };
        let complexity = fragment.model.complexity_score(fragment.root)?;
        match selected.get(&solid_id) {
            Some((_, existing)) if *existing <= complexity => {}
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
    use crate::cad_ir::{BrepFallback, CadModel, CadNode, NodeId, ProofStatus, Provenance};

    fn fallback_fragment(solid_id: u64, faces: usize) -> CadFragment {
        let mut model = CadModel::new();
        let root = model.add_node(CadNode::BrepFallback(BrepFallback {
            source_entity_ids: vec![solid_id],
            estimated_faces: faces,
            estimated_edges: 0,
            estimated_control_points: 0,
        }));
        model
            .set_provenance(
                root,
                Provenance {
                    source_entity_ids: vec![solid_id],
                    proof: ProofStatus::Exact,
                    max_residual_mm: Some(0.0),
                },
            )
            .expect("test provenance");
        CadFragment {
            source: CadFragmentSource::BrepFallback { solid_id },
            model,
            root: NodeId(root.0),
        }
    }

    #[test]
    fn cheapest_whole_solid_candidate_wins() -> Result<()> {
        let expensive = fallback_fragment(7, 100);
        let cheap = fallback_fragment(7, 1);
        let selected = select_cheapest_whole_solid_fragments(&[&expensive, &cheap])?;
        assert_eq!(selected.len(), 1);
        assert!(std::ptr::eq(selected[&7], &cheap));
        Ok(())
    }
}
