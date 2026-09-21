use anyhow::{Context, Result, bail};
use ruststep::ast::{EntityInstance, Exchange, Name, Parameter, Record};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

mod bezier_recovery;
mod brep;
pub mod cad_ir;
pub mod cad_kernel;
pub mod cad_recovery;
pub mod formed_sheet;
pub mod solid_extrusions;
pub mod compatibility;
mod curve_replicas;
mod face_coalesce;
mod geometric_intern;
mod instances;
mod line_recovery;
pub mod parameters;
mod partition_recovery;
pub mod patterns;
pub mod periodic_bodies;
pub mod periodic_chains;
pub mod periodic_resize;
mod planar_features;
mod spherical_caps;
mod surface_recovery;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputProfile {
    Compat,
    Compact,
}

#[derive(Debug, Clone)]
pub struct Options {
    pub intern_values: bool,
    pub consolidate_presentation: bool,
    pub experimental_recover_straight_bspline_lines: bool,
    pub experimental_recover_exact_bezier_curves: bool,
    pub experimental_recover_v_extrusions: bool,
    pub experimental_intern_geometric_supports: bool,
    pub experimental_recover_partitioned_bodies: bool,
    pub experimental_coalesce_same_support_faces: bool,
    pub experimental_instance_translated_bspline_curves: bool,
    pub experimental_instance_z90: bool,
    pub experimental_instance_z90_assembly: bool,
    pub experimental_instance_planar_positive_features: bool,
    pub experimental_instance_spherical_caps: bool,
    pub minify_placeholder_names: bool,
    pub dense_ids: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            intern_values: true,
            consolidate_presentation: true,
            experimental_recover_straight_bspline_lines: false,
            experimental_recover_exact_bezier_curves: false,
            experimental_recover_v_extrusions: false,
            experimental_intern_geometric_supports: false,
            experimental_recover_partitioned_bodies: false,
            experimental_coalesce_same_support_faces: false,
            experimental_instance_translated_bspline_curves: false,
            experimental_instance_z90: false,
            experimental_instance_z90_assembly: false,
            experimental_instance_planar_positive_features: false,
            experimental_instance_spherical_caps: false,
            minify_placeholder_names: false,
            dense_ids: true,
        }
    }
}

impl Options {
    pub fn for_profile(profile: OutputProfile) -> Self {
        let mut options = Self {
            experimental_recover_straight_bspline_lines: true,
            experimental_recover_exact_bezier_curves: true,
            experimental_recover_v_extrusions: true,
            experimental_intern_geometric_supports: true,
            experimental_recover_partitioned_bodies: true,
            experimental_coalesce_same_support_faces: true,
            minify_placeholder_names: true,
            ..Self::default()
        };

        if profile == OutputProfile::Compact {
            options.experimental_instance_translated_bspline_curves = true;
            options.experimental_instance_z90 = true;
            options.experimental_instance_planar_positive_features = true;
            options.experimental_instance_spherical_caps = true;
        }
        options
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Stats {
    pub input_encoding: String,
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub input_entities: usize,
    pub output_entities: usize,
    pub interned_entities: usize,
    pub consolidated_entities: usize,
    pub straight_bspline_lines_recovered: usize,
    pub straight_bspline_direction_groups: usize,
    pub straight_bspline_points_removed: usize,
    pub exact_bezier_curves_recovered: usize,
    pub v_extrusion_surfaces_recovered: usize,
    pub v_extrusion_rational_surfaces_recovered: usize,
    pub v_extrusion_profile_curves_created: usize,
    pub v_extrusion_points_removed: usize,
    pub geometric_supports_merged: usize,
    pub geometric_support_entities_removed: usize,
    pub geometric_planes_merged: usize,
    pub geometric_lines_merged: usize,
    pub geometric_cylinders_merged: usize,
    pub partition_components_recovered: usize,
    pub partition_solids_merged: usize,
    pub partition_interfaces_removed: usize,
    pub partition_styles_retargeted: usize,
    pub partition_entities_removed: usize,
    pub face_coalesce_groups: usize,
    pub face_coalesce_faces_merged: usize,
    pub face_coalesce_faces_removed: usize,
    pub face_coalesce_internal_edges_removed: usize,
    pub face_coalesce_styles_removed: usize,
    pub face_coalesce_entities_removed: usize,
    pub curve_replica_families: usize,
    pub curve_replicas: usize,
    pub curve_replica_direct_aliases: usize,
    pub curve_replica_transforms: usize,
    pub curve_replica_entities_removed: usize,
    pub curve_replica_max_residual_mm: f64,
    pub instance_groups: usize,
    pub instanced_solids: usize,
    pub instance_entities_removed: usize,
    pub instance_styles_replaced: usize,
    pub planar_feature_arrays: usize,
    pub planar_feature_families: usize,
    pub planar_feature_instances: usize,
    pub planar_feature_entities_removed: usize,
    pub planar_feature_styles_replaced: usize,
    pub spherical_cap_arrays: usize,
    pub spherical_cap_instances: usize,
    pub spherical_cap_entities_removed: usize,
    pub spherical_cap_styles_replaced: usize,
    pub placeholder_names_minified: usize,
    pub instance_patterns_detected: usize,
    pub pattern_instances_detected: usize,
    pub periodic_body_patterns_detected: usize,
    pub periodic_body_repeat_faces: usize,
    pub count_parameters_detected: usize,
    pub count_parameters_with_body_grammar: usize,
    pub byte_ratio: f64,
    pub interned_by_type: BTreeMap<String, usize>,
    pub consolidated_by_type: BTreeMap<String, usize>,
}

pub struct CleanOutput {
    pub bytes: Vec<u8>,
    pub stats: Stats,
    /// Regular instance patterns detected in the final normalized graph.
    /// Entity IDs refer to the emitted STEP after dense renumbering.
    pub patterns: Vec<patterns::InstancePattern>,
    /// Periodic planar body grammars coupled to recovered instance patterns.
    pub periodic_bodies: Vec<periodic_bodies::PeriodicBodyPattern>,
    /// Higher-level count controls recovered by coupling instance/body patterns.
    pub count_parameters: Vec<parameters::RecoveredCountParameter>,
    /// Structural compatibility audit of the emitted STEP.
    pub compatibility: compatibility::CompatibilityAudit,
}

pub struct PatternEditOutput {
    pub bytes: Vec<u8>,
    pub resize: patterns::PatternResizeStats,
    pub patterns: Vec<patterns::InstancePattern>,
    /// Periodic planar body grammars coupled to recovered instance patterns.
    pub periodic_bodies: Vec<periodic_bodies::PeriodicBodyPattern>,
    /// Higher-level count controls recovered by coupling instance/body patterns.
    pub count_parameters: Vec<parameters::RecoveredCountParameter>,
    /// Structural compatibility audit of the emitted STEP.
    pub compatibility: compatibility::CompatibilityAudit,
}

pub struct PeriodicBodyEditOutput {
    pub bytes: Vec<u8>,
    pub resize: periodic_resize::PeriodicBodyResizeStats,
}

pub struct PeriodicChainEditOutput {
    pub bytes: Vec<u8>,
    pub resize: periodic_resize::PeriodicChainResizeStats,
    pub periodic_chains: Vec<periodic_chains::PeriodicChainPattern>,
    pub compatibility: compatibility::CompatibilityAudit,
}

/// Detect read-only formed-sheet geometric evidence in a STEP exchange.
///
/// This is evidence only: it reports repeated constant-thickness signatures such as
/// coaxial cylinder radius pairs and parallel-plane offsets. It does not claim that
/// a complete editable sheet-metal construction has been recovered.
pub fn detect_formed_sheet_evidence_bytes(
    input: &[u8],
) -> Result<Vec<formed_sheet::FormedSheetEvidence>> {
    let (input_text, _) = decode_input(input)?;
    let (parser_text, had_empty_aggregate_shim) = prepare_parser_input(&input_text)?;
    let mut exchange =
        ruststep::parser::parse(&parser_text).context("parse STEP exchange structure")?;
    if had_empty_aggregate_shim {
        restore_empty_aggregates(&mut exchange)?;
    }
    if !exchange.anchor.is_empty()
        || !exchange.reference.is_empty()
        || !exchange.signature.is_empty()
    {
        bail!("ANCHOR/REFERENCE/SIGNATURE sections are not yet supported by step-redox writer");
    }

    Ok(exchange
        .data
        .iter()
        .flat_map(|section| formed_sheet::detect_formed_sheet_evidence(&section.entities))
        .collect())
}

/// Detect whole-solid line-profile extrusion grammars in a STEP exchange.
///
/// This first pass is intentionally strict: accepted solids must be closed, all-planar,
/// single-loop prisms whose cap edges and lateral connectors prove one translation.
pub fn detect_solid_extrusions_bytes(
    input: &[u8],
) -> Result<Vec<solid_extrusions::RecoveredSolidExtrusion>> {
    let (input_text, _) = decode_input(input)?;
    let (parser_text, had_empty_aggregate_shim) = prepare_parser_input(&input_text)?;
    let mut exchange =
        ruststep::parser::parse(&parser_text).context("parse STEP exchange structure")?;
    if had_empty_aggregate_shim {
        restore_empty_aggregates(&mut exchange)?;
    }
    if !exchange.anchor.is_empty()
        || !exchange.reference.is_empty()
        || !exchange.signature.is_empty()
    {
        bail!("ANCHOR/REFERENCE/SIGNATURE sections are not yet supported by step-redox writer");
    }

    Ok(exchange
        .data
        .iter()
        .flat_map(|section| solid_extrusions::detect_solid_extrusions(&section.entities))
        .collect())
}

/// Detect read-only periodic chain grammars without enabling mutation.
///
/// This analyzer is intentionally separate from the editable PeriodicBodyPattern
/// path. It can recover fused-solid site/gap/stretch/end structure even when no
/// MAPPED_ITEM instance row exists, but callers must not treat that as edit
/// permission.
pub fn detect_periodic_chains_bytes(
    input: &[u8],
) -> Result<Vec<periodic_chains::PeriodicChainPattern>> {
    let (input_text, _) = decode_input(input)?;
    let (parser_text, had_empty_aggregate_shim) = prepare_parser_input(&input_text)?;
    let mut exchange =
        ruststep::parser::parse(&parser_text).context("parse STEP exchange structure")?;
    if had_empty_aggregate_shim {
        restore_empty_aggregates(&mut exchange)?;
    }
    if !exchange.anchor.is_empty()
        || !exchange.reference.is_empty()
        || !exchange.signature.is_empty()
    {
        bail!("ANCHOR/REFERENCE/SIGNATURE sections are not yet supported by step-redox writer");
    }

    Ok(exchange
        .data
        .iter()
        .flat_map(|section| periodic_chains::detect_periodic_chains(&section.entities))
        .collect())
}

/// Resize one proven periodic fused-solid chain while keeping its start fixed.
pub fn resize_periodic_chain_bytes(
    input: &[u8],
    chain_index: usize,
    new_sites: usize,
) -> Result<PeriodicChainEditOutput> {
    resize_periodic_chain_bytes_with_anchor(
        input,
        chain_index,
        new_sites,
        CountAnchor::Start,
    )
}

/// Resize one proven periodic fused-solid chain with explicit placement anchoring.
///
/// Start and End keep the corresponding physical end fixed. Center composes
/// equal edits at both ends and therefore currently requires an even site delta.
pub fn resize_periodic_chain_bytes_with_anchor(
    input: &[u8],
    chain_index: usize,
    new_sites: usize,
    anchor: CountAnchor,
) -> Result<PeriodicChainEditOutput> {
    if anchor != CountAnchor::Center {
        return resize_periodic_chain_bytes_one_side(input, chain_index, new_sites, anchor);
    }

    let chains = detect_periodic_chains_bytes(input)?;
    let original = chains
        .get(chain_index)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("periodic chain index {chain_index} not found"))?;
    if !original.read_only_proven || !original.complete_partition {
        bail!("periodic chain {chain_index} is not sufficiently proven for editing");
    }
    if new_sites == original.sites {
        bail!("periodic-chain resize requested the existing site count {new_sites}");
    }
    if new_sites < 6 {
        bail!("center-anchored periodic-chain editing currently requires at least 6 sites");
    }

    let delta = new_sites.abs_diff(original.sites);
    if delta % 2 != 0 {
        bail!(
            "center-anchored periodic-chain resize currently requires an even site delta ({} -> {})",
            original.sites,
            new_sites
        );
    }
    let half = delta / 2;
    let mid_sites = if new_sites > original.sites {
        original.sites + half
    } else {
        original.sites - half
    };

    let first = resize_periodic_chain_bytes_one_side(
        input,
        chain_index,
        mid_sites,
        CountAnchor::Start,
    )?;
    let followup_index =
        find_matching_periodic_chain(&first.periodic_chains, mid_sites, &original)?;
    let mut second = resize_periodic_chain_bytes_one_side(
        &first.bytes,
        followup_index,
        new_sites,
        CountAnchor::End,
    )?;

    let mut combined = second.resize.clone();
    combined.old_sites = original.sites;
    combined.new_sites = new_sites;
    combined.inserted_units = first.resize.inserted_units + second.resize.inserted_units;
    combined.removed_units = first.resize.removed_units + second.resize.removed_units;
    combined.seam_edge_pairs = first.resize.seam_edge_pairs + second.resize.seam_edge_pairs;
    combined.welded_vertices = first.resize.welded_vertices + second.resize.welded_vertices;
    combined.welded_edges = first.resize.welded_edges + second.resize.welded_edges;
    combined.rebuilt_stretch_faces =
        first.resize.rebuilt_stretch_faces + second.resize.rebuilt_stretch_faces;
    combined.new_stretch_edges =
        first.resize.new_stretch_edges + second.resize.new_stretch_edges;
    combined.added_entities = first.resize.added_entities + second.resize.added_entities;
    combined.pruned_entities = first.resize.pruned_entities + second.resize.pruned_entities;
    combined.entity_delta = first.resize.entity_delta + second.resize.entity_delta;
    second.resize = combined;
    Ok(second)
}

fn resize_periodic_chain_bytes_one_side(
    input: &[u8],
    chain_index: usize,
    new_sites: usize,
    anchor: CountAnchor,
) -> Result<PeriodicChainEditOutput> {
    debug_assert!(anchor != CountAnchor::Center);
    let (input_text, _) = decode_input(input)?;
    let (parser_text, had_empty_aggregate_shim) = prepare_parser_input(&input_text)?;
    let mut exchange =
        ruststep::parser::parse(&parser_text).context("parse STEP exchange structure")?;
    if had_empty_aggregate_shim {
        restore_empty_aggregates(&mut exchange)?;
    }
    if !exchange.anchor.is_empty()
        || !exchange.reference.is_empty()
        || !exchange.signature.is_empty()
    {
        bail!("ANCHOR/REFERENCE/SIGNATURE sections are not yet supported by step-redox writer");
    }
    if exchange.data.len() != 1 {
        bail!("periodic-chain editing currently requires exactly one DATA section");
    }

    let section = &mut exchange.data[0];
    let chains = periodic_chains::detect_periodic_chains(&section.entities);
    let chain = chains
        .get(chain_index)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("periodic chain index {chain_index} not found"))?;
    if !chain.read_only_proven {
        bail!("periodic chain {chain_index} is not sufficiently proven for editing");
    }
    if new_sites == chain.sites {
        bail!("periodic-chain resize requested the existing site count {new_sites}");
    }

    let directed = if anchor == CountAnchor::End {
        reverse_periodic_chain(&chain)
    } else {
        chain.clone()
    };

    let source_entity_count = section.entities.len();
    let mut resize = if new_sites > directed.sites {
        periodic_resize::expand_periodic_chain_positive(
            &mut section.entities,
            &directed,
            new_sites,
        )?
    } else {
        periodic_resize::shrink_periodic_chain_positive(
            &mut section.entities,
            &directed,
            new_sites,
        )?
    };

    // Compact normalization has already interned the source support geometry,
    // but chain mutation creates fresh translated PLANE/LINE/CYLINDER supports.
    // Re-run the same guarded locus interning pass on the generated graph so
    // newly created supports do not survive merely because mutation happened
    // after the initial compact cleanup.
    let _ = geometric_intern::intern_geometric_supports(&mut section.entities);
    let _ = intern_section(&mut section.entities);
    let detached_vertices =
        periodic_resize::prune_detached_vertex_points(&mut section.entities);
    resize.pruned_entities += detached_vertices;
    resize.added_entities = section.entities.len().saturating_sub(source_entity_count);
    resize.entity_delta = section.entities.len() as isize - source_entity_count as isize;
    for section in &mut exchange.data {
        dense_renumber(&mut section.entities);
    }

    let periodic_chains = exchange
        .data
        .iter()
        .flat_map(|section| periodic_chains::detect_periodic_chains(&section.entities))
        .collect::<Vec<_>>();
    let verified = periodic_chains.iter().any(|candidate| {
        candidate.read_only_proven
            && candidate.complete_partition
            && candidate.sites == new_sites
            && (candidate.pitch_mm - chain.pitch_mm).abs() <= 1.0e-7
            && candidate.interior_site_face_count == chain.interior_site_face_count
            && candidate.interior_gap_face_count == chain.interior_gap_face_count
            && candidate.stretch_face_ids.len() == chain.stretch_face_ids.len()
            && candidate.fixed_negative_face_ids.len() == chain.fixed_negative_face_ids.len()
            && candidate.fixed_positive_face_ids.len() == chain.fixed_positive_face_ids.len()
            && candidate.faces_without_geometry == 0
            && candidate.nonmanifold_edges == 0
            && candidate.cross_site_edges == 0
    });
    if !verified {
        bail!(
            "post-edit periodic-chain verification failed: requested {new_sites} sites were not rediscovered with the same proven grammar"
        );
    }

    let compatibility = audit_exchange_compatibility(&exchange);
    let output = write_exchange(&exchange)?;
    Ok(PeriodicChainEditOutput {
        bytes: output.into_bytes(),
        resize,
        periodic_chains,
        compatibility,
    })
}

/// Backward-compatible growth-only wrapper around `resize_periodic_chain_bytes`.
pub fn expand_periodic_chain_bytes(
    input: &[u8],
    chain_index: usize,
    new_sites: usize,
) -> Result<PeriodicChainEditOutput> {
    let chains = detect_periodic_chains_bytes(input)?;
    let old_sites = chains
        .get(chain_index)
        .map(|chain| chain.sites)
        .ok_or_else(|| anyhow::anyhow!("periodic chain index {chain_index} not found"))?;
    if new_sites <= old_sites {
        bail!(
            "periodic-chain expansion requires growth ({old_sites} -> {new_sites})"
        );
    }
    resize_periodic_chain_bytes(input, chain_index, new_sites)
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CountAnchor {
    Start,
    Center,
    End,
}

#[derive(Debug, Clone, Serialize)]
pub struct CountResizeStats {
    pub old_sites: usize,
    pub new_sites: usize,
    pub instances_per_site: usize,
    pub anchor: CountAnchor,
    pub body_resizes: Vec<periodic_resize::PeriodicBodyResizeStats>,
    pub pattern_resizes: Vec<patterns::PatternResizeStats>,
}

pub struct CountEditOutput {
    pub bytes: Vec<u8>,
    pub resize: CountResizeStats,
    pub patterns: Vec<patterns::InstancePattern>,
    pub periodic_bodies: Vec<periodic_bodies::PeriodicBodyPattern>,
    pub count_parameters: Vec<parameters::RecoveredCountParameter>,
    pub compatibility: compatibility::CompatibilityAudit,
}

/// Resize one recovered count parameter atomically, keeping its negative end fixed.
pub fn resize_count_parameter_bytes(
    input: &[u8],
    parameter_index: usize,
    new_sites: usize,
) -> Result<CountEditOutput> {
    resize_count_parameter_bytes_with_anchor(input, parameter_index, new_sites, CountAnchor::Start)
}

/// Resize one recovered count parameter with explicit placement anchoring.
///
/// Start and End keep the corresponding physical end fixed. Center keeps the
/// geometric center fixed by performing equal edits at both ends; for now this
/// requires an even site-count delta.
pub fn resize_count_parameter_bytes_with_anchor(
    input: &[u8],
    parameter_index: usize,
    new_sites: usize,
    anchor: CountAnchor,
) -> Result<CountEditOutput> {
    if anchor != CountAnchor::Center {
        return resize_count_parameter_bytes_one_side(input, parameter_index, new_sites, anchor);
    }

    let parameters = detect_count_parameters_bytes(input)?;
    let original = parameters
        .get(parameter_index)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("count parameter index {parameter_index} not found"))?;
    if !original.body_grammar_proven || original.periodic_bodies.is_empty() {
        bail!("count parameter {parameter_index} does not have a proven periodic body grammar");
    }
    if new_sites == original.sites {
        bail!("count-parameter edit is a no-op at {new_sites} sites");
    }
    if new_sites < 4 {
        bail!("count-parameter editing currently requires at least 4 sites");
    }

    let delta = new_sites.abs_diff(original.sites);
    if delta % 2 != 0 {
        bail!(
            "center-anchored count resize currently requires an even site delta ({} -> {})",
            original.sites,
            new_sites
        );
    }
    let half = delta / 2;
    let mid_sites = if new_sites > original.sites {
        original.sites + half
    } else {
        original.sites - half
    };

    let first = resize_count_parameter_bytes_one_side(
        input,
        parameter_index,
        mid_sites,
        CountAnchor::Start,
    )?;
    let followup_index =
        find_matching_count_parameter(&first.count_parameters, mid_sites, &original)?;
    let mut second = resize_count_parameter_bytes_one_side(
        &first.bytes,
        followup_index,
        new_sites,
        CountAnchor::End,
    )?;

    let mut body_resizes = first.resize.body_resizes;
    body_resizes.extend(second.resize.body_resizes);
    let mut pattern_resizes = first.resize.pattern_resizes;
    pattern_resizes.extend(second.resize.pattern_resizes);
    second.resize = CountResizeStats {
        old_sites: original.sites,
        new_sites,
        instances_per_site: original.instances_per_site,
        anchor: CountAnchor::Center,
        body_resizes,
        pattern_resizes,
    };
    Ok(second)
}

/// One-sided primitive used by all placement policies.
fn resize_count_parameter_bytes_one_side(
    input: &[u8],
    parameter_index: usize,
    new_sites: usize,
    anchor: CountAnchor,
) -> Result<CountEditOutput> {
    debug_assert!(anchor != CountAnchor::Center);
    let (input_text, _) = decode_input(input)?;
    let (parser_text, had_empty_aggregate_shim) = prepare_parser_input(&input_text)?;
    let mut exchange =
        ruststep::parser::parse(&parser_text).context("parse STEP exchange structure")?;
    if had_empty_aggregate_shim {
        restore_empty_aggregates(&mut exchange)?;
    }
    if !exchange.anchor.is_empty()
        || !exchange.reference.is_empty()
        || !exchange.signature.is_empty()
    {
        bail!("ANCHOR/REFERENCE/SIGNATURE sections are not yet supported by step-redox writer");
    }
    if exchange.data.len() != 1 {
        bail!("count-parameter editing currently requires exactly one DATA section");
    }

    let section = &mut exchange.data[0];
    let detected_patterns = patterns::detect_instance_patterns(&section.entities, 1.0e-7, 4);
    let detected_bodies =
        periodic_bodies::detect_periodic_bodies(&section.entities, &detected_patterns);
    let detected_parameters =
        parameters::detect_count_parameters(&detected_patterns, &detected_bodies);
    let parameter = detected_parameters
        .get(parameter_index)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("count parameter index {parameter_index} not found"))?;

    if !parameter.body_grammar_proven || parameter.periodic_bodies.is_empty() {
        bail!("count parameter {parameter_index} does not have a proven periodic body grammar");
    }
    if new_sites == parameter.sites {
        bail!("count-parameter edit is a no-op at {new_sites} sites");
    }
    if new_sites < 4 {
        bail!("count-parameter editing currently requires at least 4 sites");
    }

    let body_specs = parameter
        .periodic_bodies
        .iter()
        .map(|&index| {
            detected_bodies
                .get(index)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("periodic body index {index} disappeared"))
        })
        .collect::<Result<Vec<_>>>()?;
    let pattern_specs = parameter
        .instance_patterns
        .iter()
        .map(|&index| {
            detected_patterns
                .get(index)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("instance pattern index {index} disappeared"))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut body_resizes = Vec::with_capacity(body_specs.len());
    for body in &body_specs {
        let directed = if anchor == CountAnchor::End {
            reverse_periodic_body(body)
        } else {
            body.clone()
        };
        let stats = if new_sites > directed.sites {
            periodic_resize::expand_periodic_body_positive(
                &mut section.entities,
                &directed,
                new_sites,
            )?
        } else {
            periodic_resize::shrink_periodic_body_positive(
                &mut section.entities,
                &directed,
                new_sites,
            )?
        };
        body_resizes.push(stats);
    }

    let mut pattern_resizes = Vec::with_capacity(pattern_specs.len());
    for pattern in &pattern_specs {
        let basis = pattern
            .basis
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("coupled pattern has no basis"))?;
        let axis_dot = basis[0] * parameter.axis[0]
            + basis[1] * parameter.axis[1]
            + basis[2] * parameter.axis[2];
        let pattern_anchor = match (anchor, axis_dot >= 0.0) {
            (CountAnchor::Start, true) | (CountAnchor::End, false) => {
                patterns::PatternAnchor::Start
            }
            (CountAnchor::Start, false) | (CountAnchor::End, true) => patterns::PatternAnchor::End,
            (CountAnchor::Center, _) => unreachable!(),
        };
        pattern_resizes.push(patterns::resize_filled_linear_pattern(
            &mut section.entities,
            pattern,
            new_sites,
            pattern_anchor,
        )?);
    }

    let _ = intern_section(&mut section.entities);
    for section in &mut exchange.data {
        dense_renumber(&mut section.entities);
    }

    let (patterns, periodic_bodies, count_parameters) = detect_exchange_semantics(&exchange);
    let verified = count_parameters.iter().any(|candidate| {
        candidate.body_grammar_proven
            && candidate.sites == new_sites
            && candidate.instances_per_site == parameter.instances_per_site
            && candidate.instance_patterns.len() == parameter.instance_patterns.len()
            && candidate.periodic_bodies.len() == parameter.periodic_bodies.len()
    });
    if !verified {
        bail!(
            "post-edit semantic verification failed: requested {new_sites} sites were not rediscovered with the same coupled body/instance grammar"
        );
    }

    let compatibility = audit_exchange_compatibility(&exchange);
    let output = write_exchange(&exchange)?;
    Ok(CountEditOutput {
        bytes: output.into_bytes(),
        resize: CountResizeStats {
            old_sites: parameter.sites,
            new_sites,
            instances_per_site: parameter.instances_per_site,
            anchor,
            body_resizes,
            pattern_resizes,
        },
        patterns,
        periodic_bodies,
        count_parameters,
        compatibility,
    })
}

fn find_matching_periodic_chain(
    chains: &[periodic_chains::PeriodicChainPattern],
    sites: usize,
    original: &periodic_chains::PeriodicChainPattern,
) -> Result<usize> {
    let matches = chains
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            (candidate.read_only_proven
                && candidate.complete_partition
                && candidate.sites == sites
                && (candidate.pitch_mm - original.pitch_mm).abs() <= 1.0e-7
                && candidate.interior_site_face_count == original.interior_site_face_count
                && candidate.interior_gap_face_count == original.interior_gap_face_count
                && candidate.stretch_face_ids.len() == original.stretch_face_ids.len()
                && candidate.fixed_negative_face_ids.len()
                    == original.fixed_negative_face_ids.len()
                && candidate.fixed_middle_face_ids.len() == original.fixed_middle_face_ids.len()
                && candidate.fixed_positive_face_ids.len()
                    == original.fixed_positive_face_ids.len())
            .then_some(index)
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [index] => Ok(*index),
        [] => bail!("could not rediscover the intermediate {sites}-site periodic chain"),
        _ => bail!("multiple matching intermediate {sites}-site periodic chains: {matches:?}"),
    }
}

fn reverse_periodic_chain(
    chain: &periodic_chains::PeriodicChainPattern,
) -> periodic_chains::PeriodicChainPattern {
    let mut reversed = chain.clone();
    reversed.axis = [-chain.axis[0], -chain.axis[1], -chain.axis[2]];
    reversed.site_centers_mm = chain
        .site_centers_mm
        .iter()
        .rev()
        .map(|center| -*center)
        .collect();
    reversed.site_face_counts.reverse();
    reversed.gap_face_counts.reverse();
    reversed.site_face_ids.reverse();
    reversed.gap_face_ids.reverse();
    reversed.site_adjacency_signatures.reverse();
    reversed.gap_adjacency_signatures.reverse();
    std::mem::swap(
        &mut reversed.fixed_negative_face_ids,
        &mut reversed.fixed_positive_face_ids,
    );
    reversed
}

fn reverse_periodic_body(
    body: &periodic_bodies::PeriodicBodyPattern,
) -> periodic_bodies::PeriodicBodyPattern {
    let mut reversed = body.clone();
    reversed.axis = [-body.axis[0], -body.axis[1], -body.axis[2]];
    for family in &mut reversed.repeat_face_families {
        family.face_ids.reverse();
    }
    reversed
}

fn detect_count_parameters_bytes(input: &[u8]) -> Result<Vec<parameters::RecoveredCountParameter>> {
    let (input_text, _) = decode_input(input)?;
    let (parser_text, had_empty_aggregate_shim) = prepare_parser_input(&input_text)?;
    let mut exchange =
        ruststep::parser::parse(&parser_text).context("parse STEP exchange structure")?;
    if had_empty_aggregate_shim {
        restore_empty_aggregates(&mut exchange)?;
    }
    if exchange.data.len() != 1 {
        bail!("count-parameter editing currently requires exactly one DATA section");
    }
    let section = &exchange.data[0];
    let patterns = patterns::detect_instance_patterns(&section.entities, 1.0e-7, 4);
    let bodies = periodic_bodies::detect_periodic_bodies(&section.entities, &patterns);
    Ok(parameters::detect_count_parameters(&patterns, &bodies))
}

fn find_matching_count_parameter(
    parameters: &[parameters::RecoveredCountParameter],
    sites: usize,
    original: &parameters::RecoveredCountParameter,
) -> Result<usize> {
    let matches = parameters
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            candidate.body_grammar_proven
                && candidate.sites == sites
                && candidate.instances_per_site == original.instances_per_site
                && candidate.instance_patterns.len() == original.instance_patterns.len()
                && candidate.periodic_bodies.len() == original.periodic_bodies.len()
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [index] => Ok(*index),
        [] => bail!("could not rediscover the intermediate {sites}-site count parameter"),
        _ => bail!("multiple matching intermediate {sites}-site count parameters: {matches:?}"),
    }
}

/// Backward-compatible growth-only wrapper.
pub fn expand_count_parameter_bytes(
    input: &[u8],
    parameter_index: usize,
    new_sites: usize,
) -> Result<CountEditOutput> {
    let edited = resize_count_parameter_bytes(input, parameter_index, new_sites)?;
    if edited.resize.new_sites <= edited.resize.old_sites {
        bail!(
            "expand_count_parameter_bytes requires growth ({} -> {})",
            edited.resize.old_sites,
            edited.resize.new_sites
        );
    }
    Ok(edited)
}

/// Expand one detected periodic body at its positive-axis end.
pub fn expand_periodic_body_bytes(
    input: &[u8],
    body_index: usize,
    new_sites: usize,
) -> Result<PeriodicBodyEditOutput> {
    let (input_text, _) = decode_input(input)?;
    let (parser_text, had_empty_aggregate_shim) = prepare_parser_input(&input_text)?;
    let mut exchange =
        ruststep::parser::parse(&parser_text).context("parse STEP exchange structure")?;
    if had_empty_aggregate_shim {
        restore_empty_aggregates(&mut exchange)?;
    }
    if !exchange.anchor.is_empty()
        || !exchange.reference.is_empty()
        || !exchange.signature.is_empty()
    {
        bail!("ANCHOR/REFERENCE/SIGNATURE sections are not yet supported by step-redox writer");
    }

    let mut remaining = body_index;
    let mut resize = None;
    for section in &mut exchange.data {
        let patterns = patterns::detect_instance_patterns(&section.entities, 1.0e-7, 4);
        let bodies = periodic_bodies::detect_periodic_bodies(&section.entities, &patterns);
        if remaining < bodies.len() {
            resize = Some(periodic_resize::expand_periodic_body_positive(
                &mut section.entities,
                &bodies[remaining],
                new_sites,
            )?);
            let _ = intern_section(&mut section.entities);
            break;
        }
        remaining -= bodies.len();
    }
    let resize =
        resize.ok_or_else(|| anyhow::anyhow!("periodic body index {body_index} not found"))?;

    for section in &mut exchange.data {
        dense_renumber(&mut section.entities);
    }
    let output = write_exchange(&exchange)?;
    Ok(PeriodicBodyEditOutput {
        bytes: output.into_bytes(),
        resize,
    })
}

/// Resize one fully occupied 1-D regular MAPPED_ITEM pattern in an already
/// normalized STEP file. This edits only the instance pattern; higher-level
/// package/body resizing is intentionally a separate operation.
pub fn resize_linear_pattern_bytes(
    input: &[u8],
    pattern_index: usize,
    new_count: usize,
    anchor: patterns::PatternAnchor,
) -> Result<PatternEditOutput> {
    let (input_text, _) = decode_input(input)?;
    let (parser_text, had_empty_aggregate_shim) = prepare_parser_input(&input_text)?;
    let mut exchange =
        ruststep::parser::parse(&parser_text).context("parse STEP exchange structure")?;
    if had_empty_aggregate_shim {
        restore_empty_aggregates(&mut exchange)?;
    }
    if !exchange.anchor.is_empty()
        || !exchange.reference.is_empty()
        || !exchange.signature.is_empty()
    {
        bail!("ANCHOR/REFERENCE/SIGNATURE sections are not yet supported by step-redox writer");
    }

    let mut remaining = pattern_index;
    let mut resize = None;
    for section in &mut exchange.data {
        let detected = patterns::detect_instance_patterns(&section.entities, 1.0e-7, 4);
        if remaining < detected.len() {
            resize = Some(patterns::resize_filled_linear_pattern(
                &mut section.entities,
                &detected[remaining],
                new_count,
                anchor,
            )?);
            // Normalize newly-created placements/support values immediately.
            let _ = intern_section(&mut section.entities);
            break;
        }
        remaining -= detected.len();
    }
    let resize =
        resize.ok_or_else(|| anyhow::anyhow!("pattern index {pattern_index} not found"))?;

    for section in &mut exchange.data {
        dense_renumber(&mut section.entities);
    }
    let (patterns, periodic_bodies, count_parameters) = detect_exchange_semantics(&exchange);
    let compatibility = audit_exchange_compatibility(&exchange);
    let output = write_exchange(&exchange)?;

    Ok(PatternEditOutput {
        bytes: output.into_bytes(),
        resize,
        patterns,
        periodic_bodies,
        count_parameters,
        compatibility,
    })
}

fn detect_exchange_semantics(
    exchange: &Exchange,
) -> (
    Vec<patterns::InstancePattern>,
    Vec<periodic_bodies::PeriodicBodyPattern>,
    Vec<parameters::RecoveredCountParameter>,
) {
    let mut patterns_out = Vec::new();
    let mut bodies_out = Vec::new();

    for section in &exchange.data {
        let local_patterns = patterns::detect_instance_patterns(&section.entities, 1.0e-7, 4);
        let offset = patterns_out.len();
        let mut local_bodies =
            periodic_bodies::detect_periodic_bodies(&section.entities, &local_patterns);
        for body in &mut local_bodies {
            for index in &mut body.coupled_instance_patterns {
                *index += offset;
            }
        }
        patterns_out.extend(local_patterns);
        bodies_out.extend(local_bodies);
    }

    let count_parameters = parameters::detect_count_parameters(&patterns_out, &bodies_out);
    (patterns_out, bodies_out, count_parameters)
}

fn audit_exchange_compatibility(exchange: &Exchange) -> compatibility::CompatibilityAudit {
    let mut combined = compatibility::CompatibilityAudit::default();
    for section in &exchange.data {
        let audit = compatibility::audit_entities(&section.entities);
        for (name, count) in audit.structural_risk_entities {
            *combined.structural_risk_entities.entry(name).or_insert(0) += count;
        }
    }
    combined.structural_risk_total = combined.structural_risk_entities.values().sum();
    combined.has_mapped_items = combined
        .structural_risk_entities
        .get("MAPPED_ITEM")
        .copied()
        .unwrap_or(0)
        > 0
        || combined
            .structural_risk_entities
            .get("REPRESENTATION_MAP")
            .copied()
            .unwrap_or(0)
            > 0;
    combined.has_curve_or_surface_replicas = combined
        .structural_risk_entities
        .get("CURVE_REPLICA")
        .copied()
        .unwrap_or(0)
        > 0
        || combined
            .structural_risk_entities
            .get("SURFACE_REPLICA")
            .copied()
            .unwrap_or(0)
            > 0;
    combined.has_assembly_relationships = [
        "NEXT_ASSEMBLY_USAGE_OCCURRENCE",
        "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION",
        "ITEM_DEFINED_TRANSFORMATION",
        "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION",
        "SHAPE_REPRESENTATION_RELATIONSHIP",
    ]
    .iter()
    .any(|name| {
        combined
            .structural_risk_entities
            .get(*name)
            .copied()
            .unwrap_or(0)
            > 0
    });
    combined.conservative_structure = combined.structural_risk_total == 0;
    combined
}

pub fn clean_bytes(input: &[u8], options: &Options) -> Result<CleanOutput> {
    let (input_text, input_encoding) = decode_input(input)?;
    let (parser_text, had_empty_aggregate_shim) = prepare_parser_input(&input_text)?;
    let mut exchange =
        ruststep::parser::parse(&parser_text).context("parse STEP exchange structure")?;
    if had_empty_aggregate_shim {
        restore_empty_aggregates(&mut exchange)?;
    }

    if !exchange.anchor.is_empty()
        || !exchange.reference.is_empty()
        || !exchange.signature.is_empty()
    {
        bail!("ANCHOR/REFERENCE/SIGNATURE sections are not yet supported by step-redox writer");
    }

    let input_entities: usize = exchange.data.iter().map(|d| d.entities.len()).sum();

    let mut placeholder_names_minified = 0usize;
    if options.minify_placeholder_names {
        for section in &mut exchange.data {
            placeholder_names_minified += minify_placeholder_names(&mut section.entities);
        }
    }

    let mut interned_by_type = BTreeMap::new();
    let mut interned_entities = 0usize;

    if options.intern_values {
        for section in &mut exchange.data {
            let pass = intern_section(&mut section.entities);
            interned_entities += pass.total;
            for (k, v) in pass.by_type {
                *interned_by_type.entry(k).or_insert(0) += v;
            }
        }
    }

    let mut consolidated_by_type = BTreeMap::new();
    let mut consolidated_entities = 0usize;
    if options.consolidate_presentation {
        for section in &mut exchange.data {
            let pass = consolidate_presentation(&mut section.entities);
            consolidated_entities += pass.total;
            for (k, v) in pass.by_type {
                *consolidated_by_type.entry(k).or_insert(0) += v;
            }
        }
    }

    let mut straight_bspline_lines_recovered = 0usize;
    let mut straight_bspline_direction_groups = 0usize;
    let mut straight_bspline_points_removed = 0usize;
    if options.experimental_recover_straight_bspline_lines {
        for section in &mut exchange.data {
            let pass = line_recovery::recover_straight_bspline_lines(&mut section.entities);
            straight_bspline_lines_recovered += pass.curves_recovered;
            straight_bspline_direction_groups += pass.direction_groups;
            straight_bspline_points_removed += pass.orphan_points_removed;
        }
    }

    let mut exact_bezier_curves_recovered = 0usize;
    if options.experimental_recover_exact_bezier_curves {
        for section in &mut exchange.data {
            let pass = bezier_recovery::recover_exact_bezier_curves(&mut section.entities);
            exact_bezier_curves_recovered += pass.curves_recovered;
        }
    }

    let mut v_extrusion_surfaces_recovered = 0usize;
    let mut v_extrusion_rational_surfaces_recovered = 0usize;
    let mut v_extrusion_profile_curves_created = 0usize;
    let mut v_extrusion_points_removed = 0usize;
    if options.experimental_recover_v_extrusions {
        for section in &mut exchange.data {
            let pass = surface_recovery::recover_v_extrusion_surfaces(&mut section.entities);
            v_extrusion_surfaces_recovered += pass.surfaces_recovered;
            v_extrusion_rational_surfaces_recovered += pass.rational_surfaces_recovered;
            v_extrusion_profile_curves_created += pass.profile_curves_created;
            v_extrusion_points_removed += pass.orphan_points_removed;
        }
    }

    let mut geometric_supports_merged = 0usize;
    let mut geometric_support_entities_removed = 0usize;
    let mut geometric_planes_merged = 0usize;
    let mut geometric_lines_merged = 0usize;
    let mut geometric_cylinders_merged = 0usize;
    if options.experimental_intern_geometric_supports {
        for section in &mut exchange.data {
            let pass = geometric_intern::intern_geometric_supports(&mut section.entities);
            geometric_supports_merged += pass.supports_merged;
            geometric_support_entities_removed += pass.entities_removed;
            geometric_planes_merged += pass.planes_merged;
            geometric_lines_merged += pass.lines_merged;
            geometric_cylinders_merged += pass.cylinders_merged;
        }
    }

    let mut partition_components_recovered = 0usize;
    let mut partition_solids_merged = 0usize;
    let mut partition_interfaces_removed = 0usize;
    let mut partition_styles_retargeted = 0usize;
    let mut partition_entities_removed = 0usize;
    if options.experimental_recover_partitioned_bodies {
        for section in &mut exchange.data {
            let pass = partition_recovery::recover_partitioned_bodies(&mut section.entities);
            partition_components_recovered += pass.components;
            partition_solids_merged += pass.solids_merged;
            partition_interfaces_removed += pass.interfaces_removed;
            partition_styles_retargeted += pass.styles_retargeted;
            partition_entities_removed += pass.entities_removed;
        }
    }

    let mut face_coalesce_groups = 0usize;
    let mut face_coalesce_faces_merged = 0usize;
    let mut face_coalesce_faces_removed = 0usize;
    let mut face_coalesce_internal_edges_removed = 0usize;
    let mut face_coalesce_styles_removed = 0usize;
    let mut face_coalesce_entities_removed = 0usize;
    if options.experimental_coalesce_same_support_faces {
        for section in &mut exchange.data {
            let pass = face_coalesce::coalesce_same_support_faces(&mut section.entities);
            face_coalesce_groups += pass.groups;
            face_coalesce_faces_merged += pass.faces_merged;
            face_coalesce_faces_removed += pass.faces_removed;
            face_coalesce_internal_edges_removed += pass.internal_edges_removed;
            face_coalesce_styles_removed += pass.styles_removed;
            face_coalesce_entities_removed += pass.entities_removed;
        }
    }

    let mut curve_replica_families = 0usize;
    let mut curve_replicas = 0usize;
    let mut curve_replica_direct_aliases = 0usize;
    let mut curve_replica_transforms = 0usize;
    let mut curve_replica_entities_removed = 0usize;
    let mut curve_replica_max_residual_mm = 0.0f64;

    let mut instance_groups = 0usize;
    let mut instanced_solids = 0usize;
    let mut instance_entities_removed = 0usize;
    let mut instance_styles_replaced = 0usize;
    if options.experimental_instance_z90_assembly {
        for section in &mut exchange.data {
            let pass = instances::instance_z90_solids_assembly(&mut section.entities);
            instance_groups += pass.groups;
            instanced_solids += pass.solids_replaced;
            instance_entities_removed += pass.entities_removed;
            instance_styles_replaced += pass.styles_replaced;
        }
    } else if options.experimental_instance_z90 {
        for section in &mut exchange.data {
            let pass = instances::instance_z90_solids(&mut section.entities);
            instance_groups += pass.groups;
            instanced_solids += pass.solids_replaced;
            instance_entities_removed += pass.entities_removed;
            instance_styles_replaced += pass.styles_replaced;
        }
    }

    let mut planar_feature_arrays = 0usize;
    let mut planar_feature_families = 0usize;
    let mut planar_feature_instances = 0usize;
    let mut planar_feature_entities_removed = 0usize;
    let mut planar_feature_styles_replaced = 0usize;
    if options.experimental_instance_planar_positive_features {
        for section in &mut exchange.data {
            let pass = planar_features::instance_planar_positive_features(&mut section.entities);
            planar_feature_arrays += pass.arrays;
            planar_feature_families += pass.families;
            planar_feature_instances += pass.instances;
            planar_feature_entities_removed += pass.entities_removed;
            planar_feature_styles_replaced += pass.styles_replaced;
        }
    }

    let mut spherical_cap_arrays = 0usize;
    let mut spherical_cap_instances = 0usize;
    let mut spherical_cap_entities_removed = 0usize;
    let mut spherical_cap_styles_replaced = 0usize;
    if options.experimental_instance_spherical_caps {
        for section in &mut exchange.data {
            let pass = spherical_caps::instance_planar_spherical_caps(&mut section.entities);
            spherical_cap_arrays += pass.arrays;
            spherical_cap_instances += pass.instances;
            spherical_cap_entities_removed += pass.entities_removed;
            spherical_cap_styles_replaced += pass.styles_replaced;
        }
    }

    // Low-level curve factoring comes last. Higher-level body/feature repetition
    // must be recognized against the actual geometry first; otherwise a
    // CURVE_REPLICA decomposition can leak global source coordinates into a
    // later rigid-body signature and hide obvious whole-solid instances.
    if options.experimental_instance_translated_bspline_curves {
        for section in &mut exchange.data {
            let pass = curve_replicas::instance_translated_bspline_curves(&mut section.entities);
            curve_replica_families += pass.families;
            curve_replicas += pass.replicas;
            curve_replica_direct_aliases += pass.direct_aliases;
            curve_replica_transforms += pass.transforms;
            curve_replica_entities_removed += pass.entities_removed;
            curve_replica_max_residual_mm = curve_replica_max_residual_mm.max(pass.max_residual_mm);
        }
    }

    // Experimental passes can create new placeholder-labelled entities.
    // Minify those before the post-rewrite intern pass so name normalization
    // cannot create fresh duplicates that only disappear on a second run.
    if options.minify_placeholder_names {
        for section in &mut exchange.data {
            placeholder_names_minified += minify_placeholder_names(&mut section.entities);
        }
    }

    // Experimental passes create placements/directions and other support
    // values. Normalize them in the same invocation so aggressive output is a
    // fixed point rather than requiring a second safe cleanup pass.
    if (straight_bspline_lines_recovered > 0
        || v_extrusion_surfaces_recovered > 0
        || geometric_supports_merged > 0
        || partition_components_recovered > 0
        || face_coalesce_groups > 0
        || curve_replicas > 0
        || curve_replica_direct_aliases > 0
        || instance_groups > 0
        || planar_feature_arrays > 0
        || spherical_cap_arrays > 0)
        && options.intern_values
    {
        for section in &mut exchange.data {
            let pass = intern_section(&mut section.entities);
            interned_entities += pass.total;
            for (k, v) in pass.by_type {
                *interned_by_type.entry(k).or_insert(0) += v;
            }
        }
    }

    if options.dense_ids {
        for section in &mut exchange.data {
            dense_renumber(&mut section.entities);
        }
    }

    let (patterns, periodic_bodies, count_parameters) = detect_exchange_semantics(&exchange);
    let instance_patterns_detected = patterns.len();
    let pattern_instances_detected = patterns.iter().map(|pattern| pattern.item_ids.len()).sum();
    let periodic_body_patterns_detected = periodic_bodies.len();
    let periodic_body_repeat_faces = periodic_bodies.iter().map(|body| body.repeat_faces).sum();
    let count_parameters_detected = count_parameters.len();
    let count_parameters_with_body_grammar = count_parameters
        .iter()
        .filter(|parameter| parameter.body_grammar_proven)
        .count();

    let compatibility = audit_exchange_compatibility(&exchange);

    let output = write_exchange(&exchange)?;
    let output_entities: usize = exchange.data.iter().map(|d| d.entities.len()).sum();
    let output_bytes = output.len();

    Ok(CleanOutput {
        bytes: output.into_bytes(),
        stats: Stats {
            input_encoding: input_encoding.to_string(),
            input_bytes: input.len(),
            output_bytes,
            input_entities,
            output_entities,
            interned_entities,
            consolidated_entities,
            straight_bspline_lines_recovered,
            straight_bspline_direction_groups,
            straight_bspline_points_removed,
            exact_bezier_curves_recovered,
            v_extrusion_surfaces_recovered,
            v_extrusion_rational_surfaces_recovered,
            v_extrusion_profile_curves_created,
            v_extrusion_points_removed,
            geometric_supports_merged,
            geometric_support_entities_removed,
            geometric_planes_merged,
            geometric_lines_merged,
            geometric_cylinders_merged,
            partition_components_recovered,
            partition_solids_merged,
            partition_interfaces_removed,
            partition_styles_retargeted,
            partition_entities_removed,
            face_coalesce_groups,
            face_coalesce_faces_merged,
            face_coalesce_faces_removed,
            face_coalesce_internal_edges_removed,
            face_coalesce_styles_removed,
            face_coalesce_entities_removed,
            curve_replica_families,
            curve_replicas,
            curve_replica_direct_aliases,
            curve_replica_transforms,
            curve_replica_entities_removed,
            curve_replica_max_residual_mm,
            instance_groups,
            instanced_solids,
            instance_entities_removed,
            instance_styles_replaced,
            planar_feature_arrays,
            planar_feature_families,
            planar_feature_instances,
            planar_feature_entities_removed,
            planar_feature_styles_replaced,
            spherical_cap_arrays,
            spherical_cap_instances,
            spherical_cap_entities_removed,
            spherical_cap_styles_replaced,
            placeholder_names_minified,
            instance_patterns_detected,
            pattern_instances_detected,
            periodic_body_patterns_detected,
            periodic_body_repeat_faces,
            count_parameters_detected,
            count_parameters_with_body_grammar,
            byte_ratio: output_bytes as f64 / input.len().max(1) as f64,
            interned_by_type,
            consolidated_by_type,
        },
        patterns,
        periodic_bodies,
        count_parameters,
        compatibility,
    })
}

#[derive(Default)]
struct ConsolidateStats {
    total: usize,
    by_type: BTreeMap<String, usize>,
}

fn consolidate_presentation(entities: &mut Vec<EntityInstance>) -> ConsolidateStats {
    use std::collections::{HashMap, HashSet};

    let mut refcounts: HashMap<u64, usize> = HashMap::new();
    for entity in entities.iter() {
        visit_entity_refs(entity, &mut |id| *refcounts.entry(id).or_insert(0) += 1);
    }

    #[derive(Clone)]
    struct Merge {
        into: usize,
        from: usize,
        items: Vec<Parameter>,
        ty: &'static str,
    }

    let mut groups: HashMap<String, usize> = HashMap::new();
    let mut merges = Vec::new();

    for (idx, entity) in entities.iter().enumerate() {
        let EntityInstance::Simple { id, record } = entity else {
            continue;
        };
        if refcounts.get(id).copied().unwrap_or(0) != 0 {
            continue;
        }
        let Parameter::List(params) = &record.parameter else {
            continue;
        };

        let (ty, key, items) = match record.name.as_str() {
            "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION" if params.len() == 3 => {
                let Parameter::List(items) = &params[1] else {
                    continue;
                };
                let key = format!(
                    "MDGPR|{}|{}",
                    standalone_param_key(&params[0]),
                    standalone_param_key(&params[2])
                );
                (
                    "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION",
                    key,
                    items.clone(),
                )
            }
            "PRESENTATION_LAYER_ASSIGNMENT" if params.len() == 3 => {
                let Parameter::List(items) = &params[2] else {
                    continue;
                };
                let key = format!(
                    "PLA|{}|{}",
                    standalone_param_key(&params[0]),
                    standalone_param_key(&params[1])
                );
                ("PRESENTATION_LAYER_ASSIGNMENT", key, items.clone())
            }
            _ => continue,
        };

        if let Some(&into) = groups.get(&key) {
            merges.push(Merge {
                into,
                from: idx,
                items,
                ty,
            });
        } else {
            groups.insert(key, idx);
        }
    }

    let mut additions: HashMap<usize, Vec<Parameter>> = HashMap::new();
    let mut remove = HashSet::new();
    let mut stats = ConsolidateStats::default();
    for merge in merges {
        additions.entry(merge.into).or_default().extend(merge.items);
        remove.insert(merge.from);
        stats.total += 1;
        *stats.by_type.entry(merge.ty.to_string()).or_insert(0) += 1;
    }

    for (idx, items) in additions {
        let EntityInstance::Simple { record, .. } = &mut entities[idx] else {
            unreachable!();
        };
        let Parameter::List(params) = &mut record.parameter else {
            unreachable!();
        };
        let target_idx = if record.name == "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION"
        {
            1
        } else {
            2
        };
        let Parameter::List(existing) = &mut params[target_idx] else {
            unreachable!();
        };
        existing.extend(items);
    }

    let mut i = 0usize;
    entities.retain(|_| {
        let keep = !remove.contains(&i);
        i += 1;
        keep
    });
    stats
}

fn visit_entity_refs(entity: &EntityInstance, f: &mut impl FnMut(u64)) {
    match entity {
        EntityInstance::Simple { record, .. } => visit_param_refs(&record.parameter, f),
        EntityInstance::Complex { subsuper, .. } => {
            for record in &subsuper.0 {
                visit_param_refs(&record.parameter, f);
            }
        }
    }
}

fn visit_param_refs(param: &Parameter, f: &mut impl FnMut(u64)) {
    match param {
        Parameter::Ref(Name::Entity(id)) => f(*id),
        Parameter::List(items) => {
            for item in items {
                visit_param_refs(item, f);
            }
        }
        Parameter::Typed { parameter, .. } => visit_param_refs(parameter, f),
        _ => {}
    }
}

fn standalone_param_key(param: &Parameter) -> String {
    let mut out = String::new();
    write_param_key(param, &HashMap::new(), &mut out);
    out
}

#[derive(Default)]
struct InternStats {
    total: usize,
    by_type: BTreeMap<String, usize>,
}

fn intern_section(entities: &mut Vec<EntityInstance>) -> InternStats {
    // Store redirects only. An identity map for a million-entity STEP file is
    // a surprisingly expensive way of spelling "most things survive".
    let mut alias: HashMap<u64, u64> = HashMap::new();

    // Value DAGs in the EasyEDA/SolidWorks corpus settle in a handful of
    // rounds (units -> uncertainty/context, colour -> style chains, geometry
    // primitives -> placements/surfaces). Updates are applied at the end of a
    // round so keys within that round see a stable alias map.
    for _ in 0..16 {
        let mut seen: HashMap<String, u64> = HashMap::new();
        let mut pending: Vec<(u64, u64)> = Vec::new();

        for entity in entities.iter() {
            if !is_internable(entity) {
                continue;
            }
            let id = entity_id(entity);
            if alias.contains_key(&id) {
                continue;
            }

            let key = entity_key(entity, &alias);
            if let Some(&canonical) = seen.get(&key) {
                let canonical = resolve_alias(&alias, canonical);
                if canonical != id {
                    pending.push((id, canonical));
                }
            } else {
                seen.insert(key, id);
            }
        }

        if pending.is_empty() {
            break;
        }
        for (id, canonical) in pending {
            alias.insert(id, canonical);
        }
        compress_aliases(&mut alias);
    }
    compress_aliases(&mut alias);

    let mut stats = InternStats::default();
    let original = std::mem::take(entities);
    entities.reserve(original.len().saturating_sub(alias.len()));
    for mut entity in original {
        let id = entity_id(&entity);
        let root = resolve_alias(&alias, id);
        if root != id {
            stats.total += 1;
            *stats.by_type.entry(entity_type_label(&entity)).or_insert(0) += 1;
            continue;
        }
        rewrite_entity_refs(&mut entity, &alias);
        entities.push(entity);
    }
    stats
}

fn compress_aliases(alias: &mut HashMap<u64, u64>) {
    let keys: Vec<u64> = alias.keys().copied().collect();
    for id in keys {
        let root = resolve_alias(alias, id);
        if root != id {
            alias.insert(id, root);
        }
    }
}

fn resolve_alias(alias: &HashMap<u64, u64>, mut id: u64) -> u64 {
    for _ in 0..64 {
        let Some(&next) = alias.get(&id) else {
            return id;
        };
        if next == id {
            return id;
        }
        id = next;
    }
    id
}

fn dense_renumber(entities: &mut [EntityInstance]) {
    let id_map: HashMap<u64, u64> = entities
        .iter()
        .enumerate()
        .map(|(idx, e)| (entity_id(e), idx as u64 + 1))
        .collect();

    for entity in entities.iter_mut() {
        let old = entity_id(entity);
        let new = id_map[&old];
        set_entity_id(entity, new);
        rewrite_entity_refs(entity, &id_map);
    }
}

fn entity_id(entity: &EntityInstance) -> u64 {
    match entity {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id,
    }
}

fn set_entity_id(entity: &mut EntityInstance, new_id: u64) {
    match entity {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id = new_id,
    }
}

fn entity_type_label(entity: &EntityInstance) -> String {
    match entity {
        EntityInstance::Simple { record, .. } => record.name.clone(),
        EntityInstance::Complex { subsuper, .. } => subsuper
            .0
            .iter()
            .map(|r| r.name.as_str())
            .collect::<Vec<_>>()
            .join("+"),
    }
}

fn rewrite_entity_refs(entity: &mut EntityInstance, map: &HashMap<u64, u64>) {
    match entity {
        EntityInstance::Simple { record, .. } => rewrite_param_refs(&mut record.parameter, map),
        EntityInstance::Complex { subsuper, .. } => {
            for record in &mut subsuper.0 {
                rewrite_param_refs(&mut record.parameter, map);
            }
        }
    }
}

fn rewrite_param_refs(param: &mut Parameter, map: &HashMap<u64, u64>) {
    match param {
        Parameter::Ref(Name::Entity(id)) => {
            if let Some(&new) = map.get(id) {
                *id = new;
            }
        }
        Parameter::List(items) => {
            for item in items {
                rewrite_param_refs(item, map);
            }
        }
        Parameter::Typed { parameter, .. } => rewrite_param_refs(parameter, map),
        _ => {}
    }
}

fn is_internable(entity: &EntityInstance) -> bool {
    match entity {
        EntityInstance::Simple { record, .. } => internable_record(&record.name),
        EntityInstance::Complex { subsuper, .. } => {
            !subsuper.0.is_empty() && subsuper.0.iter().all(|r| internable_record(&r.name))
        }
    }
}

// Deliberately excludes topological identity objects: VERTEX_POINT, EDGE_CURVE,
// ORIENTED_EDGE, EDGE_LOOP, FACE_*, ADVANCED_FACE, shells, and solids.
//
// Sharing these value/geometry-support objects does not merge topology. It only
// makes multiple topological objects point at the same equal geometry/style/unit
// value, which is the redundancy SolidWorks explodes in these EasyEDA files.
fn internable_record(name: &str) -> bool {
    matches!(
        name,
        // Geometry values / support geometry
        "CARTESIAN_POINT"
            | "DIRECTION"
            | "VECTOR"
            | "AXIS1_PLACEMENT"
            | "AXIS2_PLACEMENT_2D"
            | "AXIS2_PLACEMENT_3D"
            | "LINE"
            | "CIRCLE"
            | "ELLIPSE"
            | "PLANE"
            | "CYLINDRICAL_SURFACE"
            | "CONICAL_SURFACE"
            | "SPHERICAL_SURFACE"
            | "TOROIDAL_SURFACE"
            | "SURFACE_OF_LINEAR_EXTRUSION"
            | "SURFACE_OF_REVOLUTION"
            | "B_SPLINE_CURVE"
            | "B_SPLINE_CURVE_WITH_KNOTS"
            | "RATIONAL_B_SPLINE_CURVE"
            | "B_SPLINE_SURFACE"
            | "B_SPLINE_SURFACE_WITH_KNOTS"
            | "RATIONAL_B_SPLINE_SURFACE"
            // Presentation/style values
            | "COLOUR_RGB"
            | "DRAUGHTING_PRE_DEFINED_COLOUR"
            | "DRAUGHTING_PRE_DEFINED_CURVE_FONT"
            | "CURVE_STYLE"
            | "POINT_STYLE"
            | "FILL_AREA_STYLE_COLOUR"
            | "FILL_AREA_STYLE"
            | "SURFACE_STYLE_FILL_AREA"
            | "SURFACE_SIDE_STYLE"
            | "SURFACE_STYLE_USAGE"
            | "PRESENTATION_STYLE_ASSIGNMENT"
            // Units / contexts / uncertainty values
            | "NAMED_UNIT"
            | "SI_UNIT"
            | "LENGTH_UNIT"
            | "PLANE_ANGLE_UNIT"
            | "SOLID_ANGLE_UNIT"
            | "CONVERSION_BASED_UNIT"
            | "MEASURE_WITH_UNIT"
            | "UNCERTAINTY_MEASURE_WITH_UNIT"
            | "REPRESENTATION_CONTEXT"
            | "GEOMETRIC_REPRESENTATION_CONTEXT"
            | "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT"
            | "GLOBAL_UNIT_ASSIGNED_CONTEXT"
    )
}

fn entity_key(entity: &EntityInstance, alias: &HashMap<u64, u64>) -> String {
    let mut out = String::new();
    match entity {
        EntityInstance::Simple { record, .. } => write_record_key(record, alias, &mut out),
        EntityInstance::Complex { subsuper, .. } => {
            out.push('(');
            for record in &subsuper.0 {
                write_record_key(record, alias, &mut out);
            }
            out.push(')');
        }
    }
    out
}

fn write_record_key(record: &Record, alias: &HashMap<u64, u64>, out: &mut String) {
    out.push_str(&record.name);
    write_param_key(&record.parameter, alias, out);
}

fn write_param_key(param: &Parameter, alias: &HashMap<u64, u64>, out: &mut String) {
    match param {
        Parameter::Typed { keyword, parameter } => {
            out.push_str(keyword);
            out.push('(');
            write_param_key(parameter, alias, out);
            out.push(')');
        }
        Parameter::Integer(v) => {
            let _ = write!(out, "{v}");
        }
        Parameter::Real(v) => out.push_str(&format_real(*v)),
        Parameter::String(s) => write_step_string(s, out),
        Parameter::Enumeration(s) => {
            out.push('.');
            out.push_str(s);
            out.push('.');
        }
        Parameter::List(items) => {
            out.push('(');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_param_key(item, alias, out);
            }
            out.push(')');
        }
        Parameter::Ref(Name::Entity(id)) => {
            let _ = write!(out, "#{}", resolve_alias(alias, *id));
        }
        Parameter::Ref(Name::Value(id)) => {
            let _ = write!(out, "@{id}");
        }
        Parameter::Ref(Name::ConstantEntity(s)) => {
            out.push('#');
            out.push_str(s);
        }
        Parameter::Ref(Name::ConstantValue(s)) => {
            out.push('@');
            out.push_str(s);
        }
        Parameter::NotProvided => out.push('$'),
        Parameter::Omitted => out.push('*'),
    }
}

const PLACEHOLDER_NAME_TYPES: &[&str] = &[
    "ADVANCED_FACE",
    "AXIS2_PLACEMENT_3D",
    "B_SPLINE_CURVE_WITH_KNOTS",
    "B_SPLINE_SURFACE_WITH_KNOTS",
    "CARTESIAN_POINT",
    "CIRCLE",
    "CLOSED_SHELL",
    "CONICAL_SURFACE",
    "CYLINDRICAL_SURFACE",
    "DIRECTION",
    "EDGE_CURVE",
    "EDGE_LOOP",
    "FACE_BOUND",
    "FACE_OUTER_BOUND",
    "LINE",
    "MANIFOLD_SOLID_BREP",
    "ORIENTED_EDGE",
    "PLANE",
    "SPHERICAL_SURFACE",
    "STYLED_ITEM",
    "TOROIDAL_SURFACE",
    "VECTOR",
    "VERTEX_POINT",
];

fn minify_placeholder_names(entities: &mut [EntityInstance]) -> usize {
    let mut changed = 0usize;
    for entity in entities {
        match entity {
            EntityInstance::Simple { record, .. } => {
                changed += minify_placeholder_record_name(record);
            }
            EntityInstance::Complex { subsuper, .. } => {
                for record in &mut subsuper.0 {
                    changed += minify_placeholder_record_name(record);
                }
            }
        }
    }
    changed
}

fn minify_placeholder_record_name(record: &mut Record) -> usize {
    if !PLACEHOLDER_NAME_TYPES.contains(&record.name.as_str()) {
        return 0;
    }
    let Parameter::List(params) = &mut record.parameter else {
        return 0;
    };
    let Some(Parameter::String(name)) = params.first_mut() else {
        return 0;
    };
    if name != "NONE" {
        return 0;
    }
    name.clear();
    1
}

pub fn write_exchange(exchange: &Exchange) -> Result<String> {
    if !exchange.anchor.is_empty()
        || !exchange.reference.is_empty()
        || !exchange.signature.is_empty()
    {
        bail!("optional STEP sections are not supported by writer");
    }

    let mut out = String::with_capacity(
        exchange
            .data
            .iter()
            .map(|d| d.entities.len())
            .sum::<usize>()
            * 48,
    );
    out.push_str("ISO-10303-21;\nHEADER;\n");
    for record in &exchange.header {
        write_record(record, &mut out);
        out.push_str(";\n");
    }
    out.push_str("ENDSEC;\n");

    for section in &exchange.data {
        if section.meta.is_empty() {
            out.push_str("DATA;\n");
        } else {
            out.push_str("DATA(");
            for (i, param) in section.meta.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_param(param, &mut out);
            }
            out.push_str(");\n");
        }
        for entity in &section.entities {
            write_entity(entity, &mut out);
            out.push('\n');
        }
        out.push_str("ENDSEC;\n");
    }
    out.push_str("END-ISO-10303-21;\n");
    Ok(out)
}

fn write_entity(entity: &EntityInstance, out: &mut String) {
    match entity {
        EntityInstance::Simple { id, record } => {
            let _ = write!(out, "#{id}=");
            write_record(record, out);
            out.push(';');
        }
        EntityInstance::Complex { id, subsuper } => {
            let _ = write!(out, "#{id}=(");
            for record in &subsuper.0 {
                write_record(record, out);
            }
            out.push_str(");");
        }
    }
}

fn write_record(record: &Record, out: &mut String) {
    out.push_str(&record.name);
    write_param(&record.parameter, out);
}

fn write_param(param: &Parameter, out: &mut String) {
    match param {
        Parameter::Typed { keyword, parameter } => {
            out.push_str(keyword);
            out.push('(');
            write_param(parameter, out);
            out.push(')');
        }
        Parameter::Integer(v) => {
            let _ = write!(out, "{v}");
        }
        Parameter::Real(v) => out.push_str(&format_real(*v)),
        Parameter::String(s) => write_step_string(s, out),
        Parameter::Enumeration(s) => {
            out.push('.');
            out.push_str(s);
            out.push('.');
        }
        Parameter::List(items) => {
            out.push('(');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_param(item, out);
            }
            out.push(')');
        }
        Parameter::Ref(Name::Entity(id)) => {
            let _ = write!(out, "#{id}");
        }
        Parameter::Ref(Name::Value(id)) => {
            let _ = write!(out, "@{id}");
        }
        Parameter::Ref(Name::ConstantEntity(s)) => {
            out.push('#');
            out.push_str(s);
        }
        Parameter::Ref(Name::ConstantValue(s)) => {
            out.push('@');
            out.push_str(s);
        }
        Parameter::NotProvided => out.push('$'),
        Parameter::Omitted => out.push('*'),
    }
}

const EMPTY_AGGREGATE_MARKER: &str = "STEPREDOXEMPTYAGGREGATE";

fn prepare_parser_input(input: &str) -> Result<(std::borrow::Cow<'_, str>, bool)> {
    // ruststep 0.4 documents aggregate contents as optional, but its
    // comma_separated() parser currently requires at least one parameter. A
    // few valid AP214 exporters emit empty aggregates such as
    // SHAPE_REPRESENTATION('',(),#ctx). Encode those with an impossible
    // enumeration sentinel for parsing, then restore them in the AST.
    let Some(data_start) = input.find("DATA;") else {
        return Ok((std::borrow::Cow::Borrowed(input), false));
    };
    let scan_start = data_start + "DATA;".len();
    let suffix = &input[scan_start..];
    if !suffix.as_bytes().windows(2).any(|w| w == b"()")
        && !suffix.contains("( ")
        && !suffix.contains("(\t")
        && !suffix.contains("(\r")
        && !suffix.contains("(\n")
    {
        return Ok((std::borrow::Cow::Borrowed(input), false));
    }
    let marker = format!(".{EMPTY_AGGREGATE_MARKER}.");
    if input.contains(&marker) {
        bail!("STEP input collides with step-redox empty-aggregate parser marker");
    }

    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len() + 64);
    out.push_str(&input[..scan_start]);

    let mut i = scan_start;
    let mut last = scan_start;
    let mut in_string = false;
    let mut in_comment = false;
    let mut replaced = false;

    while i < bytes.len() {
        if in_comment {
            if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                in_comment = false;
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }

        if in_string {
            if bytes[i] == b'\'' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                } else {
                    in_string = false;
                    i += 1;
                }
            } else {
                i += 1;
            }
            continue;
        }

        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            in_comment = true;
            i += 2;
            continue;
        }
        if bytes[i] == b'\'' {
            in_string = true;
            i += 1;
            continue;
        }
        if bytes[i] == b'(' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b')' {
                out.push_str(&input[last..i]);
                out.push('(');
                out.push_str(&marker);
                out.push(')');
                i = j + 1;
                last = i;
                replaced = true;
                continue;
            }
        }
        i += 1;
    }

    if !replaced {
        return Ok((std::borrow::Cow::Borrowed(input), false));
    }
    out.push_str(&input[last..]);
    Ok((std::borrow::Cow::Owned(out), true))
}

fn restore_empty_aggregates(exchange: &mut Exchange) -> Result<()> {
    for section in &mut exchange.data {
        for entity in &mut section.entities {
            match entity {
                EntityInstance::Simple { record, .. } => {
                    restore_empty_aggregate_param(&mut record.parameter)?;
                }
                EntityInstance::Complex { subsuper, .. } => {
                    for record in &mut subsuper.0 {
                        restore_empty_aggregate_param(&mut record.parameter)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn restore_empty_aggregate_param(parameter: &mut Parameter) -> Result<()> {
    match parameter {
        Parameter::List(items) => {
            if items.len() == 1
                && matches!(
                    &items[0],
                    Parameter::Enumeration(value) if value == EMPTY_AGGREGATE_MARKER
                )
            {
                items.clear();
                return Ok(());
            }
            for item in items {
                restore_empty_aggregate_param(item)?;
            }
        }
        Parameter::Typed { parameter, .. } => {
            if matches!(
                parameter.as_ref(),
                Parameter::Enumeration(value) if value == EMPTY_AGGREGATE_MARKER
            ) {
                bail!("unsupported empty typed-parameter aggregate in STEP input");
            }
            restore_empty_aggregate_param(parameter)?;
        }
        Parameter::Enumeration(value) if value == EMPTY_AGGREGATE_MARKER => {
            bail!("empty-aggregate parser marker escaped its aggregate");
        }
        _ => {}
    }
    Ok(())
}

fn decode_input(input: &[u8]) -> Result<(std::borrow::Cow<'_, str>, &'static str)> {
    if let Ok(s) = std::str::from_utf8(input) {
        return Ok((std::borrow::Cow::Borrowed(s), "utf-8"));
    }

    let (decoded, _used_encoding, had_errors) = encoding_rs::GBK.decode(input);
    if had_errors {
        bail!("STEP input is neither valid UTF-8 nor valid GBK");
    }
    Ok((decoded, "gbk"))
}

fn write_step_string(s: &str, out: &mut String) {
    out.push('\'');

    let flush_non_ascii = |buf: &mut String, out: &mut String| {
        if buf.is_empty() {
            return;
        }
        out.push_str("\\X2\\");
        for unit in buf.encode_utf16() {
            let _ = write!(out, "{unit:04X}");
        }
        out.push_str("\\X0\\");
        buf.clear();
    };

    let mut non_ascii = String::new();
    for ch in s.chars() {
        if ch.is_ascii() && ch != '\'' {
            flush_non_ascii(&mut non_ascii, out);
            out.push(ch);
        } else {
            // Encode apostrophes too; this stays valid Part 21 and avoids
            // depending on ruststep's incomplete doubled-apostrophe parser.
            non_ascii.push(ch);
        }
    }
    flush_non_ascii(&mut non_ascii, out);
    out.push('\'');
}

fn format_real(v: f64) -> String {
    let mut s = v.to_string();
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        s.push('.');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wrap(data: &str) -> Vec<u8> {
        format!(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('x'),'1');\nFILE_NAME('a','b',(''),(''),'x','y','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n{data}\nENDSEC;\nEND-ISO-10303-21;\n"
        )
        .into_bytes()
    }

    #[test]
    fn writer_roundtrips_basic_exchange() {
        let src = wrap(
            "#9=CARTESIAN_POINT('NONE',(1.000000000000000000,2.500000000000000000,0.000000000000000000));\n#20=CARTESIAN_POINT('NONE',(1.0,2.5,0.0));\n#21=VERTEX_POINT('NONE',#20);",
        );
        let out = clean_bytes(&src, &Options::default()).unwrap();
        assert!(out.stats.output_bytes < out.stats.input_bytes);
        assert_eq!(out.stats.interned_entities, 1);
        ruststep::parser::parse(std::str::from_utf8(&out.bytes).unwrap()).unwrap();
    }

    #[test]
    fn equal_geometry_values_share_but_topology_identity_survives() {
        let src = wrap(
            "#1=CARTESIAN_POINT('',(1.0,2.0,3.0));\n#2=CARTESIAN_POINT('',(1.000000000000000000,2.0,3.0));\n#3=VERTEX_POINT('',#1);\n#4=VERTEX_POINT('',#2);",
        );
        let out = clean_bytes(&src, &Options::default()).unwrap();
        assert_eq!(out.stats.interned_entities, 1);
        assert_eq!(out.stats.output_entities, 3);

        let text = std::str::from_utf8(&out.bytes).unwrap();
        assert_eq!(text.matches("VERTEX_POINT").count(), 2);
        assert_eq!(text.matches("CARTESIAN_POINT").count(), 1);
    }

    #[test]
    fn consolidates_only_unreferenced_presentation_roots() {
        let src = wrap(
            "#1=CARTESIAN_POINT('',(0.0,0.0,0.0));\n\
             #8=DIRECTION('',(1.0,0.0,0.0));\n\
             #2=STYLED_ITEM('',(#8),#1);\n\
             #3=STYLED_ITEM('',(#8),#1);\n\
             #4=MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',(#2),#1);\n\
             #5=MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',(#3),#1);\n\
             #6=PRESENTATION_LAYER_ASSIGNMENT('','',(#2));\n\
             #7=PRESENTATION_LAYER_ASSIGNMENT('','',(#3));",
        );
        let out = clean_bytes(&src, &Options::default()).unwrap();
        assert_eq!(out.stats.consolidated_entities, 2);
        let text = std::str::from_utf8(&out.bytes).unwrap();
        assert_eq!(
            text.matches("MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION")
                .count(),
            1
        );
        assert_eq!(text.matches("PRESENTATION_LAYER_ASSIGNMENT").count(), 1);
    }

    #[test]
    fn referenced_presentation_records_are_not_consolidated() {
        let src = wrap(
            "#1=CARTESIAN_POINT('',(0.0,0.0,0.0));\n\
             #8=DIRECTION('',(1.0,0.0,0.0));\n\
             #2=STYLED_ITEM('',(#8),#1);\n\
             #3=STYLED_ITEM('',(#8),#1);\n\
             #4=MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',(#2),#1);\n\
             #5=MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',(#3),#1);\n\
             #6=REPRESENTATION_RELATIONSHIP('','',#4,#5);",
        );
        let out = clean_bytes(&src, &Options::default()).unwrap();
        assert_eq!(out.stats.consolidated_entities, 0);
    }

    #[test]
    fn legal_empty_aggregates_roundtrip_through_ruststep_compatibility_shim() {
        let src = wrap("#8=SHAPE_REPRESENTATION('',(),#6);\n#6=CARTESIAN_POINT('',(0.0,0.0,0.0));");
        let once = clean_bytes(&src, &Options::default()).unwrap();
        let text = std::str::from_utf8(&once.bytes).unwrap();
        assert!(text.contains("SHAPE_REPRESENTATION('',(),#"));
        assert!(!text.contains(EMPTY_AGGREGATE_MARKER));

        let twice = clean_bytes(&once.bytes, &Options::default()).unwrap();
        assert_eq!(once.bytes, twice.bytes);
    }

    #[test]
    fn empty_aggregate_shim_ignores_parentheses_inside_strings_and_comments() {
        let src = wrap(
            "#1=CARTESIAN_POINT('literal ()', (0.0,0.0,0.0));\n/* () */\n#2=SHAPE_REPRESENTATION('',( ),#1);",
        );
        let out = clean_bytes(&src, &Options::default()).unwrap();
        let text = std::str::from_utf8(&out.bytes).unwrap();
        assert!(text.contains("literal ()"));
        assert!(text.contains("SHAPE_REPRESENTATION('',(),#"));
        assert!(!text.contains(EMPTY_AGGREGATE_MARKER));
    }

    #[test]
    fn optional_placeholder_name_minification_only_touches_allowlisted_name_fields() {
        let src =
            wrap("#1=CARTESIAN_POINT('NONE',(0.0,0.0,0.0));\n#2=PRODUCT('NONE','NONE','NONE',());");
        let options = Options {
            minify_placeholder_names: true,
            ..Options::default()
        };
        let out = clean_bytes(&src, &options).unwrap();
        let text = std::str::from_utf8(&out.bytes).unwrap();
        assert!(text.contains("CARTESIAN_POINT('',"));
        assert!(text.contains("PRODUCT('NONE','NONE','NONE',())"));
        assert_eq!(out.stats.placeholder_names_minified, 1);
    }

    #[test]
    fn gbk_strings_become_standard_x2_unicode_escapes() {
        let mut src = b"ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('x'),'1');\nFILE_NAME('a','b',(''),(''),'x','y','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n#1=CARTESIAN_POINT('".to_vec();
        src.extend_from_slice(&[0xC8, 0xCE, 0xBA, 0xCE]); // 任何 in GBK
        src.extend_from_slice(b"',(0.0,0.0,0.0));\nENDSEC;\nEND-ISO-10303-21;\n");

        let out = clean_bytes(&src, &Options::default()).unwrap();
        assert_eq!(out.stats.input_encoding, "gbk");
        let text = std::str::from_utf8(&out.bytes).unwrap();
        assert!(text.contains("\\X2\\4EFB4F55\\X0\\"));
    }

    #[test]
    fn cleaning_is_byte_idempotent() {
        let src = wrap(
            "#10=DIRECTION('',(1.000000000000000000,0.0,0.0));\n#20=DIRECTION('',(1.0,0.0,0.0));\n#30=VECTOR('',#20,1000.000000000000000000);",
        );
        let once = clean_bytes(&src, &Options::default()).unwrap();
        let twice = clean_bytes(&once.bytes, &Options::default()).unwrap();
        assert_eq!(once.bytes, twice.bytes);
    }

    #[test]
    fn straight_bspline_recovery_is_byte_idempotent() {
        let src = wrap(
            "#1=CARTESIAN_POINT('',(0.,0.,0.));\n\
             #2=CARTESIAN_POINT('',(1.,0.,0.));\n\
             #3=CARTESIAN_POINT('',(2.,0.,0.));\n\
             #4=CARTESIAN_POINT('',(3.,0.,0.));\n\
             #5=VERTEX_POINT('',#1);\n\
             #6=VERTEX_POINT('',#4);\n\
             #7=B_SPLINE_CURVE_WITH_KNOTS('',3,(#1,#2,#3,#4),.UNSPECIFIED.,.F.,.F.,(4,4),(0.,1.),.UNSPECIFIED.);\n\
             #8=EDGE_CURVE('',#5,#6,#7,.T.);",
        );
        let options = Options {
            experimental_recover_straight_bspline_lines: true,
            ..Options::default()
        };
        let once = clean_bytes(&src, &options).unwrap();
        assert_eq!(once.stats.straight_bspline_lines_recovered, 1);
        let twice = clean_bytes(&once.bytes, &options).unwrap();
        assert_eq!(twice.stats.straight_bspline_lines_recovered, 0);
        assert_eq!(once.bytes, twice.bytes);
    }
}
