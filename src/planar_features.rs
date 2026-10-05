use crate::brep::{
    append_refs_to_list_param, bound_loop_edges, face_edge_curves, face_loops, face_sense,
    face_surface, manifold_shell, ref_list_param, remove_refs_from_list_param, toggle_tf,
};
use crate::instances::{
    StyleRef, collect_styles_by_target, face_topology_signature, oriented_edge_signature,
};
use crate::patterns::{
    PointLattice, PointMotifPattern, factor_point_motif_pattern, fit_point_lattice,
};
use crate::step_entities::{
    cartesian_point, entity_ref, patch_presentation_lists, push_point, push_simple,
    representation_items_and_context,
};
use crate::step_graph::{
    ReferenceGraph, build_index, entity_id, entity_ref_value, simple_record, simple_record_mut,
    visit_entity_refs,
};
use anyhow::{Result, bail};
use ruststep::ast::{EntityInstance, Parameter};
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};

const MIN_GROUP: usize = 8;
const MAX_FEATURE_FACES: usize = 128;
const MAX_BOUNDARY_FEATURE_FACES: usize = 4096;
const MIN_COMPLEX_HOST_EDGES: usize = 32;
const SIDE_TOLERANCE: f64 = 1.0e-8;

#[derive(Debug, Default, Clone)]
pub struct PlanarFeatureStats {
    pub arrays: usize,
    pub families: usize,
    pub instances: usize,
    pub entities_removed: usize,
    pub styles_replaced: usize,
}

#[derive(Debug, Clone)]
struct ShellContext {
    representation_id: u64,
    context_id: u64,
    container_id: u64,
    shell_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum StyleKey {
    Face(Vec<u64>),
    Container,
    None,
}

#[derive(Debug, Clone)]
struct Feature {
    faces: Vec<u64>,
    interface_bound: u64,
    interface_loop: u64,
    bound_orientation: String,
    center: [f64; 3],
    normalized_quarter: u8,
    signature: String,
    style_key: StyleKey,
    old_style_ids: Vec<u64>,
}

#[derive(Debug, Clone)]
struct PlaneFrame {
    surface_id: u64,
    sense: String,
    origin: [f64; 3],
    outward: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FeaturePolarity {
    Additive,
    Subtractive,
}

#[derive(Debug, Clone, Serialize)]
pub struct BoundaryFeatureInstanceEvidence {
    pub face_ids: Vec<u64>,
    pub center_mm: [f64; 3],
    pub interface_bound_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BoundaryFeatureFamilyEvidence {
    pub shell_id: u64,
    pub polarity: FeaturePolarity,
    pub host_face_ids: Vec<u64>,
    pub faces_per_instance: usize,
    pub instances: usize,
    pub signature: String,
    pub members: Vec<BoundaryFeatureInstanceEvidence>,
    pub lattice: Option<PointLattice>,
    pub motif_pattern: Option<PointMotifPattern>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BoundaryFeatureToolMaterialization {
    pub family_index: usize,
    pub solid_id: u64,
    pub shell_id: u64,
    pub polarity: FeaturePolarity,
    pub instances: usize,
    pub source_face_ids: Vec<u64>,
    pub interface_bound_ids: Vec<u64>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct BoundaryFeaturePeelStats {
    pub families: usize,
    pub instances: usize,
    pub additive_instances: usize,
    pub subtractive_instances: usize,
    pub faces_removed_from_shells: usize,
    pub interface_bounds_healed: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenChainRunEvidence {
    pub face_id: u64,
    pub bound_id: u64,
    pub outer: bool,
    pub start_edge_index: usize,
    pub motif_edges: usize,
    pub repeats: usize,
    pub covered_edges: usize,
    pub wraps_loop: bool,
    pub translation_mm: [f64; 3],
    pub motif_oriented_edge_ids: Vec<u64>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct OpenChainDiagnostics {
    pub planar_faces: usize,
    pub loops_considered: usize,
    pub patterned_loops: usize,
    pub runs: Vec<OpenChainRunEvidence>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct BoundaryFeatureDiagnostics {
    pub shell_contexts: usize,
    pub planar_hosts: usize,
    pub raw_candidates: usize,
    pub additive_candidates: usize,
    pub subtractive_candidates: usize,
    pub single_host_candidates: usize,
    pub multi_host_candidates: usize,
    pub families: Vec<BoundaryFeatureFamilyEvidence>,
}

#[derive(Debug, Clone)]
struct BoundaryFeatureCandidate {
    shell_id: u64,
    polarity: FeaturePolarity,
    host_face_ids: Vec<u64>,
    face_ids: Vec<u64>,
    interface_bound_ids: Vec<u64>,
    center: [f64; 3],
    signature: String,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct PlanarFeatureDiagnostics {
    pub shell_contexts: usize,
    pub qualifying_host_faces: usize,
    pub total_components: usize,
    pub rejected_empty_or_large: usize,
    pub rejected_non_manifold: usize,
    pub rejected_interface: usize,
    pub rejected_bound_match: usize,
    pub rejected_vertices: usize,
    pub rejected_signature: usize,
    pub positive_components: usize,
    pub negative_components: usize,
    pub straddling_components: usize,
    pub coplanar_components: usize,
    pub hosts: Vec<PlanarHostDiagnostic>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct PlanarHostDiagnostic {
    pub host_face_id: u64,
    pub bound_count: usize,
    pub outward: [f64; 3],
    pub component_count: usize,
    pub positive_components: usize,
    pub negative_components: usize,
    pub straddling_components: usize,
    pub coplanar_components: usize,
    pub largest_positive_signature_group: usize,
    pub largest_negative_signature_group: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlaneSide {
    Positive,
    Negative,
    Straddling,
    Coplanar,
}

pub fn diagnose_open_chain_patterns(entities: &[EntityInstance]) -> OpenChainDiagnostics {
    const MIN_LOOP_EDGES: usize = 16;
    const MAX_MOTIF_EDGES: usize = 32;
    const MIN_REPEATS: usize = 4;
    const TOL: f64 = 1.0e-7;

    let mut report = OpenChainDiagnostics::default();
    if entities.is_empty() {
        return report;
    }
    let index = build_index(entities);
    let contexts = collect_shell_contexts(entities, &index);

    for context in contexts {
        let Some(shell_faces) = ref_list_param(context.shell_id, 1, entities, &index) else {
            continue;
        };
        for face_id in shell_faces {
            if plane_frame(face_id, entities, &index).is_none() {
                continue;
            }
            report.planar_faces += 1;
            let Some(loops) = face_loops(face_id, entities, &index) else {
                continue;
            };

            for loop_data in loops {
                let n = loop_data.edges.len();
                if n < MIN_LOOP_EDGES {
                    continue;
                }
                report.loops_considered += 1;

                let signatures = loop_data
                    .edges
                    .iter()
                    .map(|edge| {
                        oriented_edge_signature(
                            edge.oriented_edge_id,
                            entities,
                            &index,
                            edge.start_mm,
                            0,
                        )
                    })
                    .collect::<Option<Vec<_>>>();
                let Some(signatures) = signatures else {
                    continue;
                };

                let mut candidates = Vec::<OpenChainRunEvidence>::new();
                let max_period = MAX_MOTIF_EDGES.min(n / MIN_REPEATS);
                for period in 1..=max_period {
                    if n % period == 0 {
                        collect_cyclic_open_chain_runs(
                            face_id,
                            &loop_data,
                            &signatures,
                            period,
                            MIN_REPEATS,
                            TOL,
                            &mut candidates,
                        );
                        continue;
                    }

                    for residue in 0..period {
                        let mut block = residue;
                        let mut run_start = residue;
                        let mut matched_boundaries = 0usize;

                        while block + 2 * period <= n {
                            let matched = translated_blocks_match(
                                &loop_data.edges,
                                &signatures,
                                block,
                                period,
                                TOL,
                            );
                            if matched {
                                if matched_boundaries == 0 {
                                    run_start = block;
                                }
                                matched_boundaries += 1;
                            } else {
                                if matched_boundaries + 1 >= MIN_REPEATS {
                                    candidates.push(open_chain_run(
                                        face_id,
                                        &loop_data,
                                        run_start,
                                        period,
                                        matched_boundaries + 1,
                                        false,
                                    ));
                                }
                                matched_boundaries = 0;
                            }
                            block += period;
                        }

                        if matched_boundaries + 1 >= MIN_REPEATS {
                            candidates.push(open_chain_run(
                                face_id,
                                &loop_data,
                                run_start,
                                period,
                                matched_boundaries + 1,
                                false,
                            ));
                        }
                    }
                }

                // Prefer maximal coverage, then the primitive (smallest)
                // translated motif. Greedily suppress overlapping aliases such
                // as 8-edge/35-repeat when a 4-edge/70-repeat path covers the
                // same boundary run.
                candidates.sort_by(|a, b| {
                    b.covered_edges
                        .cmp(&a.covered_edges)
                        .then_with(|| a.motif_edges.cmp(&b.motif_edges))
                        .then_with(|| b.repeats.cmp(&a.repeats))
                        .then_with(|| a.start_edge_index.cmp(&b.start_edge_index))
                });
                let mut accepted = Vec::<OpenChainRunEvidence>::new();
                let mut occupied = vec![false; n];
                for candidate in candidates {
                    let coverage = candidate.covered_edges.min(n);
                    let indices = (0..coverage)
                        .map(|offset| (candidate.start_edge_index + offset) % n)
                        .collect::<Vec<_>>();
                    if indices.iter().any(|&index| occupied[index]) {
                        continue;
                    }
                    for index in indices {
                        occupied[index] = true;
                    }
                    accepted.push(candidate);
                }
                if !accepted.is_empty() {
                    report.patterned_loops += 1;
                    report.runs.extend(accepted);
                }
            }
        }
    }

    report.runs.sort_by(|a, b| {
        b.covered_edges
            .cmp(&a.covered_edges)
            .then_with(|| b.repeats.cmp(&a.repeats))
            .then_with(|| a.face_id.cmp(&b.face_id))
            .then_with(|| a.bound_id.cmp(&b.bound_id))
            .then_with(|| a.start_edge_index.cmp(&b.start_edge_index))
    });
    report
}

fn collect_cyclic_open_chain_runs(
    face_id: u64,
    loop_data: &crate::brep::FaceLoop,
    signatures: &[String],
    period: usize,
    min_repeats: usize,
    tolerance_mm: f64,
    out: &mut Vec<OpenChainRunEvidence>,
) {
    let n = loop_data.edges.len();
    if period == 0 || n % period != 0 {
        return;
    }
    let block_count = n / period;
    if block_count < min_repeats {
        return;
    }

    for residue in 0..period {
        let matches = (0..block_count)
            .map(|block| {
                translated_blocks_match_cyclic(
                    &loop_data.edges,
                    signatures,
                    (residue + block * period) % n,
                    period,
                    tolerance_mm,
                )
            })
            .collect::<Vec<_>>();

        if matches.iter().all(|matched| *matched) {
            out.push(open_chain_run(
                face_id,
                loop_data,
                residue,
                period,
                block_count,
                false,
            ));
            continue;
        }

        let Some(cut) = matches.iter().position(|matched| !*matched) else {
            continue;
        };
        let mut run_start = None::<usize>;
        let mut run_len = 0usize;
        for step in 1..=block_count {
            let block = (cut + step) % block_count;
            if matches[block] {
                if run_start.is_none() {
                    run_start = Some(block);
                }
                run_len += 1;
                continue;
            }

            if let Some(start_block) = run_start.take() {
                let repeats = run_len + 1;
                if repeats >= min_repeats {
                    let start = (residue + start_block * period) % n;
                    let covered_edges = period.saturating_mul(repeats);
                    out.push(open_chain_run(
                        face_id,
                        loop_data,
                        start,
                        period,
                        repeats,
                        start + covered_edges > n,
                    ));
                }
                run_len = 0;
            }
        }
        if let Some(start_block) = run_start {
            let repeats = run_len + 1;
            if repeats >= min_repeats {
                let start = (residue + start_block * period) % n;
                let covered_edges = period.saturating_mul(repeats);
                out.push(open_chain_run(
                    face_id,
                    loop_data,
                    start,
                    period,
                    repeats,
                    start + covered_edges > n,
                ));
            }
        }
    }
}

fn translated_blocks_match_cyclic(
    edges: &[crate::brep::OrientedEdgeUse],
    signatures: &[String],
    start: usize,
    period: usize,
    tolerance_mm: f64,
) -> bool {
    let n = edges.len();
    if n == 0 || period == 0 || period >= n {
        return false;
    }
    let right_start = (start + period) % n;
    let translation = sub3d(edges[right_start].start_mm, edges[start].start_mm);
    if translation.iter().all(|value| value.abs() <= tolerance_mm) {
        return false;
    }

    for offset in 0..period {
        let left = (start + offset) % n;
        let right = (start + period + offset) % n;
        if signatures[left] != signatures[right]
            || point_distance(
                add3d(edges[left].start_mm, translation),
                edges[right].start_mm,
            ) > tolerance_mm
            || point_distance(add3d(edges[left].end_mm, translation), edges[right].end_mm)
                > tolerance_mm
        {
            return false;
        }
    }
    true
}

fn translated_blocks_match(
    edges: &[crate::brep::OrientedEdgeUse],
    signatures: &[String],
    start: usize,
    period: usize,
    tolerance_mm: f64,
) -> bool {
    if start + 2 * period > edges.len() {
        return false;
    }
    let translation = sub3d(edges[start + period].start_mm, edges[start].start_mm);
    if translation.iter().all(|value| value.abs() <= tolerance_mm) {
        return false;
    }

    for offset in 0..period {
        let left = start + offset;
        let right = left + period;
        if signatures[left] != signatures[right]
            || point_distance(
                add3d(edges[left].start_mm, translation),
                edges[right].start_mm,
            ) > tolerance_mm
            || point_distance(add3d(edges[left].end_mm, translation), edges[right].end_mm)
                > tolerance_mm
        {
            return false;
        }
    }
    true
}

fn open_chain_run(
    face_id: u64,
    loop_data: &crate::brep::FaceLoop,
    start: usize,
    period: usize,
    repeats: usize,
    wraps_loop: bool,
) -> OpenChainRunEvidence {
    let n = loop_data.edges.len();
    let translation = sub3d(
        loop_data.edges[(start + period) % n].start_mm,
        loop_data.edges[start].start_mm,
    );
    OpenChainRunEvidence {
        face_id,
        bound_id: loop_data.bound_id,
        outer: loop_data.outer,
        start_edge_index: start,
        motif_edges: period,
        repeats,
        covered_edges: period * repeats,
        wraps_loop,
        translation_mm: translation,
        motif_oriented_edge_ids: (0..period)
            .map(|offset| loop_data.edges[(start + offset) % n].oriented_edge_id)
            .collect(),
    }
}

fn add3d(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn sub3d(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn point_distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    let delta = sub3d(left, right);
    (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt()
}

/// Materialize one closed canonical tool solid per recovered boundary-feature
/// family without changing the source solid.
///
/// This is an analysis aid for recursive decomposition. The generated tool
/// solid reuses the canonical feature's source faces and closes each planar
/// carrier interface with a synthetic cap. Additive tools need the interface
/// cap normal opposite the source host's outward normal; subtractive tools
/// need it aligned with the host outward normal.
pub(crate) fn materialize_boundary_feature_tools_for_analysis(
    entities: &mut Vec<EntityInstance>,
    min_instances: usize,
) -> Result<(
    BoundaryFeatureDiagnostics,
    Vec<BoundaryFeatureToolMaterialization>,
)> {
    if min_instances == 0 {
        bail!("boundary feature tool materialization requires min_instances > 0");
    }

    let initial_index = build_index(entities);
    let representation_by_shell = collect_shell_contexts(entities, &initial_index)
        .into_iter()
        .map(|context| (context.shell_id, context.representation_id))
        .collect::<HashMap<_, _>>();
    let report = diagnose_boundary_features(entities);
    let mut next_id = entities
        .iter()
        .map(entity_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let mut materialized = Vec::new();

    for (family_index, family) in report.families.iter().enumerate() {
        if family.instances < min_instances {
            continue;
        }
        let Some(member) = family.members.first() else {
            continue;
        };

        let index = build_index(entities);
        let mut bound_owner = HashMap::<u64, u64>::new();
        for &host in &family.host_face_ids {
            let Some(bounds) = ref_list_param(host, 1, entities, &index) else {
                bail!("feature host face #{host} has no readable bounds");
            };
            for bound in bounds {
                if let Some(previous) = bound_owner.insert(bound, host)
                    && previous != host
                {
                    bail!("interface bound #{bound} is owned by multiple host faces");
                }
            }
        }

        let mut cap_faces = Vec::new();
        for &bound_id in &member.interface_bound_ids {
            let Some(&host_id) = bound_owner.get(&bound_id) else {
                bail!("feature interface bound #{bound_id} has no owning host face");
            };
            let Some(frame) = plane_frame(host_id, entities, &index) else {
                bail!("feature host face #{host_id} is not a supported plane");
            };
            let Some(&bound_index) = index.get(&bound_id) else {
                bail!("feature interface bound #{bound_id} is missing");
            };
            let Some(bound_record) = simple_record(&entities[bound_index]) else {
                bail!("feature interface bound #{bound_id} is complex");
            };
            if bound_record.name != "FACE_BOUND" && bound_record.name != "FACE_OUTER_BOUND" {
                bail!(
                    "feature interface #{bound_id} has unsupported type {}",
                    bound_record.name
                );
            }
            let Parameter::List(bound_params) = &bound_record.parameter else {
                bail!("feature interface bound #{bound_id} parameters are malformed");
            };
            let Some(loop_id) = bound_params.get(1).and_then(entity_ref_value) else {
                bail!("feature interface bound #{bound_id} has no loop");
            };
            let Some(bound_orientation) =
                bound_params.get(2).and_then(|parameter| match parameter {
                    Parameter::Enumeration(value) if value == "T" || value == "F" => {
                        Some(value.clone())
                    }
                    _ => None,
                })
            else {
                bail!("feature interface bound #{bound_id} has invalid orientation");
            };

            let flip = family.polarity == FeaturePolarity::Additive;
            let cap_bound = push_simple(
                entities,
                &mut next_id,
                "FACE_OUTER_BOUND",
                vec![
                    Parameter::String("NONE".to_string()),
                    entity_ref(loop_id),
                    Parameter::Enumeration(if flip {
                        toggle_tf(&bound_orientation)
                    } else {
                        bound_orientation
                    }),
                ],
            );
            let cap_face = push_simple(
                entities,
                &mut next_id,
                "ADVANCED_FACE",
                vec![
                    Parameter::String("NONE".to_string()),
                    Parameter::List(vec![entity_ref(cap_bound)]),
                    entity_ref(frame.surface_id),
                    Parameter::Enumeration(if flip {
                        toggle_tf(&frame.sense)
                    } else {
                        frame.sense
                    }),
                ],
            );
            cap_faces.push(cap_face);
        }

        let mut closed_faces = member
            .face_ids
            .iter()
            .copied()
            .map(entity_ref)
            .collect::<Vec<_>>();
        closed_faces.extend(cap_faces.iter().copied().map(entity_ref));
        let tool_shell = push_simple(
            entities,
            &mut next_id,
            "CLOSED_SHELL",
            vec![
                Parameter::String("NONE".to_string()),
                Parameter::List(closed_faces),
            ],
        );
        let tool_solid = push_simple(
            entities,
            &mut next_id,
            "MANIFOLD_SOLID_BREP",
            vec![
                Parameter::String(format!(
                    "step-redox canonical {:?} boundary feature tool",
                    family.polarity
                )),
                entity_ref(tool_shell),
            ],
        );

        let Some(&representation_id) = representation_by_shell.get(&family.shell_id) else {
            bail!(
                "feature shell #{} has no source representation",
                family.shell_id
            );
        };
        let current_index = build_index(entities);
        let Some(&representation_index) = current_index.get(&representation_id) else {
            bail!("feature representation #{representation_id} is missing");
        };
        if !append_refs_to_list_param(&mut entities[representation_index], 1, &[tool_solid]) {
            bail!("failed to append feature tool to representation #{representation_id}");
        }

        materialized.push(BoundaryFeatureToolMaterialization {
            family_index,
            solid_id: tool_solid,
            shell_id: tool_shell,
            polarity: family.polarity,
            instances: family.instances,
            source_face_ids: member.face_ids.clone(),
            interface_bound_ids: member.interface_bound_ids.clone(),
        });
    }

    Ok((report, materialized))
}

/// Remove proven patterned feature boundary patches from shell topology so the
/// simpler residual can be decomposed recursively.
///
/// This is an analysis transform, not a semantics-preserving STEP rewrite:
/// additive tools and subtractive tools are recorded separately by the caller
/// and must be replayed in reverse peel order with Union/Difference. Healing
/// the host interface is intentionally identical for both polarities.
pub(crate) fn peel_patterned_boundary_features_for_analysis(
    entities: &mut Vec<EntityInstance>,
    min_instances: usize,
) -> Result<(BoundaryFeatureDiagnostics, BoundaryFeaturePeelStats)> {
    if min_instances < 2 {
        bail!("patterned feature peel requires min_instances >= 2");
    }
    let report = diagnose_boundary_features(entities);
    let selected = report
        .families
        .iter()
        .filter(|family| family.instances >= min_instances && family.lattice.is_some())
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Ok((report, BoundaryFeaturePeelStats::default()));
    }

    let index = build_index(entities);
    let mut shell_remove = HashMap::<u64, HashSet<u64>>::new();
    let mut host_remove = HashMap::<u64, HashSet<u64>>::new();
    let mut bound_owner = HashMap::<u64, u64>::new();
    let mut seen_faces = HashSet::<u64>::new();
    let mut seen_bounds = HashSet::<u64>::new();
    let mut stats = BoundaryFeaturePeelStats::default();

    for family in selected {
        for &host in &family.host_face_ids {
            let Some(bounds) = ref_list_param(host, 1, entities, &index) else {
                bail!("feature host face #{host} has no readable bounds");
            };
            for bound in bounds {
                if let Some(previous) = bound_owner.insert(bound, host) {
                    if previous != host {
                        bail!("interface bound #{bound} is owned by multiple host faces");
                    }
                }
            }
        }

        stats.families += 1;
        stats.instances += family.instances;
        match family.polarity {
            FeaturePolarity::Additive => stats.additive_instances += family.instances,
            FeaturePolarity::Subtractive => stats.subtractive_instances += family.instances,
        }

        for member in &family.members {
            for &face in &member.face_ids {
                if !seen_faces.insert(face) {
                    bail!("feature peel candidates overlap at face #{face}");
                }
                shell_remove
                    .entry(family.shell_id)
                    .or_default()
                    .insert(face);
            }
            for &bound in &member.interface_bound_ids {
                if !seen_bounds.insert(bound) {
                    bail!("feature peel candidates overlap at interface bound #{bound}");
                }
                let Some(&host) = bound_owner.get(&bound) else {
                    bail!("feature interface bound #{bound} has no selected host owner");
                };
                host_remove.entry(host).or_default().insert(bound);
            }
        }
    }

    for (shell, faces) in &shell_remove {
        let Some(&shell_index) = index.get(shell) else {
            bail!("feature shell #{shell} is missing");
        };
        if !remove_refs_from_list_param(&mut entities[shell_index], 1, faces) {
            bail!("failed to remove feature faces from shell #{shell}");
        }
    }
    for (host, bounds) in &host_remove {
        let Some(&host_index) = index.get(host) else {
            bail!("feature host face #{host} is missing");
        };
        if !remove_refs_from_list_param(&mut entities[host_index], 1, bounds) {
            bail!("failed to heal interface bounds on host face #{host}");
        }
    }

    stats.faces_removed_from_shells = seen_faces.len();
    stats.interface_bounds_healed = seen_bounds.len();
    Ok((report, stats))
}

pub fn diagnose_boundary_features(entities: &[EntityInstance]) -> BoundaryFeatureDiagnostics {
    let mut report = BoundaryFeatureDiagnostics::default();
    if entities.is_empty() {
        return report;
    }

    let index = build_index(entities);
    let contexts = collect_shell_contexts(entities, &index);
    report.shell_contexts = contexts.len();
    let mut candidates = Vec::<BoundaryFeatureCandidate>::new();

    for context in contexts {
        let Some(shell_faces) = ref_list_param(context.shell_id, 1, entities, &index) else {
            continue;
        };
        if shell_faces.len() < 3 {
            continue;
        }

        let mut face_edges = HashMap::<u64, HashSet<u64>>::new();
        let mut edge_faces = HashMap::<u64, Vec<u64>>::new();
        for &face in &shell_faces {
            let Some(edges) = face_edge_curves(face, entities, &index) else {
                continue;
            };
            for &edge in &edges {
                edge_faces.entry(edge).or_default().push(face);
            }
            face_edges.insert(face, edges);
        }

        let mut adjacency = HashMap::<u64, Vec<u64>>::new();
        for &face in &shell_faces {
            adjacency.entry(face).or_default();
        }
        for attached in edge_faces.values() {
            if attached.len() < 2 {
                continue;
            }
            for &face in attached {
                let out = adjacency.entry(face).or_default();
                out.extend(attached.iter().copied().filter(|other| *other != face));
            }
        }
        for neighbors in adjacency.values_mut() {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        let hosts = shell_faces
            .iter()
            .copied()
            .filter_map(|face| {
                let frame = plane_frame(face, entities, &index)?;
                let bounds = ref_list_param(face, 1, entities, &index)?;
                let boundary_edges = face_edges.get(&face).map_or(0, HashSet::len);
                (bounds.len() > MIN_GROUP || boundary_edges >= MIN_COMPLEX_HOST_EDGES)
                    .then_some((face, frame, bounds))
            })
            .collect::<Vec<_>>();
        report.planar_hosts += hosts.len();

        // Single-host leaves cover both protrusions and blind pockets.  The
        // topology peel is identical for the two; only the reconstruction
        // operator (union vs difference) changes.
        for (host_face, frame, bounds) in &hosts {
            let host_set = HashSet::from([*host_face]);
            let lookups =
                HashMap::from([(*host_face, host_bound_lookup(bounds, entities, &index))]);
            for component in face_components_without_hosts(&shell_faces, &host_set, &adjacency) {
                let Some(candidate) = boundary_feature_candidate(
                    context.shell_id,
                    component,
                    &host_set,
                    &HashMap::from([(*host_face, frame.clone())]),
                    &lookups,
                    &face_edges,
                    &edge_faces,
                    entities,
                    &index,
                ) else {
                    continue;
                };
                candidates.push(candidate);
            }
        }

        // Multi-interface features are graph separators, not a special
        // "two-host hole" case. Remove every currently-qualified carrier face
        // as a barrier, flood the remaining dual graph once, and classify each
        // small component by the host bounds it actually touches. This exposes
        // through-cuts, edge cuts, and other features with two or more planar
        // interfaces without combinatorial host-pair enumeration.
        if hosts.len() >= 2 {
            let host_set = hosts
                .iter()
                .map(|(face, _, _)| *face)
                .collect::<HashSet<_>>();
            let frames = hosts
                .iter()
                .map(|(face, frame, _)| (*face, frame.clone()))
                .collect::<HashMap<_, _>>();
            let lookups = hosts
                .iter()
                .map(|(face, _, bounds)| (*face, host_bound_lookup(bounds, entities, &index)))
                .collect::<HashMap<_, _>>();

            for component in face_components_without_hosts(&shell_faces, &host_set, &adjacency) {
                let Some(candidate) = boundary_feature_candidate(
                    context.shell_id,
                    component,
                    &host_set,
                    &frames,
                    &lookups,
                    &face_edges,
                    &edge_faces,
                    entities,
                    &index,
                ) else {
                    continue;
                };
                if candidate.host_face_ids.len() >= 2 {
                    candidates.push(candidate);
                }
            }
        }
    }

    // Prefer the candidate with the richer interface set when an identical
    // face patch was rediscovered from more than one host selection.
    candidates.sort_by(|a, b| {
        a.face_ids
            .cmp(&b.face_ids)
            .then_with(|| b.host_face_ids.len().cmp(&a.host_face_ids.len()))
    });
    let mut deduped = Vec::<BoundaryFeatureCandidate>::new();
    let mut seen_faces = HashSet::<Vec<u64>>::new();
    for candidate in candidates {
        if seen_faces.insert(candidate.face_ids.clone()) {
            deduped.push(candidate);
        }
    }

    report.raw_candidates = deduped.len();
    report.additive_candidates = deduped
        .iter()
        .filter(|candidate| candidate.polarity == FeaturePolarity::Additive)
        .count();
    report.subtractive_candidates = deduped
        .iter()
        .filter(|candidate| candidate.polarity == FeaturePolarity::Subtractive)
        .count();
    report.single_host_candidates = deduped
        .iter()
        .filter(|candidate| candidate.host_face_ids.len() == 1)
        .count();
    report.multi_host_candidates = deduped
        .iter()
        .filter(|candidate| candidate.host_face_ids.len() > 1)
        .count();

    let mut groups =
        HashMap::<(u64, FeaturePolarity, Vec<u64>, String), Vec<BoundaryFeatureCandidate>>::new();
    for candidate in deduped {
        groups
            .entry((
                candidate.shell_id,
                candidate.polarity,
                candidate.host_face_ids.clone(),
                candidate.signature.clone(),
            ))
            .or_default()
            .push(candidate);
    }

    let mut families = groups
        .into_iter()
        .map(
            |((shell_id, polarity, host_face_ids, signature), mut members)| {
                members.sort_by_key(|member| member.face_ids.clone());
                let centers = members
                    .iter()
                    .map(|member| member.center)
                    .collect::<Vec<_>>();
                let lattice = fit_point_lattice(&centers, 1.0e-7);
                let motif_pattern = factor_point_motif_pattern(&centers, 1.0e-7);
                BoundaryFeatureFamilyEvidence {
                    shell_id,
                    polarity,
                    host_face_ids,
                    faces_per_instance: members[0].face_ids.len(),
                    instances: members.len(),
                    signature,
                    members: members
                        .into_iter()
                        .map(|member| BoundaryFeatureInstanceEvidence {
                            face_ids: member.face_ids,
                            center_mm: member.center,
                            interface_bound_ids: member.interface_bound_ids,
                        })
                        .collect(),
                    lattice,
                    motif_pattern,
                }
            },
        )
        .collect::<Vec<_>>();
    families.sort_by(|a, b| {
        b.instances
            .cmp(&a.instances)
            .then_with(|| b.faces_per_instance.cmp(&a.faces_per_instance))
            .then_with(|| a.host_face_ids.cmp(&b.host_face_ids))
            .then_with(|| a.signature.cmp(&b.signature))
    });
    report.families = families;
    report
}

fn boundary_feature_candidate(
    shell_id: u64,
    component: HashSet<u64>,
    requested_hosts: &HashSet<u64>,
    frames: &HashMap<u64, PlaneFrame>,
    bound_lookups: &HashMap<u64, HashMap<Vec<u64>, Vec<(u64, u64, String)>>>,
    face_edges: &HashMap<u64, HashSet<u64>>,
    edge_faces: &HashMap<u64, Vec<u64>>,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<BoundaryFeatureCandidate> {
    if component.is_empty() || component.len() > MAX_BOUNDARY_FEATURE_FACES {
        return None;
    }
    if !component_is_two_manifold_with_hosts(&component, requested_hosts, face_edges, edge_faces) {
        return None;
    }

    let mut host_face_ids = Vec::new();
    let mut interface_bound_ids = Vec::new();
    let mut sides = Vec::new();
    let vertices = component_vertex_points(&component, face_edges, entities, index)?;
    for &host in requested_hosts {
        let interface_edges = component_interface_edges(&component, host, face_edges, edge_faces);
        if interface_edges.is_empty() {
            continue;
        }
        let matches = bound_lookups
            .get(&host)?
            .get(&edge_set_key(&interface_edges))?;
        if matches.len() != 1 {
            return None;
        }
        host_face_ids.push(host);
        interface_bound_ids.push(matches[0].0);
        sides.push(plane_side(&vertices, frames.get(&host)?));
    }
    if host_face_ids.is_empty() {
        return None;
    }

    let polarity = if sides.iter().all(|side| *side == PlaneSide::Positive) {
        FeaturePolarity::Additive
    } else if sides.iter().all(|side| *side == PlaneSide::Negative) {
        FeaturePolarity::Subtractive
    } else {
        return None;
    };

    host_face_ids.sort_unstable();
    interface_bound_ids.sort_unstable();
    let center = bbox_center(&vertices)?;
    let (signature, _) = component_signature(&component, center, entities, index)?;
    let mut face_ids = component.into_iter().collect::<Vec<_>>();
    face_ids.sort_unstable();

    Some(BoundaryFeatureCandidate {
        shell_id,
        polarity,
        host_face_ids,
        face_ids,
        interface_bound_ids,
        center,
        signature,
    })
}

fn face_components_without_hosts(
    faces: &[u64],
    hosts: &HashSet<u64>,
    adjacency: &HashMap<u64, Vec<u64>>,
) -> Vec<HashSet<u64>> {
    let mut remaining = faces
        .iter()
        .copied()
        .filter(|face| !hosts.contains(face))
        .collect::<HashSet<_>>();
    let mut out = Vec::new();

    while let Some(&seed) = remaining.iter().next() {
        remaining.remove(&seed);
        let mut component = HashSet::from([seed]);
        let mut queue = VecDeque::from([seed]);
        while let Some(face) = queue.pop_front() {
            for &neighbor in adjacency.get(&face).into_iter().flatten() {
                if hosts.contains(&neighbor) || !remaining.remove(&neighbor) {
                    continue;
                }
                component.insert(neighbor);
                queue.push_back(neighbor);
            }
        }
        out.push(component);
    }
    out
}

fn component_is_two_manifold_with_hosts(
    component: &HashSet<u64>,
    hosts: &HashSet<u64>,
    face_edges: &HashMap<u64, HashSet<u64>>,
    edge_faces: &HashMap<u64, Vec<u64>>,
) -> bool {
    let mut edges = HashSet::new();
    for face in component {
        let Some(face_edges) = face_edges.get(face) else {
            return false;
        };
        edges.extend(face_edges.iter().copied());
    }

    edges.into_iter().all(|edge| {
        let Some(attached) = edge_faces.get(&edge) else {
            return false;
        };
        attached.len() == 2
            && attached
                .iter()
                .all(|face| hosts.contains(face) || component.contains(face))
            && attached.iter().filter(|face| hosts.contains(face)).count() <= 1
    })
}

pub fn diagnose_planar_features(entities: &[EntityInstance]) -> PlanarFeatureDiagnostics {
    let mut report = PlanarFeatureDiagnostics::default();
    if entities.is_empty() {
        return report;
    }

    let index = build_index(entities);
    let contexts = collect_shell_contexts(entities, &index);
    report.shell_contexts = contexts.len();

    for context in contexts {
        let Some(shell_faces) = ref_list_param(context.shell_id, 1, entities, &index) else {
            continue;
        };
        if shell_faces.len() < MIN_GROUP + 2 {
            continue;
        }

        let mut face_edges = HashMap::<u64, HashSet<u64>>::new();
        let mut edge_faces = HashMap::<u64, Vec<u64>>::new();
        for &face in &shell_faces {
            let Some(edges) = face_edge_curves(face, entities, &index) else {
                continue;
            };
            for &edge in &edges {
                edge_faces.entry(edge).or_default().push(face);
            }
            face_edges.insert(face, edges);
        }

        let mut adjacency = HashMap::<u64, Vec<u64>>::new();
        for &face in &shell_faces {
            adjacency.entry(face).or_default();
        }
        for attached in edge_faces.values() {
            if attached.len() < 2 {
                continue;
            }
            for &face in attached {
                adjacency
                    .entry(face)
                    .or_default()
                    .extend(attached.iter().copied().filter(|other| *other != face));
            }
        }
        for neighbors in adjacency.values_mut() {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        let host_faces = shell_faces
            .iter()
            .copied()
            .filter(|face| {
                face_surface(*face, entities, &index)
                    .and_then(|surface| index.get(&surface).copied())
                    .and_then(|idx| simple_record(&entities[idx]))
                    .is_some_and(|record| record.name == "PLANE")
                    && ref_list_param(*face, 1, entities, &index)
                        .is_some_and(|bounds| bounds.len() > MIN_GROUP)
            })
            .collect::<Vec<_>>();

        report.qualifying_host_faces += host_faces.len();
        for host_face in host_faces {
            let Some(frame) = plane_frame(host_face, entities, &index) else {
                continue;
            };
            let Some(host_bounds) = ref_list_param(host_face, 1, entities, &index) else {
                continue;
            };
            let bound_lookup = host_bound_lookup(&host_bounds, entities, &index);
            let components = face_components_without_host(&shell_faces, host_face, &adjacency);
            let mut host = PlanarHostDiagnostic {
                host_face_id: host_face,
                bound_count: host_bounds.len(),
                outward: frame.outward,
                component_count: components.len(),
                ..Default::default()
            };
            report.total_components += components.len();

            let mut positive_groups = HashMap::<String, usize>::new();
            let mut negative_groups = HashMap::<String, usize>::new();

            for component in components {
                if component.is_empty() || component.len() > MAX_FEATURE_FACES {
                    report.rejected_empty_or_large += 1;
                    continue;
                }
                if !component_is_two_manifold_with_host(
                    &component,
                    host_face,
                    &face_edges,
                    &edge_faces,
                ) {
                    report.rejected_non_manifold += 1;
                    continue;
                }
                let interface_edges =
                    component_interface_edges(&component, host_face, &face_edges, &edge_faces);
                if interface_edges.is_empty() {
                    report.rejected_interface += 1;
                    continue;
                }
                let Some(matches) = bound_lookup.get(&edge_set_key(&interface_edges)) else {
                    report.rejected_bound_match += 1;
                    continue;
                };
                if matches.len() != 1 {
                    report.rejected_bound_match += 1;
                    continue;
                }

                let Some(vertices) =
                    component_vertex_points(&component, &face_edges, entities, &index)
                else {
                    report.rejected_vertices += 1;
                    continue;
                };
                let Some(center) = bbox_center(&vertices) else {
                    report.rejected_vertices += 1;
                    continue;
                };
                let Some((signature, _)) =
                    component_signature(&component, center, entities, &index)
                else {
                    report.rejected_signature += 1;
                    continue;
                };

                match plane_side(&vertices, &frame) {
                    PlaneSide::Positive => {
                        report.positive_components += 1;
                        host.positive_components += 1;
                        *positive_groups.entry(signature).or_default() += 1;
                    }
                    PlaneSide::Negative => {
                        report.negative_components += 1;
                        host.negative_components += 1;
                        *negative_groups.entry(signature).or_default() += 1;
                    }
                    PlaneSide::Straddling => {
                        report.straddling_components += 1;
                        host.straddling_components += 1;
                    }
                    PlaneSide::Coplanar => {
                        report.coplanar_components += 1;
                        host.coplanar_components += 1;
                    }
                }
            }

            host.largest_positive_signature_group =
                positive_groups.values().copied().max().unwrap_or(0);
            host.largest_negative_signature_group =
                negative_groups.values().copied().max().unwrap_or(0);
            report.hosts.push(host);
        }
    }

    report
}

pub fn instance_planar_positive_features(entities: &mut Vec<EntityInstance>) -> PlanarFeatureStats {
    let mut stats = PlanarFeatureStats::default();
    if entities.is_empty() {
        return stats;
    }

    let index = build_index(entities);
    let styles_by_target = collect_styles_by_target(entities);
    let contexts = collect_shell_contexts(entities, &index);
    if contexts.is_empty() {
        return stats;
    }

    let mut next_id = entities.iter().map(entity_id).max().unwrap_or(0) + 1;
    let mut candidate_roots = HashSet::new();
    let mut delete_seed = HashSet::new();
    let mut claimed_faces = HashSet::new();
    let mut claimed_bounds = HashSet::new();

    for context in contexts {
        let Some(shell_faces) = ref_list_param(context.shell_id, 1, entities, &index) else {
            continue;
        };
        if shell_faces.len() < MIN_GROUP + 2 {
            continue;
        }

        let mut face_edges = HashMap::<u64, HashSet<u64>>::new();
        let mut edge_faces = HashMap::<u64, Vec<u64>>::new();
        for &face in &shell_faces {
            let Some(edges) = face_edge_curves(face, entities, &index) else {
                continue;
            };
            for &edge in &edges {
                edge_faces.entry(edge).or_default().push(face);
            }
            face_edges.insert(face, edges);
        }

        let mut adjacency = HashMap::<u64, Vec<u64>>::new();
        for &face in &shell_faces {
            adjacency.entry(face).or_default();
        }
        for attached in edge_faces.values() {
            if attached.len() < 2 {
                continue;
            }
            for &face in attached {
                let out = adjacency.entry(face).or_default();
                out.extend(attached.iter().copied().filter(|other| *other != face));
            }
        }
        for neighbors in adjacency.values_mut() {
            neighbors.sort_unstable();
            neighbors.dedup();
        }

        let container_styles = styles_by_target
            .get(&context.container_id)
            .cloned()
            .unwrap_or_default();

        let host_faces: Vec<u64> = shell_faces
            .iter()
            .copied()
            .filter(|face| {
                face_surface(*face, entities, &index)
                    .and_then(|surface| index.get(&surface).copied())
                    .and_then(|idx| simple_record(&entities[idx]))
                    .is_some_and(|record| record.name == "PLANE")
                    && ref_list_param(*face, 1, entities, &index)
                        .is_some_and(|bounds| bounds.len() > MIN_GROUP)
            })
            .collect();

        for host_face in host_faces {
            let Some(frame) = plane_frame(host_face, entities, &index) else {
                continue;
            };
            let Some(host_bounds) = ref_list_param(host_face, 1, entities, &index) else {
                continue;
            };
            let bound_lookup = host_bound_lookup(&host_bounds, entities, &index);
            let components = face_components_without_host(&shell_faces, host_face, &adjacency);
            let mut features = Vec::new();

            for component in components {
                if component.is_empty()
                    || component.len() > MAX_FEATURE_FACES
                    || component.iter().any(|face| claimed_faces.contains(face))
                {
                    continue;
                }

                if !component_is_two_manifold_with_host(
                    &component,
                    host_face,
                    &face_edges,
                    &edge_faces,
                ) {
                    continue;
                }
                let interface_edges =
                    component_interface_edges(&component, host_face, &face_edges, &edge_faces);
                if interface_edges.is_empty() {
                    continue;
                }
                let Some(matches) = bound_lookup.get(&edge_set_key(&interface_edges)) else {
                    continue;
                };
                if matches.len() != 1 {
                    continue;
                }
                let (interface_bound, interface_loop, bound_orientation) = matches[0].clone();
                if claimed_bounds.contains(&interface_bound) {
                    continue;
                }

                let Some(vertices) =
                    component_vertex_points(&component, &face_edges, entities, &index)
                else {
                    continue;
                };
                if !positive_side(&vertices, &frame) {
                    continue;
                }
                let Some(center) = bbox_center(&vertices) else {
                    continue;
                };
                let Some((signature, normalized_quarter)) =
                    component_signature(&component, center, entities, &index)
                else {
                    continue;
                };
                let Some((style_key, old_style_ids)) =
                    feature_style(&component, &styles_by_target, &container_styles)
                else {
                    continue;
                };

                let mut faces: Vec<u64> = component.into_iter().collect();
                faces.sort_unstable();
                features.push(Feature {
                    faces,
                    interface_bound,
                    interface_loop,
                    bound_orientation,
                    center,
                    normalized_quarter,
                    signature,
                    style_key,
                    old_style_ids,
                });
            }

            if features.len() < MIN_GROUP {
                continue;
            }

            let mut groups = HashMap::<(String, String, StyleKey), Vec<Feature>>::new();
            for feature in features {
                groups
                    .entry((
                        feature.signature.clone(),
                        feature.bound_orientation.clone(),
                        feature.style_key.clone(),
                    ))
                    .or_default()
                    .push(feature);
            }

            let mut accepted: Vec<Vec<Feature>> = groups
                .into_values()
                .filter(|group| group.len() >= MIN_GROUP)
                .collect();
            if accepted.is_empty() {
                continue;
            }
            accepted.sort_by_key(|group| {
                group
                    .iter()
                    .map(|feature| feature.faces[0])
                    .min()
                    .unwrap_or(u64::MAX)
            });

            let Some(&rep_idx) = index.get(&context.representation_id) else {
                continue;
            };

            let z_dir = push_direction(entities, &mut next_id, [0.0, 0.0, 1.0]);
            let x_dirs = [
                push_direction(entities, &mut next_id, [1.0, 0.0, 0.0]),
                push_direction(entities, &mut next_id, [0.0, 1.0, 0.0]),
                push_direction(entities, &mut next_id, [-1.0, 0.0, 0.0]),
                push_direction(entities, &mut next_id, [0.0, -1.0, 0.0]),
            ];
            let origin_point = push_point(entities, &mut next_id, [0.0, 0.0, 0.0]);
            let origin_axis = push_simple(
                entities,
                &mut next_id,
                "AXIS2_PLACEMENT_3D",
                vec![
                    Parameter::String(String::new()),
                    entity_ref(origin_point),
                    entity_ref(z_dir),
                    entity_ref(x_dirs[0]),
                ],
            );

            let mut host_mapped_ids = Vec::new();
            let mut host_remove_faces = HashSet::new();
            let mut host_remove_bounds = HashSet::new();
            let mut host_family_count = 0usize;

            for group in &mut accepted {
                group.sort_by_key(|feature| feature.faces.clone());
                let canonical = group[0].clone();

                let disk_bound = push_simple(
                    entities,
                    &mut next_id,
                    "FACE_OUTER_BOUND",
                    vec![
                        Parameter::String("NONE".to_string()),
                        entity_ref(canonical.interface_loop),
                        Parameter::Enumeration(toggle_tf(&canonical.bound_orientation)),
                    ],
                );
                let disk_face = push_simple(
                    entities,
                    &mut next_id,
                    "ADVANCED_FACE",
                    vec![
                        Parameter::String("NONE".to_string()),
                        Parameter::List(vec![entity_ref(disk_bound)]),
                        entity_ref(frame.surface_id),
                        Parameter::Enumeration(toggle_tf(&frame.sense)),
                    ],
                );

                let mut closed_faces: Vec<Parameter> =
                    canonical.faces.iter().copied().map(entity_ref).collect();
                closed_faces.push(entity_ref(disk_face));
                let feature_shell = push_simple(
                    entities,
                    &mut next_id,
                    "CLOSED_SHELL",
                    vec![
                        Parameter::String("NONE".to_string()),
                        Parameter::List(closed_faces),
                    ],
                );
                let feature_solid = push_simple(
                    entities,
                    &mut next_id,
                    "MANIFOLD_SOLID_BREP",
                    vec![
                        Parameter::String(
                            "step-redox canonical planar positive feature".to_string(),
                        ),
                        entity_ref(feature_shell),
                    ],
                );
                let source_rep = push_simple(
                    entities,
                    &mut next_id,
                    "ADVANCED_BREP_SHAPE_REPRESENTATION",
                    vec![
                        Parameter::String("step-redox planar positive feature source".to_string()),
                        Parameter::List(vec![entity_ref(feature_solid), entity_ref(origin_axis)]),
                        entity_ref(context.context_id),
                    ],
                );
                let rep_map = push_simple(
                    entities,
                    &mut next_id,
                    "REPRESENTATION_MAP",
                    vec![entity_ref(origin_axis), entity_ref(source_rep)],
                );

                let mut new_style_ids = Vec::new();
                let mut old_style_ids = HashSet::new();

                for feature in group.iter() {
                    let quarter =
                        (canonical.normalized_quarter + 4 - feature.normalized_quarter) % 4;
                    let translation = rigid_translation(canonical.center, feature.center, quarter);
                    let target_axis =
                        if quarter == 0 && translation.iter().all(|value| value.abs() <= 1.0e-12) {
                            origin_axis
                        } else {
                            let point = push_point(entities, &mut next_id, translation);
                            push_simple(
                                entities,
                                &mut next_id,
                                "AXIS2_PLACEMENT_3D",
                                vec![
                                    Parameter::String(String::new()),
                                    entity_ref(point),
                                    entity_ref(z_dir),
                                    entity_ref(x_dirs[quarter as usize]),
                                ],
                            )
                        };
                    let mapped = push_simple(
                        entities,
                        &mut next_id,
                        "MAPPED_ITEM",
                        vec![
                            Parameter::String(String::new()),
                            entity_ref(rep_map),
                            entity_ref(target_axis),
                        ],
                    );
                    host_mapped_ids.push(mapped);

                    match &feature.style_key {
                        StyleKey::Face(assignments) => {
                            let styled = push_simple(
                                entities,
                                &mut next_id,
                                "STYLED_ITEM",
                                vec![
                                    Parameter::String("NONE".to_string()),
                                    Parameter::List(
                                        assignments.iter().copied().map(entity_ref).collect(),
                                    ),
                                    entity_ref(mapped),
                                ],
                            );
                            new_style_ids.push(styled);
                            old_style_ids.extend(feature.old_style_ids.iter().copied());
                        }
                        StyleKey::Container => {
                            for style in &container_styles {
                                let styled = push_simple(
                                    entities,
                                    &mut next_id,
                                    "STYLED_ITEM",
                                    vec![
                                        Parameter::String("NONE".to_string()),
                                        Parameter::List(
                                            style
                                                .assignments
                                                .iter()
                                                .copied()
                                                .map(entity_ref)
                                                .collect(),
                                        ),
                                        entity_ref(mapped),
                                    ],
                                );
                                new_style_ids.push(styled);
                            }
                        }
                        StyleKey::None => {}
                    }

                    for &face in &feature.faces {
                        host_remove_faces.insert(face);
                        candidate_roots.insert(face);
                    }
                    host_remove_bounds.insert(feature.interface_bound);
                    candidate_roots.insert(feature.interface_bound);
                    delete_seed.insert(feature.interface_bound);
                    claimed_faces.extend(feature.faces.iter().copied());
                    claimed_bounds.insert(feature.interface_bound);
                }

                for feature in group.iter().skip(1) {
                    delete_seed.extend(feature.faces.iter().copied());
                }

                if !old_style_ids.is_empty() {
                    for &style in &old_style_ids {
                        candidate_roots.insert(style);
                        delete_seed.insert(style);
                    }
                    patch_presentation_lists(entities, &old_style_ids, &new_style_ids);
                } else if matches!(canonical.style_key, StyleKey::Container)
                    && !new_style_ids.is_empty()
                {
                    let anchors: HashSet<u64> =
                        container_styles.iter().map(|style| style.id).collect();
                    append_presentation_items_with_anchors(entities, &anchors, &new_style_ids);
                }

                host_family_count += 1;
                stats.instances += group.len();
            }

            if host_remove_faces.is_empty() {
                continue;
            }

            let Some(&shell_idx) = index.get(&context.shell_id) else {
                continue;
            };
            let Some(&host_idx) = index.get(&host_face) else {
                continue;
            };
            if !remove_refs_from_list_param(&mut entities[shell_idx], 1, &host_remove_faces) {
                continue;
            }
            if !remove_refs_from_list_param(&mut entities[host_idx], 1, &host_remove_bounds) {
                continue;
            }
            if !append_refs_to_list_param(&mut entities[rep_idx], 1, &host_mapped_ids) {
                continue;
            }

            stats.arrays += 1;
            stats.families += host_family_count;
        }
    }

    if stats.arrays == 0 {
        return stats;
    }

    let index_after = build_index(entities);
    let mut candidate = HashSet::new();
    let mut stack: Vec<u64> = candidate_roots.iter().copied().collect();
    while let Some(id) = stack.pop() {
        if !candidate.insert(id) {
            continue;
        }
        let Some(&idx) = index_after.get(&id) else {
            continue;
        };
        visit_entity_refs(&entities[idx], &mut |child| {
            if index_after.contains_key(&child) && !candidate.contains(&child) {
                stack.push(child);
            }
        });
    }

    let references = ReferenceGraph::new(entities);
    let inbound = references.inbound();
    let mut delete = delete_seed;

    loop {
        let mut changed = false;
        for &id in &candidate {
            if delete.contains(&id) {
                continue;
            }
            let parents = inbound.get(&id);
            let all_dead =
                parents.is_none_or(|parents| parents.iter().all(|parent| delete.contains(parent)));
            let child_of_dead = parents.is_some_and(|parents| {
                !parents.is_empty() && parents.iter().all(|parent| delete.contains(parent))
            });
            if all_dead && (child_of_dead || !inbound.contains_key(&id)) {
                delete.insert(id);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    for (id, children) in references.forward() {
        if delete.contains(id) {
            continue;
        }
        debug_assert!(children.iter().all(|child| !delete.contains(child)));
    }

    stats.styles_replaced = candidate_roots
        .iter()
        .filter(|id| {
            index_after
                .get(id)
                .and_then(|idx| simple_record(&entities[*idx]))
                .is_some_and(|record| record.name == "STYLED_ITEM")
                && delete.contains(id)
        })
        .count();
    stats.entities_removed = delete.len();
    entities.retain(|entity| !delete.contains(&entity_id(entity)));
    stats
}

fn collect_shell_contexts(
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Vec<ShellContext> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();

    for entity in entities {
        let Some(record) = simple_record(entity) else {
            continue;
        };
        if record.name != "ADVANCED_BREP_SHAPE_REPRESENTATION"
            && record.name != "MANIFOLD_SURFACE_SHAPE_REPRESENTATION"
        {
            continue;
        }
        let representation_id = entity_id(entity);
        let Some((items, context_id)) = representation_items_and_context(entity) else {
            continue;
        };

        for item in items {
            let Some(&item_idx) = index.get(&item) else {
                continue;
            };
            let Some(item_record) = simple_record(&entities[item_idx]) else {
                continue;
            };
            match item_record.name.as_str() {
                "MANIFOLD_SOLID_BREP" => {
                    let Some(shell_id) = manifold_shell(item, entities, index) else {
                        continue;
                    };
                    if is_closed_shell(shell_id, entities, index)
                        && seen.insert((representation_id, item, shell_id))
                    {
                        out.push(ShellContext {
                            representation_id,
                            context_id,
                            container_id: item,
                            shell_id,
                        });
                    }
                }
                "SHELL_BASED_SURFACE_MODEL" => {
                    let Some(shells) = ref_list_param(item, 1, entities, index) else {
                        continue;
                    };
                    for shell_id in shells {
                        if is_closed_shell(shell_id, entities, index)
                            && seen.insert((representation_id, item, shell_id))
                        {
                            out.push(ShellContext {
                                representation_id,
                                context_id,
                                container_id: item,
                                shell_id,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
    }
    out
}

fn is_closed_shell(
    shell_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> bool {
    index
        .get(&shell_id)
        .and_then(|idx| simple_record(&entities[*idx]))
        .is_some_and(|record| record.name == "CLOSED_SHELL")
}

fn host_bound_lookup(
    bounds: &[u64],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> HashMap<Vec<u64>, Vec<(u64, u64, String)>> {
    let mut out = HashMap::<Vec<u64>, Vec<(u64, u64, String)>>::new();
    for &bound in bounds {
        let Some(record) = index
            .get(&bound)
            .and_then(|idx| simple_record(&entities[*idx]))
        else {
            continue;
        };
        if record.name != "FACE_BOUND" {
            continue;
        }
        let Some((loop_id, edges, orientation)) = bound_loop_edges(bound, entities, index) else {
            continue;
        };
        out.entry(edge_set_key(&edges))
            .or_default()
            .push((bound, loop_id, orientation));
    }
    out
}

fn edge_set_key(edges: &HashSet<u64>) -> Vec<u64> {
    let mut key: Vec<u64> = edges.iter().copied().collect();
    key.sort_unstable();
    key
}

fn face_components_without_host(
    faces: &[u64],
    host: u64,
    adjacency: &HashMap<u64, Vec<u64>>,
) -> Vec<HashSet<u64>> {
    let mut remaining: HashSet<u64> = faces.iter().copied().filter(|face| *face != host).collect();
    let mut out = Vec::new();

    while let Some(&seed) = remaining.iter().next() {
        remaining.remove(&seed);
        let mut component = HashSet::from([seed]);
        let mut queue = VecDeque::from([seed]);
        while let Some(face) = queue.pop_front() {
            for &neighbor in adjacency.get(&face).into_iter().flatten() {
                if neighbor == host || !remaining.remove(&neighbor) {
                    continue;
                }
                component.insert(neighbor);
                queue.push_back(neighbor);
            }
        }
        out.push(component);
    }
    out
}

fn component_is_two_manifold_with_host(
    component: &HashSet<u64>,
    host: u64,
    face_edges: &HashMap<u64, HashSet<u64>>,
    edge_faces: &HashMap<u64, Vec<u64>>,
) -> bool {
    let mut edges = HashSet::new();
    for face in component {
        let Some(face_edges) = face_edges.get(face) else {
            return false;
        };
        edges.extend(face_edges.iter().copied());
    }

    edges.into_iter().all(|edge| {
        let Some(attached) = edge_faces.get(&edge) else {
            return false;
        };
        if attached.len() != 2 {
            return false;
        }
        attached
            .iter()
            .all(|face| *face == host || component.contains(face))
            && attached.iter().filter(|face| **face == host).count() <= 1
    })
}

fn component_interface_edges(
    component: &HashSet<u64>,
    host: u64,
    face_edges: &HashMap<u64, HashSet<u64>>,
    edge_faces: &HashMap<u64, Vec<u64>>,
) -> HashSet<u64> {
    let mut out = HashSet::new();
    for face in component {
        for edge in face_edges.get(face).into_iter().flatten() {
            if edge_faces
                .get(edge)
                .is_some_and(|faces| faces.contains(&host))
            {
                out.insert(*edge);
            }
        }
    }
    out
}

fn component_vertex_points(
    component: &HashSet<u64>,
    face_edges: &HashMap<u64, HashSet<u64>>,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<[f64; 3]>> {
    let mut edge_ids = HashSet::new();
    for face in component {
        edge_ids.extend(face_edges.get(face)?.iter().copied());
    }

    let mut vertex_ids = HashSet::new();
    for edge in edge_ids {
        let record = simple_record(&entities[*index.get(&edge)?])?;
        if record.name != "EDGE_CURVE" {
            return None;
        }
        let Parameter::List(params) = &record.parameter else {
            return None;
        };
        vertex_ids.insert(entity_ref_value(params.get(1)?)?);
        vertex_ids.insert(entity_ref_value(params.get(2)?)?);
    }

    let mut out = Vec::with_capacity(vertex_ids.len());
    for vertex in vertex_ids {
        let record = simple_record(&entities[*index.get(&vertex)?])?;
        if record.name != "VERTEX_POINT" {
            return None;
        }
        let Parameter::List(params) = &record.parameter else {
            return None;
        };
        let point_id = entity_ref_value(params.get(1)?)?;
        out.push(cartesian_point(point_id, entities, index)?);
    }
    Some(out)
}

fn bbox_center(points: &[[f64; 3]]) -> Option<[f64; 3]> {
    let first = *points.first()?;
    let mut min = first;
    let mut max = first;
    for point in points.iter().skip(1) {
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    Some([
        f64::midpoint(min[0], max[0]),
        f64::midpoint(min[1], max[1]),
        f64::midpoint(min[2], max[2]),
    ])
}

fn plane_frame(
    host_face: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<PlaneFrame> {
    let surface_id = face_surface(host_face, entities, index)?;
    let surface = simple_record(&entities[*index.get(&surface_id)?])?;
    if surface.name != "PLANE" {
        return None;
    }
    let Parameter::List(surface_params) = &surface.parameter else {
        return None;
    };
    let axis_id = entity_ref_value(surface_params.get(1)?)?;
    let axis = simple_record(&entities[*index.get(&axis_id)?])?;
    if axis.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let Parameter::List(axis_params) = &axis.parameter else {
        return None;
    };
    let origin_id = entity_ref_value(axis_params.get(1)?)?;
    let direction_id = entity_ref_value(axis_params.get(2)?)?;
    let origin = cartesian_point(origin_id, entities, index)?;
    let direction = direction_components(direction_id, entities, index)?;
    let length = f64::mul_add(
        direction[2],
        direction[2],
        f64::mul_add(direction[1], direction[1], direction[0] * direction[0]),
    )
    .sqrt();
    if !length.is_finite() || length <= 0.0 {
        return None;
    }
    let sense = face_sense(host_face, entities, index)?;
    let sign = match sense.as_str() {
        "T" => 1.0,
        "F" => -1.0,
        _ => return None,
    };
    Some(PlaneFrame {
        surface_id,
        sense,
        origin,
        outward: [
            sign * direction[0] / length,
            sign * direction[1] / length,
            sign * direction[2] / length,
        ],
    })
}

fn direction_components(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let record = simple_record(&entities[*index.get(&id)?])?;
    if record.name != "DIRECTION" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let Parameter::List(coords) = params.get(1)? else {
        return None;
    };
    if coords.len() != 3 {
        return None;
    }
    Some([
        numeric(&coords[0])?,
        numeric(&coords[1])?,
        numeric(&coords[2])?,
    ])
}

fn numeric(parameter: &Parameter) -> Option<f64> {
    match parameter {
        Parameter::Real(value) => Some(*value),
        Parameter::Integer(value) => crate::numeric::exact_i64_to_f64(*value),
        _ => None,
    }
}

fn plane_side(points: &[[f64; 3]], frame: &PlaneFrame) -> PlaneSide {
    let mut min_projection = f64::INFINITY;
    let mut max_projection = f64::NEG_INFINITY;
    for point in points {
        let delta = [
            point[0] - frame.origin[0],
            point[1] - frame.origin[1],
            point[2] - frame.origin[2],
        ];
        let projection = delta[2].mul_add(
            frame.outward[2],
            delta[1].mul_add(frame.outward[1], delta[0] * frame.outward[0]),
        );
        if !projection.is_finite() {
            return PlaneSide::Straddling;
        }
        min_projection = min_projection.min(projection);
        max_projection = max_projection.max(projection);
    }

    if min_projection >= -SIDE_TOLERANCE && max_projection > SIDE_TOLERANCE {
        PlaneSide::Positive
    } else if max_projection <= SIDE_TOLERANCE && min_projection < -SIDE_TOLERANCE {
        PlaneSide::Negative
    } else if min_projection.abs() <= SIDE_TOLERANCE && max_projection.abs() <= SIDE_TOLERANCE {
        PlaneSide::Coplanar
    } else {
        PlaneSide::Straddling
    }
}

fn positive_side(points: &[[f64; 3]], frame: &PlaneFrame) -> bool {
    plane_side(points, frame) == PlaneSide::Positive
}

fn component_signature(
    component: &HashSet<u64>,
    center: [f64; 3],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<(String, u8)> {
    let mut best: Option<(String, u8)> = None;
    for quarter in 0..4u8 {
        let mut faces = Vec::with_capacity(component.len());
        for &face in component {
            faces.push(face_topology_signature(
                face, entities, index, center, quarter,
            )?);
        }
        faces.sort();
        let signature = format!("F{}[{}]", component.len(), faces.join("|"));
        if best
            .as_ref()
            .is_none_or(|(current, _)| signature < *current)
        {
            best = Some((signature, quarter));
        }
    }
    best
}

fn feature_style(
    component: &HashSet<u64>,
    styles_by_target: &HashMap<u64, Vec<StyleRef>>,
    container_styles: &[StyleRef],
) -> Option<(StyleKey, Vec<u64>)> {
    let mut any_face_style = false;
    let mut assignments: Option<Vec<u64>> = None;
    let mut style_ids = Vec::new();

    for face in component {
        match styles_by_target.get(face) {
            None => {}
            Some(styles) if styles.is_empty() => {}
            Some(styles) if styles.len() == 1 && !styles[0].assignments.is_empty() => {
                any_face_style = true;
                if let Some(expected) = &assignments {
                    if *expected != styles[0].assignments {
                        return None;
                    }
                } else {
                    assignments = Some(styles[0].assignments.clone());
                }
                style_ids.push(styles[0].id);
            }
            Some(_) => return None,
        }
    }

    if any_face_style {
        if style_ids.len() != component.len() {
            return None;
        }
        Some((StyleKey::Face(assignments?), style_ids))
    } else if !container_styles.is_empty() {
        Some((StyleKey::Container, Vec::new()))
    } else {
        Some((StyleKey::None, Vec::new()))
    }
}

fn push_direction(entities: &mut Vec<EntityInstance>, next_id: &mut u64, value: [f64; 3]) -> u64 {
    push_simple(
        entities,
        next_id,
        "DIRECTION",
        vec![
            Parameter::String(String::new()),
            Parameter::List(value.into_iter().map(Parameter::Real).collect()),
        ],
    )
}

fn rigid_translation(source_center: [f64; 3], target_center: [f64; 3], quarter: u8) -> [f64; 3] {
    let (x, y) = rotate_xy(source_center[0], source_center[1], quarter);
    [
        target_center[0] - x,
        target_center[1] - y,
        target_center[2] - source_center[2],
    ]
}

fn rotate_xy(x: f64, y: f64, quarter: u8) -> (f64, f64) {
    match quarter % 4 {
        0 => (x, y),
        1 => (-y, x),
        2 => (-x, -y),
        3 => (y, -x),
        _ => unreachable!(),
    }
}

fn append_presentation_items_with_anchors(
    entities: &mut [EntityInstance],
    anchors: &HashSet<u64>,
    add: &[u64],
) {
    for entity in entities {
        let Some(record) = simple_record_mut(entity) else {
            continue;
        };
        let target_index = match record.name.as_str() {
            "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION" => 1,
            "PRESENTATION_LAYER_ASSIGNMENT" => 2,
            _ => continue,
        };
        let Parameter::List(params) = &mut record.parameter else {
            continue;
        };
        let Some(Parameter::List(items)) = params.get_mut(target_index) else {
            continue;
        };
        let has_anchor = items
            .iter()
            .filter_map(entity_ref_value)
            .any(|id| anchors.contains(&id));
        if !has_anchor {
            continue;
        }
        let existing: HashSet<u64> = items.iter().filter_map(entity_ref_value).collect();
        items.extend(
            add.iter()
                .copied()
                .filter(|id| !existing.contains(id))
                .map(entity_ref),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarter_turn_translation_maps_center() {
        let source = [2.0, 3.0, 4.0];
        let target = [10.0, -7.0, 8.0];
        for quarter in 0..4 {
            let translation = rigid_translation(source, target, quarter);
            let (x, y) = rotate_xy(source[0], source[1], quarter);
            assert!((x + translation[0] - target[0]).abs() < 1.0e-12);
            assert!((y + translation[1] - target[1]).abs() < 1.0e-12);
            assert!((source[2] + translation[2] - target[2]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn plane_side_classifies_protrusions_and_recesses() {
        let frame = PlaneFrame {
            surface_id: 1,
            sense: "T".to_string(),
            origin: [0.0, 0.0, 0.0],
            outward: [0.0, 0.0, 1.0],
        };
        let protrusion = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]];
        let recess = [[0.0, 0.0, 0.0], [0.0, 0.0, -2.0]];
        assert_eq!(plane_side(&protrusion, &frame), PlaneSide::Positive);
        assert_eq!(plane_side(&recess, &frame), PlaneSide::Negative);
        assert!(positive_side(&protrusion, &frame));
        assert!(!positive_side(&recess, &frame));
    }

    #[test]
    fn feature_edges_must_be_exactly_two_manifold_with_host() -> anyhow::Result<()> {
        let component = HashSet::from([1_u64, 2_u64]);
        let face_edges = HashMap::from([
            (1_u64, HashSet::from([10_u64, 11_u64])),
            (2_u64, HashSet::from([10_u64, 12_u64])),
        ]);
        let mut edge_faces = HashMap::from([
            (10_u64, vec![1_u64, 2_u64]),
            (11_u64, vec![1_u64, 9_u64]),
            (12_u64, vec![2_u64, 9_u64]),
        ]);

        assert!(component_is_two_manifold_with_host(
            &component,
            9,
            &face_edges,
            &edge_faces,
        ));

        edge_faces
            .get_mut(&11)
            .ok_or_else(|| anyhow::anyhow!("expected test value"))?
            .push(99);
        assert!(!component_is_two_manifold_with_host(
            &component,
            9,
            &face_edges,
            &edge_faces,
        ));
        Ok(())
    }
}
