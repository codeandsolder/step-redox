use crate::math3::{
    add, canonical_direction, cross, dot, norm, normalize as normalize3, scale, sub,
};
use crate::numeric::{
    exact_i64_to_f64, exact_usize_to_f64, floored_f64_to_i64, rounded_f64_to_i64,
};
use crate::step_entities::{
    entity_ref, entity_ref_list, numeric_list, push_simple,
    styled_items_by_target as style_records_by_target,
};
use crate::step_graph::{
    ReferenceGraph, build_index, entity_id, entity_ref_value, simple_record, simple_record_mut,
};
use anyhow::{Result, bail};
use ruststep::ast::{EntityInstance, Parameter};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

const ORIENTATION_Q: f64 = 1.0e-10;

fn quantize_vec3(vector: [f64; 3], quantum: f64) -> Option<[i64; 3]> {
    Some([
        rounded_f64_to_i64(vector[0] / quantum)?,
        rounded_f64_to_i64(vector[1] / quantum)?,
        rounded_f64_to_i64(vector[2] / quantum)?,
    ])
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InstancePattern {
    pub parent_representation: u64,
    pub representation_map: u64,
    pub item_ids: Vec<u64>,
    pub style_signature: Vec<Vec<u64>>,
    pub dimension: u8,
    pub origin: [f64; 3],
    pub basis: Vec<[f64; 3]>,
    pub pitch: Vec<f64>,
    /// Integer lattice sites. For 1-D patterns the second coordinate is zero.
    pub occupancy: Vec<[i64; 2]>,
    pub grid_shape: Vec<usize>,
    pub fill_ratio: f64,
    #[serde(default = "default_pattern_tolerance_mm")]
    pub tolerance_mm: f64,
    pub max_residual_mm: f64,
}

const fn default_pattern_tolerance_mm() -> f64 {
    1.0e-7
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PatternAnchor {
    Start,
    Center,
    End,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PatternResizeStats {
    pub old_count: usize,
    pub new_count: usize,
    pub reused_items: usize,
    pub added_items: usize,
    pub removed_items: usize,
    pub entities_removed: usize,
}

#[derive(Debug, Clone)]
struct MappedPlacement {
    item: u64,
    origin: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GroupKey {
    parent_representation: u64,
    representation_map: u64,
    orientation: [i64; 9],
    style_signature: Vec<Vec<u64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PointLattice {
    pub dimension: u8,
    pub origin: [f64; 3],
    pub basis: Vec<[f64; 3]>,
    pub pitch: Vec<f64>,
    /// Integer lattice sites. For 1-D fits the second coordinate is zero.
    pub occupancy: Vec<[i64; 2]>,
    pub grid_shape: Vec<usize>,
    pub fill_ratio: f64,
    pub max_residual_mm: f64,
}

#[must_use]
pub fn fit_point_lattice(points: &[[f64; 3]], tolerance_mm: f64) -> Option<PointLattice> {
    if points.len() < 2 || !tolerance_mm.is_finite() || tolerance_mm <= 0.0 {
        return None;
    }
    let fit = fit_lattice(points, tolerance_mm)?;
    Some(PointLattice {
        dimension: fit.dimension,
        origin: fit.origin,
        basis: fit.basis,
        pitch: fit.pitch,
        occupancy: fit.occupancy,
        grid_shape: fit.grid_shape,
        fill_ratio: fit.fill_ratio,
        max_residual_mm: fit.max_residual_mm,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PointMotifPattern {
    pub dimension: u8,
    pub origin: [f64; 3],
    pub repeat_basis: Vec<[f64; 3]>,
    pub repeat_pitch: Vec<f64>,
    pub grid_shape: Vec<usize>,
    pub motif_offsets: Vec<[f64; 3]>,
    pub repeat_count: usize,
    pub max_residual_mm: f64,
}

#[derive(Clone)]
struct MotifPoint {
    point: [f64; 3],
    integer: [i64; 2],
}

struct MotifProjection {
    groups: BTreeMap<[i64; 2], Vec<MotifPoint>>,
    max_residual_mm: f64,
}

struct GridProjection {
    first_basis: [f64; 3],
    second_basis: [f64; 3],
    first_sq: f64,
    cross_dot: f64,
    second_sq: f64,
    determinant: f64,
}

/// Factor a finite point set into a small motif repeated over a full 2-D grid.
///
/// This is a second stage after primitive lattice fitting. A staggered array
/// can be an exact subset of a finer Bravais lattice yet have a much simpler
/// CAD description as (small motif) × (coarser full grid).
#[must_use]
pub fn factor_point_motif_pattern(
    points: &[[f64; 3]],
    tolerance_mm: f64,
) -> Option<PointMotifPattern> {
    if points.len() < 4 || !tolerance_mm.is_finite() || tolerance_mm <= 0.0 {
        return None;
    }

    let candidates = motif_candidate_vectors(points, tolerance_mm, 48);
    let mut best: Option<(usize, usize, f64, PointMotifPattern)> = None;
    for left in 0..candidates.len() {
        for right in (left + 1)..candidates.len() {
            let a = candidates[left];
            let b = candidates[right];
            let Some(pattern) = factor_point_motif_with_basis(points, a, b, tolerance_mm) else {
                continue;
            };
            let motif_size = pattern.motif_offsets.len();
            let inverse_repeats = usize::MAX - pattern.repeat_count;
            let basis_cost = norm(a) + norm(b);
            match &best {
                None => best = Some((motif_size, inverse_repeats, basis_cost, pattern)),
                Some((best_motif, best_inverse_repeats, best_basis_cost, _))
                    if motif_size < *best_motif
                        || (motif_size == *best_motif
                            && inverse_repeats < *best_inverse_repeats)
                        || (motif_size == *best_motif
                            && inverse_repeats == *best_inverse_repeats
                            && basis_cost < *best_basis_cost - 1.0e-12) =>
                {
                    best = Some((motif_size, inverse_repeats, basis_cost, pattern));
                }
                _ => {}
            }
        }
    }
    best.map(|(_, _, _, pattern)| pattern)
}

fn motif_candidate_vectors(points: &[[f64; 3]], tolerance_mm: f64, limit: usize) -> Vec<[f64; 3]> {
    let quant = tolerance_mm.max(1.0e-9);
    let mut counts = HashMap::<[i64; 3], ([f64; 3], usize)>::new();
    for left in 0..points.len() {
        for right in (left + 1)..points.len() {
            let mut vector = sub(points[right], points[left]);
            if norm(vector) <= tolerance_mm {
                continue;
            }
            vector = canonical_direction(vector);
            let Some(key) = quantize_vec3(vector, quant) else {
                continue;
            };
            let entry = counts.entry(key).or_insert((vector, 0));
            entry.1 += 1;
        }
    }

    let mut ranked = counts.into_values().collect::<Vec<_>>();
    ranked.sort_by(|(left_vector, left_count), (right_vector, right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| norm(*left_vector).total_cmp(&norm(*right_vector)))
            .then_with(|| {
                left_vector[0]
                    .total_cmp(&right_vector[0])
                    .then_with(|| left_vector[1].total_cmp(&right_vector[1]))
                    .then_with(|| left_vector[2].total_cmp(&right_vector[2]))
            })
    });
    ranked
        .into_iter()
        .take(limit)
        .map(|(vector, _)| vector)
        .collect()
}

fn factor_point_motif_with_basis(
    points: &[[f64; 3]],
    first_basis: [f64; 3],
    second_basis: [f64; 3],
    tolerance_mm: f64,
) -> Option<PointMotifPattern> {
    let projection = project_motif_points(points, first_basis, second_basis, tolerance_mm)?;
    if projection.groups.is_empty() {
        return None;
    }

    let mut expected_shape = None::<[usize; 2]>;
    let mut motif_origins = Vec::<[f64; 3]>::new();
    let mut max_residual_mm = projection.max_residual_mm;
    for members in projection.groups.values() {
        let (shape, origin, residual) =
            prove_motif_grid_class(members, first_basis, second_basis, tolerance_mm)?;
        if expected_shape.is_some_and(|expected| expected != shape) {
            return None;
        }
        expected_shape = Some(shape);
        motif_origins.push(origin);
        max_residual_mm = max_residual_mm.max(residual);
    }

    let shape = expected_shape?;
    let repeat_count = shape[0].checked_mul(shape[1])?;
    if repeat_count.checked_mul(motif_origins.len())? != points.len() {
        return None;
    }

    motif_origins.sort_by(|left, right| {
        left[0]
            .total_cmp(&right[0])
            .then_with(|| left[1].total_cmp(&right[1]))
            .then_with(|| left[2].total_cmp(&right[2]))
    });
    let origin = motif_origins[0];
    let motif_offsets = motif_origins
        .iter()
        .map(|&motif_origin| sub(motif_origin, origin))
        .collect::<Vec<_>>();

    Some(PointMotifPattern {
        dimension: 2,
        origin,
        repeat_basis: vec![first_basis, second_basis],
        repeat_pitch: vec![norm(first_basis), norm(second_basis)],
        grid_shape: vec![shape[0], shape[1]],
        motif_offsets,
        repeat_count,
        max_residual_mm,
    })
}

fn project_motif_points(
    points: &[[f64; 3]],
    first_basis: [f64; 3],
    second_basis: [f64; 3],
    tolerance_mm: f64,
) -> Option<MotifProjection> {
    let first_sq = dot(first_basis, first_basis);
    let cross_dot = dot(first_basis, second_basis);
    let second_sq = dot(second_basis, second_basis);
    let determinant = cross_dot.mul_add(-cross_dot, first_sq * second_sq);
    if determinant <= 1.0e-18 {
        return None;
    }
    let base = *points.first()?;
    let coordinate_tol = tolerance_mm / norm(first_basis).min(norm(second_basis)).max(1.0e-12);
    let frac_quant = coordinate_tol.max(1.0e-10);

    let mut groups = BTreeMap::<[i64; 2], Vec<MotifPoint>>::new();
    let mut max_residual_mm = 0.0f64;
    for &point in points {
        let delta = sub(point, base);
        let first_dot = dot(first_basis, delta);
        let second_dot = dot(second_basis, delta);
        let u = second_dot.mul_add(-cross_dot, first_dot * second_sq) / determinant;
        let v = first_dot.mul_add(-cross_dot, second_dot * first_sq) / determinant;
        let reconstructed = add(base, add(scale(first_basis, u), scale(second_basis, v)));
        let residual = norm(sub(point, reconstructed));
        if !residual.is_finite() || residual > tolerance_mm {
            return None;
        }
        max_residual_mm = max_residual_mm.max(residual);

        let (integer_u, fraction_u) = split_lattice_coordinate(u, coordinate_tol)?;
        let (integer_v, fraction_v) = split_lattice_coordinate(v, coordinate_tol)?;
        let key = [
            rounded_f64_to_i64(fraction_u / frac_quant)?,
            rounded_f64_to_i64(fraction_v / frac_quant)?,
        ];
        groups.entry(key).or_default().push(MotifPoint {
            point,
            integer: [integer_u, integer_v],
        });
    }

    Some(MotifProjection {
        groups,
        max_residual_mm,
    })
}

fn prove_motif_grid_class(
    members: &[MotifPoint],
    first_basis: [f64; 3],
    second_basis: [f64; 3],
    tolerance_mm: f64,
) -> Option<([usize; 2], [f64; 3], f64)> {
    let min_u = members.iter().map(|member| member.integer[0]).min()?;
    let max_u = members.iter().map(|member| member.integer[0]).max()?;
    let min_v = members.iter().map(|member| member.integer[1]).min()?;
    let max_v = members.iter().map(|member| member.integer[1]).max()?;
    let nu = usize::try_from(max_u.checked_sub(min_u)?.checked_add(1)?).ok()?;
    let nv = usize::try_from(max_v.checked_sub(min_v)?.checked_add(1)?).ok()?;
    if nu <= 1 || nv <= 1 {
        return None;
    }
    let shape = [nu, nv];
    let occupancy = members
        .iter()
        .map(|member| {
            Some([
                member.integer[0].checked_sub(min_u)?,
                member.integer[1].checked_sub(min_v)?,
            ])
        })
        .collect::<Option<HashSet<_>>>()?;
    if occupancy.len() != nu.checked_mul(nv)? || occupancy.len() != members.len() {
        return None;
    }

    let first = members.first()?;
    let normalized = [
        first.integer[0].checked_sub(min_u)?,
        first.integer[1].checked_sub(min_v)?,
    ];
    let origin = sub(
        first.point,
        add(
            scale(first_basis, exact_i64_to_f64(normalized[0])?),
            scale(second_basis, exact_i64_to_f64(normalized[1])?),
        ),
    );
    let mut max_residual_mm = 0.0f64;
    for member in members {
        let normalized = [
            member.integer[0].checked_sub(min_u)?,
            member.integer[1].checked_sub(min_v)?,
        ];
        let reconstructed = add(
            origin,
            add(
                scale(first_basis, exact_i64_to_f64(normalized[0])?),
                scale(second_basis, exact_i64_to_f64(normalized[1])?),
            ),
        );
        let residual = norm(sub(member.point, reconstructed));
        if residual > tolerance_mm {
            return None;
        }
        max_residual_mm = max_residual_mm.max(residual);
    }
    Some((shape, origin, max_residual_mm))
}

fn split_lattice_coordinate(value: f64, tolerance: f64) -> Option<(i64, f64)> {
    let nearest = value.round();
    if (value - nearest).abs() <= tolerance {
        return Some((rounded_f64_to_i64(value)?, 0.0));
    }
    let floor = value.floor();
    let mut fraction = value - floor;
    if fraction >= 1.0 - tolerance {
        return Some((floored_f64_to_i64(value)?.checked_add(1)?, 0.0));
    }
    if fraction <= tolerance {
        fraction = 0.0;
    }
    Some((floored_f64_to_i64(value)?, fraction))
}

#[derive(Debug, Clone)]
struct Fit {
    dimension: u8,
    origin: [f64; 3],
    basis: Vec<[f64; 3]>,
    pitch: Vec<f64>,
    occupancy: Vec<[i64; 2]>,
    grid_shape: Vec<usize>,
    fill_ratio: f64,
    max_residual_mm: f64,
}

#[must_use]
pub fn detect_instance_patterns(
    entities: &[EntityInstance],
    tolerance_mm: f64,
    min_items: usize,
) -> Vec<InstancePattern> {
    let index = build_index(entities);
    detect_instance_patterns_with_index(entities, &index, tolerance_mm, min_items)
}

pub(crate) fn detect_instance_patterns_with_index(
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    tolerance_mm: f64,
    min_items: usize,
) -> Vec<InstancePattern> {
    if tolerance_mm <= 0.0 || !tolerance_mm.is_finite() || min_items < 2 {
        return Vec::new();
    }

    let parents = mapped_item_parents(entities);
    let styles = styles_by_target(entities);
    let mut groups: BTreeMap<GroupKey, Vec<MappedPlacement>> = BTreeMap::new();

    for entity in entities {
        let item = entity_id(entity);
        let Some(record) = simple_record(entity) else {
            continue;
        };
        if record.name != "MAPPED_ITEM" {
            continue;
        }
        let Parameter::List(params) = &record.parameter else {
            continue;
        };
        if params.len() < 3 {
            continue;
        }
        let Some(rep_map) = entity_ref_value(&params[1]) else {
            continue;
        };
        let Some(target) = entity_ref_value(&params[2]) else {
            continue;
        };
        let Some(parent_list) = parents.get(&item) else {
            continue;
        };
        if parent_list.len() != 1 {
            continue;
        }
        let Some((origin, orientation)) = target_frame(target, entities, index) else {
            continue;
        };
        let Some(orientation) = quantize_matrix(orientation) else {
            continue;
        };

        let mut style_signature = styles.get(&item).cloned().unwrap_or_default();
        style_signature.sort();

        groups
            .entry(GroupKey {
                parent_representation: parent_list[0],
                representation_map: rep_map,
                orientation,
                style_signature,
            })
            .or_default()
            .push(MappedPlacement { item, origin });
    }

    let mut out = Vec::new();
    for (key, mut members) in groups {
        if members.len() < min_items {
            continue;
        }
        members.sort_by_key(|m| m.item);
        let positions: Vec<[f64; 3]> = members.iter().map(|m| m.origin).collect();
        let Some(fit) = fit_lattice(&positions, tolerance_mm) else {
            continue;
        };
        let mut ordered = members
            .iter()
            .zip(fit.occupancy.iter().copied())
            .map(|(member, site)| (site, member.item))
            .collect::<Vec<_>>();
        ordered.sort_by_key(|(site, item)| (*site, *item));

        out.push(InstancePattern {
            parent_representation: key.parent_representation,
            representation_map: key.representation_map,
            item_ids: ordered.iter().map(|(_, item)| *item).collect(),
            style_signature: key.style_signature,
            dimension: fit.dimension,
            origin: fit.origin,
            basis: fit.basis,
            pitch: fit.pitch,
            occupancy: ordered.iter().map(|(site, _)| *site).collect(),
            grid_shape: fit.grid_shape,
            fill_ratio: fit.fill_ratio,
            tolerance_mm,
            max_residual_mm: fit.max_residual_mm,
        });
    }

    out.sort_by(|a, b| {
        b.item_ids
            .len()
            .cmp(&a.item_ids.len())
            .then_with(|| a.parent_representation.cmp(&b.parent_representation))
            .then_with(|| a.representation_map.cmp(&b.representation_map))
    });
    out
}

struct PatternResizeTemplates {
    old_targets: Vec<u64>,
    old_style_ids: Vec<u64>,
    axis_param: Parameter,
    refdir_param: Parameter,
    assignments: Vec<u64>,
}

struct PatternRewrite {
    new_items: Vec<u64>,
    new_styles: Vec<u64>,
    candidate_roots: Vec<u64>,
    reused: usize,
}

pub(crate) fn resize_filled_linear_pattern(
    entities: &mut Vec<EntityInstance>,
    pattern: &InstancePattern,
    new_count: usize,
    anchor: PatternAnchor,
) -> Result<PatternResizeStats> {
    let old_count = validate_resize_request(pattern, new_count)?;
    let index = build_index(entities);
    let refs_before = ReferenceGraph::new(entities);
    let styles = style_records_by_target(entities);
    let templates = collect_resize_templates(entities, pattern, &index, &styles)?;
    let new_origin = anchored_pattern_origin(pattern, old_count, new_count, anchor)?;
    let rewrite = rewrite_pattern_sites(entities, pattern, new_count, new_origin, &templates)?;
    rewrite_pattern_references(
        entities,
        pattern,
        &templates.old_style_ids,
        &rewrite.new_items,
        &rewrite.new_styles,
    )?;
    let candidate = candidate_closure(&refs_before, rewrite.candidate_roots);
    let entities_removed = prune_detached_candidates(entities, &candidate);

    Ok(PatternResizeStats {
        old_count,
        new_count,
        reused_items: rewrite.reused,
        added_items: new_count.saturating_sub(rewrite.reused),
        removed_items: old_count.saturating_sub(rewrite.reused),
        entities_removed,
    })
}

fn validate_resize_request(pattern: &InstancePattern, new_count: usize) -> Result<usize> {
    if pattern.dimension != 1 || pattern.basis.len() != 1 || pattern.grid_shape.len() != 1 {
        bail!("pattern is not one-dimensional");
    }
    if new_count == 0 {
        bail!("new pattern count must be at least 1");
    }
    let old_count = pattern.item_ids.len();
    if old_count < 2 || pattern.occupancy.len() != old_count {
        bail!("pattern does not contain enough instances");
    }
    if pattern.grid_shape[0] != old_count || (pattern.fill_ratio - 1.0).abs() > 1.0e-12 {
        bail!("pattern is not fully occupied");
    }
    for (index, site) in pattern.occupancy.iter().enumerate() {
        let expected = i64::try_from(index)
            .map_err(|_| anyhow::anyhow!("pattern occupancy index exceeds i64 range"))?;
        if *site != [expected, 0] {
            bail!("pattern occupancy is not canonical contiguous 0..N-1");
        }
    }
    Ok(old_count)
}

fn collect_resize_templates(
    entities: &[EntityInstance],
    pattern: &InstancePattern,
    index: &HashMap<u64, usize>,
    styles: &HashMap<u64, Vec<(u64, Vec<u64>)>>,
) -> Result<PatternResizeTemplates> {
    let mut old_targets = Vec::with_capacity(pattern.item_ids.len());
    let mut old_style_ids = Vec::with_capacity(pattern.item_ids.len());
    let mut style_assignments: Option<Vec<u64>> = None;
    let mut template_axis_tail: Option<(Parameter, Parameter)> = None;

    for &item in &pattern.item_ids {
        let record = simple_record(
            entities
                .get(
                    *index
                        .get(&item)
                        .ok_or_else(|| anyhow::anyhow!("missing mapped item #{item}"))?,
                )
                .ok_or_else(|| anyhow::anyhow!("mapped item #{item} is complex"))?,
        )
        .ok_or_else(|| anyhow::anyhow!("mapped item #{item} is complex"))?;
        if record.name != "MAPPED_ITEM" {
            bail!("pattern item #{item} is not MAPPED_ITEM");
        }
        let Parameter::List(params) = &record.parameter else {
            bail!("mapped item #{item} has non-list parameters");
        };
        if params.len() < 3 || entity_ref_value(&params[1]) != Some(pattern.representation_map) {
            bail!("pattern item #{item} does not reference the expected representation map");
        }
        let target = entity_ref_value(&params[2])
            .ok_or_else(|| anyhow::anyhow!("mapped item #{item} has no placement target"))?;
        collect_placement_template(entities, index, target, &mut template_axis_tail)?;
        old_targets.push(target);
        collect_style_template(styles, item, &mut style_assignments, &mut old_style_ids)?;
    }
    validate_parent_contains_pattern(entities, pattern, index)?;
    let (axis_param, refdir_param) =
        template_axis_tail.ok_or_else(|| anyhow::anyhow!("missing placement template"))?;
    let assignments = style_assignments.ok_or_else(|| anyhow::anyhow!("missing style template"))?;
    Ok(PatternResizeTemplates {
        old_targets,
        old_style_ids,
        axis_param,
        refdir_param,
        assignments,
    })
}

fn collect_placement_template(
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    target: u64,
    template_axis_tail: &mut Option<(Parameter, Parameter)>,
) -> Result<()> {
    let target_record = simple_record(
        entities
            .get(
                *index
                    .get(&target)
                    .ok_or_else(|| anyhow::anyhow!("missing target #{target}"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("target #{target} is complex"))?,
    )
    .ok_or_else(|| anyhow::anyhow!("target #{target} is complex"))?;
    if target_record.name != "AXIS2_PLACEMENT_3D" {
        bail!("pattern target #{target} is not AXIS2_PLACEMENT_3D");
    }
    let Parameter::List(target_params) = &target_record.parameter else {
        bail!("target #{target} has non-list parameters");
    };
    if target_params.len() < 4 {
        bail!("target #{target} has incomplete AXIS2_PLACEMENT_3D parameters");
    }
    match template_axis_tail {
        None => *template_axis_tail = Some((target_params[2].clone(), target_params[3].clone())),
        Some((axis, refdir)) if *axis == target_params[2] && *refdir == target_params[3] => {}
        Some(_) => bail!("pattern placements do not share exact axis/ref-direction references"),
    }
    Ok(())
}

fn collect_style_template(
    styles: &HashMap<u64, Vec<(u64, Vec<u64>)>>,
    item: u64,
    style_assignments: &mut Option<Vec<u64>>,
    old_style_ids: &mut Vec<u64>,
) -> Result<()> {
    let Some(item_styles) = styles.get(&item) else {
        bail!("pattern item #{item} has no direct style");
    };
    if item_styles.len() != 1 {
        bail!("pattern item #{item} does not have exactly one direct style");
    }
    let (style_id, assignments) = &item_styles[0];
    match style_assignments {
        None => *style_assignments = Some(assignments.clone()),
        Some(expected) if expected == assignments => {}
        Some(_) => bail!("pattern items do not share one style assignment"),
    }
    old_style_ids.push(*style_id);
    Ok(())
}

fn validate_parent_contains_pattern(
    entities: &[EntityInstance],
    pattern: &InstancePattern,
    index: &HashMap<u64, usize>,
) -> Result<()> {
    let parent_idx = *index
        .get(&pattern.parent_representation)
        .ok_or_else(|| anyhow::anyhow!("missing parent representation"))?;
    let parent = simple_record(&entities[parent_idx])
        .ok_or_else(|| anyhow::anyhow!("parent representation is complex"))?;
    let Parameter::List(params) = &parent.parameter else {
        bail!("parent representation has non-list parameters");
    };
    let items = params
        .get(1)
        .and_then(entity_ref_list)
        .ok_or_else(|| anyhow::anyhow!("parent representation has no item aggregate"))?;
    if !pattern.item_ids.iter().all(|item| items.contains(item)) {
        bail!("parent representation does not contain every pattern item");
    }
    Ok(())
}

fn anchored_pattern_origin(
    pattern: &InstancePattern,
    old_count: usize,
    new_count: usize,
    anchor: PatternAnchor,
) -> Result<[f64; 3]> {
    let basis = pattern.basis[0];
    let old_span = exact_usize_to_f64(old_count.saturating_sub(1))
        .ok_or_else(|| anyhow::anyhow!("old pattern count exceeds exact geometry range"))?;
    let new_span = exact_usize_to_f64(new_count.saturating_sub(1))
        .ok_or_else(|| anyhow::anyhow!("new pattern count exceeds exact geometry range"))?;
    let old_center = add(pattern.origin, scale(basis, old_span * 0.5));
    Ok(match anchor {
        PatternAnchor::Start => pattern.origin,
        PatternAnchor::Center => add(old_center, scale(basis, -(new_span * 0.5))),
        PatternAnchor::End => {
            let old_end = add(pattern.origin, scale(basis, old_span));
            add(old_end, scale(basis, -new_span))
        }
    })
}

fn rewrite_pattern_sites(
    entities: &mut Vec<EntityInstance>,
    pattern: &InstancePattern,
    new_count: usize,
    new_origin: [f64; 3],
    templates: &PatternResizeTemplates,
) -> Result<PatternRewrite> {
    let basis = pattern.basis[0];
    let mut next_id = entities
        .iter()
        .map(entity_id)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("STEP entity id space exhausted"))?;
    let reused = pattern.item_ids.len().min(new_count);
    let mut new_items = Vec::with_capacity(new_count);
    let mut new_styles = Vec::with_capacity(new_count);
    let mut candidate_roots = templates.old_targets.clone();

    for site in 0..new_count {
        let site_f64 = exact_usize_to_f64(site)
            .ok_or_else(|| anyhow::anyhow!("pattern site exceeds exact geometry range"))?;
        let origin = add(new_origin, scale(basis, site_f64));
        let placement = push_pattern_placement(
            entities,
            &mut next_id,
            origin,
            &templates.axis_param,
            &templates.refdir_param,
        );
        let (item, style) = if site < reused {
            rewrite_reused_pattern_item(
                entities,
                pattern.item_ids[site],
                placement,
                *templates.old_style_ids.get(site).ok_or_else(|| {
                    anyhow::anyhow!("missing style for reused pattern site {site}")
                })?,
            )?
        } else {
            push_pattern_item(
                entities,
                &mut next_id,
                pattern.representation_map,
                placement,
                &templates.assignments,
            )
        };
        new_items.push(item);
        new_styles.push(style);
    }

    let removed_items = pattern.item_ids.iter().copied().skip(reused);
    let removed_styles = templates.old_style_ids.iter().copied().skip(reused);
    candidate_roots.extend(removed_items);
    candidate_roots.extend(removed_styles);
    Ok(PatternRewrite {
        new_items,
        new_styles,
        candidate_roots,
        reused,
    })
}

fn push_pattern_placement(
    entities: &mut Vec<EntityInstance>,
    next_id: &mut u64,
    origin: [f64; 3],
    axis_param: &Parameter,
    refdir_param: &Parameter,
) -> u64 {
    let point = push_simple(
        entities,
        next_id,
        "CARTESIAN_POINT",
        vec![
            Parameter::String(String::new()),
            Parameter::List(origin.into_iter().map(Parameter::Real).collect()),
        ],
    );
    push_simple(
        entities,
        next_id,
        "AXIS2_PLACEMENT_3D",
        vec![
            Parameter::String(String::new()),
            entity_ref(point),
            axis_param.clone(),
            refdir_param.clone(),
        ],
    )
}

fn rewrite_reused_pattern_item(
    entities: &mut [EntityInstance],
    item: u64,
    placement: u64,
    style: u64,
) -> Result<(u64, u64)> {
    let current_index = build_index(entities);
    let idx = *current_index
        .get(&item)
        .ok_or_else(|| anyhow::anyhow!("mapped item disappeared during rewrite"))?;
    let record = simple_record_mut(&mut entities[idx])
        .ok_or_else(|| anyhow::anyhow!("mapped item became complex"))?;
    let Parameter::List(params) = &mut record.parameter else {
        bail!("mapped item parameters changed shape");
    };
    params[2] = entity_ref(placement);
    Ok((item, style))
}

fn push_pattern_item(
    entities: &mut Vec<EntityInstance>,
    next_id: &mut u64,
    representation_map: u64,
    placement: u64,
    assignments: &[u64],
) -> (u64, u64) {
    let item = push_simple(
        entities,
        next_id,
        "MAPPED_ITEM",
        vec![
            Parameter::String(String::new()),
            entity_ref(representation_map),
            entity_ref(placement),
        ],
    );
    let style = push_simple(
        entities,
        next_id,
        "STYLED_ITEM",
        vec![
            Parameter::String(String::new()),
            Parameter::List(assignments.iter().copied().map(entity_ref).collect()),
            entity_ref(item),
        ],
    );
    (item, style)
}

fn rewrite_pattern_references(
    entities: &mut [EntityInstance],
    pattern: &InstancePattern,
    old_style_ids: &[u64],
    new_items: &[u64],
    new_styles: &[u64],
) -> Result<()> {
    let current_index = build_index(entities);
    let idx = *current_index
        .get(&pattern.parent_representation)
        .ok_or_else(|| anyhow::anyhow!("parent representation disappeared"))?;
    rewrite_ref_sequence(&mut entities[idx], 1, &pattern.item_ids, new_items)?;
    rewrite_presentation_style_lists(entities, old_style_ids, new_styles);
    Ok(())
}

fn rewrite_presentation_style_lists(
    entities: &mut [EntityInstance],
    old_style_ids: &[u64],
    new_styles: &[u64],
) {
    let old_style_set = old_style_ids.iter().copied().collect::<HashSet<_>>();
    for entity in entities {
        let Some(record) = simple_record_mut(entity) else {
            continue;
        };
        let list_index = match record.name.as_str() {
            "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION" => 1,
            "PRESENTATION_LAYER_ASSIGNMENT" => 2,
            _ => continue,
        };
        let Parameter::List(params) = &mut record.parameter else {
            continue;
        };
        let Some(Parameter::List(items)) = params.get_mut(list_index) else {
            continue;
        };
        if !items
            .iter()
            .filter_map(entity_ref_value)
            .any(|id| old_style_set.contains(&id))
        {
            continue;
        }
        let mut out = Vec::with_capacity(items.len() + new_styles.len());
        let mut inserted = false;
        for item in items.iter() {
            if let Some(id) = entity_ref_value(item)
                && old_style_set.contains(&id)
            {
                if !inserted {
                    out.extend(new_styles.iter().copied().map(entity_ref));
                    inserted = true;
                }
                continue;
            }
            out.push(item.clone());
        }
        *items = out;
    }
}

fn candidate_closure(refs_before: &ReferenceGraph, roots: Vec<u64>) -> HashSet<u64> {
    let mut candidate = HashSet::new();
    let mut stack = roots;
    while let Some(id) = stack.pop() {
        if !candidate.insert(id) {
            continue;
        }
        stack.extend(refs_before.refs(id).iter().copied());
    }
    candidate
}

fn prune_detached_candidates(
    entities: &mut Vec<EntityInstance>,
    candidate: &HashSet<u64>,
) -> usize {
    let refs_after = ReferenceGraph::new(entities);
    let inbound = refs_after.inbound();
    let mut delete = HashSet::new();
    loop {
        let mut changed = false;
        for &id in candidate {
            if delete.contains(&id) {
                continue;
            }
            let parents = inbound.get(&id).cloned().unwrap_or_default();
            if parents.is_empty() || parents.iter().all(|parent| delete.contains(parent)) {
                delete.insert(id);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let entities_removed = delete.len();
    entities.retain(|entity| !delete.contains(&entity_id(entity)));
    entities_removed
}

fn rewrite_ref_sequence(
    entity: &mut EntityInstance,
    list_index: usize,
    old_ids: &[u64],
    new_ids: &[u64],
) -> Result<()> {
    let old_set = old_ids
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let record = simple_record_mut(entity).ok_or_else(|| anyhow::anyhow!("entity is complex"))?;
    let Parameter::List(params) = &mut record.parameter else {
        bail!("entity parameters are not a list");
    };
    let Some(Parameter::List(items)) = params.get_mut(list_index) else {
        bail!("entity does not contain the expected reference list");
    };
    let mut out = Vec::with_capacity(items.len() + new_ids.len());
    let mut inserted = false;
    for item in items.iter() {
        if let Some(id) = entity_ref_value(item)
            && old_set.contains(&id)
        {
            if !inserted {
                out.extend(new_ids.iter().copied().map(entity_ref));
                inserted = true;
            }
            continue;
        }
        out.push(item.clone());
    }
    if !inserted {
        bail!("reference sequence to replace was not found");
    }
    *items = out;
    Ok(())
}

fn fit_lattice(points: &[[f64; 3]], tol: f64) -> Option<Fit> {
    fit_line(points, tol).or_else(|| fit_grid(points, tol))
}

fn pair_differences(points: &[[f64; 3]]) -> Vec<([f64; 3], f64)> {
    let mut diffs = Vec::new();
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let d = sub(points[j], points[i]);
            let n = norm(d);
            if n.is_finite() && n > 1.0e-12 {
                diffs.push((d, n));
            }
        }
    }
    diffs.sort_by(|a, b| a.1.total_cmp(&b.1));
    diffs
}

fn lattice_candidate_vectors(points: &[[f64; 3]], tol: f64, limit: usize) -> Vec<[f64; 3]> {
    let quant = tol.max(1.0e-9);
    let mut seen = std::collections::HashSet::<[i64; 3]>::new();
    let mut out = Vec::new();
    for (mut vector, _) in pair_differences(points) {
        vector = canonical_direction(vector);
        let Some(key) = quantize_vec3(vector, quant) else {
            continue;
        };
        if !seen.insert(key) {
            continue;
        }
        out.push(vector);
        if out.len() >= limit {
            break;
        }
    }
    out
}

fn fit_line(points: &[[f64; 3]], tol: f64) -> Option<Fit> {
    let candidates = lattice_candidate_vectors(points, tol, 64);
    let base = *points.first()?;
    let mut best: Option<(f64, f64, Fit)> = None;

    for basis in candidates {
        let pitch = norm(basis);
        if pitch <= 1.0e-12 {
            continue;
        }
        let unit = scale(basis, 1.0 / pitch);
        let mut raw_sites = Vec::with_capacity(points.len());
        let mut max_residual = 0.0f64;
        let mut okay = true;

        for &point in points {
            let delta = sub(point, base);
            let coordinate = dot(delta, unit) / pitch;
            let rounded = coordinate.round();
            let Some(site) = rounded_f64_to_i64(coordinate) else {
                okay = false;
                break;
            };
            let reconstructed = add(base, scale(basis, rounded));
            let residual = norm(sub(point, reconstructed));
            if residual > tol || !residual.is_finite() {
                okay = false;
                break;
            }
            max_residual = max_residual.max(residual);
            raw_sites.push(site);
        }
        if !okay {
            continue;
        }

        raw_sites.sort_unstable();
        raw_sites.dedup();
        if raw_sites.len() != points.len() {
            continue;
        }
        let min_site = *raw_sites.first()?;
        let max_site = *raw_sites.last()?;
        let Some(span) = max_site
            .checked_sub(min_site)
            .and_then(|value| value.checked_add(1))
        else {
            continue;
        };
        if span <= 0 {
            continue;
        }
        let (Some(point_count), Some(span_f64), Some(min_site_f64), Ok(span_usize)) = (
            exact_usize_to_f64(points.len()),
            exact_i64_to_f64(span),
            exact_i64_to_f64(min_site),
            usize::try_from(span),
        ) else {
            continue;
        };
        let fill_ratio = point_count / span_f64;
        let origin = add(base, scale(basis, min_site_f64));

        let occupancy = points
            .iter()
            .map(|&point| {
                let coordinate = dot(sub(point, base), unit) / pitch;
                rounded_f64_to_i64(coordinate)?
                    .checked_sub(min_site)
                    .map(|site| [site, 0])
            })
            .collect::<Option<Vec<_>>>();
        let Some(occupancy) = occupancy else {
            continue;
        };
        let fit = Fit {
            dimension: 1,
            origin,
            basis: vec![basis],
            pitch: vec![pitch],
            occupancy,
            grid_shape: vec![span_usize],
            fill_ratio,
            max_residual_mm: max_residual,
        };

        let score = (fill_ratio, -pitch);
        match &best {
            None => best = Some((score.0, score.1, fit)),
            Some((best_fill, best_pitch, _))
                if score.0 > *best_fill + 1.0e-12
                    || ((score.0 - *best_fill).abs() <= 1.0e-12 && score.1 > *best_pitch) =>
            {
                best = Some((score.0, score.1, fit));
            }
            _ => {}
        }
    }

    best.map(|(_, _, fit)| fit)
}

fn fit_grid(points: &[[f64; 3]], tol: f64) -> Option<Fit> {
    let base = *points.first()?;
    let candidates = lattice_candidate_vectors(points, tol, 64);
    let mut best: Option<(f64, f64, Fit)> = None;

    for first_index in 0..candidates.len() {
        for second_index in (first_index + 1)..candidates.len() {
            let first_basis = canonical_direction(candidates[first_index]);
            let second_basis = canonical_direction(candidates[second_index]);
            let Some((fill_ratio, cell_area, fit)) =
                fit_grid_candidate(points, base, first_basis, second_basis, tol)
            else {
                continue;
            };
            match &best {
                None => best = Some((fill_ratio, cell_area, fit)),
                Some((best_fill, best_area, _))
                    if fill_ratio > *best_fill + 1.0e-12
                        || ((fill_ratio - *best_fill).abs() <= 1.0e-12
                            && cell_area < *best_area) =>
                {
                    best = Some((fill_ratio, cell_area, fit));
                }
                _ => {}
            }
        }
    }

    best.map(|(_, _, fit)| fit)
}

fn fit_grid_candidate(
    points: &[[f64; 3]],
    base: [f64; 3],
    first_basis: [f64; 3],
    second_basis: [f64; 3],
    tol: f64,
) -> Option<(f64, f64, Fit)> {
    let first_sq = dot(first_basis, first_basis);
    let cross_dot = dot(first_basis, second_basis);
    let second_sq = dot(second_basis, second_basis);
    let determinant = cross_dot.mul_add(-cross_dot, first_sq * second_sq);
    if determinant <= 1.0e-18 {
        return None;
    }
    let projection = GridProjection {
        first_basis,
        second_basis,
        first_sq,
        cross_dot,
        second_sq,
        determinant,
    };
    let cell_area = determinant.sqrt();
    let (mut coords, max_residual_mm) = grid_coordinates(points, base, &projection, tol)?;

    let mut unique = coords.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != points.len() {
        return None;
    }

    let u_min = coords.iter().map(|coord| coord[0]).min()?;
    let u_max = coords.iter().map(|coord| coord[0]).max()?;
    let v_min = coords.iter().map(|coord| coord[1]).min()?;
    let v_max = coords.iter().map(|coord| coord[1]).max()?;
    let u_span = u_max.checked_sub(u_min)?.checked_add(1)?;
    let v_span = v_max.checked_sub(v_min)?.checked_add(1)?;
    if u_span <= 1 || v_span <= 1 {
        return None;
    }
    let cell_count = u_span.checked_mul(v_span)?;
    let point_count = exact_usize_to_f64(points.len())?;
    let cell_count_f64 = exact_i64_to_f64(cell_count)?;
    let u_origin_coord = exact_i64_to_f64(u_min)?;
    let v_origin_coord = exact_i64_to_f64(v_min)?;
    let u_size = usize::try_from(u_span).ok()?;
    let v_size = usize::try_from(v_span).ok()?;
    let fill_ratio = point_count / cell_count_f64;
    let origin = add(
        base,
        add(
            scale(first_basis, u_origin_coord),
            scale(second_basis, v_origin_coord),
        ),
    );
    for coord in &mut coords {
        coord[0] -= u_min;
        coord[1] -= v_min;
    }
    let fit = Fit {
        dimension: 2,
        origin,
        basis: vec![first_basis, second_basis],
        pitch: vec![norm(first_basis), norm(second_basis)],
        occupancy: coords,
        grid_shape: vec![u_size, v_size],
        fill_ratio,
        max_residual_mm,
    };
    Some((fill_ratio, cell_area, fit))
}

fn grid_coordinates(
    points: &[[f64; 3]],
    base: [f64; 3],
    projection: &GridProjection,
    tol: f64,
) -> Option<(Vec<[i64; 2]>, f64)> {
    let mut coords = Vec::<[i64; 2]>::with_capacity(points.len());
    let mut max_residual_mm = 0.0f64;
    for &point in points {
        let delta = sub(point, base);
        let first_dot = dot(projection.first_basis, delta);
        let second_dot = dot(projection.second_basis, delta);
        let u = second_dot.mul_add(-projection.cross_dot, first_dot * projection.second_sq)
            / projection.determinant;
        let v = first_dot.mul_add(-projection.cross_dot, second_dot * projection.first_sq)
            / projection.determinant;
        let coordinate = [rounded_f64_to_i64(u)?, rounded_f64_to_i64(v)?];
        let reconstructed = add(
            base,
            add(
                scale(projection.first_basis, u.round()),
                scale(projection.second_basis, v.round()),
            ),
        );
        let residual = norm(sub(point, reconstructed));
        if residual > tol || !residual.is_finite() {
            return None;
        }
        max_residual_mm = max_residual_mm.max(residual);
        coords.push(coordinate);
    }
    Some((coords, max_residual_mm))
}

fn target_frame(
    target: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<([f64; 3], [[f64; 3]; 3])> {
    let record = simple_record(entities.get(*index.get(&target)?)?)?;
    match record.name.as_str() {
        "AXIS2_PLACEMENT_3D" => {
            let Parameter::List(params) = &record.parameter else {
                return None;
            };
            let origin = point_coords(entity_ref_value(params.get(1)?)?, entities, index)?;
            let z = normalize(direction_coords(
                entity_ref_value(params.get(2)?)?,
                entities,
                index,
            )?)?;
            let mut x = normalize(direction_coords(
                entity_ref_value(params.get(3)?)?,
                entities,
                index,
            )?)?;
            x = normalize(sub(x, scale(z, dot(x, z))))?;
            let y = normalize(cross(z, x))?;
            Some((origin, [x, y, z]))
        }
        "CARTESIAN_TRANSFORMATION_OPERATOR_3D" => {
            let Parameter::List(params) = &record.parameter else {
                return None;
            };
            if params.len() < 6 {
                return None;
            }
            let x = normalize(direction_coords(
                entity_ref_value(params.get(1)?)?,
                entities,
                index,
            )?)?;
            let y = normalize(direction_coords(
                entity_ref_value(params.get(2)?)?,
                entities,
                index,
            )?)?;
            let origin = point_coords(entity_ref_value(params.get(3)?)?, entities, index)?;
            let z = normalize(direction_coords(
                entity_ref_value(params.get(5)?)?,
                entities,
                index,
            )?)?;
            Some((origin, [x, y, z]))
        }
        _ => None,
    }
}

fn mapped_item_parents(entities: &[EntityInstance]) -> HashMap<u64, Vec<u64>> {
    let mut out: HashMap<u64, Vec<u64>> = HashMap::new();
    for entity in entities {
        let parent = entity_id(entity);
        let Some(record) = simple_record(entity) else {
            continue;
        };
        if !record.name.contains("SHAPE_REPRESENTATION") {
            continue;
        }
        let Parameter::List(params) = &record.parameter else {
            continue;
        };
        let Some(items) = params.get(1).and_then(entity_ref_list) else {
            continue;
        };
        for item in items {
            out.entry(item).or_default().push(parent);
        }
    }
    out
}

fn styles_by_target(entities: &[EntityInstance]) -> HashMap<u64, Vec<Vec<u64>>> {
    style_records_by_target(entities)
        .into_iter()
        .map(|(target, records)| {
            (
                target,
                records
                    .into_iter()
                    .map(|(_, assignments)| assignments)
                    .collect(),
            )
        })
        .collect()
}

fn point_coords(
    point: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let record = simple_record(entities.get(*index.get(&point)?)?)?;
    if record.name != "CARTESIAN_POINT" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let values = numeric_list(params.get(1)?)?;
    (values.len() == 3).then(|| [values[0], values[1], values[2]])
}

fn direction_coords(
    direction: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let record = simple_record(entities.get(*index.get(&direction)?)?)?;
    if record.name != "DIRECTION" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let values = numeric_list(params.get(1)?)?;
    (values.len() == 3).then(|| [values[0], values[1], values[2]])
}

fn quantize_matrix(matrix: [[f64; 3]; 3]) -> Option<[i64; 9]> {
    let mut out = [0i64; 9];
    for (idx, value) in matrix.into_iter().flatten().enumerate() {
        out[idx] = rounded_f64_to_i64(value / ORIENTATION_Q)?;
    }
    Some(out)
}

fn normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    normalize3(v, 1.0e-15)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_filled_line_pattern() -> anyhow::Result<()> {
        let points = [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ];
        let fit =
            fit_lattice(&points, 1.0e-9).ok_or_else(|| anyhow::anyhow!("expected test value"))?;
        assert_eq!(fit.dimension, 1);
        assert_eq!(fit.grid_shape, vec![4]);
        assert!((fit.pitch[0] - 0.5).abs() < 1.0e-12);
        assert!((fit.fill_ratio - 1.0).abs() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn finds_filled_grid_pattern() -> anyhow::Result<()> {
        let mut points = Vec::new();
        for x in 0..4 {
            for y in 0..3 {
                points.push([x as f64 * 0.5, y as f64 * 0.8, 0.0]);
            }
        }
        let fit =
            fit_lattice(&points, 1.0e-9).ok_or_else(|| anyhow::anyhow!("expected test value"))?;
        assert_eq!(fit.dimension, 2);
        assert_eq!(fit.occupancy.len(), 12);
        assert!((fit.fill_ratio - 1.0).abs() < 1.0e-12);
        let mut shape = fit.grid_shape.clone();
        shape.sort_unstable();
        assert_eq!(shape, vec![3, 4]);
        Ok(())
    }

    #[test]
    fn point_lattice_prefers_dense_staggered_grid_basis() {
        let mut points = Vec::new();
        for row in 0..8 {
            for column in 0..35 {
                points.push([
                    6.412_556 + column as f64 * 1.6 + row as f64 * 0.8,
                    82.505_705 - row as f64 * 1.35,
                    1.4,
                ]);
            }
        }

        let fit = fit_point_lattice(&points, 1.0e-7).expect("staggered grid");
        assert_eq!(fit.dimension, 2);
        let mut shape = fit.grid_shape.clone();
        shape.sort_unstable();
        assert_eq!(shape, vec![8, 35]);
        assert_eq!(fit.occupancy.len(), 280);
        assert!((fit.fill_ratio - 1.0).abs() <= 1.0e-12);
        assert!(fit.max_residual_mm <= 1.0e-9);
    }
    #[test]
    fn factors_alternating_stagger_into_two_point_motif() {
        let mut points = Vec::new();
        for row in 0..8 {
            for column in 0..35 {
                points.push([
                    6.412_556 + column as f64 * 1.6 + (row % 2) as f64 * 0.8,
                    82.505_705 - row as f64 * 1.35,
                    1.4,
                ]);
            }
        }

        let pattern = factor_point_motif_pattern(&points, 1.0e-7).expect("motif pattern");
        assert_eq!(pattern.motif_offsets.len(), 2);
        assert_eq!(pattern.repeat_count, 140);
        let mut shape = pattern.grid_shape.clone();
        shape.sort_unstable();
        assert_eq!(shape, vec![4, 35]);
        let mut pitch = pattern.repeat_pitch.clone();
        pitch.sort_by(f64::total_cmp);
        assert!((pitch[0] - 1.6).abs() <= 1.0e-9);
        assert!((pitch[1] - 2.7).abs() <= 1.0e-9);
        assert!(pattern.max_residual_mm <= 1.0e-9);
    }
}
