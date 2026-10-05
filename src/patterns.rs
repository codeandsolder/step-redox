use crate::step_graph::{
    build_index, entity_id, entity_ref_value, simple_record, simple_record_mut,
};
use anyhow::{Result, bail};
use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

const ORIENTATION_Q: f64 = 1.0e-10;

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

/// Factor a finite point set into a small motif repeated over a full 2-D grid.
///
/// This is a second stage after primitive lattice fitting. A staggered array
/// can be an exact subset of a finer Bravais lattice yet have a much simpler
/// CAD description as (small motif) × (coarser full grid).
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
            canonicalize_vector(&mut vector);
            let key = [
                (vector[0] / quant).round() as i64,
                (vector[1] / quant).round() as i64,
                (vector[2] / quant).round() as i64,
            ];
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
    a: [f64; 3],
    b: [f64; 3],
    tolerance_mm: f64,
) -> Option<PointMotifPattern> {
    let aa = dot(a, a);
    let ab = dot(a, b);
    let bb = dot(b, b);
    let det = aa * bb - ab * ab;
    if det <= 1.0e-18 {
        return None;
    }
    let base = *points.first()?;
    let coordinate_tol = tolerance_mm / norm(a).min(norm(b)).max(1.0e-12);

    #[derive(Clone)]
    struct MotifPoint {
        point: [f64; 3],
        integer: [i64; 2],
    }

    let mut groups = BTreeMap::<[i64; 2], Vec<MotifPoint>>::new();
    let mut max_residual = 0.0f64;
    for &point in points {
        let delta = sub(point, base);
        let ad = dot(a, delta);
        let bd = dot(b, delta);
        let u = (ad * bb - bd * ab) / det;
        let v = (bd * aa - ad * ab) / det;
        let reconstructed = add(base, add(scale(a, u), scale(b, v)));
        let residual = norm(sub(point, reconstructed));
        if !residual.is_finite() || residual > tolerance_mm {
            return None;
        }
        max_residual = max_residual.max(residual);

        let (iu, fu) = split_lattice_coordinate(u, coordinate_tol);
        let (iv, fv) = split_lattice_coordinate(v, coordinate_tol);
        let frac_quant = coordinate_tol.max(1.0e-10);
        let key = [
            (fu / frac_quant).round() as i64,
            (fv / frac_quant).round() as i64,
        ];
        groups.entry(key).or_default().push(MotifPoint {
            point,
            integer: [iu, iv],
        });
    }
    if groups.is_empty() {
        return None;
    }

    let mut expected_shape = None::<[usize; 2]>;
    let mut motif_origins = Vec::<[f64; 3]>::new();
    for members in groups.values() {
        let min_u = members.iter().map(|member| member.integer[0]).min()?;
        let max_u = members.iter().map(|member| member.integer[0]).max()?;
        let min_v = members.iter().map(|member| member.integer[1]).min()?;
        let max_v = members.iter().map(|member| member.integer[1]).max()?;
        let nu = usize::try_from(max_u - min_u + 1).ok()?;
        let nv = usize::try_from(max_v - min_v + 1).ok()?;
        if nu <= 1 || nv <= 1 {
            return None;
        }
        let shape = [nu, nv];
        if expected_shape.is_some_and(|expected| expected != shape) {
            return None;
        }
        expected_shape = Some(shape);

        let occupancy = members
            .iter()
            .map(|member| [member.integer[0] - min_u, member.integer[1] - min_v])
            .collect::<HashSet<_>>();
        if occupancy.len() != nu.saturating_mul(nv) || occupancy.len() != members.len() {
            return None;
        }

        let first = &members[0];
        let normalized = [first.integer[0] - min_u, first.integer[1] - min_v];
        let origin = sub(
            first.point,
            add(
                scale(a, normalized[0] as f64),
                scale(b, normalized[1] as f64),
            ),
        );
        for member in members {
            let normalized = [member.integer[0] - min_u, member.integer[1] - min_v];
            let reconstructed = add(
                origin,
                add(
                    scale(a, normalized[0] as f64),
                    scale(b, normalized[1] as f64),
                ),
            );
            let residual = norm(sub(member.point, reconstructed));
            if residual > tolerance_mm {
                return None;
            }
            max_residual = max_residual.max(residual);
        }
        motif_origins.push(origin);
    }

    let shape = expected_shape?;
    let repeat_count = shape[0].saturating_mul(shape[1]);
    if repeat_count.saturating_mul(motif_origins.len()) != points.len() {
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
        repeat_basis: vec![a, b],
        repeat_pitch: vec![norm(a), norm(b)],
        grid_shape: vec![shape[0], shape[1]],
        motif_offsets,
        repeat_count,
        max_residual_mm: max_residual,
    })
}

fn split_lattice_coordinate(value: f64, tolerance: f64) -> (i64, f64) {
    let nearest = value.round();
    if (value - nearest).abs() <= tolerance {
        return (nearest as i64, 0.0);
    }
    let floor = value.floor();
    let mut fraction = value - floor;
    if fraction >= 1.0 - tolerance {
        return (floor as i64 + 1, 0.0);
    }
    if fraction <= tolerance {
        fraction = 0.0;
    }
    (floor as i64, fraction)
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
    if tolerance_mm <= 0.0 || !tolerance_mm.is_finite() || min_items < 2 {
        return Vec::new();
    }

    let index = build_index(entities);
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
        let Some((origin, orientation)) = target_frame(target, entities, &index) else {
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

pub(crate) fn resize_filled_linear_pattern(
    entities: &mut Vec<EntityInstance>,
    pattern: &InstancePattern,
    new_count: usize,
    anchor: PatternAnchor,
) -> Result<PatternResizeStats> {
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
    if pattern
        .occupancy
        .iter()
        .enumerate()
        .any(|(i, site)| *site != [i as i64, 0])
    {
        bail!("pattern occupancy is not canonical contiguous 0..N-1");
    }

    let index = build_index(entities);
    let refs_before = entity_ref_map(entities);
    let styles = style_records_by_target(entities);

    let mut old_targets = Vec::with_capacity(old_count);
    let mut old_style_ids = Vec::with_capacity(old_count);
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
            bail!("mapped item #{item} does not reference the expected representation map");
        }
        let target = entity_ref_value(&params[2])
            .ok_or_else(|| anyhow::anyhow!("mapped item #{item} has no placement target"))?;
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
        match &template_axis_tail {
            None => template_axis_tail = Some((target_params[2].clone(), target_params[3].clone())),
            Some((axis, refdir)) if *axis == target_params[2] && *refdir == target_params[3] => {}
            Some(_) => bail!("pattern placements do not share exact axis/ref-direction references"),
        }
        old_targets.push(target);

        let Some(item_styles) = styles.get(&item) else {
            bail!("pattern item #{item} has no direct style");
        };
        if item_styles.len() != 1 {
            bail!("pattern item #{item} does not have exactly one direct style");
        }
        let (style_id, assignments) = &item_styles[0];
        match &style_assignments {
            None => style_assignments = Some(assignments.clone()),
            Some(expected) if expected == assignments => {}
            Some(_) => bail!("pattern items do not share one style assignment"),
        }
        old_style_ids.push(*style_id);
    }

    let parent_idx = *index
        .get(&pattern.parent_representation)
        .ok_or_else(|| anyhow::anyhow!("missing parent representation"))?;
    {
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
    }

    let basis = pattern.basis[0];
    let old_center = add(
        pattern.origin,
        scale(basis, (old_count.saturating_sub(1)) as f64 * 0.5),
    );
    let new_origin = match anchor {
        PatternAnchor::Start => pattern.origin,
        PatternAnchor::Center => add(
            old_center,
            scale(basis, -((new_count.saturating_sub(1)) as f64 * 0.5)),
        ),
        PatternAnchor::End => {
            let old_end = add(
                pattern.origin,
                scale(basis, old_count.saturating_sub(1) as f64),
            );
            add(old_end, scale(basis, -(new_count.saturating_sub(1) as f64)))
        }
    };

    let (axis_param, refdir_param) =
        template_axis_tail.ok_or_else(|| anyhow::anyhow!("missing placement template"))?;
    let assignments = style_assignments.ok_or_else(|| anyhow::anyhow!("missing style template"))?;

    let mut next_id = entities.iter().map(entity_id).max().unwrap_or(0) + 1;
    let reused = old_count.min(new_count);
    let mut new_items = Vec::with_capacity(new_count);
    let mut new_styles = Vec::with_capacity(new_count);
    let mut candidate_roots = old_targets.clone();

    for site in 0..new_count {
        let origin = add(new_origin, scale(basis, site as f64));
        let point = push_simple(
            entities,
            &mut next_id,
            "CARTESIAN_POINT",
            vec![
                Parameter::String(String::new()),
                Parameter::List(origin.into_iter().map(Parameter::Real).collect()),
            ],
        );
        let placement = push_simple(
            entities,
            &mut next_id,
            "AXIS2_PLACEMENT_3D",
            vec![
                Parameter::String(String::new()),
                entity_ref(point),
                axis_param.clone(),
                refdir_param.clone(),
            ],
        );

        if site < reused {
            let item = pattern.item_ids[site];
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
            new_items.push(item);
            new_styles.push(
                *old_style_ids.get(site).ok_or_else(|| {
                    anyhow::anyhow!("missing style for reused pattern site {site}")
                })?,
            );
        } else {
            let item = push_simple(
                entities,
                &mut next_id,
                "MAPPED_ITEM",
                vec![
                    Parameter::String(String::new()),
                    entity_ref(pattern.representation_map),
                    entity_ref(placement),
                ],
            );
            let style = push_simple(
                entities,
                &mut next_id,
                "STYLED_ITEM",
                vec![
                    Parameter::String(String::new()),
                    Parameter::List(assignments.iter().copied().map(entity_ref).collect()),
                    entity_ref(item),
                ],
            );
            new_items.push(item);
            new_styles.push(style);
        }
    }

    let removed_items: Vec<u64> = pattern.item_ids.iter().copied().skip(reused).collect();
    let removed_styles: Vec<u64> = old_style_ids.iter().copied().skip(reused).collect();
    candidate_roots.extend(removed_items.iter().copied());
    candidate_roots.extend(removed_styles.iter().copied());

    {
        let current_index = build_index(entities);
        let idx = *current_index
            .get(&pattern.parent_representation)
            .ok_or_else(|| anyhow::anyhow!("parent representation disappeared"))?;
        rewrite_ref_sequence(&mut entities[idx], 1, &pattern.item_ids, &new_items)?;
    }

    let old_style_set = old_style_ids
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    for entity in entities.iter_mut() {
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

    let mut candidate = std::collections::HashSet::new();
    let mut stack = candidate_roots;
    while let Some(id) = stack.pop() {
        if !candidate.insert(id) {
            continue;
        }
        if let Some(children) = refs_before.get(&id) {
            stack.extend(children.iter().copied());
        }
    }

    let refs_after = entity_ref_map(entities);
    let inbound = inbound_map(&refs_after);
    let mut delete = std::collections::HashSet::new();
    loop {
        let mut changed = false;
        for &id in &candidate {
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

    Ok(PatternResizeStats {
        old_count,
        new_count,
        reused_items: reused,
        added_items: new_count.saturating_sub(reused),
        removed_items: old_count.saturating_sub(reused),
        entities_removed,
    })
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
        canonicalize_vector(&mut vector);
        let key = [
            (vector[0] / quant).round() as i64,
            (vector[1] / quant).round() as i64,
            (vector[2] / quant).round() as i64,
        ];
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
            let k = dot(delta, unit) / pitch;
            let site = k.round();
            let reconstructed = add(base, scale(basis, site));
            let residual = norm(sub(point, reconstructed));
            if residual > tol || !residual.is_finite() {
                okay = false;
                break;
            }
            max_residual = max_residual.max(residual);
            raw_sites.push(site as i64);
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
        let span = max_site - min_site + 1;
        if span <= 0 {
            continue;
        }
        let fill_ratio = points.len() as f64 / span as f64;
        let origin = add(base, scale(basis, min_site as f64));

        let occupancy = points
            .iter()
            .map(|&point| {
                let k = (dot(sub(point, base), unit) / pitch).round() as i64 - min_site;
                [k, 0]
            })
            .collect::<Vec<_>>();
        let fit = Fit {
            dimension: 1,
            origin,
            basis: vec![basis],
            pitch: vec![pitch],
            occupancy,
            grid_shape: vec![span as usize],
            fill_ratio,
            max_residual_mm: max_residual,
        };

        let score = (fill_ratio, -pitch);
        match &best {
            None => best = Some((score.0, score.1, fit)),
            Some((bf, bp, _))
                if score.0 > *bf + 1.0e-12
                    || ((score.0 - *bf).abs() <= 1.0e-12 && score.1 > *bp) =>
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

    for ia in 0..candidates.len() {
        for ib in (ia + 1)..candidates.len() {
            let mut a = candidates[ia];
            let mut b = candidates[ib];
            canonicalize_vector(&mut a);
            canonicalize_vector(&mut b);

            let aa = dot(a, a);
            let ab = dot(a, b);
            let bb = dot(b, b);
            let det = ab.mul_add(-ab, aa * bb);
            if det <= 1.0e-18 {
                continue;
            }
            let cell_area = det.sqrt();
            let mut coords = Vec::<[i64; 2]>::with_capacity(points.len());
            let mut max_residual = 0.0f64;
            let mut okay = true;

            for &point in points {
                let d = sub(point, base);
                let ad = dot(a, d);
                let bd = dot(b, d);
                let u = bd.mul_add(-ab, ad * bb) / det;
                let v = ad.mul_add(-ab, bd * aa) / det;
                let iu = u.round();
                let iv = v.round();
                let reconstructed = add(base, add(scale(a, iu), scale(b, iv)));
                let residual = norm(sub(point, reconstructed));
                if residual > tol || !residual.is_finite() {
                    okay = false;
                    break;
                }
                max_residual = max_residual.max(residual);
                coords.push([iu as i64, iv as i64]);
            }
            if !okay {
                continue;
            }

            let mut unique = coords.clone();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() != points.len() {
                continue;
            }

            let min_u = coords.iter().map(|x| x[0]).min()?;
            let max_u = coords.iter().map(|x| x[0]).max()?;
            let min_v = coords.iter().map(|x| x[1]).min()?;
            let max_v = coords.iter().map(|x| x[1]).max()?;
            let nu = max_u - min_u + 1;
            let nv = max_v - min_v + 1;
            if nu <= 1 || nv <= 1 {
                continue;
            }
            let fill_ratio = points.len() as f64 / (nu * nv) as f64;
            let origin = add(base, add(scale(a, min_u as f64), scale(b, min_v as f64)));
            for coord in &mut coords {
                coord[0] -= min_u;
                coord[1] -= min_v;
            }
            let fit = Fit {
                dimension: 2,
                origin,
                basis: vec![a, b],
                pitch: vec![norm(a), norm(b)],
                occupancy: coords,
                grid_shape: vec![nu as usize, nv as usize],
                fill_ratio,
                max_residual_mm: max_residual,
            };

            match &best {
                None => best = Some((fill_ratio, cell_area, fit)),
                Some((bf, ba, _))
                    if fill_ratio > *bf + 1.0e-12
                        || ((fill_ratio - *bf).abs() <= 1.0e-12 && cell_area < *ba) =>
                {
                    best = Some((fill_ratio, cell_area, fit));
                }
                _ => {}
            }
        }
    }

    best.map(|(_, _, fit)| fit)
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

fn style_records_by_target(entities: &[EntityInstance]) -> HashMap<u64, Vec<(u64, Vec<u64>)>> {
    let mut out: HashMap<u64, Vec<(u64, Vec<u64>)>> = HashMap::new();
    for entity in entities {
        let id = entity_id(entity);
        let Some(record) = simple_record(entity) else {
            continue;
        };
        if record.name != "STYLED_ITEM" {
            continue;
        }
        let Parameter::List(params) = &record.parameter else {
            continue;
        };
        if params.len() != 3 {
            continue;
        }
        let Some(target) = entity_ref_value(&params[2]) else {
            continue;
        };
        let Some(assignments) = entity_ref_list(&params[1]) else {
            continue;
        };
        out.entry(target).or_default().push((id, assignments));
    }
    out
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
        let scaled = (value / ORIENTATION_Q).round();
        if !scaled.is_finite() || scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
            return None;
        }
        out[idx] = scaled as i64;
    }
    Some(out)
}

fn canonicalize_vector(v: &mut [f64; 3]) {
    for value in *v {
        if value.abs() > 1.0e-12 {
            if value < 0.0 {
                *v = scale(*v, -1.0);
            }
            break;
        }
    }
}

const fn entity_ref(id: u64) -> Parameter {
    Parameter::Ref(Name::Entity(id))
}

fn entity_ref_list(param: &Parameter) -> Option<Vec<u64>> {
    let Parameter::List(items) = param else {
        return None;
    };
    items.iter().map(entity_ref_value).collect()
}

fn entity_ref_map(entities: &[EntityInstance]) -> HashMap<u64, Vec<u64>> {
    let mut out = HashMap::new();
    for entity in entities {
        let id = entity_id(entity);
        let mut refs = Vec::new();
        visit_entity_refs(entity, &mut |child| refs.push(child));
        out.insert(id, refs);
    }
    out
}

fn inbound_map(refs: &HashMap<u64, Vec<u64>>) -> HashMap<u64, Vec<u64>> {
    let mut out: HashMap<u64, Vec<u64>> = HashMap::new();
    for (&parent, children) in refs {
        for &child in children {
            out.entry(child).or_default().push(parent);
        }
    }
    out
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

fn push_simple(
    entities: &mut Vec<EntityInstance>,
    next_id: &mut u64,
    name: &str,
    params: Vec<Parameter>,
) -> u64 {
    let id = *next_id;
    *next_id += 1;
    entities.push(EntityInstance::Simple {
        id,
        record: Record {
            name: name.to_string(),
            parameter: Parameter::List(params),
        },
    });
    id
}

const fn numeric_value(param: &Parameter) -> Option<f64> {
    match param {
        Parameter::Integer(v) => Some(*v as f64),
        Parameter::Real(v) => Some(*v),
        _ => None,
    }
}

fn numeric_list(param: &Parameter) -> Option<Vec<f64>> {
    let Parameter::List(items) = param else {
        return None;
    };
    items.iter().map(numeric_value).collect()
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[2].mul_add(b[2], a[1].mul_add(b[1], a[0] * b[0]))
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[2].mul_add(-b[1], a[1] * b[2]),
        a[0].mul_add(-b[2], a[2] * b[0]),
        a[1].mul_add(-b[0], a[0] * b[1]),
    ]
}

fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

fn normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = norm(v);
    if !n.is_finite() || n <= 1.0e-15 {
        return None;
    }
    Some(scale(v, 1.0 / n))
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
