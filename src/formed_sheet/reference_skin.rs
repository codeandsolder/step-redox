use super::{
    DIR_TOL, GEOM_TOL_MM, SheetCylinderPair, SheetPatchAdjacency, SheetPatchId, SheetPlanePair,
    list_params, normalize, patch_linear_index,
};
use crate::math3::{add, cross, distance, dot, mul, norm, point_to_unit_line_distance, sub};
use crate::step_entities::{
    cartesian_point, direction_components as direction, enumeration_bool, number as numeric_value,
    vertex_point,
};
use crate::step_graph::{entity_ref_value, simple_record};
use ruststep::ast::{EntityInstance, Parameter};
use serde::Serialize;
use std::collections::{BTreeSet, HashMap, VecDeque};

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SheetReferenceSkin {
    pub face_ids: Vec<u64>,
    pub patches: Vec<SheetReferencePatch>,
    pub unique_edges: usize,
    pub boundary_loops: Vec<SheetReferenceLoop>,
    pub internal_seams: Vec<SheetReferenceCurve>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SheetReferencePatch {
    pub patch: SheetPatchId,
    pub source_face_id: u64,
    pub paired_face_id: u64,
    pub surface: SheetReferenceSurface,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SheetReferenceSurface {
    Plane {
        normal: [f64; 3],
        offset_mm: f64,
        paired_offset_mm: f64,
    },
    Cylinder {
        axis_origin_mm: [f64; 3],
        axis: [f64; 3],
        radius_mm: f64,
        paired_radius_mm: f64,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SheetReferenceLoop {
    pub curves: Vec<SheetReferenceCurve>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SheetReferenceCurve {
    Line {
        source_edge_id: u64,
        patches: Vec<SheetPatchId>,
        start_mm: [f64; 3],
        end_mm: [f64; 3],
    },
    CircleArc {
        source_edge_id: u64,
        patches: Vec<SheetPatchId>,
        center_mm: [f64; 3],
        normal: [f64; 3],
        radius_mm: f64,
        start_mm: [f64; 3],
        end_mm: [f64; 3],
        sweep_angle_rad: f64,
    },
    EllipseArc {
        source_edge_id: u64,
        patches: Vec<SheetPatchId>,
        center_mm: [f64; 3],
        u_axis_mm: [f64; 3],
        v_axis_mm: [f64; 3],
        start_angle_rad: f64,
        sweep_angle_rad: f64,
        start_mm: [f64; 3],
        end_mm: [f64; 3],
    },
}

pub(super) fn recover_reference_skin(
    plane_pairs: &[SheetPlanePair],
    cylinder_pairs: &[SheetCylinderPair],
    adjacency: &[SheetPatchAdjacency],
    face_edges: &HashMap<u64, Vec<u64>>,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<SheetReferenceSkin> {
    let first = select_skin_faces(plane_pairs, cylinder_pairs, adjacency)?;
    let second = complementary_skin(&first, plane_pairs, cylinder_pairs)?;

    [first, second]
        .into_iter()
        .filter_map(|skin| {
            build_reference_skin(
                &skin,
                plane_pairs,
                cylinder_pairs,
                face_edges,
                entities,
                index,
            )
        })
        .min_by_key(|skin| {
            (
                skin.unique_edges,
                skin.boundary_loops
                    .iter()
                    .map(|loop_| loop_.curves.len())
                    .sum::<usize>(),
                skin.internal_seams.len(),
            )
        })
}

fn complementary_skin(
    skin: &[(SheetPatchId, usize, u64)],
    plane_pairs: &[SheetPlanePair],
    cylinder_pairs: &[SheetCylinderPair],
) -> Option<Vec<(SheetPatchId, usize, u64)>> {
    skin.iter()
        .map(|&(patch, side, _)| {
            if side > 1 {
                return None;
            }
            let other_side = 1 - side;
            let face = match patch {
                SheetPatchId::Plane(index) => {
                    let pair = plane_pairs.get(index)?;
                    [pair.negative_face_id, pair.positive_face_id][other_side]
                }
                SheetPatchId::Cylinder(index) => {
                    let pair = cylinder_pairs.get(index)?;
                    [pair.inner_face_id, pair.outer_face_id][other_side]
                }
            };
            Some((patch, other_side, face))
        })
        .collect()
}

fn build_reference_skin(
    skin: &[(SheetPatchId, usize, u64)],
    plane_pairs: &[SheetPlanePair],
    cylinder_pairs: &[SheetCylinderPair],
    face_edges: &HashMap<u64, Vec<u64>>,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<SheetReferenceSkin> {
    let mut face_to_patch = HashMap::<u64, (SheetPatchId, usize)>::new();
    let mut patch_side = HashMap::<SheetPatchId, usize>::new();
    let mut patches = Vec::with_capacity(skin.len());
    let mut face_ids = Vec::with_capacity(skin.len());

    for &(patch, side, face_id) in skin {
        if side > 1
            || face_to_patch.insert(face_id, (patch, side)).is_some()
            || patch_side.insert(patch, side).is_some()
        {
            return None;
        }
        face_ids.push(face_id);
        let (paired_face_id, surface) = match patch {
            SheetPatchId::Plane(patch_index) => {
                let pair = plane_pairs.get(patch_index)?;
                let offsets = [
                    pair.mid_offset_mm - pair.separation_mm * 0.5,
                    pair.mid_offset_mm + pair.separation_mm * 0.5,
                ];
                let faces = [pair.negative_face_id, pair.positive_face_id];
                (
                    faces[1 - side],
                    SheetReferenceSurface::Plane {
                        normal: pair.normal,
                        offset_mm: offsets[side],
                        paired_offset_mm: offsets[1 - side],
                    },
                )
            }
            SheetPatchId::Cylinder(patch_index) => {
                let pair = cylinder_pairs.get(patch_index)?;
                let faces = [pair.inner_face_id, pair.outer_face_id];
                let radii = [pair.inner_radius_mm, pair.outer_radius_mm];
                (
                    faces[1 - side],
                    SheetReferenceSurface::Cylinder {
                        axis_origin_mm: pair.axis_origin_mm,
                        axis: pair.axis,
                        radius_mm: radii[side],
                        paired_radius_mm: radii[1 - side],
                    },
                )
            }
        };
        patches.push(SheetReferencePatch {
            patch,
            source_face_id: face_id,
            paired_face_id,
            surface,
        });
    }

    let mut edge_patches = HashMap::<u64, Vec<SheetPatchId>>::new();
    for &(patch, _, face_id) in skin {
        for &edge in face_edges.get(&face_id)? {
            edge_patches.entry(edge).or_default().push(patch);
        }
    }
    if edge_patches
        .values()
        .any(|patches| patches.is_empty() || patches.len() > 2)
    {
        return None;
    }

    let boundary_edges = edge_patches
        .iter()
        .filter_map(|(&edge, patches)| (patches.len() == 1).then_some(edge))
        .collect::<BTreeSet<_>>();
    let internal_edges = edge_patches
        .iter()
        .filter_map(|(&edge, patches)| (patches.len() == 2).then_some(edge))
        .collect::<BTreeSet<_>>();
    if boundary_edges.is_empty() {
        return None;
    }

    let mut vertex_edges = HashMap::<u64, Vec<u64>>::new();
    for &edge in &boundary_edges {
        let (start, end, _, _) = edge_curve_info(edge, entities, index)?;
        vertex_edges.entry(start).or_default().push(edge);
        vertex_edges.entry(end).or_default().push(edge);
    }
    if vertex_edges.values().any(|edges| edges.len() != 2) {
        return None;
    }

    let mut unused = boundary_edges.clone();
    let mut boundary_loops = Vec::new();
    while let Some(&first_edge) = unused.iter().next() {
        let (first_start, first_end, _, _) = edge_curve_info(first_edge, entities, index)?;
        let start_vertex = first_start.min(first_end);
        let mut current_vertex = start_vertex;
        let mut current_edge = first_edge;
        let mut curves = Vec::new();

        loop {
            if !unused.remove(&current_edge) {
                return None;
            }
            let (edge_start, edge_end, _, _) = edge_curve_info(current_edge, entities, index)?;
            let next_vertex = if edge_start == current_vertex {
                edge_end
            } else if edge_end == current_vertex {
                edge_start
            } else {
                return None;
            };
            curves.push(reference_curve(
                current_edge,
                current_vertex,
                next_vertex,
                edge_patches.get(&current_edge)?,
                &patch_side,
                cylinder_pairs,
                entities,
                index,
            )?);

            current_vertex = next_vertex;
            if current_vertex == start_vertex {
                break;
            }
            let next = vertex_edges
                .get(&current_vertex)?
                .iter()
                .copied()
                .filter(|edge| unused.contains(edge))
                .collect::<Vec<_>>();
            if next.len() != 1 {
                return None;
            }
            current_edge = next[0];
        }
        boundary_loops.push(SheetReferenceLoop { curves });
    }
    boundary_loops.sort_by_key(|loop_| std::cmp::Reverse(loop_.curves.len()));

    let mut internal_seams = Vec::with_capacity(internal_edges.len());
    for edge in internal_edges {
        let (start, end, _, _) = edge_curve_info(edge, entities, index)?;
        internal_seams.push(reference_curve(
            edge,
            start,
            end,
            edge_patches.get(&edge)?,
            &patch_side,
            cylinder_pairs,
            entities,
            index,
        )?);
    }
    internal_seams.sort_by_key(reference_curve_source_edge);

    Some(SheetReferenceSkin {
        face_ids,
        patches,
        unique_edges: edge_patches.len(),
        boundary_loops,
        internal_seams,
    })
}

fn reference_curve_source_edge(curve: &SheetReferenceCurve) -> u64 {
    match curve {
        SheetReferenceCurve::Line { source_edge_id, .. }
        | SheetReferenceCurve::CircleArc { source_edge_id, .. }
        | SheetReferenceCurve::EllipseArc { source_edge_id, .. } => *source_edge_id,
    }
}

fn reference_curve(
    edge_id: u64,
    traversal_start: u64,
    traversal_end: u64,
    patches: &[SheetPatchId],
    patch_side: &HashMap<SheetPatchId, usize>,
    cylinder_pairs: &[SheetCylinderPair],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<SheetReferenceCurve> {
    let (edge_start, edge_end, curve_id, same_sense) = edge_curve_info(edge_id, entities, index)?;
    let start_mm = vertex_point(traversal_start, entities, index)?;
    let end_mm = vertex_point(traversal_end, entities, index)?;
    let traversal_forward = traversal_start == edge_start && traversal_end == edge_end;
    let traversal_reverse = traversal_start == edge_end && traversal_end == edge_start;
    if !traversal_forward && !traversal_reverse {
        return None;
    }
    let parameter_forward = if traversal_forward {
        same_sense
    } else {
        !same_sense
    };

    match entities.get(*index.get(&curve_id)?)? {
        EntityInstance::Simple { record, .. } if record.name == "LINE" => {
            Some(SheetReferenceCurve::Line {
                source_edge_id: edge_id,
                patches: patches.to_vec(),
                start_mm,
                end_mm,
            })
        }
        EntityInstance::Simple { record, .. } if record.name == "CIRCLE" => {
            let (center_mm, normal, x_direction, radius_mm) =
                circle_support(curve_id, entities, index)?;
            let y_direction = normalize(cross(normal, x_direction))?;
            if (distance(start_mm, center_mm) - radius_mm).abs() > GEOM_TOL_MM * 5.0
                || (distance(end_mm, center_mm) - radius_mm).abs() > GEOM_TOL_MM * 5.0
            {
                return None;
            }
            let start_angle = circle_angle(start_mm, center_mm, x_direction, y_direction)?;
            let end_angle = circle_angle(end_mm, center_mm, x_direction, y_direction)?;
            Some(SheetReferenceCurve::CircleArc {
                source_edge_id: edge_id,
                patches: patches.to_vec(),
                center_mm,
                normal,
                radius_mm,
                start_mm,
                end_mm,
                sweep_angle_rad: circle_sweep(start_angle, end_angle, parameter_forward),
            })
        }
        EntityInstance::Complex { .. } => reference_ellipse_seam(
            edge_id,
            curve_id,
            traversal_start,
            traversal_end,
            patches,
            patch_side,
            cylinder_pairs,
            entities,
            index,
        ),
        _ => None,
    }
}

fn reference_ellipse_seam(
    edge_id: u64,
    curve_id: u64,
    traversal_start: u64,
    traversal_end: u64,
    patches: &[SheetPatchId],
    patch_side: &HashMap<SheetPatchId, usize>,
    cylinder_pairs: &[SheetCylinderPair],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<SheetReferenceCurve> {
    if patches.len() != 2 {
        return None;
    }
    let SheetPatchId::Cylinder(first_index) = patches[0] else {
        return None;
    };
    let SheetPatchId::Cylinder(second_index) = patches[1] else {
        return None;
    };
    let first_pair = cylinder_pairs.get(first_index)?;
    let second_pair = cylinder_pairs.get(second_index)?;
    let first_side = *patch_side.get(&patches[0])?;
    let second_side = *patch_side.get(&patches[1])?;
    let first_radius = [first_pair.inner_radius_mm, first_pair.outer_radius_mm][first_side];
    let second_radius = [second_pair.inner_radius_mm, second_pair.outer_radius_mm][second_side];
    if (first_radius - second_radius).abs() > GEOM_TOL_MM {
        return None;
    }
    let radius_mm = f64::midpoint(first_radius, second_radius);
    let first_axis = first_pair.axis;
    let second_axis = second_pair.axis;
    if dot(first_axis, second_axis).abs() > DIR_TOL * 100.0 {
        return None;
    }
    let (center_mm, axis_gap) = closest_axis_intersection(
        first_pair.axis_origin_mm,
        first_axis,
        second_pair.axis_origin_mm,
        second_axis,
    )?;
    if axis_gap > GEOM_TOL_MM {
        return None;
    }

    let start_mm = vertex_point(traversal_start, entities, index)?;
    let end_mm = vertex_point(traversal_end, entities, index)?;
    let first_coordinate = dot(sub(start_mm, center_mm), first_axis);
    let second_coordinate = dot(sub(start_mm, center_mm), second_axis);
    if first_coordinate.abs() <= GEOM_TOL_MM || second_coordinate.abs() <= GEOM_TOL_MM {
        return None;
    }
    let branch_sign = if first_coordinate * second_coordinate >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let u_axis_mm = mul(add(mul(first_axis, branch_sign), second_axis), radius_mm);
    let v_axis_mm = mul(normalize(cross(first_axis, second_axis))?, radius_mm);

    let start_angle = ellipse_angle(start_mm, center_mm, u_axis_mm, v_axis_mm)?;
    let end_angle = ellipse_angle(end_mm, center_mm, u_axis_mm, v_axis_mm)?;
    if distance(
        ellipse_point(center_mm, u_axis_mm, v_axis_mm, start_angle),
        start_mm,
    ) > GEOM_TOL_MM * 5.0
        || distance(
            ellipse_point(center_mm, u_axis_mm, v_axis_mm, end_angle),
            end_mm,
        ) > GEOM_TOL_MM * 5.0
    {
        return None;
    }

    let samples = rational_single_span_samples(curve_id, entities, index)?;
    for sample in &samples {
        if (point_to_unit_line_distance(*sample, first_pair.axis_origin_mm, first_axis)
            - first_radius)
            .abs()
            > GEOM_TOL_MM * 5.0
            || (point_to_unit_line_distance(*sample, second_pair.axis_origin_mm, second_axis)
                - second_radius)
                .abs()
                > GEOM_TOL_MM * 5.0
        {
            return None;
        }
        let u = dot(sub(*sample, center_mm), first_axis);
        let v = dot(sub(*sample, center_mm), second_axis);
        if (u - branch_sign * v).abs() > GEOM_TOL_MM * 5.0 {
            return None;
        }
    }

    let midpoint = samples
        .get(1)
        .copied()
        .or_else(|| samples.first().copied())?;
    let midpoint_angle = ellipse_angle(midpoint, center_mm, u_axis_mm, v_axis_mm)?;
    let sweep_angle_rad = ellipse_arc_sweep(start_angle, end_angle, midpoint_angle)?;

    Some(SheetReferenceCurve::EllipseArc {
        source_edge_id: edge_id,
        patches: patches.to_vec(),
        center_mm,
        u_axis_mm,
        v_axis_mm,
        start_angle_rad: start_angle,
        sweep_angle_rad,
        start_mm,
        end_mm,
    })
}

pub(super) fn closest_axis_intersection(
    first_origin: [f64; 3],
    first_axis: [f64; 3],
    second_origin: [f64; 3],
    second_axis: [f64; 3],
) -> Option<([f64; 3], f64)> {
    let b = dot(first_axis, second_axis);
    let cross_axis = cross(first_axis, second_axis);
    let denominator = dot(cross_axis, cross_axis);
    if !denominator.is_finite() || denominator <= DIR_TOL {
        return None;
    }
    let delta = sub(first_origin, second_origin);
    let d = dot(first_axis, delta);
    let e = dot(second_axis, delta);
    let first_t = b.mul_add(e, -d) / denominator;
    let second_t = (-b).mul_add(d, e) / denominator;
    let first_point = add(first_origin, mul(first_axis, first_t));
    let second_point = add(second_origin, mul(second_axis, second_t));
    let midpoint = [
        f64::midpoint(first_point[0], second_point[0]),
        f64::midpoint(first_point[1], second_point[1]),
        f64::midpoint(first_point[2], second_point[2]),
    ];
    Some((midpoint, distance(first_point, second_point)))
}

fn ellipse_point(center: [f64; 3], u_axis: [f64; 3], v_axis: [f64; 3], angle: f64) -> [f64; 3] {
    add(
        center,
        add(mul(u_axis, angle.cos()), mul(v_axis, angle.sin())),
    )
}

fn ellipse_angle(
    point: [f64; 3],
    center: [f64; 3],
    u_axis: [f64; 3],
    v_axis: [f64; 3],
) -> Option<f64> {
    let relative = sub(point, center);
    let u_norm_sq = dot(u_axis, u_axis);
    let v_norm_sq = dot(v_axis, v_axis);
    if u_norm_sq <= DIR_TOL || v_norm_sq <= DIR_TOL {
        return None;
    }
    let cosine = dot(relative, u_axis) / u_norm_sq;
    let sine = dot(relative, v_axis) / v_norm_sq;
    Some(sine.atan2(cosine))
}

fn ellipse_arc_sweep(start: f64, end: f64, sample: f64) -> Option<f64> {
    let tau = std::f64::consts::TAU;
    let positive = (end - start).rem_euclid(tau);
    let negative = positive - tau;
    let positive_progress = (sample - start).rem_euclid(tau);
    let negative_progress = -((start - sample).rem_euclid(tau));
    let on_positive = positive_progress <= positive + 1.0e-8;
    let on_negative = negative_progress >= negative - 1.0e-8;
    match (on_positive, on_negative) {
        (true, false) => Some(positive),
        (false, true) => Some(negative),
        (true, true) => {
            if positive.abs() <= negative.abs() {
                Some(positive)
            } else {
                Some(negative)
            }
        }
        (false, false) => None,
    }
}

fn rational_single_span_samples(
    curve_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<[f64; 3]>> {
    let EntityInstance::Complex { subsuper, .. } = entities.get(*index.get(&curve_id)?)? else {
        return None;
    };
    let records = &subsuper.0;
    let record = |name: &str| {
        let mut matches = records.iter().filter(|record| record.name == name);
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    };
    let bspline = record("B_SPLINE_CURVE")?;
    let bspline_params = list_params(bspline)?;
    let degree = match bspline_params.first()? {
        Parameter::Integer(value) if *value >= 1 => *value as usize,
        _ => return None,
    };
    let Parameter::List(pole_params) = bspline_params.get(1)? else {
        return None;
    };
    if pole_params.len() != degree + 1 {
        return None;
    }
    let poles = pole_params
        .iter()
        .map(|parameter| {
            let point_id = entity_ref_value(parameter)?;
            cartesian_point(point_id, entities, index)
        })
        .collect::<Option<Vec<_>>>()?;

    let knots = record("B_SPLINE_CURVE_WITH_KNOTS")?;
    let knot_params = list_params(knots)?;
    let Parameter::List(multiplicities) = knot_params.first()? else {
        return None;
    };
    if multiplicities.len() != 2
        || !multiplicities.iter().all(
            |parameter| matches!(parameter, Parameter::Integer(value) if *value == degree as i64 + 1),
        )
    {
        return None;
    }

    let rational = record("RATIONAL_B_SPLINE_CURVE")?;
    let rational_params = list_params(rational)?;
    let Parameter::List(weight_params) = rational_params.first()? else {
        return None;
    };
    if weight_params.len() != poles.len() {
        return None;
    }
    let weights = weight_params
        .iter()
        .map(|parameter| numeric_value(parameter).filter(|value| value.is_finite() && *value > 0.0))
        .collect::<Option<Vec<_>>>()?;

    [0.25, 0.5, 0.75]
        .into_iter()
        .map(|parameter| rational_bezier_point(&poles, &weights, parameter))
        .collect()
}

pub(super) fn rational_bezier_point(
    poles: &[[f64; 3]],
    weights: &[f64],
    parameter: f64,
) -> Option<[f64; 3]> {
    if poles.len() != weights.len()
        || poles.is_empty()
        || !parameter.is_finite()
        || !(0.0..=1.0).contains(&parameter)
        || poles.iter().flatten().any(|value| !value.is_finite())
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return None;
    }

    if parameter == 0.0 {
        return poles.first().copied();
    }
    if parameter == 1.0 {
        return poles.last().copied();
    }

    let weight_scale = weights.iter().copied().max_by(|a, b| a.total_cmp(b))?;
    let mut homogeneous = poles
        .iter()
        .zip(weights)
        .map(|(&pole, &weight)| {
            let scaled_weight = weight / weight_scale;
            [
                pole[0] * scaled_weight,
                pole[1] * scaled_weight,
                pole[2] * scaled_weight,
                scaled_weight,
            ]
        })
        .collect::<Vec<_>>();

    let complement = 1.0 - parameter;
    for level in 1..homogeneous.len() {
        for index in 0..homogeneous.len() - level {
            let left = homogeneous[index];
            let right = homogeneous[index + 1];
            homogeneous[index] =
                std::array::from_fn(|axis| complement.mul_add(left[axis], parameter * right[axis]));
        }
    }

    let denominator = homogeneous[0][3];
    if !denominator.is_finite() || denominator <= 0.0 {
        return None;
    }
    let point = [
        homogeneous[0][0] / denominator,
        homogeneous[0][1] / denominator,
        homogeneous[0][2] / denominator,
    ];
    point
        .iter()
        .all(|coordinate| coordinate.is_finite())
        .then_some(point)
}

fn select_skin_faces(
    plane_pairs: &[SheetPlanePair],
    cylinder_pairs: &[SheetCylinderPair],
    adjacency: &[SheetPatchAdjacency],
) -> Option<Vec<(SheetPatchId, usize, u64)>> {
    let plane_count = plane_pairs.len();
    let patch_count = plane_count + cylinder_pairs.len();
    if patch_count == 0 {
        return None;
    }
    let mut graph = vec![Vec::<(usize, bool)>::new(); patch_count];
    for edge in adjacency {
        let a = patch_linear_index(edge.patch_a, plane_count);
        let b = patch_linear_index(edge.patch_b, plane_count);
        if a >= patch_count || b >= patch_count {
            return None;
        }
        graph[a].push((b, edge.crossed_source_sides));
        graph[b].push((a, edge.crossed_source_sides));
    }

    let mut side = vec![None::<usize>; patch_count];
    side[0] = Some(0);
    let mut queue = VecDeque::from([0usize]);
    while let Some(patch) = queue.pop_front() {
        let current = side[patch]?;
        for &(neighbor, crossed) in &graph[patch] {
            let expected = current ^ usize::from(crossed);
            match side[neighbor] {
                Some(existing) if existing != expected => return None,
                Some(_) => {}
                None => {
                    side[neighbor] = Some(expected);
                    queue.push_back(neighbor);
                }
            }
        }
    }
    if side.iter().any(Option::is_none) {
        return None;
    }

    let mut out = Vec::with_capacity(patch_count);
    for (index, selected) in side.into_iter().enumerate() {
        let selected = selected?;
        if index < plane_count {
            let pair = &plane_pairs[index];
            let faces = [pair.negative_face_id, pair.positive_face_id];
            out.push((SheetPatchId::Plane(index), selected, faces[selected]));
        } else {
            let cylinder_index = index - plane_count;
            let pair = &cylinder_pairs[cylinder_index];
            let faces = [pair.inner_face_id, pair.outer_face_id];
            out.push((
                SheetPatchId::Cylinder(cylinder_index),
                selected,
                faces[selected],
            ));
        }
    }
    Some(out)
}

fn edge_curve_info(
    edge_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<(u64, u64, u64, bool)> {
    let edge = simple_record(entities.get(*index.get(&edge_id)?)?)?;
    if edge.name != "EDGE_CURVE" {
        return None;
    }
    let params = list_params(edge)?;
    let start = params.get(1).and_then(entity_ref_value)?;
    let end = params.get(2).and_then(entity_ref_value)?;
    let curve = params.get(3).and_then(entity_ref_value)?;
    let same_sense = enumeration_bool(params.get(4)?)?;
    Some((start, end, curve, same_sense))
}

fn circle_support(
    curve_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<([f64; 3], [f64; 3], [f64; 3], f64)> {
    let circle = simple_record(entities.get(*index.get(&curve_id)?)?)?;
    if circle.name != "CIRCLE" {
        return None;
    }
    let params = list_params(circle)?;
    let placement_id = params.get(1).and_then(entity_ref_value)?;
    let radius_mm = numeric_value(params.get(2)?)?;
    let (center, normal, x_direction) = placement_frame(placement_id, entities, index)?;
    Some((center, normal, x_direction, radius_mm))
}

fn placement_frame(
    placement_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<([f64; 3], [f64; 3], [f64; 3])> {
    let placement = simple_record(entities.get(*index.get(&placement_id)?)?)?;
    if placement.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let params = list_params(placement)?;
    let origin_id = params.get(1).and_then(entity_ref_value)?;
    let axis_id = params.get(2).and_then(entity_ref_value)?;
    let reference_id = params.get(3).and_then(entity_ref_value)?;
    let origin = cartesian_point(origin_id, entities, index)?;
    let normal = normalize(direction(axis_id, entities, index)?)?;
    let reference = direction(reference_id, entities, index)?;
    let projected = sub(reference, mul(normal, dot(reference, normal)));
    let x_direction = normalize(projected)?;
    Some((origin, normal, x_direction))
}

fn circle_angle(
    point: [f64; 3],
    center: [f64; 3],
    x_direction: [f64; 3],
    y_direction: [f64; 3],
) -> Option<f64> {
    let radial = sub(point, center);
    if norm(radial) <= GEOM_TOL_MM * 0.01 {
        return None;
    }
    Some(dot(radial, y_direction).atan2(dot(radial, x_direction)))
}

fn circle_sweep(start: f64, end: f64, forward: bool) -> f64 {
    let tau = std::f64::consts::TAU;
    if forward {
        (end - start).rem_euclid(tau)
    } else {
        -((start - end).rem_euclid(tau))
    }
}
