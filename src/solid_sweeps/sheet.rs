use super::{RecoveredSweepSegment, prove_closed_source_topology};
use crate::brep::{self, CurveSupport};
use crate::formed_sheet::{
    self, SheetCylinderPair, SheetPatchId, SheetPlanePair, SheetReferenceCurve,
    SheetReferenceSurface,
};
use crate::math3::{
    add, canonical_direction, cross, distance, dot, mul, norm, normalize,
    point_to_unit_line_distance, sub,
};
use crate::step_graph::build_index;
use ruststep::ast::EntityInstance;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

const SHEET_TOL_MM: f64 = 1.0e-5;
const DIR_TOL: f64 = 1.0e-9;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecoveredRectangularTaperTip {
    pub source_face_ids: [u64; 5],
    pub base_y_mm: f64,
    pub end_y_mm: f64,
    pub end_x_range_mm: [f64; 2],
    pub end_z_range_mm: [f64; 2],
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecoveredPlanarSweepLeg {
    pub source_face_ids: [u64; 2],
    pub origin_mm: [f64; 3],
    pub x_axis: [f64; 3],
    pub y_axis: [f64; 3],
    pub z_axis: [f64; 3],
    pub profile_points_mm: Vec<[f64; 2]>,
    pub depth_mm: f64,
    pub terminal_taper: Option<RecoveredRectangularTaperTip>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecoveredOpenRectangularSweep {
    pub solid_id: u64,
    pub source_face_ids: Vec<u64>,
    pub legs: [RecoveredPlanarSweepLeg; 2],
    pub profile_width_mm: f64,
    pub profile_thickness_mm: f64,
    pub bend_path_start_mm: [f64; 3],
    pub bend_profile_x_axis: [f64; 3],
    pub bend_profile_y_axis: [f64; 3],
    pub bend_path_tangent: [f64; 3],
    pub bend_segments: Vec<RecoveredSweepSegment>,
    pub max_residual_mm: f64,
}

#[derive(Debug, Clone)]
struct CylinderSection {
    width_mm: f64,
    axis_center_mm: [f64; 3],
}

#[derive(Debug, Clone, Copy)]
struct SkinTangent {
    inward_direction: [f64; 3],
    straight_length_mm: f64,
}

#[must_use]
pub fn detect_open_rectangular_sweeps(
    entities: &[EntityInstance],
) -> Vec<RecoveredOpenRectangularSweep> {
    let index = build_index(entities);
    let mut out = formed_sheet::detect_formed_sheet_evidence_with_min_pairs(entities, 1, 1)
        .iter()
        .filter_map(|evidence| detect_one(evidence, entities, &index))
        .collect::<Vec<_>>();
    out.sort_by_key(|sweep| sweep.solid_id);
    out
}

fn detect_one(
    evidence: &formed_sheet::FormedSheetEvidence,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<RecoveredOpenRectangularSweep> {
    if evidence.cylinder_pairs.len() != 1 || !evidence.thickness_mm.is_finite() {
        return None;
    }
    let bend = &evidence.cylinder_pairs[0];
    let axis = canonical_direction(normalize(bend.axis, 0.0)?);
    let thickness_mm = bend.outer_radius_mm - bend.inner_radius_mm;
    if thickness_mm <= SHEET_TOL_MM || !near(thickness_mm, evidence.thickness_mm) {
        return None;
    }

    let plane_pairs = evidence
        .plane_pairs
        .iter()
        .filter(|pair| {
            dot(pair.normal, axis).abs() <= DIR_TOL && near(pair.separation_mm, thickness_mm)
        })
        .cloned()
        .collect::<Vec<_>>();
    if plane_pairs.len() != 2
        || dot(plane_pairs[0].normal, plane_pairs[1].normal).abs() >= 1.0 - 1.0e-7
    {
        return None;
    }

    let mut source_face_ids = brep::solid_face_ids(evidence.solid_id, entities, index)?;
    source_face_ids.sort_unstable();
    prove_closed_source_topology(&source_face_ids, entities, index)?;

    let reduced_skin = formed_sheet::recover_reference_skin_for_pairs(
        evidence.solid_id,
        &plane_pairs,
        std::slice::from_ref(bend),
        entities,
    )?;
    if reduced_skin.boundary_loops.len() != 1 || reduced_skin.patches.len() != 3 {
        return None;
    }

    let section = recover_cylinder_section(bend, axis, entities, index)?;
    let mid_radius_mm = bend.mid_radius_mm;
    if mid_radius_mm <= thickness_mm * 0.5 || !mid_radius_mm.is_finite() {
        return None;
    }

    let mut max_residual_mm: f64 = 0.0;
    let mut tangent_points = Vec::with_capacity(2);
    let mut radial_directions = Vec::with_capacity(2);
    let mut tangent_proofs = Vec::with_capacity(2);
    for (plane_index, pair) in plane_pairs.iter().enumerate() {
        let normal = normalize(pair.normal, 0.0)?;
        let signed_radius = pair.mid_offset_mm - dot(normal, section.axis_center_mm);
        max_residual_mm = max_residual_mm.max((signed_radius.abs() - mid_radius_mm).abs());
        if max_residual_mm > SHEET_TOL_MM {
            return None;
        }
        let radial = mul(normal, signed_radius);
        let radial_direction = normalize(radial, SHEET_TOL_MM)?;
        let tangent = add(section.axis_center_mm, radial);
        let tangent_proof = prove_skin_tangent(
            &reduced_skin,
            plane_index,
            pair,
            axis,
            section.width_mm,
            tangent,
        )?;
        tangent_points.push(tangent);
        radial_directions.push(radial_direction);
        tangent_proofs.push(tangent_proof);
    }

    let (start, end, bend_angle_rad, start_motion, end_motion) =
        choose_bend_order(axis, &radial_directions, &tangent_proofs)?;
    prove_cylinder_sector(
        bend,
        axis,
        section.width_mm,
        &radial_directions,
        bend_angle_rad.abs(),
        entities,
        index,
    )?;

    let mut leg0 = recover_planar_leg(&plane_pairs[0], axis, entities, index)?;
    let mut leg1 = recover_planar_leg(&plane_pairs[1], axis, entities, index)?;
    leg0.terminal_taper = recover_rectangular_taper_tip(
        &leg0,
        tangent_proofs[0].inward_direction,
        &source_face_ids,
        entities,
        index,
    );
    leg1.terminal_taper = recover_rectangular_taper_tip(
        &leg1,
        tangent_proofs[1].inward_direction,
        &source_face_ids,
        entities,
        index,
    );
    let taper_face_ids = leg0
        .terminal_taper
        .iter()
        .chain(leg1.terminal_taper.iter())
        .flat_map(|tip| tip.source_face_ids)
        .collect::<HashSet<_>>();
    prove_source_supports_covered(
        &source_face_ids,
        [&leg0, &leg1],
        bend,
        axis,
        section.axis_center_mm,
        section.width_mm,
        &taper_face_ids,
        entities,
        index,
    )?;

    let overlap_mm = section
        .width_mm
        .min(thickness_mm)
        .min(tangent_proofs[start].straight_length_mm)
        .min(tangent_proofs[end].straight_length_mm)
        * 0.25;
    if !overlap_mm.is_finite() || overlap_mm <= SHEET_TOL_MM * 4.0 {
        return None;
    }
    let bend_profile_x_axis = axis;
    let bend_path_tangent = start_motion;
    let bend_profile_y_axis =
        normalize(cross(bend_path_tangent, bend_profile_x_axis), SHEET_TOL_MM)?;
    let bend_path_start_mm = add(
        tangent_points[start],
        mul(tangent_proofs[start].inward_direction, overlap_mm),
    );
    let bend_segments = vec![
        RecoveredSweepSegment::Translation {
            vector_mm: mul(tangent_proofs[start].inward_direction, -overlap_mm),
        },
        RecoveredSweepSegment::Rotation {
            axis_origin_mm: bend.axis_origin_mm,
            axis_direction: axis,
            angle_rad: bend_angle_rad,
        },
        RecoveredSweepSegment::Translation {
            vector_mm: mul(tangent_proofs[end].inward_direction, overlap_mm),
        },
    ];
    if dot(end_motion, tangent_proofs[end].inward_direction) < 1.0 - 1.0e-8 {
        return None;
    }

    Some(RecoveredOpenRectangularSweep {
        solid_id: evidence.solid_id,
        source_face_ids,
        legs: [leg0, leg1],
        profile_width_mm: section.width_mm,
        profile_thickness_mm: thickness_mm,
        bend_path_start_mm,
        bend_profile_x_axis,
        bend_profile_y_axis,
        bend_path_tangent,
        bend_segments,
        max_residual_mm,
    })
}

#[derive(Debug, Clone, Copy)]
struct ExpectedPlane {
    normal: [f64; 3],
    offset_mm: f64,
}

fn prove_source_supports_covered(
    face_ids: &[u64],
    legs: [&RecoveredPlanarSweepLeg; 2],
    bend: &SheetCylinderPair,
    axis: [f64; 3],
    axis_center_mm: [f64; 3],
    width_mm: f64,
    extra_plane_face_ids: &HashSet<u64>,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<()> {
    let mut planes = Vec::new();
    for leg in legs {
        push_expected_plane(&mut planes, leg.z_axis, leg.origin_mm)?;
        push_expected_plane(
            &mut planes,
            leg.z_axis,
            add(leg.origin_mm, mul(leg.z_axis, leg.depth_mm)),
        )?;
        for i in 0..leg.profile_points_mm.len() {
            let a = leg.profile_points_mm[i];
            let b = leg.profile_points_mm[(i + 1) % leg.profile_points_mm.len()];
            let edge = add(mul(leg.x_axis, b[0] - a[0]), mul(leg.y_axis, b[1] - a[1]));
            let normal = normalize(cross(edge, leg.z_axis), SHEET_TOL_MM)?;
            let point = add(
                leg.origin_mm,
                add(mul(leg.x_axis, a[0]), mul(leg.y_axis, a[1])),
            );
            push_expected_plane(&mut planes, normal, point)?;
        }
    }
    for sign in [-0.5, 0.5] {
        push_expected_plane(
            &mut planes,
            axis,
            add(axis_center_mm, mul(axis, width_mm * sign)),
        )?;
    }

    for &face_id in face_ids {
        let surface_id = brep::face_surface(face_id, entities, index)?;
        match brep::surface_support(surface_id, entities, index) {
            brep::SurfaceSupport::Plane(plane) => {
                let source = canonical_plane(plane.normal, plane.origin_mm)?;
                if !planes.iter().any(|expected| same_plane(source, *expected))
                    && !extra_plane_face_ids.contains(&face_id)
                {
                    return None;
                }
            }
            brep::SurfaceSupport::Cylinder(cylinder) => {
                let radius_matches = near(cylinder.radius_mm, bend.inner_radius_mm)
                    || near(cylinder.radius_mm, bend.outer_radius_mm);
                let cylinder_axis = normalize(cylinder.axis, 0.0)?;
                if !radius_matches
                    || dot(cylinder_axis, axis).abs() < 1.0 - DIR_TOL
                    || point_to_unit_line_distance(
                        cylinder.axis_origin_mm,
                        bend.axis_origin_mm,
                        axis,
                    ) > SHEET_TOL_MM
                {
                    return None;
                }
            }
            _ => return None,
        }
    }
    Some(())
}

fn push_expected_plane(
    planes: &mut Vec<ExpectedPlane>,
    normal: [f64; 3],
    point: [f64; 3],
) -> Option<()> {
    let plane = canonical_plane(normal, point)?;
    if !planes.iter().any(|existing| same_plane(*existing, plane)) {
        planes.push(plane);
    }
    Some(())
}

fn canonical_plane(normal: [f64; 3], point: [f64; 3]) -> Option<ExpectedPlane> {
    let normal = canonical_direction(normalize(normal, 0.0)?);
    Some(ExpectedPlane {
        normal,
        offset_mm: dot(normal, point),
    })
}

fn same_plane(a: ExpectedPlane, b: ExpectedPlane) -> bool {
    dot(a.normal, b.normal) > 1.0 - DIR_TOL && (a.offset_mm - b.offset_mm).abs() <= SHEET_TOL_MM
}

fn rectangular_bounds(points: &[[f64; 2]]) -> Option<([f64; 2], [f64; 2])> {
    if points.len() != 4 {
        return None;
    }
    let min = [
        points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
        points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
    ];
    let max = [
        points
            .iter()
            .map(|p| p[0])
            .fold(f64::NEG_INFINITY, f64::max),
        points
            .iter()
            .map(|p| p[1])
            .fold(f64::NEG_INFINITY, f64::max),
    ];
    if max[0] - min[0] <= SHEET_TOL_MM || max[1] - min[1] <= SHEET_TOL_MM {
        return None;
    }
    let corners = [
        [min[0], min[1]],
        [max[0], min[1]],
        [max[0], max[1]],
        [min[0], max[1]],
    ];
    corners
        .iter()
        .all(|corner| {
            points
                .iter()
                .any(|p| near(p[0], corner[0]) && near(p[1], corner[1]))
        })
        .then_some((min, max))
}

fn leg_local_point(leg: &RecoveredPlanarSweepLeg, point: [f64; 3]) -> [f64; 3] {
    let delta = sub(point, leg.origin_mm);
    [
        dot(delta, leg.x_axis),
        dot(delta, leg.y_axis),
        dot(delta, leg.z_axis),
    ]
}

fn leg_world_point(leg: &RecoveredPlanarSweepLeg, point: [f64; 3]) -> [f64; 3] {
    add(
        leg.origin_mm,
        add(
            mul(leg.x_axis, point[0]),
            add(mul(leg.y_axis, point[1]), mul(leg.z_axis, point[2])),
        ),
    )
}

fn source_planar_polygon(
    face_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<[f64; 3]>> {
    let surface_id = brep::face_surface(face_id, entities, index)?;
    if !matches!(
        brep::surface_support(surface_id, entities, index),
        brep::SurfaceSupport::Plane(_)
    ) {
        return None;
    }
    simplify_collinear_3d(line_polygon(face_id, entities, index)?)
}

fn find_matching_planar_face(
    expected: &[[f64; 3]],
    face_ids: &[u64],
    used: &HashSet<u64>,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<u64> {
    let matches = face_ids
        .iter()
        .copied()
        .filter(|face_id| !used.contains(face_id))
        .filter(|&face_id| {
            source_planar_polygon(face_id, entities, index)
                .is_some_and(|polygon| cyclic_polygon_match(expected, &polygon))
        })
        .collect::<Vec<_>>();
    (matches.len() == 1).then_some(matches[0])
}

#[derive(Debug, Clone, Copy)]
struct TaperTerminalSection {
    flat_face_id: u64,
    base_y_mm: f64,
    x_min: f64,
    x_max: f64,
    z_min: f64,
    z_max: f64,
    end_y_mm: f64,
    end_min: [f64; 2],
    end_max: [f64; 2],
}

fn find_taper_terminal_section(
    leg: &RecoveredPlanarSweepLeg,
    terminal_direction: [f64; 3],
    face_ids: &[u64],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<TaperTerminalSection> {
    let (profile_min, profile_max) = rectangular_bounds(&leg.profile_points_mm)?;
    let direction_sign = dot(leg.y_axis, terminal_direction);
    if direction_sign.abs() < 1.0 - DIR_TOL {
        return None;
    }
    let sign = direction_sign.signum();
    let base_y_mm = if sign > 0.0 {
        profile_max[1]
    } else {
        profile_min[1]
    };
    let (x_min, x_max) = (profile_min[0], profile_max[0]);
    let (z_min, z_max) = (0.0, leg.depth_mm);
    let mut candidates = Vec::new();
    for &face_id in face_ids {
        let surface_id = brep::face_surface(face_id, entities, index)?;
        let brep::SurfaceSupport::Plane(plane) = brep::surface_support(surface_id, entities, index)
        else {
            continue;
        };
        let plane_normal = normalize(plane.normal, 0.0)?;
        if dot(plane_normal, terminal_direction).abs() < 1.0 - DIR_TOL {
            continue;
        }
        let polygon = source_planar_polygon(face_id, entities, index)?;
        if polygon.len() != 4 {
            continue;
        }
        let local = polygon
            .iter()
            .copied()
            .map(|point| leg_local_point(leg, point))
            .collect::<Vec<_>>();
        let end_y_mm = common_value(&local.iter().map(|p| p[1]).collect::<Vec<_>>())?;
        if (end_y_mm - base_y_mm) * sign <= SHEET_TOL_MM {
            continue;
        }
        let end_xz = local.iter().map(|p| [p[0], p[2]]).collect::<Vec<_>>();
        let Some((end_min, end_max)) = rectangular_bounds(&end_xz) else {
            continue;
        };
        if end_min[0] > x_min + SHEET_TOL_MM
            && end_max[0] < x_max - SHEET_TOL_MM
            && end_min[1] > z_min + SHEET_TOL_MM
            && end_max[1] < z_max - SHEET_TOL_MM
        {
            candidates.push((face_id, end_y_mm, end_min, end_max));
        }
    }
    let [(flat_face_id, end_y_mm, end_min, end_max)] = candidates.as_slice() else {
        return None;
    };
    Some(TaperTerminalSection {
        flat_face_id: *flat_face_id,
        base_y_mm,
        x_min,
        x_max,
        z_min,
        z_max,
        end_y_mm: *end_y_mm,
        end_min: *end_min,
        end_max: *end_max,
    })
}

fn recover_rectangular_taper_tip(
    leg: &RecoveredPlanarSweepLeg,
    terminal_direction: [f64; 3],
    face_ids: &[u64],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<RecoveredRectangularTaperTip> {
    let terminal = find_taper_terminal_section(leg, terminal_direction, face_ids, entities, index)?;
    let local_quads = [
        [
            [terminal.x_min, terminal.base_y_mm, terminal.z_min],
            [terminal.x_min, terminal.base_y_mm, terminal.z_max],
            [terminal.end_min[0], terminal.end_y_mm, terminal.end_max[1]],
            [terminal.end_min[0], terminal.end_y_mm, terminal.end_min[1]],
        ],
        [
            [terminal.x_max, terminal.base_y_mm, terminal.z_min],
            [terminal.end_max[0], terminal.end_y_mm, terminal.end_min[1]],
            [terminal.end_max[0], terminal.end_y_mm, terminal.end_max[1]],
            [terminal.x_max, terminal.base_y_mm, terminal.z_max],
        ],
        [
            [terminal.x_min, terminal.base_y_mm, terminal.z_min],
            [terminal.x_max, terminal.base_y_mm, terminal.z_min],
            [terminal.end_max[0], terminal.end_y_mm, terminal.end_min[1]],
            [terminal.end_min[0], terminal.end_y_mm, terminal.end_min[1]],
        ],
        [
            [terminal.x_min, terminal.base_y_mm, terminal.z_max],
            [terminal.end_min[0], terminal.end_y_mm, terminal.end_max[1]],
            [terminal.end_max[0], terminal.end_y_mm, terminal.end_max[1]],
            [terminal.x_max, terminal.base_y_mm, terminal.z_max],
        ],
    ];
    let mut used = HashSet::from([terminal.flat_face_id]);
    let mut bevel_ids = Vec::with_capacity(4);
    for local_quad in local_quads {
        let expected = local_quad.map(|point| leg_world_point(leg, point));
        let face_id = find_matching_planar_face(&expected, face_ids, &used, entities, index)?;
        used.insert(face_id);
        bevel_ids.push(face_id);
    }
    bevel_ids.sort_unstable();
    let mut source_face_ids = [
        terminal.flat_face_id,
        bevel_ids[0],
        bevel_ids[1],
        bevel_ids[2],
        bevel_ids[3],
    ];
    source_face_ids.sort_unstable();
    Some(RecoveredRectangularTaperTip {
        source_face_ids,
        base_y_mm: terminal.base_y_mm,
        end_y_mm: terminal.end_y_mm,
        end_x_range_mm: [terminal.end_min[0], terminal.end_max[0]],
        end_z_range_mm: [terminal.end_min[1], terminal.end_max[1]],
    })
}

fn recover_planar_leg(
    pair: &SheetPlanePair,
    width_axis: [f64; 3],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<RecoveredPlanarSweepLeg> {
    let normal = canonical_direction(normalize(pair.normal, 0.0)?);
    if dot(normal, width_axis).abs() > DIR_TOL {
        return None;
    }
    let mut negative = line_polygon(pair.negative_face_id, entities, index)?;
    let mut positive = line_polygon(pair.positive_face_id, entities, index)?;
    negative = simplify_collinear_3d(negative)?;
    positive = simplify_collinear_3d(positive)?;

    let positive_shifted = positive
        .iter()
        .copied()
        .map(|point| sub(point, mul(normal, pair.separation_mm)))
        .collect::<Vec<_>>();
    if !cyclic_polygon_match(&negative, &positive_shifted) {
        return None;
    }

    let x_axis = width_axis;
    let y_axis = normalize(cross(normal, x_axis), SHEET_TOL_MM)?;
    let z_axis = normal;
    let origin_mm = negative[0];
    let mut profile_points_mm = negative
        .iter()
        .copied()
        .map(|point| {
            let delta = sub(point, origin_mm);
            [dot(delta, x_axis), dot(delta, y_axis)]
        })
        .collect::<Vec<_>>();
    profile_points_mm = simplify_collinear_2d(profile_points_mm)?;
    if signed_area(&profile_points_mm).abs() <= SHEET_TOL_MM * SHEET_TOL_MM {
        return None;
    }
    if signed_area(&profile_points_mm) < 0.0 {
        profile_points_mm.reverse();
    }

    Some(RecoveredPlanarSweepLeg {
        source_face_ids: [pair.negative_face_id, pair.positive_face_id],
        origin_mm,
        x_axis,
        y_axis,
        z_axis,
        profile_points_mm,
        depth_mm: pair.separation_mm,
        terminal_taper: None,
    })
}

fn line_polygon(
    face_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<[f64; 3]>> {
    let loops = brep::face_loops(face_id, entities, index)?;
    if loops.len() != 1 || !loops[0].outer || loops[0].edges.len() < 3 {
        return None;
    }
    let loop_ = &loops[0];
    let mut points = Vec::with_capacity(loop_.edges.len());
    for (edge_index, edge) in loop_.edges.iter().enumerate() {
        if !matches!(edge.support, CurveSupport::Line(_)) {
            return None;
        }
        if edge_index > 0
            && distance(loop_.edges[edge_index - 1].end_mm, edge.start_mm) > SHEET_TOL_MM
        {
            return None;
        }
        points.push(edge.start_mm);
    }
    if distance(loop_.edges.last()?.end_mm, loop_.edges.first()?.start_mm) > SHEET_TOL_MM {
        return None;
    }
    Some(points)
}

fn recover_cylinder_section(
    pair: &SheetCylinderPair,
    axis: [f64; 3],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<CylinderSection> {
    let mut widths = Vec::new();
    let mut center_parameters = Vec::new();
    for face_id in [pair.inner_face_id, pair.outer_face_id] {
        let loops = brep::face_loops(face_id, entities, index)?;
        if loops.len() != 1 || !loops[0].outer {
            return None;
        }
        for edge in &loops[0].edges {
            let vector = sub(edge.end_mm, edge.start_mm);
            let length = norm(vector);
            if length <= SHEET_TOL_MM {
                return None;
            }
            if matches!(edge.support, CurveSupport::Line(_)) {
                let direction = normalize(vector, SHEET_TOL_MM)?;
                if dot(direction, axis).abs() < 1.0 - DIR_TOL {
                    return None;
                }
                widths.push(length);
                let midpoint = mul(add(edge.start_mm, edge.end_mm), 0.5);
                center_parameters.push(dot(sub(midpoint, pair.axis_origin_mm), axis));
            }
        }
    }
    if widths.len() != 4 || center_parameters.len() != 4 {
        return None;
    }
    let width_mm = common_value(&widths)?;
    let center_parameter = common_value(&center_parameters)?;
    Some(CylinderSection {
        width_mm,
        axis_center_mm: add(pair.axis_origin_mm, mul(axis, center_parameter)),
    })
}

fn prove_skin_tangent(
    skin: &formed_sheet::SheetReferenceSkin,
    plane_index: usize,
    pair: &SheetPlanePair,
    axis: [f64; 3],
    width_mm: f64,
    tangent_mm: [f64; 3],
) -> Option<SkinTangent> {
    let patch = SheetPatchId::Plane(plane_index);
    let seam = skin.internal_seams.iter().find_map(|curve| match curve {
        SheetReferenceCurve::Line {
            patches,
            start_mm,
            end_mm,
            ..
        } if patches.contains(&patch) && patches.contains(&SheetPatchId::Cylinder(0)) => {
            Some((*start_mm, *end_mm))
        }
        _ => None,
    })?;
    let seam_vector = sub(seam.1, seam.0);
    if !near(norm(seam_vector), width_mm)
        || dot(normalize(seam_vector, SHEET_TOL_MM)?, axis).abs() < 1.0 - DIR_TOL
    {
        return None;
    }

    let selected_offset = skin.patches.iter().find_map(|entry| {
        if entry.patch != patch {
            return None;
        }
        match entry.surface {
            SheetReferenceSurface::Plane { offset_mm, .. } => Some(offset_mm),
            _ => None,
        }
    })?;
    let normal = normalize(pair.normal, 0.0)?;
    let seam_midpoint = mul(add(seam.0, seam.1), 0.5);
    let centered = add(
        seam_midpoint,
        mul(normal, pair.mid_offset_mm - selected_offset),
    );
    if distance(centered, tangent_mm) > SHEET_TOL_MM {
        return None;
    }

    let boundary = &skin.boundary_loops[0].curves;
    let mut inward_vectors = Vec::with_capacity(2);
    for seam_point in [seam.0, seam.1] {
        let matches = boundary
            .iter()
            .filter_map(|curve| match curve {
                SheetReferenceCurve::Line {
                    patches,
                    start_mm,
                    end_mm,
                    ..
                } if patches.contains(&patch) && !patches.contains(&SheetPatchId::Cylinder(0)) => {
                    if distance(*start_mm, seam_point) <= SHEET_TOL_MM {
                        Some(sub(*end_mm, *start_mm))
                    } else if distance(*end_mm, seam_point) <= SHEET_TOL_MM {
                        Some(sub(*start_mm, *end_mm))
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return None;
        }
        inward_vectors.push(matches[0]);
    }
    let lengths = inward_vectors
        .iter()
        .map(|vector| norm(*vector))
        .collect::<Vec<_>>();
    let straight_length_mm = common_value(&lengths)?;
    let first = normalize(inward_vectors[0], SHEET_TOL_MM)?;
    let second = normalize(inward_vectors[1], SHEET_TOL_MM)?;
    if dot(first, second) < 1.0 - 1.0e-8 || dot(first, axis).abs() > DIR_TOL {
        return None;
    }
    Some(SkinTangent {
        inward_direction: first,
        straight_length_mm,
    })
}

fn choose_bend_order(
    axis: [f64; 3],
    radials: &[[f64; 3]],
    tangents: &[SkinTangent],
) -> Option<(usize, usize, f64, [f64; 3], [f64; 3])> {
    if radials.len() != 2 || tangents.len() != 2 {
        return None;
    }
    for (start, end) in [(0usize, 1usize), (1, 0)] {
        let angle =
            dot(axis, cross(radials[start], radials[end])).atan2(dot(radials[start], radials[end]));
        if !angle.is_finite()
            || angle.abs() <= 1.0e-8
            || angle.abs() >= std::f64::consts::PI - 1.0e-8
        {
            continue;
        }
        let start_motion = normalize(
            mul(cross(axis, radials[start]), angle.signum()),
            SHEET_TOL_MM,
        )?;
        let end_motion = normalize(mul(cross(axis, radials[end]), angle.signum()), SHEET_TOL_MM)?;
        if dot(start_motion, mul(tangents[start].inward_direction, -1.0)) > 1.0 - 1.0e-8
            && dot(end_motion, tangents[end].inward_direction) > 1.0 - 1.0e-8
        {
            return Some((start, end, angle, start_motion, end_motion));
        }
    }
    None
}

fn prove_cylinder_sector(
    pair: &SheetCylinderPair,
    axis: [f64; 3],
    width_mm: f64,
    radial_directions: &[[f64; 3]],
    bend_angle_rad: f64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<()> {
    let expected_angle = bend_angle_rad.abs();
    for (face_id, radius_mm) in [
        (pair.inner_face_id, pair.inner_radius_mm),
        (pair.outer_face_id, pair.outer_radius_mm),
    ] {
        let loops = brep::face_loops(face_id, entities, index)?;
        if loops.len() != 1 || !loops[0].outer || loops[0].edges.len() != 4 {
            return None;
        }
        let mut lines = 0usize;
        let mut circles = 0usize;
        for edge in &loops[0].edges {
            match edge.support {
                CurveSupport::Line(_) => {
                    let vector = sub(edge.end_mm, edge.start_mm);
                    if !near(norm(vector), width_mm)
                        || dot(normalize(vector, SHEET_TOL_MM)?, axis).abs() < 1.0 - DIR_TOL
                    {
                        return None;
                    }
                    lines += 1;
                }
                CurveSupport::Circle(circle) => {
                    if !near(circle.radius_mm, radius_mm)
                        || dot(normalize(circle.normal, 0.0)?, axis).abs() < 1.0 - DIR_TOL
                        || point_to_unit_line_distance(circle.center_mm, pair.axis_origin_mm, axis)
                            > SHEET_TOL_MM
                    {
                        return None;
                    }
                    let start_radial = radial_direction(edge.start_mm, pair.axis_origin_mm, axis)?;
                    let end_radial = radial_direction(edge.end_mm, pair.axis_origin_mm, axis)?;
                    if !endpoints_match_radials(start_radial, end_radial, radial_directions) {
                        return None;
                    }
                    let angle = dot(start_radial, end_radial).clamp(-1.0, 1.0).acos();
                    if (angle - expected_angle).abs() > 1.0e-7 {
                        return None;
                    }
                    circles += 1;
                }
                _ => return None,
            }
        }
        if lines != 2 || circles != 2 {
            return None;
        }
    }
    Some(())
}

fn radial_direction(point: [f64; 3], axis_origin: [f64; 3], axis: [f64; 3]) -> Option<[f64; 3]> {
    let along = dot(sub(point, axis_origin), axis);
    let axis_point = add(axis_origin, mul(axis, along));
    normalize(sub(point, axis_point), SHEET_TOL_MM)
}

fn endpoints_match_radials(start: [f64; 3], end: [f64; 3], expected: &[[f64; 3]]) -> bool {
    expected.len() == 2
        && ((dot(start, expected[0]) > 1.0 - 1.0e-8 && dot(end, expected[1]) > 1.0 - 1.0e-8)
            || (dot(start, expected[1]) > 1.0 - 1.0e-8 && dot(end, expected[0]) > 1.0 - 1.0e-8))
}

fn simplify_collinear_3d(mut points: Vec<[f64; 3]>) -> Option<Vec<[f64; 3]>> {
    if points.len() < 3 {
        return None;
    }
    loop {
        let mut removed = false;
        for i in 0..points.len() {
            let prev = points[(i + points.len() - 1) % points.len()];
            let current = points[i];
            let next = points[(i + 1) % points.len()];
            let a = sub(current, prev);
            let b = sub(next, current);
            let scale = norm(a).max(norm(b));
            if scale <= SHEET_TOL_MM {
                return None;
            }
            if norm(cross(a, b)) <= SHEET_TOL_MM * scale {
                points.remove(i);
                removed = true;
                break;
            }
        }
        if !removed || points.len() <= 3 {
            break;
        }
    }
    (points.len() >= 3).then_some(points)
}

fn simplify_collinear_2d(mut points: Vec<[f64; 2]>) -> Option<Vec<[f64; 2]>> {
    if points.len() < 3 {
        return None;
    }
    loop {
        let mut removed = false;
        for i in 0..points.len() {
            let prev = points[(i + points.len() - 1) % points.len()];
            let current = points[i];
            let next = points[(i + 1) % points.len()];
            let a = [current[0] - prev[0], current[1] - prev[1]];
            let b = [next[0] - current[0], next[1] - current[1]];
            let cross2 = a[0].mul_add(b[1], -a[1] * b[0]);
            let scale = a[0].hypot(a[1]).max(b[0].hypot(b[1]));
            if scale <= SHEET_TOL_MM {
                return None;
            }
            if cross2.abs() <= SHEET_TOL_MM * scale {
                points.remove(i);
                removed = true;
                break;
            }
        }
        if !removed || points.len() <= 3 {
            break;
        }
    }
    (points.len() >= 3).then_some(points)
}

fn cyclic_polygon_match(a: &[[f64; 3]], b: &[[f64; 3]]) -> bool {
    if a.len() != b.len() || a.is_empty() {
        return false;
    }
    (0..b.len()).any(|start| {
        distance(a[0], b[start]) <= SHEET_TOL_MM
            && (0..a.len()).all(|i| distance(a[i], b[(start + i) % b.len()]) <= SHEET_TOL_MM)
    }) || (0..b.len()).any(|start| {
        distance(a[0], b[start]) <= SHEET_TOL_MM
            && (0..a.len()).all(|i| {
                let j = (start + b.len() - (i % b.len())) % b.len();
                distance(a[i], b[j]) <= SHEET_TOL_MM
            })
    })
}

fn signed_area(points: &[[f64; 2]]) -> f64 {
    points
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            let b = points[(i + 1) % points.len()];
            a[0].mul_add(b[1], -a[1] * b[0])
        })
        .sum::<f64>()
        * 0.5
}

fn common_value(values: &[f64]) -> Option<f64> {
    let &first = values.first()?;
    if !first.is_finite() || values.iter().any(|&value| !near(first, value)) {
        return None;
    }
    Some(first)
}

fn near(a: f64, b: f64) -> bool {
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() <= SHEET_TOL_MM.max(scale * 1.0e-9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cyclic_polygon_match_accepts_reverse_winding() {
        let a = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let b = vec![
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        assert!(cyclic_polygon_match(&a, &b));
    }
}
