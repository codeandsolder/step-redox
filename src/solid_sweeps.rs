mod sheet;
pub use sheet::{
    RecoveredOpenRectangularSweep, RecoveredPlanarSweepLeg, RecoveredRectangularTaperTip,
    detect_open_rectangular_sweeps,
};

use crate::brep::{
    self, CircleSupport, CurveSupport, CylinderSupport, SurfaceSupport, TorusSupport,
};
use crate::math3::{
    add, canonical_direction, cross, distance, dot, mul, norm, normalize,
    point_to_unit_line_distance, sub,
};
use crate::solid_revolutions::{
    RecoveredSolidRevolution, recover_axisymmetric_subbody_with_circular_cap,
};
use crate::step_graph::{build_index, entity_id, simple_record};
use ruststep::ast::EntityInstance;
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};

const GEOM_TOL_MM: f64 = 1.0e-7;
const DIR_TOL: f64 = 1.0e-9;

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RecoveredSweepSegment {
    Translation {
        vector_mm: [f64; 3],
    },
    Rotation {
        axis_origin_mm: [f64; 3],
        axis_direction: [f64; 3],
        angle_rad: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecoveredClosedRoundSweep {
    pub solid_id: u64,
    pub source_face_ids: Vec<u64>,
    pub profile_radius_mm: f64,
    pub path_start_mm: [f64; 3],
    pub profile_x_axis: [f64; 3],
    pub profile_y_axis: [f64; 3],
    pub path_tangent: [f64; 3],
    pub segments: Vec<RecoveredSweepSegment>,
    pub max_residual_mm: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecoveredRevolvedRoundTail {
    pub solid_id: u64,
    pub source_face_ids: Vec<u64>,
    pub head: RecoveredSolidRevolution,
    pub profile_radius_mm: f64,
    pub path_start_mm: [f64; 3],
    pub profile_x_axis: [f64; 3],
    pub profile_y_axis: [f64; 3],
    pub path_tangent: [f64; 3],
    pub segments: Vec<RecoveredSweepSegment>,
    pub end_sphere_center_mm: [f64; 3],
    pub max_residual_mm: f64,
}

#[derive(Debug, Clone)]
struct CylinderRun {
    origin_mm: [f64; 3],
    direction: [f64; 3],
    radius_mm: f64,
    face_ids: Vec<u64>,
}

#[derive(Debug, Clone)]
struct TorusBend {
    center_mm: [f64; 3],
    axis: [f64; 3],
    major_radius_mm: f64,
    minor_radius_mm: f64,
    face_ids: Vec<u64>,
}

#[derive(Debug, Clone)]
struct SphereEnd {
    center_mm: [f64; 3],
    radius_mm: f64,
    face_ids: Vec<u64>,
}

struct ClosedRoundSupports {
    cylinders: Vec<CylinderRun>,
    tori: Vec<TorusBend>,
    profile_radius_mm: f64,
    corner_radius_mm: f64,
}

#[must_use]
pub fn detect_closed_round_sweeps(entities: &[EntityInstance]) -> Vec<RecoveredClosedRoundSweep> {
    let index = build_index(entities);
    let mut out = Vec::new();

    for entity in entities {
        let Some(record) = simple_record(entity) else {
            continue;
        };
        if record.name != "MANIFOLD_SOLID_BREP" {
            continue;
        }
        let solid_id = entity_id(entity);
        if let Some(sweep) = detect_one_closed_round_sweep(solid_id, entities, &index) {
            out.push(sweep);
        }
    }

    out.sort_by_key(|sweep| sweep.solid_id);
    out
}

#[must_use]
pub fn detect_revolved_round_tails(entities: &[EntityInstance]) -> Vec<RecoveredRevolvedRoundTail> {
    let index = build_index(entities);
    let mut out = Vec::new();
    for entity in entities {
        let Some(record) = simple_record(entity) else {
            continue;
        };
        if record.name != "MANIFOLD_SOLID_BREP" {
            continue;
        }
        let solid_id = entity_id(entity);
        if let Some(candidate) = detect_one_revolved_round_tail(solid_id, entities, &index) {
            out.push(candidate);
        }
    }
    out.sort_by_key(|candidate| candidate.solid_id);
    out
}

fn detect_one_closed_round_sweep(
    solid_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<RecoveredClosedRoundSweep> {
    let mut face_ids = brep::solid_face_ids(solid_id, entities, index)?;
    if face_ids.len() < 8 {
        return None;
    }
    face_ids.sort_unstable();
    prove_closed_source_topology(&face_ids, entities, index)?;

    let ClosedRoundSupports {
        cylinders,
        tori,
        profile_radius_mm,
        corner_radius_mm,
    } = collect_closed_round_supports(&face_ids, entities, index)?;

    let plane_normal = tori.first()?.axis;
    if tori
        .iter()
        .any(|torus| 1.0 - dot(plane_normal, torus.axis).abs() > DIR_TOL)
    {
        return None;
    }
    if cylinders
        .iter()
        .any(|cylinder| dot(cylinder.direction, plane_normal).abs() > DIR_TOL)
    {
        return None;
    }

    let mut cylinder_corners = Vec::<[usize; 2]>::new();
    let mut max_residual_mm: f64 = 0.0;
    for cylinder in &cylinders {
        let mut connected = Vec::new();
        for (torus_index, torus) in tori.iter().enumerate() {
            let residual = (point_to_unit_line_distance(
                torus.center_mm,
                cylinder.origin_mm,
                cylinder.direction,
            ) - corner_radius_mm)
                .abs();
            if residual <= GEOM_TOL_MM {
                connected.push(torus_index);
                max_residual_mm = max_residual_mm.max(residual);
            }
        }
        if connected.len() != 2 {
            return None;
        }
        cylinder_corners.push([connected[0], connected[1]]);
    }

    let mut corner_edges = vec![Vec::<(usize, usize)>::new(); tori.len()];
    for (cylinder_index, [a, b]) in cylinder_corners.iter().copied().enumerate() {
        corner_edges[a].push((cylinder_index, b));
        corner_edges[b].push((cylinder_index, a));
    }
    if corner_edges.iter().any(|edges| edges.len() != 2) {
        return None;
    }

    for (run_index, [first_corner, second_corner]) in cylinder_corners.iter().copied().enumerate() {
        prove_cylinder_trim(
            &cylinders[run_index],
            &tori[first_corner],
            &tori[second_corner],
            profile_radius_mm,
            entities,
            index,
        )?;
    }
    for (torus_index, attached_runs) in corner_edges.iter().enumerate() {
        prove_torus_trim(
            &tori[torus_index],
            &cylinders[attached_runs[0].0],
            &cylinders[attached_runs[1].0],
            entities,
            index,
        )?;
    }

    let start_corner =
        (0..tori.len()).min_by(|&a, &b| lex_point(tori[a].center_mm, tori[b].center_mm))?;

    let builder = ClosedRoundSweepBuilder {
        solid_id,
        face_ids: &face_ids,
        profile_radius_mm,
        corner_radius_mm,
        plane_normal,
        tori: &tori,
        cylinders: &cylinders,
        max_residual_mm,
    };
    let mut starts = corner_edges[start_corner].clone();
    starts.sort_by(|a, b| lex_point(tori[a.1].center_mm, tori[b.1].center_mm));
    for (_, next_corner) in starts {
        let Some((corner_order, edge_order)) =
            ordered_cycle(start_corner, next_corner, &corner_edges)
        else {
            continue;
        };
        if let Some(recovered) = builder.build(&corner_order, &edge_order) {
            return Some(recovered);
        }
    }
    None
}

fn detect_one_revolved_round_tail(
    solid_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<RecoveredRevolvedRoundTail> {
    let mut face_ids = brep::solid_face_ids(solid_id, entities, index)?;
    if face_ids.len() < 8 {
        return None;
    }
    face_ids.sort_unstable();
    prove_closed_manifold_source_topology(&face_ids, entities, index)?;

    let mut cylinders = Vec::<CylinderRun>::new();
    let mut tori = Vec::<TorusBend>::new();
    let mut spheres = Vec::<SphereEnd>::new();
    for &face_id in &face_ids {
        let surface_id = brep::face_surface(face_id, entities, index)?;
        match brep::surface_support(surface_id, entities, index) {
            SurfaceSupport::Cylinder(support) => {
                let mut run = canonical_cylinder(support)?;
                if let Some(existing) = cylinders
                    .iter_mut()
                    .find(|existing| same_cylinder(existing, &run))
                {
                    existing.face_ids.push(face_id);
                } else {
                    run.face_ids.push(face_id);
                    cylinders.push(run);
                }
            }
            SurfaceSupport::Torus(support) => {
                let mut bend = canonical_torus(support)?;
                if let Some(existing) = tori.iter_mut().find(|existing| same_torus(existing, &bend))
                {
                    existing.face_ids.push(face_id);
                } else {
                    bend.face_ids.push(face_id);
                    tori.push(bend);
                }
            }
            SurfaceSupport::Sphere(support) => {
                if !support.radius_mm.is_finite() || support.radius_mm <= GEOM_TOL_MM {
                    return None;
                }
                let candidate = SphereEnd {
                    center_mm: support.center_mm,
                    radius_mm: support.radius_mm,
                    face_ids: vec![face_id],
                };
                if let Some(existing) = spheres.iter_mut().find(|existing| {
                    near(existing.radius_mm, candidate.radius_mm)
                        && distance(existing.center_mm, candidate.center_mm) <= GEOM_TOL_MM
                }) {
                    existing.face_ids.push(face_id);
                } else {
                    spheres.push(candidate);
                }
            }
            _ => {}
        }
    }

    for bend in &tori {
        let radius = bend.minor_radius_mm;
        let attached = cylinders
            .iter()
            .enumerate()
            .filter(|(_, run)| {
                near(run.radius_mm, radius)
                    && dot(run.direction, bend.axis).abs() <= DIR_TOL
                    && (point_to_unit_line_distance(bend.center_mm, run.origin_mm, run.direction)
                        - bend.major_radius_mm)
                        .abs()
                        <= GEOM_TOL_MM
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if attached.len() < 2 {
            continue;
        }

        for &incoming_index in &attached {
            for &outgoing_index in &attached {
                if incoming_index == outgoing_index {
                    continue;
                }
                let incoming = &cylinders[incoming_index];
                let outgoing = &cylinders[outgoing_index];
                if dot(incoming.direction, outgoing.direction).abs() > DIR_TOL {
                    continue;
                }
                let interface_mm = project_to_line(bend.center_mm, incoming);
                let bend_end_mm = project_to_line(bend.center_mm, outgoing);
                let radial_in = normalize(sub(interface_mm, bend.center_mm), GEOM_TOL_MM)?;
                let radial_out = normalize(sub(bend_end_mm, bend.center_mm), GEOM_TOL_MM)?;
                let angle_rad =
                    dot(bend.axis, cross(radial_in, radial_out)).atan2(dot(radial_in, radial_out));
                if !angle_rad.is_finite()
                    || angle_rad.abs() <= 1.0e-8
                    || angle_rad.abs() >= std::f64::consts::PI - 1.0e-8
                {
                    continue;
                }
                let sign = angle_rad.signum();
                let path_tangent = normalize(mul(cross(bend.axis, radial_in), sign), GEOM_TOL_MM)?;
                let bend_exit_tangent =
                    normalize(mul(cross(bend.axis, radial_out), sign), GEOM_TOL_MM)?;

                for sphere in &spheres {
                    if !near(sphere.radius_mm, radius)
                        || point_to_unit_line_distance(
                            sphere.center_mm,
                            outgoing.origin_mm,
                            outgoing.direction,
                        ) > GEOM_TOL_MM
                    {
                        continue;
                    }
                    let translation = sub(sphere.center_mm, bend_end_mm);
                    let translation_len = norm(translation);
                    if translation_len <= GEOM_TOL_MM {
                        continue;
                    }
                    let translation_dir = normalize(translation, GEOM_TOL_MM)?;
                    if dot(translation_dir, bend_exit_tangent) < 1.0 - 1.0e-7 {
                        continue;
                    }

                    let mut tail_faces = HashSet::<u64>::new();
                    tail_faces.extend(bend.face_ids.iter().copied());
                    tail_faces.extend(outgoing.face_ids.iter().copied());
                    tail_faces.extend(sphere.face_ids.iter().copied());
                    let head_face_ids = face_ids
                        .iter()
                        .copied()
                        .filter(|face_id| !tail_faces.contains(face_id))
                        .collect::<Vec<_>>();
                    if head_face_ids.len() + tail_faces.len() != face_ids.len() {
                        continue;
                    }
                    let Some((head, interface_circle)) =
                        recover_axisymmetric_subbody_with_circular_cap(
                            solid_id,
                            &head_face_ids,
                            entities,
                            index,
                        )
                    else {
                        continue;
                    };
                    if !near(interface_circle.radius_mm, radius)
                        || distance(interface_circle.center_mm, interface_mm) > GEOM_TOL_MM
                        || !circle_is_cylinder_section(&interface_circle, incoming, radius)
                    {
                        continue;
                    }
                    prove_torus_trim(bend, incoming, outgoing, entities, index)?;
                    prove_cylinder_between_points(
                        outgoing,
                        bend_end_mm,
                        sphere.center_mm,
                        radius,
                        entities,
                        index,
                    )?;
                    prove_hemispherical_end(sphere, outgoing, entities, index)?;

                    let path_start_mm = interface_mm;

                    let profile_x_axis = bend.axis;
                    let profile_y_axis =
                        normalize(cross(path_tangent, profile_x_axis), GEOM_TOL_MM)?;
                    if dot(cross(profile_x_axis, profile_y_axis), path_tangent) < 1.0 - 1.0e-7 {
                        continue;
                    }
                    let max_residual_mm = head.max_residual_mm.max(
                        (distance(interface_mm, bend.center_mm) - bend.major_radius_mm)
                            .abs()
                            .max(
                                (distance(bend_end_mm, bend.center_mm) - bend.major_radius_mm)
                                    .abs(),
                            )
                            .max(point_to_unit_line_distance(
                                sphere.center_mm,
                                outgoing.origin_mm,
                                outgoing.direction,
                            )),
                    );
                    if max_residual_mm > 1.0e-5 {
                        continue;
                    }
                    return Some(RecoveredRevolvedRoundTail {
                        solid_id,
                        source_face_ids: face_ids.clone(),
                        head,
                        profile_radius_mm: radius,
                        path_start_mm,
                        profile_x_axis,
                        profile_y_axis,
                        path_tangent,
                        segments: vec![
                            RecoveredSweepSegment::Rotation {
                                axis_origin_mm: bend.center_mm,
                                axis_direction: bend.axis,
                                angle_rad,
                            },
                            RecoveredSweepSegment::Translation {
                                vector_mm: translation,
                            },
                        ],
                        end_sphere_center_mm: sphere.center_mm,
                        max_residual_mm,
                    });
                }
            }
        }
    }
    None
}

fn prove_hemispherical_end(
    sphere: &SphereEnd,
    run: &CylinderRun,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<()> {
    if !near(sphere.radius_mm, run.radius_mm) {
        return None;
    }
    let run_edges = run
        .face_ids
        .iter()
        .copied()
        .flat_map(|face_id| {
            brep::face_loops(face_id, entities, index)
                .into_iter()
                .flatten()
        })
        .flat_map(|loop_| loop_.edges)
        .map(|edge| edge.edge_id)
        .collect::<HashSet<_>>();
    let mut edge_counts = HashMap::<u64, (usize, Option<brep::OrientedEdgeUse>)>::new();
    for &face_id in &sphere.face_ids {
        let SurfaceSupport::Sphere(support) = brep::surface_support(
            brep::face_surface(face_id, entities, index)?,
            entities,
            index,
        ) else {
            return None;
        };
        if !near(support.radius_mm, sphere.radius_mm)
            || distance(support.center_mm, sphere.center_mm) > GEOM_TOL_MM
        {
            return None;
        }
        for loop_ in brep::face_loops(face_id, entities, index)? {
            for edge in loop_.edges {
                for point in [edge.start_mm, edge.end_mm] {
                    if (distance(point, sphere.center_mm) - sphere.radius_mm).abs() > GEOM_TOL_MM {
                        return None;
                    }
                }
                let entry = edge_counts.entry(edge.edge_id).or_insert((0, None));
                entry.0 += 1;
                entry.1.get_or_insert(edge);
            }
        }
    }
    let boundary = edge_counts
        .values()
        .filter(|(count, _)| *count == 1)
        .filter_map(|(_, edge)| edge.as_ref())
        .collect::<Vec<_>>();
    if boundary.is_empty()
        || boundary
            .iter()
            .any(|edge| !run_edges.contains(&edge.edge_id))
    {
        return None;
    }
    boundary
        .iter()
        .all(|edge| {
            let CurveSupport::Circle(circle) = edge.support else {
                return false;
            };
            near(circle.radius_mm, sphere.radius_mm)
                && distance(circle.center_mm, sphere.center_mm) <= GEOM_TOL_MM
                && 1.0
                    - dot(
                        normalize(circle.normal, GEOM_TOL_MM).unwrap_or([0.0; 3]),
                        run.direction,
                    )
                    .abs()
                    <= DIR_TOL
        })
        .then_some(())
}

fn collect_closed_round_supports(
    face_ids: &[u64],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<ClosedRoundSupports> {
    let mut cylinders = Vec::<CylinderRun>::new();
    let mut tori = Vec::<TorusBend>::new();
    let mut radius_samples = Vec::new();
    let mut bend_radius_samples = Vec::new();

    for &face_id in face_ids {
        let surface_id = brep::face_surface(face_id, entities, index)?;
        match brep::surface_support(surface_id, entities, index) {
            SurfaceSupport::Cylinder(support) => {
                let mut run = canonical_cylinder(support)?;
                radius_samples.push(run.radius_mm);
                if let Some(existing) = cylinders
                    .iter_mut()
                    .find(|existing| same_cylinder(existing, &run))
                {
                    existing.face_ids.push(face_id);
                } else {
                    run.face_ids.push(face_id);
                    cylinders.push(run);
                }
            }
            SurfaceSupport::Torus(support) => {
                let mut bend = canonical_torus(support)?;
                radius_samples.push(bend.minor_radius_mm);
                bend_radius_samples.push(bend.major_radius_mm);
                if let Some(existing) = tori.iter_mut().find(|existing| same_torus(existing, &bend))
                {
                    existing.face_ids.push(face_id);
                } else {
                    bend.face_ids.push(face_id);
                    tori.push(bend);
                }
            }
            _ => return None,
        }
    }

    // STEP commonly splits one analytic support at a parameter seam, so the
    // source face count is deliberately not prescribed here.
    if cylinders.len() != 4 || tori.len() != 4 {
        return None;
    }
    let profile_radius_mm = common_value(&radius_samples)?;
    let corner_radius_mm = common_value(&bend_radius_samples)?;
    (profile_radius_mm > 0.0 && corner_radius_mm > 0.0).then_some(ClosedRoundSupports {
        cylinders,
        tori,
        profile_radius_mm,
        corner_radius_mm,
    })
}

fn prove_closed_manifold_source_topology(
    face_ids: &[u64],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<()> {
    if face_ids.is_empty() {
        return None;
    }
    let mut edge_faces = HashMap::<u64, Vec<usize>>::new();
    let mut face_neighbors = vec![HashSet::<usize>::new(); face_ids.len()];
    for (face_index, &face_id) in face_ids.iter().enumerate() {
        let loops = brep::face_loops(face_id, entities, index)?;
        if loops.is_empty() {
            return None;
        }
        for loop_ in loops {
            if loop_.edges.len() < 2 {
                return None;
            }
            for (edge, next) in loop_.edges.iter().zip(loop_.edges.iter().cycle().skip(1)) {
                if edge.end_vertex != next.start_vertex {
                    return None;
                }
                edge_faces.entry(edge.edge_id).or_default().push(face_index);
            }
        }
    }
    for attached in edge_faces.values() {
        if attached.len() != 2 {
            return None;
        }
        let [a, b] = attached.as_slice() else {
            return None;
        };
        if a != b {
            face_neighbors[*a].insert(*b);
            face_neighbors[*b].insert(*a);
        }
    }
    let mut seen = vec![false; face_ids.len()];
    let mut queue = VecDeque::from([0_usize]);
    seen[0] = true;
    while let Some(face) = queue.pop_front() {
        for &neighbor in &face_neighbors[face] {
            if !seen[neighbor] {
                seen[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }
    seen.into_iter().all(|value| value).then_some(())
}

fn prove_closed_source_topology(
    face_ids: &[u64],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<()> {
    if face_ids.is_empty() {
        return None;
    }
    let mut edge_faces = HashMap::<u64, Vec<usize>>::new();
    let mut face_neighbors = vec![HashSet::<usize>::new(); face_ids.len()];

    for (face_index, &face_id) in face_ids.iter().enumerate() {
        let loops = brep::face_loops(face_id, entities, index)?;
        if loops.len() != 1 || !loops[0].outer || loops[0].edges.len() < 2 {
            return None;
        }
        let edges = &loops[0].edges;
        for (edge, next) in edges.iter().zip(edges.iter().cycle().skip(1)) {
            if edge.end_vertex != next.start_vertex {
                return None;
            }
            edge_faces.entry(edge.edge_id).or_default().push(face_index);
        }
    }

    for attached in edge_faces.values() {
        if attached.len() != 2 {
            return None;
        }
        let a = attached[0];
        let b = attached[1];
        if a != b {
            face_neighbors[a].insert(b);
            face_neighbors[b].insert(a);
        }
    }

    let mut seen = vec![false; face_ids.len()];
    let mut queue = VecDeque::from([0_usize]);
    seen[0] = true;
    while let Some(face) = queue.pop_front() {
        for &neighbor in &face_neighbors[face] {
            if !seen[neighbor] {
                seen[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }
    seen.into_iter().all(|value| value).then_some(())
}

fn prove_cylinder_trim(
    run: &CylinderRun,
    first_corner: &TorusBend,
    second_corner: &TorusBend,
    profile_radius_mm: f64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<()> {
    prove_cylinder_between_points(
        run,
        project_to_line(first_corner.center_mm, run),
        project_to_line(second_corner.center_mm, run),
        profile_radius_mm,
        entities,
        index,
    )
}

fn prove_cylinder_between_points(
    run: &CylinderRun,
    start_mm: [f64; 3],
    end_mm: [f64; 3],
    profile_radius_mm: f64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<()> {
    let tangent = normalize(sub(end_mm, start_mm), GEOM_TOL_MM)?;
    let length_mm = distance(start_mm, end_mm);
    if length_mm <= GEOM_TOL_MM {
        return None;
    }

    let mut min_along = f64::INFINITY;
    let mut max_along = f64::NEG_INFINITY;
    for &face_id in &run.face_ids {
        for loop_ in brep::face_loops(face_id, entities, index)? {
            for edge in loop_.edges {
                for point in [edge.start_mm, edge.end_mm] {
                    if (point_to_unit_line_distance(point, run.origin_mm, run.direction)
                        - profile_radius_mm)
                        .abs()
                        > GEOM_TOL_MM
                    {
                        return None;
                    }
                    let along = dot(sub(point, start_mm), tangent);
                    if along < -GEOM_TOL_MM || along > length_mm + GEOM_TOL_MM {
                        return None;
                    }
                    min_along = min_along.min(along);
                    max_along = max_along.max(along);
                }
                match edge.support {
                    CurveSupport::Line(line) => {
                        let direction = normalize(line.direction, GEOM_TOL_MM)?;
                        if 1.0 - dot(direction, run.direction).abs() > DIR_TOL
                            || (point_to_unit_line_distance(
                                line.origin_mm,
                                run.origin_mm,
                                run.direction,
                            ) - profile_radius_mm)
                                .abs()
                                > GEOM_TOL_MM
                        {
                            return None;
                        }
                    }
                    CurveSupport::Circle(circle) => {
                        if !circle_is_cylinder_section(&circle, run, profile_radius_mm) {
                            return None;
                        }
                    }
                    CurveSupport::BSpline(_) | CurveSupport::Other { .. } => return None,
                }
            }
        }
    }

    (min_along.abs() <= GEOM_TOL_MM && (max_along - length_mm).abs() <= GEOM_TOL_MM).then_some(())
}

fn circle_is_cylinder_section(
    circle: &CircleSupport,
    run: &CylinderRun,
    profile_radius_mm: f64,
) -> bool {
    let Some(normal) = normalize(circle.normal, GEOM_TOL_MM) else {
        return false;
    };
    1.0 - dot(normal, run.direction).abs() <= DIR_TOL
        && near(circle.radius_mm, profile_radius_mm)
        && point_to_unit_line_distance(circle.center_mm, run.origin_mm, run.direction)
            <= GEOM_TOL_MM
}

fn prove_torus_trim(
    bend: &TorusBend,
    first_run: &CylinderRun,
    second_run: &CylinderRun,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<()> {
    let radial_a = normalize(
        sub(project_to_line(bend.center_mm, first_run), bend.center_mm),
        GEOM_TOL_MM,
    )?;
    let radial_b = normalize(
        sub(project_to_line(bend.center_mm, second_run), bend.center_mm),
        GEOM_TOL_MM,
    )?;
    let bend_angle = unit_angle(radial_a, radial_b)?;
    if bend_angle <= 1.0e-8 || bend_angle >= std::f64::consts::PI - 1.0e-8 {
        return None;
    }

    let mut saw_first_end = false;
    let mut saw_second_end = false;
    let mut saw_major_trace = false;
    for &face_id in &bend.face_ids {
        for loop_ in brep::face_loops(face_id, entities, index)? {
            for edge in loop_.edges {
                for point in [edge.start_mm, edge.end_mm] {
                    if !point_on_torus(point, bend) {
                        return None;
                    }
                    let direction = torus_major_direction(point, bend)?;
                    if !direction_on_minor_arc(radial_a, radial_b, direction, bend_angle) {
                        return None;
                    }
                    saw_first_end |= 1.0 - dot(direction, radial_a) <= DIR_TOL;
                    saw_second_end |= 1.0 - dot(direction, radial_b) <= DIR_TOL;
                }

                let CurveSupport::Circle(circle) = edge.support else {
                    return None;
                };
                match classify_torus_boundary_circle(&circle, bend)? {
                    TorusBoundaryCircle::CrossSection { radial_direction } => {
                        if !direction_on_minor_arc(radial_a, radial_b, radial_direction, bend_angle)
                        {
                            return None;
                        }
                    }
                    TorusBoundaryCircle::MajorTrace => {
                        let sweep = directed_circle_sweep(&edge, &circle)?;
                        if sweep <= 1.0e-10 || sweep > bend_angle + 1.0e-7 {
                            return None;
                        }
                        saw_major_trace = true;
                    }
                }
            }
        }
    }

    (saw_first_end && saw_second_end && saw_major_trace).then_some(())
}

#[derive(Debug, Clone, Copy)]
enum TorusBoundaryCircle {
    CrossSection { radial_direction: [f64; 3] },
    MajorTrace,
}

fn classify_torus_boundary_circle(
    circle: &CircleSupport,
    bend: &TorusBend,
) -> Option<TorusBoundaryCircle> {
    if !circle.radius_mm.is_finite() || circle.radius_mm <= 0.0 {
        return None;
    }
    let normal = normalize(circle.normal, GEOM_TOL_MM)?;
    let center_delta = sub(circle.center_mm, bend.center_mm);
    let axial = dot(center_delta, bend.axis);
    let radial_center = sub(center_delta, mul(bend.axis, axial));
    let radial_distance = norm(radial_center);

    if near(circle.radius_mm, bend.minor_radius_mm)
        && axial.abs() <= GEOM_TOL_MM
        && (radial_distance - bend.major_radius_mm).abs() <= GEOM_TOL_MM
    {
        let radial_direction = normalize(radial_center, GEOM_TOL_MM)?;
        let tangent_direction = normalize(cross(bend.axis, radial_direction), GEOM_TOL_MM)?;
        if 1.0 - dot(normal, tangent_direction).abs() <= DIR_TOL {
            return Some(TorusBoundaryCircle::CrossSection { radial_direction });
        }
    }

    if radial_distance <= GEOM_TOL_MM
        && 1.0 - dot(normal, bend.axis).abs() <= DIR_TOL
        && ((circle.radius_mm - bend.major_radius_mm).hypot(axial) - bend.minor_radius_mm).abs()
            <= GEOM_TOL_MM
    {
        return Some(TorusBoundaryCircle::MajorTrace);
    }
    None
}

fn point_on_torus(point: [f64; 3], bend: &TorusBend) -> bool {
    let delta = sub(point, bend.center_mm);
    let axial = dot(delta, bend.axis);
    let radial = sub(delta, mul(bend.axis, axial));
    ((norm(radial) - bend.major_radius_mm).hypot(axial) - bend.minor_radius_mm).abs() <= GEOM_TOL_MM
}

fn torus_major_direction(point: [f64; 3], bend: &TorusBend) -> Option<[f64; 3]> {
    let delta = sub(point, bend.center_mm);
    let axial = dot(delta, bend.axis);
    normalize(sub(delta, mul(bend.axis, axial)), GEOM_TOL_MM)
}

fn unit_angle(a: [f64; 3], b: [f64; 3]) -> Option<f64> {
    let cosine = dot(a, b).clamp(-1.0, 1.0);
    let angle = cosine.acos();
    angle.is_finite().then_some(angle)
}

fn direction_on_minor_arc(
    start: [f64; 3],
    end: [f64; 3],
    point: [f64; 3],
    total_angle: f64,
) -> bool {
    let Some(first) = unit_angle(start, point) else {
        return false;
    };
    let Some(second) = unit_angle(point, end) else {
        return false;
    };
    (first + second - total_angle).abs() <= 1.0e-7
}

fn directed_circle_sweep(edge: &brep::OrientedEdgeUse, circle: &CircleSupport) -> Option<f64> {
    let normal = normalize(circle.normal, GEOM_TOL_MM)?;
    let x_axis = normalize(circle.x_direction, GEOM_TOL_MM)?;
    let y_axis = normalize(cross(normal, x_axis), GEOM_TOL_MM)?;
    let parameter = |point: [f64; 3]| {
        let radial = sub(point, circle.center_mm);
        dot(radial, y_axis).atan2(dot(radial, x_axis))
    };
    let start = parameter(edge.start_mm);
    let end = parameter(edge.end_mm);
    let sweep = if edge.parameter_forward {
        (end - start).rem_euclid(std::f64::consts::TAU)
    } else {
        (start - end).rem_euclid(std::f64::consts::TAU)
    };
    sweep.is_finite().then_some(sweep)
}

fn canonical_cylinder(support: CylinderSupport) -> Option<CylinderRun> {
    if !support.radius_mm.is_finite() || support.radius_mm <= 0.0 {
        return None;
    }
    let direction = canonical_direction(normalize(support.axis, 0.0)?);
    let origin_mm =
        crate::math3::closest_point_on_unit_line_to_origin(support.axis_origin_mm, direction);
    Some(CylinderRun {
        origin_mm,
        direction,
        radius_mm: support.radius_mm,
        face_ids: Vec::new(),
    })
}

fn canonical_torus(support: TorusSupport) -> Option<TorusBend> {
    if !support.major_radius_mm.is_finite()
        || support.major_radius_mm <= 0.0
        || !support.minor_radius_mm.is_finite()
        || support.minor_radius_mm <= 0.0
        || support.center_mm.iter().any(|value| !value.is_finite())
    {
        return None;
    }
    Some(TorusBend {
        center_mm: support.center_mm,
        axis: canonical_direction(normalize(support.axis, 0.0)?),
        major_radius_mm: support.major_radius_mm,
        minor_radius_mm: support.minor_radius_mm,
        face_ids: Vec::new(),
    })
}

fn same_cylinder(a: &CylinderRun, b: &CylinderRun) -> bool {
    near(a.radius_mm, b.radius_mm)
        && 1.0 - dot(a.direction, b.direction).abs() <= DIR_TOL
        && point_to_unit_line_distance(b.origin_mm, a.origin_mm, a.direction) <= GEOM_TOL_MM
}

fn same_torus(a: &TorusBend, b: &TorusBend) -> bool {
    near(a.major_radius_mm, b.major_radius_mm)
        && near(a.minor_radius_mm, b.minor_radius_mm)
        && distance(a.center_mm, b.center_mm) <= GEOM_TOL_MM
        && 1.0 - dot(a.axis, b.axis).abs() <= DIR_TOL
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
    (a - b).abs() <= GEOM_TOL_MM.max(scale * 1.0e-9)
}

fn lex_point(a: [f64; 3], b: [f64; 3]) -> std::cmp::Ordering {
    a[0].total_cmp(&b[0])
        .then(a[1].total_cmp(&b[1]))
        .then(a[2].total_cmp(&b[2]))
}

fn ordered_cycle(
    start: usize,
    next: usize,
    corner_edges: &[Vec<(usize, usize)>],
) -> Option<(Vec<usize>, Vec<usize>)> {
    let first_edge = corner_edges[start]
        .iter()
        .find_map(|&(edge, other)| (other == next).then_some(edge))?;
    let mut corners = vec![start, next];
    let mut edges = vec![first_edge];
    let mut previous = start;
    let mut current = next;

    while corners.len() < corner_edges.len() {
        let &(edge, following) = corner_edges[current]
            .iter()
            .find(|&&(_, other)| other != previous)?;
        if corners.contains(&following) {
            return None;
        }
        edges.push(edge);
        corners.push(following);
        previous = current;
        current = following;
    }

    let closing_edge = corner_edges[current]
        .iter()
        .find_map(|&(edge, other)| (other == start).then_some(edge))?;
    edges.push(closing_edge);
    Some((corners, edges))
}

struct ClosedRoundSweepBuilder<'a> {
    solid_id: u64,
    face_ids: &'a [u64],
    profile_radius_mm: f64,
    corner_radius_mm: f64,
    plane_normal: [f64; 3],
    tori: &'a [TorusBend],
    cylinders: &'a [CylinderRun],
    max_residual_mm: f64,
}

impl ClosedRoundSweepBuilder<'_> {
    fn build(&self, corners: &[usize], edges: &[usize]) -> Option<RecoveredClosedRoundSweep> {
        if corners.len() != 4 || edges.len() != 4 {
            return None;
        }

        let mut max_residual_mm = self.max_residual_mm;
        let mut segments = Vec::with_capacity(8);
        let mut start_mm = None;
        let mut start_tangent = None;

        for i in 0..4 {
            let current_corner = &self.tori[corners[i]];
            let next_corner = &self.tori[corners[(i + 1) % 4]];
            let run = &self.cylinders[edges[i]];
            let next_run = &self.cylinders[edges[(i + 1) % 4]];

            let line_start = project_to_line(current_corner.center_mm, run);
            let line_end = project_to_line(next_corner.center_mm, run);
            let translation = sub(line_end, line_start);
            let tangent = normalize(translation, GEOM_TOL_MM)?;
            if start_mm.is_none() {
                start_mm = Some(line_start);
                start_tangent = Some(tangent);
            }

            let radial_in = sub(line_end, next_corner.center_mm);
            let radial_out = sub(
                project_to_line(next_corner.center_mm, next_run),
                next_corner.center_mm,
            );
            let radial_in_length = norm(radial_in);
            let radial_out_length = norm(radial_out);
            max_residual_mm = max_residual_mm
                .max((radial_in_length - self.corner_radius_mm).abs())
                .max((radial_out_length - self.corner_radius_mm).abs());
            if max_residual_mm > GEOM_TOL_MM {
                return None;
            }

            let angle = dot(self.plane_normal, cross(radial_in, radial_out))
                .atan2(dot(radial_in, radial_out));
            if !angle.is_finite()
                || angle.abs() <= 1.0e-8
                || angle.abs() >= std::f64::consts::PI - 1.0e-8
            {
                return None;
            }

            let sign = angle.signum();
            let expected_in =
                normalize(mul(cross(self.plane_normal, radial_in), sign), GEOM_TOL_MM)?;
            if dot(expected_in, tangent) < 1.0 - 1.0e-7 {
                return None;
            }
            let next_line_start = project_to_line(next_corner.center_mm, next_run);
            let following_corner = &self.tori[corners[(i + 2) % 4]];
            let next_line_end = project_to_line(following_corner.center_mm, next_run);
            let outgoing = normalize(sub(next_line_end, next_line_start), GEOM_TOL_MM)?;
            let expected_out =
                normalize(mul(cross(self.plane_normal, radial_out), sign), GEOM_TOL_MM)?;
            if dot(expected_out, outgoing) < 1.0 - 1.0e-7 {
                return None;
            }

            segments.push(RecoveredSweepSegment::Translation {
                vector_mm: translation,
            });
            segments.push(RecoveredSweepSegment::Rotation {
                axis_origin_mm: next_corner.center_mm,
                axis_direction: self.plane_normal,
                angle_rad: angle,
            });
        }

        let path_tangent = start_tangent?;
        let section_normal = self.plane_normal;
        let section_binormal = normalize(cross(path_tangent, section_normal), GEOM_TOL_MM)?;
        if dot(cross(section_normal, section_binormal), path_tangent) < 1.0 - 1.0e-7 {
            return None;
        }

        Some(RecoveredClosedRoundSweep {
            solid_id: self.solid_id,
            source_face_ids: self.face_ids.to_vec(),
            profile_radius_mm: self.profile_radius_mm,
            path_start_mm: start_mm?,
            profile_x_axis: section_normal,
            profile_y_axis: section_binormal,
            path_tangent,
            segments,
            max_residual_mm,
        })
    }
}

fn project_to_line(point: [f64; 3], line: &CylinderRun) -> [f64; 3] {
    add(
        line.origin_mm,
        mul(
            line.direction,
            dot(sub(point, line.origin_mm), line.direction),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torus_boundary_circle_classification_uses_major_tangent_normal() -> anyhow::Result<()> {
        let bend = TorusBend {
            center_mm: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 1.0],
            major_radius_mm: 2.0,
            minor_radius_mm: 0.25,
            face_ids: Vec::new(),
        };
        let cross_section = CircleSupport {
            center_mm: [2.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            x_direction: [0.0, 0.0, 1.0],
            radius_mm: 0.25,
        };
        let Some(TorusBoundaryCircle::CrossSection { radial_direction }) =
            classify_torus_boundary_circle(&cross_section, &bend)
        else {
            anyhow::bail!("minor torus section was not classified");
        };
        assert_eq!(radial_direction, [1.0, 0.0, 0.0]);

        let outer_trace = CircleSupport {
            center_mm: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            x_direction: [1.0, 0.0, 0.0],
            radius_mm: 2.25,
        };
        assert!(matches!(
            classify_torus_boundary_circle(&outer_trace, &bend),
            Some(TorusBoundaryCircle::MajorTrace)
        ));

        let wrong_normal = CircleSupport {
            normal: [1.0, 0.0, 0.0],
            ..cross_section
        };
        assert!(classify_torus_boundary_circle(&wrong_normal, &bend).is_none());
        Ok(())
    }

    #[test]
    fn rounded_rectangle_builds_exact_line_arc_cycle() -> anyhow::Result<()> {
        let tori = vec![
            TorusBend {
                center_mm: [0.0, 0.0, 0.0],
                axis: [0.0, 0.0, 1.0],
                major_radius_mm: 0.25,
                minor_radius_mm: 0.05,
                face_ids: Vec::new(),
            },
            TorusBend {
                center_mm: [2.0, 0.0, 0.0],
                axis: [0.0, 0.0, 1.0],
                major_radius_mm: 0.25,
                minor_radius_mm: 0.05,
                face_ids: Vec::new(),
            },
            TorusBend {
                center_mm: [2.0, 1.0, 0.0],
                axis: [0.0, 0.0, 1.0],
                major_radius_mm: 0.25,
                minor_radius_mm: 0.05,
                face_ids: Vec::new(),
            },
            TorusBend {
                center_mm: [0.0, 1.0, 0.0],
                axis: [0.0, 0.0, 1.0],
                major_radius_mm: 0.25,
                minor_radius_mm: 0.05,
                face_ids: Vec::new(),
            },
        ];
        let cylinders = vec![
            CylinderRun {
                origin_mm: [0.0, -0.25, 0.0],
                direction: [1.0, 0.0, 0.0],
                radius_mm: 0.05,
                face_ids: Vec::new(),
            },
            CylinderRun {
                origin_mm: [2.25, 0.0, 0.0],
                direction: [0.0, 1.0, 0.0],
                radius_mm: 0.05,
                face_ids: Vec::new(),
            },
            CylinderRun {
                origin_mm: [0.0, 1.25, 0.0],
                direction: [1.0, 0.0, 0.0],
                radius_mm: 0.05,
                face_ids: Vec::new(),
            },
            CylinderRun {
                origin_mm: [-0.25, 0.0, 0.0],
                direction: [0.0, 1.0, 0.0],
                radius_mm: 0.05,
                face_ids: Vec::new(),
            },
        ];
        let builder = ClosedRoundSweepBuilder {
            solid_id: 7,
            face_ids: &[10, 11],
            profile_radius_mm: 0.05,
            corner_radius_mm: 0.25,
            plane_normal: [0.0, 0.0, 1.0],
            tori: &tori,
            cylinders: &cylinders,
            max_residual_mm: 0.0,
        };
        let Some(sweep) = builder.build(&[0, 1, 2, 3], &[0, 1, 2, 3]) else {
            anyhow::bail!("rounded rectangle was not recovered");
        };
        assert_eq!(sweep.segments.len(), 8);
        assert_eq!(sweep.path_start_mm, [0.0, -0.25, 0.0]);
        assert_eq!(sweep.path_tangent, [1.0, 0.0, 0.0]);
        let RecoveredSweepSegment::Translation { vector_mm } = sweep.segments[0] else {
            anyhow::bail!("first segment must be a straight run");
        };
        assert_eq!(vector_mm, [2.0, 0.0, 0.0]);
        let RecoveredSweepSegment::Rotation { angle_rad, .. } = sweep.segments[1] else {
            anyhow::bail!("second segment must be a bend");
        };
        assert!((angle_rad - std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);
        assert!(sweep.max_residual_mm <= GEOM_TOL_MM);
        Ok(())
    }

    #[test]
    fn cycle_order_is_deterministic() -> anyhow::Result<()> {
        let edges = vec![
            vec![(0, 1), (3, 3)],
            vec![(0, 0), (1, 2)],
            vec![(1, 1), (2, 3)],
            vec![(2, 2), (3, 0)],
        ];
        let Some((corners, runs)) = ordered_cycle(0, 1, &edges) else {
            anyhow::bail!("cycle was not recovered");
        };
        assert_eq!(corners, vec![0, 1, 2, 3]);
        assert_eq!(runs, vec![0, 1, 2, 3]);
        Ok(())
    }
}
