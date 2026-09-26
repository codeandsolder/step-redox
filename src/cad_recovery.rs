use crate::cad_ir::{
    Axis3, BrepFallback, CadModel, CadNode, Curve2d, NodeId, PatternSpec, Profile2d, ProfileLoop,
    ProofStatus, Provenance, RigidTransform,
};
use crate::patterns::InstancePattern;
use crate::profile_curves::RecoveredProfileCurve;
use crate::solid_extrusions::RecoveredSolidExtrusion;
use crate::solid_revolutions::{
    MAX_REVOLUTION_SOURCE_UNCERTAINTY_MM, REVOLUTION_SOURCE_SUPPORT_TOL_MM,
    RecoveredSolidRevolution,
};
use anyhow::{Result, bail};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CadFragment {
    pub source: CadFragmentSource,
    pub model: CadModel,
    pub root: NodeId,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CadFragmentSource {
    InstancePattern {
        parent_representation: u64,
        representation_map: u64,
        item_ids: Vec<u64>,
    },
    SolidExtrusion {
        solid_id: u64,
        cap_face_ids: [u64; 2],
        side_face_ids: Vec<u64>,
    },
    SolidRevolution {
        solid_id: u64,
        face_ids: Vec<u64>,
    },
}

/// Recover a constructive CAD fragment from a geometrically-proven solid extrusion.
///
/// The sketch is expressed in a canonical local XY frame and extruded along local +Z.
/// A rigid transform then places that constructive body back into source coordinates.
pub fn recover_solid_extrusion_fragment(
    extrusion: &RecoveredSolidExtrusion,
) -> Result<CadFragment> {
    validate_solid_extrusion(extrusion)?;

    let mut model = CadModel::new();
    let profile = recovered_extrusion_profile(extrusion)?;
    let body = model.add_node(CadNode::Extrude {
        profile,
        vector_mm: [0.0, 0.0, extrusion.height_mm],
    });

    let profile_edge_count = extrusion
        .profile_loops()
        .flatten()
        .map(|curve| curve.source_edge_ids().len())
        .sum::<usize>();
    let mut source_entity_ids =
        Vec::with_capacity(extrusion.side_face_ids.len() + profile_edge_count + 3);
    source_entity_ids.push(extrusion.solid_id);
    source_entity_ids.extend(extrusion.cap_face_ids);
    source_entity_ids.extend(extrusion.side_face_ids.iter().copied());
    source_entity_ids.extend(
        extrusion
            .profile_loops()
            .flatten()
            .flat_map(|curve| curve.source_edge_ids().iter().copied()),
    );
    source_entity_ids.sort_unstable();
    source_entity_ids.dedup();

    let proof = Provenance {
        source_entity_ids: source_entity_ids.clone(),
        proof: ProofStatus::WithinTolerance,
        max_residual_mm: Some(extrusion.max_residual_mm),
    };
    model.set_provenance(body, proof.clone())?;

    let root = model.add_node(CadNode::Transform {
        transform: local_frame_transform(
            extrusion.origin_mm,
            extrusion.x_axis,
            extrusion.y_axis,
            extrusion.z_axis,
        ),
        child: body,
    });
    model.set_provenance(root, proof)?;
    model.add_root(root)?;
    model.validate()?;

    Ok(CadFragment {
        source: CadFragmentSource::SolidExtrusion {
            solid_id: extrusion.solid_id,
            cap_face_ids: extrusion.cap_face_ids,
            side_face_ids: extrusion.side_face_ids.clone(),
        },
        model,
        root,
    })
}

pub fn recover_solid_extrusion_fragments(
    extrusions: &[RecoveredSolidExtrusion],
) -> Result<Vec<CadFragment>> {
    extrusions
        .iter()
        .map(recover_solid_extrusion_fragment)
        .collect()
}

/// Recover a constructive CAD fragment from a geometrically-proven full revolution.
///
/// The detector expresses the meridian sketch directly in local [radius, axial]
/// coordinates. Local +Y is the revolution axis; the rigid transform maps local
/// +X to the detector's deterministic radial direction and local +Y to the world axis.
pub fn recover_solid_revolution_fragment(
    revolution: &RecoveredSolidRevolution,
) -> Result<CadFragment> {
    validate_solid_revolution(revolution)?;

    let mut model = CadModel::new();
    let profile = Profile2d {
        loops: vec![ProfileLoop {
            curves: revolution
                .profile_curves
                .iter()
                .map(recovered_profile_curve_to_ir)
                .collect(),
        }],
    };
    let body = model.add_node(CadNode::Revolve {
        profile,
        axis: Axis3 {
            origin_mm: [0.0, 0.0, 0.0],
            direction: [0.0, 1.0, 0.0],
        },
        angle_rad: std::f64::consts::TAU,
    });

    let mut source_entity_ids = Vec::with_capacity(revolution.face_ids.len() + 1);
    source_entity_ids.push(revolution.solid_id);
    source_entity_ids.extend(revolution.face_ids.iter().copied());
    source_entity_ids.sort_unstable();
    source_entity_ids.dedup();

    let proof = Provenance {
        source_entity_ids,
        proof: ProofStatus::WithinTolerance,
        max_residual_mm: Some(revolution.max_residual_mm),
    };
    model.set_provenance(body, proof.clone())?;

    let z_axis = cross(revolution.radial_direction, revolution.axis_direction);
    let root = model.add_node(CadNode::Transform {
        transform: local_frame_transform(
            revolution.axis_origin_mm,
            revolution.radial_direction,
            revolution.axis_direction,
            z_axis,
        ),
        child: body,
    });
    model.set_provenance(root, proof)?;
    model.add_root(root)?;
    model.validate()?;

    Ok(CadFragment {
        source: CadFragmentSource::SolidRevolution {
            solid_id: revolution.solid_id,
            face_ids: revolution.face_ids.clone(),
        },
        model,
        root,
    })
}

pub fn recover_solid_revolution_fragments(
    revolutions: &[RecoveredSolidRevolution],
) -> Result<Vec<CadFragment>> {
    revolutions
        .iter()
        .map(recover_solid_revolution_fragment)
        .collect()
}

fn recovered_extrusion_profile(extrusion: &RecoveredSolidExtrusion) -> Result<Profile2d> {
    let loops = extrusion
        .profile_loops()
        .map(|curves| {
            let curves = curves
                .iter()
                .map(recovered_profile_curve_to_ir)
                .collect::<Vec<_>>();
            ProfileLoop { curves }
        })
        .collect::<Vec<_>>();
    Ok(Profile2d { loops })
}

fn recovered_profile_curve_to_ir(curve: &RecoveredProfileCurve) -> Curve2d {
    match curve {
        RecoveredProfileCurve::Line {
            start_mm, end_mm, ..
        } => Curve2d::Line {
            start_mm: *start_mm,
            end_mm: *end_mm,
        },
        RecoveredProfileCurve::CircleArc {
            center_mm,
            radius_mm,
            start_angle_rad,
            end_angle_rad,
            ..
        } => Curve2d::CircleArc {
            center_mm: *center_mm,
            radius_mm: *radius_mm,
            start_angle_rad: *start_angle_rad,
            end_angle_rad: *end_angle_rad,
        },
        RecoveredProfileCurve::Bezier {
            control_points_mm, ..
        } => Curve2d::Bezier {
            control_points_mm: control_points_mm.clone(),
        },
        RecoveredProfileCurve::BSpline {
            degree,
            control_points_mm,
            knots,
            weights,
            ..
        } => Curve2d::BSpline {
            degree: *degree,
            control_points_mm: control_points_mm.clone(),
            knots: knots.clone(),
            weights: weights.clone(),
        },
    }
}

fn validate_solid_revolution(revolution: &RecoveredSolidRevolution) -> Result<()> {
    if revolution.profile_curves.is_empty() {
        bail!("solid revolution needs a non-empty meridian profile");
    }
    for curve in &revolution.profile_curves {
        validate_profile_curve_geometry(curve)?;
        match curve {
            RecoveredProfileCurve::Line {
                start_mm, end_mm, ..
            } => {
                if start_mm[0] < -1.0e-7 || end_mm[0] < -1.0e-7 {
                    bail!("solid revolution profile contains a negative radius");
                }
            }
            RecoveredProfileCurve::CircleArc {
                center_mm,
                radius_mm,
                start_angle_rad,
                end_angle_rad,
                ..
            } => {
                let sweep = end_angle_rad - start_angle_rad;
                if sweep.abs() <= 1.0e-12 || sweep.abs() > std::f64::consts::TAU + 1.0e-9 {
                    bail!("solid revolution circular meridian has invalid sweep");
                }
                if circle_arc_min_radius(center_mm[0], *radius_mm, *start_angle_rad, *end_angle_rad)
                    < -1.0e-7
                {
                    bail!("solid revolution circular meridian crosses negative radius");
                }
            }
            RecoveredProfileCurve::Bezier { .. } | RecoveredProfileCurve::BSpline { .. } => {
                bail!("solid revolution spline meridians are not yet supported");
            }
        }
    }
    for index in 0..revolution.profile_curves.len() {
        let current = &revolution.profile_curves[index];
        let next = &revolution.profile_curves[(index + 1) % revolution.profile_curves.len()];
        let Some(end) = current.end_point() else {
            bail!("solid revolution profile curve has no endpoint");
        };
        let Some(start) = next.start_point() else {
            bail!("solid revolution profile curve has no start point");
        };
        if ((end[0] - start[0]).powi(2) + (end[1] - start[1]).powi(2)).sqrt() > 1.0e-7 {
            bail!("solid revolution profile curves are not topologically continuous");
        }
    }

    if revolution
        .axis_origin_mm
        .iter()
        .any(|value| !value.is_finite())
        || revolution
            .axis_direction
            .iter()
            .any(|value| !value.is_finite())
        || revolution
            .radial_direction
            .iter()
            .any(|value| !value.is_finite())
    {
        bail!("solid revolution frame contains non-finite values");
    }
    if (norm(revolution.axis_direction) - 1.0).abs() > 1.0e-10
        || (norm(revolution.radial_direction) - 1.0).abs() > 1.0e-10
        || dot(revolution.axis_direction, revolution.radial_direction).abs() > 1.0e-10
    {
        bail!("solid revolution axis/radial basis is not orthonormal");
    }
    let z_axis = cross(revolution.radial_direction, revolution.axis_direction);
    if (norm(z_axis) - 1.0).abs() > 1.0e-10 {
        bail!("solid revolution frame does not define a unit profile normal");
    }

    if !revolution.source_tolerance_mm.is_finite()
        || revolution.source_tolerance_mm < REVOLUTION_SOURCE_SUPPORT_TOL_MM
        || revolution.source_tolerance_mm > MAX_REVOLUTION_SOURCE_UNCERTAINTY_MM
    {
        bail!("solid revolution has invalid source tolerance");
    }
    if !revolution.max_residual_mm.is_finite()
        || revolution.max_residual_mm < 0.0
        || revolution.max_residual_mm > revolution.source_tolerance_mm + 1.0e-15
    {
        bail!("solid revolution has invalid proof residual");
    }
    Ok(())
}

fn validate_solid_extrusion(extrusion: &RecoveredSolidExtrusion) -> Result<()> {
    if extrusion.profile_curves.is_empty() {
        bail!("solid extrusion needs a non-empty outer profile loop");
    }
    for (loop_index, curves) in extrusion.profile_loops().enumerate() {
        if curves.is_empty() {
            bail!("solid extrusion profile loop {loop_index} is empty");
        }
        for curve in curves {
            validate_recovered_profile_curve(curve)?;
        }
    }
    if !extrusion.height_mm.is_finite() || extrusion.height_mm <= 0.0 {
        bail!("solid extrusion height must be finite and positive");
    }
    if !extrusion.max_residual_mm.is_finite()
        || extrusion.max_residual_mm < 0.0
        || extrusion.max_residual_mm > 1.0e-7 + 1.0e-15
    {
        bail!("solid extrusion has invalid proof residual");
    }
    for vector in [
        extrusion.origin_mm,
        extrusion.x_axis,
        extrusion.y_axis,
        extrusion.z_axis,
    ] {
        if vector.iter().any(|value| !value.is_finite()) {
            bail!("solid extrusion frame contains non-finite values");
        }
    }
    for axis in [extrusion.x_axis, extrusion.y_axis, extrusion.z_axis] {
        if (norm(axis) - 1.0).abs() > 1.0e-10 {
            bail!("solid extrusion frame axis is not unit length");
        }
    }
    if dot(extrusion.x_axis, extrusion.y_axis).abs() > 1.0e-10
        || dot(extrusion.x_axis, extrusion.z_axis).abs() > 1.0e-10
        || dot(extrusion.y_axis, extrusion.z_axis).abs() > 1.0e-10
    {
        bail!("solid extrusion frame is not orthogonal");
    }
    let handedness = dot(cross(extrusion.x_axis, extrusion.y_axis), extrusion.z_axis);
    if (handedness - 1.0).abs() > 1.0e-10 {
        bail!("solid extrusion frame is not right-handed");
    }
    Ok(())
}

fn validate_recovered_profile_curve(curve: &RecoveredProfileCurve) -> Result<()> {
    if curve.source_edge_ids().is_empty() {
        bail!("recovered profile curve has no source edges");
    }
    validate_profile_curve_geometry(curve)
}

fn validate_profile_curve_geometry(curve: &RecoveredProfileCurve) -> Result<()> {
    let finite_point = |point: &[f64; 2]| point.iter().all(|value| value.is_finite());
    match curve {
        RecoveredProfileCurve::Line {
            start_mm, end_mm, ..
        } => {
            if !finite_point(start_mm) || !finite_point(end_mm) || start_mm == end_mm {
                bail!("recovered line has invalid endpoints");
            }
        }
        RecoveredProfileCurve::CircleArc {
            center_mm,
            radius_mm,
            start_angle_rad,
            end_angle_rad,
            ..
        } => {
            if !finite_point(center_mm)
                || !radius_mm.is_finite()
                || *radius_mm <= 0.0
                || !start_angle_rad.is_finite()
                || !end_angle_rad.is_finite()
            {
                bail!("recovered circle arc has invalid geometry");
            }
        }
        RecoveredProfileCurve::Bezier {
            control_points_mm, ..
        } => {
            if control_points_mm.len() < 2 || !control_points_mm.iter().all(finite_point) {
                bail!("recovered Bezier curve has invalid control points");
            }
        }
        RecoveredProfileCurve::BSpline {
            degree,
            control_points_mm,
            knots,
            weights,
            ..
        } => {
            let Some(min_control_points) = degree.checked_add(1) else {
                bail!("recovered B-spline degree overflows dimensions");
            };
            let Some(expected_knots) = control_points_mm.len().checked_add(min_control_points)
            else {
                bail!("recovered B-spline knot count overflows dimensions");
            };
            let endpoint_clamped = min_control_points.checked_mul(2).is_some_and(|min_knots| {
                knots.len() >= min_knots
                    && knots[..min_control_points]
                        .iter()
                        .all(|knot| *knot == knots[0])
                    && knots[knots.len() - min_control_points..]
                        .iter()
                        .all(|knot| *knot == knots[knots.len() - 1])
            });
            if *degree == 0
                || control_points_mm.len() < min_control_points
                || knots.len() != expected_knots
                || !control_points_mm.iter().all(finite_point)
                || !knots.iter().all(|value| value.is_finite())
                || knots.windows(2).any(|pair| pair[0] > pair[1])
                || knots.first() == knots.last()
                || !endpoint_clamped
            {
                bail!("recovered B-spline has invalid dimensions or knots");
            }
            if let Some(weights) = weights
                && (weights.len() != control_points_mm.len()
                    || weights
                        .iter()
                        .any(|weight| !weight.is_finite() || *weight <= 0.0))
            {
                bail!("recovered B-spline has invalid weights");
            }
        }
    }
    Ok(())
}

fn circle_arc_min_radius(center_radius: f64, radius: f64, start_angle: f64, end_angle: f64) -> f64 {
    let mut minimum =
        (center_radius + radius * start_angle.cos()).min(center_radius + radius * end_angle.cos());
    if angle_on_sweep(std::f64::consts::PI, start_angle, end_angle) {
        minimum = minimum.min(center_radius - radius);
    }
    minimum
}

fn angle_on_sweep(angle: f64, start: f64, end: f64) -> bool {
    let sweep = end - start;
    if sweep >= 0.0 {
        (angle - start).rem_euclid(std::f64::consts::TAU) <= sweep + 1.0e-12
    } else {
        (start - angle).rem_euclid(std::f64::consts::TAU) <= -sweep + 1.0e-12
    }
}

fn local_frame_transform(
    origin: [f64; 3],
    x_axis: [f64; 3],
    y_axis: [f64; 3],
    z_axis: [f64; 3],
) -> RigidTransform {
    RigidTransform {
        matrix: [
            [x_axis[0], y_axis[0], z_axis[0], origin[0]],
            [x_axis[1], y_axis[1], z_axis[1], origin[1]],
            [x_axis[2], y_axis[2], z_axis[2], origin[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
}

/// Recover a constructive CAD fragment from an already-proven STEP instance pattern.
///
/// The child is the first actual MAPPED_ITEM occurrence, preserved as an exact B-rep
/// fallback leaf. This is deliberate: using only the REPRESENTATION_MAP would lose
/// the STEP mapping-origin transform when that origin is not global zero. The pattern
/// node then describes only the repeated translation lattice.
pub fn recover_instance_pattern_fragment(pattern: &InstancePattern) -> Result<CadFragment> {
    validate_instance_pattern(pattern)?;

    let first_item = pattern.item_ids[0];
    let mut model = CadModel::new();
    let child = model.add_node(CadNode::BrepFallback(BrepFallback {
        source_entity_ids: vec![first_item],
        estimated_faces: 0,
        estimated_edges: 0,
        estimated_control_points: 0,
    }));
    model.set_provenance(
        child,
        Provenance {
            source_entity_ids: vec![first_item, pattern.representation_map],
            proof: ProofStatus::Exact,
            max_residual_mm: Some(0.0),
        },
    )?;

    let (pattern_spec, canonicalization_residual_mm) = pattern_spec(pattern)?;
    let root = model.add_node(CadNode::Pattern {
        pattern: pattern_spec,
        child,
    });

    let mut source_entity_ids = Vec::with_capacity(pattern.item_ids.len() + 2);
    source_entity_ids.push(pattern.parent_representation);
    source_entity_ids.push(pattern.representation_map);
    source_entity_ids.extend(pattern.item_ids.iter().copied());
    source_entity_ids.sort_unstable();
    source_entity_ids.dedup();

    model.set_provenance(
        root,
        Provenance {
            source_entity_ids,
            // InstancePattern quantizes the shared orientation before grouping.
            // A zero positional residual therefore still does not prove a bit-exact
            // orientation identity across every source occurrence.
            proof: ProofStatus::WithinTolerance,
            max_residual_mm: Some(pattern.max_residual_mm + canonicalization_residual_mm),
        },
    )?;
    model.add_root(root)?;
    model.validate()?;

    Ok(CadFragment {
        source: CadFragmentSource::InstancePattern {
            parent_representation: pattern.parent_representation,
            representation_map: pattern.representation_map,
            item_ids: pattern.item_ids.clone(),
        },
        model,
        root,
    })
}

pub fn recover_instance_pattern_fragments(
    patterns: &[InstancePattern],
) -> Result<Vec<CadFragment>> {
    patterns
        .iter()
        .map(recover_instance_pattern_fragment)
        .collect()
}

fn validate_instance_pattern(pattern: &InstancePattern) -> Result<()> {
    if pattern.item_ids.len() < 2 {
        bail!("instance pattern needs at least two items");
    }
    if pattern.occupancy.len() != pattern.item_ids.len() {
        bail!("instance pattern occupancy/item count mismatch");
    }
    if !pattern.tolerance_mm.is_finite() || pattern.tolerance_mm <= 0.0 {
        bail!("instance pattern has invalid tolerance");
    }
    if !pattern.max_residual_mm.is_finite()
        || pattern.max_residual_mm < 0.0
        || pattern.max_residual_mm > pattern.tolerance_mm + 1.0e-15
    {
        bail!("instance pattern has invalid residual");
    }
    if pattern.origin.iter().any(|value| !value.is_finite()) {
        bail!("instance pattern origin is not finite");
    }
    if pattern.occupancy.first() != Some(&[0, 0]) {
        bail!("first pattern item is not the canonical lattice origin occurrence");
    }

    let dimensions = usize::from(pattern.dimension);
    if !(1..=2).contains(&dimensions) {
        bail!("only one- and two-dimensional instance patterns are supported");
    }
    if pattern.basis.len() != dimensions
        || pattern.pitch.len() != dimensions
        || pattern.grid_shape.len() != dimensions
    {
        bail!("instance pattern lattice metadata has inconsistent dimensions");
    }
    if pattern.grid_shape.contains(&0) {
        bail!("instance pattern contains a zero-sized lattice dimension");
    }

    let declared_cells = pattern.grid_shape.iter().copied().product::<usize>();
    let expected_fill = pattern.item_ids.len() as f64 / declared_cells as f64;
    if !pattern.fill_ratio.is_finite() || (pattern.fill_ratio - expected_fill).abs() > 1.0e-12 {
        bail!("instance pattern fill ratio disagrees with occupancy/grid shape");
    }

    for (axis, (&basis, &pitch)) in pattern.basis.iter().zip(&pattern.pitch).enumerate() {
        if basis.iter().any(|value| !value.is_finite()) || !pitch.is_finite() || pitch <= 0.0 {
            bail!("instance pattern axis {axis} is not finite/positive");
        }
        let basis_norm = norm(basis);
        let tolerance = 1.0e-10_f64.max(pitch.abs() * 1.0e-10);
        if (basis_norm - pitch).abs() > tolerance {
            bail!(
                "instance pattern axis {axis} basis norm {basis_norm} disagrees with pitch {pitch}"
            );
        }
    }

    let mut occupancy = pattern.occupancy.clone();
    occupancy.sort_unstable();
    occupancy.dedup();
    if occupancy.len() != pattern.occupancy.len() {
        bail!("instance pattern has duplicate occupied lattice sites");
    }

    let nu = pattern.grid_shape[0] as i64;
    let nv = if dimensions == 2 {
        pattern.grid_shape[1] as i64
    } else {
        1
    };
    for &[u, v] in &pattern.occupancy {
        if u < 0 || u >= nu || v < 0 || v >= nv {
            bail!("instance pattern occupancy lies outside its declared grid");
        }
        if dimensions == 1 && v != 0 {
            bail!("one-dimensional instance pattern has a nonzero second lattice coordinate");
        }
    }

    Ok(())
}

fn pattern_spec(pattern: &InstancePattern) -> Result<(PatternSpec, f64)> {
    let remaining_budget_mm = (pattern.tolerance_mm - pattern.max_residual_mm).max(0.0);
    let dimensions = usize::from(pattern.dimension);
    let per_axis_budget_mm = remaining_budget_mm / dimensions as f64;
    let mut canonical_basis = Vec::with_capacity(dimensions);
    let mut canonicalization_residual_mm = 0.0;
    for axis in 0..dimensions {
        let span = pattern.grid_shape[axis];
        let (basis, added_residual) =
            canonicalize_step(pattern.basis[axis], span, per_axis_budget_mm);
        canonical_basis.push(basis);
        canonicalization_residual_mm += added_residual;
    }

    let spec = match pattern.dimension {
        1 => {
            let full_contiguous = pattern.grid_shape[0] == pattern.item_ids.len()
                && pattern
                    .occupancy
                    .iter()
                    .enumerate()
                    .all(|(index, site)| *site == [index as i64, 0]);
            if full_contiguous {
                PatternSpec::Linear {
                    count: pattern.item_ids.len(),
                    step_mm: canonical_basis[0],
                }
            } else {
                PatternSpec::Grid {
                    counts: [pattern.grid_shape[0], 1, 1],
                    step_vectors_mm: [canonical_basis[0], [0.0; 3], [0.0; 3]],
                    occupancy: Some(
                        pattern
                            .occupancy
                            .iter()
                            .map(|site| [site[0] as usize, 0, 0])
                            .collect(),
                    ),
                }
            }
        }
        2 => {
            let expected = pattern.grid_shape[0].saturating_mul(pattern.grid_shape[1]);
            let occupancy = if expected == pattern.item_ids.len()
                && covers_full_grid(
                    &pattern.occupancy,
                    pattern.grid_shape[0],
                    pattern.grid_shape[1],
                ) {
                None
            } else {
                Some(
                    pattern
                        .occupancy
                        .iter()
                        .map(|site| [site[0] as usize, site[1] as usize, 0])
                        .collect(),
                )
            };
            PatternSpec::Grid {
                counts: [pattern.grid_shape[0], pattern.grid_shape[1], 1],
                step_vectors_mm: [canonical_basis[0], canonical_basis[1], [0.0; 3]],
                occupancy,
            }
        }
        _ => unreachable!("validated above"),
    };
    Ok((spec, canonicalization_residual_mm))
}

fn canonicalize_step(step: [f64; 3], span: usize, budget_mm: f64) -> ([f64; 3], f64) {
    let repeats = span.saturating_sub(1) as f64;
    if repeats == 0.0 || budget_mm <= 0.0 {
        return (step, 0.0);
    }

    let max_abs = step.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let start_exponent = if max_abs > 0.0 {
        max_abs.log10().ceil() as i32
    } else {
        0
    };

    for exponent in (-15..=start_exponent).rev() {
        let quantum = 10_f64.powi(exponent);
        let mut candidate = step;
        for value in &mut candidate {
            *value = (*value / quantum).round() * quantum;
            if *value == -0.0 {
                *value = 0.0;
            }
        }
        let added_residual_mm = norm(sub(candidate, step)) * repeats;
        if added_residual_mm <= budget_mm + 1.0e-15 {
            return (candidate, added_residual_mm);
        }
    }

    (step, 0.0)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn covers_full_grid(occupancy: &[[i64; 2]], nu: usize, nv: usize) -> bool {
    if occupancy.len() != nu.saturating_mul(nv) {
        return false;
    }
    let mut expected = Vec::with_capacity(occupancy.len());
    for u in 0..nu {
        for v in 0..nv {
            expected.push([u as i64, v as i64]);
        }
    }
    let mut actual = occupancy.to_vec();
    actual.sort_unstable();
    expected.sort_unstable();
    actual == expected
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_pattern(occupancy: Vec<[i64; 2]>, span: usize, residual: f64) -> InstancePattern {
        let fill_ratio = occupancy.len() as f64 / span as f64;
        InstancePattern {
            parent_representation: 100,
            representation_map: 200,
            item_ids: (0..occupancy.len())
                .map(|index| 300 + index as u64)
                .collect(),
            style_signature: vec![vec![700]],
            dimension: 1,
            origin: [12.0, 3.0, -1.0],
            basis: vec![[2.54, 0.0, 0.0]],
            pitch: vec![2.54],
            occupancy,
            grid_shape: vec![span],
            fill_ratio,
            tolerance_mm: 1.0e-7,
            max_residual_mm: residual,
        }
    }

    #[test]
    fn solid_extrusion_becomes_local_extrude_plus_world_frame() -> Result<()> {
        let extrusion = RecoveredSolidExtrusion {
            solid_id: 10,
            cap_face_ids: [20, 21],
            side_face_ids: vec![30, 31, 32, 33],
            profile_curves: vec![
                RecoveredProfileCurve::Line {
                    source_edge_ids: vec![100],
                    start_mm: [0.0, 0.0],
                    end_mm: [4.0, 0.0],
                },
                RecoveredProfileCurve::Line {
                    source_edge_ids: vec![101],
                    start_mm: [4.0, 0.0],
                    end_mm: [4.0, 2.0],
                },
                RecoveredProfileCurve::Line {
                    source_edge_ids: vec![102],
                    start_mm: [4.0, 2.0],
                    end_mm: [0.0, 2.0],
                },
                RecoveredProfileCurve::Line {
                    source_edge_ids: vec![103],
                    start_mm: [0.0, 2.0],
                    end_mm: [0.0, 0.0],
                },
            ],
            inner_profile_loops: Vec::new(),
            origin_mm: [12.0, -3.0, 7.0],
            x_axis: [0.0, 1.0, 0.0],
            y_axis: [0.0, 0.0, 1.0],
            z_axis: [1.0, 0.0, 0.0],
            height_mm: 5.0,
            max_residual_mm: 2.0e-12,
        };

        let serialized = serde_json::to_value(&extrusion)?;
        assert!(serialized.get("inner_profile_loops").is_none());

        let fragment = recover_solid_extrusion_fragment(&extrusion)?;
        let CadNode::Transform { transform, child } = fragment.model.node(fragment.root)? else {
            panic!("expected transform root");
        };
        assert_eq!(
            transform.matrix,
            [
                [0.0, 0.0, 1.0, 12.0],
                [1.0, 0.0, 0.0, -3.0],
                [0.0, 1.0, 0.0, 7.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        );
        let CadNode::Extrude { profile, vector_mm } = fragment.model.node(*child)? else {
            panic!("expected local extrusion child");
        };
        assert_eq!(*vector_mm, [0.0, 0.0, 5.0]);
        assert_eq!(
            profile.single_polygon_points(),
            Some(vec![[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])
        );
        assert_eq!(
            fragment.model.provenance[&fragment.root].proof,
            ProofStatus::WithinTolerance
        );
        assert_eq!(
            fragment.model.provenance[&fragment.root].max_residual_mm,
            Some(2.0e-12)
        );

        let mut holed = extrusion.clone();
        holed.inner_profile_loops = vec![vec![
            RecoveredProfileCurve::Line {
                source_edge_ids: vec![200],
                start_mm: [1.0, 0.5],
                end_mm: [1.0, 1.5],
            },
            RecoveredProfileCurve::Line {
                source_edge_ids: vec![201],
                start_mm: [1.0, 1.5],
                end_mm: [3.0, 1.5],
            },
            RecoveredProfileCurve::Line {
                source_edge_ids: vec![202],
                start_mm: [3.0, 1.5],
                end_mm: [3.0, 0.5],
            },
            RecoveredProfileCurve::Line {
                source_edge_ids: vec![203],
                start_mm: [3.0, 0.5],
                end_mm: [1.0, 0.5],
            },
        ]];
        let holed_fragment = recover_solid_extrusion_fragment(&holed)?;
        let CadNode::Transform { child, .. } = holed_fragment.model.node(holed_fragment.root)?
        else {
            panic!("expected transform root");
        };
        let CadNode::Extrude { profile, .. } = holed_fragment.model.node(*child)? else {
            panic!("expected local extrusion child");
        };
        assert_eq!(profile.loops.len(), 2);
        for source_id in 200..=203 {
            assert!(
                holed_fragment.model.provenance[&holed_fragment.root]
                    .source_entity_ids
                    .contains(&source_id)
            );
        }
        assert!(
            serde_json::to_value(&holed)?
                .get("inner_profile_loops")
                .is_some()
        );
        Ok(())
    }

    #[test]
    fn solid_revolution_becomes_local_full_revolve_plus_world_frame() -> Result<()> {
        let revolution = RecoveredSolidRevolution {
            solid_id: 50,
            face_ids: vec![60, 61, 62],
            profile_curves: vec![
                RecoveredProfileCurve::Line {
                    source_edge_ids: Vec::new(),
                    start_mm: [0.0, 0.0],
                    end_mm: [2.0, 0.0],
                },
                RecoveredProfileCurve::Line {
                    source_edge_ids: Vec::new(),
                    start_mm: [2.0, 0.0],
                    end_mm: [2.0, 3.0],
                },
                RecoveredProfileCurve::Line {
                    source_edge_ids: Vec::new(),
                    start_mm: [2.0, 3.0],
                    end_mm: [0.0, 3.0],
                },
                RecoveredProfileCurve::Line {
                    source_edge_ids: Vec::new(),
                    start_mm: [0.0, 3.0],
                    end_mm: [0.0, 0.0],
                },
            ],
            axis_origin_mm: [10.0, 20.0, 30.0],
            axis_direction: [0.0, 1.0, 0.0],
            radial_direction: [1.0, 0.0, 0.0],
            max_residual_mm: 0.5 * REVOLUTION_SOURCE_SUPPORT_TOL_MM,
            source_tolerance_mm: REVOLUTION_SOURCE_SUPPORT_TOL_MM,
        };

        let fragment = recover_solid_revolution_fragment(&revolution)?;
        let CadNode::Transform { transform, child } = fragment.model.node(fragment.root)? else {
            panic!("expected transform root");
        };
        assert_eq!(
            transform.matrix,
            [
                [1.0, 0.0, 0.0, 10.0],
                [0.0, 1.0, 0.0, 20.0],
                [0.0, 0.0, 1.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        );
        let CadNode::Revolve {
            profile,
            axis,
            angle_rad,
        } = fragment.model.node(*child)?
        else {
            panic!("expected local revolution child");
        };
        assert_eq!(profile.single_polygon_points(), revolution.polygon_points());
        assert_eq!(axis.origin_mm, [0.0, 0.0, 0.0]);
        assert_eq!(axis.direction, [0.0, 1.0, 0.0]);
        assert_eq!(*angle_rad, std::f64::consts::TAU);
        assert_eq!(
            fragment.model.provenance[&fragment.root].max_residual_mm,
            Some(0.5 * REVOLUTION_SOURCE_SUPPORT_TOL_MM)
        );

        let mut excessive_residual = revolution.clone();
        excessive_residual.max_residual_mm = 2.0 * REVOLUTION_SOURCE_SUPPORT_TOL_MM;
        assert!(recover_solid_revolution_fragment(&excessive_residual).is_err());

        let mut excessive_tolerance = revolution.clone();
        excessive_tolerance.source_tolerance_mm = 2.0 * MAX_REVOLUTION_SOURCE_UNCERTAINTY_MM;
        assert!(recover_solid_revolution_fragment(&excessive_tolerance).is_err());

        let mut too_strict_tolerance = revolution.clone();
        too_strict_tolerance.source_tolerance_mm = 0.5 * REVOLUTION_SOURCE_SUPPORT_TOL_MM;
        assert!(recover_solid_revolution_fragment(&too_strict_tolerance).is_err());

        #[cfg(feature = "cad-kernel-monstertruck")]
        {
            use crate::cad_kernel::CadKernel;
            let kernel = crate::cad_kernel::monstertruck::MonstertruckKernel;
            let evaluated = kernel.evaluate(&fragment.model, fragment.root)?;
            assert!(kernel.summarize(&evaluated).geometrically_consistent);
            ruststep::parser::parse(&kernel.to_step(&evaluated)?)?;
        }
        Ok(())
    }

    #[test]
    fn malformed_spline_extrusions_fail_closed() {
        let base = RecoveredSolidExtrusion {
            solid_id: 10,
            cap_face_ids: [20, 21],
            side_face_ids: vec![30],
            profile_curves: vec![RecoveredProfileCurve::Bezier {
                source_edge_ids: vec![100],
                control_points_mm: Vec::new(),
            }],
            inner_profile_loops: Vec::new(),
            origin_mm: [0.0, 0.0, 0.0],
            x_axis: [1.0, 0.0, 0.0],
            y_axis: [0.0, 1.0, 0.0],
            z_axis: [0.0, 0.0, 1.0],
            height_mm: 1.0,
            max_residual_mm: 0.0,
        };
        assert!(recover_solid_extrusion_fragment(&base).is_err());

        let mut empty_inner_loop = base.clone();
        empty_inner_loop.profile_curves = vec![
            RecoveredProfileCurve::Line {
                source_edge_ids: vec![104],
                start_mm: [0.0, 0.0],
                end_mm: [1.0, 0.0],
            },
            RecoveredProfileCurve::Line {
                source_edge_ids: vec![105],
                start_mm: [1.0, 0.0],
                end_mm: [0.0, 0.0],
            },
        ];
        empty_inner_loop.inner_profile_loops = vec![Vec::new()];
        assert!(recover_solid_extrusion_fragment(&empty_inner_loop).is_err());

        let mut invalid_bspline = base;
        invalid_bspline.profile_curves = vec![RecoveredProfileCurve::BSpline {
            source_edge_ids: vec![101],
            degree: 2,
            control_points_mm: vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
            knots: vec![0.0, 0.0, 1.0],
            weights: Some(vec![1.0, -1.0, 1.0]),
        }];
        assert!(recover_solid_extrusion_fragment(&invalid_bspline).is_err());

        let mut overflowing_degree = invalid_bspline;
        overflowing_degree.profile_curves = vec![RecoveredProfileCurve::BSpline {
            source_edge_ids: vec![102],
            degree: usize::MAX,
            control_points_mm: vec![[0.0, 0.0], [1.0, 1.0]],
            knots: Vec::new(),
            weights: None,
        }];
        assert!(recover_solid_extrusion_fragment(&overflowing_degree).is_err());

        let mut unclamped = overflowing_degree;
        unclamped.profile_curves = vec![RecoveredProfileCurve::BSpline {
            source_edge_ids: vec![103],
            degree: 2,
            control_points_mm: vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
            knots: vec![0.0, 0.1, 0.2, 0.8, 0.9, 1.0],
            weights: None,
        }];
        assert!(recover_solid_extrusion_fragment(&unclamped).is_err());
    }

    #[test]
    fn full_linear_pattern_becomes_constructive_pattern_over_first_occurrence() -> Result<()> {
        let pattern = linear_pattern(vec![[0, 0], [1, 0], [2, 0], [3, 0]], 4, 0.0);
        let fragment = recover_instance_pattern_fragment(&pattern)?;

        let CadNode::Pattern {
            pattern: PatternSpec::Linear { count, step_mm },
            child,
        } = fragment.model.node(fragment.root)?
        else {
            panic!("expected linear pattern root");
        };
        assert_eq!(*count, 4);
        assert_eq!(*step_mm, [2.54, 0.0, 0.0]);

        let CadNode::BrepFallback(fallback) = fragment.model.node(*child)? else {
            panic!("expected B-rep fallback child");
        };
        assert_eq!(fallback.source_entity_ids, vec![300]);
        assert_eq!(
            fragment.model.provenance[&fragment.root].proof,
            ProofStatus::WithinTolerance
        );
        Ok(())
    }

    #[test]
    fn canonicalizes_floating_lattice_noise_within_far_end_budget() -> Result<()> {
        let mut pattern = linear_pattern((0..24).map(|index| [index, 0]).collect(), 24, 2.4e-13);
        pattern.basis = vec![[
            1.9999999999999893,
            4.440892098500626e-16,
            -4.440892098500626e-16,
        ]];
        pattern.pitch = vec![1.9999999999999893];

        let fragment = recover_instance_pattern_fragment(&pattern)?;
        let CadNode::Pattern {
            pattern: PatternSpec::Linear { step_mm, .. },
            ..
        } = fragment.model.node(fragment.root)?
        else {
            panic!("expected linear pattern root");
        };
        assert_eq!(*step_mm, [2.0, 0.0, 0.0]);
        assert!(
            fragment.model.provenance[&fragment.root]
                .max_residual_mm
                .unwrap()
                <= pattern.tolerance_mm
        );
        Ok(())
    }

    #[test]
    fn sparse_linear_pattern_preserves_holes_as_grid_occupancy() -> Result<()> {
        let pattern = linear_pattern(vec![[0, 0], [2, 0], [4, 0]], 5, 3.0e-8);
        let fragment = recover_instance_pattern_fragment(&pattern)?;

        let CadNode::Pattern {
            pattern: PatternSpec::Grid {
                counts, occupancy, ..
            },
            ..
        } = fragment.model.node(fragment.root)?
        else {
            panic!("expected sparse grid pattern root");
        };
        assert_eq!(*counts, [5, 1, 1]);
        assert_eq!(
            occupancy.as_deref(),
            Some(&[[0, 0, 0], [2, 0, 0], [4, 0, 0]][..])
        );
        assert_eq!(
            fragment.model.provenance[&fragment.root].proof,
            ProofStatus::WithinTolerance
        );
        assert_eq!(
            fragment.model.provenance[&fragment.root].max_residual_mm,
            Some(3.0e-8)
        );
        Ok(())
    }

    #[test]
    fn full_two_dimensional_pattern_becomes_grid() -> Result<()> {
        let pattern = InstancePattern {
            parent_representation: 10,
            representation_map: 20,
            item_ids: vec![30, 31, 32, 33],
            style_signature: Vec::new(),
            dimension: 2,
            origin: [0.0, 0.0, 0.0],
            basis: vec![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            pitch: vec![1.0, 2.0],
            occupancy: vec![[0, 0], [0, 1], [1, 0], [1, 1]],
            grid_shape: vec![2, 2],
            fill_ratio: 1.0,
            tolerance_mm: 1.0e-7,
            max_residual_mm: 0.0,
        };
        let fragment = recover_instance_pattern_fragment(&pattern)?;
        let CadNode::Pattern {
            pattern: PatternSpec::Grid {
                counts, occupancy, ..
            },
            ..
        } = fragment.model.node(fragment.root)?
        else {
            panic!("expected grid pattern root");
        };
        assert_eq!(*counts, [2, 2, 1]);
        assert!(occupancy.is_none());
        Ok(())
    }

    #[test]
    fn malformed_pattern_fails_closed() {
        let mut pattern = linear_pattern(vec![[0, 0], [1, 0]], 2, 0.0);
        pattern.occupancy[0] = [1, 0];
        assert!(recover_instance_pattern_fragment(&pattern).is_err());
    }
}
