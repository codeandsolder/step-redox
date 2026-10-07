use super::{CadFragment, CadFragmentSource};
use crate::cad_ir::{
    Axis3, BooleanOp, CadModel, CadNode, Curve2d, Profile2d, ProfileLoop, ProofStatus, Provenance,
    RigidTransform, SweepPath3d, SweepSegment3d,
};
use crate::math3::{cross, dot, norm};
use crate::profile_curves::RecoveredProfileCurve;
use crate::solid_extrusions::RecoveredSolidExtrusion;
use crate::solid_revolutions::{
    MAX_REVOLUTION_SOURCE_UNCERTAINTY_MM, REVOLUTION_SOURCE_SUPPORT_TOL_MM,
    RecoveredRadialSlotRevolution, RecoveredSolidRevolution,
};
use crate::solid_sweeps::{
    RecoveredClosedRoundSweep, RecoveredOpenRectangularSweep, RecoveredPlanarSweepLeg,
    RecoveredSweepSegment,
};
use anyhow::{Result, bail};

/// Recover a constructive CAD fragment from a geometrically-proven solid extrusion.
///
/// The sketch is expressed in a canonical local XY frame and extruded along local +Z.
/// A rigid transform then places that constructive body back into source coordinates.
///
/// # Errors
/// Returns an error if the recovered extrusion proof is inconsistent or its canonical CAD fragment cannot be constructed.
pub fn recover_solid_extrusion_fragment(
    extrusion: &RecoveredSolidExtrusion,
) -> Result<CadFragment> {
    validate_solid_extrusion(extrusion)?;

    let mut model = CadModel::new();
    let profile = recovered_extrusion_profile(extrusion);
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

///
/// # Errors
/// Returns the first error produced while lowering one of the recovered extrusions.
pub fn recover_solid_extrusion_fragments(
    extrusions: &[RecoveredSolidExtrusion],
) -> Result<Vec<CadFragment>> {
    extrusions
        .iter()
        .map(recover_solid_extrusion_fragment)
        .collect()
}

/// Recover a proven constant-round-section closed composite sweep.
///
/// The detector has already proven a planar cycle of exact cylinder and torus
/// supports. The CAD node keeps that proof as a circular profile plus exact
/// rigid translation/rotation path segments; no tessellation enters the IR.
///
/// # Errors
/// Returns an error if the recovered frame/path is malformed or cannot form valid CAD IR.
pub fn recover_closed_round_sweep_fragment(
    sweep: &RecoveredClosedRoundSweep,
) -> Result<CadFragment> {
    if sweep.solid_id == 0
        || !sweep.profile_radius_mm.is_finite()
        || sweep.profile_radius_mm <= 0.0
        || !sweep.max_residual_mm.is_finite()
        || sweep.max_residual_mm > 1.0e-6
        || sweep.segments.is_empty()
    {
        bail!("closed round sweep proof is invalid");
    }
    for axis in [
        sweep.profile_x_axis,
        sweep.profile_y_axis,
        sweep.path_tangent,
    ] {
        if (norm(axis) - 1.0).abs() > 1.0e-8 {
            bail!("closed round sweep frame is not orthonormal");
        }
    }
    if dot(sweep.profile_x_axis, sweep.profile_y_axis).abs() > 1.0e-8
        || dot(sweep.profile_x_axis, sweep.path_tangent).abs() > 1.0e-8
        || dot(sweep.profile_y_axis, sweep.path_tangent).abs() > 1.0e-8
    {
        bail!("closed round sweep frame axes are not perpendicular");
    }

    let profile = Profile2d {
        loops: vec![ProfileLoop {
            curves: vec![
                Curve2d::CircleArc {
                    center_mm: [0.0, 0.0],
                    radius_mm: sweep.profile_radius_mm,
                    start_angle_rad: 0.0,
                    end_angle_rad: std::f64::consts::PI,
                },
                Curve2d::CircleArc {
                    center_mm: [0.0, 0.0],
                    radius_mm: sweep.profile_radius_mm,
                    start_angle_rad: std::f64::consts::PI,
                    end_angle_rad: std::f64::consts::TAU,
                },
            ],
        }],
    };

    let segments = sweep
        .segments
        .iter()
        .map(|segment| match segment {
            RecoveredSweepSegment::Translation { vector_mm } => SweepSegment3d::Translation {
                vector_mm: *vector_mm,
            },
            RecoveredSweepSegment::Rotation {
                axis_origin_mm,
                axis_direction,
                angle_rad,
            } => SweepSegment3d::Rotation {
                axis: Axis3 {
                    origin_mm: *axis_origin_mm,
                    direction: *axis_direction,
                },
                angle_rad: *angle_rad,
            },
        })
        .collect::<Vec<_>>();

    let mut model = CadModel::new();
    let root = model.add_node(CadNode::Sweep {
        profile,
        path: SweepPath3d {
            initial_transform: local_frame_transform(
                sweep.path_start_mm,
                sweep.profile_x_axis,
                sweep.profile_y_axis,
                sweep.path_tangent,
            ),
            segments,
            closed: true,
        },
    });

    let mut source_entity_ids = Vec::with_capacity(sweep.source_face_ids.len() + 1);
    source_entity_ids.push(sweep.solid_id);
    source_entity_ids.extend(sweep.source_face_ids.iter().copied());
    source_entity_ids.sort_unstable();
    source_entity_ids.dedup();
    model.set_provenance(
        root,
        Provenance {
            source_entity_ids,
            proof: ProofStatus::WithinTolerance,
            max_residual_mm: Some(sweep.max_residual_mm),
        },
    )?;
    model.add_root(root)?;
    model.validate()?;

    Ok(CadFragment {
        source: CadFragmentSource::SolidSweep {
            solid_id: sweep.solid_id,
            face_ids: sweep.source_face_ids.clone(),
            closed: true,
        },
        model,
        root,
    })
}

/// Lower every proven closed round sweep into CAD IR.
///
/// # Errors
/// Returns the first malformed detector result.
pub fn recover_closed_round_sweep_fragments(
    sweeps: &[RecoveredClosedRoundSweep],
) -> Result<Vec<CadFragment>> {
    sweeps
        .iter()
        .map(recover_closed_round_sweep_fragment)
        .collect()
}

/// Recover a one-bend rectangular-section body as two exact planar prisms plus
/// one exact rotational sweep through the source cylindrical sector.
///
/// The straight-leg polygons come directly from paired source faces, so terminal
/// flanges and line-only chamfers remain part of the constructive model rather
/// than being approximated by a constant-width path.
///
/// # Errors
/// Returns an error if the recovered dimensions/frame are malformed or the
/// resulting constructive union is not valid CAD IR.
pub fn recover_open_rectangular_sweep_fragment(
    sweep: &RecoveredOpenRectangularSweep,
) -> Result<CadFragment> {
    if sweep.solid_id == 0
        || !sweep.profile_width_mm.is_finite()
        || sweep.profile_width_mm <= 0.0
        || !sweep.profile_thickness_mm.is_finite()
        || sweep.profile_thickness_mm <= 0.0
        || !sweep.max_residual_mm.is_finite()
        || sweep.max_residual_mm > 1.0e-5
    {
        bail!("open rectangular sweep proof is invalid");
    }
    for axis in [
        sweep.bend_profile_x_axis,
        sweep.bend_profile_y_axis,
        sweep.bend_path_tangent,
    ] {
        if (norm(axis) - 1.0).abs() > 1.0e-8 {
            bail!("open rectangular sweep frame is not orthonormal");
        }
    }
    if dot(sweep.bend_profile_x_axis, sweep.bend_profile_y_axis).abs() > 1.0e-8
        || dot(sweep.bend_profile_x_axis, sweep.bend_path_tangent).abs() > 1.0e-8
        || dot(sweep.bend_profile_y_axis, sweep.bend_path_tangent).abs() > 1.0e-8
        || dot(
            cross(sweep.bend_profile_x_axis, sweep.bend_profile_y_axis),
            sweep.bend_path_tangent,
        ) < 1.0 - 1.0e-8
    {
        bail!("open rectangular sweep frame axes are inconsistent");
    }

    let mut model = CadModel::new();
    let leg_roots = sweep
        .legs
        .iter()
        .map(|leg| add_planar_sweep_leg(&mut model, leg))
        .collect::<Result<Vec<_>>>()?;

    let half_width = sweep.profile_width_mm * 0.5;
    let half_thickness = sweep.profile_thickness_mm * 0.5;
    let bend_profile = Profile2d::polygon(vec![
        [-half_width, -half_thickness],
        [half_width, -half_thickness],
        [half_width, half_thickness],
        [-half_width, half_thickness],
    ])?;
    if sweep.bend_segments.len() != 3 {
        bail!("one-bend rectangular sweep must contain overlap, rotation, overlap");
    }
    let bend_segments = sweep
        .bend_segments
        .iter()
        .map(|segment| match segment {
            RecoveredSweepSegment::Translation { vector_mm } => SweepSegment3d::Translation {
                vector_mm: *vector_mm,
            },
            RecoveredSweepSegment::Rotation {
                axis_origin_mm,
                axis_direction,
                angle_rad,
            } => SweepSegment3d::Rotation {
                axis: Axis3 {
                    origin_mm: *axis_origin_mm,
                    direction: *axis_direction,
                },
                angle_rad: *angle_rad,
            },
        })
        .collect::<Vec<_>>();
    let bend = model.add_node(CadNode::Sweep {
        profile: bend_profile,
        path: SweepPath3d {
            initial_transform: local_frame_transform(
                sweep.bend_path_start_mm,
                sweep.bend_profile_x_axis,
                sweep.bend_profile_y_axis,
                sweep.bend_path_tangent,
            ),
            segments: bend_segments,
            closed: false,
        },
    });

    let root = model.add_node(CadNode::Boolean {
        op: BooleanOp::Union,
        children: vec![leg_roots[0], bend, leg_roots[1]],
    });
    let mut source_entity_ids = Vec::with_capacity(sweep.source_face_ids.len() + 1);
    source_entity_ids.push(sweep.solid_id);
    source_entity_ids.extend(sweep.source_face_ids.iter().copied());
    source_entity_ids.sort_unstable();
    source_entity_ids.dedup();
    model.set_provenance(
        root,
        Provenance {
            source_entity_ids,
            proof: ProofStatus::WithinTolerance,
            max_residual_mm: Some(sweep.max_residual_mm),
        },
    )?;
    model.add_root(root)?;
    model.validate()?;

    Ok(CadFragment {
        source: CadFragmentSource::SolidSweep {
            solid_id: sweep.solid_id,
            face_ids: sweep.source_face_ids.clone(),
            closed: false,
        },
        model,
        root,
    })
}

fn add_planar_sweep_leg(
    model: &mut CadModel,
    leg: &RecoveredPlanarSweepLeg,
) -> Result<crate::cad_ir::NodeId> {
    if !leg.depth_mm.is_finite() || leg.depth_mm <= 0.0 || leg.profile_points_mm.len() < 3 {
        bail!("open rectangular sweep leg is malformed");
    }
    let profile = Profile2d::polygon(leg.profile_points_mm.clone())?;
    let body = model.add_node(CadNode::Extrude {
        profile,
        vector_mm: [0.0, 0.0, leg.depth_mm],
    });
    Ok(model.add_node(CadNode::Transform {
        transform: local_frame_transform(leg.origin_mm, leg.x_axis, leg.y_axis, leg.z_axis),
        child: body,
    }))
}

/// Lower every proven one-bend rectangular sweep into CAD IR.
///
/// # Errors
/// Returns the first malformed detector result.
pub fn recover_open_rectangular_sweep_fragments(
    sweeps: &[RecoveredOpenRectangularSweep],
) -> Result<Vec<CadFragment>> {
    sweeps
        .iter()
        .map(recover_open_rectangular_sweep_fragment)
        .collect()
}

/// Recover a constructive CAD fragment from a geometrically-proven full revolution.
///
/// The detector expresses the meridian sketch directly in local [radius, axial]
/// coordinates. Local +Y is the revolution axis; the rigid transform maps local
/// +X to the detector's deterministic radial direction and local +Y to the world axis.
///
/// # Errors
/// Returns an error if the recovered revolution proof is inconsistent or its canonical CAD fragment cannot be constructed.
pub fn recover_solid_revolution_fragment(
    revolution: &RecoveredSolidRevolution,
) -> Result<CadFragment> {
    validate_solid_revolution(revolution)?;

    let mut model = CadModel::new();
    let profile = recovered_revolution_profile(revolution);
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

///
/// # Errors
/// Returns the first error produced while lowering one of the recovered revolutions.
pub fn recover_solid_revolution_fragments(
    revolutions: &[RecoveredSolidRevolution],
) -> Result<Vec<CadFragment>> {
    revolutions
        .iter()
        .map(recover_solid_revolution_fragment)
        .collect()
}

/// Recover a proven radially-slotted turned solid as a boolean difference.
///
/// The detector proves an axisymmetric host plus three planes describing one
/// constant-width slot.  The canonical IR therefore keeps the host as a full
/// revolution and subtracts a rectangular prism extending from the axis plane
/// past the host's maximum radius.
///
/// # Errors
/// Returns an error if the recovered host or slot proof is internally
/// inconsistent, if the cutter profile cannot be constructed, or if the CAD
/// model cannot be validated.
pub fn recover_radial_slot_revolution_fragment(
    slotted: &RecoveredRadialSlotRevolution,
) -> Result<CadFragment> {
    validate_radial_slot_revolution(slotted)?;

    let base = &slotted.base;
    let mut model = CadModel::new();
    let host_profile = recovered_revolution_profile(base);
    let host_body = model.add_node(CadNode::Revolve {
        profile: host_profile,
        axis: Axis3 {
            origin_mm: [0.0, 0.0, 0.0],
            direction: [0.0, 1.0, 0.0],
        },
        angle_rad: std::f64::consts::TAU,
    });
    let host_z = cross(base.radial_direction, base.axis_direction);
    let host = model.add_node(CadNode::Transform {
        transform: local_frame_transform(
            base.axis_origin_mm,
            base.radial_direction,
            base.axis_direction,
            host_z,
        ),
        child: host_body,
    });

    let max_radius_mm = maximum_revolution_radius(base)?;
    let cutter_margin_mm = (10.0 * base.source_tolerance_mm).max(1.0e-5);
    let cutter_depth_mm = max_radius_mm + cutter_margin_mm;
    // The detector proves that the open end coincides with exactly one global
    // axial extreme of the host. Extend the cutter only through that already-open
    // end so Boolean kernels never have to resolve a coincident cutter/host cap.
    // This extension lies outside the source solid and therefore does not change
    // the occupied geometry.
    let extended_open_end_axial_mm = if slotted.slot_open_end_axial_mm > slotted.slot_root_axial_mm
    {
        slotted.slot_open_end_axial_mm + cutter_margin_mm
    } else {
        slotted.slot_open_end_axial_mm - cutter_margin_mm
    };
    let axial_min = slotted.slot_root_axial_mm.min(extended_open_end_axial_mm);
    let axial_max = slotted.slot_root_axial_mm.max(extended_open_end_axial_mm);
    let half_width = slotted.slot_half_width_mm;
    let cutter_profile = Profile2d::polygon(vec![
        [-half_width, axial_min],
        [half_width, axial_min],
        [half_width, axial_max],
        [-half_width, axial_max],
    ])?;
    let cutter_body = model.add_node(CadNode::Extrude {
        profile: cutter_profile,
        vector_mm: [0.0, 0.0, cutter_depth_mm],
    });
    let cutter_x = normalize3(cross(base.axis_direction, slotted.slot_outward_direction))?;
    let cutter = model.add_node(CadNode::Transform {
        transform: local_frame_transform(
            base.axis_origin_mm,
            cutter_x,
            base.axis_direction,
            slotted.slot_outward_direction,
        ),
        child: cutter_body,
    });

    let root = model.add_node(CadNode::Boolean {
        op: BooleanOp::Difference,
        children: vec![host, cutter],
    });

    let mut source_entity_ids = Vec::with_capacity(base.face_ids.len() + 4);
    source_entity_ids.push(base.solid_id);
    source_entity_ids.extend(base.face_ids.iter().copied());
    source_entity_ids.extend(slotted.slot_face_ids);
    source_entity_ids.sort_unstable();
    source_entity_ids.dedup();
    let proof = Provenance {
        source_entity_ids,
        proof: ProofStatus::WithinTolerance,
        max_residual_mm: Some(base.max_residual_mm.max(slotted.slot_max_residual_mm)),
    };
    model.set_provenance(root, proof)?;
    model.add_root(root)?;
    model.validate()?;

    Ok(CadFragment {
        source: CadFragmentSource::RadialSlotRevolution {
            solid_id: base.solid_id,
            base_face_ids: base.face_ids.clone(),
            slot_face_ids: slotted.slot_face_ids,
        },
        model,
        root,
    })
}

/// Recover several proven radial-slot revolutions into canonical CAD fragments.
///
/// # Errors
/// Returns the first recovery error from [`recover_radial_slot_revolution_fragment`].
pub fn recover_radial_slot_revolution_fragments(
    revolutions: &[RecoveredRadialSlotRevolution],
) -> Result<Vec<CadFragment>> {
    revolutions
        .iter()
        .map(recover_radial_slot_revolution_fragment)
        .collect()
}

fn recovered_revolution_profile(revolution: &RecoveredSolidRevolution) -> Profile2d {
    Profile2d {
        loops: vec![ProfileLoop {
            curves: revolution
                .profile_curves
                .iter()
                .map(recovered_profile_curve_to_ir)
                .collect(),
        }],
    }
}

fn maximum_revolution_radius(revolution: &RecoveredSolidRevolution) -> Result<f64> {
    let mut maximum = 0.0_f64;
    for curve in &revolution.profile_curves {
        let candidate = match curve {
            RecoveredProfileCurve::Line {
                start_mm, end_mm, ..
            } => start_mm[0].max(end_mm[0]),
            RecoveredProfileCurve::CircleArc {
                center_mm,
                radius_mm,
                ..
            } => center_mm[0].abs() + radius_mm,
            RecoveredProfileCurve::Bezier { .. } | RecoveredProfileCurve::BSpline { .. } => {
                bail!("radial-slot revolution has unsupported spline meridian")
            }
        };
        maximum = maximum.max(candidate);
    }
    if !maximum.is_finite() || maximum <= 0.0 {
        bail!("radial-slot revolution has no positive radial extent");
    }
    Ok(maximum)
}

fn validate_radial_slot_revolution(slotted: &RecoveredRadialSlotRevolution) -> Result<()> {
    validate_solid_revolution(&slotted.base)?;
    if slotted.slot_face_ids.contains(&0) {
        bail!("radial-slot source face ids must be nonzero");
    }
    let mut unique_faces = slotted.slot_face_ids;
    unique_faces.sort_unstable();
    if unique_faces.windows(2).any(|pair| pair[0] == pair[1]) {
        bail!("radial-slot source face ids must be distinct");
    }
    if !slotted.slot_half_width_mm.is_finite() || slotted.slot_half_width_mm <= 0.0 {
        bail!("radial-slot half width must be finite and positive");
    }
    if !slotted.slot_root_axial_mm.is_finite()
        || !slotted.slot_open_end_axial_mm.is_finite()
        || (slotted.slot_root_axial_mm - slotted.slot_open_end_axial_mm).abs()
            <= slotted.base.source_tolerance_mm
    {
        bail!("radial-slot axial span must be finite and nonzero");
    }
    for direction in [slotted.slot_outward_direction, slotted.slot_side_direction] {
        if direction.iter().any(|component| !component.is_finite())
            || (norm(direction) - 1.0).abs() > 1.0e-10
        {
            bail!("radial-slot direction must be a finite unit vector");
        }
    }
    if dot(slotted.slot_outward_direction, slotted.base.axis_direction).abs() > 1.0e-10
        || dot(slotted.slot_side_direction, slotted.base.axis_direction).abs() > 1.0e-10
        || dot(slotted.slot_outward_direction, slotted.slot_side_direction).abs() > 1.0e-10
    {
        bail!("radial-slot frame must be orthogonal to the revolution axis");
    }
    if !slotted.slot_max_residual_mm.is_finite()
        || slotted.slot_max_residual_mm < 0.0
        || slotted.slot_max_residual_mm > slotted.base.source_tolerance_mm + 1.0e-15
    {
        bail!("radial-slot proof residual exceeds the source tolerance");
    }
    Ok(())
}

fn recovered_extrusion_profile(extrusion: &RecoveredSolidExtrusion) -> Profile2d {
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
    Profile2d { loops }
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
        if (end[0] - start[0]).hypot(end[1] - start[1]) > 1.0e-7 {
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

fn same_f64(left: f64, right: f64) -> bool {
    left.partial_cmp(&right)
        .is_some_and(std::cmp::Ordering::is_eq)
}

fn validate_profile_curve_geometry(curve: &RecoveredProfileCurve) -> Result<()> {
    let finite_point = |point: &[f64; 2]| point.iter().all(|value| value.is_finite());
    match curve {
        RecoveredProfileCurve::Line {
            start_mm, end_mm, ..
        } => {
            let identical_endpoints = start_mm
                .iter()
                .zip(end_mm)
                .all(|(&start, &end)| same_f64(start, end));
            if !finite_point(start_mm) || !finite_point(end_mm) || identical_endpoints {
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
                        .all(|knot| same_f64(*knot, knots[0]))
                    && knots[knots.len() - min_control_points..]
                        .iter()
                        .all(|knot| same_f64(*knot, knots[knots.len() - 1]))
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
    let mut minimum = radius
        .mul_add(start_angle.cos(), center_radius)
        .min(radius.mul_add(end_angle.cos(), center_radius));
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

const fn local_frame_transform(
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

fn normalize3(vector: [f64; 3]) -> Result<[f64; 3]> {
    let length = norm(vector);
    if !length.is_finite() || length <= 1.0e-12 {
        bail!("cannot normalize zero or non-finite CAD frame vector");
    }
    Ok([vector[0] / length, vector[1] / length, vector[2] / length])
}

#[cfg(test)]
mod tests {
    use super::*;

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
            bail!("expected transform root");
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
            bail!("expected local extrusion child");
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
            bail!("expected transform root");
        };
        let CadNode::Extrude { profile, .. } = holed_fragment.model.node(*child)? else {
            bail!("expected local extrusion child");
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
            bail!("expected transform root");
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
            bail!("expected local revolution child");
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
    fn production_radial_slot_fixture_recovers_and_evaluates() -> Result<()> {
        let bytes = include_bytes!("../../validation/fixtures/production_radial_slot.step");
        let recovered = crate::detect_radial_slot_revolutions_bytes(bytes)?;
        assert_eq!(recovered.len(), 1);

        let fragment = recover_radial_slot_revolution_fragment(&recovered[0])?;
        let CadNode::Boolean { op, children } = fragment.model.node(fragment.root)? else {
            bail!("expected production radial-slot boolean root");
        };
        assert_eq!(*op, BooleanOp::Difference);
        assert_eq!(children.len(), 2);

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
    fn radial_slot_revolution_becomes_revolve_minus_extrude() -> Result<()> {
        let slotted = RecoveredRadialSlotRevolution {
            base: RecoveredSolidRevolution {
                solid_id: 10,
                face_ids: vec![20, 21, 22],
                profile_curves: vec![
                    RecoveredProfileCurve::Line {
                        source_edge_ids: vec![100],
                        start_mm: [0.0, -2.0],
                        end_mm: [2.0, -2.0],
                    },
                    RecoveredProfileCurve::Line {
                        source_edge_ids: vec![101],
                        start_mm: [2.0, -2.0],
                        end_mm: [2.0, 2.0],
                    },
                    RecoveredProfileCurve::Line {
                        source_edge_ids: vec![102],
                        start_mm: [2.0, 2.0],
                        end_mm: [0.0, 2.0],
                    },
                    RecoveredProfileCurve::Line {
                        source_edge_ids: vec![103],
                        start_mm: [0.0, 2.0],
                        end_mm: [0.0, -2.0],
                    },
                ],
                axis_origin_mm: [0.0, 0.0, 0.0],
                axis_direction: [0.0, 1.0, 0.0],
                radial_direction: [1.0, 0.0, 0.0],
                max_residual_mm: 0.0,
                source_tolerance_mm: REVOLUTION_SOURCE_SUPPORT_TOL_MM,
            },
            slot_face_ids: [30, 31, 32],
            slot_outward_direction: [1.0, 0.0, 0.0],
            slot_side_direction: [0.0, 0.0, 1.0],
            slot_half_width_mm: 0.25,
            slot_root_axial_mm: 1.0,
            slot_open_end_axial_mm: 2.0,
            slot_max_residual_mm: 0.0,
        };

        let fragment = recover_radial_slot_revolution_fragment(&slotted)?;
        let CadNode::Boolean { op, children } = fragment.model.node(fragment.root)? else {
            bail!("expected radial-slot boolean root");
        };
        assert_eq!(*op, BooleanOp::Difference);
        assert_eq!(children.len(), 2);
        assert_eq!(
            fragment.model.provenance[&fragment.root].source_entity_ids,
            vec![10, 20, 21, 22, 30, 31, 32]
        );

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
}
