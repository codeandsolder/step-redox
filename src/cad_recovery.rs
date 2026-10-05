use crate::cad_ir::{
    Axis3, BooleanOp, BrepFallback, CadModel, CadNode, Curve2d, FusedPeriodicChain, IndexedCount,
    NodeId, PatternSpec, Profile2d, ProfileLoop, ProofStatus, Provenance, RigidTransform,
};
use crate::numeric::exact_usize_to_f64;
use crate::patterns::InstancePattern;
use crate::periodic_chains::PeriodicChainPattern;
use crate::periodic_decomposition::decompose_periodic_chain;
use crate::profile_curves::RecoveredProfileCurve;
use crate::solid_extrusions::RecoveredSolidExtrusion;
use crate::solid_revolutions::{
    MAX_REVOLUTION_SOURCE_UNCERTAINTY_MM, REVOLUTION_SOURCE_SUPPORT_TOL_MM,
    RecoveredRadialSlotRevolution, RecoveredSolidRevolution,
};
use anyhow::{Result, anyhow, bail};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CadFragment {
    pub source: CadFragmentSource,
    pub model: CadModel,
    pub root: NodeId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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
    RadialSlotRevolution {
        solid_id: u64,
        base_face_ids: Vec<u64>,
        slot_face_ids: [u64; 3],
    },
    PeriodicChain {
        solid_id: u64,
        sites: usize,
    },
    PeriodicChainSurfaceDecomposition {
        solid_id: u64,
        sites: usize,
        motif_period_sites: usize,
    },
    BrepFallback {
        solid_id: u64,
    },
}

fn add_packed_face_patch(
    model: &mut CadModel,
    solid_id: u64,
    face_ids: &[u64],
    entities: &[ruststep::ast::EntityInstance],
) -> Result<NodeId> {
    if face_ids.is_empty() {
        bail!("periodic decomposition patch must contain at least one face");
    }
    let packed = crate::compact_brep::build_packed_brep_faces(solid_id, face_ids, entities)?;
    let node = model.add_node(CadNode::BrepFallback(BrepFallback::from_packed(
        face_ids.to_vec(),
        packed,
    )));
    model.set_provenance(
        node,
        Provenance {
            source_entity_ids: face_ids.to_vec(),
            proof: ProofStatus::Exact,
            max_residual_mm: Some(0.0),
        },
    )?;
    Ok(node)
}

fn add_linear_face_pattern(
    model: &mut CadModel,
    solid_id: u64,
    source_face_ids: &[u64],
    repeat_count: usize,
    step_mm: [f64; 3],
    all_source_face_ids: Vec<u64>,
    entities: &[ruststep::ast::EntityInstance],
) -> Result<NodeId> {
    let child = add_packed_face_patch(model, solid_id, source_face_ids, entities)?;
    if repeat_count == 1 {
        return Ok(child);
    }
    if repeat_count == 0 || step_mm.iter().any(|value| !value.is_finite()) {
        bail!("invalid periodic face-pattern metadata");
    }
    let root = model.add_node(CadNode::Pattern {
        pattern: PatternSpec::Linear {
            count: repeat_count,
            step_mm,
        },
        child,
    });
    model.set_provenance(
        root,
        Provenance {
            source_entity_ids: all_source_face_ids,
            proof: ProofStatus::WithinTolerance,
            max_residual_mm: Some(1.0e-5),
        },
    )?;
    Ok(root)
}

/// Lower a proven periodic fused solid into exact packed source patches plus
/// translation-pattern nodes. The result is an assembly of surface patches,
/// not a claim that constructive boolean/manifold history has been recovered.
///
/// # Errors
/// Returns an error if the periodic proof cannot be exactly decomposed, packed,
/// or shown to cover the source solid's face set exactly once.
pub fn recover_periodic_chain_surface_decomposition_fragment(
    chain: &PeriodicChainPattern,
    entities: &[ruststep::ast::EntityInstance],
) -> Result<CadFragment> {
    let decomposition = decompose_periodic_chain(chain, entities)
        .ok_or_else(|| anyhow!("periodic chain does not have a complete exact decomposition"))?;

    let index = crate::instances::build_index(entities);
    let mut source_faces = crate::brep::solid_face_ids(chain.solid_id, entities, &index)
        .ok_or_else(|| anyhow!("periodic solid has no readable closed-shell face set"))?;
    source_faces.sort_unstable();
    source_faces.dedup();

    let mut represented_faces = Vec::<u64>::new();
    let mut model = CadModel::new();
    let mut children = Vec::<NodeId>::new();

    for pattern in &decomposition.site_patterns {
        let all_faces = (pattern.source_index..chain.site_face_ids.len())
            .step_by(decomposition.motif_period_sites)
            .flat_map(|index| chain.site_face_ids[index].iter().copied())
            .collect::<Vec<_>>();
        represented_faces.extend(all_faces.iter().copied());
        children.push(add_linear_face_pattern(
            &mut model,
            chain.solid_id,
            &pattern.source_face_ids,
            pattern.repeat_count,
            pattern.step_mm,
            all_faces,
            entities,
        )?);
    }

    for pattern in &decomposition.gap_patterns {
        let all_faces = (pattern.source_index..chain.gap_face_ids.len())
            .step_by(decomposition.motif_period_sites)
            .flat_map(|index| chain.gap_face_ids[index].iter().copied())
            .collect::<Vec<_>>();
        represented_faces.extend(all_faces.iter().copied());
        children.push(add_linear_face_pattern(
            &mut model,
            chain.solid_id,
            &pattern.source_face_ids,
            pattern.repeat_count,
            pattern.step_mm,
            all_faces,
            entities,
        )?);
    }

    if !decomposition.residual_face_ids.is_empty() {
        represented_faces.extend(decomposition.residual_face_ids.iter().copied());
        children.push(add_packed_face_patch(
            &mut model,
            chain.solid_id,
            &decomposition.residual_face_ids,
            entities,
        )?);
    }

    represented_faces.sort_unstable();
    if represented_faces.windows(2).any(|pair| pair[0] == pair[1]) {
        bail!("periodic decomposition represents a source face more than once");
    }
    if represented_faces != source_faces {
        bail!(
            "periodic decomposition/source face-set mismatch: represented={} source={}",
            represented_faces.len(),
            source_faces.len()
        );
    }
    if represented_faces.len() != decomposition.covered_face_count {
        bail!("periodic decomposition covered-face count is inconsistent");
    }
    if children.is_empty() {
        bail!("periodic decomposition produced no CAD nodes");
    }

    let root = model.add_node(CadNode::Assembly { children });
    model.set_provenance(
        root,
        Provenance {
            source_entity_ids: vec![chain.solid_id],
            proof: ProofStatus::WithinTolerance,
            max_residual_mm: Some(1.0e-5),
        },
    )?;
    model.add_root(root)?;
    model.validate()?;

    Ok(CadFragment {
        source: CadFragmentSource::PeriodicChainSurfaceDecomposition {
            solid_id: chain.solid_id,
            sites: chain.sites,
            motif_period_sites: decomposition.motif_period_sites,
        },
        model,
        root,
    })
}

/// Admit any source solid into CAD IR without pretending it has been
/// constructively recovered yet.
///
/// The fallback is exact by provenance: it names the original B-rep root and
/// carries only complexity estimates. Later recovery passes can replace this
/// leaf with an Extrude/Revolve/Boolean/etc. without changing the rule that
/// every parseable source body is representable from day one.
///
/// # Errors
/// Returns an error if the source solid id is invalid or the fallback CAD fragment cannot be validated.
pub fn recover_brep_fallback_fragment(
    solid_id: u64,
    estimated_faces: usize,
    estimated_edges: usize,
    estimated_control_points: usize,
) -> Result<CadFragment> {
    if solid_id == 0 {
        bail!("B-rep fallback solid id must be nonzero");
    }

    let mut model = CadModel::new();
    let root = model.add_node(CadNode::BrepFallback(BrepFallback::source_reference(
        vec![solid_id],
        estimated_faces,
        estimated_edges,
        estimated_control_points,
    )));
    model.set_provenance(
        root,
        Provenance {
            source_entity_ids: vec![solid_id],
            proof: ProofStatus::Exact,
            max_residual_mm: Some(0.0),
        },
    )?;
    model.add_root(root)?;
    model.validate()?;

    Ok(CadFragment {
        source: CadFragmentSource::BrepFallback { solid_id },
        model,
        root,
    })
}

/// Build an exact compact fallback from the parsed source B-rep.
///
/// The packed payload carries indexed topology and geometry in the CAD IR itself.
/// If every source geometry kind is supported, `BrepFallback::is_self_contained`
/// is true and the fragment no longer depends on the original STEP entity graph.
///
/// # Errors
/// Returns an error if the source solid id is invalid or its B-rep cannot be packed.
pub fn recover_packed_brep_fallback_fragment(
    solid_id: u64,
    entities: &[ruststep::ast::EntityInstance],
) -> Result<CadFragment> {
    if solid_id == 0 {
        bail!("B-rep fallback solid id must be nonzero");
    }

    let packed = crate::compact_brep::build_packed_brep(solid_id, entities)?;
    let mut model = CadModel::new();
    let root = model.add_node(CadNode::BrepFallback(BrepFallback::from_packed(
        vec![solid_id],
        packed,
    )));
    model.set_provenance(
        root,
        Provenance {
            source_entity_ids: vec![solid_id],
            proof: ProofStatus::Exact,
            max_residual_mm: Some(0.0),
        },
    )?;
    model.add_root(root)?;
    model.validate()?;

    Ok(CadFragment {
        source: CadFragmentSource::BrepFallback { solid_id },
        model,
        root,
    })
}

/// Recover a first-class fused periodic-chain IR fragment from a proven source partition.
///
/// This node is descriptive, not yet executable by Monstertruck/KCL. It references the exact
/// source solid while recording the proven pitch/count grammar; detailed face ownership stays
/// in the detector/debug evidence rather than bloating high-level IR. Backends that do not
/// implement fused periodic chains must reject the node explicitly instead of lowering it as
/// a pattern of independent solids.
///
/// # Errors
/// Returns an error if the chain is not fully proven or its partition is inconsistent.
pub fn recover_periodic_chain_fragment(chain: &PeriodicChainPattern) -> Result<CadFragment> {
    validate_periodic_chain(chain)?;

    let adjacent_site_edges_per_boundary = chain
        .adjacent_site_edge_counts
        .first()
        .copied()
        .unwrap_or(0);

    let mut model = CadModel::new();
    let root = model.add_node(CadNode::PeriodicChain(FusedPeriodicChain {
        source_solid_id: chain.solid_id,
        axis: chain.axis,
        pitch_mm: chain.pitch_mm,
        sites: chain.sites,
        first_site_center_mm: chain.site_centers_mm[0],
        interior_site_face_count: chain.interior_site_face_count,
        site_face_count_overrides: count_overrides(
            &chain.site_face_counts,
            chain.interior_site_face_count,
        ),
        interior_gap_face_count: chain.interior_gap_face_count,
        gap_face_count_overrides: count_overrides(
            &chain.gap_face_counts,
            chain.interior_gap_face_count,
        ),
        stretch_face_count: chain.stretch_face_ids.len(),
        fixed_negative_face_count: chain.fixed_negative_face_ids.len(),
        fixed_positive_face_count: chain.fixed_positive_face_ids.len(),
        adjacent_site_edges_per_boundary,
        repeat_coverage_ratio: chain.repeat_coverage_ratio,
    }));

    model.set_provenance(
        root,
        Provenance {
            source_entity_ids: vec![chain.solid_id],
            // This is a structural partition/grammar proof, not a reconstructed
            // geometry comparison, so there is no geometric residual to invent.
            proof: ProofStatus::StructurallyProven,
            max_residual_mm: None,
        },
    )?;
    model.add_root(root)?;
    model.validate()?;

    Ok(CadFragment {
        source: CadFragmentSource::PeriodicChain {
            solid_id: chain.solid_id,
            sites: chain.sites,
        },
        model,
        root,
    })
}

/// Recover compact CAD-IR fragments for every periodic chain whose detector proof is complete.
///
/// Diagnostic candidates that are not `read_only_proven` remain available through the
/// periodic-chain report, but they are deliberately excluded from canonical CAD IR.
///
/// # Errors
/// Returns the first inconsistency found in a chain already marked as proven.
pub fn recover_periodic_chain_fragments(
    chains: &[PeriodicChainPattern],
) -> Result<Vec<CadFragment>> {
    chains
        .iter()
        .filter(|chain| chain.read_only_proven)
        .map(recover_periodic_chain_fragment)
        .collect()
}

fn count_overrides(counts: &[usize], baseline: usize) -> Vec<IndexedCount> {
    counts
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, count)| (count != baseline).then_some(IndexedCount { index, count }))
        .collect()
}

fn validate_periodic_chain(chain: &PeriodicChainPattern) -> Result<()> {
    if !chain.read_only_proven
        || !chain.complete_partition
        || chain.faces_without_geometry != 0
        || chain.nonmanifold_edges != 0
        || chain.nonlocal_cross_site_edges != 0
        || !chain.fixed_middle_face_ids.is_empty()
    {
        bail!("periodic-chain source proof is not complete and editable");
    }
    if chain.solid_id == 0 || chain.sites < 2 {
        bail!("periodic-chain source identity/site count is invalid");
    }
    if !chain.pitch_mm.is_finite() || chain.pitch_mm <= 0.0 {
        bail!("periodic-chain source pitch is invalid");
    }
    if chain.axis.iter().any(|value| !value.is_finite()) {
        bail!("periodic-chain source axis is non-finite");
    }
    let axis_norm = chain.axis[2]
        .mul_add(
            chain.axis[2],
            chain.axis[1].mul_add(chain.axis[1], chain.axis[0] * chain.axis[0]),
        )
        .sqrt();
    if (axis_norm - 1.0).abs() > 1.0e-9 {
        bail!("periodic-chain source axis is not unit length");
    }
    if chain.site_centers_mm.iter().any(|value| !value.is_finite()) {
        bail!("periodic-chain source site centers are non-finite");
    }
    let pitch_tolerance = 1.0e-5_f64.max(chain.pitch_mm.abs() * 1.0e-9);
    if chain
        .site_centers_mm
        .windows(2)
        .any(|pair| ((pair[1] - pair[0]).abs() - chain.pitch_mm).abs() > pitch_tolerance)
    {
        bail!("periodic-chain source site centers disagree with pitch");
    }
    if chain.site_face_ids.len() != chain.sites
        || chain.site_face_counts.len() != chain.sites
        || chain.site_centers_mm.len() != chain.sites
        || chain.gap_face_ids.len() + 1 != chain.sites
        || chain.gap_face_counts.len() + 1 != chain.sites
        || chain.adjacent_site_edge_counts.len() + 1 != chain.sites
    {
        bail!("periodic-chain proof arrays disagree with site count");
    }
    if chain.interior_site_face_count == 0 || chain.site_face_counts.contains(&0) {
        bail!("periodic-chain proof contains an empty site");
    }
    if chain
        .site_face_ids
        .iter()
        .zip(&chain.site_face_counts)
        .any(|(faces, &count)| faces.len() != count)
        || chain
            .gap_face_ids
            .iter()
            .zip(&chain.gap_face_counts)
            .any(|(faces, &count)| faces.len() != count)
    {
        bail!("periodic-chain face counts disagree with face partitions");
    }
    if chain
        .adjacent_site_edge_counts
        .first()
        .is_some_and(|first| {
            chain
                .adjacent_site_edge_counts
                .iter()
                .any(|count| count != first)
        })
    {
        bail!("periodic-chain adjacent-site seam width is inconsistent");
    }

    let mut owned = std::collections::HashSet::new();
    for face in chain
        .site_face_ids
        .iter()
        .flatten()
        .chain(chain.gap_face_ids.iter().flatten())
        .chain(chain.stretch_face_ids.iter())
        .chain(chain.fixed_negative_face_ids.iter())
        .chain(chain.fixed_positive_face_ids.iter())
    {
        if *face == 0 || !owned.insert(*face) {
            bail!("periodic-chain source face partition is invalid");
        }
    }
    Ok(())
}

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

/// Recover a constructive CAD fragment from an already-proven STEP instance pattern.
///
/// The child is the first actual `MAPPED_ITEM` occurrence, preserved as an exact B-rep
/// fallback leaf. This is deliberate: using only the `REPRESENTATION_MAP` would lose
/// the STEP mapping-origin transform when that origin is not global zero. The pattern
/// node then describes only the repeated translation lattice.
///
/// # Errors
/// Returns an error if the instance-pattern proof is inconsistent or its canonical CAD fragment cannot be constructed.
pub fn recover_instance_pattern_fragment(pattern: &InstancePattern) -> Result<CadFragment> {
    validate_instance_pattern(pattern)?;

    let first_item = pattern.item_ids[0];
    let mut model = CadModel::new();
    let child = model.add_node(CadNode::BrepFallback(BrepFallback::source_reference(
        vec![first_item],
        0,
        0,
        0,
    )));
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

///
/// # Errors
/// Returns the first error produced while lowering one of the instance patterns.
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

    let declared_cells = pattern
        .grid_shape
        .iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .ok_or_else(|| anyhow!("instance pattern grid size overflows usize"))?;
    let Some(item_count) = exact_usize_to_f64(pattern.item_ids.len()) else {
        bail!("instance pattern item count exceeds exact binary64 integer range");
    };
    let Some(declared_cells) = exact_usize_to_f64(declared_cells) else {
        bail!("instance pattern grid size exceeds exact binary64 integer range");
    };
    let expected_fill = item_count / declared_cells;
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

    let nu = i64::try_from(pattern.grid_shape[0])
        .map_err(|_| anyhow!("instance pattern U grid size exceeds i64 range"))?;
    let nv = if dimensions == 2 {
        i64::try_from(pattern.grid_shape[1])
            .map_err(|_| anyhow!("instance pattern V grid size exceeds i64 range"))?
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
    let Some(dimensions_f64) = exact_usize_to_f64(dimensions) else {
        bail!("instance pattern dimension exceeds exact binary64 integer range");
    };
    let per_axis_budget_mm = remaining_budget_mm / dimensions_f64;
    let mut canonical_basis = Vec::with_capacity(dimensions);
    let mut canonicalization_residual_mm = 0.0;
    for axis in 0..dimensions {
        let span = pattern.grid_shape[axis];
        let (basis, added_residual) =
            canonicalize_step(pattern.basis[axis], span, per_axis_budget_mm)?;
        canonical_basis.push(basis);
        canonicalization_residual_mm += added_residual;
    }

    let spec = match pattern.dimension {
        1 => {
            let full_contiguous = pattern.grid_shape[0] == pattern.item_ids.len()
                && pattern.occupancy.iter().enumerate().all(|(index, site)| {
                    i64::try_from(index).is_ok_and(|index| *site == [index, 0])
                });
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
                            .map(occupancy_site_to_grid_index)
                            .collect::<Result<Vec<_>>>()?,
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
                        .map(occupancy_site_to_grid_index)
                        .collect::<Result<Vec<_>>>()?,
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

fn occupancy_site_to_grid_index(site: &[i64; 2]) -> Result<[usize; 3]> {
    let u = usize::try_from(site[0])
        .map_err(|_| anyhow!("instance pattern U occupancy is negative or too large"))?;
    let v = usize::try_from(site[1])
        .map_err(|_| anyhow!("instance pattern V occupancy is negative or too large"))?;
    Ok([u, v, 0])
}

fn canonicalize_step(step: [f64; 3], span: usize, budget_mm: f64) -> Result<([f64; 3], f64)> {
    let Some(repeats) = exact_usize_to_f64(span.saturating_sub(1)) else {
        bail!("instance pattern span exceeds exact binary64 integer range");
    };
    if repeats == 0.0 || budget_mm <= 0.0 {
        return Ok((step, 0.0));
    }

    let max_abs = step.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    if max_abs > 0.0 && max_abs < 1.0e-15 {
        return Ok((step, 0.0));
    }
    let start_exponent = (-15..=308)
        .find(|exponent| 10_f64.powi(*exponent) >= max_abs)
        .unwrap_or(308);
    if !max_abs.is_finite() {
        bail!("instance pattern basis contains non-finite magnitude");
    }

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
            return Ok((candidate, added_residual_mm));
        }
    }

    Ok((step, 0.0))
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
        let Ok(u) = i64::try_from(u) else {
            return false;
        };
        for v in 0..nv {
            let Ok(v) = i64::try_from(v) else {
                return false;
            };
            expected.push([u, v]);
        }
    }
    let mut actual = occupancy.to_vec();
    actual.sort_unstable();
    expected.sort_unstable();
    actual == expected
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

fn normalize3(vector: [f64; 3]) -> Result<[f64; 3]> {
    let length = norm(vector);
    if !length.is_finite() || length <= 1.0e-12 {
        bail!("cannot normalize zero or non-finite CAD frame vector");
    }
    Ok([vector[0] / length, vector[1] / length, vector[2] / length])
}

fn norm(vector: [f64; 3]) -> f64 {
    vector[2]
        .mul_add(
            vector[2],
            vector[1].mul_add(vector[1], vector[0] * vector[0]),
        )
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_pattern(
        occupancy: Vec<[i64; 2]>,
        span: usize,
        residual: f64,
    ) -> Result<InstancePattern> {
        let item_count = exact_usize_to_f64(occupancy.len())
            .ok_or_else(|| anyhow!("test item count exceeds exact binary64 integer range"))?;
        let span_f64 = exact_usize_to_f64(span)
            .ok_or_else(|| anyhow!("test span exceeds exact binary64 integer range"))?;
        let item_ids = (0..occupancy.len())
            .map(|index| u64::try_from(index).map(|index| 300 + index))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(InstancePattern {
            parent_representation: 100,
            representation_map: 200,
            item_ids,
            style_signature: vec![vec![700]],
            dimension: 1,
            origin: [12.0, 3.0, -1.0],
            basis: vec![[2.54, 0.0, 0.0]],
            pitch: vec![2.54],
            occupancy,
            grid_shape: vec![span],
            fill_ratio: item_count / span_f64,
            tolerance_mm: 1.0e-7,
            max_residual_mm: residual,
        })
    }

    #[test]
    fn arbitrary_solid_can_enter_cad_ir_as_exact_brep_fallback() -> Result<()> {
        let fragment = recover_brep_fallback_fragment(74952, 132, 395, 8)?;
        assert_eq!(
            fragment.source,
            CadFragmentSource::BrepFallback { solid_id: 74952 }
        );
        let CadNode::BrepFallback(fallback) = fragment.model.node(fragment.root)? else {
            bail!("expected B-rep fallback root");
        };
        assert_eq!(fallback.source_entity_ids, vec![74952]);
        assert_eq!(fallback.estimated_faces, 132);
        assert_eq!(fallback.estimated_edges, 395);
        assert_eq!(fallback.estimated_control_points, 8);
        assert_eq!(
            fragment.model.provenance[&fragment.root].proof,
            ProofStatus::Exact
        );
        assert_eq!(
            fragment.model.provenance[&fragment.root].max_residual_mm,
            Some(0.0)
        );
        assert!(recover_brep_fallback_fragment(0, 0, 0, 0).is_err());
        Ok(())
    }

    fn periodic_chain_fixture() -> PeriodicChainPattern {
        PeriodicChainPattern {
            solid_id: 42,
            axis: [1.0, 0.0, 0.0],
            pitch_mm: 2.54,
            sites: 4,
            site_centers_mm: vec![0.0, 2.54, 5.08, 7.62],
            lattice_face_family_votes: 8,
            complete_partition: true,
            read_only_proven: true,
            faces_without_geometry: 0,
            nonmanifold_edges: 0,
            cross_site_edges: 6,
            nonlocal_cross_site_edges: 0,
            adjacent_site_edge_counts: vec![2, 2, 2],
            site_face_counts: vec![2, 2, 2, 3],
            interior_site_face_count: 2,
            gap_face_counts: vec![1, 1, 1],
            interior_gap_face_count: 1,
            site_face_ids: vec![vec![10, 11], vec![20, 21], vec![30, 31], vec![40, 41, 42]],
            gap_face_ids: vec![vec![50], vec![51], vec![52]],
            stretch_face_ids: vec![60],
            fixed_negative_face_ids: vec![70],
            fixed_middle_face_ids: Vec::new(),
            fixed_positive_face_ids: vec![71],
            repeat_coverage_ratio: 0.9,
            edge_category_counts: std::collections::BTreeMap::new(),
            site_adjacency_signatures: vec![std::collections::BTreeMap::new(); 4],
            gap_adjacency_signatures: vec![std::collections::BTreeMap::new(); 3],
        }
    }

    #[test]
    fn proven_fused_periodic_chain_becomes_first_class_ir() -> Result<()> {
        let chain = periodic_chain_fixture();
        let fragment = recover_periodic_chain_fragment(&chain)?;
        assert_eq!(
            fragment.source,
            CadFragmentSource::PeriodicChain {
                solid_id: 42,
                sites: 4,
            }
        );
        let CadNode::PeriodicChain(node) = fragment.model.node(fragment.root)? else {
            bail!("expected fused periodic-chain root");
        };
        assert_eq!(node.sites, 4);
        assert_eq!(node.pitch_mm, 2.54);
        assert_eq!(node.adjacent_site_edges_per_boundary, 2);
        assert_eq!(node.interior_site_face_count, 2);
        assert_eq!(node.interior_gap_face_count, 1);
        assert_eq!(
            node.site_face_count_overrides,
            vec![IndexedCount { index: 3, count: 3 }]
        );
        assert_eq!(node.gap_face_count_overrides, Vec::<IndexedCount>::new());
        assert_eq!(node.stretch_face_count, 1);
        assert_eq!(
            fragment.model.provenance[&fragment.root].proof,
            ProofStatus::StructurallyProven
        );
        assert_eq!(
            fragment.model.provenance[&fragment.root].max_residual_mm,
            None
        );
        assert!(fragment.model.complexity_score(fragment.root)? < 200);
        Ok(())
    }

    #[test]
    fn periodic_chain_recovery_fails_closed_on_incomplete_or_ambiguous_proof() -> Result<()> {
        let mut chain = periodic_chain_fixture();
        chain.read_only_proven = false;
        assert!(recover_periodic_chain_fragment(&chain).is_err());

        let mut chain = periodic_chain_fixture();
        chain.site_face_ids[1][0] = chain.site_face_ids[0][0];
        assert!(recover_periodic_chain_fragment(&chain).is_err());

        let mut chain = periodic_chain_fixture();
        chain.adjacent_site_edge_counts[1] = 3;
        assert!(recover_periodic_chain_fragment(&chain).is_err());

        let mut diagnostic = periodic_chain_fixture();
        diagnostic.read_only_proven = false;
        let fragments = recover_periodic_chain_fragments(&[periodic_chain_fixture(), diagnostic])?;
        assert_eq!(fragments.len(), 1);
        Ok(())
    }

    #[test]
    fn packed_brep_fallback_is_self_contained() -> Result<()> {
        let src = b"ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('x'),'1');
FILE_NAME('a','b',(''),(''),'x','y','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(1.,0.,0.));
#3=CARTESIAN_POINT('',(0.,1.,0.));
#4=VERTEX_POINT('',#1);
#5=VERTEX_POINT('',#2);
#6=VERTEX_POINT('',#3);
#7=DIRECTION('',(1.,0.,0.));
#8=VECTOR('',#7,1.);
#9=LINE('',#1,#8);
#10=DIRECTION('',(-1.,1.,0.));
#11=VECTOR('',#10,1.4142135623730951);
#12=LINE('',#2,#11);
#13=DIRECTION('',(0.,-1.,0.));
#14=VECTOR('',#13,1.);
#15=LINE('',#3,#14);
#16=EDGE_CURVE('',#4,#5,#9,.T.);
#17=EDGE_CURVE('',#5,#6,#12,.T.);
#18=EDGE_CURVE('',#6,#4,#15,.T.);
#19=ORIENTED_EDGE('',*,*,#16,.T.);
#20=ORIENTED_EDGE('',*,*,#17,.T.);
#21=ORIENTED_EDGE('',*,*,#18,.T.);
#22=EDGE_LOOP('',(#19,#20,#21));
#23=FACE_OUTER_BOUND('',#22,.T.);
#24=DIRECTION('',(0.,0.,1.));
#25=DIRECTION('',(1.,0.,0.));
#26=AXIS2_PLACEMENT_3D('',#1,#24,#25);
#27=PLANE('',#26);
#28=ADVANCED_FACE('',(#23),#27,.T.);
#29=CLOSED_SHELL('',(#28));
#30=MANIFOLD_SOLID_BREP('',#29);
ENDSEC;
END-ISO-10303-21;
";
        let exchange = ruststep::parser::parse(std::str::from_utf8(src)?)?;
        let fragment = recover_packed_brep_fallback_fragment(30, &exchange.data[0].entities)?;
        let CadNode::BrepFallback(fallback) = fragment.model.node(fragment.root)? else {
            panic!("expected B-rep fallback root");
        };
        assert!(fallback.is_self_contained());
        assert_eq!(fallback.estimated_faces, 1);
        assert_eq!(fallback.estimated_edges, 3);
        assert!(fallback.packed_core_bytes().is_some_and(|bytes| bytes > 0));
        assert!(
            fallback
                .packed_provenance_bytes()
                .is_some_and(|bytes| bytes > 0)
        );
        Ok(())
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
        let bytes = include_bytes!("../validation/fixtures/production_radial_slot.step");
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

    #[test]
    fn full_linear_pattern_becomes_constructive_pattern_over_first_occurrence() -> Result<()> {
        let pattern = linear_pattern(vec![[0, 0], [1, 0], [2, 0], [3, 0]], 4, 0.0)?;
        let fragment = recover_instance_pattern_fragment(&pattern)?;

        let CadNode::Pattern {
            pattern: PatternSpec::Linear { count, step_mm },
            child,
        } = fragment.model.node(fragment.root)?
        else {
            bail!("expected linear pattern root");
        };
        assert_eq!(*count, 4);
        assert_eq!(*step_mm, [2.54, 0.0, 0.0]);

        let CadNode::BrepFallback(fallback) = fragment.model.node(*child)? else {
            bail!("expected B-rep fallback child");
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
        let mut pattern = linear_pattern((0..24).map(|index| [index, 0]).collect(), 24, 2.4e-13)?;
        pattern.basis = vec![[
            1.999_999_999_999_989_3,
            4.440_892_098_500_626e-16,
            -4.440_892_098_500_626e-16,
        ]];
        pattern.pitch = vec![1.999_999_999_999_989_3];

        let fragment = recover_instance_pattern_fragment(&pattern)?;
        let CadNode::Pattern {
            pattern: PatternSpec::Linear { step_mm, .. },
            ..
        } = fragment.model.node(fragment.root)?
        else {
            bail!("expected linear pattern root");
        };
        assert_eq!(*step_mm, [2.0, 0.0, 0.0]);
        let Some(max_residual_mm) = fragment.model.provenance[&fragment.root].max_residual_mm
        else {
            bail!("pattern provenance should carry a residual");
        };
        assert!(max_residual_mm <= pattern.tolerance_mm);
        Ok(())
    }

    #[test]
    fn sparse_linear_pattern_preserves_holes_as_grid_occupancy() -> Result<()> {
        let pattern = linear_pattern(vec![[0, 0], [2, 0], [4, 0]], 5, 3.0e-8)?;
        let fragment = recover_instance_pattern_fragment(&pattern)?;

        let CadNode::Pattern {
            pattern: PatternSpec::Grid {
                counts, occupancy, ..
            },
            ..
        } = fragment.model.node(fragment.root)?
        else {
            bail!("expected sparse grid pattern root");
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
            bail!("expected grid pattern root");
        };
        assert_eq!(*counts, [2, 2, 1]);
        assert!(occupancy.is_none());
        Ok(())
    }

    #[test]
    fn malformed_pattern_fails_closed() -> Result<()> {
        let mut pattern = linear_pattern(vec![[0, 0], [1, 0]], 2, 0.0)?;
        pattern.occupancy[0] = [1, 0];
        assert!(recover_instance_pattern_fragment(&pattern).is_err());
        Ok(())
    }
}
