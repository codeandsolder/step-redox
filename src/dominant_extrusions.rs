use crate::math3::{canonical_direction, dot, norm};
use crate::solid_revolutions::{
    SolidFaceSignature, SolidSurfaceGeometrySignature, SolidSurfaceSignature,
};
use serde::Serialize;
use std::cmp::Ordering;

const GEOM_TOL_MM: f64 = 1.0e-7;
const DIR_TOL: f64 = 1.0e-8;
const MIN_COMPATIBLE_SIDE_FRACTION: f64 = 0.5;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DominantExtrusionResidualFace {
    pub face_id: u64,
    pub support: String,
    /// Absolute alignment with the candidate extrusion axis. For planes this is
    /// `|normal·axis|`; for axis-bearing surfaces it is `|surface_axis·axis|`.
    pub axial_alignment: Option<f64>,
    pub loop_edge_counts: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DominantExtrusionCandidate {
    pub solid_id: u64,
    pub cap_face_ids: [u64; 2],
    pub axis: [f64; 3],
    pub height_mm: f64,
    pub cap_edge_count: usize,
    pub face_count: usize,
    pub side_face_count: usize,
    pub compatible_side_face_ids: Vec<u64>,
    pub residual_faces: Vec<DominantExtrusionResidualFace>,
    pub compatible_side_fraction: f64,
}

#[must_use]
pub fn detect_dominant_extrusion_candidates(
    signatures: &[SolidSurfaceSignature],
) -> Vec<DominantExtrusionCandidate> {
    signatures.iter().filter_map(detect_one).collect::<Vec<_>>()
}

#[derive(Debug, Clone, Copy)]
struct CapPairHypothesis {
    first_index: usize,
    second_index: usize,
    axis: [f64; 3],
    height_mm: f64,
    cap_edge_count: usize,
    side_face_count: usize,
    compatible_side_count: usize,
    residual_side_count: usize,
    compatible_side_fraction: f64,
}

fn detect_one(signature: &SolidSurfaceSignature) -> Option<DominantExtrusionCandidate> {
    if !signature.closed_two_manifold || signature.faces.len() < 4 {
        return None;
    }

    // Side compatibility depends only on the first cap's axis, not on which
    // parallel mate is chosen. Compute it once per possible first cap, then
    // rank cap pairs using scalar metadata. The old implementation built a
    // full Vec of compatible/residual faces for every O(F²) cap pair, turning
    // large planar solids into O(F³) work and O(F³) retained allocations.
    let side_face_count = signature.faces.len().checked_sub(2)?;
    let mut best = None::<CapPairHypothesis>;
    for first_index in 0..signature.faces.len() {
        let first = &signature.faces[first_index];
        let SolidSurfaceGeometrySignature::Plane {
            normal: first_normal,
            max_residual_mm: first_residual,
            ..
        } = &first.geometry
        else {
            continue;
        };
        if *first_residual > GEOM_TOL_MM {
            continue;
        }
        let axis = canonical_direction(*first_normal);
        let compatible_side_count = signature
            .faces
            .iter()
            .filter(|face| surface_is_linear_extrusion_side(&face.geometry, axis))
            .count();
        if compatible_side_count < 2 {
            continue;
        }
        let compatible_count = u32::try_from(compatible_side_count).ok()?;
        let total_side_count = u32::try_from(side_face_count).ok()?;
        let compatible_side_fraction = f64::from(compatible_count) / f64::from(total_side_count);
        if compatible_side_fraction < MIN_COMPATIBLE_SIDE_FRACTION {
            continue;
        }
        let residual_side_count = side_face_count.checked_sub(compatible_side_count)?;

        for second_index in first_index + 1..signature.faces.len() {
            let Some((height_mm, cap_edge_count)) =
                cap_pair_geometry(first, &signature.faces[second_index], axis)
            else {
                continue;
            };
            let hypothesis = CapPairHypothesis {
                first_index,
                second_index,
                axis,
                height_mm,
                cap_edge_count,
                side_face_count,
                compatible_side_count,
                residual_side_count,
                compatible_side_fraction,
            };
            if best
                .as_ref()
                .is_none_or(|current| hypothesis_order(signature, &hypothesis, current).is_lt())
            {
                best = Some(hypothesis);
            }
        }
    }

    materialize_candidate(signature, best?)
}

fn cap_pair_geometry(
    first: &SolidFaceSignature,
    second: &SolidFaceSignature,
    axis: [f64; 3],
) -> Option<(f64, usize)> {
    let (
        SolidSurfaceGeometrySignature::Plane {
            origin_mm: first_origin,
            normal: first_normal,
            max_residual_mm: first_residual,
        },
        SolidSurfaceGeometrySignature::Plane {
            origin_mm: second_origin,
            normal: second_normal,
            max_residual_mm: second_residual,
        },
    ) = (&first.geometry, &second.geometry)
    else {
        return None;
    };
    if *first_residual > GEOM_TOL_MM
        || *second_residual > GEOM_TOL_MM
        || (dot(*first_normal, *second_normal).abs() - 1.0).abs() > DIR_TOL
        || !loop_shapes_match(first, second)
    {
        return None;
    }

    let separation = dot(axis, *second_origin) - dot(axis, *first_origin);
    if !separation.is_finite() || separation.abs() <= GEOM_TOL_MM {
        return None;
    }
    Some((
        separation.abs(),
        first.loops.iter().map(|loop_| loop_.edge_count).sum(),
    ))
}

fn materialize_candidate(
    signature: &SolidSurfaceSignature,
    hypothesis: CapPairHypothesis,
) -> Option<DominantExtrusionCandidate> {
    let mut compatible_side_face_ids = Vec::with_capacity(hypothesis.compatible_side_count);
    let mut residual_faces = Vec::with_capacity(hypothesis.residual_side_count);
    for (index, face) in signature.faces.iter().enumerate() {
        if index == hypothesis.first_index || index == hypothesis.second_index {
            continue;
        }
        if surface_is_linear_extrusion_side(&face.geometry, hypothesis.axis) {
            compatible_side_face_ids.push(face.face_id);
        } else {
            residual_faces.push(DominantExtrusionResidualFace {
                face_id: face.face_id,
                support: face.support.clone(),
                axial_alignment: axial_alignment(&face.geometry, hypothesis.axis),
                loop_edge_counts: face.loops.iter().map(|loop_| loop_.edge_count).collect(),
            });
        }
    }
    if compatible_side_face_ids.len() != hypothesis.compatible_side_count
        || residual_faces.len() != hypothesis.residual_side_count
    {
        return None;
    }

    Some(DominantExtrusionCandidate {
        solid_id: signature.solid_id,
        cap_face_ids: [
            signature.faces[hypothesis.first_index].face_id,
            signature.faces[hypothesis.second_index].face_id,
        ],
        axis: hypothesis.axis,
        height_mm: hypothesis.height_mm,
        cap_edge_count: hypothesis.cap_edge_count,
        face_count: signature.face_count,
        side_face_count: hypothesis.side_face_count,
        compatible_side_face_ids,
        residual_faces,
        compatible_side_fraction: hypothesis.compatible_side_fraction,
    })
}

fn hypothesis_order(
    signature: &SolidSurfaceSignature,
    first: &CapPairHypothesis,
    second: &CapPairHypothesis,
) -> Ordering {
    second
        .compatible_side_fraction
        .total_cmp(&first.compatible_side_fraction)
        .then_with(|| first.residual_side_count.cmp(&second.residual_side_count))
        .then_with(|| second.cap_edge_count.cmp(&first.cap_edge_count))
        .then_with(|| first.height_mm.total_cmp(&second.height_mm))
        .then_with(|| {
            let first_ids = [
                signature.faces[first.first_index].face_id,
                signature.faces[first.second_index].face_id,
            ];
            let second_ids = [
                signature.faces[second.first_index].face_id,
                signature.faces[second.second_index].face_id,
            ];
            first_ids.cmp(&second_ids)
        })
}

fn loop_shapes_match(first: &SolidFaceSignature, second: &SolidFaceSignature) -> bool {
    if first.loops.len() != second.loops.len() {
        return false;
    }
    let mut used = vec![false; second.loops.len()];
    for first_loop in &first.loops {
        let Some(match_index) = second
            .loops
            .iter()
            .enumerate()
            .find_map(|(index, second_loop)| {
                (!used[index]
                    && first_loop.edge_count == second_loop.edge_count
                    && first_loop.curve_counts == second_loop.curve_counts)
                    .then_some(index)
            })
        else {
            return false;
        };
        used[match_index] = true;
    }
    true
}

fn surface_is_linear_extrusion_side(
    surface: &SolidSurfaceGeometrySignature,
    axis: [f64; 3],
) -> bool {
    match surface {
        SolidSurfaceGeometrySignature::Plane { normal, .. } => dot(*normal, axis).abs() <= DIR_TOL,
        SolidSurfaceGeometrySignature::Cylinder {
            axis: surface_axis, ..
        } => parallel(*surface_axis, axis),
        SolidSurfaceGeometrySignature::SplineExtrusion { extrusion_mm, .. } => {
            norm(*extrusion_mm) > GEOM_TOL_MM && parallel(*extrusion_mm, axis)
        }
        SolidSurfaceGeometrySignature::Cone { .. }
        | SolidSurfaceGeometrySignature::Sphere { .. }
        | SolidSurfaceGeometrySignature::Torus { .. }
        | SolidSurfaceGeometrySignature::Revolution { .. }
        | SolidSurfaceGeometrySignature::Other { .. } => false,
    }
}

fn axial_alignment(surface: &SolidSurfaceGeometrySignature, axis: [f64; 3]) -> Option<f64> {
    let surface_axis = match surface {
        SolidSurfaceGeometrySignature::Plane { normal, .. } => {
            return Some(dot(*normal, axis).abs());
        }
        SolidSurfaceGeometrySignature::Cylinder {
            axis: surface_axis, ..
        }
        | SolidSurfaceGeometrySignature::Cone {
            axis: surface_axis, ..
        }
        | SolidSurfaceGeometrySignature::Sphere {
            axis: surface_axis, ..
        }
        | SolidSurfaceGeometrySignature::Torus {
            axis: surface_axis, ..
        }
        | SolidSurfaceGeometrySignature::Revolution {
            axis: surface_axis, ..
        } => *surface_axis,
        SolidSurfaceGeometrySignature::SplineExtrusion { extrusion_mm, .. } => *extrusion_mm,
        SolidSurfaceGeometrySignature::Other { .. } => return None,
    };
    let length = norm(surface_axis);
    (length > GEOM_TOL_MM).then_some((dot(surface_axis, axis) / length).abs())
}

fn parallel(first: [f64; 3], second: [f64; 3]) -> bool {
    let first_norm = norm(first);
    let second_norm = norm(second);
    if first_norm <= GEOM_TOL_MM || second_norm <= GEOM_TOL_MM {
        return false;
    }
    let alignment = dot(first, second).abs() / first_norm / second_norm;
    (alignment - 1.0).abs() <= DIR_TOL
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solid_revolutions::{SolidLoopSignature, SolidSurfaceGeometrySignature};
    use std::collections::BTreeMap;

    fn loop_signature(edge_count: usize) -> SolidLoopSignature {
        SolidLoopSignature {
            outer: true,
            edge_count,
            curve_counts: BTreeMap::from([("line".to_owned(), edge_count)]),
        }
    }

    fn plane(face_id: u64, x: f64, normal: [f64; 3], edges: usize) -> SolidFaceSignature {
        SolidFaceSignature {
            face_id,
            support: "plane".to_owned(),
            geometry: SolidSurfaceGeometrySignature::Plane {
                origin_mm: [x, 0.0, 0.0],
                normal,
                max_residual_mm: 0.0,
            },
            loops: vec![loop_signature(edges)],
        }
    }

    #[test]
    fn dominant_extrusion_keeps_off_axis_treatment_residual() {
        let signature = SolidSurfaceSignature {
            solid_id: 10,
            face_count: 6,
            support_counts: BTreeMap::new(),
            faces: vec![
                plane(1, 0.0, [1.0, 0.0, 0.0], 4),
                plane(2, 1.0, [1.0, 0.0, 0.0], 4),
                plane(3, 0.0, [0.0, 1.0, 0.0], 4),
                plane(4, 0.0, [0.0, 0.0, 1.0], 4),
                plane(5, 0.0, [0.0, -1.0, 0.0], 4),
                plane(
                    6,
                    0.0,
                    [
                        std::f64::consts::FRAC_1_SQRT_2,
                        0.0,
                        std::f64::consts::FRAC_1_SQRT_2,
                    ],
                    4,
                ),
            ],
            unique_edge_count: 12,
            edge_use_count: 24,
            closed_two_manifold: true,
        };
        let candidates = detect_dominant_extrusion_candidates(&[signature]);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].cap_face_ids, [1, 2]);
        assert_eq!(candidates[0].residual_faces.len(), 1);
        assert_eq!(candidates[0].residual_faces[0].face_id, 6);
    }

    #[test]
    fn dominant_extrusion_many_parallel_cap_pairs_stays_bounded() {
        let mut faces = Vec::new();
        for index in 0_u32..128 {
            faces.push(plane(
                1_000 + u64::from(index),
                f64::from(index) * 0.1,
                [1.0, 0.0, 0.0],
                4,
            ));
        }
        for index in 0_u32..160 {
            faces.push(plane(
                10_000 + u64::from(index),
                0.0,
                if index % 2 == 0 {
                    [0.0, 1.0, 0.0]
                } else {
                    [0.0, 0.0, 1.0]
                },
                4,
            ));
        }
        let signature = SolidSurfaceSignature {
            solid_id: 11,
            face_count: faces.len(),
            support_counts: BTreeMap::new(),
            faces,
            unique_edge_count: 0,
            edge_use_count: 0,
            closed_two_manifold: true,
        };

        let candidates = detect_dominant_extrusion_candidates(&[signature]);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].compatible_side_face_ids.len(), 160);
        assert_eq!(candidates[0].residual_faces.len(), 126);
    }
}
