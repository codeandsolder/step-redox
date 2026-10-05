use crate::periodic_chains::PeriodicChainPattern;
use ruststep::ast::EntityInstance;
use serde::Serialize;

const MAX_MOTIF_PERIOD_SITES: usize = 8;

type PatchSignature = Vec<String>;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PeriodicPatchPattern {
    pub source_index: usize,
    pub source_face_ids: Vec<u64>,
    pub repeat_count: usize,
    pub step_mm: [f64; 3],
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PeriodicChainDecomposition {
    pub solid_id: u64,
    pub primitive_pitch_mm: f64,
    pub motif_period_sites: usize,
    pub motif_pitch_mm: f64,
    pub site_patterns: Vec<PeriodicPatchPattern>,
    pub gap_patterns: Vec<PeriodicPatchPattern>,
    pub residual_face_ids: Vec<u64>,
    pub covered_face_count: usize,
}

/// Factor a currently-proven periodic chain into exact repeated source patches.
///
/// Unlike the historical detector-side decomposition, this consumes the current
/// `PeriodicChainPattern` after overlay promotion and seam validation. That keeps
/// edit correctness in one detector while still recovering a compact geometric
/// representation of the repeated surface patches.
#[must_use]
pub fn decompose_periodic_chain(
    chain: &PeriodicChainPattern,
    entities: &[EntityInstance],
) -> Option<PeriodicChainDecomposition> {
    if !chain.read_only_proven
        || !chain.complete_partition
        || chain.faces_without_geometry != 0
        || chain.nonmanifold_edges != 0
        || chain.nonlocal_cross_site_edges != 0
        || chain.site_face_ids.len() != chain.sites
        || chain.gap_face_ids.len() + 1 != chain.sites
        || chain.site_centers_mm.len() != chain.sites
    {
        return None;
    }

    let axis_index = chain
        .axis
        .iter()
        .position(|component| component.abs() > 0.5)?;
    if chain
        .axis
        .iter()
        .enumerate()
        .any(|(index, component)| index != axis_index && component.abs() > f64::EPSILON)
    {
        return None;
    }

    let index = crate::step_graph::build_index(entities);
    let site_signatures = chain
        .site_face_ids
        .iter()
        .zip(&chain.site_centers_mm)
        .map(|(faces, center)| patch_signature(faces, axis_index, *center, entities, &index))
        .collect::<Option<Vec<_>>>()?;
    let gap_signatures = chain
        .gap_face_ids
        .iter()
        .enumerate()
        .map(|(gap_index, faces)| {
            let midpoint = f64::midpoint(
                chain.site_centers_mm[gap_index],
                chain.site_centers_mm[gap_index + 1],
            );
            patch_signature(faces, axis_index, midpoint, entities, &index)
        })
        .collect::<Option<Vec<_>>>()?;

    let motif_period_sites = infer_exact_period(&site_signatures, &gap_signatures)?;
    let motif_pitch_mm = chain.pitch_mm * motif_period_sites as f64;
    if !motif_pitch_mm.is_finite() || motif_pitch_mm <= 0.0 {
        return None;
    }
    let step_mm = [
        chain.axis[0] * motif_pitch_mm,
        chain.axis[1] * motif_pitch_mm,
        chain.axis[2] * motif_pitch_mm,
    ];

    let mut site_patterns = Vec::with_capacity(motif_period_sites);
    for residue in 0..motif_period_sites {
        site_patterns.push(PeriodicPatchPattern {
            source_index: residue,
            source_face_ids: chain.site_face_ids[residue].clone(),
            repeat_count: (chain.sites - residue).div_ceil(motif_period_sites),
            step_mm,
        });
    }

    let mut gap_patterns = Vec::new();
    for residue in 0..motif_period_sites.min(chain.gap_face_ids.len()) {
        gap_patterns.push(PeriodicPatchPattern {
            source_index: residue,
            source_face_ids: chain.gap_face_ids[residue].clone(),
            repeat_count: (chain.gap_face_ids.len() - residue).div_ceil(motif_period_sites),
            step_mm,
        });
    }

    let mut residual_face_ids = chain.stretch_face_ids.clone();
    residual_face_ids.extend(chain.fixed_negative_face_ids.iter().copied());
    residual_face_ids.extend(chain.fixed_middle_face_ids.iter().copied());
    residual_face_ids.extend(chain.fixed_positive_face_ids.iter().copied());
    residual_face_ids.sort_unstable();
    residual_face_ids.dedup();

    let mut covered = std::collections::HashSet::<u64>::new();
    for faces in &chain.site_face_ids {
        if !insert_unique(&mut covered, faces) {
            return None;
        }
    }
    for faces in &chain.gap_face_ids {
        if !insert_unique(&mut covered, faces) {
            return None;
        }
    }
    if !insert_unique(&mut covered, &residual_face_ids) {
        return None;
    }

    let mut source_faces = crate::brep::solid_face_ids(chain.solid_id, entities, &index)?;
    source_faces.sort_unstable();
    source_faces.dedup();
    let mut covered_faces = covered.iter().copied().collect::<Vec<_>>();
    covered_faces.sort_unstable();
    if source_faces != covered_faces {
        return None;
    }

    Some(PeriodicChainDecomposition {
        solid_id: chain.solid_id,
        primitive_pitch_mm: chain.pitch_mm,
        motif_period_sites,
        motif_pitch_mm,
        site_patterns,
        gap_patterns,
        residual_face_ids,
        covered_face_count: covered.len(),
    })
}

fn patch_signature(
    faces: &[u64],
    axis: usize,
    origin_mm: f64,
    entities: &[EntityInstance],
    index: &std::collections::HashMap<u64, usize>,
) -> Option<PatchSignature> {
    let mut origin = [0.0; 3];
    origin[axis] = origin_mm;
    let mut signatures = Vec::with_capacity(faces.len());
    for &face in faces {
        signatures.push(crate::shape_identity::face_topology_signature(
            face, entities, index, origin, 0,
        )?);
    }
    signatures.sort();
    Some(signatures)
}

fn infer_exact_period(
    site_signatures: &[PatchSignature],
    gap_signatures: &[PatchSignature],
) -> Option<usize> {
    let sites = site_signatures.len();
    if sites < 4 || gap_signatures.len() + 1 != sites {
        return None;
    }
    for period in 1..=MAX_MOTIF_PERIOD_SITES.min(sites) {
        if sites % period != 0 || sites / period < 4 {
            continue;
        }
        let sites_match = site_signatures
            .iter()
            .enumerate()
            .all(|(index, signature)| *signature == site_signatures[index % period]);
        if !sites_match {
            continue;
        }
        let gaps_match = gap_signatures
            .iter()
            .enumerate()
            .all(|(index, signature)| *signature == gap_signatures[index % period]);
        if gaps_match {
            return Some(period);
        }
    }
    None
}

fn insert_unique(seen: &mut std::collections::HashSet<u64>, faces: &[u64]) -> bool {
    faces.iter().all(|face| seen.insert(*face))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signature(value: &str) -> PatchSignature {
        vec![value.to_string()]
    }

    #[test]
    fn exact_period_detects_two_site_motif() {
        let sites = (0..8)
            .map(|index| signature(if index % 2 == 0 { "a" } else { "b" }))
            .collect::<Vec<_>>();
        let gaps = (0..7)
            .map(|index| signature(if index % 2 == 0 { "x" } else { "y" }))
            .collect::<Vec<_>>();
        assert_eq!(infer_exact_period(&sites, &gaps), Some(2));
    }

    #[test]
    fn exact_period_rejects_exceptional_tile() {
        let mut sites = vec![signature("a"); 8];
        sites[5] = signature("exception");
        let gaps = vec![signature("x"); 7];
        assert_eq!(infer_exact_period(&sites, &gaps), None);
    }
}
