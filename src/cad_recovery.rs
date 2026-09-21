use crate::cad_ir::{
    BrepFallback, CadModel, CadNode, NodeId, PatternSpec, ProofStatus, Provenance,
};
use crate::patterns::InstancePattern;
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
