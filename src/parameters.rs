use crate::math3::{canonical_direction, cross, norm, normalize as normalize3};
use serde::Serialize;

use crate::patterns::{InstancePattern, linear_pattern_sites};
use crate::periodic_bodies::PeriodicBodyPattern;

const TOL_MM: f64 = 1.0e-5;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RecoveredCountParameter {
    pub sites: usize,
    pub pitch_mm: f64,
    pub axis: [f64; 3],
    pub parent_representation: u64,
    pub instance_patterns: Vec<usize>,
    pub periodic_bodies: Vec<usize>,
    pub instances_per_site: usize,
    pub total_instances: usize,
    pub body_grammar_proven: bool,
    pub max_residual_mm: f64,
}

pub fn detect_count_parameters(
    patterns: &[InstancePattern],
    bodies: &[PeriodicBodyPattern],
) -> Vec<RecoveredCountParameter> {
    let mut out = Vec::<RecoveredCountParameter>::new();
    let mut used = vec![false; patterns.len()];

    // A periodic body is the strongest evidence that several instance rows are
    // one count parameter. Build these groups first.
    for (body_index, body) in bodies.iter().enumerate() {
        let mut indices = body.coupled_instance_patterns.clone();
        indices.sort_unstable();
        indices.dedup();
        indices.retain(|&index| index < patterns.len());
        if indices.is_empty() {
            continue;
        }

        if let Some(existing) = out.iter_mut().find(|parameter| {
            parameter.instance_patterns == indices
                && parameter.sites == body.sites
                && (parameter.pitch_mm - body.pitch_mm).abs() <= TOL_MM
        }) {
            existing.periodic_bodies.push(body_index);
            existing.body_grammar_proven = true;
            existing.max_residual_mm = existing.max_residual_mm.max(body.max_residual_mm);
            for &index in &indices {
                used[index] = true;
            }
            continue;
        }

        let first = &patterns[indices[0]];
        let Some(total_instances) = indices.iter().try_fold(0usize, |total, &index| {
            total.checked_add(linear_pattern_sites(&patterns[index])?)
        }) else {
            continue;
        };
        let parent_representation = first.parent_representation;
        out.push(RecoveredCountParameter {
            sites: body.sites,
            pitch_mm: body.pitch_mm,
            axis: canonical_direction(body.axis),
            parent_representation,
            instance_patterns: indices.clone(),
            periodic_bodies: vec![body_index],
            instances_per_site: total_instances / body.sites.max(1),
            total_instances,
            body_grammar_proven: true,
            max_residual_mm: body.max_residual_mm,
        });
        for &index in &indices {
            used[index] = true;
        }
    }

    // Remaining regular rows can still expose a count parameter, but are
    // explicitly marked as lacking a proven coupled body grammar.
    for index in 0..patterns.len() {
        if used[index] || !eligible(&patterns[index]) {
            continue;
        }
        let p = &patterns[index];
        let axis = canonical_direction(normalize3(p.basis[0], 1.0e-15).unwrap_or([1.0, 0.0, 0.0]));
        let pitch = norm(p.basis[0]);
        let Some(sites) = linear_pattern_sites(p) else {
            continue;
        };

        let mut indices = Vec::new();
        for other in index..patterns.len() {
            if used[other] || !eligible(&patterns[other]) {
                continue;
            }
            let q = &patterns[other];
            let q_sites = linear_pattern_sites(q);
            if q.parent_representation != p.parent_representation
                || q_sites != Some(sites)
                || (norm(q.basis[0]) - pitch).abs() > TOL_MM
            {
                continue;
            }
            let q_axis =
                canonical_direction(normalize3(q.basis[0], 1.0e-15).unwrap_or([1.0, 0.0, 0.0]));
            if norm(cross(axis, q_axis)) > 1.0e-8 {
                continue;
            }
            indices.push(other);
        }
        if indices.is_empty() {
            continue;
        }
        for &other in &indices {
            used[other] = true;
        }
        let Some(total_instances) = indices.iter().try_fold(0usize, |total, &other| {
            total.checked_add(linear_pattern_sites(&patterns[other])?)
        }) else {
            continue;
        };
        let max_residual_mm = indices
            .iter()
            .map(|&other| patterns[other].max_residual_mm)
            .fold(0.0f64, f64::max);
        out.push(RecoveredCountParameter {
            sites,
            pitch_mm: pitch,
            axis,
            parent_representation: p.parent_representation,
            instance_patterns: indices,
            periodic_bodies: Vec::new(),
            instances_per_site: total_instances / sites.max(1),
            total_instances,
            body_grammar_proven: false,
            max_residual_mm,
        });
    }

    out.sort_by(|a, b| {
        b.body_grammar_proven
            .cmp(&a.body_grammar_proven)
            .then_with(|| b.total_instances.cmp(&a.total_instances))
            .then_with(|| a.parent_representation.cmp(&b.parent_representation))
    });
    out
}

fn eligible(pattern: &InstancePattern) -> bool {
    linear_pattern_sites(pattern).is_some_and(|sites| sites >= 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_axis_is_sign_stable() {
        assert_eq!(canonical_direction([-1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]);
        assert_eq!(canonical_direction([0.0, -1.0, 0.0]), [0.0, 1.0, 0.0]);
    }
}
