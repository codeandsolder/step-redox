use serde::Serialize;

use crate::patterns::InstancePattern;
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
        let total_instances = indices
            .iter()
            .map(|&index| patterns[index].item_ids.len())
            .sum::<usize>();
        let parent_representation = first.parent_representation;
        out.push(RecoveredCountParameter {
            sites: body.sites,
            pitch_mm: body.pitch_mm,
            axis: canonical_axis(body.axis),
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
        let axis = canonical_axis(normalized(p.basis[0]).unwrap_or([1.0, 0.0, 0.0]));
        let pitch = norm(p.basis[0]);
        let sites = p.item_ids.len();

        let mut indices = Vec::new();
        for other in index..patterns.len() {
            if used[other] || !eligible(&patterns[other]) {
                continue;
            }
            let q = &patterns[other];
            if q.parent_representation != p.parent_representation
                || q.item_ids.len() != sites
                || (norm(q.basis[0]) - pitch).abs() > TOL_MM
            {
                continue;
            }
            let q_axis = canonical_axis(normalized(q.basis[0]).unwrap_or([1.0, 0.0, 0.0]));
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
        let total_instances = indices
            .iter()
            .map(|&other| patterns[other].item_ids.len())
            .sum::<usize>();
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
    pattern.dimension == 1
        && pattern.item_ids.len() >= 2
        && pattern.grid_shape.len() == 1
        && pattern.grid_shape[0] == pattern.item_ids.len()
        && pattern.basis.len() == 1
        && (pattern.fill_ratio - 1.0).abs() <= 1.0e-12
}

fn canonical_axis(mut axis: [f64; 3]) -> [f64; 3] {
    for value in axis {
        if value.abs() > 1.0e-12 {
            if value < 0.0 {
                axis = [-axis[0], -axis[1], -axis[2]];
            }
            break;
        }
    }
    axis
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn normalized(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = norm(v);
    (n.is_finite() && n > 1.0e-15).then(|| [v[0] / n, v[1] / n, v[2] / n])
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_axis_is_sign_stable() {
        assert_eq!(canonical_axis([-1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]);
        assert_eq!(canonical_axis([0.0, -1.0, 0.0]), [0.0, 1.0, 0.0]);
    }
}
