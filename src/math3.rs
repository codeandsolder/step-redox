pub(crate) fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn scale(v: [f64; 3], scalar: f64) -> [f64; 3] {
    [v[0] * scalar, v[1] * scalar, v[2] * scalar]
}

pub(crate) use scale as mul;

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[2].mul_add(b[2], a[1].mul_add(b[1], a[0] * b[0]))
}

pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[2].mul_add(-b[1], a[1] * b[2]),
        a[0].mul_add(-b[2], a[2] * b[0]),
        a[1].mul_add(-b[0], a[0] * b[1]),
    ]
}

pub(crate) fn norm(v: [f64; 3]) -> f64 {
    v[0].hypot(v[1]).hypot(v[2])
}

pub(crate) fn normalize(v: [f64; 3], min_norm: f64) -> Option<[f64; 3]> {
    let length = normalization_length(v, min_norm)?;
    // Divide each component directly so each coordinate gets one rounding step.
    Some([v[0] / length, v[1] / length, v[2] / length])
}

pub(crate) fn canonical_direction(mut direction: [f64; 3]) -> [f64; 3] {
    let dominant = (0..3)
        .max_by(|&a, &b| direction[a].abs().total_cmp(&direction[b].abs()))
        .unwrap_or(0);
    if direction[dominant] < 0.0 {
        direction = scale(direction, -1.0);
    }
    direction.map(|value| if value == 0.0 { 0.0 } else { value })
}

pub(crate) fn canonical_unit_direction(direction: [f64; 3], min_norm: f64) -> Option<[f64; 3]> {
    Some(canonical_direction(normalize(direction, min_norm)?))
}

pub(crate) fn closest_point_on_unit_line_to_origin(
    origin: [f64; 3],
    direction: [f64; 3],
) -> [f64; 3] {
    sub(origin, scale(direction, dot(origin, direction)))
}

pub(crate) fn point_to_unit_line_distance(
    point: [f64; 3],
    origin: [f64; 3],
    direction: [f64; 3],
) -> f64 {
    norm(cross(sub(point, origin), direction))
}

fn normalization_length(v: [f64; 3], min_norm: f64) -> Option<f64> {
    let length = norm(v);
    if !length.is_finite() || !min_norm.is_finite() || min_norm < 0.0 || length <= min_norm {
        return None;
    }
    Some(length)
}

pub(crate) fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_keeps_tolerance_policy_explicit() {
        assert_eq!(normalize([3.0, 0.0, 4.0], 0.0), Some([0.6, 0.0, 0.8]));
        assert_eq!(normalize([1.0e-9, 0.0, 0.0], 1.0e-9), None);
        assert_eq!(normalize([0.0, 0.0, 0.0], 0.0), None);
        assert_eq!(normalize([f64::NAN, 0.0, 0.0], 0.0), None);
        assert_eq!(normalize([1.0, 0.0, 0.0], f64::NAN), None);
        assert!(norm([1.0e308, 1.0e308, 0.0]).is_finite());
        assert!(norm([1.0e-300, 0.0, 0.0]) > 0.0);
        assert_eq!(
            canonical_direction([1.0e-13, -1.0, -0.0]),
            [-1.0e-13, 1.0, 0.0]
        );
        assert_eq!(
            closest_point_on_unit_line_to_origin([4.0, 3.0, 0.0], [1.0, 0.0, 0.0]),
            [0.0, 3.0, 0.0]
        );
        assert_eq!(
            point_to_unit_line_distance([2.0, 3.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            3.0
        );
    }
}
