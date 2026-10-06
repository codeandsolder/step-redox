pub(crate) fn sub(left: [f64; 2], right: [f64; 2]) -> [f64; 2] {
    [left[0] - right[0], left[1] - right[1]]
}

pub(crate) fn dot(left: [f64; 2], right: [f64; 2]) -> f64 {
    left[1].mul_add(right[1], left[0] * right[0])
}

pub(crate) fn cross(left: [f64; 2], right: [f64; 2]) -> f64 {
    left[1].mul_add(-right[0], left[0] * right[1])
}

pub(crate) fn norm(vector: [f64; 2]) -> f64 {
    vector[0].hypot(vector[1])
}

pub(crate) fn distance(left: [f64; 2], right: [f64; 2]) -> f64 {
    norm(sub(left, right))
}

pub(crate) fn normalize(vector: [f64; 2], min_norm: f64) -> Option<[f64; 2]> {
    let length = norm(vector);
    if !length.is_finite() || length <= min_norm {
        return None;
    }
    Some([vector[0] / length, vector[1] / length])
}

#[cfg(test)]
mod tests {
    use super::{cross, distance, dot, norm, normalize, sub};

    #[test]
    fn basic_operations_are_stable() {
        assert_eq!(sub([3.0, 4.0], [1.0, 1.0]), [2.0, 3.0]);
        assert_eq!(dot([1.0, 2.0], [3.0, 4.0]), 11.0);
        assert_eq!(cross([1.0, 0.0], [0.0, 1.0]), 1.0);
        assert_eq!(norm([3.0, 4.0]), 5.0);
        assert_eq!(distance([3.0, 4.0], [0.0, 0.0]), 5.0);
        assert_eq!(normalize([3.0, 4.0], 1.0e-12), Some([0.6, 0.8]));
        assert_eq!(normalize([0.0, 0.0], 0.0), None);
    }
}
