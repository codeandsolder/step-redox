const MAX_EXACT_F64_INTEGER: u64 = 1_u64 << 53;
const TWO_POW_32: f64 = 4_294_967_296.0;

/// Convert an `i64` to `f64` only when the integer is exactly representable.
///
/// STEP permits integer-valued numeric parameters wider than the exact integer
/// range of IEEE-754 binary64. Geometry recovery must not silently round them.
pub(super) fn exact_i64_to_f64(value: i64) -> Option<f64> {
    let magnitude = value.unsigned_abs();
    if magnitude > MAX_EXACT_F64_INTEGER {
        return None;
    }

    let high = u32::try_from(magnitude >> 32).ok()?;
    let low = u32::try_from(magnitude & u64::from(u32::MAX)).ok()?;
    let magnitude_f64 = f64::from(high).mul_add(TWO_POW_32, f64::from(low));
    Some(if value.is_negative() {
        -magnitude_f64
    } else {
        magnitude_f64
    })
}

/// Convert a `usize` to `f64` only when the integer is exactly representable.
pub(super) fn exact_usize_to_f64(value: usize) -> Option<f64> {
    let value = u64::try_from(value).ok()?;
    if value > MAX_EXACT_F64_INTEGER {
        return None;
    }

    let high = u32::try_from(value >> 32).ok()?;
    let low = u32::try_from(value & u64::from(u32::MAX)).ok()?;
    Some(f64::from(high).mul_add(TWO_POW_32, f64::from(low)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i64_conversion_preserves_exact_binary64_integer_range() {
        assert_eq!(exact_i64_to_f64(0), Some(0.0));
        assert_eq!(exact_i64_to_f64(-42), Some(-42.0));
        assert_eq!(
            exact_i64_to_f64(9_007_199_254_740_992),
            Some(9_007_199_254_740_992.0)
        );
        assert_eq!(
            exact_i64_to_f64(-9_007_199_254_740_992),
            Some(-9_007_199_254_740_992.0)
        );
        assert_eq!(exact_i64_to_f64(9_007_199_254_740_993), None);
        assert_eq!(exact_i64_to_f64(-9_007_199_254_740_993), None);
        assert_eq!(exact_i64_to_f64(i64::MIN), None);
        assert_eq!(exact_i64_to_f64(i64::MAX), None);
    }

    #[test]
    fn usize_conversion_preserves_exact_binary64_integer_range() {
        assert_eq!(exact_usize_to_f64(0), Some(0.0));
        assert_eq!(exact_usize_to_f64(42), Some(42.0));
        if let (Ok(last_exact), Ok(first_inexact)) = (
            usize::try_from(9_007_199_254_740_992_u64),
            usize::try_from(9_007_199_254_740_993_u64),
        ) {
            assert_eq!(
                exact_usize_to_f64(last_exact),
                Some(9_007_199_254_740_992.0)
            );
            assert_eq!(exact_usize_to_f64(first_inexact), None);
        }
    }
}
