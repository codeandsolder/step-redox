use super::{CompactCurve, CompactSurface, fbits};
use crate::brep::BSplineSupport;
use crate::surface_recovery::BSplineSurfaceSupport;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

fn hash_f64(hasher: &mut impl Hasher, value: f64) {
    hasher.write_u64(fbits(value));
}

fn hash_vec3(hasher: &mut impl Hasher, value: [f64; 3]) {
    for component in value {
        hash_f64(hasher, component);
    }
}

fn hash_f64_slice(hasher: &mut impl Hasher, values: &[f64]) {
    hasher.write_usize(values.len());
    for &value in values {
        hash_f64(hasher, value);
    }
}

fn hash_bspline_curve(hasher: &mut impl Hasher, spline: &BSplineSupport) {
    hasher.write_usize(spline.degree);
    hasher.write_usize(spline.control_points_mm.len());
    for &point in &spline.control_points_mm {
        hash_vec3(hasher, point);
    }
    hash_f64_slice(hasher, &spline.knots);
    match &spline.weights {
        Some(weights) => {
            hasher.write_u8(1);
            hash_f64_slice(hasher, weights);
        }
        None => hasher.write_u8(0),
    }
}

fn hash_bspline_surface(hasher: &mut impl Hasher, spline: &BSplineSurfaceSupport) {
    hasher.write_usize(spline.u_degree);
    hasher.write_usize(spline.v_degree);
    hasher.write_usize(spline.control_points_mm.len());
    for row in &spline.control_points_mm {
        hasher.write_usize(row.len());
        for &point in row {
            hash_vec3(hasher, point);
        }
    }
    hash_f64_slice(hasher, &spline.u_knots);
    hash_f64_slice(hasher, &spline.v_knots);
    match &spline.weights {
        Some(rows) => {
            hasher.write_u8(1);
            hasher.write_usize(rows.len());
            for row in rows {
                hash_f64_slice(hasher, row);
            }
        }
        None => hasher.write_u8(0),
    }
}

pub(super) fn curve_hash(curve: &CompactCurve) -> u64 {
    let mut hasher = DefaultHasher::new();
    match curve {
        CompactCurve::Line {
            origin_mm,
            direction,
        } => {
            hasher.write_u8(0);
            hash_vec3(&mut hasher, *origin_mm);
            hash_vec3(&mut hasher, *direction);
        }
        CompactCurve::Circle {
            center_mm,
            normal,
            x_direction,
            radius_mm,
        } => {
            hasher.write_u8(1);
            hash_vec3(&mut hasher, *center_mm);
            hash_vec3(&mut hasher, *normal);
            hash_vec3(&mut hasher, *x_direction);
            hash_f64(&mut hasher, *radius_mm);
        }
        CompactCurve::BSpline { spline, .. } => {
            hasher.write_u8(2);
            hash_bspline_curve(&mut hasher, spline);
        }
        CompactCurve::Source { source_entity_id } => {
            hasher.write_u8(3);
            hasher.write_u64(*source_entity_id);
        }
    }
    hasher.finish()
}

pub(super) fn surface_hash(surface: &CompactSurface) -> u64 {
    let mut hasher = DefaultHasher::new();
    match surface {
        CompactSurface::Plane { origin_mm, normal } => {
            hasher.write_u8(0);
            hash_vec3(&mut hasher, *origin_mm);
            hash_vec3(&mut hasher, *normal);
        }
        CompactSurface::Cylinder {
            axis_origin_mm,
            axis,
            x_direction,
            radius_mm,
        } => {
            hasher.write_u8(1);
            hash_vec3(&mut hasher, *axis_origin_mm);
            hash_vec3(&mut hasher, *axis);
            hash_vec3(&mut hasher, *x_direction);
            hash_f64(&mut hasher, *radius_mm);
        }
        CompactSurface::Cone {
            reference_origin_mm,
            axis,
            x_direction,
            reference_radius_mm,
            semi_angle_rad,
        } => {
            hasher.write_u8(2);
            hash_vec3(&mut hasher, *reference_origin_mm);
            hash_vec3(&mut hasher, *axis);
            hash_vec3(&mut hasher, *x_direction);
            hash_f64(&mut hasher, *reference_radius_mm);
            hash_f64(&mut hasher, *semi_angle_rad);
        }
        CompactSurface::BSpline { spline, .. } => {
            hasher.write_u8(3);
            hash_bspline_surface(&mut hasher, spline);
        }
        CompactSurface::Sphere {
            center_mm,
            axis,
            x_direction,
            radius_mm,
        } => {
            hasher.write_u8(4);
            hash_vec3(&mut hasher, *center_mm);
            hash_vec3(&mut hasher, *axis);
            hash_vec3(&mut hasher, *x_direction);
            hash_f64(&mut hasher, *radius_mm);
        }
        CompactSurface::Torus {
            center_mm,
            axis,
            x_direction,
            major_radius_mm,
            minor_radius_mm,
        } => {
            hasher.write_u8(5);
            hash_vec3(&mut hasher, *center_mm);
            hash_vec3(&mut hasher, *axis);
            hash_vec3(&mut hasher, *x_direction);
            hash_f64(&mut hasher, *major_radius_mm);
            hash_f64(&mut hasher, *minor_radius_mm);
        }
        CompactSurface::Revolution { source_entity_id } => {
            hasher.write_u8(6);
            hasher.write_u64(*source_entity_id);
        }
        CompactSurface::Source { source_entity_id } => {
            hasher.write_u8(7);
            hasher.write_u64(*source_entity_id);
        }
    }
    hasher.finish()
}

fn f64_structural_eq(left: f64, right: f64) -> bool {
    fbits(left) == fbits(right)
}

fn vec3_structural_eq(left: [f64; 3], right: [f64; 3]) -> bool {
    left.into_iter()
        .zip(right)
        .all(|(left, right)| f64_structural_eq(left, right))
}

fn f64_slice_structural_eq(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(&left, &right)| f64_structural_eq(left, right))
}

fn bspline_curve_structural_eq(left: &BSplineSupport, right: &BSplineSupport) -> bool {
    left.degree == right.degree
        && left.control_points_mm.len() == right.control_points_mm.len()
        && left
            .control_points_mm
            .iter()
            .zip(&right.control_points_mm)
            .all(|(&left, &right)| vec3_structural_eq(left, right))
        && f64_slice_structural_eq(&left.knots, &right.knots)
        && match (&left.weights, &right.weights) {
            (Some(left), Some(right)) => f64_slice_structural_eq(left, right),
            (None, None) => true,
            _ => false,
        }
}

fn bspline_surface_structural_eq(
    left: &BSplineSurfaceSupport,
    right: &BSplineSurfaceSupport,
) -> bool {
    left.u_degree == right.u_degree
        && left.v_degree == right.v_degree
        && left.control_points_mm.len() == right.control_points_mm.len()
        && left
            .control_points_mm
            .iter()
            .zip(&right.control_points_mm)
            .all(|(left, right)| {
                left.len() == right.len()
                    && left
                        .iter()
                        .zip(right)
                        .all(|(&left, &right)| vec3_structural_eq(left, right))
            })
        && f64_slice_structural_eq(&left.u_knots, &right.u_knots)
        && f64_slice_structural_eq(&left.v_knots, &right.v_knots)
        && match (&left.weights, &right.weights) {
            (Some(left), Some(right)) => {
                left.len() == right.len()
                    && left
                        .iter()
                        .zip(right)
                        .all(|(left, right)| f64_slice_structural_eq(left, right))
            }
            (None, None) => true,
            _ => false,
        }
}

pub(super) fn curve_structural_eq(left: &CompactCurve, right: &CompactCurve) -> bool {
    match (left, right) {
        (
            CompactCurve::Line {
                origin_mm: left_origin,
                direction: left_direction,
            },
            CompactCurve::Line {
                origin_mm: right_origin,
                direction: right_direction,
            },
        ) => {
            vec3_structural_eq(*left_origin, *right_origin)
                && vec3_structural_eq(*left_direction, *right_direction)
        }
        (
            CompactCurve::Circle {
                center_mm: left_center,
                normal: left_normal,
                x_direction: left_x,
                radius_mm: left_radius,
            },
            CompactCurve::Circle {
                center_mm: right_center,
                normal: right_normal,
                x_direction: right_x,
                radius_mm: right_radius,
            },
        ) => {
            vec3_structural_eq(*left_center, *right_center)
                && vec3_structural_eq(*left_normal, *right_normal)
                && vec3_structural_eq(*left_x, *right_x)
                && f64_structural_eq(*left_radius, *right_radius)
        }
        (
            CompactCurve::BSpline { spline: left, .. },
            CompactCurve::BSpline { spline: right, .. },
        ) => bspline_curve_structural_eq(left, right),
        (
            CompactCurve::Source {
                source_entity_id: left,
            },
            CompactCurve::Source {
                source_entity_id: right,
            },
        ) => left == right,
        _ => false,
    }
}

pub(super) fn surface_structural_eq(left: &CompactSurface, right: &CompactSurface) -> bool {
    match (left, right) {
        (
            CompactSurface::Plane {
                origin_mm: left_origin,
                normal: left_normal,
            },
            CompactSurface::Plane {
                origin_mm: right_origin,
                normal: right_normal,
            },
        ) => {
            vec3_structural_eq(*left_origin, *right_origin)
                && vec3_structural_eq(*left_normal, *right_normal)
        }
        (
            CompactSurface::Cylinder {
                axis_origin_mm: left_origin,
                axis: left_axis,
                x_direction: left_x,
                radius_mm: left_radius,
            },
            CompactSurface::Cylinder {
                axis_origin_mm: right_origin,
                axis: right_axis,
                x_direction: right_x,
                radius_mm: right_radius,
            },
        ) => {
            vec3_structural_eq(*left_origin, *right_origin)
                && vec3_structural_eq(*left_axis, *right_axis)
                && vec3_structural_eq(*left_x, *right_x)
                && f64_structural_eq(*left_radius, *right_radius)
        }
        (
            CompactSurface::Cone {
                reference_origin_mm: left_origin,
                axis: left_axis,
                x_direction: left_x,
                reference_radius_mm: left_radius,
                semi_angle_rad: left_angle,
            },
            CompactSurface::Cone {
                reference_origin_mm: right_origin,
                axis: right_axis,
                x_direction: right_x,
                reference_radius_mm: right_radius,
                semi_angle_rad: right_angle,
            },
        ) => {
            vec3_structural_eq(*left_origin, *right_origin)
                && vec3_structural_eq(*left_axis, *right_axis)
                && vec3_structural_eq(*left_x, *right_x)
                && f64_structural_eq(*left_radius, *right_radius)
                && f64_structural_eq(*left_angle, *right_angle)
        }
        (
            CompactSurface::BSpline { spline: left, .. },
            CompactSurface::BSpline { spline: right, .. },
        ) => bspline_surface_structural_eq(left, right),
        (
            CompactSurface::Sphere {
                center_mm: left_center,
                axis: left_axis,
                x_direction: left_x,
                radius_mm: left_radius,
            },
            CompactSurface::Sphere {
                center_mm: right_center,
                axis: right_axis,
                x_direction: right_x,
                radius_mm: right_radius,
            },
        ) => {
            vec3_structural_eq(*left_center, *right_center)
                && vec3_structural_eq(*left_axis, *right_axis)
                && vec3_structural_eq(*left_x, *right_x)
                && f64_structural_eq(*left_radius, *right_radius)
        }
        (
            CompactSurface::Torus {
                center_mm: left_center,
                axis: left_axis,
                x_direction: left_x,
                major_radius_mm: left_major,
                minor_radius_mm: left_minor,
            },
            CompactSurface::Torus {
                center_mm: right_center,
                axis: right_axis,
                x_direction: right_x,
                major_radius_mm: right_major,
                minor_radius_mm: right_minor,
            },
        ) => {
            vec3_structural_eq(*left_center, *right_center)
                && vec3_structural_eq(*left_axis, *right_axis)
                && vec3_structural_eq(*left_x, *right_x)
                && f64_structural_eq(*left_major, *right_major)
                && f64_structural_eq(*left_minor, *right_minor)
        }
        (
            CompactSurface::Revolution {
                source_entity_id: left,
            },
            CompactSurface::Revolution {
                source_entity_id: right,
            },
        )
        | (
            CompactSurface::Source {
                source_entity_id: left,
            },
            CompactSurface::Source {
                source_entity_id: right,
            },
        ) => left == right,
        _ => false,
    }
}
