use crate::brep::{self, BSplineSupport, CurveSupport, FaceLoop, SurfaceSupport};
use crate::step_graph::{build_index, simple_record};
use crate::surface_recovery::{self, BSplineSurfaceSupport};
use anyhow::{Context, Result, bail};
use ruststep::ast::EntityInstance;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CompactBrep {
    pub source_solid_id: u64,
    pub vertices: Vec<CompactVertex>,
    pub curves: Vec<CompactCurve>,
    pub edges: Vec<CompactEdge>,
    pub loops: Vec<CompactLoop>,
    pub surfaces: Vec<CompactSurface>,
    pub faces: Vec<CompactFace>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CompactVertex {
    pub source_vertex_id: u64,
    pub point_mm: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum CompactCurve {
    Line {
        origin_mm: [f64; 3],
        direction: [f64; 3],
    },
    Circle {
        center_mm: [f64; 3],
        normal: [f64; 3],
        x_direction: [f64; 3],
        radius_mm: f64,
    },
    BSpline {
        source_entity_id: u64,
        spline: BSplineSupport,
    },
    Source {
        source_entity_id: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CompactEdge {
    pub source_edge_id: u64,
    pub start_vertex: u32,
    pub end_vertex: u32,
    pub curve: u32,
    pub curve_same_sense: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CompactEdgeUse {
    pub edge: u32,
    pub forward: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CompactLoop {
    pub source_loop_id: u64,
    pub edges: Vec<CompactEdgeUse>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum CompactSurface {
    Plane {
        origin_mm: [f64; 3],
        normal: [f64; 3],
    },
    Cylinder {
        axis_origin_mm: [f64; 3],
        axis: [f64; 3],
        x_direction: [f64; 3],
        radius_mm: f64,
    },
    Cone {
        reference_origin_mm: [f64; 3],
        axis: [f64; 3],
        x_direction: [f64; 3],
        reference_radius_mm: f64,
        semi_angle_rad: f64,
    },
    BSpline {
        source_entity_id: u64,
        spline: BSplineSurfaceSupport,
    },
    Sphere {
        center_mm: [f64; 3],
        axis: [f64; 3],
        x_direction: [f64; 3],
        radius_mm: f64,
    },
    Torus {
        center_mm: [f64; 3],
        axis: [f64; 3],
        x_direction: [f64; 3],
        major_radius_mm: f64,
        minor_radius_mm: f64,
    },
    Revolution {
        source_entity_id: u64,
    },
    Source {
        source_entity_id: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CompactBoundary {
    pub loop_index: u32,
    pub outer: bool,
    pub orientation: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CompactFace {
    pub source_face_id: u64,
    pub surface: u32,
    pub same_sense: bool,
    pub boundaries: Vec<CompactBoundary>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CompactBrepStats {
    pub source_closure_entities: usize,
    pub vertices: usize,
    pub curves: usize,
    pub edges: usize,
    pub edge_uses: usize,
    pub loops: usize,
    pub surfaces: usize,
    pub faces: usize,
    pub boundaries: usize,
    pub compact_records: usize,
    pub serialized_json_bytes: usize,
    pub unresolved_curve_geometries: usize,
    pub unresolved_surface_geometries: usize,
    pub unresolved_curve_geometry_ids: Vec<u64>,
    pub unresolved_surface_geometry_ids: Vec<u64>,
    pub resident_payload_bytes_estimate: usize,
    pub packed_core_bytes_estimate: usize,
    pub provenance_sidecar_bytes_estimate: usize,
    pub packed_actual_core_bytes: usize,
    pub packed_actual_provenance_bytes: usize,
    pub packed_actual_json_bytes: usize,
    pub packed_self_contained: bool,
    pub vec3_occurrences: usize,
    pub unique_vec3_values: usize,
    pub vec3_pool_bytes_estimate: usize,
    pub vec3_pool_savings_estimate: usize,
    pub unique_scalar_values: usize,
    pub scalar_pool_index_bytes: usize,
    pub scalar_pool_bytes_estimate: usize,
    pub scalar_pool_savings_estimate: usize,
    pub u16_topology_bytes_estimate: Option<usize>,
    pub u16_topology_savings_estimate: Option<usize>,
    pub line_curves: usize,
    pub circle_curves: usize,
    pub bspline_curves: usize,
    pub plane_surfaces: usize,
    pub cylinder_surfaces: usize,
    pub cone_surfaces: usize,
    pub bspline_surfaces: usize,
    pub sphere_surfaces: usize,
    pub torus_surfaces: usize,
    pub revolution_surfaces: usize,
    pub bspline_control_points: usize,
    pub bspline_payload_bytes: usize,
    pub bspline_sequence_occurrences: usize,
    pub unique_bspline_sequences: usize,
    pub bspline_sequence_pool_bytes_estimate: usize,
    pub bspline_sequence_pool_savings_estimate: usize,
}

impl CompactBrep {
    pub fn stats(&self, source_closure_entities: usize) -> CompactBrepStats {
        let edge_uses = self.loops.iter().map(|loop_| loop_.edges.len()).sum();
        let boundaries = self.faces.iter().map(|face| face.boundaries.len()).sum();
        let compact_records = self.vertices.len()
            + self.curves.len()
            + self.edges.len()
            + edge_uses
            + self.loops.len()
            + self.surfaces.len()
            + self.faces.len()
            + boundaries;
        let unresolved_curve_geometry_ids = self
            .curves
            .iter()
            .filter_map(|curve| match curve {
                CompactCurve::Source { source_entity_id } => Some(*source_entity_id),
                _ => None,
            })
            .collect::<Vec<_>>();
        let unresolved_surface_geometry_ids = self
            .surfaces
            .iter()
            .filter_map(|surface| match surface {
                CompactSurface::Source { source_entity_id } => Some(*source_entity_id),
                _ => None,
            })
            .collect::<Vec<_>>();
        let resident_payload_bytes_estimate = self.resident_payload_bytes_estimate();
        let packed_core_bytes_estimate = self.packed_core_bytes_estimate();
        let provenance_sidecar_bytes_estimate = self.provenance_sidecar_bytes_estimate();
        let (
            vec3_occurrences,
            unique_vec3_values,
            vec3_pool_bytes_estimate,
            vec3_pool_savings_estimate,
            unique_scalar_values,
            scalar_pool_index_bytes,
            scalar_pool_bytes_estimate,
            scalar_pool_savings_estimate,
        ) = self.vec3_pool_stats();
        let (u16_topology_bytes_estimate, u16_topology_savings_estimate) =
            self.u16_topology_stats();
        let line_curves = self
            .curves
            .iter()
            .filter(|curve| matches!(curve, CompactCurve::Line { .. }))
            .count();
        let circle_curves = self
            .curves
            .iter()
            .filter(|curve| matches!(curve, CompactCurve::Circle { .. }))
            .count();
        let bspline_curves = self
            .curves
            .iter()
            .filter(|curve| matches!(curve, CompactCurve::BSpline { .. }))
            .count();
        let plane_surfaces = self
            .surfaces
            .iter()
            .filter(|surface| matches!(surface, CompactSurface::Plane { .. }))
            .count();
        let cylinder_surfaces = self
            .surfaces
            .iter()
            .filter(|surface| matches!(surface, CompactSurface::Cylinder { .. }))
            .count();
        let cone_surfaces = self
            .surfaces
            .iter()
            .filter(|surface| matches!(surface, CompactSurface::Cone { .. }))
            .count();
        let bspline_surfaces = self
            .surfaces
            .iter()
            .filter(|surface| matches!(surface, CompactSurface::BSpline { .. }))
            .count();
        let sphere_surfaces = self
            .surfaces
            .iter()
            .filter(|surface| matches!(surface, CompactSurface::Sphere { .. }))
            .count();
        let torus_surfaces = self
            .surfaces
            .iter()
            .filter(|surface| matches!(surface, CompactSurface::Torus { .. }))
            .count();
        let revolution_surfaces = self
            .surfaces
            .iter()
            .filter(|surface| matches!(surface, CompactSurface::Revolution { .. }))
            .count();
        let bspline_control_points = self
            .curves
            .iter()
            .filter_map(|curve| match curve {
                CompactCurve::BSpline { spline, .. } => Some(spline.control_points_mm.len()),
                _ => None,
            })
            .sum::<usize>()
            + self
                .surfaces
                .iter()
                .filter_map(|surface| match surface {
                    CompactSurface::BSpline { spline, .. } => {
                        Some(spline.control_points_mm.iter().map(Vec::len).sum::<usize>())
                    }
                    _ => None,
                })
                .sum::<usize>();
        let bspline_payload_bytes = self
            .curves
            .iter()
            .filter_map(|curve| match curve {
                CompactCurve::BSpline { spline, .. } => Some(bspline_curve_heap_bytes(spline)),
                _ => None,
            })
            .sum::<usize>()
            + self
                .surfaces
                .iter()
                .filter_map(|surface| match surface {
                    CompactSurface::BSpline { spline, .. } => {
                        Some(bspline_surface_heap_bytes(spline))
                    }
                    _ => None,
                })
                .sum::<usize>();
        let (
            bspline_sequence_occurrences,
            unique_bspline_sequences,
            bspline_sequence_pool_bytes_estimate,
            bspline_sequence_pool_savings_estimate,
        ) = self.bspline_sequence_pool_stats();

        CompactBrepStats {
            source_closure_entities,
            vertices: self.vertices.len(),
            curves: self.curves.len(),
            edges: self.edges.len(),
            edge_uses,
            loops: self.loops.len(),
            surfaces: self.surfaces.len(),
            faces: self.faces.len(),
            boundaries,
            compact_records,
            serialized_json_bytes: 0,
            unresolved_curve_geometries: unresolved_curve_geometry_ids.len(),
            unresolved_surface_geometries: unresolved_surface_geometry_ids.len(),
            unresolved_curve_geometry_ids,
            unresolved_surface_geometry_ids,
            resident_payload_bytes_estimate,
            packed_core_bytes_estimate,
            provenance_sidecar_bytes_estimate,
            packed_actual_core_bytes: 0,
            packed_actual_provenance_bytes: 0,
            packed_actual_json_bytes: 0,
            packed_self_contained: false,
            vec3_occurrences,
            unique_vec3_values,
            vec3_pool_bytes_estimate,
            vec3_pool_savings_estimate,
            unique_scalar_values,
            scalar_pool_index_bytes,
            scalar_pool_bytes_estimate,
            scalar_pool_savings_estimate,
            u16_topology_bytes_estimate,
            u16_topology_savings_estimate,
            line_curves,
            circle_curves,
            bspline_curves,
            plane_surfaces,
            cylinder_surfaces,
            cone_surfaces,
            bspline_surfaces,
            sphere_surfaces,
            torus_surfaces,
            revolution_surfaces,
            bspline_control_points,
            bspline_payload_bytes,
            bspline_sequence_occurrences,
            unique_bspline_sequences,
            bspline_sequence_pool_bytes_estimate,
            bspline_sequence_pool_savings_estimate,
        }
    }

    fn bspline_sequence_pool_stats(&self) -> (usize, usize, usize, usize) {
        let mut occurrences = 0usize;
        let mut raw_bytes = 0usize;
        let mut unique = HashMap::<Vec<u64>, usize>::new();

        let mut add = |values: &[f64]| {
            if values.is_empty() {
                return;
            }
            occurrences += 1;
            raw_bytes += std::mem::size_of_val(values);
            let key = values.iter().map(|&value| fbits(value)).collect::<Vec<_>>();
            unique.entry(key).or_insert(values.len());
        };

        for curve in &self.curves {
            if let CompactCurve::BSpline { spline, .. } = curve {
                add(&spline.knots);
                if let Some(weights) = &spline.weights {
                    add(weights);
                }
            }
        }
        for surface in &self.surfaces {
            if let CompactSurface::BSpline { spline, .. } = surface {
                add(&spline.u_knots);
                add(&spline.v_knots);
                if let Some(weight_rows) = &spline.weights {
                    let flattened = weight_rows
                        .iter()
                        .flat_map(|row| row.iter().copied())
                        .collect::<Vec<_>>();
                    add(&flattened);
                }
            }
        }

        let pooled_payload_bytes =
            unique.values().copied().sum::<usize>() * std::mem::size_of::<f64>();
        let pooled_bytes = pooled_payload_bytes + occurrences * std::mem::size_of::<u32>();
        (
            occurrences,
            unique.len(),
            pooled_bytes,
            raw_bytes.saturating_sub(pooled_bytes),
        )
    }

    fn vec3_pool_stats(&self) -> (usize, usize, usize, usize, usize, usize, usize, usize) {
        let mut unique = HashSet::<[u64; 3]>::new();
        let mut occurrences = 0usize;
        let mut add = |value: [f64; 3]| {
            occurrences += 1;
            unique.insert([fbits(value[0]), fbits(value[1]), fbits(value[2])]);
        };

        for vertex in &self.vertices {
            add(vertex.point_mm);
        }
        for curve in &self.curves {
            match curve {
                CompactCurve::Line {
                    origin_mm,
                    direction,
                } => {
                    add(*origin_mm);
                    add(*direction);
                }
                CompactCurve::Circle {
                    center_mm,
                    normal,
                    x_direction,
                    ..
                } => {
                    add(*center_mm);
                    add(*normal);
                    add(*x_direction);
                }
                CompactCurve::BSpline { spline, .. } => {
                    for &point in &spline.control_points_mm {
                        add(point);
                    }
                }
                CompactCurve::Source { .. } => {}
            }
        }
        for surface in &self.surfaces {
            match surface {
                CompactSurface::Plane { origin_mm, normal } => {
                    add(*origin_mm);
                    add(*normal);
                }
                CompactSurface::Cylinder {
                    axis_origin_mm,
                    axis,
                    x_direction,
                    ..
                } => {
                    add(*axis_origin_mm);
                    add(*axis);
                    add(*x_direction);
                }
                CompactSurface::Cone {
                    reference_origin_mm,
                    axis,
                    x_direction,
                    ..
                } => {
                    add(*reference_origin_mm);
                    add(*axis);
                    add(*x_direction);
                }
                CompactSurface::BSpline { spline, .. } => {
                    for row in &spline.control_points_mm {
                        for &point in row {
                            add(point);
                        }
                    }
                }
                CompactSurface::Sphere {
                    center_mm,
                    axis,
                    x_direction,
                    ..
                }
                | CompactSurface::Torus {
                    center_mm,
                    axis,
                    x_direction,
                    ..
                } => {
                    add(*center_mm);
                    add(*axis);
                    add(*x_direction);
                }
                CompactSurface::Revolution { .. } | CompactSurface::Source { .. } => {}
            }
        }

        let current_bytes = occurrences * std::mem::size_of::<[f64; 3]>();
        let pooled_bytes = unique.len() * std::mem::size_of::<[f64; 3]>()
            + occurrences * std::mem::size_of::<u32>();

        let mut scalar_values = HashSet::<u64>::new();
        for value in &unique {
            scalar_values.extend(value);
        }
        let scalar_index_bytes = if scalar_values.len() <= u16::MAX as usize + 1 {
            std::mem::size_of::<u16>()
        } else {
            std::mem::size_of::<u32>()
        };
        // A scalar-pooled representation replaces each unique Vec3 payload with
        // three scalar handles; occurrence handles remain u32.
        let scalar_pooled_bytes = scalar_values.len() * std::mem::size_of::<f64>()
            + unique.len() * 3 * scalar_index_bytes
            + occurrences * std::mem::size_of::<u32>();

        (
            occurrences,
            unique.len(),
            pooled_bytes,
            current_bytes.saturating_sub(pooled_bytes),
            scalar_values.len(),
            scalar_index_bytes,
            scalar_pooled_bytes,
            pooled_bytes.saturating_sub(scalar_pooled_bytes),
        )
    }

    fn u16_topology_stats(&self) -> (Option<usize>, Option<usize>) {
        let edge_uses = self
            .loops
            .iter()
            .map(|loop_| loop_.edges.len())
            .sum::<usize>();
        let boundaries = self
            .faces
            .iter()
            .map(|face| face.boundaries.len())
            .sum::<usize>();
        let fits = self.vertices.len() <= (u16::MAX as usize + 1)
            && self.curves.len() <= (1usize << 15)
            && self.edges.len() <= (1usize << 15)
            && edge_uses <= u16::MAX as usize
            && self.surfaces.len() <= (1usize << 15)
            && self.loops.len() <= (1usize << 14)
            && boundaries <= u16::MAX as usize;
        if !fits {
            return (None, None);
        }

        let entry_count = self.edges.len() * 3
            + edge_uses
            + self.loops.len()
            + 1
            + self.faces.len()
            + self.faces.len()
            + 1
            + boundaries;
        let current = entry_count * std::mem::size_of::<u32>();
        let narrowed = entry_count * std::mem::size_of::<u16>();
        (Some(narrowed), Some(current - narrowed))
    }

    fn resident_payload_bytes_estimate(&self) -> usize {
        use std::mem::size_of;

        let mut bytes = size_of::<Self>()
            + self.vertices.len() * size_of::<CompactVertex>()
            + self.curves.len() * size_of::<CompactCurve>()
            + self.edges.len() * size_of::<CompactEdge>()
            + self.loops.len() * size_of::<CompactLoop>()
            + self.surfaces.len() * size_of::<CompactSurface>()
            + self.faces.len() * size_of::<CompactFace>();
        bytes += self
            .loops
            .iter()
            .map(|loop_| loop_.edges.len() * size_of::<CompactEdgeUse>())
            .sum::<usize>();
        bytes += self
            .faces
            .iter()
            .map(|face| face.boundaries.len() * size_of::<CompactBoundary>())
            .sum::<usize>();
        for curve in &self.curves {
            if let CompactCurve::BSpline { spline, .. } = curve {
                bytes += bspline_curve_heap_bytes(spline);
            }
        }
        for surface in &self.surfaces {
            if let CompactSurface::BSpline { spline, .. } = surface {
                bytes += bspline_surface_heap_bytes(spline);
            }
        }
        bytes
    }

    fn packed_core_bytes_estimate(&self) -> usize {
        // Target layout: f64 geometry tables plus u32 topology/typed handles.
        // Direction/orientation flags use the spare high bits of u32 handles.
        let mut bytes = self.vertices.len() * 3 * size_of::<f64>();
        bytes += self.edges.len() * 3 * size_of::<u32>();
        bytes += self
            .loops
            .iter()
            .map(|loop_| loop_.edges.len())
            .sum::<usize>()
            * size_of::<u32>();
        bytes += (self.loops.len() + 1) * size_of::<u32>();

        // Curve handle table plus type-specific payload tables.
        bytes += self.curves.len() * size_of::<u32>();
        for curve in &self.curves {
            bytes += match curve {
                CompactCurve::Line { .. } => 6 * size_of::<f64>(),
                CompactCurve::Circle { .. } => 10 * size_of::<f64>(),
                CompactCurve::BSpline { spline, .. } => {
                    6 * size_of::<u32>()
                        + spline.control_points_mm.len() * 3 * size_of::<f64>()
                        + spline.knots.len() * size_of::<f64>()
                        + spline
                            .weights
                            .as_ref()
                            .map_or(0, |weights| weights.len() * size_of::<f64>())
                }
                CompactCurve::Source { .. } => size_of::<u64>(),
            };
        }

        // Surface handle table plus type-specific payload tables.
        bytes += self.surfaces.len() * size_of::<u32>();
        for surface in &self.surfaces {
            bytes += match surface {
                CompactSurface::Plane { .. } => 6 * size_of::<f64>(),
                CompactSurface::Cylinder { .. } => 10 * size_of::<f64>(),
                CompactSurface::Cone { .. } => 11 * size_of::<f64>(),
                CompactSurface::BSpline { spline, .. } => {
                    let point_count = spline.control_points_mm.iter().map(Vec::len).sum::<usize>();
                    let weight_count = spline
                        .weights
                        .as_ref()
                        .map_or(0, |rows| rows.iter().map(Vec::len).sum::<usize>());
                    size_of::<[u32; 8]>()
                        + point_count * 3 * size_of::<f64>()
                        + (spline.u_knots.len() + spline.v_knots.len()) * size_of::<f64>()
                        + weight_count * size_of::<f64>()
                }
                CompactSurface::Sphere { .. } => 10 * size_of::<f64>(),
                CompactSurface::Torus { .. } => 11 * size_of::<f64>(),
                CompactSurface::Revolution { .. } => size_of::<u64>(),
                CompactSurface::Source { .. } => size_of::<u64>(),
            };
        }

        // Faces: surface handle, face-boundary offset table, one bit same-sense.
        bytes += self.faces.len() * size_of::<u32>();
        bytes += (self.faces.len() + 1) * size_of::<u32>();
        bytes += self.faces.len().div_ceil(8);
        // Boundaries: loop index plus two flag bits in the same u32.
        bytes += self
            .faces
            .iter()
            .map(|face| face.boundaries.len())
            .sum::<usize>()
            * size_of::<u32>();
        bytes
    }

    fn provenance_sidecar_bytes_estimate(&self) -> usize {
        // Source STEP IDs can live out-of-band from the geometric core.
        // Keep u64 here so the estimate remains valid for arbitrary Part 21 IDs.
        let spline_curves = self
            .curves
            .iter()
            .filter(|curve| matches!(curve, CompactCurve::BSpline { .. }))
            .count();
        let spline_surfaces = self
            .surfaces
            .iter()
            .filter(|surface| matches!(surface, CompactSurface::BSpline { .. }))
            .count();
        (1 + self.vertices.len()
            + self.edges.len()
            + self.loops.len()
            + self.faces.len()
            + spline_curves
            + spline_surfaces)
            * size_of::<u64>()
    }
}

fn bspline_curve_heap_bytes(spline: &BSplineSupport) -> usize {
    spline.control_points_mm.len() * std::mem::size_of::<[f64; 3]>()
        + spline.knots.len() * std::mem::size_of::<f64>()
        + spline
            .weights
            .as_ref()
            .map_or(0, |weights| weights.len() * std::mem::size_of::<f64>())
}

fn bspline_surface_heap_bytes(spline: &BSplineSurfaceSupport) -> usize {
    let mut bytes = spline.control_points_mm.len() * std::mem::size_of::<Vec<[f64; 3]>>();
    bytes += spline
        .control_points_mm
        .iter()
        .map(|row| row.len() * std::mem::size_of::<[f64; 3]>())
        .sum::<usize>();
    bytes += (spline.u_knots.len() + spline.v_knots.len()) * std::mem::size_of::<f64>();
    if let Some(weights) = &spline.weights {
        bytes += weights.len() * std::mem::size_of::<Vec<f64>>();
        bytes += weights
            .iter()
            .map(|row| row.len() * std::mem::size_of::<f64>())
            .sum::<usize>();
    }
    bytes
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct PackedBrep {
    pub source_solid_id: u64,
    /// Exact scalar dictionary used by the Vec3 dictionary.
    pub scalar_pool: Vec<f64>,
    /// Exact de-duplicated vector/point table referencing scalar_pool.
    pub vec3_pool: PackedVec3Pool,
    /// One Vec3 handle per topological vertex.
    pub vertex_positions: PackedVec3Handles,
    /// One typed handle per unique curve support.
    pub curve_handles: Vec<u32>,
    pub lines: Vec<PackedLine>,
    pub circles: Vec<PackedCircle>,
    pub spline_sequences: PackedScalarSequences,
    pub bspline_curves: Vec<PackedBSplineCurve>,
    /// Non-self-contained escape hatch. Expected to be empty for optimized large bodies.
    pub unresolved_curve_entity_ids: Vec<u64>,
    /// Topology uses 16-bit indices when every cardinality/flagged handle fits,
    /// otherwise it transparently falls back to the original 32-bit layout.
    pub topology: PackedTopology,
    /// One typed handle per unique surface support.
    pub surface_handles: Vec<u32>,
    pub planes: Vec<PackedPlane>,
    pub cylinders: Vec<PackedCylinder>,
    pub cones: Vec<PackedCone>,
    pub bspline_surfaces: Vec<PackedBSplineSurface>,
    pub spheres: Vec<PackedSphere>,
    pub tori: Vec<PackedTorus>,
    /// Non-self-contained escape hatch. Expected to be empty for optimized large bodies.
    pub unresolved_surface_entity_ids: Vec<u64>,
    pub provenance: PackedBrepProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) enum PackedVec3Pool {
    U16(Vec<[u16; 3]>),
    U32(Vec<[u32; 3]>),
}

impl PackedVec3Pool {
    fn len(&self) -> usize {
        match self {
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
        }
    }

    fn payload_bytes(&self) -> usize {
        match self {
            Self::U16(values) => values.len() * std::mem::size_of::<[u16; 3]>(),
            Self::U32(values) => values.len() * std::mem::size_of::<[u32; 3]>(),
        }
    }

    fn validate(&self, scalar_count: usize) -> Result<()> {
        match self {
            Self::U16(values) => {
                for (index, value) in values.iter().enumerate() {
                    if value
                        .iter()
                        .any(|&component| component as usize >= scalar_count)
                    {
                        bail!("packed Vec3 {index} references scalar outside {scalar_count}");
                    }
                }
            }
            Self::U32(values) => {
                for (index, value) in values.iter().enumerate() {
                    if value
                        .iter()
                        .any(|&component| component as usize >= scalar_count)
                    {
                        bail!("packed Vec3 {index} references scalar outside {scalar_count}");
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) enum PackedVec3Handles {
    U16(Box<[u16]>),
    U32(Box<[u32]>),
}

impl PackedVec3Handles {
    fn from_u32(values: Vec<u32>) -> Self {
        if values.iter().all(|&value| value <= u16::MAX as u32) {
            Self::U16(
                values
                    .into_iter()
                    .map(|value| value as u16)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
        } else {
            Self::U32(values.into_boxed_slice())
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::U16(values) => values.len(),
            Self::U32(values) => values.len(),
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn payload_bytes(&self) -> usize {
        match self {
            Self::U16(values) => std::mem::size_of_val(values.as_ref()),
            Self::U32(values) => std::mem::size_of_val(values.as_ref()),
        }
    }

    fn validate(&self, vec3_count: usize) -> Result<()> {
        match self {
            Self::U16(values) => {
                if values.iter().any(|&value| value as usize >= vec3_count) {
                    bail!("packed Vec3 handle references entry outside {vec3_count}");
                }
            }
            Self::U32(values) => {
                if values.iter().any(|&value| value as usize >= vec3_count) {
                    bail!("packed Vec3 handle references entry outside {vec3_count}");
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) enum PackedTopology {
    U16 {
        edges: Vec<[u16; 3]>,
        edge_uses: Vec<u16>,
        loop_offsets: Vec<u16>,
        face_surfaces: Vec<u16>,
        face_boundary_offsets: Vec<u16>,
        boundaries: Vec<u16>,
    },
    U32 {
        edges: Vec<[u32; 3]>,
        edge_uses: Vec<u32>,
        loop_offsets: Vec<u32>,
        face_surfaces: Vec<u32>,
        face_boundary_offsets: Vec<u32>,
        boundaries: Vec<u32>,
    },
}

impl PackedTopology {
    fn from_u32(
        edges: Vec<[u32; 3]>,
        edge_uses: Vec<u32>,
        loop_offsets: Vec<u32>,
        face_surfaces: Vec<u32>,
        face_boundary_offsets: Vec<u32>,
        boundaries: Vec<u32>,
    ) -> Self {
        let fits_u16 = edges.iter().all(|edge| {
            edge[0] <= u16::MAX as u32
                && edge[1] <= u16::MAX as u32
                && (edge[2] & PACKED_INDEX_MASK_31) <= 0x7fff
        }) && edge_uses
            .iter()
            .all(|value| (value & PACKED_INDEX_MASK_31) <= 0x7fff)
            && loop_offsets.iter().all(|&value| value <= u16::MAX as u32)
            && face_surfaces
                .iter()
                .all(|value| (value & PACKED_INDEX_MASK_31) <= 0x7fff)
            && face_boundary_offsets
                .iter()
                .all(|&value| value <= u16::MAX as u32)
            && boundaries
                .iter()
                .all(|value| (value & PACKED_INDEX_MASK_30) <= 0x3fff);

        if !fits_u16 {
            return Self::U32 {
                edges,
                edge_uses,
                loop_offsets,
                face_surfaces,
                face_boundary_offsets,
                boundaries,
            };
        }

        let edges = edges
            .into_iter()
            .map(|edge| {
                [
                    edge[0] as u16,
                    edge[1] as u16,
                    (edge[2] & PACKED_INDEX_MASK_31) as u16
                        | if edge[2] >> 31 != 0 { 1u16 << 15 } else { 0 },
                ]
            })
            .collect();
        let edge_uses = edge_uses
            .into_iter()
            .map(|value| {
                (value & PACKED_INDEX_MASK_31) as u16
                    | if value >> 31 != 0 { 1u16 << 15 } else { 0 }
            })
            .collect();
        let loop_offsets = loop_offsets.into_iter().map(|value| value as u16).collect();
        let face_surfaces = face_surfaces
            .into_iter()
            .map(|value| {
                (value & PACKED_INDEX_MASK_31) as u16
                    | if value >> 31 != 0 { 1u16 << 15 } else { 0 }
            })
            .collect();
        let face_boundary_offsets = face_boundary_offsets
            .into_iter()
            .map(|value| value as u16)
            .collect();
        let boundaries = boundaries
            .into_iter()
            .map(|value| {
                (value & PACKED_INDEX_MASK_30) as u16
                    | if value & (1u32 << 30) != 0 {
                        1u16 << 14
                    } else {
                        0
                    }
                    | if value & (1u32 << 31) != 0 {
                        1u16 << 15
                    } else {
                        0
                    }
            })
            .collect();
        Self::U16 {
            edges,
            edge_uses,
            loop_offsets,
            face_surfaces,
            face_boundary_offsets,
            boundaries,
        }
    }

    fn edge_count(&self) -> usize {
        match self {
            Self::U16 { edges, .. } => edges.len(),
            Self::U32 { edges, .. } => edges.len(),
        }
    }

    fn loop_count(&self) -> usize {
        match self {
            Self::U16 { loop_offsets, .. } => loop_offsets.len().saturating_sub(1),
            Self::U32 { loop_offsets, .. } => loop_offsets.len().saturating_sub(1),
        }
    }

    fn face_count(&self) -> usize {
        match self {
            Self::U16 { face_surfaces, .. } => face_surfaces.len(),
            Self::U32 { face_surfaces, .. } => face_surfaces.len(),
        }
    }

    fn payload_bytes(&self) -> usize {
        use std::mem::size_of;
        match self {
            Self::U16 {
                edges,
                edge_uses,
                loop_offsets,
                face_surfaces,
                face_boundary_offsets,
                boundaries,
            } => {
                edges.len() * size_of::<[u16; 3]>()
                    + (edge_uses.len()
                        + loop_offsets.len()
                        + face_surfaces.len()
                        + face_boundary_offsets.len()
                        + boundaries.len())
                        * size_of::<u16>()
            }
            Self::U32 {
                edges,
                edge_uses,
                loop_offsets,
                face_surfaces,
                face_boundary_offsets,
                boundaries,
            } => {
                edges.len() * size_of::<[u32; 3]>()
                    + (edge_uses.len()
                        + loop_offsets.len()
                        + face_surfaces.len()
                        + face_boundary_offsets.len()
                        + boundaries.len())
                        * size_of::<u32>()
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct TopologyCounts {
    vertex_count: usize,
    curve_count: usize,
    surface_count: usize,
}

impl PackedTopology {
    fn validate(
        &self,
        vertex_count: usize,
        curve_count: usize,
        surface_count: usize,
    ) -> Result<()> {
        let counts = TopologyCounts {
            vertex_count,
            curve_count,
            surface_count,
        };
        match self {
            Self::U16 {
                edges,
                edge_uses,
                loop_offsets,
                face_surfaces,
                face_boundary_offsets,
                boundaries,
            } => validate_topology_u16(
                edges,
                edge_uses,
                loop_offsets,
                face_surfaces,
                face_boundary_offsets,
                boundaries,
                counts,
            ),
            Self::U32 {
                edges,
                edge_uses,
                loop_offsets,
                face_surfaces,
                face_boundary_offsets,
                boundaries,
            } => validate_topology_u32(
                edges,
                edge_uses,
                loop_offsets,
                face_surfaces,
                face_boundary_offsets,
                boundaries,
                counts,
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub(crate) struct PackedLine {
    pub origin: u32,
    pub direction: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub(crate) struct PackedCircle {
    pub center: u32,
    pub normal: u32,
    pub x_direction: u32,
    pub radius_mm: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct PackedScalarSequences {
    pub values: Vec<f64>,
    pub offsets: Vec<u32>,
}

impl PackedScalarSequences {
    fn get(&self, handle: u32) -> Result<&[f64]> {
        let index = handle as usize;
        let Some((&start, &end)) = self.offsets.get(index).zip(self.offsets.get(index + 1)) else {
            bail!("scalar-sequence handle {handle} is outside sequence table");
        };
        Ok(&self.values[start as usize..end as usize])
    }

    fn validate(&self) -> Result<()> {
        if self.offsets.is_empty() || self.offsets[0] != 0 {
            bail!("packed scalar-sequence offsets must start at zero");
        }
        validate_csr_offsets(&self.offsets, self.values.len(), "scalar sequence")?;
        if self.values.iter().any(|value| !value.is_finite()) {
            bail!("packed scalar sequence table contains non-finite value");
        }
        Ok(())
    }

    fn payload_bytes(&self) -> usize {
        self.values.len() * std::mem::size_of::<f64>()
            + self.offsets.len() * std::mem::size_of::<u32>()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct PackedBSplineCurve {
    pub degree: u32,
    pub control_points: PackedVec3Handles,
    pub knots_sequence: u32,
    pub weights_sequence: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub(crate) struct PackedPlane {
    pub origin: u32,
    pub normal: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub(crate) struct PackedCylinder {
    pub axis_origin: u32,
    pub axis: u32,
    pub x_direction: u32,
    pub radius_mm: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub(crate) struct PackedCone {
    pub reference_origin: u32,
    pub axis: u32,
    pub x_direction: u32,
    pub reference_radius_mm: f64,
    pub semi_angle_rad: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct PackedBSplineSurface {
    pub u_degree: u32,
    pub v_degree: u32,
    pub u_count: u32,
    pub v_count: u32,
    /// Row-major Vec3 handles; length must equal u_count * v_count.
    pub control_points: PackedVec3Handles,
    pub u_knots_sequence: u32,
    pub v_knots_sequence: u32,
    /// Row-major when present; u32::MAX means absent.
    pub weights_sequence: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub(crate) struct PackedSphere {
    pub center: u32,
    pub axis: u32,
    pub x_direction: u32,
    pub radius_mm: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub(crate) struct PackedTorus {
    pub center: u32,
    pub axis: u32,
    pub x_direction: u32,
    pub major_radius_mm: f64,
    pub minor_radius_mm: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) enum PackedBrepProvenance {
    U32 {
        vertex_source_ids: Vec<u32>,
        edge_source_ids: Vec<u32>,
        loop_source_ids: Vec<u32>,
        face_source_ids: Vec<u32>,
    },
    U64 {
        vertex_source_ids: Vec<u64>,
        edge_source_ids: Vec<u64>,
        loop_source_ids: Vec<u64>,
        face_source_ids: Vec<u64>,
    },
}

impl PackedBrepProvenance {
    fn from_u64(
        vertex_source_ids: Vec<u64>,
        edge_source_ids: Vec<u64>,
        loop_source_ids: Vec<u64>,
        face_source_ids: Vec<u64>,
    ) -> Self {
        let fits_u32 = vertex_source_ids
            .iter()
            .chain(&edge_source_ids)
            .chain(&loop_source_ids)
            .chain(&face_source_ids)
            .all(|&value| u32::try_from(value).is_ok());
        if fits_u32 {
            return Self::U32 {
                vertex_source_ids: vertex_source_ids
                    .into_iter()
                    .map(|value| value as u32)
                    .collect(),
                edge_source_ids: edge_source_ids
                    .into_iter()
                    .map(|value| value as u32)
                    .collect(),
                loop_source_ids: loop_source_ids
                    .into_iter()
                    .map(|value| value as u32)
                    .collect(),
                face_source_ids: face_source_ids
                    .into_iter()
                    .map(|value| value as u32)
                    .collect(),
            };
        }
        Self::U64 {
            vertex_source_ids,
            edge_source_ids,
            loop_source_ids,
            face_source_ids,
        }
    }

    fn matches_counts(&self, vertices: usize, edges: usize, loops: usize, faces: usize) -> bool {
        match self {
            Self::U32 {
                vertex_source_ids,
                edge_source_ids,
                loop_source_ids,
                face_source_ids,
            } => {
                vertex_source_ids.len() == vertices
                    && edge_source_ids.len() == edges
                    && loop_source_ids.len() == loops
                    && face_source_ids.len() == faces
            }
            Self::U64 {
                vertex_source_ids,
                edge_source_ids,
                loop_source_ids,
                face_source_ids,
            } => {
                vertex_source_ids.len() == vertices
                    && edge_source_ids.len() == edges
                    && loop_source_ids.len() == loops
                    && face_source_ids.len() == faces
            }
        }
    }

    fn payload_bytes(&self) -> usize {
        match self {
            Self::U32 {
                vertex_source_ids,
                edge_source_ids,
                loop_source_ids,
                face_source_ids,
            } => {
                (vertex_source_ids.len()
                    + edge_source_ids.len()
                    + loop_source_ids.len()
                    + face_source_ids.len())
                    * std::mem::size_of::<u32>()
            }
            Self::U64 {
                vertex_source_ids,
                edge_source_ids,
                loop_source_ids,
                face_source_ids,
            } => {
                (vertex_source_ids.len()
                    + edge_source_ids.len()
                    + loop_source_ids.len()
                    + face_source_ids.len())
                    * std::mem::size_of::<u64>()
            }
        }
    }
}

const PACKED_INDEX_MASK_31: u32 = 0x7fff_ffff;
const PACKED_INDEX_MASK_30: u32 = 0x3fff_ffff;
const CURVE_KIND_SHIFT: u32 = 30;
const CURVE_INDEX_MASK: u32 = (1u32 << CURVE_KIND_SHIFT) - 1;
const SURFACE_KIND_SHIFT: u32 = 29;
const SURFACE_INDEX_MASK: u32 = (1u32 << SURFACE_KIND_SHIFT) - 1;

const NO_SEQUENCE: u32 = u32::MAX;

struct ScalarSequencePool {
    values: Vec<f64>,
    offsets: Vec<u32>,
    by_bits: HashMap<Vec<u64>, u32>,
}

impl ScalarSequencePool {
    fn new() -> Self {
        Self {
            values: Vec::new(),
            offsets: vec![0],
            by_bits: HashMap::new(),
        }
    }

    fn intern(&mut self, values: &[f64]) -> Result<u32> {
        let key = values.iter().map(|&value| fbits(value)).collect::<Vec<_>>();
        if let Some(&existing) = self.by_bits.get(&key) {
            return Ok(existing);
        }
        let handle = as_u32(self.offsets.len() - 1, "scalar sequence")?;
        self.values.extend(key.iter().copied().map(f64::from_bits));
        self.offsets
            .push(as_u32(self.values.len(), "scalar sequence value")?);
        self.by_bits.insert(key, handle);
        Ok(handle)
    }

    fn finish(self) -> PackedScalarSequences {
        PackedScalarSequences {
            values: self.values,
            offsets: self.offsets,
        }
    }
}

#[derive(Default)]
struct Vec3Pool {
    values: Vec<[f64; 3]>,
    by_bits: HashMap<[u64; 3], u32>,
}

impl Vec3Pool {
    fn intern(&mut self, value: [f64; 3]) -> Result<u32> {
        let value = clean_vec(value);
        let key = [fbits(value[0]), fbits(value[1]), fbits(value[2])];
        if let Some(&existing) = self.by_bits.get(&key) {
            return Ok(existing);
        }
        let index = as_u32(self.values.len(), "Vec3 pool")?;
        self.values.push(value);
        self.by_bits.insert(key, index);
        Ok(index)
    }

    fn finish(self) -> Result<(Vec<f64>, PackedVec3Pool)> {
        let mut scalar_pool = Vec::<f64>::new();
        let mut scalar_by_bits = HashMap::<u64, u32>::new();
        let mut triples = Vec::<[u32; 3]>::with_capacity(self.values.len());

        for value in self.values {
            let mut triple = [0u32; 3];
            for axis in 0..3 {
                let bits = fbits(value[axis]);
                let handle = if let Some(&existing) = scalar_by_bits.get(&bits) {
                    existing
                } else {
                    let index = as_u32(scalar_pool.len(), "scalar pool")?;
                    scalar_pool.push(f64::from_bits(bits));
                    scalar_by_bits.insert(bits, index);
                    index
                };
                triple[axis] = handle;
            }
            triples.push(triple);
        }

        let vec3_pool = if scalar_pool.len() <= u16::MAX as usize + 1 {
            let values = triples
                .into_iter()
                .map(|value| {
                    Ok([
                        u16::try_from(value[0]).context("Vec3 scalar index exceeds u16")?,
                        u16::try_from(value[1]).context("Vec3 scalar index exceeds u16")?,
                        u16::try_from(value[2]).context("Vec3 scalar index exceeds u16")?,
                    ])
                })
                .collect::<Result<Vec<_>>>()?;
            PackedVec3Pool::U16(values)
        } else {
            PackedVec3Pool::U32(triples)
        };
        Ok((scalar_pool, vec3_pool))
    }
}

fn pack_bspline_curve(
    spline: BSplineSupport,
    vec3_pool: &mut Vec3Pool,
    sequences: &mut ScalarSequencePool,
) -> Result<PackedBSplineCurve> {
    let BSplineSupport {
        degree,
        control_points_mm,
        knots,
        weights,
    } = spline;
    let control_points = PackedVec3Handles::from_u32(
        control_points_mm
            .into_iter()
            .map(|point| vec3_pool.intern(point))
            .collect::<Result<Vec<_>>>()?,
    );
    let knots_sequence = sequences.intern(&knots)?;
    let weights_sequence = match weights {
        Some(weights) => sequences.intern(&weights)?,
        None => NO_SEQUENCE,
    };
    Ok(PackedBSplineCurve {
        degree: as_u32(degree, "B-spline degree")?,
        control_points,
        knots_sequence,
        weights_sequence,
    })
}

fn pack_bspline_surface(
    spline: BSplineSurfaceSupport,
    vec3_pool: &mut Vec3Pool,
    sequences: &mut ScalarSequencePool,
) -> Result<PackedBSplineSurface> {
    let BSplineSurfaceSupport {
        u_degree,
        v_degree,
        control_points_mm,
        u_knots,
        v_knots,
        weights,
    } = spline;
    let u_count = control_points_mm.len();
    let v_count = control_points_mm.first().map(Vec::len).unwrap_or(0);
    if u_count == 0 || v_count == 0 || control_points_mm.iter().any(|row| row.len() != v_count) {
        bail!("cannot pack empty or ragged B-spline surface control grid");
    }
    let control_points = PackedVec3Handles::from_u32(
        control_points_mm
            .into_iter()
            .flatten()
            .map(|point| vec3_pool.intern(point))
            .collect::<Result<Vec<_>>>()?,
    );
    let u_knots_sequence = sequences.intern(&u_knots)?;
    let v_knots_sequence = sequences.intern(&v_knots)?;
    let weights_sequence = match weights {
        Some(rows) => {
            if rows.len() != u_count || rows.iter().any(|row| row.len() != v_count) {
                bail!("cannot pack ragged B-spline surface weight grid");
            }
            let values = rows.into_iter().flatten().collect::<Vec<_>>();
            sequences.intern(&values)?
        }
        None => NO_SEQUENCE,
    };
    Ok(PackedBSplineSurface {
        u_degree: as_u32(u_degree, "B-spline surface U degree")?,
        v_degree: as_u32(v_degree, "B-spline surface V degree")?,
        u_count: as_u32(u_count, "B-spline surface U control-point count")?,
        v_count: as_u32(v_count, "B-spline surface V control-point count")?,
        control_points,
        u_knots_sequence,
        v_knots_sequence,
        weights_sequence,
    })
}

impl CompactBrep {
    pub(crate) fn into_packed(self) -> Result<PackedBrep> {
        let mut vec3_pool = Vec3Pool::default();
        let mut sequences = ScalarSequencePool::new();
        let vertex_positions = PackedVec3Handles::from_u32(
            self.vertices
                .iter()
                .map(|vertex| vec3_pool.intern(vertex.point_mm))
                .collect::<Result<Vec<_>>>()?,
        );
        let mut curve_handles = Vec::with_capacity(self.curves.len());
        let mut lines = Vec::new();
        let mut circles = Vec::new();
        let mut bspline_curves = Vec::new();
        let mut unresolved_curve_entity_ids = Vec::new();
        for curve in self.curves {
            let handle = match curve {
                CompactCurve::Line {
                    origin_mm,
                    direction,
                } => {
                    let index = packed_typed_index(lines.len(), CURVE_INDEX_MASK, "line")?;
                    lines.push(PackedLine {
                        origin: vec3_pool.intern(origin_mm)?,
                        direction: vec3_pool.intern(direction)?,
                    });
                    pack_typed_handle(0, index, CURVE_KIND_SHIFT)
                }
                CompactCurve::Circle {
                    center_mm,
                    normal,
                    x_direction,
                    radius_mm,
                } => {
                    let index = packed_typed_index(circles.len(), CURVE_INDEX_MASK, "circle")?;
                    circles.push(PackedCircle {
                        center: vec3_pool.intern(center_mm)?,
                        normal: vec3_pool.intern(normal)?,
                        x_direction: vec3_pool.intern(x_direction)?,
                        radius_mm,
                    });
                    pack_typed_handle(1, index, CURVE_KIND_SHIFT)
                }
                CompactCurve::BSpline { spline, .. } => {
                    let index =
                        packed_typed_index(bspline_curves.len(), CURVE_INDEX_MASK, "B-spline")?;
                    bspline_curves.push(pack_bspline_curve(
                        spline,
                        &mut vec3_pool,
                        &mut sequences,
                    )?);
                    pack_typed_handle(2, index, CURVE_KIND_SHIFT)
                }
                CompactCurve::Source { source_entity_id } => {
                    let index = packed_typed_index(
                        unresolved_curve_entity_ids.len(),
                        CURVE_INDEX_MASK,
                        "unresolved curve",
                    )?;
                    unresolved_curve_entity_ids.push(source_entity_id);
                    pack_typed_handle(3, index, CURVE_KIND_SHIFT)
                }
            };
            curve_handles.push(handle);
        }

        let mut surface_handles = Vec::with_capacity(self.surfaces.len());
        let mut planes = Vec::new();
        let mut cylinders = Vec::new();
        let mut cones = Vec::new();
        let mut bspline_surfaces = Vec::new();
        let mut spheres = Vec::new();
        let mut tori = Vec::new();
        let mut unresolved_surface_entity_ids = Vec::new();
        for surface in self.surfaces {
            let (kind, index) = match surface {
                CompactSurface::Plane { origin_mm, normal } => {
                    let index = packed_typed_index(planes.len(), SURFACE_INDEX_MASK, "plane")?;
                    planes.push(PackedPlane {
                        origin: vec3_pool.intern(origin_mm)?,
                        normal: vec3_pool.intern(normal)?,
                    });
                    (0, index)
                }
                CompactSurface::Cylinder {
                    axis_origin_mm,
                    axis,
                    x_direction,
                    radius_mm,
                } => {
                    let index =
                        packed_typed_index(cylinders.len(), SURFACE_INDEX_MASK, "cylinder")?;
                    cylinders.push(PackedCylinder {
                        axis_origin: vec3_pool.intern(axis_origin_mm)?,
                        axis: vec3_pool.intern(axis)?,
                        x_direction: vec3_pool.intern(x_direction)?,
                        radius_mm,
                    });
                    (1, index)
                }
                CompactSurface::Cone {
                    reference_origin_mm,
                    axis,
                    x_direction,
                    reference_radius_mm,
                    semi_angle_rad,
                } => {
                    let index = packed_typed_index(cones.len(), SURFACE_INDEX_MASK, "cone")?;
                    cones.push(PackedCone {
                        reference_origin: vec3_pool.intern(reference_origin_mm)?,
                        axis: vec3_pool.intern(axis)?,
                        x_direction: vec3_pool.intern(x_direction)?,
                        reference_radius_mm,
                        semi_angle_rad,
                    });
                    (2, index)
                }
                CompactSurface::BSpline { spline, .. } => {
                    let index = packed_typed_index(
                        bspline_surfaces.len(),
                        SURFACE_INDEX_MASK,
                        "B-spline surface",
                    )?;
                    bspline_surfaces.push(pack_bspline_surface(
                        spline,
                        &mut vec3_pool,
                        &mut sequences,
                    )?);
                    (3, index)
                }
                CompactSurface::Sphere {
                    center_mm,
                    axis,
                    x_direction,
                    radius_mm,
                } => {
                    let index = packed_typed_index(spheres.len(), SURFACE_INDEX_MASK, "sphere")?;
                    spheres.push(PackedSphere {
                        center: vec3_pool.intern(center_mm)?,
                        axis: vec3_pool.intern(axis)?,
                        x_direction: vec3_pool.intern(x_direction)?,
                        radius_mm,
                    });
                    (4, index)
                }
                CompactSurface::Torus {
                    center_mm,
                    axis,
                    x_direction,
                    major_radius_mm,
                    minor_radius_mm,
                } => {
                    let index = packed_typed_index(tori.len(), SURFACE_INDEX_MASK, "torus")?;
                    tori.push(PackedTorus {
                        center: vec3_pool.intern(center_mm)?,
                        axis: vec3_pool.intern(axis)?,
                        x_direction: vec3_pool.intern(x_direction)?,
                        major_radius_mm,
                        minor_radius_mm,
                    });
                    (5, index)
                }
                CompactSurface::Revolution { source_entity_id }
                | CompactSurface::Source { source_entity_id } => {
                    let index = packed_typed_index(
                        unresolved_surface_entity_ids.len(),
                        SURFACE_INDEX_MASK,
                        "unresolved surface",
                    )?;
                    unresolved_surface_entity_ids.push(source_entity_id);
                    (7, index)
                }
            };
            surface_handles.push(pack_typed_handle(kind, index, SURFACE_KIND_SHIFT));
        }

        let vertex_source_ids = self
            .vertices
            .iter()
            .map(|vertex| vertex.source_vertex_id)
            .collect::<Vec<_>>();

        let mut edges = Vec::with_capacity(self.edges.len());
        let edge_source_ids = self
            .edges
            .iter()
            .map(|edge| edge.source_edge_id)
            .collect::<Vec<_>>();
        for edge in &self.edges {
            if edge.start_vertex > PACKED_INDEX_MASK_31
                || edge.end_vertex > PACKED_INDEX_MASK_31
                || edge.curve > PACKED_INDEX_MASK_31
            {
                bail!("compact B-rep exceeds 31-bit topology index capacity");
            }
            let curve_and_sense = edge.curve | if edge.curve_same_sense { 1u32 << 31 } else { 0 };
            edges.push([edge.start_vertex, edge.end_vertex, curve_and_sense]);
        }

        let mut edge_uses = Vec::new();
        let mut loop_offsets = Vec::with_capacity(self.loops.len() + 1);
        let mut loop_source_ids = Vec::with_capacity(self.loops.len());
        loop_offsets.push(0);
        for loop_ in &self.loops {
            loop_source_ids.push(loop_.source_loop_id);
            for edge_use in &loop_.edges {
                if edge_use.edge > PACKED_INDEX_MASK_31 {
                    bail!("compact B-rep edge-use index exceeds 31 bits");
                }
                edge_uses.push(edge_use.edge | if edge_use.forward { 1u32 << 31 } else { 0 });
            }
            loop_offsets.push(as_u32(edge_uses.len(), "edge use")?);
        }

        let mut face_surfaces = Vec::with_capacity(self.faces.len());
        let mut face_boundary_offsets = Vec::with_capacity(self.faces.len() + 1);
        let mut boundaries = Vec::new();
        let mut face_source_ids = Vec::with_capacity(self.faces.len());
        face_boundary_offsets.push(0);
        for face in &self.faces {
            if face.surface > PACKED_INDEX_MASK_31 {
                bail!("compact B-rep face surface index exceeds 31 bits");
            }
            face_source_ids.push(face.source_face_id);
            face_surfaces.push(face.surface | if face.same_sense { 1u32 << 31 } else { 0 });
            for boundary in &face.boundaries {
                if boundary.loop_index > PACKED_INDEX_MASK_30 {
                    bail!("compact B-rep boundary loop index exceeds 30 bits");
                }
                boundaries.push(
                    boundary.loop_index
                        | if boundary.outer { 1u32 << 30 } else { 0 }
                        | if boundary.orientation { 1u32 << 31 } else { 0 },
                );
            }
            face_boundary_offsets.push(as_u32(boundaries.len(), "face boundary")?);
        }

        let topology = PackedTopology::from_u32(
            edges,
            edge_uses,
            loop_offsets,
            face_surfaces,
            face_boundary_offsets,
            boundaries,
        );
        let (scalar_pool, vec3_pool) = vec3_pool.finish()?;
        let spline_sequences = sequences.finish();
        let packed = PackedBrep {
            source_solid_id: self.source_solid_id,
            scalar_pool,
            vec3_pool,
            vertex_positions,
            curve_handles,
            lines,
            circles,
            spline_sequences,
            bspline_curves,
            unresolved_curve_entity_ids,
            topology,
            surface_handles,
            planes,
            cylinders,
            cones,
            bspline_surfaces,
            spheres,
            tori,
            unresolved_surface_entity_ids,
            provenance: PackedBrepProvenance::from_u64(
                vertex_source_ids,
                edge_source_ids,
                loop_source_ids,
                face_source_ids,
            ),
        };
        packed.validate()?;
        Ok(packed)
    }
}

pub(crate) fn build_packed_brep(solid_id: u64, entities: &[EntityInstance]) -> Result<PackedBrep> {
    build_compact_brep(solid_id, entities)?.into_packed()
}

pub(crate) fn build_packed_brep_faces(
    solid_id: u64,
    face_ids: &[u64],
    entities: &[EntityInstance],
) -> Result<PackedBrep> {
    build_compact_brep_faces(solid_id, face_ids, entities)?.into_packed()
}

impl PackedBrep {
    pub(crate) fn validate(&self) -> Result<()> {
        let vec3_count = self.vec3_pool.len();
        let vertex_count = self.vertex_positions.len();
        let curve_count = self.curve_handles.len();
        let edge_count = self.topology.edge_count();
        let loop_count = self.topology.loop_count();
        let surface_count = self.surface_handles.len();
        let face_count = self.topology.face_count();
        self.topology
            .validate(vertex_count, curve_count, surface_count)?;

        for (index, scalar) in self.scalar_pool.iter().copied().enumerate() {
            if !scalar.is_finite() {
                bail!("packed B-rep scalar pool entry {index} is not finite");
            }
        }
        self.vec3_pool.validate(self.scalar_pool.len())?;
        self.vertex_positions.validate(vec3_count)?;

        for (index, handle) in self.curve_handles.iter().copied().enumerate() {
            let kind = handle >> CURVE_KIND_SHIFT;
            let item = (handle & CURVE_INDEX_MASK) as usize;
            let len = match kind {
                0 => self.lines.len(),
                1 => self.circles.len(),
                2 => self.bspline_curves.len(),
                3 => self.unresolved_curve_entity_ids.len(),
                _ => bail!("packed B-rep curve handle {index} has invalid kind {kind}"),
            };
            if item >= len {
                bail!(
                    "packed B-rep curve handle {index} points to {item}, but kind {kind} has {len} items"
                );
            }
        }

        for (index, handle) in self.surface_handles.iter().copied().enumerate() {
            let kind = handle >> SURFACE_KIND_SHIFT;
            let item = (handle & SURFACE_INDEX_MASK) as usize;
            let len = match kind {
                0 => self.planes.len(),
                1 => self.cylinders.len(),
                2 => self.cones.len(),
                3 => self.bspline_surfaces.len(),
                4 => self.spheres.len(),
                5 => self.tori.len(),
                7 => self.unresolved_surface_entity_ids.len(),
                _ => bail!("packed B-rep surface handle {index} has invalid kind {kind}"),
            };
            if item >= len {
                bail!(
                    "packed B-rep surface handle {index} points to {item}, but kind {kind} has {len} items"
                );
            }
        }

        if !self
            .provenance
            .matches_counts(vertex_count, edge_count, loop_count, face_count)
        {
            bail!("packed B-rep provenance cardinalities disagree with topology tables");
        }

        for (index, line) in self.lines.iter().enumerate() {
            validate_vec3_handle(line.origin, vec3_count, "line origin", index)?;
            validate_vec3_handle(line.direction, vec3_count, "line direction", index)?;
        }
        for (index, circle) in self.circles.iter().enumerate() {
            validate_vec3_handle(circle.center, vec3_count, "circle center", index)?;
            validate_vec3_handle(circle.normal, vec3_count, "circle normal", index)?;
            validate_vec3_handle(circle.x_direction, vec3_count, "circle x direction", index)?;
            if !circle.radius_mm.is_finite() || circle.radius_mm <= 0.0 {
                bail!("packed B-rep circle {index} has invalid radius");
            }
        }
        for (index, plane) in self.planes.iter().enumerate() {
            validate_vec3_handle(plane.origin, vec3_count, "plane origin", index)?;
            validate_vec3_handle(plane.normal, vec3_count, "plane normal", index)?;
        }
        for (index, cylinder) in self.cylinders.iter().enumerate() {
            validate_vec3_handle(
                cylinder.axis_origin,
                vec3_count,
                "cylinder axis origin",
                index,
            )?;
            validate_vec3_handle(cylinder.axis, vec3_count, "cylinder axis", index)?;
            validate_vec3_handle(
                cylinder.x_direction,
                vec3_count,
                "cylinder x direction",
                index,
            )?;
            if !cylinder.radius_mm.is_finite() || cylinder.radius_mm <= 0.0 {
                bail!("packed B-rep cylinder {index} has invalid radius");
            }
        }
        for (index, cone) in self.cones.iter().enumerate() {
            validate_vec3_handle(
                cone.reference_origin,
                vec3_count,
                "cone reference origin",
                index,
            )?;
            validate_vec3_handle(cone.axis, vec3_count, "cone axis", index)?;
            validate_vec3_handle(cone.x_direction, vec3_count, "cone x direction", index)?;
            if !cone.reference_radius_mm.is_finite()
                || !cone.semi_angle_rad.is_finite()
                || cone.reference_radius_mm < 0.0
            {
                bail!("packed B-rep cone {index} has invalid scalar geometry");
            }
        }
        for (index, sphere) in self.spheres.iter().enumerate() {
            validate_vec3_handle(sphere.center, vec3_count, "sphere center", index)?;
            validate_vec3_handle(sphere.axis, vec3_count, "sphere axis", index)?;
            validate_vec3_handle(sphere.x_direction, vec3_count, "sphere x direction", index)?;
            if !sphere.radius_mm.is_finite() || sphere.radius_mm <= 0.0 {
                bail!("packed B-rep sphere {index} has invalid radius");
            }
        }
        for (index, torus) in self.tori.iter().enumerate() {
            validate_vec3_handle(torus.center, vec3_count, "torus center", index)?;
            validate_vec3_handle(torus.axis, vec3_count, "torus axis", index)?;
            validate_vec3_handle(torus.x_direction, vec3_count, "torus x direction", index)?;
            if !torus.major_radius_mm.is_finite()
                || !torus.minor_radius_mm.is_finite()
                || torus.major_radius_mm <= 0.0
                || torus.minor_radius_mm <= 0.0
            {
                bail!("packed B-rep torus {index} has invalid radii");
            }
        }

        self.spline_sequences.validate()?;
        for (index, spline) in self.bspline_curves.iter().enumerate() {
            validate_packed_bspline_curve(spline, vec3_count, &self.spline_sequences)
                .with_context(|| format!("packed B-rep B-spline curve {index}"))?;
        }
        for (index, spline) in self.bspline_surfaces.iter().enumerate() {
            validate_packed_bspline_surface(spline, vec3_count, &self.spline_sequences)
                .with_context(|| format!("packed B-rep B-spline surface {index}"))?;
        }
        Ok(())
    }

    pub(crate) fn self_contained(&self) -> bool {
        self.unresolved_curve_entity_ids.is_empty() && self.unresolved_surface_entity_ids.is_empty()
    }

    pub(crate) fn edge_count(&self) -> usize {
        self.topology.edge_count()
    }

    pub(crate) fn face_count(&self) -> usize {
        self.topology.face_count()
    }

    pub(crate) fn control_point_count(&self) -> usize {
        self.bspline_curves
            .iter()
            .map(|spline| spline.control_points.len())
            .sum::<usize>()
            + self
                .bspline_surfaces
                .iter()
                .map(|spline| spline.control_points.len())
                .sum::<usize>()
    }

    pub(crate) fn core_payload_bytes(&self) -> usize {
        use std::mem::size_of;
        let mut bytes = size_of::<Self>()
            + self.scalar_pool.len() * size_of::<f64>()
            + self.vec3_pool.payload_bytes()
            + self.vertex_positions.payload_bytes()
            + self.curve_handles.len() * size_of::<u32>()
            + self.lines.len() * size_of::<PackedLine>()
            + self.circles.len() * size_of::<PackedCircle>()
            + self.spline_sequences.payload_bytes()
            + self.bspline_curves.len() * size_of::<PackedBSplineCurve>()
            + self.unresolved_curve_entity_ids.len() * size_of::<u64>()
            + self.topology.payload_bytes()
            + self.surface_handles.len() * size_of::<u32>()
            + self.planes.len() * size_of::<PackedPlane>()
            + self.cylinders.len() * size_of::<PackedCylinder>()
            + self.cones.len() * size_of::<PackedCone>()
            + self.bspline_surfaces.len() * size_of::<PackedBSplineSurface>()
            + self.spheres.len() * size_of::<PackedSphere>()
            + self.tori.len() * size_of::<PackedTorus>()
            + self.unresolved_surface_entity_ids.len() * size_of::<u64>();
        for spline in &self.bspline_curves {
            bytes += spline.control_points.payload_bytes();
        }
        for spline in &self.bspline_surfaces {
            bytes += spline.control_points.payload_bytes();
        }
        bytes
    }

    pub(crate) fn provenance_payload_bytes(&self) -> usize {
        self.provenance.payload_bytes()
    }
}

fn validate_vec3_handle(handle: u32, pool_len: usize, what: &str, index: usize) -> Result<()> {
    if handle as usize >= pool_len {
        bail!("packed B-rep {what} {index} references Vec3 {handle} outside {pool_len}");
    }
    Ok(())
}

fn validate_topology_u16(
    edges: &[[u16; 3]],
    edge_uses: &[u16],
    loop_offsets: &[u16],
    face_surfaces: &[u16],
    face_boundary_offsets: &[u16],
    boundaries: &[u16],
    counts: TopologyCounts,
) -> Result<()> {
    let edges = edges
        .iter()
        .map(|edge| {
            [
                edge[0] as u32,
                edge[1] as u32,
                (edge[2] & 0x7fff) as u32 | if edge[2] & 0x8000 != 0 { 1u32 << 31 } else { 0 },
            ]
        })
        .collect::<Vec<_>>();
    let edge_uses = edge_uses
        .iter()
        .map(|&value| (value & 0x7fff) as u32 | if value & 0x8000 != 0 { 1u32 << 31 } else { 0 })
        .collect::<Vec<_>>();
    let loop_offsets = loop_offsets
        .iter()
        .map(|&value| value as u32)
        .collect::<Vec<_>>();
    let face_surfaces = face_surfaces
        .iter()
        .map(|&value| (value & 0x7fff) as u32 | if value & 0x8000 != 0 { 1u32 << 31 } else { 0 })
        .collect::<Vec<_>>();
    let face_boundary_offsets = face_boundary_offsets
        .iter()
        .map(|&value| value as u32)
        .collect::<Vec<_>>();
    let boundaries = boundaries
        .iter()
        .map(|&value| {
            (value & 0x3fff) as u32
                | if value & 0x4000 != 0 { 1u32 << 30 } else { 0 }
                | if value & 0x8000 != 0 { 1u32 << 31 } else { 0 }
        })
        .collect::<Vec<_>>();
    validate_topology_u32(
        &edges,
        &edge_uses,
        &loop_offsets,
        &face_surfaces,
        &face_boundary_offsets,
        &boundaries,
        counts,
    )
}

fn validate_topology_u32(
    edges: &[[u32; 3]],
    edge_uses: &[u32],
    loop_offsets: &[u32],
    face_surfaces: &[u32],
    face_boundary_offsets: &[u32],
    boundaries: &[u32],
    counts: TopologyCounts,
) -> Result<()> {
    let TopologyCounts {
        vertex_count,
        curve_count,
        surface_count,
    } = counts;
    validate_csr_offsets(loop_offsets, edge_uses.len(), "loop")?;
    validate_csr_offsets(face_boundary_offsets, boundaries.len(), "face boundary")?;
    if face_boundary_offsets.len() != face_surfaces.len() + 1 {
        bail!("packed B-rep face-boundary offsets disagree with face count");
    }
    let loop_count = loop_offsets.len().saturating_sub(1);
    for (index, edge) in edges.iter().enumerate() {
        let start = edge[0] as usize;
        let end = edge[1] as usize;
        let curve = (edge[2] & PACKED_INDEX_MASK_31) as usize;
        if start >= vertex_count || end >= vertex_count {
            bail!("packed B-rep edge {index} references vertex outside {vertex_count}");
        }
        if curve >= curve_count {
            bail!("packed B-rep edge {index} references curve {curve} outside {curve_count}");
        }
    }
    for (index, value) in edge_uses.iter().copied().enumerate() {
        let edge = (value & PACKED_INDEX_MASK_31) as usize;
        if edge >= edges.len() {
            bail!(
                "packed B-rep edge use {index} references edge {edge} outside {}",
                edges.len()
            );
        }
    }
    for (index, value) in face_surfaces.iter().copied().enumerate() {
        let surface = (value & PACKED_INDEX_MASK_31) as usize;
        if surface >= surface_count {
            bail!("packed B-rep face {index} references surface {surface} outside {surface_count}");
        }
    }
    for (index, value) in boundaries.iter().copied().enumerate() {
        let loop_index = (value & PACKED_INDEX_MASK_30) as usize;
        if loop_index >= loop_count {
            bail!(
                "packed B-rep boundary {index} references loop {loop_index} outside {loop_count}"
            );
        }
    }
    Ok(())
}

fn validate_csr_offsets(offsets: &[u32], payload_len: usize, what: &str) -> Result<()> {
    if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        bail!("packed B-rep {what} offsets are not monotonic");
    }
    let last = offsets.last().copied().unwrap_or(0) as usize;
    if last != payload_len {
        bail!("packed B-rep {what} offsets end at {last}, expected payload length {payload_len}");
    }
    Ok(())
}

fn validate_packed_bspline_curve(
    spline: &PackedBSplineCurve,
    vec3_count: usize,
    sequences: &PackedScalarSequences,
) -> Result<()> {
    let degree = spline.degree as usize;
    if degree == 0 || spline.control_points.is_empty() {
        bail!("has invalid degree/control-point count");
    }
    spline.control_points.validate(vec3_count)?;
    let knots = sequences.get(spline.knots_sequence)?;
    if knots.len() != spline.control_points.len() + degree + 1 {
        bail!(
            "knot count {} disagrees with {} control points and degree {}",
            knots.len(),
            spline.control_points.len(),
            degree
        );
    }
    if knots.windows(2).any(|pair| pair[0] > pair[1]) {
        bail!("knot vector is invalid");
    }
    if spline.weights_sequence != NO_SEQUENCE {
        let weights = sequences.get(spline.weights_sequence)?;
        if weights.len() != spline.control_points.len()
            || weights.iter().any(|weight| *weight <= 0.0)
        {
            bail!("weights disagree with control points or are invalid");
        }
    }
    Ok(())
}

fn validate_packed_bspline_surface(
    spline: &PackedBSplineSurface,
    vec3_count: usize,
    sequences: &PackedScalarSequences,
) -> Result<()> {
    let u_degree = spline.u_degree as usize;
    let v_degree = spline.v_degree as usize;
    let u_count = spline.u_count as usize;
    let v_count = spline.v_count as usize;
    if u_degree == 0 || v_degree == 0 || u_count == 0 || v_count == 0 {
        bail!("has invalid degrees/control-point grid");
    }
    let expected_points = u_count
        .checked_mul(v_count)
        .context("B-spline surface control grid overflows usize")?;
    if spline.control_points.len() != expected_points {
        bail!("has invalid control-point grid");
    }
    spline.control_points.validate(vec3_count)?;
    let u_knots = sequences.get(spline.u_knots_sequence)?;
    let v_knots = sequences.get(spline.v_knots_sequence)?;
    if u_knots.len() != u_count + u_degree + 1
        || v_knots.len() != v_count + v_degree + 1
        || u_knots.windows(2).any(|pair| pair[0] > pair[1])
        || v_knots.windows(2).any(|pair| pair[0] > pair[1])
    {
        bail!("has invalid knot vectors");
    }
    if spline.weights_sequence != NO_SEQUENCE {
        let weights = sequences.get(spline.weights_sequence)?;
        if weights.len() != expected_points || weights.iter().any(|weight| *weight <= 0.0) {
            bail!("has invalid weight grid");
        }
    }
    Ok(())
}

fn packed_typed_index(value: usize, max: u32, what: &str) -> Result<u32> {
    let value = as_u32(value, what)?;
    if value > max {
        bail!("{what} index {value} exceeds packed handle capacity");
    }
    Ok(value)
}

fn pack_typed_handle(kind: u32, index: u32, kind_shift: u32) -> u32 {
    debug_assert!(kind < (1u32 << (32 - kind_shift)));
    debug_assert!(index < (1u32 << kind_shift));
    (kind << kind_shift) | index
}

pub fn build_compact_brep(solid_id: u64, entities: &[EntityInstance]) -> Result<CompactBrep> {
    let index = build_index(entities);
    let face_ids = brep::solid_face_ids(solid_id, entities, &index)
        .with_context(|| format!("solid #{solid_id} has no readable closed-shell face set"))?;
    build_compact_brep_faces_with_index(solid_id, &face_ids, entities, &index)
}

pub fn build_compact_brep_faces(
    solid_id: u64,
    face_ids: &[u64],
    entities: &[EntityInstance],
) -> Result<CompactBrep> {
    let index = build_index(entities);
    build_compact_brep_faces_with_index(solid_id, face_ids, entities, &index)
}

fn build_compact_brep_faces_with_index(
    solid_id: u64,
    face_ids: &[u64],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Result<CompactBrep> {
    if face_ids.is_empty() {
        bail!("compact B-rep face subset must not be empty");
    }
    let mut seen = HashSet::new();
    let mut builder = Builder::new(solid_id, entities, index);
    for &face_id in face_ids {
        if !seen.insert(face_id) {
            bail!("compact B-rep face subset contains duplicate face #{face_id}");
        }
        builder.add_face(face_id)?;
    }
    Ok(builder.finish())
}

struct Builder<'a> {
    solid_id: u64,
    entities: &'a [EntityInstance],
    index: &'a HashMap<u64, usize>,
    vertices: Vec<CompactVertex>,
    vertex_by_source: HashMap<u64, u32>,
    curves: Vec<CompactCurve>,
    curve_by_key: HashMap<String, u32>,
    edges: Vec<CompactEdge>,
    edge_by_source: HashMap<u64, u32>,
    loops: Vec<CompactLoop>,
    loop_by_source: HashMap<u64, u32>,
    surfaces: Vec<CompactSurface>,
    surface_by_key: HashMap<String, u32>,
    faces: Vec<CompactFace>,
}

impl<'a> Builder<'a> {
    fn new(solid_id: u64, entities: &'a [EntityInstance], index: &'a HashMap<u64, usize>) -> Self {
        Self {
            solid_id,
            entities,
            index,
            vertices: Vec::new(),
            vertex_by_source: HashMap::new(),
            curves: Vec::new(),
            curve_by_key: HashMap::new(),
            edges: Vec::new(),
            edge_by_source: HashMap::new(),
            loops: Vec::new(),
            loop_by_source: HashMap::new(),
            surfaces: Vec::new(),
            surface_by_key: HashMap::new(),
            faces: Vec::new(),
        }
    }

    fn finish(self) -> CompactBrep {
        CompactBrep {
            source_solid_id: self.solid_id,
            vertices: self.vertices,
            curves: self.curves,
            edges: self.edges,
            loops: self.loops,
            surfaces: self.surfaces,
            faces: self.faces,
        }
    }

    fn add_face(&mut self, face_id: u64) -> Result<()> {
        let surface_id = brep::face_surface(face_id, self.entities, self.index)
            .with_context(|| format!("face #{face_id} has no support surface"))?;
        let surface = self.intern_surface(surface_id);
        let same_sense = match brep::face_sense(face_id, self.entities, self.index).as_deref() {
            Some("T") => true,
            Some("F") => false,
            other => bail!("face #{face_id} has invalid same_sense {other:?}"),
        };
        let loops = brep::face_loops(face_id, self.entities, self.index)
            .with_context(|| format!("face #{face_id} has unreadable bounds"))?;
        let boundaries = loops
            .into_iter()
            .map(|loop_| {
                let loop_index = self.intern_loop(&loop_)?;
                Ok(CompactBoundary {
                    loop_index,
                    outer: loop_.outer,
                    orientation: loop_.orientation,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.faces.push(CompactFace {
            source_face_id: face_id,
            surface,
            same_sense,
            boundaries,
        });
        Ok(())
    }

    fn intern_loop(&mut self, loop_: &FaceLoop) -> Result<u32> {
        if let Some(&existing) = self.loop_by_source.get(&loop_.loop_id) {
            return Ok(existing);
        }
        let mut uses = Vec::with_capacity(loop_.edges.len());
        for edge_use in &loop_.edges {
            let edge = self.intern_edge(edge_use)?;
            let forward = edge_use.parameter_forward == edge_use.curve_same_sense;
            uses.push(CompactEdgeUse { edge, forward });
        }
        let index = as_u32(self.loops.len(), "loop")?;
        self.loops.push(CompactLoop {
            source_loop_id: loop_.loop_id,
            edges: uses,
        });
        self.loop_by_source.insert(loop_.loop_id, index);
        Ok(index)
    }

    fn intern_edge(&mut self, edge_use: &brep::OrientedEdgeUse) -> Result<u32> {
        if let Some(&existing) = self.edge_by_source.get(&edge_use.edge_id) {
            return Ok(existing);
        }
        let forward = edge_use.parameter_forward == edge_use.curve_same_sense;
        let (raw_start, raw_end) = if forward {
            (edge_use.start_vertex, edge_use.end_vertex)
        } else {
            (edge_use.end_vertex, edge_use.start_vertex)
        };
        let start_vertex = self.intern_vertex(raw_start)?;
        let end_vertex = self.intern_vertex(raw_end)?;
        let curve = self.intern_curve(edge_use.curve_id, &edge_use.support);
        let index = as_u32(self.edges.len(), "edge")?;
        self.edges.push(CompactEdge {
            source_edge_id: edge_use.edge_id,
            start_vertex,
            end_vertex,
            curve,
            curve_same_sense: edge_use.curve_same_sense,
        });
        self.edge_by_source.insert(edge_use.edge_id, index);
        Ok(index)
    }

    fn intern_vertex(&mut self, source_vertex_id: u64) -> Result<u32> {
        if let Some(&existing) = self.vertex_by_source.get(&source_vertex_id) {
            return Ok(existing);
        }
        let point_mm = brep::vertex_point(source_vertex_id, self.entities, self.index)
            .with_context(|| format!("vertex #{source_vertex_id} has no Cartesian point"))?;
        let index = as_u32(self.vertices.len(), "vertex")?;
        self.vertices.push(CompactVertex {
            source_vertex_id,
            point_mm,
        });
        self.vertex_by_source.insert(source_vertex_id, index);
        Ok(index)
    }

    fn intern_curve(&mut self, source_entity_id: u64, support: &CurveSupport) -> u32 {
        let curve = if matches!(support, CurveSupport::Other { .. }) {
            brep::exact_bspline_curve_support(source_entity_id, self.entities, self.index)
                .map(|spline| CompactCurve::BSpline {
                    source_entity_id,
                    spline,
                })
                .unwrap_or_else(|| compact_curve(source_entity_id, support))
        } else {
            compact_curve(source_entity_id, support)
        };
        let key = curve_key(&curve);
        if let Some(&existing) = self.curve_by_key.get(&key) {
            return existing;
        }
        let index = self.curves.len() as u32;
        self.curves.push(curve);
        self.curve_by_key.insert(key, index);
        index
    }

    fn intern_surface(&mut self, source_entity_id: u64) -> u32 {
        let surface = if let Some(spline) =
            surface_recovery::bspline_surface_support(source_entity_id, self.entities, self.index)
        {
            CompactSurface::BSpline {
                source_entity_id,
                spline,
            }
        } else {
            let support = brep::surface_support(source_entity_id, self.entities, self.index);
            let source_kind = self
                .index
                .get(&source_entity_id)
                .and_then(|&idx| simple_record(&self.entities[idx]))
                .map(|record| record.name.as_str());
            compact_surface(source_entity_id, source_kind, &support)
        };
        let key = surface_key(&surface);
        if let Some(&existing) = self.surface_by_key.get(&key) {
            return existing;
        }
        let index = self.surfaces.len() as u32;
        self.surfaces.push(surface);
        self.surface_by_key.insert(key, index);
        index
    }
}

fn compact_curve(source_entity_id: u64, support: &CurveSupport) -> CompactCurve {
    match support {
        CurveSupport::Line(line) => CompactCurve::Line {
            origin_mm: clean_vec(line.origin_mm),
            direction: clean_vec(line.direction),
        },
        CurveSupport::Circle(circle) => CompactCurve::Circle {
            center_mm: clean_vec(circle.center_mm),
            normal: clean_vec(circle.normal),
            x_direction: clean_vec(circle.x_direction),
            radius_mm: clean_zero(circle.radius_mm),
        },
        CurveSupport::BSpline(spline) => CompactCurve::BSpline {
            source_entity_id,
            spline: spline.clone(),
        },
        CurveSupport::Other { .. } => CompactCurve::Source { source_entity_id },
    }
}

fn compact_surface(
    source_entity_id: u64,
    source_kind: Option<&str>,
    support: &SurfaceSupport,
) -> CompactSurface {
    match (source_kind, support) {
        (Some("PLANE"), SurfaceSupport::Plane(plane)) => CompactSurface::Plane {
            origin_mm: clean_vec(plane.origin_mm),
            normal: clean_vec(plane.normal),
        },
        (Some("CYLINDRICAL_SURFACE"), SurfaceSupport::Cylinder(cylinder)) => {
            CompactSurface::Cylinder {
                axis_origin_mm: clean_vec(cylinder.axis_origin_mm),
                axis: clean_vec(cylinder.axis),
                x_direction: clean_vec(cylinder.x_direction),
                radius_mm: clean_zero(cylinder.radius_mm),
            }
        }
        (Some("CONICAL_SURFACE"), SurfaceSupport::Cone(cone)) => CompactSurface::Cone {
            reference_origin_mm: clean_vec(cone.reference_origin_mm),
            axis: clean_vec(cone.axis),
            x_direction: clean_vec(cone.x_direction),
            reference_radius_mm: clean_zero(cone.reference_radius_mm),
            semi_angle_rad: clean_zero(cone.semi_angle_rad),
        },
        (Some("SPHERICAL_SURFACE"), SurfaceSupport::Sphere(sphere)) => CompactSurface::Sphere {
            center_mm: clean_vec(sphere.center_mm),
            axis: clean_vec(sphere.axis),
            x_direction: clean_vec(sphere.x_direction),
            radius_mm: clean_zero(sphere.radius_mm),
        },
        (Some("TOROIDAL_SURFACE"), SurfaceSupport::Torus(torus)) => CompactSurface::Torus {
            center_mm: clean_vec(torus.center_mm),
            axis: clean_vec(torus.axis),
            x_direction: clean_vec(torus.x_direction),
            major_radius_mm: clean_zero(torus.major_radius_mm),
            minor_radius_mm: clean_zero(torus.minor_radius_mm),
        },
        (Some("SURFACE_OF_REVOLUTION"), SurfaceSupport::Revolution(_)) => {
            CompactSurface::Revolution { source_entity_id }
        }
        _ => CompactSurface::Source { source_entity_id },
    }
}

fn curve_key(curve: &CompactCurve) -> String {
    match curve {
        CompactCurve::Line {
            origin_mm,
            direction,
        } => format!("L:{}:{}", vec_key(*origin_mm), vec_key(*direction)),
        CompactCurve::Circle {
            center_mm,
            normal,
            x_direction,
            radius_mm,
        } => format!(
            "C:{}:{}:{}:{:016x}",
            vec_key(*center_mm),
            vec_key(*normal),
            vec_key(*x_direction),
            fbits(*radius_mm)
        ),
        CompactCurve::BSpline { spline, .. } => bspline_curve_key(spline),
        CompactCurve::Source { source_entity_id } => format!("SRC:{source_entity_id}"),
    }
}

fn surface_key(surface: &CompactSurface) -> String {
    match surface {
        CompactSurface::Plane { origin_mm, normal } => {
            format!("P:{}:{}", vec_key(*origin_mm), vec_key(*normal))
        }
        CompactSurface::Cylinder {
            axis_origin_mm,
            axis,
            x_direction,
            radius_mm,
        } => format!(
            "CY:{}:{}:{}:{:016x}",
            vec_key(*axis_origin_mm),
            vec_key(*axis),
            vec_key(*x_direction),
            fbits(*radius_mm)
        ),
        CompactSurface::Cone {
            reference_origin_mm,
            axis,
            x_direction,
            reference_radius_mm,
            semi_angle_rad,
        } => format!(
            "CO:{}:{}:{}:{:016x}:{:016x}",
            vec_key(*reference_origin_mm),
            vec_key(*axis),
            vec_key(*x_direction),
            fbits(*reference_radius_mm),
            fbits(*semi_angle_rad)
        ),
        CompactSurface::BSpline { spline, .. } => bspline_surface_key(spline),
        CompactSurface::Sphere {
            center_mm,
            axis,
            x_direction,
            radius_mm,
        } => format!(
            "S:{}:{}:{}:{:016x}",
            vec_key(*center_mm),
            vec_key(*axis),
            vec_key(*x_direction),
            fbits(*radius_mm)
        ),
        CompactSurface::Torus {
            center_mm,
            axis,
            x_direction,
            major_radius_mm,
            minor_radius_mm,
        } => format!(
            "T:{}:{}:{}:{:016x}:{:016x}",
            vec_key(*center_mm),
            vec_key(*axis),
            vec_key(*x_direction),
            fbits(*major_radius_mm),
            fbits(*minor_radius_mm)
        ),
        CompactSurface::Revolution { source_entity_id } => format!("R:{source_entity_id}"),
        CompactSurface::Source { source_entity_id } => format!("SRC:{source_entity_id}"),
    }
}

fn push_bits(out: &mut String, value: f64) {
    use std::fmt::Write as _;
    let _ = write!(out, "{:016x}", fbits(value));
}

fn bspline_curve_key(spline: &BSplineSupport) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(
        32 + spline.control_points_mm.len() * 49
            + spline.knots.len() * 17
            + spline
                .weights
                .as_ref()
                .map_or(0, |weights| weights.len() * 17),
    );
    let _ = write!(
        out,
        "BS:d{}:cp{}:",
        spline.degree,
        spline.control_points_mm.len()
    );
    for point in &spline.control_points_mm {
        for &value in point {
            push_bits(&mut out, value);
        }
        out.push(';');
    }
    let _ = write!(out, ":k{}:", spline.knots.len());
    for &value in &spline.knots {
        push_bits(&mut out, value);
        out.push(',');
    }
    match &spline.weights {
        Some(weights) => {
            let _ = write!(out, ":w{}:", weights.len());
            for &value in weights {
                push_bits(&mut out, value);
                out.push(',');
            }
        }
        None => out.push_str(":w-"),
    }
    out
}

fn bspline_surface_key(spline: &BSplineSurfaceSupport) -> String {
    use std::fmt::Write as _;
    let point_count = spline.control_points_mm.iter().map(Vec::len).sum::<usize>();
    let weight_count = spline
        .weights
        .as_ref()
        .map_or(0, |rows| rows.iter().map(Vec::len).sum::<usize>());
    let mut out = String::with_capacity(
        48 + point_count * 49 + (spline.u_knots.len() + spline.v_knots.len() + weight_count) * 17,
    );
    let columns = spline.control_points_mm.first().map_or(0, Vec::len);
    let _ = write!(
        out,
        "BSS:du{}:dv{}:r{}:c{}:",
        spline.u_degree,
        spline.v_degree,
        spline.control_points_mm.len(),
        columns
    );
    for row in &spline.control_points_mm {
        let _ = write!(out, "[{}:", row.len());
        for point in row {
            for &value in point {
                push_bits(&mut out, value);
            }
            out.push(';');
        }
        out.push(']');
    }
    let _ = write!(out, ":uk{}:", spline.u_knots.len());
    for &value in &spline.u_knots {
        push_bits(&mut out, value);
        out.push(',');
    }
    let _ = write!(out, ":vk{}:", spline.v_knots.len());
    for &value in &spline.v_knots {
        push_bits(&mut out, value);
        out.push(',');
    }
    match &spline.weights {
        Some(rows) => {
            let _ = write!(out, ":wr{}:", rows.len());
            for row in rows {
                let _ = write!(out, "[{}:", row.len());
                for &value in row {
                    push_bits(&mut out, value);
                    out.push(',');
                }
                out.push(']');
            }
        }
        None => out.push_str(":w-"),
    }
    out
}

fn as_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("too many {what}s for compact B-rep: {value}"))
}

fn clean_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn clean_vec(v: [f64; 3]) -> [f64; 3] {
    [clean_zero(v[0]), clean_zero(v[1]), clean_zero(v[2])]
}

fn fbits(value: f64) -> u64 {
    clean_zero(value).to_bits()
}

fn vec_key(v: [f64; 3]) -> String {
    format!(
        "{:016x},{:016x},{:016x}",
        fbits(v[0]),
        fbits(v[1]),
        fbits(v[2])
    )
}

pub fn closure_entity_count(solid_id: u64, entities: &[EntityInstance]) -> usize {
    let index = build_index(entities);
    crate::step_entities::closure_from(solid_id, entities, &index).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_line_supports_intern_when_parameter_frames_match() {
        let a = compact_curve(
            1,
            &CurveSupport::Line(brep::LineSupport {
                origin_mm: [10.0, 2.0, 3.0],
                direction: [1.0, 0.0, 0.0],
            }),
        );
        let b = compact_curve(
            2,
            &CurveSupport::Line(brep::LineSupport {
                origin_mm: [10.0, 2.0, 3.0],
                direction: [1.0, 0.0, 0.0],
            }),
        );
        assert_eq!(curve_key(&a), curve_key(&b));
    }

    #[test]
    fn line_direction_sign_is_not_discarded_without_orientation_parity() {
        let a = compact_curve(
            1,
            &CurveSupport::Line(brep::LineSupport {
                origin_mm: [10.0, 2.0, 3.0],
                direction: [1.0, 0.0, 0.0],
            }),
        );
        let b = compact_curve(
            2,
            &CurveSupport::Line(brep::LineSupport {
                origin_mm: [10.0, 2.0, 3.0],
                direction: [-1.0, 0.0, 0.0],
            }),
        );
        assert_ne!(curve_key(&a), curve_key(&b));
    }
    #[test]
    fn identical_bspline_curve_payloads_ignore_source_entity_id() {
        let spline = BSplineSupport {
            degree: 2,
            control_points_mm: vec![[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0]],
            knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            weights: Some(vec![1.0, 0.5, 1.0]),
        };
        let a = CompactCurve::BSpline {
            source_entity_id: 10,
            spline: spline.clone(),
        };
        let b = CompactCurve::BSpline {
            source_entity_id: 999,
            spline: spline.clone(),
        };
        assert_eq!(curve_key(&a), curve_key(&b));

        let mut changed = spline;
        changed.control_points_mm[1][1] =
            f64::from_bits(changed.control_points_mm[1][1].to_bits() + 1);
        let c = CompactCurve::BSpline {
            source_entity_id: 10,
            spline: changed,
        };
        assert_ne!(curve_key(&a), curve_key(&c));
    }

    #[test]
    fn identical_bspline_surface_payloads_ignore_source_entity_id() {
        let spline = BSplineSurfaceSupport {
            u_degree: 1,
            v_degree: 1,
            control_points_mm: vec![
                vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                vec![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            ],
            u_knots: vec![0.0, 0.0, 1.0, 1.0],
            v_knots: vec![0.0, 0.0, 1.0, 1.0],
            weights: Some(vec![vec![1.0, 1.0], vec![1.0, 1.0]]),
        };
        let a = CompactSurface::BSpline {
            source_entity_id: 20,
            spline: spline.clone(),
        };
        let b = CompactSurface::BSpline {
            source_entity_id: 777,
            spline: spline.clone(),
        };
        assert_eq!(surface_key(&a), surface_key(&b));

        let mut changed = spline;
        changed.weights.as_mut().unwrap()[1][1] =
            f64::from_bits(changed.weights.as_ref().unwrap()[1][1].to_bits() + 1);
        let c = CompactSurface::BSpline {
            source_entity_id: 20,
            spline: changed,
        };
        assert_ne!(surface_key(&a), surface_key(&c));
    }

    #[test]
    fn scalar_sequence_pool_interns_exact_payloads() {
        let mut pool = ScalarSequencePool::new();
        let a = pool.intern(&[0.0, 0.5, 1.0]).unwrap();
        let b = pool.intern(&[0.0, 0.5, 1.0]).unwrap();
        let changed = f64::from_bits(0.5f64.to_bits() + 1);
        let c = pool.intern(&[0.0, changed, 1.0]).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);

        let packed = pool.finish();
        packed.validate().unwrap();
        assert_eq!(packed.get(a).unwrap(), &[0.0, 0.5, 1.0]);
        assert_eq!(packed.get(c).unwrap()[1].to_bits(), changed.to_bits());
    }

    #[test]
    fn u16_topology_remaps_flag_bits_without_changing_indices() {
        let topology = PackedTopology::from_u32(
            vec![[1, 2, 3 | (1u32 << 31)]],
            vec![4 | (1u32 << 31)],
            vec![0, 1],
            vec![5 | (1u32 << 31)],
            vec![0, 1],
            vec![6 | (1u32 << 30) | (1u32 << 31)],
        );
        match topology {
            PackedTopology::U16 {
                edges,
                edge_uses,
                face_surfaces,
                boundaries,
                ..
            } => {
                assert_eq!(edges[0], [1, 2, 3 | (1u16 << 15)]);
                assert_eq!(edge_uses[0], 4 | (1u16 << 15));
                assert_eq!(face_surfaces[0], 5 | (1u16 << 15));
                assert_eq!(boundaries[0], 6 | (1u16 << 14) | (1u16 << 15));
            }
            PackedTopology::U32 { .. } => panic!("small topology should use u16 storage"),
        }
    }

    #[test]
    fn provenance_uses_u32_until_an_entity_id_requires_u64() {
        let compact = PackedBrepProvenance::from_u64(vec![1], vec![2], vec![3], vec![4]);
        assert!(matches!(compact, PackedBrepProvenance::U32 { .. }));
        assert_eq!(compact.payload_bytes(), 4 * std::mem::size_of::<u32>());

        let wide =
            PackedBrepProvenance::from_u64(vec![u32::MAX as u64 + 1], vec![], vec![], vec![]);
        assert!(matches!(wide, PackedBrepProvenance::U64 { .. }));
        assert_eq!(wide.payload_bytes(), std::mem::size_of::<u64>());
    }

    #[test]
    fn square_face_compacts_to_indexed_topology() {
        let text = "ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('x'),'1');
FILE_NAME('a','b',(''),(''),'x','y','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=CARTESIAN_POINT('',(1.,0.,0.));
#3=CARTESIAN_POINT('',(1.,1.,0.));
#4=CARTESIAN_POINT('',(0.,1.,0.));
#5=VERTEX_POINT('',#1);
#6=VERTEX_POINT('',#2);
#7=VERTEX_POINT('',#3);
#8=VERTEX_POINT('',#4);
#9=DIRECTION('',(1.,0.,0.));
#10=DIRECTION('',(0.,1.,0.));
#11=DIRECTION('',(-1.,0.,0.));
#12=DIRECTION('',(0.,-1.,0.));
#13=DIRECTION('',(0.,0.,1.));
#14=VECTOR('',#9,1.);
#15=VECTOR('',#10,1.);
#16=VECTOR('',#11,1.);
#17=VECTOR('',#12,1.);
#18=LINE('',#1,#14);
#19=LINE('',#2,#15);
#20=LINE('',#3,#16);
#21=LINE('',#4,#17);
#22=EDGE_CURVE('',#5,#6,#18,.T.);
#23=EDGE_CURVE('',#6,#7,#19,.T.);
#24=EDGE_CURVE('',#7,#8,#20,.T.);
#25=EDGE_CURVE('',#8,#5,#21,.T.);
#26=ORIENTED_EDGE('',*,*,#22,.T.);
#27=ORIENTED_EDGE('',*,*,#23,.T.);
#28=ORIENTED_EDGE('',*,*,#24,.T.);
#29=ORIENTED_EDGE('',*,*,#25,.T.);
#30=EDGE_LOOP('',(#26,#27,#28,#29));
#31=FACE_OUTER_BOUND('',#30,.T.);
#32=AXIS2_PLACEMENT_3D('',#1,#13,#9);
#33=PLANE('',#32);
#34=ADVANCED_FACE('',(#31),#33,.T.);
#35=CLOSED_SHELL('',(#34));
#36=MANIFOLD_SOLID_BREP('',#35);
ENDSEC;
END-ISO-10303-21;
";
        let exchange = ruststep::parser::parse(text).unwrap();
        let compact = build_compact_brep(36, &exchange.data[0].entities).unwrap();
        let stats = compact.stats(closure_entity_count(36, &exchange.data[0].entities));
        assert_eq!(stats.vertices, 4);

        let subset = build_compact_brep_faces(36, &[34], &exchange.data[0].entities).unwrap();
        assert_eq!(subset.faces.len(), 1);
        assert_eq!(subset.edges.len(), 4);
        assert!(build_compact_brep_faces(36, &[], &exchange.data[0].entities).is_err());
        assert!(build_compact_brep_faces(36, &[34, 34], &exchange.data[0].entities).is_err());
        assert_eq!(stats.edges, 4);
        assert_eq!(stats.edge_uses, 4);
        assert_eq!(stats.loops, 1);
        assert_eq!(stats.faces, 1);
        assert_eq!(stats.boundaries, 1);
        assert_eq!(compact.faces[0].boundaries[0].loop_index, 0);
        assert!(compact.loops[0].edges.iter().all(|edge| edge.forward));

        let mut packed = compact.into_packed().unwrap();
        packed.validate().unwrap();
        match &mut packed.topology {
            PackedTopology::U16 { edges, .. } => {
                let sense = edges[0][2] & 0x8000;
                edges[0][2] = sense | u16::try_from(packed.curve_handles.len()).unwrap();
            }
            PackedTopology::U32 { edges, .. } => {
                let sense = edges[0][2] & (1u32 << 31);
                edges[0][2] = sense | packed.curve_handles.len() as u32;
            }
        }
        assert!(packed.validate().is_err());
    }
}
