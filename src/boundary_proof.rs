use crate::brep::{self, BSplineSupport, CurveSupport, SplineExtrusionSupport, SurfaceSupport};
use crate::math3::{add, canonical_direction, cross, distance, dot, mul, norm, normalize, sub};
use crate::step_graph::{build_index, entity_ref_value, simple_record};
use num_traits::ToPrimitive;
use ruststep::ast::{EntityInstance, Parameter};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};

const POINT_Q_MM: f64 = 1.0e-7;
const DIRECTION_Q: f64 = 1.0e-10;
const SCALAR_Q_MM: f64 = 1.0e-7;
const BINOMIAL_2: [f64; 3] = [1.0, 2.0, 1.0];
const BINOMIAL_4: [f64; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
pub(crate) const ANALYTIC_BOUNDARY_PROOF_TOL_MM: f64 = 4.0 * SCALAR_Q_MM;

type PointKey = [i64; 3];
type GroupedBoundarySegments = BTreeMap<[OrientedSupportKey; 2], Vec<(PointKey, PointKey)>>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OrientedSupportKey {
    Plane {
        normal: [i64; 3],
        offset_mm: i64,
        outward_positive: bool,
    },
    Cylinder {
        axis: [i64; 3],
        closest_mm: [i64; 3],
        radius_mm: i64,
        radial_outward: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum SupportLocusKey {
    Plane {
        normal: [i64; 3],
        offset_mm: i64,
    },
    Cylinder {
        axis: [i64; 3],
        closest_mm: [i64; 3],
        radius_mm: i64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BoundaryChainKey {
    Open {
        supports: [OrientedSupportKey; 2],
        endpoints_mm: [PointKey; 2],
    },
    Cycle {
        supports: [OrientedSupportKey; 2],
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AnalyticBoundarySignature {
    /// One entry per connected same-support patch after face-partition collapse.
    pub supports: Vec<OrientedSupportKey>,
    /// Physical boundaries between distinct oriented analytic support patches.
    pub chains: Vec<BoundaryChainKey>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalyticBoundaryDiagnostic {
    pub solid_id: u64,
    pub supported: bool,
    pub reason: Option<String>,
    pub signature: Option<AnalyticBoundarySignature>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalyticBoundaryComparison {
    pub equivalent: bool,
    pub first: AnalyticBoundaryDiagnostic,
    pub second: AnalyticBoundaryDiagnostic,
}

#[derive(Debug, Clone)]
struct FaceInfo {
    support: OrientedSupportKey,
}

#[derive(Debug, Clone)]
struct EdgeGeometry {
    start: PointKey,
    end: PointKey,
    curve_id: u64,
    support: CurveSupport,
}

#[derive(Debug)]
struct BoundaryGraph {
    faces: Vec<FaceInfo>,
    edge_faces: HashMap<u64, Vec<usize>>,
    edge_geometry: HashMap<u64, EdgeGeometry>,
}

#[derive(Debug)]
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
        }
    }

    fn find(&mut self, index: usize) -> usize {
        let mut root = index;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut cursor = index;
        while self.parent[cursor] != root {
            let next = self.parent[cursor];
            self.parent[cursor] = root;
            cursor = next;
        }
        root
    }

    fn union(&mut self, first: usize, second: usize) {
        let first = self.find(first);
        let second = self.find(second);
        if first == second {
            return;
        }
        let (keep, replace) = if first < second {
            (first, second)
        } else {
            (second, first)
        };
        self.parent[replace] = keep;
    }
}

#[must_use]
pub fn compare_analytic_boundaries(
    first_solid_id: u64,
    first_entities: &[EntityInstance],
    second_solid_id: u64,
    second_entities: &[EntityInstance],
) -> AnalyticBoundaryComparison {
    let first = analytic_boundary_diagnostic(first_solid_id, first_entities);
    let second = analytic_boundary_diagnostic(second_solid_id, second_entities);
    let equivalent = first.supported
        && second.supported
        && first.signature.is_some()
        && first.signature == second.signature;
    AnalyticBoundaryComparison {
        equivalent,
        first,
        second,
    }
}

#[must_use]
pub fn analytic_boundary_diagnostic(
    solid_id: u64,
    entities: &[EntityInstance],
) -> AnalyticBoundaryDiagnostic {
    match analytic_boundary_signature(solid_id, entities) {
        Ok(signature) => AnalyticBoundaryDiagnostic {
            solid_id,
            supported: true,
            reason: None,
            signature: Some(signature),
        },
        Err(reason) => AnalyticBoundaryDiagnostic {
            solid_id,
            supported: false,
            reason: Some(reason),
            signature: None,
        },
    }
}

fn analytic_boundary_signature(
    solid_id: u64,
    entities: &[EntityInstance],
) -> Result<AnalyticBoundarySignature, String> {
    let index = build_index(entities);
    let face_ids = brep::solid_face_ids(solid_id, entities, &index)
        .ok_or_else(|| "solid has no supported manifold face list".to_owned())?;
    if face_ids.is_empty() {
        return Err("solid has no faces".to_owned());
    }

    let graph = collect_boundary_graph(&face_ids, entities, &index)?;
    let (mut union_find, component_support) = merge_support_components(&graph)?;
    let grouped_segments = group_physical_boundaries(
        &graph,
        &mut union_find,
        &component_support,
        entities,
        &index,
    )?;

    let mut chains = Vec::new();
    for (supports, segments) in grouped_segments {
        chains.extend(normalize_support_pair_chains(&supports, &segments)?);
    }
    chains.sort();

    let mut supports = component_support.into_values().collect::<Vec<_>>();
    supports.sort();
    prove_all_supports_are_bounded(&supports, &chains)?;

    Ok(AnalyticBoundarySignature { supports, chains })
}

fn collect_boundary_graph(
    face_ids: &[u64],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Result<BoundaryGraph, String> {
    let mut faces = Vec::with_capacity(face_ids.len());
    let mut edge_faces = HashMap::<u64, Vec<usize>>::new();
    let mut edge_geometry = HashMap::<u64, EdgeGeometry>::new();

    for (face_index, face_id) in face_ids.iter().copied().enumerate() {
        let support = oriented_support_key(face_id, entities, index)
            .ok_or_else(|| format!("face #{face_id} is not a supported plane/cylinder"))?;
        let loops = brep::face_loops(face_id, entities, index)
            .ok_or_else(|| format!("face #{face_id} has unsupported boundary topology"))?;
        if loops.is_empty() {
            return Err(format!("face #{face_id} has no boundary loops"));
        }

        for edge in loops.iter().flat_map(|loop_| &loop_.edges) {
            edge_faces.entry(edge.edge_id).or_default().push(face_index);
            let geometry = EdgeGeometry {
                start: point_key(edge.start_mm)
                    .ok_or_else(|| format!("edge #{} has non-finite start", edge.edge_id))?,
                end: point_key(edge.end_mm)
                    .ok_or_else(|| format!("edge #{} has non-finite end", edge.edge_id))?,
                curve_id: edge.curve_id,
                support: edge.support.clone(),
            };
            validate_or_insert_edge_geometry(edge.edge_id, geometry, &mut edge_geometry)?;
        }
        faces.push(FaceInfo { support });
    }

    Ok(BoundaryGraph {
        faces,
        edge_faces,
        edge_geometry,
    })
}

fn validate_or_insert_edge_geometry(
    edge_id: u64,
    geometry: EdgeGeometry,
    edge_geometry: &mut HashMap<u64, EdgeGeometry>,
) -> Result<(), String> {
    let Some(existing) = edge_geometry.get(&edge_id) else {
        edge_geometry.insert(edge_id, geometry);
        return Ok(());
    };
    let same = existing.start == geometry.start && existing.end == geometry.end;
    let reversed = existing.start == geometry.end && existing.end == geometry.start;
    if !same && !reversed {
        return Err(format!(
            "edge #{edge_id} has inconsistent endpoint geometry"
        ));
    }
    if existing.curve_id != geometry.curve_id || existing.support != geometry.support {
        return Err(format!("edge #{edge_id} has inconsistent curve geometry"));
    }
    Ok(())
}

fn merge_support_components(
    graph: &BoundaryGraph,
) -> Result<(UnionFind, HashMap<usize, OrientedSupportKey>), String> {
    let mut union_find = UnionFind::new(graph.faces.len());
    for attached in graph.edge_faces.values() {
        let unique = attached.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() > 2 {
            return Err("non-manifold edge has more than two attached faces".to_owned());
        }
        if let [first, second] = unique.iter().copied().collect::<Vec<_>>().as_slice()
            && graph.faces[*first].support == graph.faces[*second].support
        {
            union_find.union(*first, *second);
        }
    }

    let mut component_support = HashMap::<usize, OrientedSupportKey>::new();
    for (face_index, face) in graph.faces.iter().enumerate() {
        let root = union_find.find(face_index);
        if let Some(existing) = component_support.get(&root) {
            if existing != &face.support {
                return Err("same-support merge crossed an oriented support".to_owned());
            }
        } else {
            component_support.insert(root, face.support.clone());
        }
    }
    Ok((union_find, component_support))
}

fn group_physical_boundaries(
    graph: &BoundaryGraph,
    union_find: &mut UnionFind,
    component_support: &HashMap<usize, OrientedSupportKey>,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Result<GroupedBoundarySegments, String> {
    let mut grouped_segments = BTreeMap::new();
    for (edge_id, attached) in &graph.edge_faces {
        let unique_faces = attached.iter().copied().collect::<BTreeSet<_>>();
        let roots = unique_faces
            .iter()
            .copied()
            .map(|face_index| union_find.find(face_index))
            .collect::<BTreeSet<_>>();
        if roots.len() == 1 {
            if attached.len() >= 2 {
                continue;
            }
            return Err(format!("edge #{edge_id} is open"));
        }
        if roots.len() != 2 || unique_faces.len() != 2 {
            return Err(format!("edge #{edge_id} has unsupported incidence"));
        }

        let roots = roots.into_iter().collect::<Vec<_>>();
        let first = component_support
            .get(&roots[0])
            .ok_or_else(|| "missing first component support".to_owned())?
            .clone();
        let second = component_support
            .get(&roots[1])
            .ok_or_else(|| "missing second component support".to_owned())?
            .clone();
        let supports = sorted_pair(first, second);
        let geometry = graph
            .edge_geometry
            .get(edge_id)
            .ok_or_else(|| format!("edge #{edge_id} has no geometry"))?;
        if geometry.start == geometry.end {
            return Err(format!("edge #{edge_id} collapses after quantization"));
        }
        if !edge_proves_support_intersection(geometry, &supports, entities, index) {
            return Err(format!(
                "edge #{edge_id} is not proven on the exact intersection locus of its carriers"
            ));
        }
        grouped_segments
            .entry(supports)
            .or_insert_with(Vec::new)
            .push((geometry.start, geometry.end));
    }
    Ok(grouped_segments)
}

fn prove_all_supports_are_bounded(
    supports: &[OrientedSupportKey],
    chains: &[BoundaryChainKey],
) -> Result<(), String> {
    let used_supports = chains
        .iter()
        .flat_map(|chain| match chain {
            BoundaryChainKey::Open { supports, .. } | BoundaryChainKey::Cycle { supports } => {
                supports.iter()
            }
        })
        .cloned()
        .collect::<BTreeSet<_>>();
    if supports
        .iter()
        .any(|support| !used_supports.contains(support))
    {
        return Err("analytic support component has no physical boundary".to_owned());
    }
    Ok(())
}

fn normalize_support_pair_chains(
    supports: &[OrientedSupportKey; 2],
    segments: &[(PointKey, PointKey)],
) -> Result<Vec<BoundaryChainKey>, String> {
    if segments.is_empty() {
        return Ok(Vec::new());
    }

    let mut adjacency = BTreeMap::<PointKey, Vec<usize>>::new();
    for (segment_index, &(start, end)) in segments.iter().enumerate() {
        adjacency.entry(start).or_default().push(segment_index);
        adjacency.entry(end).or_default().push(segment_index);
    }
    if adjacency.values().any(|edges| edges.len() > 2) {
        return Err("support-pair boundary graph branches".to_owned());
    }

    let mut visited = vec![false; segments.len()];
    let mut out = Vec::new();
    for seed in 0..segments.len() {
        if visited[seed] {
            continue;
        }
        let mut stack = vec![seed];
        let mut component_vertices = BTreeSet::new();
        while let Some(segment_index) = stack.pop() {
            if visited[segment_index] {
                continue;
            }
            visited[segment_index] = true;
            let (start, end) = segments[segment_index];
            component_vertices.insert(start);
            component_vertices.insert(end);
            for vertex in [start, end] {
                if let Some(neighbors) = adjacency.get(&vertex) {
                    for &neighbor in neighbors {
                        if !visited[neighbor] {
                            stack.push(neighbor);
                        }
                    }
                }
            }
        }

        let endpoints = component_vertices
            .iter()
            .copied()
            .filter(|vertex| adjacency.get(vertex).is_some_and(|edges| edges.len() == 1))
            .collect::<Vec<_>>();
        let all_degree_two = component_vertices
            .iter()
            .all(|vertex| adjacency.get(vertex).is_some_and(|edges| edges.len() == 2));

        if endpoints.len() == 2 {
            let mut endpoints_mm = [endpoints[0], endpoints[1]];
            endpoints_mm.sort_unstable();
            out.push(BoundaryChainKey::Open {
                supports: supports.to_owned(),
                endpoints_mm,
            });
        } else if endpoints.is_empty() && all_degree_two {
            out.push(BoundaryChainKey::Cycle {
                supports: supports.to_owned(),
            });
        } else {
            return Err("support-pair component is neither a simple path nor cycle".to_owned());
        }
    }
    Ok(out)
}

const fn support_locus_key(support: &OrientedSupportKey) -> SupportLocusKey {
    match support {
        OrientedSupportKey::Plane {
            normal, offset_mm, ..
        } => SupportLocusKey::Plane {
            normal: *normal,
            offset_mm: *offset_mm,
        },
        OrientedSupportKey::Cylinder {
            axis,
            closest_mm,
            radius_mm,
            ..
        } => SupportLocusKey::Cylinder {
            axis: *axis,
            closest_mm: *closest_mm,
            radius_mm: *radius_mm,
        },
    }
}

fn edge_proves_support_intersection(
    edge: &EdgeGeometry,
    supports: &[OrientedSupportKey; 2],
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> bool {
    let loci = [
        support_locus_key(&supports[0]),
        support_locus_key(&supports[1]),
    ];
    if !support_pair_can_intersect(&loci) {
        return false;
    }

    let support = match &edge.support {
        CurveSupport::Other { .. } => {
            resolve_geometric_curve_support(edge.curve_id, entities, index)
        }
        support => Some(support.clone()),
    };
    match support {
        Some(CurveSupport::Line(line)) => {
            loci.iter().all(|locus| line_lies_on_support(line, locus))
        }
        Some(CurveSupport::Circle(circle)) => loci
            .iter()
            .all(|locus| circle_lies_on_support(circle, locus)),
        Some(CurveSupport::BSpline(spline)) => bspline_lies_on_supports(&spline, &loci),
        Some(CurveSupport::Other { .. }) | None => false,
    }
}

fn resolve_geometric_curve_support(
    curve_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<CurveSupport> {
    let mut current = curve_id;
    let mut seen = BTreeSet::new();
    for _ in 0..8 {
        if !seen.insert(current) {
            return None;
        }
        let support = brep::curve_support(current, entities, index);
        if !matches!(support, CurveSupport::Other { .. }) {
            return Some(support);
        }
        let record = simple_record(&entities[*index.get(&current)?])?;
        if !matches!(
            record.name.as_str(),
            "SURFACE_CURVE" | "SEAM_CURVE" | "INTERSECTION_CURVE"
        ) {
            return None;
        }
        let Parameter::List(params) = &record.parameter else {
            return None;
        };
        current = entity_ref_value(params.get(1)?)?;
    }
    None
}

fn bspline_lies_on_supports(spline: &BSplineSupport, loci: &[SupportLocusKey; 2]) -> bool {
    if let Some(line) = bspline_line_support(spline)
        && loci.iter().all(|locus| line_lies_on_support(line, locus))
    {
        return true;
    }
    bspline_is_circular_section_of_supports(spline, loci)
}

fn bspline_line_support(spline: &BSplineSupport) -> Option<brep::LineSupport> {
    const LOCUS_TOLERANCE_MM: f64 = 4.0 * SCALAR_Q_MM;
    if spline.control_points_mm.len() < 2 {
        return None;
    }
    match &spline.weights {
        Some(weights)
            if weights.len() == spline.control_points_mm.len()
                && weights
                    .iter()
                    .all(|weight| weight.is_finite() && *weight > 0.0) => {}
        Some(_) => return None,
        None => {}
    }
    let origin = spline.control_points_mm[0];
    let farthest = spline
        .control_points_mm
        .iter()
        .copied()
        .max_by(|first, second| distance(origin, *first).total_cmp(&distance(origin, *second)))?;
    let direction = normalize(sub(farthest, origin), LOCUS_TOLERANCE_MM)?;
    if spline
        .control_points_mm
        .iter()
        .any(|point| norm(cross(sub(*point, origin), direction)) > LOCUS_TOLERANCE_MM)
    {
        return None;
    }
    Some(brep::LineSupport {
        origin_mm: origin,
        direction,
    })
}

fn bspline_is_circular_section_of_supports(
    spline: &BSplineSupport,
    loci: &[SupportLocusKey; 2],
) -> bool {
    const ANGULAR_TOLERANCE: f64 = 4.0 * DIRECTION_Q;
    const LOCUS_TOLERANCE_MM: f64 = 4.0 * SCALAR_Q_MM;
    let (normal, offset_mm, axis, closest_mm, radius_mm) = match (&loci[0], &loci[1]) {
        (
            SupportLocusKey::Plane { normal, offset_mm },
            SupportLocusKey::Cylinder {
                axis,
                closest_mm,
                radius_mm,
            },
        )
        | (
            SupportLocusKey::Cylinder {
                axis,
                closest_mm,
                radius_mm,
            },
            SupportLocusKey::Plane { normal, offset_mm },
        ) => (*normal, *offset_mm, *axis, *closest_mm, *radius_mm),
        _ => return false,
    };
    let Some(normal) = dequantize_vector(normal, DIRECTION_Q) else {
        return false;
    };
    let Some(axis) = dequantize_vector(axis, DIRECTION_Q) else {
        return false;
    };
    if norm(cross(normal, axis)) > ANGULAR_TOLERANCE {
        return false;
    }
    let Some(closest) = dequantize_vector(closest_mm, POINT_Q_MM) else {
        return false;
    };
    let Some(offset) = dequantize_scalar(offset_mm) else {
        return false;
    };
    let Some(radius) = dequantize_scalar(radius_mm) else {
        return false;
    };
    let denominator = dot(normal, axis);
    if denominator.abs() <= ANGULAR_TOLERANCE {
        return false;
    }
    let center = add(
        closest,
        mul(axis, (offset - dot(normal, closest)) / denominator),
    );
    certify_rational_bspline_circle(spline, center, axis, radius, LOCUS_TOLERANCE_MM) == Some(true)
}

fn support_pair_can_intersect(loci: &[SupportLocusKey; 2]) -> bool {
    const ANGULAR_TOLERANCE: f64 = 4.0 * DIRECTION_Q;
    const LOCUS_TOLERANCE_MM: f64 = 4.0 * SCALAR_Q_MM;
    match (&loci[0], &loci[1]) {
        (
            SupportLocusKey::Plane { normal: first, .. },
            SupportLocusKey::Plane { normal: second, .. },
        ) => {
            let Some(first) = dequantize_vector(*first, DIRECTION_Q) else {
                return false;
            };
            let Some(second) = dequantize_vector(*second, DIRECTION_Q) else {
                return false;
            };
            norm(cross(first, second)) > ANGULAR_TOLERANCE
        }
        (
            SupportLocusKey::Plane { normal, offset_mm },
            SupportLocusKey::Cylinder {
                axis,
                closest_mm,
                radius_mm,
            },
        )
        | (
            SupportLocusKey::Cylinder {
                axis,
                closest_mm,
                radius_mm,
            },
            SupportLocusKey::Plane { normal, offset_mm },
        ) => {
            let Some(normal) = dequantize_vector(*normal, DIRECTION_Q) else {
                return false;
            };
            let Some(axis) = dequantize_vector(*axis, DIRECTION_Q) else {
                return false;
            };
            if dot(normal, axis).abs() > ANGULAR_TOLERANCE {
                return true;
            }
            let Some(closest) = dequantize_vector(*closest_mm, POINT_Q_MM) else {
                return false;
            };
            let Some(offset) = dequantize_scalar(*offset_mm) else {
                return false;
            };
            let Some(radius) = dequantize_scalar(*radius_mm) else {
                return false;
            };
            (dot(normal, closest) - offset).abs() <= radius + LOCUS_TOLERANCE_MM
        }
        (SupportLocusKey::Cylinder { .. }, SupportLocusKey::Cylinder { .. }) => false,
    }
}

fn line_lies_on_support(line: brep::LineSupport, support: &SupportLocusKey) -> bool {
    const ANGULAR_TOLERANCE: f64 = 4.0 * DIRECTION_Q;
    const LOCUS_TOLERANCE_MM: f64 = 4.0 * SCALAR_Q_MM;
    let Some(direction) = normalize(line.direction, 0.0) else {
        return false;
    };
    match support {
        SupportLocusKey::Plane { normal, offset_mm } => {
            let Some(normal) = dequantize_vector(*normal, DIRECTION_Q) else {
                return false;
            };
            let Some(offset) = dequantize_scalar(*offset_mm) else {
                return false;
            };
            (dot(normal, line.origin_mm) - offset).abs() <= LOCUS_TOLERANCE_MM
                && dot(normal, direction).abs() <= ANGULAR_TOLERANCE
        }
        SupportLocusKey::Cylinder {
            axis,
            closest_mm,
            radius_mm,
        } => {
            let Some(axis) = dequantize_vector(*axis, DIRECTION_Q) else {
                return false;
            };
            let Some(closest) = dequantize_vector(*closest_mm, POINT_Q_MM) else {
                return false;
            };
            let Some(radius) = dequantize_scalar(*radius_mm) else {
                return false;
            };
            let delta = sub(line.origin_mm, closest);
            let radial = sub(delta, mul(axis, dot(delta, axis)));
            norm(cross(direction, axis)) <= ANGULAR_TOLERANCE
                && (norm(radial) - radius).abs() <= LOCUS_TOLERANCE_MM
        }
    }
}

fn circle_lies_on_support(circle: brep::CircleSupport, support: &SupportLocusKey) -> bool {
    const ANGULAR_TOLERANCE: f64 = 4.0 * DIRECTION_Q;
    const LOCUS_TOLERANCE_MM: f64 = 4.0 * SCALAR_Q_MM;
    let Some(normal) = normalize(circle.normal, 0.0) else {
        return false;
    };
    match support {
        SupportLocusKey::Plane {
            normal: plane_normal,
            offset_mm,
        } => {
            let Some(plane_normal) = dequantize_vector(*plane_normal, DIRECTION_Q) else {
                return false;
            };
            let Some(offset) = dequantize_scalar(*offset_mm) else {
                return false;
            };
            norm(cross(normal, plane_normal)) <= ANGULAR_TOLERANCE
                && (dot(plane_normal, circle.center_mm) - offset).abs() <= LOCUS_TOLERANCE_MM
        }
        SupportLocusKey::Cylinder {
            axis,
            closest_mm,
            radius_mm,
        } => {
            let Some(axis) = dequantize_vector(*axis, DIRECTION_Q) else {
                return false;
            };
            let Some(closest) = dequantize_vector(*closest_mm, POINT_Q_MM) else {
                return false;
            };
            let Some(radius) = dequantize_scalar(*radius_mm) else {
                return false;
            };
            let delta = sub(circle.center_mm, closest);
            let radial_center = sub(delta, mul(axis, dot(delta, axis)));
            norm(cross(normal, axis)) <= ANGULAR_TOLERANCE
                && norm(radial_center) <= LOCUS_TOLERANCE_MM
                && (circle.radius_mm - radius).abs() <= LOCUS_TOLERANCE_MM
        }
    }
}

fn dequantize_scalar(value: i64) -> Option<f64> {
    Some(value.to_f64()? * SCALAR_Q_MM)
}

fn dequantize_vector(vector: [i64; 3], quantum: f64) -> Option<[f64; 3]> {
    Some([
        vector[0].to_f64()? * quantum,
        vector[1].to_f64()? * quantum,
        vector[2].to_f64()? * quantum,
    ])
}

fn oriented_support_key(
    face_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<OrientedSupportKey> {
    let surface_id = brep::face_surface(face_id, entities, index)?;
    let forward = match brep::face_sense(face_id, entities, index)?.as_str() {
        "T" => true,
        "F" => false,
        _ => return None,
    };

    match brep::surface_support(surface_id, entities, index) {
        SurfaceSupport::Plane(plane) => {
            let raw_normal = normalize(plane.normal, 0.0)?;
            let normal = canonical_direction(raw_normal);
            let outward = if forward {
                raw_normal
            } else {
                mul(raw_normal, -1.0)
            };
            Some(OrientedSupportKey::Plane {
                normal: vector_key(normal, DIRECTION_Q)?,
                offset_mm: quantize(dot(normal, plane.origin_mm), SCALAR_Q_MM)?,
                outward_positive: dot(outward, normal) >= 0.0,
            })
        }
        SurfaceSupport::Cylinder(cylinder) => {
            let raw_axis = normalize(cylinder.axis, 0.0)?;
            let axis = canonical_direction(raw_axis);
            let closest = sub(
                cylinder.axis_origin_mm,
                mul(axis, dot(axis, cylinder.axis_origin_mm)),
            );
            Some(OrientedSupportKey::Cylinder {
                axis: vector_key(axis, DIRECTION_Q)?,
                closest_mm: point_key(closest)?,
                radius_mm: quantize(cylinder.radius_mm, SCALAR_Q_MM)?,
                radial_outward: forward,
            })
        }
        SurfaceSupport::SplineExtrusion(surface) => {
            spline_extrusion_cylinder_key(&surface, forward)
        }
        _ => None,
    }
}

/// Normalize a spline-extrusion surface only when its profile is structurally
/// certified as a circular arc.
///
/// The three-point construction below only proposes a center/radius. Acceptance
/// is finite and fail-closed: the rational B-spline is Bézierized and every
/// homogeneous Bernstein coefficient is checked against the circle and axial
/// plane polynomial identities. This cannot hide a between-sample bump.
fn spline_extrusion_cylinder_key(
    surface: &SplineExtrusionSupport,
    forward: bool,
) -> Option<OrientedSupportKey> {
    const FIT_TOL_MM: f64 = 1.0e-7;

    if surface.max_residual_mm > FIT_TOL_MM {
        return None;
    }
    let raw_axis = normalize(surface.extrusion_mm, 0.0)?;
    let axis = canonical_direction(raw_axis);
    let (t0, t1) = bspline_domain(&surface.profile)?;
    let p0 = evaluate_bspline(&surface.profile, t0)?;
    let p1 = evaluate_bspline(&surface.profile, t1)?;
    let pm = evaluate_bspline(&surface.profile, f64::midpoint(t0, t1))?;

    let chord = sub(p1, p0);
    let chord_in_plane = sub(chord, mul(axis, dot(chord, axis)));
    let u = normalize(chord_in_plane, FIT_TOL_MM)?;
    let v = normalize(cross(axis, u), FIT_TOL_MM)?;
    let q1x = dot(sub(p1, p0), u);
    let q2 = sub(pm, p0);
    let q2x = dot(q2, u);
    let q2y = dot(q2, v);
    if q1x.abs() <= FIT_TOL_MM || q2y.abs() <= FIT_TOL_MM {
        return None;
    }
    let center_x = q1x * 0.5;
    let center_y = q1x.mul_add(-q2x, q2x.mul_add(q2x, q2y * q2y)) / (2.0 * q2y);
    if !center_y.is_finite() {
        return None;
    }
    let center = add(p0, add(mul(u, center_x), mul(v, center_y)));
    let radius = distance(center, p0);
    if !radius.is_finite() || radius <= FIT_TOL_MM {
        return None;
    }

    if !certify_rational_bspline_circle(&surface.profile, center, axis, radius, FIT_TOL_MM)? {
        return None;
    }

    let delta_t = (t1 - t0) * 1.0e-5;
    if !delta_t.is_finite() || delta_t.abs() <= f64::EPSILON {
        return None;
    }
    let middle = f64::midpoint(t0, t1);
    let before = evaluate_bspline(&surface.profile, middle - delta_t)?;
    let after = evaluate_bspline(&surface.profile, middle + delta_t)?;
    let tangent = sub(after, before);
    let natural_normal = normalize(cross(tangent, surface.extrusion_mm), 0.0)?;
    let radial = normalize(sub(pm, center), 0.0)?;
    let natural_radial_outward = dot(natural_normal, radial);
    if natural_radial_outward.abs() < 0.9 {
        return None;
    }
    let radial_outward = if forward {
        natural_radial_outward > 0.0
    } else {
        natural_radial_outward < 0.0
    };

    let closest = sub(center, mul(axis, dot(axis, center)));
    Some(OrientedSupportKey::Cylinder {
        axis: vector_key(axis, DIRECTION_Q)?,
        closest_mm: point_key(closest)?,
        radius_mm: quantize(radius, SCALAR_Q_MM)?,
        radial_outward,
    })
}

/// Prove that a quadratic rational B-spline lies on one circle whose plane is
/// normal to `axis`.
///
/// Each non-empty knot span of a degree-2 homogeneous B-spline is one
/// quadratic polynomial. Three homogeneous evaluations therefore recover its
/// complete Bernstein representation, after which both
/// `dot(P - C W, axis) = 0` and
/// `|P - C W|^2 - r^2 W^2 = 0` are polynomial identities. Bounding every
/// Bernstein coefficient is a bound over the whole span, not a sample test.
fn certify_rational_bspline_circle(
    spline: &BSplineSupport,
    center: [f64; 3],
    axis: [f64; 3],
    radius: f64,
    tolerance_mm: f64,
) -> Option<bool> {
    if spline.degree != 2
        || !radius.is_finite()
        || radius <= tolerance_mm
        || !tolerance_mm.is_finite()
        || tolerance_mm <= 0.0
    {
        return Some(false);
    }
    let (domain_start, domain_end) = bspline_domain(spline)?;
    let mut span_count = 0usize;

    for knots in spline.knots.windows(2) {
        let start = knots[0];
        let end = knots[1];
        if end <= start || end <= domain_start || start >= domain_end {
            continue;
        }
        if start < domain_start || end > domain_end {
            return Some(false);
        }

        let middle = f64::midpoint(start, end);
        let h0 = evaluate_homogeneous_bspline(spline, start)?;
        let hm = evaluate_homogeneous_bspline(spline, middle)?;
        let h2 = evaluate_homogeneous_bspline(spline, end)?;
        let h1 = std::array::from_fn(|coordinate| {
            f64::midpoint(h0[coordinate], h2[coordinate]).mul_add(-1.0, 2.0 * hm[coordinate])
        });
        let controls = [h0, h1, h2];
        if controls.iter().flatten().any(|value| !value.is_finite())
            || controls.iter().any(|control| control[3] <= 0.0)
        {
            return Some(false);
        }

        let min_weight = controls
            .iter()
            .map(|control| control[3])
            .fold(f64::INFINITY, f64::min);
        let centered = controls.map(|control| {
            [
                center[0].mul_add(-control[3], control[0]),
                center[1].mul_add(-control[3], control[1]),
                center[2].mul_add(-control[3], control[2]),
            ]
        });

        let axial_bound = tolerance_mm * min_weight;
        if centered
            .iter()
            .any(|point| dot(*point, axis).abs() > axial_bound)
        {
            return Some(false);
        }

        let mut circle_coefficients = [0.0_f64; 5];
        for first in 0..3 {
            for second in 0..3 {
                let degree = first + second;
                let factor = BINOMIAL_2[first] * BINOMIAL_2[second] / BINOMIAL_4[degree];
                let radial_dot = dot(centered[first], centered[second]);
                let weight_product = controls[first][3] * controls[second][3];
                let residual = (radius * radius).mul_add(-weight_product, radial_dot);
                circle_coefficients[degree] = factor.mul_add(residual, circle_coefficients[degree]);
            }
        }

        let radial_bound = tolerance_mm.mul_add(tolerance_mm, 2.0 * radius * tolerance_mm)
            * min_weight
            * min_weight;
        if circle_coefficients
            .iter()
            .any(|coefficient| coefficient.abs() > radial_bound)
        {
            return Some(false);
        }

        span_count += 1;
    }

    Some(span_count != 0)
}

fn evaluate_homogeneous_bspline(spline: &BSplineSupport, parameter: f64) -> Option<[f64; 4]> {
    let control_count = spline.control_points_mm.len();
    let (domain_start, domain_end) = bspline_domain(spline)?;
    if !parameter.is_finite() || parameter < domain_start || parameter > domain_end {
        return None;
    }

    let weights = match &spline.weights {
        Some(weights) if weights.len() == control_count => weights.clone(),
        Some(_) => return None,
        None => vec![1.0; control_count],
    };
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return None;
    }
    if parameter >= domain_end {
        let point = *spline.control_points_mm.last()?;
        let weight = *weights.last()?;
        return Some([
            point[0] * weight,
            point[1] * weight,
            point[2] * weight,
            weight,
        ]);
    }

    let mut basis = vec![0.0; control_count + spline.degree];
    for (index, basis_value) in basis.iter_mut().enumerate().take(control_count) {
        if spline.knots[index] <= parameter && parameter < spline.knots[index + 1] {
            *basis_value = 1.0;
        }
    }
    for degree in 1..=spline.degree {
        let mut next = vec![0.0; basis.len()];
        for index in 0..control_count {
            let left_denominator = spline.knots[index + degree] - spline.knots[index];
            let left = if left_denominator.abs() > f64::EPSILON {
                (parameter - spline.knots[index]) / left_denominator * basis[index]
            } else {
                0.0
            };
            let right_denominator = spline.knots[index + degree + 1] - spline.knots[index + 1];
            let right = if right_denominator.abs() > f64::EPSILON {
                (spline.knots[index + degree + 1] - parameter) / right_denominator
                    * basis[index + 1]
            } else {
                0.0
            };
            next[index] = left + right;
        }
        basis = next;
    }

    let mut homogeneous = [0.0_f64; 4];
    for ((point, weight), basis) in spline
        .control_points_mm
        .iter()
        .zip(&weights)
        .zip(&basis[..control_count])
    {
        let coefficient = weight * basis;
        for coordinate in 0..3 {
            homogeneous[coordinate] =
                coefficient.mul_add(point[coordinate], homogeneous[coordinate]);
        }
        homogeneous[3] += coefficient;
    }
    homogeneous
        .iter()
        .all(|value| value.is_finite())
        .then_some(homogeneous)
}

fn bspline_domain(spline: &BSplineSupport) -> Option<(f64, f64)> {
    let control_count = spline.control_points_mm.len();
    if control_count == 0
        || spline.degree == 0
        || spline.knots.len() != control_count.checked_add(spline.degree)?.checked_add(1)?
    {
        return None;
    }
    let start = *spline.knots.get(spline.degree)?;
    let end = *spline.knots.get(control_count)?;
    (start.is_finite() && end.is_finite() && end > start).then_some((start, end))
}

fn evaluate_bspline(spline: &BSplineSupport, parameter: f64) -> Option<[f64; 3]> {
    let control_count = spline.control_points_mm.len();
    let (domain_start, domain_end) = bspline_domain(spline)?;
    if !parameter.is_finite() || parameter < domain_start || parameter > domain_end {
        return None;
    }
    if parameter >= domain_end {
        return spline.control_points_mm.last().copied();
    }

    let weights = match &spline.weights {
        Some(weights) if weights.len() == control_count => weights.clone(),
        Some(_) => return None,
        None => vec![1.0; control_count],
    };
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return None;
    }

    let mut basis = vec![0.0; control_count + spline.degree];
    for (index, basis_value) in basis.iter_mut().enumerate().take(control_count) {
        if spline.knots[index] <= parameter && parameter < spline.knots[index + 1] {
            *basis_value = 1.0;
        }
    }
    for degree in 1..=spline.degree {
        let mut next = vec![0.0; basis.len()];
        for index in 0..control_count {
            let left_denominator = spline.knots[index + degree] - spline.knots[index];
            let left = if left_denominator.abs() > f64::EPSILON {
                (parameter - spline.knots[index]) / left_denominator * basis[index]
            } else {
                0.0
            };
            let right_denominator = spline.knots[index + degree + 1] - spline.knots[index + 1];
            let right = if right_denominator.abs() > f64::EPSILON {
                (spline.knots[index + degree + 1] - parameter) / right_denominator
                    * basis[index + 1]
            } else {
                0.0
            };
            next[index] = left + right;
        }
        basis = next;
    }

    let mut numerator = [0.0; 3];
    let mut denominator = 0.0;
    for ((point, weight), basis) in spline
        .control_points_mm
        .iter()
        .zip(&weights)
        .zip(&basis[..control_count])
    {
        let weighted_basis = weight * basis;
        denominator += weighted_basis;
        for axis in 0..3 {
            numerator[axis] = weighted_basis.mul_add(point[axis], numerator[axis]);
        }
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        return None;
    }
    let point = numerator.map(|coordinate| coordinate / denominator);
    point.iter().all(|value| value.is_finite()).then_some(point)
}

fn sorted_pair(first: OrientedSupportKey, second: OrientedSupportKey) -> [OrientedSupportKey; 2] {
    if first <= second {
        [first, second]
    } else {
        [second, first]
    }
}

fn point_key(point: [f64; 3]) -> Option<PointKey> {
    vector_key(point, POINT_Q_MM)
}

fn vector_key(vector: [f64; 3], quantum: f64) -> Option<[i64; 3]> {
    Some([
        quantize(vector[0], quantum)?,
        quantize(vector[1], quantum)?,
        quantize(vector[2], quantum)?,
    ])
}

fn quantize(value: f64, quantum: f64) -> Option<i64> {
    if !value.is_finite() || !quantum.is_finite() || quantum <= 0.0 {
        return None;
    }
    (value / quantum).round().to_i64()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(offset_mm: i64, outward_positive: bool) -> OrientedSupportKey {
        OrientedSupportKey::Plane {
            normal: [0, 0, 10_000_000_000],
            offset_mm,
            outward_positive,
        }
    }

    #[test]
    fn line_carrier_proof_rejects_off_plane_line() {
        let support = SupportLocusKey::Plane {
            normal: [0, 0, 10_000_000_000],
            offset_mm: 0,
        };
        let on_plane = brep::LineSupport {
            origin_mm: [0.0, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0],
        };
        let off_plane = brep::LineSupport {
            origin_mm: [0.0, 0.0, 1.0e-3],
            direction: [1.0, 0.0, 0.0],
        };
        assert!(line_lies_on_support(on_plane, &support));
        assert!(!line_lies_on_support(off_plane, &support));
    }

    #[test]
    fn circle_carrier_proof_rejects_wrong_cylinder_radius() {
        let support = SupportLocusKey::Cylinder {
            axis: [0, 0, 10_000_000_000],
            closest_mm: [0, 0, 0],
            radius_mm: 20_000_000,
        };
        let exact = brep::CircleSupport {
            center_mm: [0.0, 0.0, 5.0],
            normal: [0.0, 0.0, 1.0],
            x_direction: [1.0, 0.0, 0.0],
            radius_mm: 2.0,
        };
        let wrong = brep::CircleSupport {
            radius_mm: 2.001,
            ..exact
        };
        assert!(circle_lies_on_support(exact, &support));
        assert!(!circle_lies_on_support(wrong, &support));
    }

    #[test]
    fn rational_quadratic_circle_certificate_rejects_between_span_bump() {
        let weight = 0.5_f64.sqrt();
        let exact = BSplineSupport {
            degree: 2,
            control_points_mm: vec![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            weights: Some(vec![1.0, weight, 1.0]),
        };
        assert_eq!(
            certify_rational_bspline_circle(&exact, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.0, 1.0e-7,),
            Some(true)
        );

        let mut perturbed = exact;
        perturbed.control_points_mm[1][0] += 1.0e-3;
        assert_eq!(
            certify_rational_bspline_circle(
                &perturbed,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                1.0,
                1.0e-7,
            ),
            Some(false)
        );
    }

    #[test]
    fn open_chain_normalization_ignores_degree_two_partitions() {
        let supports = [plane(0, true), plane(10_000_000, true)];
        let one = normalize_support_pair_chains(&supports, &[([0, 0, 0], [20, 0, 0])]);
        let split = normalize_support_pair_chains(
            &supports,
            &[([0, 0, 0], [7, 0, 0]), ([7, 0, 0], [20, 0, 0])],
        );
        assert!(one.is_ok());
        assert_eq!(one, split);
    }

    #[test]
    fn cycle_normalization_ignores_partition_vertices() {
        let supports = [plane(0, true), plane(10_000_000, true)];
        let first = normalize_support_pair_chains(
            &supports,
            &[
                ([0, 0, 0], [10, 0, 0]),
                ([10, 0, 0], [10, 10, 0]),
                ([10, 10, 0], [0, 10, 0]),
                ([0, 10, 0], [0, 0, 0]),
            ],
        );
        let split = normalize_support_pair_chains(
            &supports,
            &[
                ([0, 0, 0], [5, 0, 0]),
                ([5, 0, 0], [10, 0, 0]),
                ([10, 0, 0], [10, 10, 0]),
                ([10, 10, 0], [0, 10, 0]),
                ([0, 10, 0], [0, 0, 0]),
            ],
        );
        assert!(first.is_ok());
        assert_eq!(first, split);
    }

    #[test]
    fn branch_is_rejected() {
        let supports = [plane(0, true), plane(10_000_000, true)];
        let result = normalize_support_pair_chains(
            &supports,
            &[
                ([0, 0, 0], [10, 0, 0]),
                ([10, 0, 0], [20, 0, 0]),
                ([10, 0, 0], [10, 10, 0]),
            ],
        );
        assert!(result.is_err());
    }

    #[test]
    fn parallel_distinct_planes_do_not_define_boundary_curve() {
        let loci = [
            SupportLocusKey::Plane {
                normal: [0, 0, 10_000_000_000],
                offset_mm: 0,
            },
            SupportLocusKey::Plane {
                normal: [0, 0, 10_000_000_000],
                offset_mm: 10_000_000,
            },
        ];
        assert!(!support_pair_can_intersect(&loci));
    }

    #[test]
    fn plane_outside_parallel_cylinder_does_not_intersect() {
        let loci = [
            SupportLocusKey::Plane {
                normal: [10_000_000_000, 0, 0],
                offset_mm: 20_000_000,
            },
            SupportLocusKey::Cylinder {
                axis: [0, 0, 10_000_000_000],
                closest_mm: [0, 0, 0],
                radius_mm: 10_000_000,
            },
        ];
        assert!(!support_pair_can_intersect(&loci));
    }

    #[test]
    fn oblique_plane_and_cylinder_have_intersection() {
        let loci = [
            SupportLocusKey::Plane {
                normal: [0, 0, 10_000_000_000],
                offset_mm: 0,
            },
            SupportLocusKey::Cylinder {
                axis: [10_000_000_000, 0, 0],
                closest_mm: [0, 0, 0],
                radius_mm: 10_000_000,
            },
        ];
        assert!(support_pair_can_intersect(&loci));
    }

    #[test]
    fn rational_linear_bspline_requires_collinear_control_geometry() {
        let exact = BSplineSupport {
            degree: 1,
            control_points_mm: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            knots: vec![0.0, 0.0, 0.5, 1.0, 1.0],
            weights: Some(vec![1.0, 0.8, 1.0]),
        };
        assert!(bspline_line_support(&exact).is_some());

        let mut bent = exact;
        bent.control_points_mm[1][1] = 1.0e-3;
        assert!(bspline_line_support(&bent).is_none());
    }
}
