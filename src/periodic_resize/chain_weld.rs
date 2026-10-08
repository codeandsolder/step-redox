use super::graph_editor::GraphEditor;
use super::{entity_ref, entity_ref_list, list_params, numeric_value, quantize_coord};
use crate::periodic_chains::PeriodicChainPattern;
use crate::step_graph::entity_ref_value;
use anyhow::{Result, anyhow, bail};
use ruststep::ast::{EntityInstance, Parameter, Record};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum ChainCurveKey {
    Line,
    RationalSingleSpan {
        degree: i64,
        poles: Vec<[i64; 3]>,
        weights: Vec<i64>,
    },
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ChainEdgeKey {
    endpoints: [[i64; 3]; 2],
    curve: ChainCurveKey,
}

pub(super) struct ChainWeldPlan {
    pub(super) seam_pairs: Vec<(u64, u64)>,
    pub(super) vertex_map: HashMap<u64, u64>,
    pub(super) edge_map: HashMap<u64, u64>,
}

impl ChainWeldPlan {
    fn from_pairs(graph: &GraphEditor<'_>, seam_pairs: Vec<(u64, u64)>) -> Result<Self> {
        let (vertex_map, edge_map) = chain_build_weld_maps(graph, &seam_pairs)?;
        Ok(Self {
            seam_pairs,
            vertex_map,
            edge_map,
        })
    }

    pub(super) fn seed_prune_roots(&self, prune_roots: &mut HashSet<u64>) {
        prune_roots.extend(self.seam_pairs.iter().map(|&(_, duplicate)| duplicate));
        // Welding rewrites duplicate EDGE_CURVE endpoints to canonical vertices.
        // Seed detached duplicate vertices explicitly: after the rewrite they
        // are no longer descendants of duplicate seam-edge roots.
        prune_roots.extend(self.vertex_map.keys().copied());
    }
}

fn mapped_seam_edges(mapping: &HashMap<u64, u64>, edges: &[u64], label: &str) -> Result<Vec<u64>> {
    edges
        .iter()
        .map(|edge| {
            mapping
                .get(edge)
                .copied()
                .ok_or_else(|| anyhow!("{label} missing seam edge #{edge}"))
        })
        .collect()
}

pub(super) fn plan_chain_expansion_welds(
    graph: &GraphEditor<'_>,
    chain: &PeriodicChainPattern,
    source_edge_faces: &HashMap<u64, Vec<u64>>,
    unit_maps: &[HashMap<u64, u64>],
    tail_map: &HashMap<u64, u64>,
) -> Result<ChainWeldPlan> {
    let old_sites = chain.sites;
    let unit_gap_index = old_sites - 4;
    let unit_site_index = old_sites - 3;
    let tail_gap0 = old_sites - 3;
    let tail_site0 = old_sites - 2;

    let left_site = chain.site_face_ids[unit_gap_index]
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let unit_gap = chain.gap_face_ids[unit_gap_index]
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let unit_site = chain.site_face_ids[unit_site_index]
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let first_tail_gap = chain.gap_face_ids[tail_gap0]
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let overlay_left_site = chain.site_face_ids[unit_site_index - 1]
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let overlay_tail_site = chain.site_face_ids[tail_site0]
        .iter()
        .copied()
        .collect::<HashSet<_>>();

    let source_unit_left = chain_interface_edges(source_edge_faces, &unit_gap, &left_site);
    let source_unit_right = chain_interface_edges(source_edge_faces, &unit_site, &first_tail_gap);
    if source_unit_left.is_empty() || source_unit_right.is_empty() {
        bail!(
            "periodic-chain site/gap seam is empty: left={} right={}",
            source_unit_left.len(),
            source_unit_right.len()
        );
    }
    chain_require_supported_seam_edges(
        graph,
        source_unit_left
            .iter()
            .chain(source_unit_right.iter())
            .copied(),
    )?;

    let overlay_edges_per_boundary = chain.adjacent_site_edge_counts[0];
    let source_overlay_left =
        chain_interface_edges(source_edge_faces, &overlay_left_site, &unit_site);
    let source_overlay_right =
        chain_interface_edges(source_edge_faces, &unit_site, &overlay_tail_site);
    if source_overlay_left.len() != overlay_edges_per_boundary
        || source_overlay_right.len() != overlay_edges_per_boundary
    {
        bail!(
            "periodic-chain adjacent-site overlay seam differs from proof: expected={overlay_edges_per_boundary} left={} right={}",
            source_overlay_left.len(),
            source_overlay_right.len()
        );
    }
    chain_require_supported_seam_edges(
        graph,
        source_overlay_left
            .iter()
            .chain(source_overlay_right.iter())
            .copied(),
    )?;

    let first_unit = unit_maps
        .first()
        .ok_or_else(|| anyhow!("periodic-chain expansion has no inserted unit map"))?;
    let last_unit = unit_maps
        .last()
        .ok_or_else(|| anyhow!("periodic-chain expansion has no inserted unit map"))?;
    let mut seam_pairs = Vec::new();

    let first_left = mapped_seam_edges(first_unit, &source_unit_left, "first inserted unit")?;
    seam_pairs.extend(chain_pair_edges_by_geometry(
        graph,
        &source_unit_right,
        &first_left,
    )?);

    for pair in unit_maps.windows(2) {
        let left = mapped_seam_edges(&pair[0], &source_unit_right, "inserted unit right")?;
        let right = mapped_seam_edges(&pair[1], &source_unit_left, "inserted unit left")?;
        seam_pairs.extend(chain_pair_edges_by_geometry(graph, &left, &right)?);
    }

    let final_right = mapped_seam_edges(last_unit, &source_unit_right, "last inserted unit")?;
    let tail_left = mapped_seam_edges(tail_map, &source_unit_right, "translated tail")?;
    seam_pairs.extend(chain_pair_edges_by_geometry(
        graph,
        &final_right,
        &tail_left,
    )?);

    if overlay_edges_per_boundary > 0 {
        let first_overlay_left = mapped_seam_edges(
            first_unit,
            &source_overlay_left,
            "first inserted unit overlay-left",
        )?;
        seam_pairs.extend(chain_pair_edges_by_geometry(
            graph,
            &source_overlay_right,
            &first_overlay_left,
        )?);

        for pair in unit_maps.windows(2) {
            let left = mapped_seam_edges(
                &pair[0],
                &source_overlay_right,
                "inserted unit overlay-right",
            )?;
            let right =
                mapped_seam_edges(&pair[1], &source_overlay_left, "inserted unit overlay-left")?;
            seam_pairs.extend(chain_pair_edges_by_geometry(graph, &left, &right)?);
        }

        let final_overlay_right = mapped_seam_edges(
            last_unit,
            &source_overlay_right,
            "last inserted unit overlay-right",
        )?;
        let tail_overlay_left = mapped_seam_edges(
            tail_map,
            &source_overlay_right,
            "translated tail overlay-left",
        )?;
        seam_pairs.extend(chain_pair_edges_by_geometry(
            graph,
            &final_overlay_right,
            &tail_overlay_left,
        )?);
    }

    ChainWeldPlan::from_pairs(graph, seam_pairs)
}

pub(super) fn plan_chain_shrink_welds(
    graph: &GraphEditor<'_>,
    chain: &PeriodicChainPattern,
    source_edge_faces: &HashMap<u64, Vec<u64>>,
    tail_map: &HashMap<u64, u64>,
    new_sites: usize,
) -> Result<ChainWeldPlan> {
    let old_sites = chain.sites;
    let kept_last_site = new_sites - 3;
    let remove_site_start = new_sites - 2;
    let remove_gap_start = new_sites - 3;
    let tail_gap0 = old_sites - 3;
    let tail_site0 = old_sites - 2;

    let kept_site = chain.site_face_ids[kept_last_site]
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let first_removed_site = chain.site_face_ids[remove_site_start]
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let first_removed_gap = chain.gap_face_ids[remove_gap_start]
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let last_removed_site = chain.site_face_ids[tail_gap0]
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let first_tail_gap = chain.gap_face_ids[tail_gap0]
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let tail_site = chain.site_face_ids[tail_site0]
        .iter()
        .copied()
        .collect::<HashSet<_>>();

    let source_kept_right =
        chain_interface_edges(source_edge_faces, &kept_site, &first_removed_gap);
    let source_tail_left =
        chain_interface_edges(source_edge_faces, &last_removed_site, &first_tail_gap);
    if source_kept_right.is_empty() || source_tail_left.is_empty() {
        bail!(
            "periodic-chain shrink seam is empty: kept={} tail={}",
            source_kept_right.len(),
            source_tail_left.len()
        );
    }
    chain_require_supported_seam_edges(
        graph,
        source_kept_right
            .iter()
            .chain(source_tail_left.iter())
            .copied(),
    )?;

    let mapped_tail_left = mapped_seam_edges(tail_map, &source_tail_left, "translated tail")?;
    let mut seam_pairs =
        chain_pair_edges_by_geometry(graph, &source_kept_right, &mapped_tail_left)?;

    let overlay_edges_per_boundary = chain.adjacent_site_edge_counts[0];
    let source_kept_overlay =
        chain_interface_edges(source_edge_faces, &kept_site, &first_removed_site);
    let source_tail_overlay =
        chain_interface_edges(source_edge_faces, &last_removed_site, &tail_site);
    if source_kept_overlay.len() != overlay_edges_per_boundary
        || source_tail_overlay.len() != overlay_edges_per_boundary
    {
        bail!(
            "periodic-chain shrink overlay seam differs from proof: expected={overlay_edges_per_boundary} kept={} tail={}",
            source_kept_overlay.len(),
            source_tail_overlay.len()
        );
    }
    chain_require_supported_seam_edges(
        graph,
        source_kept_overlay
            .iter()
            .chain(source_tail_overlay.iter())
            .copied(),
    )?;
    if overlay_edges_per_boundary > 0 {
        let mapped_tail_overlay =
            mapped_seam_edges(tail_map, &source_tail_overlay, "translated tail overlay")?;
        seam_pairs.extend(chain_pair_edges_by_geometry(
            graph,
            &source_kept_overlay,
            &mapped_tail_overlay,
        )?);
    }

    ChainWeldPlan::from_pairs(graph, seam_pairs)
}

fn chain_interface_edges(
    edge_faces: &HashMap<u64, Vec<u64>>,
    left: &HashSet<u64>,
    right: &HashSet<u64>,
) -> Vec<u64> {
    let mut out = edge_faces
        .iter()
        .filter_map(|(&edge, faces)| {
            (faces.iter().any(|face| left.contains(face))
                && faces.iter().any(|face| right.contains(face)))
            .then_some(edge)
        })
        .collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn chain_edge_curve_id(graph: &GraphEditor<'_>, edge: u64) -> Result<u64> {
    let record = graph
        .simple_record(edge)
        .ok_or_else(|| anyhow!("missing chain seam edge #{edge}"))?;
    let params = list_params(record).ok_or_else(|| anyhow!("chain edge params invalid"))?;
    params
        .get(3)
        .and_then(entity_ref_value)
        .ok_or_else(|| anyhow!("chain seam edge #{edge} missing curve support"))
}

fn chain_complex_record<'a>(records: &'a [Record], name: &str) -> Result<&'a Record> {
    let mut matches = records.iter().filter(|record| record.name == name);
    let record = matches
        .next()
        .ok_or_else(|| anyhow!("complex seam curve missing {name} record"))?;
    if matches.next().is_some() {
        bail!("complex seam curve has duplicate {name} records");
    }
    Ok(record)
}

fn chain_rational_single_span_key(
    graph: &GraphEditor<'_>,
    curve: u64,
    records: &[Record],
) -> Result<ChainCurveKey> {
    const EXPECTED: [&str; 7] = [
        "BOUNDED_CURVE",
        "B_SPLINE_CURVE",
        "B_SPLINE_CURVE_WITH_KNOTS",
        "CURVE",
        "GEOMETRIC_REPRESENTATION_ITEM",
        "RATIONAL_B_SPLINE_CURVE",
        "REPRESENTATION_ITEM",
    ];
    let mut actual = records
        .iter()
        .map(|record| record.name.as_str())
        .collect::<Vec<_>>();
    actual.sort_unstable();
    let mut expected = EXPECTED.to_vec();
    expected.sort_unstable();
    if actual != expected {
        bail!("complex seam curve #{curve} is not the supported rational single-span form");
    }

    let bspline = chain_complex_record(records, "B_SPLINE_CURVE")?;
    let bp = list_params(bspline).ok_or_else(|| anyhow!("B_SPLINE_CURVE params invalid"))?;
    if bp.len() != 5 {
        bail!("complex seam curve #{curve} has unexpected B_SPLINE_CURVE arity");
    }
    let degree = match bp.first() {
        Some(Parameter::Integer(value)) if *value >= 1 => *value,
        _ => bail!("complex seam curve #{curve} has invalid spline degree"),
    };
    let poles = bp
        .get(1)
        .and_then(entity_ref_list)
        .ok_or_else(|| anyhow!("complex seam curve #{curve} has invalid pole list"))?;
    if poles.len() != degree as usize + 1 {
        bail!("complex seam curve #{curve} is not a single-span Bezier-equivalent spline");
    }
    if !matches!(bp.get(2), Some(Parameter::Enumeration(value)) if value == "UNSPECIFIED")
        || !matches!(bp.get(3), Some(Parameter::Enumeration(value)) if value == "F")
        || !matches!(bp.get(4), Some(Parameter::Enumeration(value)) if value == "F")
    {
        bail!("complex seam curve #{curve} uses unsupported spline flags");
    }

    let knots = chain_complex_record(records, "B_SPLINE_CURVE_WITH_KNOTS")?;
    let kp =
        list_params(knots).ok_or_else(|| anyhow!("B_SPLINE_CURVE_WITH_KNOTS params invalid"))?;
    if kp.len() != 3 {
        bail!("complex seam curve #{curve} has unexpected knot-record arity");
    }
    let multiplicities = match kp.first() {
        Some(Parameter::List(items)) => items
            .iter()
            .map(|item| match item {
                Parameter::Integer(value) => Ok(*value),
                _ => bail!("complex seam curve #{curve} has non-integer knot multiplicity"),
            })
            .collect::<Result<Vec<_>>>()?,
        _ => bail!("complex seam curve #{curve} has invalid knot multiplicities"),
    };
    if multiplicities != [degree + 1, degree + 1] {
        bail!("complex seam curve #{curve} is not clamped single-span");
    }
    let knot_values = match kp.get(1) {
        Some(Parameter::List(items)) => items
            .iter()
            .map(|item| {
                numeric_value(item)
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| anyhow!("complex seam curve #{curve} has invalid knot value"))
            })
            .collect::<Result<Vec<_>>>()?,
        _ => bail!("complex seam curve #{curve} has invalid knot list"),
    };
    if knot_values.len() != 2 || knot_values[0] == knot_values[1] {
        bail!("complex seam curve #{curve} is not a finite single knot span");
    }
    if !matches!(kp.get(2), Some(Parameter::Enumeration(value)) if value == "UNSPECIFIED") {
        bail!("complex seam curve #{curve} uses unsupported knot specification");
    }

    let rational = chain_complex_record(records, "RATIONAL_B_SPLINE_CURVE")?;
    let rp =
        list_params(rational).ok_or_else(|| anyhow!("RATIONAL_B_SPLINE_CURVE params invalid"))?;
    let weights = match rp {
        [Parameter::List(items)] => items
            .iter()
            .map(|item| {
                let value = numeric_value(item)
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .ok_or_else(|| anyhow!("complex seam curve #{curve} has invalid weight"))?;
                Ok((value * 1.0e12).round() as i64)
            })
            .collect::<Result<Vec<_>>>()?,
        _ => bail!("complex seam curve #{curve} has invalid rational weights"),
    };
    if weights.len() != poles.len() {
        bail!("complex seam curve #{curve} has a pole/weight count mismatch");
    }

    let poles = poles
        .into_iter()
        .map(|point| graph.cartesian_point(point).map(quantize_coord))
        .collect::<Result<Vec<_>>>()?;
    Ok(ChainCurveKey::RationalSingleSpan {
        degree,
        poles,
        weights,
    })
}

pub(super) fn chain_curve_key(graph: &GraphEditor<'_>, edge: u64) -> Result<ChainCurveKey> {
    let curve = chain_edge_curve_id(graph, edge)?;
    match graph
        .entity(curve)
        .ok_or_else(|| anyhow!("missing chain seam curve #{curve}"))?
    {
        EntityInstance::Simple { record, .. } if record.name == "LINE" => Ok(ChainCurveKey::Line),
        EntityInstance::Complex { subsuper, .. } => {
            chain_rational_single_span_key(graph, curve, &subsuper.0)
        }
        entity => bail!(
            "periodic-chain seam edge #{edge} uses unsupported {} curve support",
            match entity {
                EntityInstance::Simple { record, .. } => record.name.as_str(),
                EntityInstance::Complex { .. } => "COMPLEX",
            }
        ),
    }
}

fn chain_require_supported_seam_edges(
    graph: &GraphEditor<'_>,
    edges: impl IntoIterator<Item = u64>,
) -> Result<()> {
    for edge in edges {
        chain_curve_key(graph, edge)?;
    }
    Ok(())
}

fn chain_edge_key(graph: &GraphEditor<'_>, edge: u64) -> Result<ChainEdgeKey> {
    let [va, vb] = graph.edge_vertices(edge)?;
    let mut endpoints = [
        quantize_coord(graph.vertex_coord(va)?),
        quantize_coord(graph.vertex_coord(vb)?),
    ];
    endpoints.sort_unstable();
    Ok(ChainEdgeKey {
        endpoints,
        curve: chain_curve_key(graph, edge)?,
    })
}

fn chain_pair_edges_by_geometry(
    graph: &GraphEditor<'_>,
    canonical: &[u64],
    duplicate: &[u64],
) -> Result<Vec<(u64, u64)>> {
    let mut a = HashMap::<ChainEdgeKey, Vec<u64>>::new();
    let mut b = HashMap::<ChainEdgeKey, Vec<u64>>::new();
    for &edge in canonical {
        a.entry(chain_edge_key(graph, edge)?)
            .or_default()
            .push(edge);
    }
    for &edge in duplicate {
        b.entry(chain_edge_key(graph, edge)?)
            .or_default()
            .push(edge);
    }
    if a.keys().collect::<HashSet<_>>() != b.keys().collect::<HashSet<_>>() {
        bail!("periodic-chain seam geometry differs after translation");
    }

    let mut keys = a.keys().cloned().collect::<Vec<_>>();
    keys.sort_by(|x, y| {
        x.endpoints
            .cmp(&y.endpoints)
            .then_with(|| x.curve.cmp(&y.curve))
    });
    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        let left = &a[&key];
        let right = &b[&key];
        if left.len() != 1 || right.len() != 1 {
            bail!("periodic-chain seam geometry is ambiguous: {left:?} vs {right:?}");
        }
        out.push((left[0], right[0]));
    }
    Ok(out)
}

fn chain_build_weld_maps(
    graph: &GraphEditor<'_>,
    pairs: &[(u64, u64)],
) -> Result<(HashMap<u64, u64>, HashMap<u64, u64>)> {
    let mut vertex_map = HashMap::<u64, u64>::new();
    let mut edge_map = HashMap::<u64, u64>::new();

    for &(canonical, duplicate) in pairs {
        edge_map.insert(duplicate, canonical);
        let canonical_vertices = graph.edge_vertices(canonical)?;
        let duplicate_vertices = graph.edge_vertices(duplicate)?;
        let a = canonical_vertices
            .into_iter()
            .map(|vertex| Ok((quantize_coord(graph.vertex_coord(vertex)?), vertex)))
            .collect::<Result<HashMap<_, _>>>()?;
        let b = duplicate_vertices
            .into_iter()
            .map(|vertex| Ok((quantize_coord(graph.vertex_coord(vertex)?), vertex)))
            .collect::<Result<HashMap<_, _>>>()?;
        if a.keys().collect::<HashSet<_>>() != b.keys().collect::<HashSet<_>>() {
            bail!(
                "periodic-chain seam endpoint mismatch between edges #{canonical} and #{duplicate}"
            );
        }
        for (coord, &duplicate_vertex) in &b {
            let canonical_vertex = a[coord];
            if let Some(existing) = vertex_map.insert(duplicate_vertex, canonical_vertex)
                && existing != canonical_vertex
            {
                bail!(
                    "conflicting periodic-chain vertex weld for #{duplicate_vertex}: #{existing} / #{canonical_vertex}"
                );
            }
        }
    }
    Ok((vertex_map, edge_map))
}

fn chain_face_oriented_edges(graph: &GraphEditor<'_>, face: u64) -> Result<Vec<u64>> {
    let record = graph
        .simple_record(face)
        .ok_or_else(|| anyhow!("missing face #{face}"))?;
    let params = list_params(record).ok_or_else(|| anyhow!("face params invalid"))?;
    let bounds = params
        .get(1)
        .and_then(entity_ref_list)
        .ok_or_else(|| anyhow!("face #{face} has no bounds"))?;
    let mut out = Vec::new();
    for bound in bounds {
        let bound_record = graph
            .simple_record(bound)
            .ok_or_else(|| anyhow!("missing bound #{bound}"))?;
        let bparams = list_params(bound_record).ok_or_else(|| anyhow!("bound params invalid"))?;
        let loop_id = bparams
            .get(1)
            .and_then(entity_ref_value)
            .ok_or_else(|| anyhow!("bound #{bound} missing EDGE_LOOP"))?;
        let loop_record = graph
            .simple_record(loop_id)
            .ok_or_else(|| anyhow!("missing loop #{loop_id}"))?;
        let lparams = list_params(loop_record).ok_or_else(|| anyhow!("loop params invalid"))?;
        out.extend(
            lparams
                .get(1)
                .and_then(entity_ref_list)
                .ok_or_else(|| anyhow!("loop #{loop_id} has no oriented edges"))?,
        );
    }
    Ok(out)
}

fn chain_oriented_edge_info(graph: &GraphEditor<'_>, oe: u64) -> Result<(u64, bool)> {
    let record = graph
        .simple_record(oe)
        .ok_or_else(|| anyhow!("missing ORIENTED_EDGE #{oe}"))?;
    if record.name != "ORIENTED_EDGE" {
        bail!("#{oe} is not ORIENTED_EDGE");
    }
    let params = list_params(record).ok_or_else(|| anyhow!("ORIENTED_EDGE params invalid"))?;
    let edge = params
        .get(3)
        .and_then(entity_ref_value)
        .ok_or_else(|| anyhow!("ORIENTED_EDGE #{oe} missing EDGE_CURVE"))?;
    let forward = matches!(
        params.get(4),
        Some(Parameter::Enumeration(value)) if value == "T"
    );
    Ok((edge, forward))
}

fn chain_set_edge_vertices(graph: &mut GraphEditor<'_>, edge: u64, va: u64, vb: u64) -> Result<()> {
    let record = graph
        .simple_record_mut(edge)
        .ok_or_else(|| anyhow!("missing EDGE_CURVE #{edge}"))?;
    if record.name != "EDGE_CURVE" {
        bail!("#{edge} is not EDGE_CURVE");
    }
    let Parameter::List(params) = &mut record.parameter else {
        bail!("EDGE_CURVE #{edge} params invalid");
    };
    if params.len() < 5 {
        bail!("EDGE_CURVE #{edge} params incomplete");
    }
    params[1] = entity_ref(va);
    params[2] = entity_ref(vb);
    Ok(())
}

fn chain_set_oriented_edge(
    graph: &mut GraphEditor<'_>,
    oe: u64,
    edge: u64,
    forward: bool,
) -> Result<()> {
    let record = graph
        .simple_record_mut(oe)
        .ok_or_else(|| anyhow!("missing ORIENTED_EDGE #{oe}"))?;
    if record.name != "ORIENTED_EDGE" {
        bail!("#{oe} is not ORIENTED_EDGE");
    }
    let Parameter::List(params) = &mut record.parameter else {
        bail!("ORIENTED_EDGE #{oe} params invalid");
    };
    if params.len() < 5 {
        bail!("ORIENTED_EDGE #{oe} params incomplete");
    }
    params[3] = entity_ref(edge);
    params[4] = Parameter::Enumeration(if forward { "T" } else { "F" }.to_string());
    Ok(())
}

pub(super) fn chain_apply_weld(
    graph: &mut GraphEditor<'_>,
    faces: &HashSet<u64>,
    vertex_map: &HashMap<u64, u64>,
    edge_map: &HashMap<u64, u64>,
) -> Result<()> {
    let mut edges = HashSet::<u64>::new();
    let mut oriented = HashSet::<u64>::new();
    for &face in faces {
        edges.extend(graph.face_edges_ordered(face)?);
        oriented.extend(chain_face_oriented_edges(graph, face)?);
    }

    for edge in edges {
        let [va, vb] = graph.edge_vertices(edge)?;
        let nva = vertex_map.get(&va).copied().unwrap_or(va);
        let nvb = vertex_map.get(&vb).copied().unwrap_or(vb);
        if (nva, nvb) != (va, vb) {
            chain_set_edge_vertices(graph, edge, nva, nvb)?;
        }
    }

    for oe in oriented {
        let (old_edge, forward) = chain_oriented_edge_info(graph, oe)?;
        let Some(&canonical) = edge_map.get(&old_edge) else {
            continue;
        };
        let [ova, ovb] = graph.edge_vertices(old_edge)?;
        let (mut start, mut end) = if forward { (ova, ovb) } else { (ovb, ova) };
        start = vertex_map.get(&start).copied().unwrap_or(start);
        end = vertex_map.get(&end).copied().unwrap_or(end);

        let [mut cva, mut cvb] = graph.edge_vertices(canonical)?;
        cva = vertex_map.get(&cva).copied().unwrap_or(cva);
        cvb = vertex_map.get(&cvb).copied().unwrap_or(cvb);

        let new_forward = if (start, end) == (cva, cvb) {
            true
        } else if (start, end) == (cvb, cva) {
            false
        } else {
            bail!("ORIENTED_EDGE #{oe} traversal cannot map seam edge #{old_edge} -> #{canonical}");
        };
        chain_set_oriented_edge(graph, oe, canonical, new_forward)?;
    }
    Ok(())
}
