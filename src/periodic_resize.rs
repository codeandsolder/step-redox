use crate::math3::{add, dot, norm, scale};
use crate::step_graph::{
    ReferenceGraph, build_index, entity_id, entity_ref_value, simple_record, visit_entity_refs,
};
use anyhow::{Result, anyhow, bail};
use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

use crate::instances::{StyleRef, collect_styles_by_target};
use crate::periodic_bodies::PeriodicBodyPattern;
use crate::periodic_chains::PeriodicChainPattern;

mod graph_editor;
use graph_editor::{GraphEditor, SourceLoop};

const COORD_TOL_MM: f64 = 1.0e-7;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PeriodicBodyResizeStats {
    pub old_sites: usize,
    pub new_sites: usize,
    pub canonical_site: usize,
    pub cloned_repeat_faces: usize,
    pub cloned_positive_fixed_faces: usize,
    pub rebuilt_stretch_faces: usize,
    pub new_stretch_edges: usize,
    pub added_entities: usize,
}

#[derive(Debug, Clone, Copy)]
struct SsRow {
    center: f64,
    span: f64,
    edge: u64,
}

struct BodyResizeContext<'a> {
    graph: GraphEditor<'a>,
    shell: u64,
    shell_faces: Vec<u64>,
    axis: [f64; 3],
    pitch: f64,
    stretch_faces: HashSet<u64>,
    fixed_faces: HashSet<u64>,
    repeat_faces: HashSet<u64>,
    edge_faces: HashMap<u64, Vec<u64>>,
    site_faces: Vec<Vec<u64>>,
    right_fixed: HashSet<u64>,
    left_fixed: HashSet<u64>,
    source_loops: HashMap<u64, Vec<SourceLoop>>,
}

impl<'a> BodyResizeContext<'a> {
    fn new(entities: &'a mut Vec<EntityInstance>, body: &PeriodicBodyPattern) -> Result<Self> {
        let old_sites = body.sites;
        if body.repeat_face_families.is_empty()
            || body
                .repeat_face_families
                .iter()
                .any(|family| family.face_ids.len() != old_sites)
        {
            bail!("periodic body face families do not all span every site");
        }

        let axis = normalize(body.axis).ok_or_else(|| anyhow!("periodic body axis is zero"))?;
        let pitch = body.pitch_mm;
        if !pitch.is_finite() || pitch <= COORD_TOL_MM {
            bail!("periodic body pitch is invalid");
        }

        let graph = GraphEditor::new(entities);
        let solid = body.solid_id;
        let shell = graph
            .simple_record(solid)
            .and_then(|record| {
                list_params(record).and_then(|params| {
                    params
                        .iter()
                        .filter_map(entity_ref_value)
                        .find(|id| graph.entity_type(*id) == Some("CLOSED_SHELL"))
                })
            })
            .ok_or_else(|| anyhow!("periodic body solid #{solid} has no CLOSED_SHELL"))?;
        let shell_faces = graph.shell_faces(shell)?;

        let stretch_faces = body
            .stretch_face_ids
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let fixed_faces = body.fixed_face_ids.iter().copied().collect::<HashSet<_>>();
        let repeat_faces = body
            .repeat_face_families
            .iter()
            .flat_map(|family| family.face_ids.iter().copied())
            .collect::<HashSet<_>>();
        let housing_faces = repeat_faces
            .iter()
            .chain(stretch_faces.iter())
            .chain(fixed_faces.iter())
            .copied()
            .collect::<HashSet<_>>();
        let edge_faces = graph.edge_faces(&housing_faces)?;

        let site_faces = (0..old_sites)
            .map(|site| {
                body.repeat_face_families
                    .iter()
                    .map(|family| family.face_ids[site])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let site_projection = site_faces
            .iter()
            .map(|faces| -> Result<f64> {
                let total = faces
                    .iter()
                    .map(|face| graph.face_center(*face).map(|center| dot(center, axis)))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .sum::<f64>();
                Ok(total / faces.len().max(1) as f64)
            })
            .collect::<Result<Vec<_>>>()?;
        let right_threshold = pitch.mul_add(0.5, site_projection[old_sites - 1]);

        let mut right_fixed = HashSet::new();
        let mut left_fixed = HashSet::new();
        for &face in &fixed_faces {
            let projected = dot(graph.face_center(face)?, axis);
            if projected > right_threshold - COORD_TOL_MM {
                right_fixed.insert(face);
            } else {
                left_fixed.insert(face);
            }
        }
        if right_fixed.is_empty() || left_fixed.is_empty() {
            bail!(
                "could not split fixed faces into negative/positive caps: left={} right={}",
                left_fixed.len(),
                right_fixed.len()
            );
        }

        let source_loops = stretch_faces
            .iter()
            .map(|&face| Ok((face, graph.source_loops(face)?)))
            .collect::<Result<HashMap<_, _>>>()?;

        Ok(Self {
            graph,
            shell,
            shell_faces,
            axis,
            pitch,
            stretch_faces,
            fixed_faces,
            repeat_faces,
            edge_faces,
            site_faces,
            right_fixed,
            left_fixed,
            source_loops,
        })
    }
}

fn insert_stretch_edge(
    stretch_edges: &mut HashMap<u64, HashSet<u64>>,
    face: u64,
    edge: u64,
) -> Result<()> {
    stretch_edges
        .get_mut(&face)
        .ok_or_else(|| anyhow!("missing target edge set for stretch face #{face}"))?
        .insert(edge);
    Ok(())
}

/// Expand a proven 1-D periodic body at its positive-axis end.
///
/// This deliberately handles only growth for the first production increment.
/// The graph transformation is exact: repeat-cell faces are cloned by rigid
/// translation and the spanning planar faces receive freshly traced boundary
/// loops over the target shared-edge graph.
///
/// # Errors
/// Returns an error if the periodic-body proof is incomplete, the requested expansion is invalid, or the STEP graph cannot be rewritten safely.
pub fn expand_periodic_body_positive(
    entities: &mut Vec<EntityInstance>,
    body: &PeriodicBodyPattern,
    new_sites: usize,
) -> Result<PeriodicBodyResizeStats> {
    let old_sites = body.sites;
    if new_sites <= old_sites {
        bail!("periodic body expansion requires new_sites > old_sites");
    }
    if old_sites < 2 {
        bail!("periodic body does not contain enough repeated structure");
    }

    let BodyResizeContext {
        mut graph,
        shell,
        shell_faces,
        axis,
        pitch,
        stretch_faces,
        fixed_faces,
        repeat_faces,
        edge_faces,
        site_faces,
        right_fixed,
        left_fixed,
        source_loops,
    } = BodyResizeContext::new(entities, body)?;
    let extra = new_sites - old_sites;
    let delta_total = scale(axis, extra as f64 * pitch);
    let canonical_site = old_sites / 2;
    let canonical_faces = site_faces[canonical_site].clone();

    let cap_map = graph.clone_descendants(&right_fixed, delta_total)?;
    let new_right_fixed = right_fixed
        .iter()
        .map(|face| {
            cap_map
                .get(face)
                .copied()
                .ok_or_else(|| anyhow!("positive cap clone missing face #{face}"))
        })
        .collect::<Result<HashSet<_>>>()?;

    let mut site_maps = HashMap::<usize, HashMap<u64, u64>>::new();
    let mut new_site_faces = Vec::<Vec<u64>>::new();
    for site in old_sites..new_sites {
        let delta = scale(
            axis,
            (site as isize - canonical_site as isize) as f64 * pitch,
        );
        let mapping = graph.clone_descendants(&canonical_faces.iter().copied().collect(), delta)?;
        let faces = canonical_faces
            .iter()
            .map(|face| {
                mapping
                    .get(face)
                    .copied()
                    .ok_or_else(|| anyhow!("repeat-cell clone missing face #{face}"))
            })
            .collect::<Result<Vec<_>>>()?;
        new_site_faces.push(faces);
        site_maps.insert(site, mapping);
    }

    let mut target_patch_faces = site_faces
        .iter()
        .flat_map(|faces| faces.iter().copied())
        .collect::<HashSet<_>>();
    for faces in &new_site_faces {
        target_patch_faces.extend(faces.iter().copied());
    }
    target_patch_faces.extend(left_fixed.iter().copied());
    target_patch_faces.extend(new_right_fixed.iter().copied());

    let mut coord_vertices = HashMap::<[i64; 3], Vec<u64>>::new();
    for face in target_patch_faces {
        for vertex in graph.face_vertices(face)? {
            coord_vertices
                .entry(quantize_coord(graph.vertex_coord(vertex)?))
                .or_default()
                .push(vertex);
        }
    }
    for vertices in coord_vertices.values_mut() {
        vertices.sort_unstable();
        vertices.dedup();
    }

    let find_vertex = |point: [f64; 3]| -> Result<u64> {
        let Some(vertices) = coord_vertices.get(&quantize_coord(point)) else {
            bail!("no target vertex at {point:?}");
        };
        if vertices.len() != 1 {
            bail!("ambiguous target vertex at {point:?}: {vertices:?}");
        }
        Ok(vertices[0])
    };

    let mut stretch_edges = stretch_faces
        .iter()
        .map(|&face| (face, HashSet::<u64>::new()))
        .collect::<HashMap<_, _>>();

    let canonical_set = canonical_faces.iter().copied().collect::<HashSet<_>>();
    let mut canonical_boundary = Vec::<(u64, u64)>::new();

    for (&edge, faces) in &edge_faces {
        let repeated = faces.iter().any(|face| repeat_faces.contains(face));
        if repeated {
            for stretch in faces.iter().filter(|face| stretch_faces.contains(face)) {
                insert_stretch_edge(&mut stretch_edges, *stretch, edge)?;
            }
        }

        let is_canonical = faces.iter().any(|face| canonical_set.contains(face));
        if is_canonical
            && let Some(stretch) = faces.iter().find(|face| stretch_faces.contains(face))
        {
            canonical_boundary.push((edge, *stretch));
        }
    }

    for mapping in site_maps.values() {
        for &(source_edge, stretch) in &canonical_boundary {
            let target_edge = mapping
                .get(&source_edge)
                .copied()
                .ok_or_else(|| anyhow!("cell clone missing boundary edge #{source_edge}"))?;
            insert_stretch_edge(&mut stretch_edges, stretch, target_edge)?;
        }
    }

    // Fixed-cap to stretch-face interfaces.
    for (&edge, faces) in &edge_faces {
        let fixed = faces
            .iter()
            .find(|face| fixed_faces.contains(face))
            .copied();
        let Some(fixed) = fixed else {
            continue;
        };
        let stretches = faces
            .iter()
            .filter(|face| stretch_faces.contains(face))
            .copied()
            .collect::<Vec<_>>();
        if stretches.is_empty() {
            continue;
        }
        let target_edge = if right_fixed.contains(&fixed) {
            cap_map
                .get(&edge)
                .copied()
                .ok_or_else(|| anyhow!("positive cap clone missing interface edge #{edge}"))?
        } else {
            edge
        };
        for stretch in stretches {
            insert_stretch_edge(&mut stretch_edges, stretch, target_edge)?;
        }
    }

    // Stretch-to-stretch grammar.
    let mut ss_pairs = HashMap::<(u64, u64), Vec<SsRow>>::new();
    for (&edge, faces) in &edge_faces {
        if faces.len() != 2 || !faces.iter().all(|face| stretch_faces.contains(face)) {
            continue;
        }
        let mut pair = [faces[0], faces[1]];
        pair.sort_unstable();
        let (center, span) = graph.edge_center_span(edge, axis)?;
        ss_pairs
            .entry((pair[0], pair[1]))
            .or_default()
            .push(SsRow { center, span, edge });
    }

    let mut new_stretch_edges = 0usize;
    for (pair, mut rows) in ss_pairs {
        rows.sort_by(|a, b| a.center.total_cmp(&b.center));
        let full_threshold = (old_sites.saturating_sub(1)) as f64 * pitch;
        let full = rows
            .iter()
            .copied()
            .filter(|row| row.span > full_threshold)
            .collect::<Vec<_>>();
        let short = rows
            .iter()
            .copied()
            .filter(|row| row.span <= full_threshold)
            .collect::<Vec<_>>();

        for row in full {
            let [mut va, mut vb] = graph.edge_vertices(row.edge)?;
            let mut pa = graph.vertex_coord(va)?;
            let mut pb = graph.vertex_coord(vb)?;
            if dot(pa, axis) > dot(pb, axis) {
                std::mem::swap(&mut va, &mut vb);
                std::mem::swap(&mut pa, &mut pb);
            }
            let target_right = add(pb, delta_total);
            let nv = find_vertex(target_right)?;
            let edge = graph.make_edge_like(row.edge, va, nv)?;
            new_stretch_edges += 1;
            insert_stretch_edge(&mut stretch_edges, pair.0, edge)?;
            insert_stretch_edge(&mut stretch_edges, pair.1, edge)?;
        }

        if short.is_empty() {
            continue;
        }
        if short.len() != old_sites + 1 {
            bail!(
                "unexpected short stretch-edge grammar for faces {pair:?}: {} rows, expected {}",
                short.len(),
                old_sites + 1
            );
        }
        let left_end = short[0];
        let right_end = short[short.len() - 1];
        let gaps = &short[1..short.len() - 1];

        for row in std::iter::once(&left_end).chain(gaps.iter()) {
            insert_stretch_edge(&mut stretch_edges, pair.0, row.edge)?;
            insert_stretch_edge(&mut stretch_edges, pair.1, row.edge)?;
        }

        let prototype = gaps
            .last()
            .copied()
            .ok_or_else(|| anyhow!("stretch grammar contains no inter-site gap"))?;
        for step in 1..=extra {
            let delta = scale(axis, step as f64 * pitch);
            let [va, vb] = graph.edge_vertices(prototype.edge)?;
            let pva = add(graph.vertex_coord(va)?, delta);
            let pvb = add(graph.vertex_coord(vb)?, delta);
            let nva = find_vertex(pva)?;
            let nvb = find_vertex(pvb)?;
            let edge = graph.make_edge_like(prototype.edge, nva, nvb)?;
            new_stretch_edges += 1;
            insert_stretch_edge(&mut stretch_edges, pair.0, edge)?;
            insert_stretch_edge(&mut stretch_edges, pair.1, edge)?;
        }

        let [va, vb] = graph.edge_vertices(right_end.edge)?;
        let nva = find_vertex(add(graph.vertex_coord(va)?, delta_total))?;
        let nvb = find_vertex(add(graph.vertex_coord(vb)?, delta_total))?;
        let edge = graph.make_edge_like(right_end.edge, nva, nvb)?;
        new_stretch_edges += 1;
        insert_stretch_edge(&mut stretch_edges, pair.0, edge)?;
        insert_stretch_edge(&mut stretch_edges, pair.1, edge)?;
    }

    for &face in &body.stretch_face_ids {
        let edges = stretch_edges
            .get(&face)
            .ok_or_else(|| anyhow!("missing target edge set for stretch face #{face}"))?;
        let loops = source_loops
            .get(&face)
            .ok_or_else(|| anyhow!("missing source loop semantics for stretch face #{face}"))?;
        graph.rebuild_face_bounds(face, edges, loops)?;
    }

    // Replace positive fixed faces in the shell, keep everything else, append
    // the newly-created repeat faces.
    let mut target_shell_faces =
        Vec::with_capacity(shell_faces.len() + extra * body.faces_per_site);
    for face in shell_faces {
        if right_fixed.contains(&face) {
            target_shell_faces.push(
                cap_map
                    .get(&face)
                    .copied()
                    .ok_or_else(|| anyhow!("cap clone missing shell face #{face}"))?,
            );
        } else {
            target_shell_faces.push(face);
        }
    }
    for faces in &new_site_faces {
        target_shell_faces.extend(faces.iter().copied());
    }
    graph.set_shell_faces(shell, &target_shell_faces)?;

    let added_entities = graph.added_entities();

    Ok(PeriodicBodyResizeStats {
        old_sites,
        new_sites,
        canonical_site,
        cloned_repeat_faces: extra * canonical_faces.len(),
        cloned_positive_fixed_faces: right_fixed.len(),
        rebuilt_stretch_faces: body.stretch_face_ids.len(),
        new_stretch_edges,
        added_entities,
    })
}

/// Shrink a proven 1-D periodic body at its positive-axis end.
///
/// The negative end and the first `new_sites` repeat cells remain in place.
/// The positive fixed cap is cloned inward, the removed repeat faces are
/// omitted from the `CLOSED_SHELL`, and the spanning-face boundary grammar is
/// rebuilt over the shortened cell/gap sequence.
///
/// # Errors
/// Returns an error if the periodic-body proof is incomplete, the requested shrink is invalid, or the STEP graph cannot be rewritten safely.
pub fn shrink_periodic_body_positive(
    entities: &mut Vec<EntityInstance>,
    body: &PeriodicBodyPattern,
    new_sites: usize,
) -> Result<PeriodicBodyResizeStats> {
    let old_sites = body.sites;
    if new_sites >= old_sites {
        bail!("periodic body shrink requires new_sites < old_sites");
    }
    // Instance/count semantic recovery currently requires at least four sites,
    // so do not emit an edit that the postcondition checker cannot prove.
    if new_sites < 4 {
        bail!("periodic body shrink currently requires at least 4 sites");
    }

    let BodyResizeContext {
        mut graph,
        shell,
        shell_faces,
        axis,
        pitch,
        stretch_faces,
        fixed_faces,
        repeat_faces: _,
        edge_faces,
        site_faces,
        right_fixed,
        left_fixed,
        source_loops,
    } = BodyResizeContext::new(entities, body)?;
    let removed = old_sites - new_sites;
    let delta_total = scale(axis, -(removed as f64) * pitch);
    let canonical_site = new_sites / 2;

    let cap_map = graph.clone_descendants(&right_fixed, delta_total)?;
    let new_right_fixed = right_fixed
        .iter()
        .map(|face| {
            cap_map
                .get(face)
                .copied()
                .ok_or_else(|| anyhow!("positive cap clone missing face #{face}"))
        })
        .collect::<Result<HashSet<_>>>()?;

    let kept_repeat_faces = site_faces[..new_sites]
        .iter()
        .flat_map(|faces| faces.iter().copied())
        .collect::<HashSet<_>>();
    let removed_repeat_faces = site_faces[new_sites..]
        .iter()
        .flat_map(|faces| faces.iter().copied())
        .collect::<HashSet<_>>();

    let mut target_patch_faces = kept_repeat_faces.clone();
    target_patch_faces.extend(left_fixed.iter().copied());
    target_patch_faces.extend(new_right_fixed.iter().copied());

    let mut coord_vertices = HashMap::<[i64; 3], Vec<u64>>::new();
    for face in target_patch_faces {
        for vertex in graph.face_vertices(face)? {
            coord_vertices
                .entry(quantize_coord(graph.vertex_coord(vertex)?))
                .or_default()
                .push(vertex);
        }
    }
    for vertices in coord_vertices.values_mut() {
        vertices.sort_unstable();
        vertices.dedup();
    }

    let find_vertex = |point: [f64; 3]| -> Result<u64> {
        let Some(vertices) = coord_vertices.get(&quantize_coord(point)) else {
            bail!("no target vertex at {point:?}");
        };
        if vertices.len() != 1 {
            bail!("ambiguous target vertex at {point:?}: {vertices:?}");
        }
        Ok(vertices[0])
    };

    let mut stretch_edges = stretch_faces
        .iter()
        .map(|&face| (face, HashSet::<u64>::new()))
        .collect::<HashMap<_, _>>();

    // Keep only repeat-cell interfaces that belong to surviving sites.
    for (&edge, faces) in &edge_faces {
        let repeated = faces.iter().any(|face| kept_repeat_faces.contains(face));
        if repeated {
            for stretch in faces.iter().filter(|face| stretch_faces.contains(face)) {
                insert_stretch_edge(&mut stretch_edges, *stretch, edge)?;
            }
        }
    }

    // Fixed-cap interfaces: negative cap stays in place; positive cap moves
    // inward by the removed pitch span.
    for (&edge, faces) in &edge_faces {
        let fixed = faces
            .iter()
            .find(|face| fixed_faces.contains(face))
            .copied();
        let Some(fixed) = fixed else {
            continue;
        };
        let stretches = faces
            .iter()
            .filter(|face| stretch_faces.contains(face))
            .copied()
            .collect::<Vec<_>>();
        if stretches.is_empty() {
            continue;
        }
        let target_edge = if right_fixed.contains(&fixed) {
            cap_map
                .get(&edge)
                .copied()
                .ok_or_else(|| anyhow!("positive cap clone missing interface edge #{edge}"))?
        } else {
            edge
        };
        for stretch in stretches {
            insert_stretch_edge(&mut stretch_edges, stretch, target_edge)?;
        }
    }

    // Rebuild the stretch-to-stretch grammar. Full-span rails receive a new
    // positive endpoint; gap runs keep only the first new_sites-1 gaps.
    let mut ss_pairs = HashMap::<(u64, u64), Vec<SsRow>>::new();
    for (&edge, faces) in &edge_faces {
        if faces.len() != 2 || !faces.iter().all(|face| stretch_faces.contains(face)) {
            continue;
        }
        let mut pair = [faces[0], faces[1]];
        pair.sort_unstable();
        let (center, span) = graph.edge_center_span(edge, axis)?;
        ss_pairs
            .entry((pair[0], pair[1]))
            .or_default()
            .push(SsRow { center, span, edge });
    }

    let mut new_stretch_edges = 0usize;
    for (pair, mut rows) in ss_pairs {
        rows.sort_by(|a, b| a.center.total_cmp(&b.center));
        let full_threshold = (old_sites.saturating_sub(1)) as f64 * pitch;
        let full = rows
            .iter()
            .copied()
            .filter(|row| row.span > full_threshold)
            .collect::<Vec<_>>();
        let short = rows
            .iter()
            .copied()
            .filter(|row| row.span <= full_threshold)
            .collect::<Vec<_>>();

        for row in full {
            let [mut va, mut vb] = graph.edge_vertices(row.edge)?;
            let mut pa = graph.vertex_coord(va)?;
            let mut pb = graph.vertex_coord(vb)?;
            if dot(pa, axis) > dot(pb, axis) {
                std::mem::swap(&mut va, &mut vb);
                std::mem::swap(&mut pa, &mut pb);
            }
            let target_right = add(pb, delta_total);
            let nv = find_vertex(target_right)?;
            let edge = graph.make_edge_like(row.edge, va, nv)?;
            new_stretch_edges += 1;
            insert_stretch_edge(&mut stretch_edges, pair.0, edge)?;
            insert_stretch_edge(&mut stretch_edges, pair.1, edge)?;
        }

        if short.is_empty() {
            continue;
        }
        if short.len() != old_sites + 1 {
            bail!(
                "unexpected short stretch-edge grammar for faces {pair:?}: {} rows, expected {}",
                short.len(),
                old_sites + 1
            );
        }
        let left_end = short[0];
        let right_end = short[short.len() - 1];
        let gaps = &short[1..short.len() - 1];

        insert_stretch_edge(&mut stretch_edges, pair.0, left_end.edge)?;
        insert_stretch_edge(&mut stretch_edges, pair.1, left_end.edge)?;
        for row in gaps.iter().take(new_sites.saturating_sub(1)) {
            insert_stretch_edge(&mut stretch_edges, pair.0, row.edge)?;
            insert_stretch_edge(&mut stretch_edges, pair.1, row.edge)?;
        }

        let [va, vb] = graph.edge_vertices(right_end.edge)?;
        let nva = find_vertex(add(graph.vertex_coord(va)?, delta_total))?;
        let nvb = find_vertex(add(graph.vertex_coord(vb)?, delta_total))?;
        let edge = graph.make_edge_like(right_end.edge, nva, nvb)?;
        new_stretch_edges += 1;
        insert_stretch_edge(&mut stretch_edges, pair.0, edge)?;
        insert_stretch_edge(&mut stretch_edges, pair.1, edge)?;
    }

    for &face in &body.stretch_face_ids {
        let edges = stretch_edges
            .get(&face)
            .ok_or_else(|| anyhow!("missing target edge set for stretch face #{face}"))?;
        let loops = source_loops
            .get(&face)
            .ok_or_else(|| anyhow!("missing source loop semantics for stretch face #{face}"))?;
        graph.rebuild_face_bounds(face, edges, loops)?;
    }

    let mut target_shell_faces =
        Vec::with_capacity(shell_faces.len().saturating_sub(removed_repeat_faces.len()));
    for face in shell_faces {
        if right_fixed.contains(&face) {
            target_shell_faces.push(
                cap_map
                    .get(&face)
                    .copied()
                    .ok_or_else(|| anyhow!("cap clone missing shell face #{face}"))?,
            );
        } else if removed_repeat_faces.contains(&face) {
            continue;
        } else {
            target_shell_faces.push(face);
        }
    }
    graph.set_shell_faces(shell, &target_shell_faces)?;

    let added_entities = graph.added_entities();

    Ok(PeriodicBodyResizeStats {
        old_sites,
        new_sites,
        canonical_site,
        cloned_repeat_faces: 0,
        cloned_positive_fixed_faces: right_fixed.len(),
        rebuilt_stretch_faces: body.stretch_face_ids.len(),
        new_stretch_edges,
        added_entities,
    })
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct PeriodicChainResizeStats {
    pub old_sites: usize,
    pub new_sites: usize,
    pub inserted_units: usize,
    pub removed_units: usize,
    pub unit_faces: usize,
    pub tail_faces: usize,
    pub seam_edge_pairs: usize,
    pub welded_vertices: usize,
    pub welded_edges: usize,
    pub rebuilt_stretch_faces: usize,
    pub new_stretch_edges: usize,
    pub added_entities: usize,
    pub pruned_entities: usize,
    pub cloned_style_items: usize,
    pub removed_style_items: usize,
    pub entity_delta: isize,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum ChainCurveKey {
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

struct ChainResizeContext<'a> {
    graph: GraphEditor<'a>,
    shell: u64,
    source_shell_faces: Vec<u64>,
    axis: [f64; 3],
    pitch: f64,
    stretch_faces: HashSet<u64>,
    styles_by_target: HashMap<u64, Vec<StyleRef>>,
    style_parents: HashMap<u64, Vec<u64>>,
    source_edge_faces: HashMap<u64, Vec<u64>>,
    source_loops: HashMap<u64, Vec<SourceLoop>>,
}

impl<'a> ChainResizeContext<'a> {
    fn new(
        entities: &'a mut Vec<EntityInstance>,
        chain: &PeriodicChainPattern,
        touched_roots: &HashSet<u64>,
    ) -> Result<Self> {
        validate_chain_editability(chain)?;

        let axis = normalize(chain.axis).ok_or_else(|| anyhow!("periodic chain axis is zero"))?;
        let pitch = chain.pitch_mm;
        if !pitch.is_finite() || pitch <= COORD_TOL_MM {
            bail!("periodic chain pitch is invalid");
        }

        let styles_by_target = collect_styles_by_target(entities);
        let preflight_index = build_index(entities);
        let touched_descendants =
            entity_descendant_closure(entities, &preflight_index, touched_roots)?;
        require_face_only_direct_styles(
            entities,
            &preflight_index,
            &touched_descendants,
            &styles_by_target,
        )?;
        let style_parents = collect_style_container_parents(
            entities,
            &preflight_index,
            &styles_by_target,
            &touched_descendants,
        )?;

        let graph = GraphEditor::new(entities);
        let shell = graph
            .simple_record(chain.solid_id)
            .and_then(|record| {
                list_params(record).and_then(|params| {
                    params
                        .iter()
                        .filter_map(entity_ref_value)
                        .find(|id| graph.entity_type(*id) == Some("CLOSED_SHELL"))
                })
            })
            .ok_or_else(|| {
                anyhow!(
                    "periodic chain solid #{} has no CLOSED_SHELL",
                    chain.solid_id
                )
            })?;
        let source_shell_faces = graph.shell_faces(shell)?;

        let stretch_faces = chain
            .stretch_face_ids
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let mut all_faces = HashSet::<u64>::new();
        for faces in &chain.site_face_ids {
            all_faces.extend(faces.iter().copied());
        }
        for faces in &chain.gap_face_ids {
            all_faces.extend(faces.iter().copied());
        }
        all_faces.extend(stretch_faces.iter().copied());
        all_faces.extend(chain.fixed_negative_face_ids.iter().copied());
        all_faces.extend(chain.fixed_positive_face_ids.iter().copied());
        let source_edge_faces = graph.edge_faces(&all_faces)?;
        let source_loops = stretch_faces
            .iter()
            .map(|&face| Ok((face, graph.source_loops(face)?)))
            .collect::<Result<HashMap<_, _>>>()?;

        Ok(Self {
            graph,
            shell,
            source_shell_faces,
            axis,
            pitch,
            stretch_faces,
            styles_by_target,
            style_parents,
            source_edge_faces,
            source_loops,
        })
    }
}

fn validate_chain_editability(chain: &PeriodicChainPattern) -> Result<()> {
    let old_sites = chain.sites;
    if !chain.read_only_proven || !chain.complete_partition {
        bail!("periodic chain is not fully proven");
    }
    if chain.nonmanifold_edges != 0 || chain.nonlocal_cross_site_edges != 0 {
        bail!(
            "periodic chain topology is not editable: nonmanifold={} nonlocal_cross_site={}",
            chain.nonmanifold_edges,
            chain.nonlocal_cross_site_edges
        );
    }
    if chain.adjacent_site_edge_counts.len() + 1 != old_sites
        || chain
            .adjacent_site_edge_counts
            .first()
            .is_some_and(|first| {
                chain
                    .adjacent_site_edge_counts
                    .iter()
                    .any(|count| count != first)
            })
    {
        bail!("periodic chain has inconsistent adjacent-site overlay seams");
    }
    if !chain.fixed_middle_face_ids.is_empty() {
        bail!("periodic chain contains fixed middle geometry");
    }
    Ok(())
}

struct ChainInterfaceSource<'a> {
    faces: &'a HashSet<u64>,
    mapping: Option<&'a HashMap<u64, u64>>,
}

struct ChainStretchRewrite<'a> {
    chain: &'a PeriodicChainPattern,
    axis: [f64; 3],
    pitch: f64,
    delta_total: [f64; 3],
    stretch_faces: &'a HashSet<u64>,
    source_edge_faces: &'a HashMap<u64, Vec<u64>>,
    source_loops: &'a HashMap<u64, Vec<SourceLoop>>,
    target_nonstretch: &'a HashSet<u64>,
    interfaces: Vec<ChainInterfaceSource<'a>>,
    vertex_map: &'a HashMap<u64, u64>,
    edge_map: &'a HashMap<u64, u64>,
}

impl ChainStretchRewrite<'_> {
    fn run(self, graph: &mut GraphEditor<'_>, prune_roots: &mut HashSet<u64>) -> Result<usize> {
        let target_faces_for_weld = self
            .target_nonstretch
            .iter()
            .chain(self.stretch_faces.iter())
            .copied()
            .collect::<HashSet<_>>();
        chain_apply_weld(
            graph,
            &target_faces_for_weld,
            self.vertex_map,
            self.edge_map,
        )?;

        let mut coord_vertices = HashMap::<[i64; 3], Vec<u64>>::new();
        for &face in self.target_nonstretch {
            for vertex in graph.face_vertices(face)? {
                coord_vertices
                    .entry(quantize_coord(graph.vertex_coord(vertex)?))
                    .or_default()
                    .push(*self.vertex_map.get(&vertex).unwrap_or(&vertex));
            }
        }
        for vertices in coord_vertices.values_mut() {
            vertices.sort_unstable();
            vertices.dedup();
        }
        let find_vertex = |point: [f64; 3]| -> Result<u64> {
            let Some(vertices) = coord_vertices.get(&quantize_coord(point)) else {
                bail!("no target chain vertex at {point:?}");
            };
            if vertices.len() != 1 {
                bail!("ambiguous target chain vertex at {point:?}: {vertices:?}");
            }
            Ok(vertices[0])
        };

        let mut stretch_edges = self
            .stretch_faces
            .iter()
            .map(|&face| (face, HashSet::<u64>::new()))
            .collect::<HashMap<_, _>>();
        for source in &self.interfaces {
            for (&edge, faces) in self.source_edge_faces {
                if !faces.iter().any(|face| source.faces.contains(face)) {
                    continue;
                }
                let target_stretch = faces
                    .iter()
                    .filter(|face| self.stretch_faces.contains(face))
                    .copied()
                    .collect::<Vec<_>>();
                if target_stretch.is_empty() {
                    continue;
                }
                let mut target_edge = source
                    .mapping
                    .and_then(|map| map.get(&edge).copied())
                    .unwrap_or(edge);
                target_edge = self
                    .edge_map
                    .get(&target_edge)
                    .copied()
                    .unwrap_or(target_edge);
                for stretch in target_stretch {
                    insert_stretch_edge(&mut stretch_edges, stretch, target_edge)?;
                }
            }
        }

        let mut stretch_rows = Vec::<(SsRow, [u64; 2])>::new();
        for (&edge, faces) in self.source_edge_faces {
            if faces.len() == 2 && faces.iter().all(|face| self.stretch_faces.contains(face)) {
                let mut pair = [faces[0], faces[1]];
                pair.sort_unstable();
                let (center, span) = graph.edge_center_span(edge, self.axis)?;
                stretch_rows.push((SsRow { center, span, edge }, pair));
            }
        }
        if stretch_rows.is_empty() {
            bail!("periodic-chain stretch grammar has no stretch/stretch edges");
        }

        let chain_span = self.chain.sites.saturating_sub(1) as f64 * self.pitch;
        let chain_mid = f64::midpoint(
            self.chain.site_centers_mm[0],
            self.chain.site_centers_mm[self.chain.sites - 1],
        );
        let mut new_stretch_edges = 0usize;
        for (row, pair) in stretch_rows {
            let [mut va, mut vb] = graph.edge_vertices(row.edge)?;
            let mut pa = graph.vertex_coord(va)?;
            let mut pb = graph.vertex_coord(vb)?;

            let target_edge = if row.span > chain_span {
                if dot(pa, self.axis) > dot(pb, self.axis) {
                    std::mem::swap(&mut va, &mut vb);
                    std::mem::swap(&mut pa, &mut pb);
                }
                let nv = find_vertex(add(pb, self.delta_total))?;
                new_stretch_edges += 1;
                graph.make_edge_like(row.edge, va, nv)?
            } else if row.center < chain_mid {
                row.edge
            } else {
                let nva = find_vertex(add(pa, self.delta_total))?;
                let nvb = find_vertex(add(pb, self.delta_total))?;
                new_stretch_edges += 1;
                graph.make_edge_like(row.edge, nva, nvb)?
            };
            insert_stretch_edge(&mut stretch_edges, pair[0], target_edge)?;
            insert_stretch_edge(&mut stretch_edges, pair[1], target_edge)?;
        }

        for &face in &self.chain.stretch_face_ids {
            let edges = stretch_edges
                .get(&face)
                .ok_or_else(|| anyhow!("missing target edges for chain stretch face #{face}"))?;
            let loops = self
                .source_loops
                .get(&face)
                .ok_or_else(|| anyhow!("missing source loops for chain stretch face #{face}"))?;
            let replaced_bounds = graph.rebuild_face_bounds(face, edges, loops)?;
            prune_roots.extend(replaced_bounds);
        }
        Ok(new_stretch_edges)
    }
}

struct ChainWeldPlan {
    seam_pairs: Vec<(u64, u64)>,
    vertex_map: HashMap<u64, u64>,
    edge_map: HashMap<u64, u64>,
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

    fn seed_prune_roots(&self, prune_roots: &mut HashSet<u64>) {
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

fn plan_chain_expansion_welds(
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

fn plan_chain_shrink_welds(
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

/// Expand a proven fused-solid periodic chain at its positive-axis end.
///
/// The chain detector has already proved a complete manifold partition into
/// per-site patches, inter-site gap patches, spanning faces, and symmetric
/// fixed end regions. Growth clones one generic (gap + interior-site) unit per
/// added site, translates the positive two-site/end-cap tail, welds supported
/// seam edges by geometric identity, and rebuilds only the spanning face loops.
///
/// # Errors
/// Returns an error if the periodic-chain proof is incomplete, the requested expansion is invalid, or the STEP graph cannot be rewritten safely.
pub fn expand_periodic_chain_positive(
    entities: &mut Vec<EntityInstance>,
    chain: &PeriodicChainPattern,
    new_sites: usize,
) -> Result<PeriodicChainResizeStats> {
    let old_sites = chain.sites;

    if new_sites <= old_sites {
        bail!("periodic chain expansion requires new_sites > old_sites");
    }
    if old_sites < 6
        || chain.site_face_ids.len() != old_sites
        || chain.gap_face_ids.len() + 1 != old_sites
    {
        bail!("periodic chain does not have the expected 1-D site/gap structure");
    }

    let prefix_site_end = old_sites - 3;
    let unit_gap_index = old_sites - 4;
    let unit_site_index = old_sites - 3;
    let tail_gap0 = old_sites - 3;
    let tail_site0 = old_sites - 2;
    let tail_gap1 = old_sites - 2;
    let tail_site1 = old_sites - 1;

    let mut prefix_sites = HashSet::<u64>::new();
    for faces in &chain.site_face_ids[..=prefix_site_end] {
        prefix_sites.extend(faces.iter().copied());
    }
    let mut prefix_gaps = HashSet::<u64>::new();
    for faces in &chain.gap_face_ids[..unit_gap_index] {
        prefix_gaps.extend(faces.iter().copied());
    }
    let prefix_faces = prefix_sites
        .iter()
        .chain(prefix_gaps.iter())
        .chain(chain.fixed_negative_face_ids.iter())
        .copied()
        .collect::<HashSet<_>>();
    let unit_faces = chain.gap_face_ids[unit_gap_index]
        .iter()
        .chain(chain.site_face_ids[unit_site_index].iter())
        .copied()
        .collect::<HashSet<_>>();
    let tail_faces = chain.gap_face_ids[tail_gap0]
        .iter()
        .chain(chain.site_face_ids[tail_site0].iter())
        .chain(chain.gap_face_ids[tail_gap1].iter())
        .chain(chain.site_face_ids[tail_site1].iter())
        .chain(chain.fixed_positive_face_ids.iter())
        .copied()
        .collect::<HashSet<_>>();
    let mut prune_roots = tail_faces.clone();

    if prefix_faces.iter().any(|face| tail_faces.contains(face))
        || unit_faces.iter().any(|face| tail_faces.contains(face))
    {
        bail!("periodic chain positive tail overlaps kept/insertion geometry");
    }

    let clone_roots = unit_faces
        .iter()
        .chain(tail_faces.iter())
        .copied()
        .collect::<HashSet<_>>();
    let ChainResizeContext {
        mut graph,
        shell,
        source_shell_faces,
        axis,
        pitch,
        stretch_faces,
        styles_by_target,
        style_parents,
        source_edge_faces,
        source_loops,
    } = ChainResizeContext::new(entities, chain, &clone_roots)?;
    let extra = new_sites - old_sites;
    let delta_total = scale(axis, extra as f64 * pitch);
    let mut cloned_style_items = 0usize;
    let mut removed_style_items = 0usize;

    let mut unit_maps = Vec::<HashMap<u64, u64>>::new();
    let mut unit_face_sets = Vec::<HashSet<u64>>::new();
    for step in 1..=extra {
        let mapping = graph.clone_descendants(&unit_faces, scale(axis, step as f64 * pitch))?;
        cloned_style_items +=
            graph.clone_face_styles(&styles_by_target, &style_parents, &mapping)?;
        let mapped_faces = unit_faces
            .iter()
            .map(|face| {
                mapping
                    .get(face)
                    .copied()
                    .ok_or_else(|| anyhow!("inserted unit clone missing face #{face}"))
            })
            .collect::<Result<HashSet<_>>>()?;
        unit_face_sets.push(mapped_faces);
        unit_maps.push(mapping);
    }

    let tail_map = graph.clone_descendants(&tail_faces, delta_total)?;
    cloned_style_items += graph.clone_face_styles(&styles_by_target, &style_parents, &tail_map)?;
    let target_tail = tail_faces
        .iter()
        .map(|face| {
            tail_map
                .get(face)
                .copied()
                .ok_or_else(|| anyhow!("positive tail clone missing face #{face}"))
        })
        .collect::<Result<HashSet<_>>>()?;

    let weld_plan =
        plan_chain_expansion_welds(&graph, chain, &source_edge_faces, &unit_maps, &tail_map)?;
    weld_plan.seed_prune_roots(&mut prune_roots);
    let seam_edge_pairs = weld_plan.seam_pairs.len();
    let vertex_map = &weld_plan.vertex_map;
    let edge_map = &weld_plan.edge_map;

    let kept_source_faces = prefix_faces
        .iter()
        .chain(unit_faces.iter())
        .copied()
        .collect::<HashSet<_>>();
    let mut target_nonstretch = kept_source_faces.clone();
    target_nonstretch.extend(target_tail.iter().copied());
    for faces in &unit_face_sets {
        target_nonstretch.extend(faces.iter().copied());
    }

    let mut interfaces = Vec::with_capacity(unit_maps.len() + 2);
    interfaces.push(ChainInterfaceSource {
        faces: &kept_source_faces,
        mapping: None,
    });
    for mapping in &unit_maps {
        interfaces.push(ChainInterfaceSource {
            faces: &unit_faces,
            mapping: Some(mapping),
        });
    }
    interfaces.push(ChainInterfaceSource {
        faces: &tail_faces,
        mapping: Some(&tail_map),
    });
    let new_stretch_edges = ChainStretchRewrite {
        chain,
        axis,
        pitch,
        delta_total,
        stretch_faces: &stretch_faces,
        source_edge_faces: &source_edge_faces,
        source_loops: &source_loops,
        target_nonstretch: &target_nonstretch,
        interfaces,
        vertex_map,
        edge_map,
    }
    .run(&mut graph, &mut prune_roots)?;

    let mut target_shell_faces =
        Vec::with_capacity(source_shell_faces.len() + extra * unit_faces.len());
    for face in source_shell_faces {
        if tail_faces.contains(&face) {
            target_shell_faces.push(
                tail_map
                    .get(&face)
                    .copied()
                    .ok_or_else(|| anyhow!("translated tail missing shell face #{face}"))?,
            );
        } else {
            target_shell_faces.push(face);
        }
    }
    for mapping in &unit_maps {
        let mut faces = unit_faces.iter().copied().collect::<Vec<_>>();
        faces.sort_unstable();
        for face in faces {
            target_shell_faces.push(
                mapping
                    .get(&face)
                    .copied()
                    .ok_or_else(|| anyhow!("inserted unit missing shell face #{face}"))?,
            );
        }
    }
    graph.set_shell_faces(shell, &target_shell_faces)?;

    removed_style_items +=
        graph.remove_face_styles(&styles_by_target, &style_parents, &tail_faces)?;
    let pruned_entities = graph.prune_unreachable_descendants(&prune_roots)?;
    let added_entities = graph.added_entities();
    let entity_delta = graph.entity_delta();
    Ok(PeriodicChainResizeStats {
        old_sites,
        new_sites,
        inserted_units: extra,
        removed_units: 0,
        unit_faces: unit_faces.len(),
        tail_faces: tail_faces.len(),
        seam_edge_pairs,
        welded_vertices: vertex_map.len(),
        welded_edges: edge_map.len(),
        rebuilt_stretch_faces: stretch_faces.len(),
        new_stretch_edges,
        added_entities,
        pruned_entities,
        cloned_style_items,
        removed_style_items,
        entity_delta,
    })
}

/// Shrink a proven fused-solid periodic chain at its positive-axis end.
///
/// Shrinking keeps the negative prefix fixed, removes whole periodic units
/// immediately before the proven positive tail, clones/translates that tail
/// toward the negative end, welds the new site-gap seam, and rebuilds only the
/// spanning face loops. No booleans or tessellation are used.
///
/// # Errors
/// Returns an error if the periodic-chain proof is incomplete, the requested shrink is invalid, or the STEP graph cannot be rewritten safely.
pub fn shrink_periodic_chain_positive(
    entities: &mut Vec<EntityInstance>,
    chain: &PeriodicChainPattern,
    new_sites: usize,
) -> Result<PeriodicChainResizeStats> {
    let old_sites = chain.sites;

    if new_sites >= old_sites {
        bail!("periodic chain shrink requires new_sites < old_sites");
    }
    if new_sites < 6
        || chain.site_face_ids.len() != old_sites
        || chain.gap_face_ids.len() + 1 != old_sites
    {
        bail!("periodic chain does not have the expected shrinkable 1-D site/gap structure");
    }

    let kept_last_site = new_sites - 3;
    let kept_last_gap = new_sites - 4;
    let remove_site_start = new_sites - 2;
    let remove_gap_start = new_sites - 3;
    let tail_gap0 = old_sites - 3;
    let tail_site0 = old_sites - 2;
    let tail_gap1 = old_sites - 2;
    let tail_site1 = old_sites - 1;

    let mut kept_sites = HashSet::<u64>::new();
    for faces in &chain.site_face_ids[..=kept_last_site] {
        kept_sites.extend(faces.iter().copied());
    }
    let mut kept_gaps = HashSet::<u64>::new();
    for faces in &chain.gap_face_ids[..=kept_last_gap] {
        kept_gaps.extend(faces.iter().copied());
    }
    let kept_faces = kept_sites
        .iter()
        .chain(kept_gaps.iter())
        .chain(chain.fixed_negative_face_ids.iter())
        .copied()
        .collect::<HashSet<_>>();

    let mut remove_faces = HashSet::<u64>::new();
    for faces in &chain.site_face_ids[remove_site_start..tail_site0] {
        remove_faces.extend(faces.iter().copied());
    }
    for faces in &chain.gap_face_ids[remove_gap_start..tail_gap0] {
        remove_faces.extend(faces.iter().copied());
    }
    let tail_faces = chain.gap_face_ids[tail_gap0]
        .iter()
        .chain(chain.site_face_ids[tail_site0].iter())
        .chain(chain.gap_face_ids[tail_gap1].iter())
        .chain(chain.site_face_ids[tail_site1].iter())
        .chain(chain.fixed_positive_face_ids.iter())
        .copied()
        .collect::<HashSet<_>>();
    let unit_faces =
        chain.gap_face_ids[remove_gap_start].len() + chain.site_face_ids[remove_site_start].len();

    if remove_faces.is_empty()
        || kept_faces.iter().any(|face| tail_faces.contains(face))
        || remove_faces
            .iter()
            .any(|face| kept_faces.contains(face) || tail_faces.contains(face))
    {
        bail!("periodic chain shrink partition overlaps or removes no periodic unit");
    }

    let touched_roots = remove_faces
        .iter()
        .chain(tail_faces.iter())
        .copied()
        .collect::<HashSet<_>>();
    let ChainResizeContext {
        mut graph,
        shell,
        source_shell_faces,
        axis,
        pitch,
        stretch_faces,
        styles_by_target,
        style_parents,
        source_edge_faces,
        source_loops,
    } = ChainResizeContext::new(entities, chain, &touched_roots)?;
    let removed_units = old_sites - new_sites;
    let delta_total = scale(axis, -(removed_units as f64) * pitch);
    let mut cloned_style_items = 0usize;
    let mut removed_style_items = 0usize;

    let tail_map = graph.clone_descendants(&tail_faces, delta_total)?;
    cloned_style_items += graph.clone_face_styles(&styles_by_target, &style_parents, &tail_map)?;
    let target_tail = tail_faces
        .iter()
        .map(|face| {
            tail_map
                .get(face)
                .copied()
                .ok_or_else(|| anyhow!("translated tail clone missing face #{face}"))
        })
        .collect::<Result<HashSet<_>>>()?;

    let weld_plan =
        plan_chain_shrink_welds(&graph, chain, &source_edge_faces, &tail_map, new_sites)?;
    let mut prune_roots = remove_faces.clone();
    prune_roots.extend(tail_faces.iter().copied());
    weld_plan.seed_prune_roots(&mut prune_roots);
    let seam_edge_pairs = weld_plan.seam_pairs.len();
    let vertex_map = &weld_plan.vertex_map;
    let edge_map = &weld_plan.edge_map;

    let mut target_nonstretch = kept_faces.clone();
    target_nonstretch.extend(target_tail.iter().copied());

    let interfaces = vec![
        ChainInterfaceSource {
            faces: &kept_faces,
            mapping: None,
        },
        ChainInterfaceSource {
            faces: &tail_faces,
            mapping: Some(&tail_map),
        },
    ];
    let new_stretch_edges = ChainStretchRewrite {
        chain,
        axis,
        pitch,
        delta_total,
        stretch_faces: &stretch_faces,
        source_edge_faces: &source_edge_faces,
        source_loops: &source_loops,
        target_nonstretch: &target_nonstretch,
        interfaces,
        vertex_map,
        edge_map,
    }
    .run(&mut graph, &mut prune_roots)?;

    let mut target_shell_faces = Vec::with_capacity(source_shell_faces.len());
    for face in source_shell_faces {
        if remove_faces.contains(&face) {
            continue;
        }
        if tail_faces.contains(&face) {
            target_shell_faces.push(
                tail_map
                    .get(&face)
                    .copied()
                    .ok_or_else(|| anyhow!("translated tail missing shell face #{face}"))?,
            );
        } else {
            target_shell_faces.push(face);
        }
    }
    graph.set_shell_faces(shell, &target_shell_faces)?;

    let removed_face_roots = remove_faces
        .iter()
        .chain(tail_faces.iter())
        .copied()
        .collect::<HashSet<_>>();
    removed_style_items +=
        graph.remove_face_styles(&styles_by_target, &style_parents, &removed_face_roots)?;
    let pruned_entities = graph.prune_unreachable_descendants(&prune_roots)?;
    let added_entities = graph.added_entities();
    let entity_delta = graph.entity_delta();
    Ok(PeriodicChainResizeStats {
        old_sites,
        new_sites,
        inserted_units: 0,
        removed_units,
        unit_faces,
        tail_faces: tail_faces.len(),
        seam_edge_pairs,
        welded_vertices: vertex_map.len(),
        welded_edges: edge_map.len(),
        rebuilt_stretch_faces: stretch_faces.len(),
        new_stretch_edges,
        added_entities,
        pruned_entities,
        cloned_style_items,
        removed_style_items,
        entity_delta,
    })
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

fn chain_curve_key(graph: &GraphEditor<'_>, edge: u64) -> Result<ChainCurveKey> {
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

fn collect_style_container_parents(
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    styles_by_target: &HashMap<u64, Vec<StyleRef>>,
    relevant_targets: &HashSet<u64>,
) -> Result<HashMap<u64, Vec<u64>>> {
    let relevant_styles = relevant_targets
        .iter()
        .flat_map(|target| styles_by_target.get(target).into_iter().flatten())
        .map(|style| style.id)
        .collect::<HashSet<_>>();
    let mut inbound = HashMap::<u64, Vec<u64>>::with_capacity(relevant_styles.len());
    for entity in entities {
        let parent = entity_id(entity);
        visit_entity_refs(entity, &mut |child| {
            if relevant_styles.contains(&child) {
                inbound.entry(child).or_default().push(parent);
            }
        });
    }
    let mut out = HashMap::<u64, Vec<u64>>::new();
    for style in relevant_targets
        .iter()
        .flat_map(|target| styles_by_target.get(target).into_iter().flatten())
    {
        let mut parents = inbound.get(&style.id).cloned().unwrap_or_default();
        parents.sort_unstable();
        parents.dedup();
        if parents.is_empty() {
            bail!("STYLED_ITEM #{} has no presentation container", style.id);
        }
        for &parent in &parents {
            let Some(&idx) = index.get(&parent) else {
                bail!(
                    "STYLED_ITEM #{} references missing parent #{parent}",
                    style.id
                );
            };
            let Some(record) = simple_record(&entities[idx]) else {
                bail!(
                    "STYLED_ITEM #{} has complex presentation parent #{parent}",
                    style.id
                );
            };
            if !matches!(
                record.name.as_str(),
                "PRESENTATION_LAYER_ASSIGNMENT"
                    | "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION"
            ) {
                bail!(
                    "STYLED_ITEM #{} has unsupported presentation parent {} #{parent}",
                    style.id,
                    record.name
                );
            }
        }
        out.insert(style.id, parents);
    }
    Ok(out)
}

fn require_face_only_direct_styles(
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    touched: &HashSet<u64>,
    styles_by_target: &HashMap<u64, Vec<StyleRef>>,
) -> Result<()> {
    for &target in touched {
        if !styles_by_target.contains_key(&target) {
            continue;
        }
        let Some(&idx) = index.get(&target) else {
            bail!("styled target #{target} is missing");
        };
        let Some(record) = simple_record(&entities[idx]) else {
            bail!("styled moved target #{target} is complex");
        };
        if record.name != "ADVANCED_FACE" {
            bail!(
                "periodic-chain presentation surgery only supports directly styled ADVANCED_FACE targets, got {} #{target}",
                record.name
            );
        }
    }
    Ok(())
}

fn chain_apply_weld(
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

fn list_params(record: &Record) -> Option<&[Parameter]> {
    match &record.parameter {
        Parameter::List(params) => Some(params),
        _ => None,
    }
}

const fn entity_ref(id: u64) -> Parameter {
    Parameter::Ref(Name::Entity(id))
}

fn entity_ref_list(param: &Parameter) -> Option<Vec<u64>> {
    let Parameter::List(items) = param else {
        return None;
    };
    items.iter().map(entity_ref_value).collect()
}

fn numeric_value(param: &Parameter) -> Option<f64> {
    match param {
        Parameter::Integer(value) => crate::numeric::exact_i64_to_f64(*value),
        Parameter::Real(value) => Some(*value),
        _ => None,
    }
}

fn quantize_coord(point: [f64; 3]) -> [i64; 3] {
    [
        (point[0] / COORD_TOL_MM).round() as i64,
        (point[1] / COORD_TOL_MM).round() as i64,
        (point[2] / COORD_TOL_MM).round() as i64,
    ]
}

fn normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = norm(v);
    if !n.is_finite() || n <= 1.0e-15 {
        None
    } else {
        Some(scale(v, 1.0 / n))
    }
}

#[cfg(test)]
mod tests;

fn entity_descendant_closure(
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    seeds: &HashSet<u64>,
) -> Result<HashSet<u64>> {
    let mut descendants = HashSet::with_capacity(seeds.len());
    let mut stack = seeds.iter().copied().collect::<Vec<_>>();
    while let Some(id) = stack.pop() {
        if !descendants.insert(id) {
            continue;
        }
        let idx = *index
            .get(&id)
            .ok_or_else(|| anyhow!("descendant graph references missing entity #{id}"))?;
        visit_entity_refs(&entities[idx], &mut |child| stack.push(child));
    }
    Ok(descendants)
}

/// Remove detached topological vertex roots left after later support/value
/// interning. A bare `VERTEX_POINT` with no inbound STEP reference cannot
/// participate in any represented B-rep; descendants are collected only when
/// every surviving parent is collected with it.
pub(crate) fn prune_detached_vertex_points(entities: &mut Vec<EntityInstance>) -> usize {
    if entities.is_empty() {
        return 0;
    }
    let references = ReferenceGraph::new(entities);
    let refs = references.forward();
    let inbound = references.inbound();

    let seeds = entities
        .iter()
        .filter_map(|entity| {
            let id = entity_id(entity);
            simple_record(entity)
                .is_some_and(|record| record.name == "VERTEX_POINT")
                .then_some(id)
                .filter(|id| inbound.get(id).is_none_or(Vec::is_empty))
        })
        .collect::<HashSet<_>>();
    if seeds.is_empty() {
        return 0;
    }

    let mut candidate = HashSet::<u64>::new();
    let mut stack = seeds.iter().copied().collect::<Vec<_>>();
    while let Some(id) = stack.pop() {
        if !candidate.insert(id) {
            continue;
        }
        stack.extend(refs.get(&id).into_iter().flatten().copied());
    }

    let mut delete = HashSet::<u64>::new();
    loop {
        let mut changed = false;
        for &id in &candidate {
            if delete.contains(&id) {
                continue;
            }
            let all_dead = inbound
                .get(&id)
                .is_none_or(|parents| parents.iter().all(|parent| delete.contains(parent)));
            if all_dead {
                delete.insert(id);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let removed = delete.len();
    entities.retain(|entity| !delete.contains(&entity_id(entity)));
    removed
}
