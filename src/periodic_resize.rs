use crate::math3::{add, dot, normalize as normalize3, scale};
use crate::step_entities::number as numeric_value;
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

mod chain_weld;
#[cfg(test)]
use chain_weld::chain_curve_key;
use chain_weld::{chain_apply_weld, plan_chain_expansion_welds, plan_chain_shrink_welds};

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

enum BodyStretchMode<'a> {
    Expand {
        repeat_faces: &'a HashSet<u64>,
        canonical_faces: &'a HashSet<u64>,
        site_maps: &'a [HashMap<u64, u64>],
        extra: usize,
    },
    Shrink {
        kept_repeat_faces: &'a HashSet<u64>,
        new_sites: usize,
    },
}

struct BodyStretchRewrite<'a> {
    axis: [f64; 3],
    pitch: f64,
    old_sites: usize,
    delta_total: [f64; 3],
    stretch_face_ids: &'a [u64],
    stretch_faces: &'a HashSet<u64>,
    fixed_faces: &'a HashSet<u64>,
    right_fixed: &'a HashSet<u64>,
    edge_faces: &'a HashMap<u64, Vec<u64>>,
    source_loops: &'a HashMap<u64, Vec<SourceLoop>>,
    target_patch_faces: &'a HashSet<u64>,
    cap_map: &'a HashMap<u64, u64>,
    mode: BodyStretchMode<'a>,
}

impl BodyStretchRewrite<'_> {
    fn run(&self, graph: &mut GraphEditor<'_>) -> Result<usize> {
        let mut coord_vertices = HashMap::<[i64; 3], Vec<u64>>::new();
        for &face in self.target_patch_faces {
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

        let mut stretch_edges = self
            .stretch_faces
            .iter()
            .map(|&face| (face, HashSet::<u64>::new()))
            .collect::<HashMap<_, _>>();

        match &self.mode {
            BodyStretchMode::Expand {
                repeat_faces,
                canonical_faces,
                site_maps,
                ..
            } => {
                let mut canonical_boundary = Vec::<(u64, u64)>::new();
                for (&edge, faces) in self.edge_faces {
                    if faces.iter().any(|face| repeat_faces.contains(face)) {
                        for stretch in faces
                            .iter()
                            .filter(|face| self.stretch_faces.contains(face))
                        {
                            insert_stretch_edge(&mut stretch_edges, *stretch, edge)?;
                        }
                    }
                    if faces.iter().any(|face| canonical_faces.contains(face))
                        && let Some(stretch) =
                            faces.iter().find(|face| self.stretch_faces.contains(face))
                    {
                        canonical_boundary.push((edge, *stretch));
                    }
                }
                for mapping in *site_maps {
                    for &(source_edge, stretch) in &canonical_boundary {
                        let target_edge = mapping.get(&source_edge).copied().ok_or_else(|| {
                            anyhow!("cell clone missing boundary edge #{source_edge}")
                        })?;
                        insert_stretch_edge(&mut stretch_edges, stretch, target_edge)?;
                    }
                }
            }
            BodyStretchMode::Shrink {
                kept_repeat_faces, ..
            } => {
                for (&edge, faces) in self.edge_faces {
                    if faces.iter().any(|face| kept_repeat_faces.contains(face)) {
                        for stretch in faces
                            .iter()
                            .filter(|face| self.stretch_faces.contains(face))
                        {
                            insert_stretch_edge(&mut stretch_edges, *stretch, edge)?;
                        }
                    }
                }
            }
        }

        for (&edge, faces) in self.edge_faces {
            let fixed = faces
                .iter()
                .find(|face| self.fixed_faces.contains(face))
                .copied();
            let Some(fixed) = fixed else {
                continue;
            };
            let target_edge = if self.right_fixed.contains(&fixed) {
                self.cap_map
                    .get(&edge)
                    .copied()
                    .ok_or_else(|| anyhow!("positive cap clone missing interface edge #{edge}"))?
            } else {
                edge
            };
            for stretch in faces
                .iter()
                .filter(|face| self.stretch_faces.contains(face))
            {
                insert_stretch_edge(&mut stretch_edges, *stretch, target_edge)?;
            }
        }

        let mut ss_pairs = HashMap::<(u64, u64), Vec<SsRow>>::new();
        for (&edge, faces) in self.edge_faces {
            if faces.len() != 2 || !faces.iter().all(|face| self.stretch_faces.contains(face)) {
                continue;
            }
            let mut pair = [faces[0], faces[1]];
            pair.sort_unstable();
            let (center, span) = graph.edge_center_span(edge, self.axis)?;
            ss_pairs
                .entry((pair[0], pair[1]))
                .or_default()
                .push(SsRow { center, span, edge });
        }

        let mut new_stretch_edges = 0usize;
        for (pair, mut rows) in ss_pairs {
            rows.sort_by(|a, b| a.center.total_cmp(&b.center));
            let full_threshold = self.old_sites.saturating_sub(1) as f64 * self.pitch;
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
                if dot(pa, self.axis) > dot(pb, self.axis) {
                    std::mem::swap(&mut va, &mut vb);
                    std::mem::swap(&mut pa, &mut pb);
                }
                let nv = find_vertex(add(pb, self.delta_total))?;
                let edge = graph.make_edge_like(row.edge, va, nv)?;
                new_stretch_edges += 1;
                insert_stretch_edge(&mut stretch_edges, pair.0, edge)?;
                insert_stretch_edge(&mut stretch_edges, pair.1, edge)?;
            }

            if short.is_empty() {
                continue;
            }
            if short.len() != self.old_sites + 1 {
                bail!(
                    "unexpected short stretch-edge grammar for faces {pair:?}: {} rows, expected {}",
                    short.len(),
                    self.old_sites + 1
                );
            }
            let left_end = short[0];
            let right_end = short[short.len() - 1];
            let gaps = &short[1..short.len() - 1];
            insert_stretch_edge(&mut stretch_edges, pair.0, left_end.edge)?;
            insert_stretch_edge(&mut stretch_edges, pair.1, left_end.edge)?;

            match self.mode {
                BodyStretchMode::Expand { extra, .. } => {
                    for row in gaps {
                        insert_stretch_edge(&mut stretch_edges, pair.0, row.edge)?;
                        insert_stretch_edge(&mut stretch_edges, pair.1, row.edge)?;
                    }
                    let prototype = gaps
                        .last()
                        .copied()
                        .ok_or_else(|| anyhow!("stretch grammar contains no inter-site gap"))?;
                    for step in 1..=extra {
                        let delta = scale(self.axis, step as f64 * self.pitch);
                        let [va, vb] = graph.edge_vertices(prototype.edge)?;
                        let nva = find_vertex(add(graph.vertex_coord(va)?, delta))?;
                        let nvb = find_vertex(add(graph.vertex_coord(vb)?, delta))?;
                        let edge = graph.make_edge_like(prototype.edge, nva, nvb)?;
                        new_stretch_edges += 1;
                        insert_stretch_edge(&mut stretch_edges, pair.0, edge)?;
                        insert_stretch_edge(&mut stretch_edges, pair.1, edge)?;
                    }
                }
                BodyStretchMode::Shrink { new_sites, .. } => {
                    for row in gaps.iter().take(new_sites.saturating_sub(1)) {
                        insert_stretch_edge(&mut stretch_edges, pair.0, row.edge)?;
                        insert_stretch_edge(&mut stretch_edges, pair.1, row.edge)?;
                    }
                }
            }

            let [va, vb] = graph.edge_vertices(right_end.edge)?;
            let nva = find_vertex(add(graph.vertex_coord(va)?, self.delta_total))?;
            let nvb = find_vertex(add(graph.vertex_coord(vb)?, self.delta_total))?;
            let edge = graph.make_edge_like(right_end.edge, nva, nvb)?;
            new_stretch_edges += 1;
            insert_stretch_edge(&mut stretch_edges, pair.0, edge)?;
            insert_stretch_edge(&mut stretch_edges, pair.1, edge)?;
        }

        for &face in self.stretch_face_ids {
            let edges = stretch_edges
                .get(&face)
                .ok_or_else(|| anyhow!("missing target edge set for stretch face #{face}"))?;
            let loops = self
                .source_loops
                .get(&face)
                .ok_or_else(|| anyhow!("missing source loop semantics for stretch face #{face}"))?;
            graph.rebuild_face_bounds(face, edges, loops)?;
        }
        Ok(new_stretch_edges)
    }
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

    let canonical_face_set = canonical_faces.iter().copied().collect::<HashSet<_>>();
    let mut site_maps = Vec::<HashMap<u64, u64>>::with_capacity(extra);
    let mut new_site_faces = Vec::<Vec<u64>>::with_capacity(extra);
    for site in old_sites..new_sites {
        let delta = scale(
            axis,
            (site as isize - canonical_site as isize) as f64 * pitch,
        );
        let mapping = graph.clone_descendants(&canonical_face_set, delta)?;
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
        site_maps.push(mapping);
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

    let new_stretch_edges = BodyStretchRewrite {
        axis,
        pitch,
        old_sites,
        delta_total,
        stretch_face_ids: &body.stretch_face_ids,
        stretch_faces: &stretch_faces,
        fixed_faces: &fixed_faces,
        right_fixed: &right_fixed,
        edge_faces: &edge_faces,
        source_loops: &source_loops,
        target_patch_faces: &target_patch_faces,
        cap_map: &cap_map,
        mode: BodyStretchMode::Expand {
            repeat_faces: &repeat_faces,
            canonical_faces: &canonical_face_set,
            site_maps: &site_maps,
            extra,
        },
    }
    .run(&mut graph)?;

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

    let new_stretch_edges = BodyStretchRewrite {
        axis,
        pitch,
        old_sites,
        delta_total,
        stretch_face_ids: &body.stretch_face_ids,
        stretch_faces: &stretch_faces,
        fixed_faces: &fixed_faces,
        right_fixed: &right_fixed,
        edge_faces: &edge_faces,
        source_loops: &source_loops,
        target_patch_faces: &target_patch_faces,
        cap_map: &cap_map,
        mode: BodyStretchMode::Shrink {
            kept_repeat_faces: &kept_repeat_faces,
            new_sites,
        },
    }
    .run(&mut graph)?;

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
            ReferenceGraph::new(entities).descendant_closure(&preflight_index, touched_roots);
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
    let pruned_entities = graph.prune_unreachable_descendants(&prune_roots);
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
    let pruned_entities = graph.prune_unreachable_descendants(&prune_roots);
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

fn quantize_coord(point: [f64; 3]) -> [i64; 3] {
    [
        (point[0] / COORD_TOL_MM).round() as i64,
        (point[1] / COORD_TOL_MM).round() as i64,
        (point[2] / COORD_TOL_MM).round() as i64,
    ]
}

fn normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    normalize3(v, 1.0e-15)
}

#[cfg(test)]
mod tests;

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
