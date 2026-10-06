use super::graph_editor::{GraphEditor, SourceLoop};
use super::{COORD_TOL_MM, SsRow, insert_stretch_edge, list_params, normalize, quantize_coord};
use crate::math3::{add, dot, scale};
use crate::periodic_bodies::PeriodicBodyPattern;
use crate::step_graph::entity_ref_value;
use anyhow::{Result, anyhow, bail};
use ruststep::ast::EntityInstance;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

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
