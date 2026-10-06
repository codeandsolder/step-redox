use super::chain_weld::{chain_apply_weld, plan_chain_expansion_welds, plan_chain_shrink_welds};
use super::graph_editor::{GraphEditor, SourceLoop};
use super::{
    COORD_TOL_MM, SsRow, collect_style_container_parents, insert_stretch_edge, list_params,
    normalize, quantize_coord, require_face_only_direct_styles,
};
use crate::instances::{StyleRef, collect_styles_by_target};
use crate::math3::{add, dot, scale};
use crate::periodic_chains::PeriodicChainPattern;
use crate::step_graph::{ReferenceGraph, build_index, entity_ref_value};
use anyhow::{Result, anyhow, bail};
use ruststep::ast::EntityInstance;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

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
