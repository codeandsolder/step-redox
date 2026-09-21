use anyhow::{Result, anyhow, bail};
use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use serde::Serialize;
use std::collections::{HashMap, HashSet};

use crate::periodic_bodies::PeriodicBodyPattern;

const COORD_TOL_MM: f64 = 1.0e-7;

#[derive(Debug, Clone, Serialize, PartialEq)]
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
struct Traversal {
    edge: u64,
    start: u64,
    end: u64,
}

#[derive(Debug, Clone)]
struct SourceLoop {
    outer: bool,
    traversal: Vec<Traversal>,
}

#[derive(Debug, Clone, Copy)]
struct SsRow {
    center: f64,
    span: f64,
    edge: u64,
}

/// Expand a proven 1-D periodic body at its positive-axis end.
///
/// This deliberately handles only growth for the first production increment.
/// The graph transformation is exact: repeat-cell faces are cloned by rigid
/// translation and the spanning planar faces receive freshly traced boundary
/// loops over the target shared-edge graph.
pub fn expand_periodic_body_positive(
    entities: &mut Vec<EntityInstance>,
    body: &PeriodicBodyPattern,
    new_sites: usize,
) -> Result<PeriodicBodyResizeStats> {
    let old_sites = body.sites;
    if new_sites <= old_sites {
        bail!("periodic body expansion requires new_sites > old_sites");
    }
    if old_sites < 2 || body.repeat_face_families.is_empty() {
        bail!("periodic body does not contain enough repeated structure");
    }
    if body
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
    let extra = new_sites - old_sites;
    let delta_total = scale(axis, extra as f64 * pitch);

    let mut graph = GraphEditor::new(entities);
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

    let stretch_faces = body.stretch_face_ids.iter().copied().collect::<HashSet<_>>();
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
    let canonical_site = old_sites / 2;
    let canonical_faces = site_faces[canonical_site].clone();

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
    let right_threshold = site_projection[old_sites - 1] + pitch * 0.5;

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

    // Preserve source stretch-loop semantics before any face rewrite.
    let source_loops = stretch_faces
        .iter()
        .map(|&face| Ok((face, graph.source_loops(face)?)))
        .collect::<Result<HashMap<_, _>>>()?;

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
        let delta = scale(axis, (site as isize - canonical_site as isize) as f64 * pitch);
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
                stretch_edges.get_mut(stretch).unwrap().insert(edge);
            }
        }

        let is_canonical = faces.iter().any(|face| canonical_set.contains(face));
        if is_canonical {
            if let Some(stretch) = faces.iter().find(|face| stretch_faces.contains(face)) {
                canonical_boundary.push((edge, *stretch));
            }
        }
    }

    for mapping in site_maps.values() {
        for &(source_edge, stretch) in &canonical_boundary {
            let target_edge = mapping
                .get(&source_edge)
                .copied()
                .ok_or_else(|| anyhow!("cell clone missing boundary edge #{source_edge}"))?;
            stretch_edges.get_mut(&stretch).unwrap().insert(target_edge);
        }
    }

    // Fixed-cap to stretch-face interfaces.
    for (&edge, faces) in &edge_faces {
        let fixed = faces.iter().find(|face| fixed_faces.contains(face)).copied();
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
            stretch_edges.get_mut(&stretch).unwrap().insert(target_edge);
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
            stretch_edges.get_mut(&pair.0).unwrap().insert(edge);
            stretch_edges.get_mut(&pair.1).unwrap().insert(edge);
        }

        if short.is_empty() {
            continue;
        }
        if short.len() != old_sites + 1 {
            bail!(
                "unexpected short stretch-edge grammar for faces {:?}: {} rows, expected {}",
                pair,
                short.len(),
                old_sites + 1
            );
        }
        let left_end = short[0];
        let right_end = short[short.len() - 1];
        let gaps = &short[1..short.len() - 1];

        for row in std::iter::once(&left_end).chain(gaps.iter()) {
            stretch_edges.get_mut(&pair.0).unwrap().insert(row.edge);
            stretch_edges.get_mut(&pair.1).unwrap().insert(row.edge);
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
            stretch_edges.get_mut(&pair.0).unwrap().insert(edge);
            stretch_edges.get_mut(&pair.1).unwrap().insert(edge);
        }

        let [va, vb] = graph.edge_vertices(right_end.edge)?;
        let nva = find_vertex(add(graph.vertex_coord(va)?, delta_total))?;
        let nvb = find_vertex(add(graph.vertex_coord(vb)?, delta_total))?;
        let edge = graph.make_edge_like(right_end.edge, nva, nvb)?;
        new_stretch_edges += 1;
        stretch_edges.get_mut(&pair.0).unwrap().insert(edge);
        stretch_edges.get_mut(&pair.1).unwrap().insert(edge);
    }

    for &face in &body.stretch_face_ids {
        let edges = stretch_edges
            .get(&face)
            .ok_or_else(|| anyhow!("missing target edge set for stretch face #{face}"))?
            .clone();
        let loops = source_loops
            .get(&face)
            .ok_or_else(|| anyhow!("missing source loop semantics for stretch face #{face}"))?;
        graph.rebuild_face_bounds(face, &edges, loops)?;
    }

    // Replace positive fixed faces in the shell, keep everything else, append
    // the newly-created repeat faces.
    let mut target_shell_faces = Vec::with_capacity(
        shell_faces.len() + extra * body.faces_per_site,
    );
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

    let added_entities = graph.entities.len().saturating_sub(graph.initial_len);

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
/// omitted from the CLOSED_SHELL, and the spanning-face boundary grammar is
/// rebuilt over the shortened cell/gap sequence.
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
    let removed = old_sites - new_sites;
    let delta_total = scale(axis, -(removed as f64) * pitch);

    let mut graph = GraphEditor::new(entities);
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
    let fixed_faces = body
        .fixed_face_ids
        .iter()
        .copied()
        .collect::<HashSet<_>>();
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
    let canonical_site = new_sites / 2;

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
    let right_threshold = site_projection[old_sites - 1] + pitch * 0.5;

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
        let repeated = faces
            .iter()
            .any(|face| kept_repeat_faces.contains(face));
        if repeated {
            for stretch in faces.iter().filter(|face| stretch_faces.contains(face)) {
                stretch_edges.get_mut(stretch).unwrap().insert(edge);
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
            stretch_edges.get_mut(&stretch).unwrap().insert(target_edge);
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
            stretch_edges.get_mut(&pair.0).unwrap().insert(edge);
            stretch_edges.get_mut(&pair.1).unwrap().insert(edge);
        }

        if short.is_empty() {
            continue;
        }
        if short.len() != old_sites + 1 {
            bail!(
                "unexpected short stretch-edge grammar for faces {:?}: {} rows, expected {}",
                pair,
                short.len(),
                old_sites + 1
            );
        }
        let left_end = short[0];
        let right_end = short[short.len() - 1];
        let gaps = &short[1..short.len() - 1];

        stretch_edges.get_mut(&pair.0).unwrap().insert(left_end.edge);
        stretch_edges.get_mut(&pair.1).unwrap().insert(left_end.edge);
        for row in gaps.iter().take(new_sites.saturating_sub(1)) {
            stretch_edges.get_mut(&pair.0).unwrap().insert(row.edge);
            stretch_edges.get_mut(&pair.1).unwrap().insert(row.edge);
        }

        let [va, vb] = graph.edge_vertices(right_end.edge)?;
        let nva = find_vertex(add(graph.vertex_coord(va)?, delta_total))?;
        let nvb = find_vertex(add(graph.vertex_coord(vb)?, delta_total))?;
        let edge = graph.make_edge_like(right_end.edge, nva, nvb)?;
        new_stretch_edges += 1;
        stretch_edges.get_mut(&pair.0).unwrap().insert(edge);
        stretch_edges.get_mut(&pair.1).unwrap().insert(edge);
    }

    for &face in &body.stretch_face_ids {
        let edges = stretch_edges
            .get(&face)
            .ok_or_else(|| anyhow!("missing target edge set for stretch face #{face}"))?
            .clone();
        let loops = source_loops
            .get(&face)
            .ok_or_else(|| anyhow!("missing source loop semantics for stretch face #{face}"))?;
        graph.rebuild_face_bounds(face, &edges, loops)?;
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

    let added_entities = graph.entities.len().saturating_sub(graph.initial_len);

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

struct GraphEditor<'a> {
    entities: &'a mut Vec<EntityInstance>,
    index: HashMap<u64, usize>,
    next_id: u64,
    initial_len: usize,
}

impl<'a> GraphEditor<'a> {
    fn new(entities: &'a mut Vec<EntityInstance>) -> Self {
        let index = build_index(entities);
        let next_id = entities.iter().map(entity_id).max().unwrap_or(0) + 1;
        let initial_len = entities.len();
        Self {
            entities,
            index,
            next_id,
            initial_len,
        }
    }

    fn entity_type(&self, id: u64) -> Option<&str> {
        let idx = *self.index.get(&id)?;
        match &self.entities[idx] {
            EntityInstance::Simple { record, .. } => Some(record.name.as_str()),
            EntityInstance::Complex { .. } => Some("COMPLEX"),
        }
    }

    fn simple_record(&self, id: u64) -> Option<&Record> {
        let idx = *self.index.get(&id)?;
        simple_record(&self.entities[idx])
    }

    fn simple_record_mut(&mut self, id: u64) -> Option<&mut Record> {
        let idx = *self.index.get(&id)?;
        simple_record_mut(&mut self.entities[idx])
    }

    fn push_simple(&mut self, name: &str, params: Vec<Parameter>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let idx = self.entities.len();
        self.entities.push(EntityInstance::Simple {
            id,
            record: Record {
                name: name.to_string(),
                parameter: Parameter::List(params),
            },
        });
        self.index.insert(id, idx);
        id
    }

    fn shell_faces(&self, shell: u64) -> Result<Vec<u64>> {
        let record = self
            .simple_record(shell)
            .ok_or_else(|| anyhow!("missing shell #{shell}"))?;
        if record.name != "CLOSED_SHELL" {
            bail!("#{shell} is not CLOSED_SHELL");
        }
        let params = list_params(record).ok_or_else(|| anyhow!("shell parameters are not a list"))?;
        params
            .get(1)
            .and_then(entity_ref_list)
            .ok_or_else(|| anyhow!("shell #{shell} has no face aggregate"))
    }

    fn set_shell_faces(&mut self, shell: u64, faces: &[u64]) -> Result<()> {
        let record = self
            .simple_record_mut(shell)
            .ok_or_else(|| anyhow!("missing shell #{shell}"))?;
        let Parameter::List(params) = &mut record.parameter else {
            bail!("shell #{shell} parameters are not a list");
        };
        if params.len() < 2 {
            bail!("shell #{shell} has incomplete parameters");
        }
        params[1] = Parameter::List(faces.iter().copied().map(entity_ref).collect());
        Ok(())
    }

    fn refs_map(&self) -> HashMap<u64, Vec<u64>> {
        entity_ref_map(self.entities)
    }

    fn clone_descendants(
        &mut self,
        seeds: &HashSet<u64>,
        delta: [f64; 3],
    ) -> Result<HashMap<u64, u64>> {
        let refs = self.refs_map();
        let mut descendants = HashSet::new();
        let mut stack = seeds.iter().copied().collect::<Vec<_>>();
        while let Some(id) = stack.pop() {
            if !descendants.insert(id) {
                continue;
            }
            let Some(children) = refs.get(&id) else {
                bail!("clone seed graph references missing entity #{id}");
            };
            stack.extend(children.iter().copied());
        }

        let mut ids = descendants.into_iter().collect::<Vec<_>>();
        ids.sort_unstable();
        let mut mapping = HashMap::with_capacity(ids.len());
        for &old in &ids {
            let new = self.next_id;
            self.next_id += 1;
            mapping.insert(old, new);
        }

        let old_index = self.index.clone();
        let mut clones = Vec::with_capacity(ids.len());
        for &old in &ids {
            let idx = *old_index
                .get(&old)
                .ok_or_else(|| anyhow!("clone graph missing entity #{old}"))?;
            let mut entity = self.entities[idx].clone();
            set_entity_id(&mut entity, mapping[&old]);
            remap_entity_refs(&mut entity, &mapping);
            translate_cartesian_point(&mut entity, delta)?;
            clones.push(entity);
        }

        for entity in clones {
            let id = entity_id(&entity);
            let idx = self.entities.len();
            self.entities.push(entity);
            self.index.insert(id, idx);
        }
        Ok(mapping)
    }

    fn face_edges_ordered(&self, face: u64) -> Result<Vec<u64>> {
        let record = self
            .simple_record(face)
            .ok_or_else(|| anyhow!("missing face #{face}"))?;
        if record.name != "ADVANCED_FACE" {
            bail!("#{face} is not ADVANCED_FACE");
        }
        let params = list_params(record).ok_or_else(|| anyhow!("face params are not a list"))?;
        let bounds = params
            .get(1)
            .and_then(entity_ref_list)
            .ok_or_else(|| anyhow!("face #{face} has no bounds"))?;
        let mut out = Vec::new();
        for bound in bounds {
            let bound_record = self
                .simple_record(bound)
                .ok_or_else(|| anyhow!("missing face bound #{bound}"))?;
            if bound_record.name != "FACE_BOUND" && bound_record.name != "FACE_OUTER_BOUND" {
                bail!("#{bound} is not a face bound");
            }
            let bound_params =
                list_params(bound_record).ok_or_else(|| anyhow!("bound params are not a list"))?;
            let loop_id = entity_ref_value(
                bound_params
                    .get(1)
                    .ok_or_else(|| anyhow!("bound #{bound} missing loop"))?,
            )
            .ok_or_else(|| anyhow!("bound #{bound} loop is not a reference"))?;
            let loop_record = self
                .simple_record(loop_id)
                .ok_or_else(|| anyhow!("missing edge loop #{loop_id}"))?;
            if loop_record.name != "EDGE_LOOP" {
                bail!("#{loop_id} is not EDGE_LOOP");
            }
            let loop_params =
                list_params(loop_record).ok_or_else(|| anyhow!("loop params are not a list"))?;
            let oriented = loop_params
                .get(1)
                .and_then(entity_ref_list)
                .ok_or_else(|| anyhow!("loop #{loop_id} has no edge aggregate"))?;
            for oe in oriented {
                let oe_record = self
                    .simple_record(oe)
                    .ok_or_else(|| anyhow!("missing oriented edge #{oe}"))?;
                if oe_record.name != "ORIENTED_EDGE" {
                    bail!("#{oe} is not ORIENTED_EDGE");
                }
                let oe_params =
                    list_params(oe_record).ok_or_else(|| anyhow!("oriented edge params invalid"))?;
                let edge = entity_ref_value(
                    oe_params
                        .get(3)
                        .ok_or_else(|| anyhow!("oriented edge #{oe} missing EDGE_CURVE"))?,
                )
                .ok_or_else(|| anyhow!("oriented edge #{oe} edge is not a ref"))?;
                out.push(edge);
            }
        }
        Ok(out)
    }

    fn edge_faces(&self, faces: &HashSet<u64>) -> Result<HashMap<u64, Vec<u64>>> {
        let mut out = HashMap::<u64, Vec<u64>>::new();
        for &face in faces {
            let edges = self.face_edges_ordered(face)?;
            let mut unique = HashSet::new();
            for edge in edges {
                if unique.insert(edge) {
                    out.entry(edge).or_default().push(face);
                }
            }
        }
        Ok(out)
    }

    fn edge_vertices(&self, edge: u64) -> Result<[u64; 2]> {
        let record = self
            .simple_record(edge)
            .ok_or_else(|| anyhow!("missing edge #{edge}"))?;
        if record.name != "EDGE_CURVE" {
            bail!("#{edge} is not EDGE_CURVE");
        }
        let params = list_params(record).ok_or_else(|| anyhow!("edge params are not a list"))?;
        let va = entity_ref_value(
            params
                .get(1)
                .ok_or_else(|| anyhow!("edge #{edge} missing start vertex"))?,
        )
        .ok_or_else(|| anyhow!("edge #{edge} start vertex is not a ref"))?;
        let vb = entity_ref_value(
            params
                .get(2)
                .ok_or_else(|| anyhow!("edge #{edge} missing end vertex"))?,
        )
        .ok_or_else(|| anyhow!("edge #{edge} end vertex is not a ref"))?;
        Ok([va, vb])
    }

    fn vertex_coord(&self, vertex: u64) -> Result<[f64; 3]> {
        let record = self
            .simple_record(vertex)
            .ok_or_else(|| anyhow!("missing vertex #{vertex}"))?;
        if record.name != "VERTEX_POINT" {
            bail!("#{vertex} is not VERTEX_POINT");
        }
        let params = list_params(record).ok_or_else(|| anyhow!("vertex params are not a list"))?;
        let point = entity_ref_value(
            params
                .get(1)
                .ok_or_else(|| anyhow!("vertex #{vertex} missing point"))?,
        )
        .ok_or_else(|| anyhow!("vertex #{vertex} point is not a ref"))?;
        self.cartesian_point(point)
    }

    fn cartesian_point(&self, point: u64) -> Result<[f64; 3]> {
        let record = self
            .simple_record(point)
            .ok_or_else(|| anyhow!("missing point #{point}"))?;
        if record.name != "CARTESIAN_POINT" {
            bail!("#{point} is not CARTESIAN_POINT");
        }
        let params = list_params(record).ok_or_else(|| anyhow!("point params are not a list"))?;
        let Parameter::List(coords) = params
            .get(1)
            .ok_or_else(|| anyhow!("point #{point} missing coordinates"))?
        else {
            bail!("point #{point} coordinates are not a list");
        };
        if coords.len() != 3 {
            bail!("point #{point} is not 3-D");
        }
        Ok([
            numeric_value(&coords[0]).ok_or_else(|| anyhow!("point coord is not numeric"))?,
            numeric_value(&coords[1]).ok_or_else(|| anyhow!("point coord is not numeric"))?,
            numeric_value(&coords[2]).ok_or_else(|| anyhow!("point coord is not numeric"))?,
        ])
    }

    fn face_vertices(&self, face: u64) -> Result<HashSet<u64>> {
        let mut vertices = HashSet::new();
        for edge in self.face_edges_ordered(face)? {
            vertices.extend(self.edge_vertices(edge)?);
        }
        Ok(vertices)
    }

    fn face_center(&self, face: u64) -> Result<[f64; 3]> {
        let vertices = self.face_vertices(face)?;
        if vertices.is_empty() {
            bail!("face #{face} has no vertices");
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for vertex in vertices {
            let p = self.vertex_coord(vertex)?;
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        Ok([
            (lo[0] + hi[0]) * 0.5,
            (lo[1] + hi[1]) * 0.5,
            (lo[2] + hi[2]) * 0.5,
        ])
    }

    fn edge_center_span(&self, edge: u64, axis: [f64; 3]) -> Result<(f64, f64)> {
        let [va, vb] = self.edge_vertices(edge)?;
        let pa = dot(self.vertex_coord(va)?, axis);
        let pb = dot(self.vertex_coord(vb)?, axis);
        Ok(((pa + pb) * 0.5, (pb - pa).abs()))
    }

    fn make_edge_like(&mut self, prototype: u64, va: u64, vb: u64) -> Result<u64> {
        let record = self
            .simple_record(prototype)
            .ok_or_else(|| anyhow!("missing edge prototype #{prototype}"))?
            .clone();
        if record.name != "EDGE_CURVE" {
            bail!("edge prototype #{prototype} is not EDGE_CURVE");
        }
        let Parameter::List(mut params) = record.parameter else {
            bail!("edge prototype params are not a list");
        };
        if params.len() < 5 {
            bail!("edge prototype has incomplete parameters");
        }
        params[1] = entity_ref(va);
        params[2] = entity_ref(vb);
        Ok(self.push_simple("EDGE_CURVE", params))
    }

    fn source_loops(&self, face: u64) -> Result<Vec<SourceLoop>> {
        let record = self
            .simple_record(face)
            .ok_or_else(|| anyhow!("missing face #{face}"))?;
        let params = list_params(record).ok_or_else(|| anyhow!("face params invalid"))?;
        let bounds = params
            .get(1)
            .and_then(entity_ref_list)
            .ok_or_else(|| anyhow!("face #{face} has no bounds"))?;
        let mut out = Vec::new();

        for bound in bounds {
            let bound_record = self
                .simple_record(bound)
                .ok_or_else(|| anyhow!("missing bound #{bound}"))?;
            let outer = bound_record.name == "FACE_OUTER_BOUND";
            let bparams = list_params(bound_record).ok_or_else(|| anyhow!("bound params invalid"))?;
            let loop_id = entity_ref_value(
                bparams
                    .get(1)
                    .ok_or_else(|| anyhow!("bound #{bound} missing loop"))?,
            )
            .ok_or_else(|| anyhow!("bound loop is not a ref"))?;
            let loop_record = self
                .simple_record(loop_id)
                .ok_or_else(|| anyhow!("missing loop #{loop_id}"))?;
            let lparams = list_params(loop_record).ok_or_else(|| anyhow!("loop params invalid"))?;
            let oes = lparams
                .get(1)
                .and_then(entity_ref_list)
                .ok_or_else(|| anyhow!("loop #{loop_id} has no edges"))?;
            let mut traversal = Vec::with_capacity(oes.len());
            for oe in oes {
                let oe_record = self
                    .simple_record(oe)
                    .ok_or_else(|| anyhow!("missing oriented edge #{oe}"))?;
                let params = list_params(oe_record).ok_or_else(|| anyhow!("OE params invalid"))?;
                let edge = entity_ref_value(
                    params
                        .get(3)
                        .ok_or_else(|| anyhow!("OE #{oe} missing EDGE_CURVE"))?,
                )
                .ok_or_else(|| anyhow!("OE edge is not a ref"))?;
                let forward = matches!(
                    params.get(4),
                    Some(Parameter::Enumeration(value)) if value == "T"
                );
                let [va, vb] = self.edge_vertices(edge)?;
                traversal.push(if forward {
                    Traversal {
                        edge,
                        start: va,
                        end: vb,
                    }
                } else {
                    Traversal {
                        edge,
                        start: vb,
                        end: va,
                    }
                });
            }
            out.push(SourceLoop { outer, traversal });
        }
        Ok(out)
    }

    fn rebuild_face_bounds(
        &mut self,
        face: u64,
        target_edges: &HashSet<u64>,
        source_loops: &[SourceLoop],
    ) -> Result<()> {
        let source_outer = source_loops
            .iter()
            .find(|loop_| loop_.outer)
            .ok_or_else(|| anyhow!("face #{face} has no source outer loop"))?;
        let ref_area = self.area_vector(&source_outer.traversal)?;
        if norm(ref_area) <= 1.0e-12 {
            bail!("face #{face} source outer loop has zero area");
        }
        let hole_sign = if let Some(hole) = source_loops.iter().find(|loop_| !loop_.outer) {
            if dot(self.area_vector(&hole.traversal)?, ref_area) > 0.0 {
                1.0
            } else {
                -1.0
            }
        } else {
            -1.0
        };

        let mut cycles = self.trace_cycles(target_edges)?;
        if cycles.is_empty() {
            bail!("face #{face} target boundary has no cycles");
        }
        let scored = cycles
            .iter()
            .map(|cycle| self.area_vector(cycle).map(|area| dot(area, ref_area).abs()))
            .collect::<Result<Vec<_>>>()?;
        let outer_index = scored
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap();

        let mut new_bounds = Vec::with_capacity(cycles.len());
        for (index, cycle) in cycles.iter_mut().enumerate() {
            let desired = if index == outer_index { 1.0 } else { hole_sign };
            let actual = dot(self.area_vector(cycle)?, ref_area);
            if actual == 0.0 {
                bail!("face #{face} target cycle has zero area");
            }
            if (actual > 0.0) != (desired > 0.0) {
                cycle.reverse();
                for step in cycle.iter_mut() {
                    std::mem::swap(&mut step.start, &mut step.end);
                }
            }

            let mut oriented = Vec::with_capacity(cycle.len());
            for step in cycle.iter() {
                let [va, vb] = self.edge_vertices(step.edge)?;
                let forward = if step.start == va && step.end == vb {
                    true
                } else if step.start == vb && step.end == va {
                    false
                } else {
                    bail!("face #{face} cycle endpoint mismatch");
                };
                let oe = self.push_simple(
                    "ORIENTED_EDGE",
                    vec![
                        Parameter::String(String::new()),
                        Parameter::Omitted,
                        Parameter::Omitted,
                        entity_ref(step.edge),
                        Parameter::Enumeration(if forward { "T" } else { "F" }.to_string()),
                    ],
                );
                oriented.push(oe);
            }
            let loop_id = self.push_simple(
                "EDGE_LOOP",
                vec![
                    Parameter::String(String::new()),
                    Parameter::List(oriented.into_iter().map(entity_ref).collect()),
                ],
            );
            let bound_name = if index == outer_index {
                "FACE_OUTER_BOUND"
            } else {
                "FACE_BOUND"
            };
            let bound = self.push_simple(
                bound_name,
                vec![
                    Parameter::String(String::new()),
                    entity_ref(loop_id),
                    Parameter::Enumeration("T".to_string()),
                ],
            );
            new_bounds.push(bound);
        }

        let record = self
            .simple_record_mut(face)
            .ok_or_else(|| anyhow!("missing face #{face} during rewrite"))?;
        let Parameter::List(params) = &mut record.parameter else {
            bail!("face #{face} parameters changed shape");
        };
        if params.len() < 4 {
            bail!("face #{face} has incomplete ADVANCED_FACE parameters");
        }
        params[1] = Parameter::List(new_bounds.into_iter().map(entity_ref).collect());
        Ok(())
    }

    fn trace_cycles(&self, edges: &HashSet<u64>) -> Result<Vec<Vec<Traversal>>> {
        let mut incident = HashMap::<u64, Vec<u64>>::new();
        for &edge in edges {
            let [a, b] = self.edge_vertices(edge)?;
            incident.entry(a).or_default().push(edge);
            incident.entry(b).or_default().push(edge);
        }
        for (&vertex, members) in &incident {
            if members.len() != 2 {
                bail!(
                    "target face boundary is not a disjoint set of cycles: vertex #{vertex} has degree {}",
                    members.len()
                );
            }
        }

        let mut unused = edges.clone();
        let mut cycles = Vec::new();
        while !unused.is_empty() {
            let edge0 = *unused.iter().min().unwrap();
            let [start_vertex, _] = self.edge_vertices(edge0)?;
            let mut current_vertex = start_vertex;
            let mut current_edge = edge0;
            let mut cycle = Vec::new();

            loop {
                unused.remove(&current_edge);
                let [a, b] = self.edge_vertices(current_edge)?;
                let next_vertex = if current_vertex == a {
                    b
                } else if current_vertex == b {
                    a
                } else {
                    bail!("cycle traversal lost endpoint identity");
                };
                cycle.push(Traversal {
                    edge: current_edge,
                    start: current_vertex,
                    end: next_vertex,
                });
                current_vertex = next_vertex;
                if current_vertex == start_vertex {
                    break;
                }
                let candidates = incident
                    .get(&current_vertex)
                    .into_iter()
                    .flatten()
                    .filter(|edge| unused.contains(edge))
                    .copied()
                    .collect::<Vec<_>>();
                if candidates.len() != 1 {
                    bail!(
                        "cycle continuation at vertex #{} is ambiguous: {:?}",
                        current_vertex,
                        candidates
                    );
                }
                current_edge = candidates[0];
            }
            cycles.push(cycle);
        }
        Ok(cycles)
    }

    fn area_vector(&self, traversal: &[Traversal]) -> Result<[f64; 3]> {
        if traversal.len() < 3 {
            return Ok([0.0; 3]);
        }
        let points = traversal
            .iter()
            .map(|step| self.vertex_coord(step.start))
            .collect::<Result<Vec<_>>>()?;
        let mut acc = [0.0; 3];
        for index in 0..points.len() {
            let a = points[index];
            let b = points[(index + 1) % points.len()];
            acc = add(acc, cross(a, b));
        }
        Ok(scale(acc, 0.5))
    }
}

fn translate_cartesian_point(entity: &mut EntityInstance, delta: [f64; 3]) -> Result<()> {
    let EntityInstance::Simple { record, .. } = entity else {
        return Ok(());
    };
    if record.name != "CARTESIAN_POINT" {
        return Ok(());
    }
    let Parameter::List(params) = &mut record.parameter else {
        bail!("CARTESIAN_POINT parameters are not a list");
    };
    let Some(Parameter::List(coords)) = params.get_mut(1) else {
        bail!("CARTESIAN_POINT lacks coordinate aggregate");
    };
    if coords.len() != 3 {
        bail!("CARTESIAN_POINT is not 3-D");
    }
    for index in 0..3 {
        let value = numeric_value(&coords[index])
            .ok_or_else(|| anyhow!("CARTESIAN_POINT coordinate is not numeric"))?;
        coords[index] = Parameter::Real(value + delta[index]);
    }
    Ok(())
}

fn remap_entity_refs(entity: &mut EntityInstance, mapping: &HashMap<u64, u64>) {
    match entity {
        EntityInstance::Simple { record, .. } => remap_param_refs(&mut record.parameter, mapping),
        EntityInstance::Complex { subsuper, .. } => {
            for record in &mut subsuper.0 {
                remap_param_refs(&mut record.parameter, mapping);
            }
        }
    }
}

fn remap_param_refs(param: &mut Parameter, mapping: &HashMap<u64, u64>) {
    match param {
        Parameter::Ref(Name::Entity(id)) => {
            if let Some(new) = mapping.get(id) {
                *id = *new;
            }
        }
        Parameter::List(items) => {
            for item in items {
                remap_param_refs(item, mapping);
            }
        }
        Parameter::Typed { parameter, .. } => remap_param_refs(parameter, mapping),
        _ => {}
    }
}

fn set_entity_id(entity: &mut EntityInstance, id: u64) {
    match entity {
        EntityInstance::Simple { id: current, .. }
        | EntityInstance::Complex { id: current, .. } => *current = id,
    }
}

fn entity_ref_map(entities: &[EntityInstance]) -> HashMap<u64, Vec<u64>> {
    let mut out = HashMap::new();
    for entity in entities {
        let id = entity_id(entity);
        let mut refs = Vec::new();
        visit_entity_refs(entity, &mut |child| refs.push(child));
        out.insert(id, refs);
    }
    out
}

fn visit_entity_refs(entity: &EntityInstance, f: &mut impl FnMut(u64)) {
    match entity {
        EntityInstance::Simple { record, .. } => visit_param_refs(&record.parameter, f),
        EntityInstance::Complex { subsuper, .. } => {
            for record in &subsuper.0 {
                visit_param_refs(&record.parameter, f);
            }
        }
    }
}

fn visit_param_refs(param: &Parameter, f: &mut impl FnMut(u64)) {
    match param {
        Parameter::Ref(Name::Entity(id)) => f(*id),
        Parameter::List(items) => {
            for item in items {
                visit_param_refs(item, f);
            }
        }
        Parameter::Typed { parameter, .. } => visit_param_refs(parameter, f),
        _ => {}
    }
}

fn build_index(entities: &[EntityInstance]) -> HashMap<u64, usize> {
    entities
        .iter()
        .enumerate()
        .map(|(index, entity)| (entity_id(entity), index))
        .collect()
}

fn simple_record(entity: &EntityInstance) -> Option<&Record> {
    match entity {
        EntityInstance::Simple { record, .. } => Some(record),
        EntityInstance::Complex { .. } => None,
    }
}

fn simple_record_mut(entity: &mut EntityInstance) -> Option<&mut Record> {
    match entity {
        EntityInstance::Simple { record, .. } => Some(record),
        EntityInstance::Complex { .. } => None,
    }
}

fn entity_id(entity: &EntityInstance) -> u64 {
    match entity {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id,
    }
}

fn list_params(record: &Record) -> Option<&[Parameter]> {
    match &record.parameter {
        Parameter::List(params) => Some(params),
        _ => None,
    }
}

fn entity_ref(id: u64) -> Parameter {
    Parameter::Ref(Name::Entity(id))
}

fn entity_ref_value(param: &Parameter) -> Option<u64> {
    match param {
        Parameter::Ref(Name::Entity(id)) => Some(*id),
        _ => None,
    }
}

fn entity_ref_list(param: &Parameter) -> Option<Vec<u64>> {
    let Parameter::List(items) = param else {
        return None;
    };
    items.iter().map(entity_ref_value).collect()
}

fn numeric_value(param: &Parameter) -> Option<f64> {
    match param {
        Parameter::Integer(value) => Some(*value as f64),
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

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(v: [f64; 3], scalar: f64) -> [f64; 3] {
    [v[0] * scalar, v[1] * scalar, v[2] * scalar]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
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
mod tests {
    use super::*;

    #[test]
    fn quantized_coordinate_is_stable_under_small_noise() {
        assert_eq!(
            quantize_coord([1.0, 2.0, 3.0]),
            quantize_coord([1.0 + 1.0e-9, 2.0 - 1.0e-9, 3.0])
        );
    }
}
