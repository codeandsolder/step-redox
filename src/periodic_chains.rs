use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};

const GEOM_TOL_MM: f64 = 1.0e-5;
const MIN_FAMILY_INSTANCES: usize = 4;
const MIN_LATTICE_FAMILY_VOTES: usize = 4;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PeriodicChainPattern {
    pub solid_id: u64,
    pub axis: [f64; 3],
    pub pitch_mm: f64,
    pub sites: usize,
    pub site_centers_mm: Vec<f64>,
    pub lattice_face_family_votes: usize,
    pub complete_partition: bool,
    pub read_only_proven: bool,
    pub faces_without_geometry: usize,
    pub nonmanifold_edges: usize,
    pub cross_site_edges: usize,
    pub nonlocal_cross_site_edges: usize,
    pub adjacent_site_edge_counts: Vec<usize>,
    pub site_face_counts: Vec<usize>,
    pub interior_site_face_count: usize,
    pub gap_face_counts: Vec<usize>,
    pub interior_gap_face_count: usize,
    pub site_face_ids: Vec<Vec<u64>>,
    pub gap_face_ids: Vec<Vec<u64>>,
    pub stretch_face_ids: Vec<u64>,
    pub fixed_negative_face_ids: Vec<u64>,
    pub fixed_middle_face_ids: Vec<u64>,
    pub fixed_positive_face_ids: Vec<u64>,
    pub repeat_coverage_ratio: f64,
    pub edge_category_counts: BTreeMap<String, usize>,
    pub site_adjacency_signatures: Vec<BTreeMap<String, usize>>,
    pub gap_adjacency_signatures: Vec<BTreeMap<String, usize>>,
}

#[derive(Debug, Clone)]
struct FaceGeom {
    id: u64,
    lo: [f64; 3],
    hi: [f64; 3],
    center: [f64; 3],
    span: [f64; 3],
    surface_type: String,
    edges: Vec<u64>,
    bounds: usize,
    same_sense: bool,
    support_axis: [i64; 3],
    local_points: Vec<[i64; 3]>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct IntrinsicFaceKey {
    surface_type: String,
    edges: usize,
    bounds: usize,
    same_sense: bool,
    support_axis: [i64; 3],
    local_points: Vec<[i64; 3]>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct CoarseFaceKey {
    span_ticks: i64,
    surface_type: String,
    edges: usize,
}

#[derive(Debug, Clone)]
struct LatticeCandidate {
    axis_index: usize,
    pitch_ticks: i64,
    sites: usize,
    votes: usize,
    site_family_rows: Vec<Vec<(i64, u64)>>,
}

#[derive(Debug, Clone)]
struct Partition {
    centers: Vec<f64>,
    site_of: HashMap<u64, usize>,
    sites: Vec<Vec<u64>>,
    gap_of: HashMap<u64, usize>,
    gaps: Vec<Vec<u64>>,
    stretch: HashSet<u64>,
    fixed: HashSet<u64>,
    score: [i64; 8],
}

#[must_use]
pub fn detect_periodic_chains(entities: &[EntityInstance]) -> Vec<PeriodicChainPattern> {
    let index = build_index(entities);
    let solids = entities
        .iter()
        .filter_map(|entity| {
            let record = simple_record(entity)?;
            (record.name == "MANIFOLD_SOLID_BREP").then_some(entity_id(entity))
        })
        .collect::<Vec<_>>();

    let mut out = Vec::new();
    for solid in solids {
        let Some(face_ids) = solid_faces(solid, entities, &index) else {
            continue;
        };
        if face_ids.len() < 16 {
            continue;
        }

        let mut geoms = HashMap::<u64, FaceGeom>::new();
        for &face in &face_ids {
            if let Some(geom) = face_geometry(face, entities, &index) {
                geoms.insert(face, geom);
            }
        }
        let missing = face_ids.len().saturating_sub(geoms.len());
        if geoms.len() < 16 {
            continue;
        }

        let Some(candidate) = infer_lattice(&geoms) else {
            continue;
        };
        if candidate.votes < MIN_LATTICE_FAMILY_VOTES {
            continue;
        }

        let Some(partition) = choose_partition(&geoms, &candidate) else {
            continue;
        };

        let pattern = build_pattern(solid, &face_ids, &geoms, &candidate, partition, missing);
        out.push(pattern);
    }

    out.sort_by(|a, b| {
        b.read_only_proven
            .cmp(&a.read_only_proven)
            .then_with(|| b.repeat_coverage_ratio.total_cmp(&a.repeat_coverage_ratio))
            .then_with(|| b.sites.cmp(&a.sites))
            .then_with(|| a.solid_id.cmp(&b.solid_id))
    });
    out
}

fn infer_lattice(geoms: &HashMap<u64, FaceGeom>) -> Option<LatticeCandidate> {
    let mut families = HashMap::<(usize, i64, usize), Vec<Vec<(i64, u64)>>>::new();

    for axis in 0..3 {
        let other = other_axes(axis);
        let mut rows = HashMap::<(IntrinsicFaceKey, i64, i64), Vec<(i64, u64)>>::new();

        for geom in geoms.values() {
            let key = IntrinsicFaceKey {
                surface_type: geom.surface_type.clone(),
                edges: geom.edges.len(),
                bounds: geom.bounds,
                same_sense: geom.same_sense,
                support_axis: geom.support_axis,
                local_points: geom.local_points.clone(),
            };
            rows.entry((
                key,
                quantize_mm(geom.center[other[0]]),
                quantize_mm(geom.center[other[1]]),
            ))
            .or_default()
            .push((quantize_mm(geom.center[axis]), geom.id));
        }

        for row in rows.values_mut() {
            row.sort_unstable_by_key(|entry| entry.0);
            row.dedup_by_key(|entry| entry.0);
            if row.len() < MIN_FAMILY_INSTANCES {
                continue;
            }
            let pitch = row[1].0 - row[0].0;
            if pitch <= 0 {
                continue;
            }
            if row.windows(2).any(|pair| pair[1].0 - pair[0].0 != pitch) {
                continue;
            }
            families
                .entry((axis, pitch, row.len()))
                .or_default()
                .push(row.clone());
        }
    }

    families
        .into_iter()
        .map(
            |((axis_index, pitch_ticks, sites), site_family_rows)| LatticeCandidate {
                axis_index,
                pitch_ticks,
                sites,
                votes: site_family_rows.len(),
                site_family_rows,
            },
        )
        .max_by(|a, b| {
            a.votes
                .cmp(&b.votes)
                .then_with(|| a.sites.cmp(&b.sites))
                .then_with(|| b.pitch_ticks.cmp(&a.pitch_ticks))
        })
}
fn choose_partition(
    geoms: &HashMap<u64, FaceGeom>,
    candidate: &LatticeCandidate,
) -> Option<Partition> {
    let axis = candidate.axis_index;
    let pitch_ticks = candidate.pitch_ticks;
    let pitch = pitch_ticks as f64 * GEOM_TOL_MM;
    let sites = candidate.sites;

    let mut phases = HashMap::<i64, Vec<i64>>::new();
    for geom in geoms.values() {
        if geom.span[axis] > pitch + GEOM_TOL_MM {
            continue;
        }
        let tick = quantize_mm(geom.center[axis]);
        phases
            .entry(tick.rem_euclid(pitch_ticks))
            .or_default()
            .push(tick);
    }

    let mut windows = Vec::<(i64, Vec<f64>)>::new();
    for ticks in phases.values() {
        let mut coords = ticks.clone();
        coords.sort_unstable();
        coords.dedup();
        let coord_set = coords.iter().copied().collect::<HashSet<_>>();
        for &start in &coords {
            if coord_set.contains(&(start - pitch_ticks)) {
                continue;
            }
            let mut run = Vec::new();
            let mut x = start;
            while coord_set.contains(&x) {
                run.push(x);
                x += pitch_ticks;
            }
            if run.len() < sites {
                continue;
            }
            for offset in 0..=run.len() - sites {
                let window = &run[offset..offset + sites];
                let observation_score = ticks
                    .iter()
                    .filter(|&&tick| tick >= window[0] && tick <= window[sites - 1])
                    .count() as i64;
                windows.push((
                    observation_score,
                    window
                        .iter()
                        .map(|tick| *tick as f64 * GEOM_TOL_MM)
                        .collect(),
                ));
            }
        }
    }

    let mut best: Option<Partition> = None;
    for (observation_score, centers) in windows {
        let mut partition = classify_faces(geoms, &centers, axis, pitch, candidate)?;
        partition.score = partition_score(geoms, &partition, axis, observation_score);
        if best
            .as_ref()
            .is_none_or(|existing| partition.score > existing.score)
        {
            best = Some(partition);
        }
    }
    if let Some(partition) = best.as_mut() {
        promote_periodic_overlays(geoms, candidate, partition);
    }
    best
}

fn promote_periodic_overlays(
    geoms: &HashMap<u64, FaceGeom>,
    candidate: &LatticeCandidate,
    partition: &mut Partition,
) {
    let sites = partition.centers.len();
    let min_run = sites.saturating_sub(2).max(MIN_FAMILY_INSTANCES);
    if sites < MIN_FAMILY_INSTANCES || candidate.pitch_ticks <= 0 {
        return;
    }

    let Some(pitch_mm) = crate::numeric::exact_i64_to_f64(candidate.pitch_ticks) else {
        return;
    };
    let pitch_mm = pitch_mm * GEOM_TOL_MM;
    let axis = candidate.axis_index;
    let other = other_axes(axis);
    let center_ticks = partition
        .centers
        .iter()
        .map(|&center| quantize_mm(center))
        .collect::<Vec<_>>();
    let mut rows = HashMap::<(IntrinsicFaceKey, i64, i64), Vec<(i64, u64)>>::new();

    for &face in &partition.fixed {
        let geom = &geoms[&face];
        if geom.span[axis] > pitch_mm + GEOM_TOL_MM {
            continue;
        }
        let key = IntrinsicFaceKey {
            surface_type: geom.surface_type.clone(),
            edges: geom.edges.len(),
            bounds: geom.bounds,
            same_sense: geom.same_sense,
            support_axis: geom.support_axis,
            local_points: geom.local_points.clone(),
        };
        rows.entry((
            key,
            quantize_mm(geom.center[other[0]]),
            quantize_mm(geom.center[other[1]]),
        ))
        .or_default()
        .push((quantize_mm(geom.center[axis]), face));
    }

    let mut promote = Vec::<(u64, usize)>::new();
    for row in rows.values_mut() {
        row.sort_unstable_by_key(|entry| entry.0);
        row.dedup_by_key(|entry| entry.0);

        let mut start = 0usize;
        while start < row.len() {
            let mut end = start + 1;
            while end < row.len() && row[end].0 - row[end - 1].0 == candidate.pitch_ticks {
                end += 1;
            }
            let run = &row[start..end];
            if run.len() >= min_run
                && run.len() <= sites
                && let Some(site_start) =
                    overlay_site_alignment(run, &center_ticks, sites, candidate.pitch_ticks)
            {
                for (index, &(_, face)) in run.iter().enumerate() {
                    promote.push((face, site_start + index));
                }
            }
            start = end;
        }
    }

    promote.sort_unstable();
    if promote.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return;
    }
    for (face, site) in promote {
        if !partition.fixed.remove(&face) {
            continue;
        }
        partition.site_of.insert(face, site);
        partition.sites[site].push(face);
    }
    for site in &mut partition.sites {
        site.sort_unstable();
    }
}

fn overlay_site_alignment(
    run: &[(i64, u64)],
    center_ticks: &[i64],
    sites: usize,
    pitch_ticks: i64,
) -> Option<usize> {
    let mut best: Option<(i64, usize)> = None;
    let mut ambiguous = false;
    for (site_start, &site_tick) in center_ticks.iter().take(sites - run.len() + 1).enumerate() {
        let score = (run[0].0 - site_tick).abs();
        if score > pitch_ticks / 2 + 1 {
            continue;
        }
        match best {
            None => {
                best = Some((score, site_start));
                ambiguous = false;
            }
            Some((best_score, _)) if score < best_score => {
                best = Some((score, site_start));
                ambiguous = false;
            }
            Some((best_score, best_start)) if score == best_score && site_start != best_start => {
                ambiguous = true;
            }
            Some(_) => {}
        }
    }
    if ambiguous {
        None
    } else {
        best.map(|(_, site_start)| site_start)
    }
}

fn classify_faces(
    geoms: &HashMap<u64, FaceGeom>,
    centers: &[f64],
    axis: usize,
    pitch: f64,
    candidate: &LatticeCandidate,
) -> Option<Partition> {
    let sites_count = centers.len();
    if sites_count < 2 {
        return None;
    }

    let mut site_of = HashMap::<u64, usize>::new();
    let mut sites = vec![Vec::<u64>::new(); sites_count];
    let mut stretch = HashSet::<u64>::new();
    let mut remaining = geoms.keys().copied().collect::<HashSet<_>>();

    // The lattice inference already proved exact translated face families. Use
    // those families as site membership instead of treating every face that
    // happens to fit inside a site window as periodic. The latter swallows
    // unique connector end-cap/housing geometry into the end sites.
    let center_ticks = centers.iter().map(|&x| quantize_mm(x)).collect::<Vec<_>>();
    for row in &candidate.site_family_rows {
        if row.len() != sites_count {
            continue;
        }
        let offset = row[0].0 - center_ticks[0];
        if !row
            .iter()
            .enumerate()
            .all(|(site, (tick, _))| *tick - center_ticks[site] == offset)
        {
            continue;
        }
        for (site, &(_, face)) in row.iter().enumerate() {
            if site_of.insert(face, site).is_some() {
                return None;
            }
            sites[site].push(face);
            remaining.remove(&face);
        }
    }

    for geom in geoms.values() {
        if remaining.contains(&geom.id) && geom.span[axis] > pitch + GEOM_TOL_MM {
            stretch.insert(geom.id);
            remaining.remove(&geom.id);
        }
    }

    let mids = centers
        .windows(2)
        .map(|pair| f64::midpoint(pair[0], pair[1]))
        .collect::<Vec<_>>();
    let mut gap_candidates = vec![Vec::<u64>::new(); mids.len()];

    for &face in &remaining {
        let geom = &geoms[&face];
        let (gap_index, distance) = mids
            .iter()
            .enumerate()
            .map(|(index, &mid)| (index, (geom.center[axis] - mid).abs()))
            .min_by(|a, b| a.1.total_cmp(&b.1))?;
        if distance > pitch {
            continue;
        }
        if geom.lo[axis] < mids[gap_index] + GEOM_TOL_MM
            && geom.hi[axis] > mids[gap_index] - GEOM_TOL_MM
            && geom.lo[axis] >= centers[gap_index] - GEOM_TOL_MM
            && geom.hi[axis] <= centers[gap_index + 1] + GEOM_TOL_MM
        {
            gap_candidates[gap_index].push(face);
        }
    }

    let mut census_hist = HashMap::<Vec<(CoarseFaceKey, usize)>, usize>::new();
    for faces in &gap_candidates {
        let mut census = BTreeMap::<CoarseFaceKey, usize>::new();
        for &face in faces {
            *census.entry(coarse_key(&geoms[&face], axis)).or_insert(0) += 1;
        }
        let canonical = census.into_iter().collect::<Vec<_>>();
        *census_hist.entry(canonical).or_insert(0) += 1;
    }

    let generic = census_hist
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(census, _)| census)
        .unwrap_or_default();
    let generic_map = generic.into_iter().collect::<HashMap<_, _>>();

    let mut gap_of = HashMap::<u64, usize>::new();
    let mut gaps = vec![Vec::<u64>::new(); mids.len()];
    for (index, faces) in gap_candidates.iter().enumerate() {
        let mut need = generic_map.clone();
        let mut ordered = faces.clone();
        ordered.sort_unstable();
        for face in ordered {
            let key = coarse_key(&geoms[&face], axis);
            let Some(count) = need.get_mut(&key) else {
                continue;
            };
            if *count == 0 {
                continue;
            }
            *count -= 1;
            gap_of.insert(face, index);
            gaps[index].push(face);
            remaining.remove(&face);
        }
    }

    for site in &mut sites {
        site.sort_unstable();
    }
    for gap in &mut gaps {
        gap.sort_unstable();
    }

    Some(Partition {
        centers: centers.to_vec(),
        site_of,
        sites,
        gap_of,
        gaps,
        stretch,
        fixed: remaining,
        score: [0; 8],
    })
}

fn partition_score(
    geoms: &HashMap<u64, FaceGeom>,
    partition: &Partition,
    axis: usize,
    observation_score: i64,
) -> [i64; 8] {
    let site_counts = partition.sites.iter().map(Vec::len).collect::<Vec<_>>();
    let gap_counts = partition.gaps.iter().map(Vec::len).collect::<Vec<_>>();

    let site_mode_n = mode_frequency(&site_counts) as i64;
    let gap_mode_n = mode_frequency(&gap_counts) as i64;
    let site_sym = symmetric_abs_difference(&site_counts) as i64;
    let gap_sym = symmetric_abs_difference(&gap_counts) as i64;

    let midpoint = f64::midpoint(
        partition.centers[0],
        partition.centers[partition.centers.len() - 1],
    );
    let mut negative = 0usize;
    let mut positive = 0usize;
    let mut middle = 0usize;
    for face in &partition.fixed {
        let x = geoms[face].center[axis];
        if x < midpoint - GEOM_TOL_MM {
            negative += 1;
        } else if x > midpoint + GEOM_TOL_MM {
            positive += 1;
        } else {
            middle += 1;
        }
    }

    [
        -(negative.abs_diff(positive) as i64),
        -site_sym,
        -gap_sym,
        gap_mode_n,
        site_mode_n,
        -(partition.fixed.len() as i64),
        -(middle as i64),
        observation_score,
    ]
}

fn build_pattern(
    solid_id: u64,
    all_face_ids: &[u64],
    geoms: &HashMap<u64, FaceGeom>,
    candidate: &LatticeCandidate,
    partition: Partition,
    faces_without_geometry: usize,
) -> PeriodicChainPattern {
    let axis_index = candidate.axis_index;
    let mut axis = [0.0; 3];
    axis[axis_index] = 1.0;

    let site_face_counts = partition.sites.iter().map(Vec::len).collect::<Vec<_>>();
    let gap_face_counts = partition.gaps.iter().map(Vec::len).collect::<Vec<_>>();
    let interior_site_face_count = mode_value(&site_face_counts).unwrap_or(0);
    let interior_gap_face_count = mode_value(&gap_face_counts).unwrap_or(0);

    // Positive-end editing preserves the prefix through site N-3 and moves
    // the tail beginning at the gap before site N-2. Classify fixed geometry
    // against that actual edit seam, not the whole-chain midpoint: a central
    // key/housing feature must remain stationary even if it happens to lie on
    // the positive half of the model.
    let tail_split = f64::midpoint(
        partition.centers[partition.centers.len() - 3],
        partition.centers[partition.centers.len() - 2],
    );
    let mut fixed_negative = Vec::new();
    let mut fixed_middle = Vec::new();
    let mut fixed_positive = Vec::new();
    for &face in &partition.fixed {
        let x = geoms[&face].center[axis_index];
        if x > tail_split + GEOM_TOL_MM {
            fixed_positive.push(face);
        } else if x < tail_split - GEOM_TOL_MM {
            fixed_negative.push(face);
        } else {
            fixed_middle.push(face);
        }
    }
    fixed_negative.sort_unstable();
    fixed_middle.sort_unstable();
    fixed_positive.sort_unstable();

    let mut edge_faces = HashMap::<u64, Vec<u64>>::new();
    for geom in geoms.values() {
        for &edge in &geom.edges {
            edge_faces.entry(edge).or_default().push(geom.id);
        }
    }

    let mut edge_category_counts = BTreeMap::<String, usize>::new();
    let mut site_adj = vec![BTreeMap::<String, usize>::new(); partition.sites.len()];
    let mut gap_adj = vec![BTreeMap::<String, usize>::new(); partition.gaps.len()];
    let mut nonmanifold_edges = 0usize;
    let mut cross_site_edges = 0usize;
    let mut nonlocal_cross_site_edges = 0usize;
    let mut adjacent_site_edge_counts = vec![0usize; partition.sites.len().saturating_sub(1)];

    for faces in edge_faces.values() {
        if faces.len() != 2 {
            nonmanifold_edges += 1;
            continue;
        }
        let a = faces[0];
        let b = faces[1];
        let ca = face_category(a, &partition);
        let cb = face_category(b, &partition);

        let mut names = [ca.0, cb.0];
        names.sort_unstable();
        *edge_category_counts
            .entry(format!("{}|{}", names[0], names[1]))
            .or_insert(0) += 1;

        if ca.0 == "site" && cb.0 == "site" && ca.1 != cb.1 {
            cross_site_edges += 1;
            if let (Some(a_site), Some(b_site)) = (ca.1, cb.1) {
                if a_site.abs_diff(b_site) == 1 {
                    adjacent_site_edge_counts[a_site.min(b_site)] += 1;
                } else {
                    nonlocal_cross_site_edges += 1;
                }
            } else {
                nonlocal_cross_site_edges += 1;
            }
        }

        add_adjacency(&mut site_adj, &mut gap_adj, ca, cb);
        add_adjacency(&mut site_adj, &mut gap_adj, cb, ca);
    }

    let periodic_faces = partition.site_of.len() + partition.gap_of.len();
    let repeat_coverage_ratio = periodic_faces as f64 / all_face_ids.len().max(1) as f64;

    // The editor preserves both terminal regions and clones only an interior
    // (gap + site) unit. Terminal site/end-cap geometry may therefore be
    // intentionally asymmetric; requiring whole-chain mirror symmetry rejects
    // valid counted parts such as keyed or end-featured connector headers.
    let gap_symmetric = symmetric_abs_difference(&gap_face_counts) == 0;
    let gap_consistent = gap_face_counts
        .iter()
        .all(|&count| count == interior_gap_face_count);
    let interior_consistent = if partition.sites.len() > 4 {
        partition.sites[2..partition.sites.len() - 2]
            .iter()
            .all(|faces| faces.len() == interior_site_face_count)
    } else {
        site_face_counts
            .iter()
            .all(|&count| count == interior_site_face_count)
    };

    let adjacent_site_consistent = adjacent_site_edge_counts
        .first()
        .is_none_or(|first| adjacent_site_edge_counts.iter().all(|count| count == first));

    let complete_partition = faces_without_geometry == 0
        && partition.site_of.len()
            + partition.gap_of.len()
            + partition.stretch.len()
            + partition.fixed.len()
            == all_face_ids.len();

    let read_only_proven = candidate.votes >= MIN_LATTICE_FAMILY_VOTES
        && complete_partition
        && nonmanifold_edges == 0
        && nonlocal_cross_site_edges == 0
        && adjacent_site_consistent
        && gap_symmetric
        && fixed_middle.is_empty()
        && gap_consistent
        && interior_consistent
        && repeat_coverage_ratio >= 0.5;

    let mut stretch_face_ids = partition.stretch.into_iter().collect::<Vec<_>>();
    stretch_face_ids.sort_unstable();

    PeriodicChainPattern {
        solid_id,
        axis,
        pitch_mm: candidate.pitch_ticks as f64 * GEOM_TOL_MM,
        sites: candidate.sites,
        site_centers_mm: partition.centers,
        lattice_face_family_votes: candidate.votes,
        complete_partition,
        read_only_proven,
        faces_without_geometry,
        nonmanifold_edges,
        cross_site_edges,
        nonlocal_cross_site_edges,
        adjacent_site_edge_counts,
        site_face_counts,
        interior_site_face_count,
        gap_face_counts,
        interior_gap_face_count,
        site_face_ids: partition.sites,
        gap_face_ids: partition.gaps,
        stretch_face_ids,
        fixed_negative_face_ids: fixed_negative,
        fixed_middle_face_ids: fixed_middle,
        fixed_positive_face_ids: fixed_positive,
        repeat_coverage_ratio,
        edge_category_counts,
        site_adjacency_signatures: site_adj,
        gap_adjacency_signatures: gap_adj,
    }
}

fn add_adjacency(
    site_adj: &mut [BTreeMap<String, usize>],
    gap_adj: &mut [BTreeMap<String, usize>],
    source: (&'static str, Option<usize>),
    target: (&'static str, Option<usize>),
) {
    match source {
        ("site", Some(site)) => {
            let label = match target {
                ("site", Some(other)) => format!("site:{:+}", other as isize - site as isize),
                ("gap", Some(other)) => format!("gap:{:+}", other as isize - site as isize),
                (name, _) => name.to_string(),
            };
            *site_adj[site].entry(label).or_insert(0) += 1;
        }
        ("gap", Some(gap)) => {
            let label = match target {
                ("site", Some(other)) => format!("site:{:+}", other as isize - gap as isize),
                ("gap", Some(other)) => format!("gap:{:+}", other as isize - gap as isize),
                (name, _) => name.to_string(),
            };
            *gap_adj[gap].entry(label).or_insert(0) += 1;
        }
        _ => {}
    }
}

fn face_category(face: u64, partition: &Partition) -> (&'static str, Option<usize>) {
    if let Some(&site) = partition.site_of.get(&face) {
        ("site", Some(site))
    } else if let Some(&gap) = partition.gap_of.get(&face) {
        ("gap", Some(gap))
    } else if partition.stretch.contains(&face) {
        ("stretch", None)
    } else if partition.fixed.contains(&face) {
        ("fixed", None)
    } else {
        ("unknown", None)
    }
}

fn face_geometry(
    face: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<FaceGeom> {
    let record = simple_record(entities.get(*index.get(&face)?)?)?;
    if record.name != "ADVANCED_FACE" {
        return None;
    }
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let bounds = match params.get(1)? {
        Parameter::List(items) => items.len(),
        _ => return None,
    };
    let surface = entity_ref_value(params.get(2)?)?;
    let same_sense = matches!(params.get(3)?, Parameter::Enumeration(value) if value == "T");
    let surface_record = simple_record(entities.get(*index.get(&surface)?)?)?;
    let surface_type = surface_record.name.clone();
    let support_axis = surface_axis_signature(surface_record, entities, index);

    let edges = face_edges(face, entities, index)?;
    if edges.is_empty() {
        return None;
    }
    let mut points = HashSet::<u64>::new();
    for &edge in &edges {
        let edge_record = simple_record(entities.get(*index.get(&edge)?)?)?;
        if edge_record.name != "EDGE_CURVE" {
            return None;
        }
        let Parameter::List(edge_params) = &edge_record.parameter else {
            return None;
        };
        for param in [edge_params.get(1)?, edge_params.get(2)?] {
            let vertex = entity_ref_value(param)?;
            let vertex_record = simple_record(entities.get(*index.get(&vertex)?)?)?;
            if vertex_record.name != "VERTEX_POINT" {
                return None;
            }
            let Parameter::List(vertex_params) = &vertex_record.parameter else {
                return None;
            };
            points.insert(entity_ref_value(vertex_params.get(1)?)?);
        }
    }
    // A valid face loop can contain two EDGE_CURVEs joining the same two
    // vertices (for example two arcs forming a closed planar lens/slot).
    // Two distinct topological points are therefore sufficient for the
    // translation/lattice analysis. One-point/full-circle faces still need a
    // richer curve-extrema path before they can participate.
    if points.len() < 2 {
        return None;
    }

    let coords = points
        .into_iter()
        .map(|point| cartesian_point(point, entities, index))
        .collect::<Option<Vec<_>>>()?;
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for point in &coords {
        for axis in 0..3 {
            lo[axis] = lo[axis].min(point[axis]);
            hi[axis] = hi[axis].max(point[axis]);
        }
    }
    let center = [
        f64::midpoint(lo[0], hi[0]),
        f64::midpoint(lo[1], hi[1]),
        f64::midpoint(lo[2], hi[2]),
    ];
    let span = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let mut local_points = coords
        .iter()
        .map(|point| {
            [
                quantize_mm(point[0] - center[0]),
                quantize_mm(point[1] - center[1]),
                quantize_mm(point[2] - center[2]),
            ]
        })
        .collect::<Vec<_>>();
    local_points.sort_unstable();
    local_points.dedup();

    Some(FaceGeom {
        id: face,
        lo,
        hi,
        center,
        span,
        surface_type,
        edges,
        bounds,
        same_sense,
        support_axis,
        local_points,
    })
}

fn surface_axis_signature(
    surface: &Record,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> [i64; 3] {
    if surface.name != "PLANE"
        && surface.name != "CYLINDRICAL_SURFACE"
        && surface.name != "CONICAL_SURFACE"
    {
        return [0; 3];
    }
    let Some(params) = list_params(surface) else {
        return [0; 3];
    };
    let Some(placement) = params.get(1).and_then(entity_ref_value) else {
        return [0; 3];
    };
    let Some(place_record) = index
        .get(&placement)
        .and_then(|&i| simple_record(entities.get(i)?))
    else {
        return [0; 3];
    };
    if place_record.name != "AXIS2_PLACEMENT_3D" {
        return [0; 3];
    }
    let Some(place_params) = list_params(place_record) else {
        return [0; 3];
    };
    let Some(direction) = place_params.get(2).and_then(entity_ref_value) else {
        return [0; 3];
    };
    let Some(dir_record) = index
        .get(&direction)
        .and_then(|&i| simple_record(entities.get(i)?))
    else {
        return [0; 3];
    };
    if dir_record.name != "DIRECTION" {
        return [0; 3];
    }
    let Some(dir_params) = list_params(dir_record) else {
        return [0; 3];
    };
    let Some(Parameter::List(coords)) = dir_params.get(1) else {
        return [0; 3];
    };
    if coords.len() != 3 {
        return [0; 3];
    }
    [
        quantize_dir(numeric_value(&coords[0]).unwrap_or(0.0)),
        quantize_dir(numeric_value(&coords[1]).unwrap_or(0.0)),
        quantize_dir(numeric_value(&coords[2]).unwrap_or(0.0)),
    ]
}

fn coarse_key(geom: &FaceGeom, axis: usize) -> CoarseFaceKey {
    CoarseFaceKey {
        span_ticks: quantize_mm(geom.span[axis]),
        surface_type: geom.surface_type.clone(),
        edges: geom.edges.len(),
    }
}

fn mode_frequency(values: &[usize]) -> usize {
    let mut counts = HashMap::<usize, usize>::new();
    for &value in values {
        *counts.entry(value).or_insert(0) += 1;
    }
    counts.into_values().max().unwrap_or(0)
}

fn mode_value(values: &[usize]) -> Option<usize> {
    let mut counts = HashMap::<usize, usize>::new();
    for &value in values {
        *counts.entry(value).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(value, count)| (count, value))
        .map(|(value, _)| value)
}

fn symmetric_abs_difference(values: &[usize]) -> usize {
    (0..values.len() / 2)
        .map(|index| values[index].abs_diff(values[values.len() - 1 - index]))
        .sum()
}

const fn other_axes(axis: usize) -> [usize; 2] {
    match axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    }
}

fn quantize_mm(value: f64) -> i64 {
    (value / GEOM_TOL_MM).round() as i64
}

fn quantize_dir(value: f64) -> i64 {
    (value / 1.0e-10).round() as i64
}

fn solid_faces(
    solid: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<u64>> {
    let record = simple_record(entities.get(*index.get(&solid)?)?)?;
    let params = list_params(record)?;
    let shell = params.get(1).and_then(entity_ref_value)?;
    let shell_record = simple_record(entities.get(*index.get(&shell)?)?)?;
    if shell_record.name != "CLOSED_SHELL" {
        return None;
    }
    let shell_params = list_params(shell_record)?;
    let Parameter::List(face_refs) = shell_params.get(1)? else {
        return None;
    };
    face_refs.iter().map(entity_ref_value).collect()
}

fn face_edges(
    face: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<u64>> {
    let record = simple_record(entities.get(*index.get(&face)?)?)?;
    let params = list_params(record)?;
    let Parameter::List(bound_refs) = params.get(1)? else {
        return None;
    };
    let mut edges = HashSet::new();
    for bound_ref in bound_refs {
        let bound = entity_ref_value(bound_ref)?;
        let bound_record = simple_record(entities.get(*index.get(&bound)?)?)?;
        if bound_record.name != "FACE_BOUND" && bound_record.name != "FACE_OUTER_BOUND" {
            return None;
        }
        let bound_params = list_params(bound_record)?;
        let loop_id = bound_params.get(1).and_then(entity_ref_value)?;
        let loop_record = simple_record(entities.get(*index.get(&loop_id)?)?)?;
        if loop_record.name != "EDGE_LOOP" {
            return None;
        }
        let loop_params = list_params(loop_record)?;
        let Parameter::List(oriented_refs) = loop_params.get(1)? else {
            return None;
        };
        for oriented_ref in oriented_refs {
            let oriented = entity_ref_value(oriented_ref)?;
            let oriented_record = simple_record(entities.get(*index.get(&oriented)?)?)?;
            if oriented_record.name != "ORIENTED_EDGE" {
                return None;
            }
            let oriented_params = list_params(oriented_record)?;
            edges.insert(oriented_params.get(3).and_then(entity_ref_value)?);
        }
    }
    let mut out = edges.into_iter().collect::<Vec<_>>();
    out.sort_unstable();
    Some(out)
}

fn cartesian_point(
    point: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let record = simple_record(entities.get(*index.get(&point)?)?)?;
    if record.name != "CARTESIAN_POINT" {
        return None;
    }
    let params = list_params(record)?;
    let Parameter::List(coords) = params.get(1)? else {
        return None;
    };
    if coords.len() != 3 {
        return None;
    }
    Some([
        numeric_value(&coords[0])?,
        numeric_value(&coords[1])?,
        numeric_value(&coords[2])?,
    ])
}

fn list_params(record: &Record) -> Option<&[Parameter]> {
    match &record.parameter {
        Parameter::List(params) => Some(params),
        _ => None,
    }
}

fn numeric_value(param: &Parameter) -> Option<f64> {
    match param {
        Parameter::Integer(value) => crate::numeric::exact_i64_to_f64(*value),
        Parameter::Real(value) => Some(*value),
        _ => None,
    }
}

const fn entity_ref_value(param: &Parameter) -> Option<u64> {
    match param {
        Parameter::Ref(Name::Entity(id)) => Some(*id),
        _ => None,
    }
}

fn build_index(entities: &[EntityInstance]) -> HashMap<u64, usize> {
    entities
        .iter()
        .enumerate()
        .map(|(index, entity)| (entity_id(entity), index))
        .collect()
}

const fn simple_record(entity: &EntityInstance) -> Option<&Record> {
    match entity {
        EntityInstance::Simple { record, .. } => Some(record),
        EntityInstance::Complex { .. } => None,
    }
}

const fn entity_id(entity: &EntityInstance) -> u64 {
    match entity {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_face(id: u64, center: [f64; 3]) -> FaceGeom {
        FaceGeom {
            id,
            lo: [center[0] - 1.12, center[1], center[2]],
            hi: [center[0] + 1.12, center[1], center[2]],
            center,
            span: [2.24, 0.0, 0.0],
            surface_type: "PLANE".to_string(),
            edges: vec![id * 10, id * 10 + 1, id * 10 + 2, id * 10 + 3],
            bounds: 1,
            same_sense: true,
            support_axis: [0, 0, quantize_dir(1.0)],
            local_points: vec![[quantize_mm(-1.12), 0, 0], [quantize_mm(1.12), 0, 0]],
        }
    }

    #[test]
    fn promotes_long_exact_overlay_row_into_sites() -> anyhow::Result<()> {
        let pitch = 2.54;
        let centers = (0_i32..6).map(|i| f64::from(i) * pitch).collect::<Vec<_>>();
        let mut geoms = HashMap::new();
        let mut fixed = HashSet::new();
        for (index, &center) in centers.iter().take(4).enumerate() {
            let id = u64::try_from(index)? + 1;
            geoms.insert(id, test_face(id, [center + 0.4, 3.81, 8.5]));
            fixed.insert(id);
        }
        let candidate = LatticeCandidate {
            axis_index: 0,
            pitch_ticks: quantize_mm(pitch),
            sites: centers.len(),
            votes: MIN_LATTICE_FAMILY_VOTES,
        };
        let mut partition = Partition {
            centers,
            site_of: HashMap::new(),
            sites: vec![Vec::new(); 6],
            gap_of: HashMap::new(),
            gaps: vec![Vec::new(); 5],
            stretch: HashSet::new(),
            fixed,
            score: [0; 8],
        };

        promote_periodic_overlays(&geoms, &candidate, &mut partition);

        assert!(partition.fixed.is_empty());
        assert_eq!(
            partition.sites.iter().map(Vec::len).collect::<Vec<_>>(),
            [1, 1, 1, 1, 0, 0]
        );
        Ok(())
    }

    #[test]
    fn ambiguous_half_pitch_overlay_row_stays_fixed() -> anyhow::Result<()> {
        let pitch = 2.54;
        let centers = (0_i32..6).map(|i| f64::from(i) * pitch).collect::<Vec<_>>();
        let mut geoms = HashMap::new();
        let mut fixed = HashSet::new();
        for index in 0_i32..4 {
            let id = u64::try_from(index)? + 1;
            let x = pitch * (f64::from(index) + 0.5);
            geoms.insert(id, test_face(id, [x, 3.81, 8.5]));
            fixed.insert(id);
        }
        let candidate = LatticeCandidate {
            axis_index: 0,
            pitch_ticks: quantize_mm(pitch),
            sites: centers.len(),
            votes: MIN_LATTICE_FAMILY_VOTES,
        };
        let mut partition = Partition {
            centers,
            site_of: HashMap::new(),
            sites: vec![Vec::new(); 6],
            gap_of: HashMap::new(),
            gaps: vec![Vec::new(); 5],
            stretch: HashSet::new(),
            fixed: fixed.clone(),
            score: [0; 8],
        };

        promote_periodic_overlays(&geoms, &candidate, &mut partition);

        assert_eq!(partition.fixed, fixed);
        assert!(partition.sites.iter().all(Vec::is_empty));
        Ok(())
    }

    #[test]
    fn symmetry_metric_is_zero_for_palindrome() {
        assert_eq!(symmetric_abs_difference(&[74, 71, 70, 71, 74]), 0);
    }

    #[test]
    fn symmetry_metric_detects_shifted_origin() {
        assert!(symmetric_abs_difference(&[78, 71, 70, 71, 74]) > 0);
    }
}
