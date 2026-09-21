#[path = "../instances.rs"]
mod instances;

use anyhow::{Context, Result};
use ruststep::ast::{EntityInstance, Parameter, Record};
use std::collections::{HashSet, VecDeque};
use std::env;
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};

fn entity_id(e: &EntityInstance) -> u64 {
    match e {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id,
    }
}

fn record_name(e: &EntityInstance) -> Option<&str> {
    instances::simple_record(e).map(|r| r.name.as_str())
}

fn descendants(
    root: u64,
    entities: &[EntityInstance],
    index: &std::collections::HashMap<u64, usize>,
) -> HashSet<u64> {
    let mut seen = HashSet::new();
    let mut q = VecDeque::from([root]);
    while let Some(id) = q.pop_front() {
        if !seen.insert(id) {
            continue;
        }
        let Some(&idx) = index.get(&id) else {
            continue;
        };
        instances::visit_entity_refs(&entities[idx], &mut |child| {
            if index.contains_key(&child) && !seen.contains(&child) {
                q.push_back(child);
            }
        });
    }
    seen
}

fn vertex_center(
    face: u64,
    entities: &[EntityInstance],
    index: &std::collections::HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let mut point_ids = HashSet::new();
    for id in descendants(face, entities, index) {
        let &idx = index.get(&id)?;
        let Some(rec) = instances::simple_record(&entities[idx]) else {
            continue;
        };
        if rec.name != "VERTEX_POINT" {
            continue;
        }
        let pid = instances::nth_entity_ref(&rec.parameter, 1)?;
        point_ids.insert(pid);
    }
    if point_ids.is_empty() {
        return None;
    }
    let mut sum = [0.0; 3];
    let mut n = 0usize;
    for pid in point_ids {
        if let Some(p) = instances::cartesian_point(pid, entities, index) {
            sum[0] += p[0];
            sum[1] += p[1];
            sum[2] += p[2];
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    Some([sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64])
}

fn surface_type(
    face: &EntityInstance,
    entities: &[EntityInstance],
    index: &std::collections::HashMap<u64, usize>,
) -> Option<String> {
    let rec = instances::simple_record(face)?;
    let Parameter::List(params) = &rec.parameter else {
        return None;
    };
    let sid = instances::entity_ref_value(params.get(2)?)?;
    let &idx = index.get(&sid)?;
    match &entities[idx] {
        EntityInstance::Simple { record, .. } => Some(record.name.clone()),
        EntityInstance::Complex { subsuper, .. } => Some(format!(
            "COMPLEX[{}]",
            subsuper
                .0
                .iter()
                .map(|r| r.name.as_str())
                .collect::<Vec<_>>()
                .join("|")
        )),
    }
}

fn main() -> Result<()> {
    let path = env::args().nth(1).context("usage: face-scan FILE.step")?;
    let text = fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
    let exchange = ruststep::parser::parse(&text).context("parse STEP")?;
    let entities = exchange
        .data
        .into_iter()
        .next()
        .context("no DATA section")?
        .entities;
    let index = instances::build_index(&entities);

    for e in &entities {
        let Some("ADVANCED_FACE") = record_name(e) else {
            continue;
        };
        let id = entity_id(e);
        let Some(center) = vertex_center(id, &entities, &index) else {
            continue;
        };
        let mut best: Option<(String, u8)> = None;
        for quarter in 0..4u8 {
            let Some(sig) =
                instances::face_topology_signature(id, &entities, &index, center, quarter)
            else {
                continue;
            };
            if best
                .as_ref()
                .is_none_or(|(current, _)| sig.as_str() < current.as_str())
            {
                best = Some((sig, quarter));
            }
        }
        let Some((sig, quarter)) = best else {
            continue;
        };
        let mut h = DefaultHasher::new();
        sig.hash(&mut h);
        let hash = h.finish();
        let st = surface_type(e, &entities, &index).unwrap_or_else(|| "?".to_string());
        println!(
            "{id}\t{hash:016x}\t{}\t{:.15}\t{:.15}\t{:.15}\t{}\t{}",
            sig.len(),
            center[0],
            center[1],
            center[2],
            st,
            quarter
        );
    }
    Ok(())
}
