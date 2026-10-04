#[path = "../instances.rs"]
mod instances;
#[path = "../numeric.rs"]
mod numeric;
#[path = "../units.rs"]
mod units;

use anyhow::{Context, Result, bail};
use encoding_rs::GBK;
use std::{borrow::Cow, collections::HashMap, env, fs};

fn decode_input(input: &[u8]) -> Result<Cow<'_, str>> {
    if let Ok(text) = std::str::from_utf8(input) {
        return Ok(Cow::Borrowed(text));
    }
    let (decoded, _, had_errors) = GBK.decode(input);
    if had_errors {
        bail!("STEP input is neither valid UTF-8 nor valid GBK");
    }
    Ok(decoded)
}

fn main() -> Result<()> {
    let mut args = env::args();
    let _ = args.next();
    let path = args.next().ok_or_else(|| anyhow::anyhow!("FILE"))?;
    let a: u64 = args.next().ok_or_else(|| anyhow::anyhow!("A"))?.parse()?;
    let b: u64 = args.next().ok_or_else(|| anyhow::anyhow!("B"))?.parse()?;
    let bytes = fs::read(&path).with_context(|| format!("read {path}"))?;
    let text = decode_input(&bytes)?;
    let exchange = ruststep::parser::parse(&text)?;
    let entities = &exchange
        .data
        .first()
        .ok_or_else(|| anyhow::anyhow!("no DATA"))?
        .entities;
    let index: HashMap<u64, usize> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| (instances::entity_id(e), i))
        .collect();
    let (ka, ca, qa, cla) = instances::solid_shape_key(a, entities, &index)
        .ok_or_else(|| anyhow::anyhow!("no key A"))?;
    let (kb, cb, qb, clb) = instances::solid_shape_key(b, entities, &index)
        .ok_or_else(|| anyhow::anyhow!("no key B"))?;
    println!(
        "A center={ca:?} q={qa} closure={cla} counts={}/{}/{}/{}",
        ka.vertices, ka.edges, ka.oriented_edges, ka.faces
    );
    println!(
        "B center={cb:?} q={qb} closure={clb} counts={}/{}/{}/{}",
        kb.vertices, kb.edges, kb.oriented_edges, kb.faces
    );
    println!(
        "equal={} points={} edge_geom={} face_geom={} topology={}",
        ka == kb,
        ka.points == kb.points,
        ka.edge_geometry == kb.edge_geometry,
        ka.face_geometry == kb.face_geometry,
        ka.topology == kb.topology
    );
    if ka.points != kb.points {
        println!("point lens {} {}", ka.points.len(), kb.points.len());
        for (i, (x, y)) in ka.points.iter().zip(&kb.points).enumerate() {
            if x != y {
                println!("POINT_DIFF {i} {x:?} {y:?}");
                if i > 20 {
                    break;
                }
            }
        }
    }
    if ka.edge_geometry != kb.edge_geometry {
        println!(
            "EDGE_GEOM lens {} {}",
            ka.edge_geometry.len(),
            kb.edge_geometry.len()
        );
        for (i, (x, y)) in ka.edge_geometry.iter().zip(&kb.edge_geometry).enumerate() {
            if x != y {
                println!("EDGE_DIFF {i} {x:?} {y:?}");
                if i > 20 {
                    break;
                }
            }
        }
    }
    if ka.face_geometry != kb.face_geometry {
        println!(
            "FACE_GEOM lens {} {}",
            ka.face_geometry.len(),
            kb.face_geometry.len()
        );
        for (i, (x, y)) in ka.face_geometry.iter().zip(&kb.face_geometry).enumerate() {
            if x != y {
                println!("FACE_DIFF {i} {x:?} {y:?}");
                if i > 20 {
                    break;
                }
            }
        }
    }
    if ka.topology != kb.topology {
        println!("TOPO lens {} {}", ka.topology.len(), kb.topology.len());
        let aa = ka.topology.as_bytes();
        let bb = kb.topology.as_bytes();
        let pos = aa
            .iter()
            .zip(bb)
            .position(|(x, y)| x != y)
            .unwrap_or(aa.len().min(bb.len()));
        let lo = pos.saturating_sub(120);
        let hi = (pos + 500).min(ka.topology.len()).min(kb.topology.len());
        println!(
            "TOPO_DIFF_AT {pos}\nA={}\nB={}",
            &ka.topology[lo..hi],
            &kb.topology[lo..hi]
        );
    }
    Ok(())
}
