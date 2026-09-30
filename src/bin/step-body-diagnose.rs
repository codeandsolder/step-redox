#[expect(
    dead_code,
    reason = "diagnostic target reuses the production instance parser but exercises only its diagnosis subset"
)]
#[path = "../instances.rs"]
mod instances;
#[path = "../numeric.rs"]
mod numeric;

use anyhow::{Context, Result, bail};
use encoding_rs::GBK;
use ruststep::ast::{EntityInstance, Parameter};
use std::{borrow::Cow, env, fs, path::Path};

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

fn entity_name(entity: &EntityInstance) -> String {
    match entity {
        EntityInstance::Simple { record, .. } => record.name.clone(),
        EntityInstance::Complex { subsuper, .. } => subsuper
            .0
            .iter()
            .map(|record| record.name.as_str())
            .collect::<Vec<_>>()
            .join("+"),
    }
}

fn refs_in_parameter(param: &Parameter, out: &mut Vec<u64>) {
    match param {
        Parameter::Ref(ruststep::ast::Name::Entity(id)) => out.push(*id),
        Parameter::List(items) => {
            for item in items {
                refs_in_parameter(item, out);
            }
        }
        Parameter::Typed { parameter, .. } => refs_in_parameter(parameter, out),
        _ => {}
    }
}

fn record_refs(entity: &EntityInstance) -> Vec<u64> {
    let mut refs = Vec::new();
    match entity {
        EntityInstance::Simple { record, .. } => refs_in_parameter(&record.parameter, &mut refs),
        EntityInstance::Complex { subsuper, .. } => {
            for record in &subsuper.0 {
                refs_in_parameter(&record.parameter, &mut refs);
            }
        }
    }
    refs
}

fn enum_bool_like(parameter: &Parameter) -> bool {
    matches!(
        parameter,
        Parameter::Enumeration(value) if value == "T" || value == "F"
    )
}

fn vertex_point_ok(
    vertex_id: u64,
    entities: &[EntityInstance],
    index: &std::collections::HashMap<u64, usize>,
) -> bool {
    let Some(&idx) = index.get(&vertex_id) else {
        return false;
    };
    let Some(record) = instances::simple_record(&entities[idx]) else {
        return false;
    };
    if record.name != "VERTEX_POINT" {
        return false;
    }
    let Some(point_id) = instances::nth_entity_ref(&record.parameter, 1) else {
        return false;
    };
    instances::cartesian_point(point_id, entities, index).is_some()
}

fn oriented_edge_failure(
    oriented_id: u64,
    entities: &[EntityInstance],
    index: &std::collections::HashMap<u64, usize>,
) -> Option<String> {
    let &oriented_idx = index.get(&oriented_id)?;
    let oriented = instances::simple_record(&entities[oriented_idx])?;
    if oriented.name != "ORIENTED_EDGE" {
        return Some(format!("type {}", oriented.name));
    }
    let Parameter::List(oriented_params) = &oriented.parameter else {
        return Some("parameters not list".into());
    };
    let Some(edge_id) = oriented_params.get(3).and_then(instances::entity_ref_value) else {
        return Some("missing edge_element ref".into());
    };
    if !oriented_params.get(4).is_some_and(enum_bool_like) {
        return Some("invalid ORIENTED_EDGE.orientation".into());
    }

    let Some(&edge_idx) = index.get(&edge_id) else {
        return Some(format!("missing EDGE_CURVE #{edge_id}"));
    };
    let Some(edge) = instances::simple_record(&entities[edge_idx]) else {
        return Some(format!("edge #{edge_id} is complex"));
    };
    if edge.name != "EDGE_CURVE" {
        return Some(format!("edge #{edge_id} type {}", edge.name));
    }
    let Parameter::List(edge_params) = &edge.parameter else {
        return Some(format!("EDGE_CURVE #{edge_id} parameters not list"));
    };
    let Some(start_id) = edge_params.get(1).and_then(instances::entity_ref_value) else {
        return Some(format!("EDGE_CURVE #{edge_id} missing start"));
    };
    let Some(end_id) = edge_params.get(2).and_then(instances::entity_ref_value) else {
        return Some(format!("EDGE_CURVE #{edge_id} missing end"));
    };
    let Some(curve_id) = edge_params.get(3).and_then(instances::entity_ref_value) else {
        return Some(format!("EDGE_CURVE #{edge_id} missing curve"));
    };
    if !edge_params.get(4).is_some_and(enum_bool_like) {
        return Some(format!("EDGE_CURVE #{edge_id} invalid same_sense"));
    }
    if !vertex_point_ok(start_id, entities, index) {
        return Some(format!(
            "EDGE_CURVE #{edge_id} bad start vertex #{start_id}"
        ));
    }
    if !vertex_point_ok(end_id, entities, index) {
        return Some(format!("EDGE_CURVE #{edge_id} bad end vertex #{end_id}"));
    }
    if !index.contains_key(&curve_id) {
        return Some(format!(
            "EDGE_CURVE #{edge_id} missing curve entity #{curve_id}"
        ));
    }
    None
}

fn diagnose_face_bounds(
    face_id: u64,
    entities: &[EntityInstance],
    index: &std::collections::HashMap<u64, usize>,
) {
    let Some(&face_idx) = index.get(&face_id) else {
        return;
    };
    let Some(record) = instances::simple_record(&entities[face_idx]) else {
        return;
    };
    let Parameter::List(params) = &record.parameter else {
        return;
    };
    let Some(Parameter::List(bounds)) = params.get(1) else {
        println!("    bounds: malformed");
        return;
    };
    for bound_ref in bounds {
        let Some(bound_id) = instances::entity_ref_value(bound_ref) else {
            println!("    FAIL bound: non-entity ref {bound_ref:?}");
            continue;
        };
        let Some(&bound_idx) = index.get(&bound_id) else {
            println!("    FAIL bound #{bound_id}: missing");
            continue;
        };
        let Some(bound) = instances::simple_record(&entities[bound_idx]) else {
            println!("    FAIL bound #{bound_id}: complex");
            continue;
        };
        if bound.name != "FACE_OUTER_BOUND" && bound.name != "FACE_BOUND" {
            println!("    FAIL bound #{bound_id}: type {}", bound.name);
            continue;
        }
        let Parameter::List(bound_params) = &bound.parameter else {
            println!("    FAIL bound #{bound_id}: params not list");
            continue;
        };
        let Some(loop_id) = bound_params.get(1).and_then(instances::entity_ref_value) else {
            println!("    FAIL bound #{bound_id}: missing loop ref");
            continue;
        };
        if !bound_params.get(2).is_some_and(enum_bool_like) {
            println!("    FAIL bound #{bound_id}: invalid orientation");
            continue;
        }
        let Some(&loop_idx) = index.get(&loop_id) else {
            println!("    FAIL bound #{bound_id}: missing loop #{loop_id}");
            continue;
        };
        let Some(loop_record) = instances::simple_record(&entities[loop_idx]) else {
            println!("    FAIL loop #{loop_id}: complex");
            continue;
        };
        if loop_record.name != "EDGE_LOOP" {
            println!("    FAIL loop #{loop_id}: type {}", loop_record.name);
            continue;
        }
        let Parameter::List(loop_params) = &loop_record.parameter else {
            println!("    FAIL loop #{loop_id}: params not list");
            continue;
        };
        let Some(Parameter::List(oriented_edges)) = loop_params.get(1) else {
            println!("    FAIL loop #{loop_id}: malformed oriented-edge list");
            continue;
        };
        for oriented_ref in oriented_edges {
            let Some(oriented_id) = instances::entity_ref_value(oriented_ref) else {
                println!("    FAIL loop #{loop_id}: non-entity oriented edge {oriented_ref:?}");
                continue;
            };
            if let Some(reason) = oriented_edge_failure(oriented_id, entities, index) {
                println!("    FAIL oriented edge #{oriented_id}: {reason}");
                if let Some(&idx) = index.get(&oriented_id) {
                    let oriented = &entities[idx];
                    println!(
                        "      {} refs={:?}",
                        entity_name(oriented),
                        record_refs(oriented)
                    );
                    for id in record_refs(oriented) {
                        if let Some(&child_idx) = index.get(&id) {
                            let child = &entities[child_idx];
                            println!(
                                "        -> #{id} {} refs={:?}",
                                entity_name(child),
                                record_refs(child)
                            );
                        }
                    }
                }
            }
        }
    }
}

fn deterministic_centroid(mut points: Vec<[f64; 3]>) -> Option<[f64; 3]> {
    if points.is_empty() {
        return None;
    }
    points.sort_by(|a, b| {
        a[0].total_cmp(&b[0])
            .then_with(|| a[1].total_cmp(&b[1]))
            .then_with(|| a[2].total_cmp(&b[2]))
    });
    let mut sum = [0.0; 3];
    let mut compensation = [0.0; 3];
    for point in &points {
        for axis in 0..3 {
            let y = point[axis] - compensation[axis];
            let t = sum[axis] + y;
            compensation[axis] = (t - sum[axis]) - y;
            sum[axis] = t;
        }
    }
    let n = points.len() as f64;
    Some([sum[0] / n, sum[1] / n, sum[2] / n])
}

fn main() -> Result<()> {
    let mut args = env::args_os();
    let _exe = args.next();
    let path = args
        .next()
        .context("usage: step-body-diagnose FILE SOLID_ID")?;
    let solid_id: u64 = args
        .next()
        .context("usage: step-body-diagnose FILE SOLID_ID")?
        .to_string_lossy()
        .parse()
        .context("invalid solid id")?;
    if args.next().is_some() {
        bail!("usage: step-body-diagnose FILE SOLID_ID");
    }

    let bytes = fs::read(Path::new(&path)).context("read STEP")?;
    let text = decode_input(&bytes)?;
    let exchange = ruststep::parser::parse(&text).context("parse STEP")?;

    for (section_number, section) in exchange.data.iter().enumerate() {
        let index = instances::build_index(&section.entities);
        let Some(&solid_idx) = index.get(&solid_id) else {
            continue;
        };
        let solid = &section.entities[solid_idx];
        let Some(record) = instances::simple_record(solid) else {
            bail!("solid #{solid_id} is complex");
        };
        if record.name != "MANIFOLD_SOLID_BREP" {
            bail!("#{solid_id} is {}, not MANIFOLD_SOLID_BREP", record.name);
        }

        let closure = instances::closure_from(solid_id, &section.entities, &index);
        let mut points = Vec::new();
        for id in &closure {
            let Some(&idx) = index.get(id) else { continue };
            let Some(record) = instances::simple_record(&section.entities[idx]) else {
                continue;
            };
            if record.name == "VERTEX_POINT"
                && let Some(point_id) = instances::nth_entity_ref(&record.parameter, 1)
                && let Some(point) = instances::cartesian_point(point_id, &section.entities, &index)
            {
                points.push(point);
            }
        }
        let center = deterministic_centroid(points).context("solid has no parseable vertices")?;

        let shell_id = instances::nth_entity_ref(&record.parameter, 1).context("missing shell")?;
        let shell = &section.entities[*index.get(&shell_id).context("missing shell entity")?];
        let shell_record = instances::simple_record(shell).context("complex shell")?;
        let Parameter::List(shell_params) = &shell_record.parameter else {
            bail!("shell parameter is not a list");
        };
        let Parameter::List(face_refs) = shell_params.get(1).context("missing shell face list")?
        else {
            bail!("shell faces are not a list");
        };
        let face_ids = instances::manifold_solid_face_ids(solid_id, &section.entities, &index)
            .context("shell semantic face list is invalid")?;

        println!(
            "section={section_number} solid=#{solid_id} shell=#{shell_id} raw_members={} semantic_faces={} closure={} center={center:?}",
            face_refs.len(),
            face_ids.len(),
            closure.len()
        );

        let mut failed = 0usize;
        for face_id in face_ids {
            let ok = (0..4u8).all(|quarter| {
                instances::face_topology_signature(
                    face_id,
                    &section.entities,
                    &index,
                    center,
                    quarter,
                )
                .is_some()
            });
            if ok {
                continue;
            }
            failed += 1;
            let Some(&face_idx) = index.get(&face_id) else {
                println!("FAIL face #{face_id}: missing entity");
                continue;
            };
            let face = &section.entities[face_idx];
            let refs = record_refs(face);
            println!(
                "FAIL face #{face_id} type={} refs={refs:?}",
                entity_name(face)
            );
            for id in refs {
                if let Some(&idx) = index.get(&id) {
                    println!("  -> #{id} {}", entity_name(&section.entities[idx]));
                }
            }
            diagnose_face_bounds(face_id, &section.entities, &index);
        }
        println!("failed_faces={failed}");

        let whole = instances::solid_shape_key(solid_id, &section.entities, &index);
        println!(
            "whole_shape_key={}",
            if whole.is_some() { "ok" } else { "FAIL" }
        );
        return Ok(());
    }

    bail!("solid #{solid_id} not found")
}
