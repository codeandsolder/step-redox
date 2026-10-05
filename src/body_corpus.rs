use crate::{instances, step_entities, step_graph, step_io::ParsedExchange, units};
use anyhow::{Context, Result, bail};
use ruststep::ast::{DataSection, EntityInstance, Exchange, Parameter, Record};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::Hasher;
use std::path::Path;

#[derive(Debug, Serialize)]
struct ScanReport {
    manifold_solids: usize,
    analyzed_solids: usize,
    skipped_solids: usize,
    fallback_identity_solids: usize,
    solids_without_shape_representation: usize,
    solids: Vec<SolidScan>,
}

#[derive(Debug, Serialize)]
struct SolidScan {
    data_section: usize,
    solid_id: u64,
    geometry_fingerprint: Option<String>,
    identity_kind: &'static str,
    unit_scale_mm: Option<f64>,
    center_source_units: Option<[f64; 3]>,
    canonical_quarter_turn: Option<u8>,
    closure_entities: usize,
    vertices: usize,
    edges: usize,
    oriented_edges: usize,
    faces: usize,
    has_shape_representation: bool,
}

#[derive(Clone, Copy)]
struct Fnv64(u64);

impl Fnv64 {
    const fn seeded(seed: u64) -> Self {
        Self(seed)
    }
}

impl Hasher for Fnv64 {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
}

fn feed_u64(hashers: &mut [Fnv64; 2], value: u64) {
    for hasher in hashers {
        hasher.write(&value.to_le_bytes());
    }
}

fn feed_i64(hashers: &mut [Fnv64; 2], value: i64) {
    for hasher in hashers {
        hasher.write(&value.to_le_bytes());
    }
}

fn feed_bytes(hashers: &mut [Fnv64; 2], value: &[u8]) {
    feed_u64(hashers, value.len() as u64);
    for hasher in hashers {
        hasher.write(value);
    }
}

fn fingerprint(value: &instances::ShapeKey) -> String {
    let mut hashers = [
        Fnv64::seeded(0xcbf29ce484222325),
        Fnv64::seeded(0x84222325cbf29ce4),
    ];
    for count in [
        value.vertices,
        value.edges,
        value.oriented_edges,
        value.faces,
    ] {
        feed_u64(&mut hashers, count as u64);
    }
    feed_u64(&mut hashers, value.points.len() as u64);
    for point in &value.points {
        for coordinate in point {
            feed_i64(&mut hashers, *coordinate);
        }
    }
    for geometry in [&value.edge_geometry, &value.face_geometry] {
        feed_u64(&mut hashers, geometry.len() as u64);
        for (name, scalars) in geometry {
            feed_bytes(&mut hashers, name.as_bytes());
            feed_u64(&mut hashers, scalars.len() as u64);
            for scalar in scalars {
                feed_i64(&mut hashers, *scalar);
            }
        }
    }
    feed_bytes(&mut hashers, value.topology.as_bytes());
    format!("{:016x}{:016x}", hashers[0].finish(), hashers[1].finish())
}

fn parse_exchange(path: &Path) -> Result<ruststep::ast::Exchange> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(ParsedExchange::parse(&bytes)?.exchange)
}

fn representation_ownership(entities: &[EntityInstance]) -> HashMap<u64, (u64, u64)> {
    let mut owners = HashMap::new();
    for entity in entities {
        let Some(record) = step_graph::simple_record(entity) else {
            continue;
        };
        if record.name != "ADVANCED_BREP_SHAPE_REPRESENTATION"
            && record.name != "SHAPE_REPRESENTATION"
        {
            continue;
        }
        let Some((items, context_id)) = step_entities::representation_items_and_context(entity)
        else {
            continue;
        };
        let representation_id = step_graph::entity_id(entity);
        for item in items {
            owners
                .entry(item)
                .or_insert((representation_id, context_id));
        }
    }
    owners
}

fn shape_definition_root_for_representation(
    representation_id: u64,
    entities: &[EntityInstance],
) -> Option<(u64, u64)> {
    entities.iter().find_map(|entity| {
        let record = step_graph::simple_record(entity)?;
        if record.name != "SHAPE_DEFINITION_REPRESENTATION" {
            return None;
        }
        let Parameter::List(params) = &record.parameter else {
            return None;
        };
        let [definition, representation] = params.as_slice() else {
            return None;
        };
        (step_graph::entity_ref_value(representation)? == representation_id).then_some((
            step_graph::entity_id(entity),
            step_graph::entity_ref_value(definition)?,
        ))
    })
}

fn entity_record_named<'a>(entity: &'a EntityInstance, name: &str) -> Option<&'a Record> {
    match entity {
        EntityInstance::Simple { record, .. } => (record.name == name).then_some(record),
        EntityInstance::Complex { subsuper, .. } => {
            subsuper.0.iter().find(|record| record.name == name)
        }
    }
}

fn representation_length_scale_mm(
    context_id: u64,
    entities: &[EntityInstance],
    index: &std::collections::HashMap<u64, usize>,
) -> Option<f64> {
    let entity = entities.get(*index.get(&context_id)?)?;
    let record = entity_record_named(entity, "GLOBAL_UNIT_ASSIGNED_CONTEXT")?;
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let [Parameter::List(units)] = params.as_slice() else {
        return None;
    };
    let mut scales = units
        .iter()
        .filter_map(step_graph::entity_ref_value)
        .filter_map(|unit_id| units::length_unit_scale_mm(unit_id, entities, index));
    let scale = scales.next()?;
    scales.next().is_none().then_some(scale)
}

#[derive(Debug, Clone, Copy)]
struct BasicSolidStats {
    closure_entities: usize,
    vertices: usize,
    edges: usize,
    oriented_edges: usize,
    faces: usize,
}

fn solid_basic_stats(
    root: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> BasicSolidStats {
    let closure = instances::semantic_solid_closure(root, entities, index)
        .unwrap_or_else(|| step_entities::closure_from(root, entities, index));
    let mut stats = BasicSolidStats {
        closure_entities: closure.len(),
        vertices: 0,
        edges: 0,
        oriented_edges: 0,
        faces: 0,
    };
    for id in closure {
        let Some(&entity_index) = index.get(&id) else {
            continue;
        };
        let Some(record) = step_graph::simple_record(&entities[entity_index]) else {
            continue;
        };
        match record.name.as_str() {
            "VERTEX_POINT" => stats.vertices += 1,
            "EDGE_CURVE" => stats.edges += 1,
            "ORIENTED_EDGE" => stats.oriented_edges += 1,
            "ADVANCED_FACE" => stats.faces += 1,
            _ => {}
        }
    }
    stats
}

fn scan(path: &Path) -> Result<ScanReport> {
    let exchange = parse_exchange(path)?;
    let mut solids = Vec::new();
    let mut manifold_solids = 0usize;
    let skipped_solids = 0usize;
    let mut fallback_identity_solids = 0usize;
    let mut solids_without_shape_representation = 0usize;

    for (data_section, section) in exchange.data.iter().enumerate() {
        let index = step_graph::build_index(&section.entities);
        let ownership = representation_ownership(&section.entities);
        let mut unit_scale_cache = HashMap::<u64, Option<f64>>::new();
        for entity in &section.entities {
            let Some(record) = step_graph::simple_record(entity) else {
                continue;
            };
            if record.name != "MANIFOLD_SOLID_BREP" {
                continue;
            }
            manifold_solids += 1;
            let solid_id = step_graph::entity_id(entity);
            let owning = ownership.get(&solid_id).copied();
            let has_shape_representation = owning.is_some();
            solids_without_shape_representation += usize::from(!has_shape_representation);
            let unit_scale_mm = owning.and_then(|(_, context_id)| {
                *unit_scale_cache.entry(context_id).or_insert_with(|| {
                    representation_length_scale_mm(context_id, &section.entities, &index)
                })
            });

            let canonical = instances::solid_shape_key(solid_id, &section.entities, &index);
            let (
                geometry_fingerprint,
                identity_kind,
                center_source_units,
                canonical_quarter_turn,
                stats,
            ) = if let Some((key, center, quarter, closure_entities)) = canonical {
                (
                    Some(fingerprint(&key)),
                    "canonical",
                    Some(center),
                    Some(quarter),
                    BasicSolidStats {
                        closure_entities,
                        vertices: key.vertices,
                        edges: key.edges,
                        oriented_edges: key.oriented_edges,
                        faces: key.faces,
                    },
                )
            } else {
                fallback_identity_solids += 1;
                (
                    None,
                    "file_local_fallback",
                    None,
                    None,
                    solid_basic_stats(solid_id, &section.entities, &index),
                )
            };

            solids.push(SolidScan {
                data_section,
                solid_id,
                geometry_fingerprint,
                identity_kind,
                unit_scale_mm,
                center_source_units,
                canonical_quarter_turn,
                closure_entities: stats.closure_entities,
                vertices: stats.vertices,
                edges: stats.edges,
                oriented_edges: stats.oriented_edges,
                faces: stats.faces,
                has_shape_representation,
            });
        }
    }
    solids.sort_by_key(|row| (row.data_section, row.solid_id));
    Ok(ScanReport {
        manifold_solids,
        analyzed_solids: solids.len(),
        skipped_solids,
        fallback_identity_solids,
        solids_without_shape_representation,
        solids,
    })
}

fn parse_selection(raw: &str) -> Result<(usize, u64, String)> {
    let mut parts = raw.splitn(3, ':');
    let section = parts
        .next()
        .context("missing DATA section")?
        .parse::<usize>()
        .context("invalid DATA section")?;
    let solid = parts
        .next()
        .context("missing solid id")?
        .parse::<u64>()
        .context("invalid solid id")?;
    let fingerprint = parts.next().context("missing fingerprint")?.to_string();
    if fingerprint.is_empty() || !fingerprint.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("invalid fingerprint in {raw:?}");
    }
    Ok((section, solid, fingerprint))
}

fn set_shell_faces(entity: &mut EntityInstance, face_ids: &[u64]) -> Result<()> {
    let record = step_graph::simple_record_mut(entity).context("shell is complex")?;
    if record.name != "CLOSED_SHELL" {
        bail!("expected CLOSED_SHELL, got {}", record.name);
    }
    let Parameter::List(params) = &mut record.parameter else {
        bail!("shell parameters are not a list");
    };
    let Some(faces) = params.get_mut(1) else {
        bail!("shell has no face list");
    };
    *faces = Parameter::List(
        face_ids
            .iter()
            .copied()
            .map(step_entities::entity_ref)
            .collect(),
    );
    Ok(())
}

fn set_representation_single_item(entity: &mut EntityInstance, solid_id: u64) -> Result<()> {
    let record = step_graph::simple_record_mut(entity).context("representation is complex")?;
    let Parameter::List(params) = &mut record.parameter else {
        bail!("representation parameters are not a list");
    };
    let Some(items) = params.get_mut(1) else {
        bail!("representation has no item list");
    };
    *items = Parameter::List(vec![step_entities::entity_ref(solid_id)]);
    Ok(())
}

fn extract_one(
    exchange: &Exchange,
    data_section: usize,
    solid_id: u64,
    representation_id: u64,
    context_id: u64,
    index: &HashMap<u64, usize>,
    out_path: &Path,
) -> Result<usize> {
    let section = exchange
        .data
        .get(data_section)
        .with_context(|| format!("missing DATA section {data_section}"))?;
    let &solid_index = index
        .get(&solid_id)
        .with_context(|| format!("missing solid #{solid_id}"))?;
    let record = step_graph::simple_record(&section.entities[solid_index])
        .with_context(|| format!("solid #{solid_id} is complex"))?;
    if record.name != "MANIFOLD_SOLID_BREP" {
        bail!("#{solid_id} is {}, not MANIFOLD_SOLID_BREP", record.name);
    }

    let shell_id = step_entities::nth_entity_ref(&record.parameter, 1)
        .context("solid has no shell reference")?;
    let face_ids = instances::manifold_solid_face_ids(solid_id, &section.entities, index)
        .context("solid has no semantic face list")?;
    let mut keep = instances::semantic_solid_closure(solid_id, &section.entities, index)
        .context("cannot build semantic solid closure")?;
    keep.extend(step_entities::closure_from(
        context_id,
        &section.entities,
        index,
    ));
    keep.insert(representation_id);

    // A bare ADVANCED_BREP_SHAPE_REPRESENTATION is valid exchange data but is
    // not a standalone product root for importers such as OpenCascade. Preserve
    // the smallest original product-definition chain that points at this
    // representation so an extracted body is independently loadable.
    if let Some((shape_definition_id, product_shape_id)) =
        shape_definition_root_for_representation(representation_id, &section.entities)
    {
        keep.insert(shape_definition_id);
        keep.extend(step_entities::closure_from(
            product_shape_id,
            &section.entities,
            index,
        ));
    }

    let mut entities = Vec::with_capacity(keep.len());
    for entity in &section.entities {
        let id = step_graph::entity_id(entity);
        if !keep.contains(&id) {
            continue;
        }
        let mut cloned = entity.clone();
        if id == shell_id {
            set_shell_faces(&mut cloned, &face_ids)?;
        }
        if id == representation_id {
            set_representation_single_item(&mut cloned, solid_id)?;
        }
        entities.push(cloned);
    }

    let output = Exchange {
        header: exchange.header.clone(),
        anchor: Vec::new(),
        reference: Vec::new(),
        data: vec![DataSection {
            meta: section.meta.clone(),
            entities,
        }],
        signature: Vec::new(),
    };

    let serialized = crate::write_exchange(&output)?;
    // The isolated graph is already semantics-complete. Run only the normal
    // non-experimental cleanup pass before persisting it: value interning,
    // presentation consolidation and dense entity IDs substantially reduce
    // corpus size without invoking any constructive recovery heuristics.
    let cleaned = crate::clean_bytes(serialized.as_bytes(), &crate::Options::default())?;
    fs::write(out_path, cleaned.bytes).with_context(|| format!("write {}", out_path.display()))?;
    Ok(keep.len())
}

fn extract(path: &Path, out_dir: &Path, selections: &[String]) -> Result<()> {
    let exchange = parse_exchange(path)?;
    fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;

    let mut parsed = Vec::with_capacity(selections.len());
    let mut seen = HashSet::new();
    for raw in selections {
        let (section, solid_id, fingerprint) = parse_selection(raw)?;
        if !seen.insert(fingerprint.clone()) {
            bail!("duplicate output fingerprint {fingerprint}");
        }
        parsed.push((section, solid_id, fingerprint));
    }

    let mut indexes = Vec::with_capacity(exchange.data.len());
    let mut ownership = Vec::with_capacity(exchange.data.len());
    for section in &exchange.data {
        indexes.push(step_graph::build_index(&section.entities));
        ownership.push(representation_ownership(&section.entities));
    }

    for (section_id, solid_id, fingerprint) in parsed {
        let index = indexes
            .get(section_id)
            .with_context(|| format!("missing DATA section {section_id}"))?;
        let (representation_id, context_id) = ownership
            .get(section_id)
            .and_then(|owners| owners.get(&solid_id))
            .copied()
            .with_context(|| {
                format!("solid #{solid_id} is not a direct item of a shape representation")
            })?;

        let shard = &fingerprint[..2.min(fingerprint.len())];
        let dir = out_dir.join(shard);
        fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        let out_path = dir.join(format!("{fingerprint}.step"));
        extract_one(
            &exchange,
            section_id,
            solid_id,
            representation_id,
            context_id,
            index,
            &out_path,
        )?;
    }
    Ok(())
}

/// Scan one STEP file using the production body-identity implementation and
/// return the body-corpus report in the stable JSON shape consumed by corpus tools.
///
/// # Errors
/// Returns an error if the file cannot be read/decoded/parsed or serialization fails.
pub fn scan_json(path: &Path) -> Result<String> {
    Ok(serde_json::to_string(&scan(path)?)?)
}

/// Extract selected body fingerprints into standalone STEP files.
///
/// # Errors
/// Returns an error if selection metadata is invalid or a body cannot be isolated/written.
pub fn extract_selected(path: &Path, out_dir: &Path, selections: &[String]) -> Result<()> {
    extract(path, out_dir, selections)
}
