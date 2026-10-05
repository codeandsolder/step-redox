use crate::step_graph::visit_entity_refs;
use crate::step_io::{format_real, write_step_string};
use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

#[derive(Default)]
pub(super) struct ConsolidateStats {
    pub(super) total: usize,
    pub(super) by_type: BTreeMap<String, usize>,
}

pub(super) fn consolidate_presentation(entities: &mut Vec<EntityInstance>) -> ConsolidateStats {
    use std::collections::{HashMap, HashSet};

    let mut refcounts: HashMap<u64, usize> = HashMap::new();
    for entity in entities.iter() {
        visit_entity_refs(entity, &mut |id| *refcounts.entry(id).or_insert(0) += 1);
    }

    #[derive(Clone)]
    struct Merge {
        into: usize,
        from: usize,
        items: Vec<Parameter>,
        ty: &'static str,
    }

    let mut groups: HashMap<String, usize> = HashMap::new();
    let mut merges = Vec::new();

    for (idx, entity) in entities.iter().enumerate() {
        let EntityInstance::Simple { id, record } = entity else {
            continue;
        };
        if refcounts.get(id).copied().unwrap_or(0) != 0 {
            continue;
        }
        let Parameter::List(params) = &record.parameter else {
            continue;
        };

        let (ty, key, items) = match record.name.as_str() {
            "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION" if params.len() == 3 => {
                let Parameter::List(items) = &params[1] else {
                    continue;
                };
                let key = format!(
                    "MDGPR|{}|{}",
                    standalone_param_key(&params[0]),
                    standalone_param_key(&params[2])
                );
                (
                    "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION",
                    key,
                    items.clone(),
                )
            }
            "PRESENTATION_LAYER_ASSIGNMENT" if params.len() == 3 => {
                let Parameter::List(items) = &params[2] else {
                    continue;
                };
                let key = format!(
                    "PLA|{}|{}",
                    standalone_param_key(&params[0]),
                    standalone_param_key(&params[1])
                );
                ("PRESENTATION_LAYER_ASSIGNMENT", key, items.clone())
            }
            _ => continue,
        };

        if let Some(&into) = groups.get(&key) {
            merges.push(Merge {
                into,
                from: idx,
                items,
                ty,
            });
        } else {
            groups.insert(key, idx);
        }
    }

    let mut additions: HashMap<usize, Vec<Parameter>> = HashMap::new();
    let mut remove = HashSet::new();
    let mut stats = ConsolidateStats::default();
    for merge in merges {
        additions.entry(merge.into).or_default().extend(merge.items);
        remove.insert(merge.from);
        stats.total += 1;
        *stats.by_type.entry(merge.ty.to_string()).or_insert(0) += 1;
    }

    for (idx, items) in additions {
        let EntityInstance::Simple { record, .. } = &mut entities[idx] else {
            unreachable!();
        };
        let Parameter::List(params) = &mut record.parameter else {
            unreachable!();
        };
        let target_idx = if record.name == "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION"
        {
            1
        } else {
            2
        };
        let Parameter::List(existing) = &mut params[target_idx] else {
            unreachable!();
        };
        existing.extend(items);
    }

    let mut i = 0usize;
    entities.retain(|_| {
        let keep = !remove.contains(&i);
        i += 1;
        keep
    });
    stats
}

fn standalone_param_key(param: &Parameter) -> String {
    let mut out = String::new();
    write_param_key(param, &HashMap::new(), &mut out);
    out
}

#[derive(Default)]
pub(super) struct InternStats {
    pub(super) total: usize,
    pub(super) by_type: BTreeMap<String, usize>,
}

pub(super) fn intern_section(entities: &mut Vec<EntityInstance>) -> InternStats {
    // Store redirects only. An identity map for a million-entity STEP file is
    // a surprisingly expensive way of spelling "most things survive".
    let mut alias: HashMap<u64, u64> = HashMap::new();

    // Value DAGs in the EasyEDA/SolidWorks corpus settle in a handful of
    // rounds (units -> uncertainty/context, colour -> style chains, geometry
    // primitives -> placements/surfaces). Updates are applied at the end of a
    // round so keys within that round see a stable alias map.
    for _ in 0..16 {
        let mut seen: HashMap<String, u64> = HashMap::new();
        let mut pending: Vec<(u64, u64)> = Vec::new();

        for entity in entities.iter() {
            if !is_internable(entity) {
                continue;
            }
            let id = entity_id(entity);
            if alias.contains_key(&id) {
                continue;
            }

            let key = entity_key(entity, &alias);
            if let Some(&canonical) = seen.get(&key) {
                let canonical = resolve_alias(&alias, canonical);
                if canonical != id {
                    pending.push((id, canonical));
                }
            } else {
                seen.insert(key, id);
            }
        }

        if pending.is_empty() {
            break;
        }
        for (id, canonical) in pending {
            alias.insert(id, canonical);
        }
        compress_aliases(&mut alias);
    }
    compress_aliases(&mut alias);

    let mut stats = InternStats::default();
    let original = std::mem::take(entities);
    entities.reserve(original.len().saturating_sub(alias.len()));
    for mut entity in original {
        let id = entity_id(&entity);
        let root = resolve_alias(&alias, id);
        if root != id {
            stats.total += 1;
            *stats.by_type.entry(entity_type_label(&entity)).or_insert(0) += 1;
            continue;
        }
        rewrite_entity_refs(&mut entity, &alias);
        entities.push(entity);
    }
    stats
}

fn compress_aliases(alias: &mut HashMap<u64, u64>) {
    let keys: Vec<u64> = alias.keys().copied().collect();
    for id in keys {
        let root = resolve_alias(alias, id);
        if root != id {
            alias.insert(id, root);
        }
    }
}

fn resolve_alias(alias: &HashMap<u64, u64>, mut id: u64) -> u64 {
    for _ in 0..64 {
        let Some(&next) = alias.get(&id) else {
            return id;
        };
        if next == id {
            return id;
        }
        id = next;
    }
    id
}

pub(super) fn dense_renumber(entities: &mut [EntityInstance]) {
    let id_map: HashMap<u64, u64> = entities
        .iter()
        .enumerate()
        .map(|(idx, e)| (entity_id(e), idx as u64 + 1))
        .collect();

    for entity in entities.iter_mut() {
        let old = entity_id(entity);
        let new = id_map[&old];
        set_entity_id(entity, new);
        rewrite_entity_refs(entity, &id_map);
    }
}

const fn entity_id(entity: &EntityInstance) -> u64 {
    match entity {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id,
    }
}

const fn set_entity_id(entity: &mut EntityInstance, new_id: u64) {
    match entity {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id = new_id,
    }
}

fn entity_type_label(entity: &EntityInstance) -> String {
    match entity {
        EntityInstance::Simple { record, .. } => record.name.clone(),
        EntityInstance::Complex { subsuper, .. } => subsuper
            .0
            .iter()
            .map(|r| r.name.as_str())
            .collect::<Vec<_>>()
            .join("+"),
    }
}

fn rewrite_entity_refs(entity: &mut EntityInstance, map: &HashMap<u64, u64>) {
    match entity {
        EntityInstance::Simple { record, .. } => rewrite_param_refs(&mut record.parameter, map),
        EntityInstance::Complex { subsuper, .. } => {
            for record in &mut subsuper.0 {
                rewrite_param_refs(&mut record.parameter, map);
            }
        }
    }
}

fn rewrite_param_refs(param: &mut Parameter, map: &HashMap<u64, u64>) {
    match param {
        Parameter::Ref(Name::Entity(id)) => {
            if let Some(&new) = map.get(id) {
                *id = new;
            }
        }
        Parameter::List(items) => {
            for item in items {
                rewrite_param_refs(item, map);
            }
        }
        Parameter::Typed { parameter, .. } => rewrite_param_refs(parameter, map),
        _ => {}
    }
}

fn is_internable(entity: &EntityInstance) -> bool {
    match entity {
        EntityInstance::Simple { record, .. } => internable_record(&record.name),
        EntityInstance::Complex { subsuper, .. } => {
            !subsuper.0.is_empty() && subsuper.0.iter().all(|r| internable_record(&r.name))
        }
    }
}

// Deliberately excludes topological identity objects: VERTEX_POINT, EDGE_CURVE,
// ORIENTED_EDGE, EDGE_LOOP, FACE_*, ADVANCED_FACE, shells, and solids.
//
// Sharing these value/geometry-support objects does not merge topology. It only
// makes multiple topological objects point at the same equal geometry/style/unit
// value, which is the redundancy SolidWorks explodes in these EasyEDA files.
fn internable_record(name: &str) -> bool {
    matches!(
        name,
        // Geometry values / support geometry
        "CARTESIAN_POINT"
            | "DIRECTION"
            | "VECTOR"
            | "AXIS1_PLACEMENT"
            | "AXIS2_PLACEMENT_2D"
            | "AXIS2_PLACEMENT_3D"
            | "LINE"
            | "CIRCLE"
            | "ELLIPSE"
            | "PLANE"
            | "CYLINDRICAL_SURFACE"
            | "CONICAL_SURFACE"
            | "SPHERICAL_SURFACE"
            | "TOROIDAL_SURFACE"
            | "SURFACE_OF_LINEAR_EXTRUSION"
            | "SURFACE_OF_REVOLUTION"
            | "B_SPLINE_CURVE"
            | "B_SPLINE_CURVE_WITH_KNOTS"
            | "RATIONAL_B_SPLINE_CURVE"
            | "B_SPLINE_SURFACE"
            | "B_SPLINE_SURFACE_WITH_KNOTS"
            | "RATIONAL_B_SPLINE_SURFACE"
            // Presentation/style values
            | "COLOUR_RGB"
            | "DRAUGHTING_PRE_DEFINED_COLOUR"
            | "DRAUGHTING_PRE_DEFINED_CURVE_FONT"
            | "CURVE_STYLE"
            | "POINT_STYLE"
            | "FILL_AREA_STYLE_COLOUR"
            | "FILL_AREA_STYLE"
            | "SURFACE_STYLE_FILL_AREA"
            | "SURFACE_SIDE_STYLE"
            | "SURFACE_STYLE_USAGE"
            | "PRESENTATION_STYLE_ASSIGNMENT"
            // Units / contexts / uncertainty values
            | "NAMED_UNIT"
            | "SI_UNIT"
            | "LENGTH_UNIT"
            | "PLANE_ANGLE_UNIT"
            | "SOLID_ANGLE_UNIT"
            | "CONVERSION_BASED_UNIT"
            | "MEASURE_WITH_UNIT"
            | "UNCERTAINTY_MEASURE_WITH_UNIT"
            | "REPRESENTATION_CONTEXT"
            | "GEOMETRIC_REPRESENTATION_CONTEXT"
            | "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT"
            | "GLOBAL_UNIT_ASSIGNED_CONTEXT"
    )
}

fn entity_key(entity: &EntityInstance, alias: &HashMap<u64, u64>) -> String {
    let mut out = String::new();
    match entity {
        EntityInstance::Simple { record, .. } => write_record_key(record, alias, &mut out),
        EntityInstance::Complex { subsuper, .. } => {
            out.push('(');
            for record in &subsuper.0 {
                write_record_key(record, alias, &mut out);
            }
            out.push(')');
        }
    }
    out
}

fn write_record_key(record: &Record, alias: &HashMap<u64, u64>, out: &mut String) {
    out.push_str(&record.name);
    write_param_key(&record.parameter, alias, out);
}

fn write_param_key(param: &Parameter, alias: &HashMap<u64, u64>, out: &mut String) {
    match param {
        Parameter::Typed { keyword, parameter } => {
            out.push_str(keyword);
            out.push('(');
            write_param_key(parameter, alias, out);
            out.push(')');
        }
        Parameter::Integer(v) => {
            let _ = write!(out, "{v}");
        }
        Parameter::Real(v) => out.push_str(&format_real(*v)),
        Parameter::String(s) => write_step_string(s, out),
        Parameter::Enumeration(s) => {
            out.push('.');
            out.push_str(s);
            out.push('.');
        }
        Parameter::List(items) => {
            out.push('(');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_param_key(item, alias, out);
            }
            out.push(')');
        }
        Parameter::Ref(Name::Entity(id)) => {
            let _ = write!(out, "#{}", resolve_alias(alias, *id));
        }
        Parameter::Ref(Name::Value(id)) => {
            let _ = write!(out, "@{id}");
        }
        Parameter::Ref(Name::ConstantEntity(s)) => {
            out.push('#');
            out.push_str(s);
        }
        Parameter::Ref(Name::ConstantValue(s)) => {
            out.push('@');
            out.push_str(s);
        }
        Parameter::NotProvided => out.push('$'),
        Parameter::Omitted => out.push('*'),
    }
}

const PLACEHOLDER_NAME_TYPES: &[&str] = &[
    "ADVANCED_FACE",
    "AXIS2_PLACEMENT_3D",
    "B_SPLINE_CURVE_WITH_KNOTS",
    "B_SPLINE_SURFACE_WITH_KNOTS",
    "CARTESIAN_POINT",
    "CIRCLE",
    "CLOSED_SHELL",
    "CONICAL_SURFACE",
    "CYLINDRICAL_SURFACE",
    "DIRECTION",
    "EDGE_CURVE",
    "EDGE_LOOP",
    "FACE_BOUND",
    "FACE_OUTER_BOUND",
    "LINE",
    "MANIFOLD_SOLID_BREP",
    "ORIENTED_EDGE",
    "PLANE",
    "SPHERICAL_SURFACE",
    "STYLED_ITEM",
    "TOROIDAL_SURFACE",
    "VECTOR",
    "VERTEX_POINT",
];

pub(super) fn minify_placeholder_names(entities: &mut [EntityInstance]) -> usize {
    let mut changed = 0usize;
    for entity in entities {
        match entity {
            EntityInstance::Simple { record, .. } => {
                changed += minify_placeholder_record_name(record);
            }
            EntityInstance::Complex { subsuper, .. } => {
                for record in &mut subsuper.0 {
                    changed += minify_placeholder_record_name(record);
                }
            }
        }
    }
    changed
}

fn minify_placeholder_record_name(record: &mut Record) -> usize {
    if !PLACEHOLDER_NAME_TYPES.contains(&record.name.as_str()) {
        return 0;
    }
    let Parameter::List(params) = &mut record.parameter else {
        return 0;
    };
    let Some(Parameter::String(name)) = params.first_mut() else {
        return 0;
    };
    if name != "NONE" {
        return 0;
    }
    name.clear();
    1
}
