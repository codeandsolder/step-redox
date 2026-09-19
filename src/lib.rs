use anyhow::{Context, Result, bail};
use ruststep::ast::{EntityInstance, Exchange, Name, Parameter, Record};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

mod brep;
mod instances;
mod line_recovery;
mod planar_features;
mod spherical_caps;

#[derive(Debug, Clone)]
pub struct Options {
    pub intern_values: bool,
    pub consolidate_presentation: bool,
    pub experimental_recover_straight_bspline_lines: bool,
    pub experimental_instance_z90: bool,
    pub experimental_instance_planar_positive_features: bool,
    pub experimental_instance_spherical_caps: bool,
    pub minify_placeholder_names: bool,
    pub dense_ids: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            intern_values: true,
            consolidate_presentation: true,
            experimental_recover_straight_bspline_lines: false,
            experimental_instance_z90: false,
            experimental_instance_planar_positive_features: false,
            experimental_instance_spherical_caps: false,
            minify_placeholder_names: false,
            dense_ids: true,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Stats {
    pub input_encoding: String,
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub input_entities: usize,
    pub output_entities: usize,
    pub interned_entities: usize,
    pub consolidated_entities: usize,
    pub straight_bspline_lines_recovered: usize,
    pub straight_bspline_direction_groups: usize,
    pub straight_bspline_points_removed: usize,
    pub instance_groups: usize,
    pub instanced_solids: usize,
    pub instance_entities_removed: usize,
    pub instance_styles_replaced: usize,
    pub planar_feature_arrays: usize,
    pub planar_feature_families: usize,
    pub planar_feature_instances: usize,
    pub planar_feature_entities_removed: usize,
    pub planar_feature_styles_replaced: usize,
    pub spherical_cap_arrays: usize,
    pub spherical_cap_instances: usize,
    pub spherical_cap_entities_removed: usize,
    pub spherical_cap_styles_replaced: usize,
    pub placeholder_names_minified: usize,
    pub byte_ratio: f64,
    pub interned_by_type: BTreeMap<String, usize>,
    pub consolidated_by_type: BTreeMap<String, usize>,
}

pub struct CleanOutput {
    pub bytes: Vec<u8>,
    pub stats: Stats,
}

pub fn clean_bytes(input: &[u8], options: &Options) -> Result<CleanOutput> {
    let (input_text, input_encoding) = decode_input(input)?;
    let (parser_text, had_empty_aggregate_shim) = prepare_parser_input(&input_text)?;
    let mut exchange =
        ruststep::parser::parse(&parser_text).context("parse STEP exchange structure")?;
    if had_empty_aggregate_shim {
        restore_empty_aggregates(&mut exchange)?;
    }

    if !exchange.anchor.is_empty()
        || !exchange.reference.is_empty()
        || !exchange.signature.is_empty()
    {
        bail!("ANCHOR/REFERENCE/SIGNATURE sections are not yet supported by step-redox writer");
    }

    let input_entities: usize = exchange.data.iter().map(|d| d.entities.len()).sum();

    let mut placeholder_names_minified = 0usize;
    if options.minify_placeholder_names {
        for section in &mut exchange.data {
            placeholder_names_minified += minify_placeholder_names(&mut section.entities);
        }
    }

    let mut interned_by_type = BTreeMap::new();
    let mut interned_entities = 0usize;

    if options.intern_values {
        for section in &mut exchange.data {
            let pass = intern_section(&mut section.entities);
            interned_entities += pass.total;
            for (k, v) in pass.by_type {
                *interned_by_type.entry(k).or_insert(0) += v;
            }
        }
    }

    let mut consolidated_by_type = BTreeMap::new();
    let mut consolidated_entities = 0usize;
    if options.consolidate_presentation {
        for section in &mut exchange.data {
            let pass = consolidate_presentation(&mut section.entities);
            consolidated_entities += pass.total;
            for (k, v) in pass.by_type {
                *consolidated_by_type.entry(k).or_insert(0) += v;
            }
        }
    }

    let mut straight_bspline_lines_recovered = 0usize;
    let mut straight_bspline_direction_groups = 0usize;
    let mut straight_bspline_points_removed = 0usize;
    if options.experimental_recover_straight_bspline_lines {
        for section in &mut exchange.data {
            let pass = line_recovery::recover_straight_bspline_lines(&mut section.entities);
            straight_bspline_lines_recovered += pass.curves_recovered;
            straight_bspline_direction_groups += pass.direction_groups;
            straight_bspline_points_removed += pass.orphan_points_removed;
        }
    }

    let mut instance_groups = 0usize;
    let mut instanced_solids = 0usize;
    let mut instance_entities_removed = 0usize;
    let mut instance_styles_replaced = 0usize;
    if options.experimental_instance_z90 {
        for section in &mut exchange.data {
            let pass = instances::instance_z90_solids(&mut section.entities);
            instance_groups += pass.groups;
            instanced_solids += pass.solids_replaced;
            instance_entities_removed += pass.entities_removed;
            instance_styles_replaced += pass.styles_replaced;
        }
    }

    let mut planar_feature_arrays = 0usize;
    let mut planar_feature_families = 0usize;
    let mut planar_feature_instances = 0usize;
    let mut planar_feature_entities_removed = 0usize;
    let mut planar_feature_styles_replaced = 0usize;
    if options.experimental_instance_planar_positive_features {
        for section in &mut exchange.data {
            let pass = planar_features::instance_planar_positive_features(&mut section.entities);
            planar_feature_arrays += pass.arrays;
            planar_feature_families += pass.families;
            planar_feature_instances += pass.instances;
            planar_feature_entities_removed += pass.entities_removed;
            planar_feature_styles_replaced += pass.styles_replaced;
        }
    }

    let mut spherical_cap_arrays = 0usize;
    let mut spherical_cap_instances = 0usize;
    let mut spherical_cap_entities_removed = 0usize;
    let mut spherical_cap_styles_replaced = 0usize;
    if options.experimental_instance_spherical_caps {
        for section in &mut exchange.data {
            let pass = spherical_caps::instance_planar_spherical_caps(&mut section.entities);
            spherical_cap_arrays += pass.arrays;
            spherical_cap_instances += pass.instances;
            spherical_cap_entities_removed += pass.entities_removed;
            spherical_cap_styles_replaced += pass.styles_replaced;
        }
    }

    // Experimental passes can create new placeholder-labelled entities.
    // Minify those before the post-rewrite intern pass so name normalization
    // cannot create fresh duplicates that only disappear on a second run.
    if options.minify_placeholder_names {
        for section in &mut exchange.data {
            placeholder_names_minified += minify_placeholder_names(&mut section.entities);
        }
    }

    // Experimental passes create placements/directions and other support
    // values. Normalize them in the same invocation so aggressive output is a
    // fixed point rather than requiring a second safe cleanup pass.
    if (straight_bspline_lines_recovered > 0
        || instance_groups > 0
        || planar_feature_arrays > 0
        || spherical_cap_arrays > 0)
        && options.intern_values
    {
        for section in &mut exchange.data {
            let pass = intern_section(&mut section.entities);
            interned_entities += pass.total;
            for (k, v) in pass.by_type {
                *interned_by_type.entry(k).or_insert(0) += v;
            }
        }
    }

    if options.dense_ids {
        for section in &mut exchange.data {
            dense_renumber(&mut section.entities);
        }
    }

    let output = write_exchange(&exchange)?;
    let output_entities: usize = exchange.data.iter().map(|d| d.entities.len()).sum();
    let output_bytes = output.len();

    Ok(CleanOutput {
        bytes: output.into_bytes(),
        stats: Stats {
            input_encoding: input_encoding.to_string(),
            input_bytes: input.len(),
            output_bytes,
            input_entities,
            output_entities,
            interned_entities,
            consolidated_entities,
            straight_bspline_lines_recovered,
            straight_bspline_direction_groups,
            straight_bspline_points_removed,
            instance_groups,
            instanced_solids,
            instance_entities_removed,
            instance_styles_replaced,
            planar_feature_arrays,
            planar_feature_families,
            planar_feature_instances,
            planar_feature_entities_removed,
            planar_feature_styles_replaced,
            spherical_cap_arrays,
            spherical_cap_instances,
            spherical_cap_entities_removed,
            spherical_cap_styles_replaced,
            placeholder_names_minified,
            byte_ratio: output_bytes as f64 / input.len().max(1) as f64,
            interned_by_type,
            consolidated_by_type,
        },
    })
}

#[derive(Default)]
struct ConsolidateStats {
    total: usize,
    by_type: BTreeMap<String, usize>,
}

fn consolidate_presentation(entities: &mut Vec<EntityInstance>) -> ConsolidateStats {
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

fn standalone_param_key(param: &Parameter) -> String {
    let mut out = String::new();
    write_param_key(param, &HashMap::new(), &mut out);
    out
}

#[derive(Default)]
struct InternStats {
    total: usize,
    by_type: BTreeMap<String, usize>,
}

fn intern_section(entities: &mut Vec<EntityInstance>) -> InternStats {
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

fn dense_renumber(entities: &mut [EntityInstance]) {
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

fn entity_id(entity: &EntityInstance) -> u64 {
    match entity {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id,
    }
}

fn set_entity_id(entity: &mut EntityInstance, new_id: u64) {
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

fn minify_placeholder_names(entities: &mut [EntityInstance]) -> usize {
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

pub fn write_exchange(exchange: &Exchange) -> Result<String> {
    if !exchange.anchor.is_empty()
        || !exchange.reference.is_empty()
        || !exchange.signature.is_empty()
    {
        bail!("optional STEP sections are not supported by writer");
    }

    let mut out = String::with_capacity(
        exchange
            .data
            .iter()
            .map(|d| d.entities.len())
            .sum::<usize>()
            * 48,
    );
    out.push_str("ISO-10303-21;\nHEADER;\n");
    for record in &exchange.header {
        write_record(record, &mut out);
        out.push_str(";\n");
    }
    out.push_str("ENDSEC;\n");

    for section in &exchange.data {
        if section.meta.is_empty() {
            out.push_str("DATA;\n");
        } else {
            out.push_str("DATA(");
            for (i, param) in section.meta.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_param(param, &mut out);
            }
            out.push_str(");\n");
        }
        for entity in &section.entities {
            write_entity(entity, &mut out);
            out.push('\n');
        }
        out.push_str("ENDSEC;\n");
    }
    out.push_str("END-ISO-10303-21;\n");
    Ok(out)
}

fn write_entity(entity: &EntityInstance, out: &mut String) {
    match entity {
        EntityInstance::Simple { id, record } => {
            let _ = write!(out, "#{id}=");
            write_record(record, out);
            out.push(';');
        }
        EntityInstance::Complex { id, subsuper } => {
            let _ = write!(out, "#{id}=(");
            for record in &subsuper.0 {
                write_record(record, out);
            }
            out.push_str(");");
        }
    }
}

fn write_record(record: &Record, out: &mut String) {
    out.push_str(&record.name);
    write_param(&record.parameter, out);
}

fn write_param(param: &Parameter, out: &mut String) {
    match param {
        Parameter::Typed { keyword, parameter } => {
            out.push_str(keyword);
            out.push('(');
            write_param(parameter, out);
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
                write_param(item, out);
            }
            out.push(')');
        }
        Parameter::Ref(Name::Entity(id)) => {
            let _ = write!(out, "#{id}");
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

const EMPTY_AGGREGATE_MARKER: &str = "STEPREDOXEMPTYAGGREGATE";

fn prepare_parser_input(input: &str) -> Result<(std::borrow::Cow<'_, str>, bool)> {
    // ruststep 0.4 documents aggregate contents as optional, but its
    // comma_separated() parser currently requires at least one parameter. A
    // few valid AP214 exporters emit empty aggregates such as
    // SHAPE_REPRESENTATION('',(),#ctx). Encode those with an impossible
    // enumeration sentinel for parsing, then restore them in the AST.
    let Some(data_start) = input.find("DATA;") else {
        return Ok((std::borrow::Cow::Borrowed(input), false));
    };
    let scan_start = data_start + "DATA;".len();
    let suffix = &input[scan_start..];
    if !suffix.as_bytes().windows(2).any(|w| w == b"()")
        && !suffix.contains("( ")
        && !suffix.contains("(\t")
        && !suffix.contains("(\r")
        && !suffix.contains("(\n")
    {
        return Ok((std::borrow::Cow::Borrowed(input), false));
    }
    let marker = format!(".{EMPTY_AGGREGATE_MARKER}.");
    if input.contains(&marker) {
        bail!("STEP input collides with step-redox empty-aggregate parser marker");
    }

    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len() + 64);
    out.push_str(&input[..scan_start]);

    let mut i = scan_start;
    let mut last = scan_start;
    let mut in_string = false;
    let mut in_comment = false;
    let mut replaced = false;

    while i < bytes.len() {
        if in_comment {
            if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                in_comment = false;
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }

        if in_string {
            if bytes[i] == b'\'' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' {
                    i += 2;
                } else {
                    in_string = false;
                    i += 1;
                }
            } else {
                i += 1;
            }
            continue;
        }

        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            in_comment = true;
            i += 2;
            continue;
        }
        if bytes[i] == b'\'' {
            in_string = true;
            i += 1;
            continue;
        }
        if bytes[i] == b'(' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b')' {
                out.push_str(&input[last..i]);
                out.push('(');
                out.push_str(&marker);
                out.push(')');
                i = j + 1;
                last = i;
                replaced = true;
                continue;
            }
        }
        i += 1;
    }

    if !replaced {
        return Ok((std::borrow::Cow::Borrowed(input), false));
    }
    out.push_str(&input[last..]);
    Ok((std::borrow::Cow::Owned(out), true))
}

fn restore_empty_aggregates(exchange: &mut Exchange) -> Result<()> {
    for section in &mut exchange.data {
        for entity in &mut section.entities {
            match entity {
                EntityInstance::Simple { record, .. } => {
                    restore_empty_aggregate_param(&mut record.parameter)?;
                }
                EntityInstance::Complex { subsuper, .. } => {
                    for record in &mut subsuper.0 {
                        restore_empty_aggregate_param(&mut record.parameter)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn restore_empty_aggregate_param(parameter: &mut Parameter) -> Result<()> {
    match parameter {
        Parameter::List(items) => {
            if items.len() == 1
                && matches!(
                    &items[0],
                    Parameter::Enumeration(value) if value == EMPTY_AGGREGATE_MARKER
                )
            {
                items.clear();
                return Ok(());
            }
            for item in items {
                restore_empty_aggregate_param(item)?;
            }
        }
        Parameter::Typed { parameter, .. } => {
            if matches!(
                parameter.as_ref(),
                Parameter::Enumeration(value) if value == EMPTY_AGGREGATE_MARKER
            ) {
                bail!("unsupported empty typed-parameter aggregate in STEP input");
            }
            restore_empty_aggregate_param(parameter)?;
        }
        Parameter::Enumeration(value) if value == EMPTY_AGGREGATE_MARKER => {
            bail!("empty-aggregate parser marker escaped its aggregate");
        }
        _ => {}
    }
    Ok(())
}

fn decode_input(input: &[u8]) -> Result<(std::borrow::Cow<'_, str>, &'static str)> {
    if let Ok(s) = std::str::from_utf8(input) {
        return Ok((std::borrow::Cow::Borrowed(s), "utf-8"));
    }

    let (decoded, _used_encoding, had_errors) = encoding_rs::GBK.decode(input);
    if had_errors {
        bail!("STEP input is neither valid UTF-8 nor valid GBK");
    }
    Ok((decoded, "gbk"))
}

fn write_step_string(s: &str, out: &mut String) {
    out.push('\'');

    let flush_non_ascii = |buf: &mut String, out: &mut String| {
        if buf.is_empty() {
            return;
        }
        out.push_str("\\X2\\");
        for unit in buf.encode_utf16() {
            let _ = write!(out, "{unit:04X}");
        }
        out.push_str("\\X0\\");
        buf.clear();
    };

    let mut non_ascii = String::new();
    for ch in s.chars() {
        if ch.is_ascii() && ch != '\'' {
            flush_non_ascii(&mut non_ascii, out);
            out.push(ch);
        } else {
            // Encode apostrophes too; this stays valid Part 21 and avoids
            // depending on ruststep's incomplete doubled-apostrophe parser.
            non_ascii.push(ch);
        }
    }
    flush_non_ascii(&mut non_ascii, out);
    out.push('\'');
}

fn format_real(v: f64) -> String {
    let mut s = v.to_string();
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        s.push('.');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wrap(data: &str) -> Vec<u8> {
        format!(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('x'),'1');\nFILE_NAME('a','b',(''),(''),'x','y','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n{data}\nENDSEC;\nEND-ISO-10303-21;\n"
        )
        .into_bytes()
    }

    #[test]
    fn writer_roundtrips_basic_exchange() {
        let src = wrap(
            "#9=CARTESIAN_POINT('NONE',(1.000000000000000000,2.500000000000000000,0.000000000000000000));\n#20=CARTESIAN_POINT('NONE',(1.0,2.5,0.0));\n#21=VERTEX_POINT('NONE',#20);",
        );
        let out = clean_bytes(&src, &Options::default()).unwrap();
        assert!(out.stats.output_bytes < out.stats.input_bytes);
        assert_eq!(out.stats.interned_entities, 1);
        ruststep::parser::parse(std::str::from_utf8(&out.bytes).unwrap()).unwrap();
    }

    #[test]
    fn equal_geometry_values_share_but_topology_identity_survives() {
        let src = wrap(
            "#1=CARTESIAN_POINT('',(1.0,2.0,3.0));\n#2=CARTESIAN_POINT('',(1.000000000000000000,2.0,3.0));\n#3=VERTEX_POINT('',#1);\n#4=VERTEX_POINT('',#2);",
        );
        let out = clean_bytes(&src, &Options::default()).unwrap();
        assert_eq!(out.stats.interned_entities, 1);
        assert_eq!(out.stats.output_entities, 3);

        let text = std::str::from_utf8(&out.bytes).unwrap();
        assert_eq!(text.matches("VERTEX_POINT").count(), 2);
        assert_eq!(text.matches("CARTESIAN_POINT").count(), 1);
    }

    #[test]
    fn consolidates_only_unreferenced_presentation_roots() {
        let src = wrap(
            "#1=CARTESIAN_POINT('',(0.0,0.0,0.0));\n\
             #8=DIRECTION('',(1.0,0.0,0.0));\n\
             #2=STYLED_ITEM('',(#8),#1);\n\
             #3=STYLED_ITEM('',(#8),#1);\n\
             #4=MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',(#2),#1);\n\
             #5=MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',(#3),#1);\n\
             #6=PRESENTATION_LAYER_ASSIGNMENT('','',(#2));\n\
             #7=PRESENTATION_LAYER_ASSIGNMENT('','',(#3));",
        );
        let out = clean_bytes(&src, &Options::default()).unwrap();
        assert_eq!(out.stats.consolidated_entities, 2);
        let text = std::str::from_utf8(&out.bytes).unwrap();
        assert_eq!(
            text.matches("MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION")
                .count(),
            1
        );
        assert_eq!(text.matches("PRESENTATION_LAYER_ASSIGNMENT").count(), 1);
    }

    #[test]
    fn referenced_presentation_records_are_not_consolidated() {
        let src = wrap(
            "#1=CARTESIAN_POINT('',(0.0,0.0,0.0));\n\
             #8=DIRECTION('',(1.0,0.0,0.0));\n\
             #2=STYLED_ITEM('',(#8),#1);\n\
             #3=STYLED_ITEM('',(#8),#1);\n\
             #4=MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',(#2),#1);\n\
             #5=MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',(#3),#1);\n\
             #6=REPRESENTATION_RELATIONSHIP('','',#4,#5);",
        );
        let out = clean_bytes(&src, &Options::default()).unwrap();
        assert_eq!(out.stats.consolidated_entities, 0);
    }

    #[test]
    fn legal_empty_aggregates_roundtrip_through_ruststep_compatibility_shim() {
        let src = wrap("#8=SHAPE_REPRESENTATION('',(),#6);\n#6=CARTESIAN_POINT('',(0.0,0.0,0.0));");
        let once = clean_bytes(&src, &Options::default()).unwrap();
        let text = std::str::from_utf8(&once.bytes).unwrap();
        assert!(text.contains("SHAPE_REPRESENTATION('',(),#"));
        assert!(!text.contains(EMPTY_AGGREGATE_MARKER));

        let twice = clean_bytes(&once.bytes, &Options::default()).unwrap();
        assert_eq!(once.bytes, twice.bytes);
    }

    #[test]
    fn empty_aggregate_shim_ignores_parentheses_inside_strings_and_comments() {
        let src = wrap(
            "#1=CARTESIAN_POINT('literal ()', (0.0,0.0,0.0));\n/* () */\n#2=SHAPE_REPRESENTATION('',( ),#1);",
        );
        let out = clean_bytes(&src, &Options::default()).unwrap();
        let text = std::str::from_utf8(&out.bytes).unwrap();
        assert!(text.contains("literal ()"));
        assert!(text.contains("SHAPE_REPRESENTATION('',(),#"));
        assert!(!text.contains(EMPTY_AGGREGATE_MARKER));
    }

    #[test]
    fn optional_placeholder_name_minification_only_touches_allowlisted_name_fields() {
        let src =
            wrap("#1=CARTESIAN_POINT('NONE',(0.0,0.0,0.0));\n#2=PRODUCT('NONE','NONE','NONE',());");
        let options = Options {
            minify_placeholder_names: true,
            ..Options::default()
        };
        let out = clean_bytes(&src, &options).unwrap();
        let text = std::str::from_utf8(&out.bytes).unwrap();
        assert!(text.contains("CARTESIAN_POINT('',"));
        assert!(text.contains("PRODUCT('NONE','NONE','NONE',())"));
        assert_eq!(out.stats.placeholder_names_minified, 1);
    }

    #[test]
    fn gbk_strings_become_standard_x2_unicode_escapes() {
        let mut src = b"ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('x'),'1');\nFILE_NAME('a','b',(''),(''),'x','y','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n#1=CARTESIAN_POINT('".to_vec();
        src.extend_from_slice(&[0xC8, 0xCE, 0xBA, 0xCE]); // 任何 in GBK
        src.extend_from_slice(b"',(0.0,0.0,0.0));\nENDSEC;\nEND-ISO-10303-21;\n");

        let out = clean_bytes(&src, &Options::default()).unwrap();
        assert_eq!(out.stats.input_encoding, "gbk");
        let text = std::str::from_utf8(&out.bytes).unwrap();
        assert!(text.contains("\\X2\\4EFB4F55\\X0\\"));
    }

    #[test]
    fn cleaning_is_byte_idempotent() {
        let src = wrap(
            "#10=DIRECTION('',(1.000000000000000000,0.0,0.0));\n#20=DIRECTION('',(1.0,0.0,0.0));\n#30=VECTOR('',#20,1000.000000000000000000);",
        );
        let once = clean_bytes(&src, &Options::default()).unwrap();
        let twice = clean_bytes(&once.bytes, &Options::default()).unwrap();
        assert_eq!(once.bytes, twice.bytes);
    }

    #[test]
    fn straight_bspline_recovery_is_byte_idempotent() {
        let src = wrap(
            "#1=CARTESIAN_POINT('',(0.,0.,0.));\n\
             #2=CARTESIAN_POINT('',(1.,0.,0.));\n\
             #3=CARTESIAN_POINT('',(2.,0.,0.));\n\
             #4=CARTESIAN_POINT('',(3.,0.,0.));\n\
             #5=VERTEX_POINT('',#1);\n\
             #6=VERTEX_POINT('',#4);\n\
             #7=B_SPLINE_CURVE_WITH_KNOTS('',3,(#1,#2,#3,#4),.UNSPECIFIED.,.F.,.F.,(4,4),(0.,1.),.UNSPECIFIED.);\n\
             #8=EDGE_CURVE('',#5,#6,#7,.T.);",
        );
        let options = Options {
            experimental_recover_straight_bspline_lines: true,
            ..Options::default()
        };
        let once = clean_bytes(&src, &options).unwrap();
        assert_eq!(once.stats.straight_bspline_lines_recovered, 1);
        let twice = clean_bytes(&once.bytes, &options).unwrap();
        assert_eq!(twice.stats.straight_bspline_lines_recovered, 0);
        assert_eq!(once.bytes, twice.bytes);
    }
}
