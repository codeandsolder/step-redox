use super::{CleanOutput, Options, Stats, audit_exchange_compatibility, detect_exchange_semantics};
use crate::normalization::{
    consolidate_presentation, dense_renumber, intern_section, minify_placeholder_names,
};
use crate::step_io::decode_input;
use crate::{
    bezier_recovery, curve_replicas, face_coalesce, geometric_intern, instances, line_recovery,
    partition_recovery, planar_features, spherical_caps, surface_recovery, surface_replicas,
    write_exchange,
};
use anyhow::{Context, Result, bail};
use ruststep::ast::Exchange;

pub(super) fn clean_bytes(input: &[u8], options: &Options) -> Result<CleanOutput> {
    let (input_text, input_encoding) = decode_input(input)?;
    let mut exchange =
        ruststep::parser::parse(&input_text).context("parse STEP exchange structure")?;

    if !exchange.anchor.is_empty()
        || !exchange.reference.is_empty()
        || !exchange.signature.is_empty()
    {
        bail!("ANCHOR/REFERENCE/SIGNATURE sections are not yet supported by step-redox writer");
    }

    let input_entities: usize = exchange.data.iter().map(|d| d.entities.len()).sum();
    let mut stats = Stats {
        input_encoding: input_encoding.to_string(),
        input_bytes: input.len(),
        input_entities,
        ..Stats::default()
    };

    run_initial_normalization(&mut exchange, options, &mut stats);

    run_geometry_recovery(&mut exchange, options, &mut stats);

    run_instancing(&mut exchange, options, &mut stats);

    stabilize_output(&mut exchange, options, &mut stats);

    let (patterns, periodic_bodies, count_parameters) = detect_exchange_semantics(&exchange);
    stats.instance_patterns_detected = patterns.len();
    stats.pattern_instances_detected = patterns.iter().map(|pattern| pattern.item_ids.len()).sum();
    stats.periodic_body_patterns_detected = periodic_bodies.len();
    stats.periodic_body_repeat_faces = periodic_bodies.iter().map(|body| body.repeat_faces).sum();
    stats.count_parameters_detected = count_parameters.len();
    stats.count_parameters_with_body_grammar = count_parameters
        .iter()
        .filter(|parameter| parameter.body_grammar_proven)
        .count();

    let compatibility = audit_exchange_compatibility(&exchange);
    let output = write_exchange(&exchange)?;
    stats.output_entities = exchange.data.iter().map(|d| d.entities.len()).sum();
    stats.output_bytes = output.len();
    stats.byte_ratio = stats.output_bytes as f64 / input.len().max(1) as f64;

    Ok(CleanOutput {
        bytes: output.into_bytes(),
        stats,
        patterns,
        periodic_bodies,
        count_parameters,
        compatibility,
    })
}
fn run_initial_normalization(exchange: &mut Exchange, options: &Options, stats: &mut Stats) {
    if options.minify_placeholder_names {
        for section in &mut exchange.data {
            stats.placeholder_names_minified += minify_placeholder_names(&mut section.entities);
        }
    }

    if options.intern_values {
        for section in &mut exchange.data {
            let pass = intern_section(&mut section.entities);
            stats.interned_entities += pass.total;
            for (k, v) in pass.by_type {
                *stats.interned_by_type.entry(k).or_insert(0) += v;
            }
        }
    }

    if options.consolidate_presentation {
        for section in &mut exchange.data {
            let pass = consolidate_presentation(&mut section.entities);
            stats.consolidated_entities += pass.total;
            for (k, v) in pass.by_type {
                *stats.consolidated_by_type.entry(k).or_insert(0) += v;
            }
        }
    }
}

fn run_geometry_recovery(exchange: &mut Exchange, options: &Options, stats: &mut Stats) {
    if options.experimental_recover_straight_bspline_lines {
        for section in &mut exchange.data {
            let pass = line_recovery::recover_straight_bspline_lines(&mut section.entities);
            stats.straight_bspline_lines_recovered += pass.curves_recovered;
            stats.straight_bspline_direction_groups += pass.direction_groups;
            stats.straight_bspline_points_removed += pass.orphan_points_removed;
        }
    }

    if options.experimental_recover_exact_bezier_curves {
        for section in &mut exchange.data {
            let pass = bezier_recovery::recover_exact_bezier_curves(&mut section.entities);
            stats.exact_bezier_curves_recovered += pass.curves_recovered;
        }
    }

    if options.experimental_recover_v_extrusions {
        for section in &mut exchange.data {
            let pass = surface_recovery::recover_v_extrusion_surfaces(&mut section.entities);
            stats.v_extrusion_surfaces_recovered += pass.surfaces_recovered;
            stats.v_extrusion_rational_surfaces_recovered += pass.rational_surfaces_recovered;
            stats.v_extrusion_profile_curves_created += pass.profile_curves_created;
            stats.v_extrusion_points_removed += pass.orphan_points_removed;
        }
    }

    // V-extrusion recovery materializes profile B-splines after the first
    // Bezier pass. Revisit exact Bezier recovery once so eligible generated
    // profiles are canonicalized in the same clean instead of on clean #2.
    if options.experimental_recover_exact_bezier_curves
        && stats.v_extrusion_profile_curves_created > 0
    {
        for section in &mut exchange.data {
            let pass = bezier_recovery::recover_exact_bezier_curves(&mut section.entities);
            stats.exact_bezier_curves_recovered += pass.curves_recovered;
        }
    }

    if options.experimental_intern_geometric_supports {
        for section in &mut exchange.data {
            let pass = geometric_intern::intern_geometric_supports(&mut section.entities);
            stats.geometric_supports_merged += pass.supports_merged;
            stats.geometric_support_entities_removed += pass.entities_removed;
            stats.geometric_planes_merged += pass.planes_merged;
            stats.geometric_lines_merged += pass.lines_merged;
            stats.geometric_cylinders_merged += pass.cylinders_merged;
        }
    }

    if options.experimental_recover_partitioned_bodies {
        for section in &mut exchange.data {
            let pass = partition_recovery::recover_partitioned_bodies(&mut section.entities);
            stats.partition_components_recovered += pass.components;
            stats.partition_solids_merged += pass.solids_merged;
            stats.partition_interfaces_removed += pass.interfaces_removed;
            stats.partition_styles_retargeted += pass.styles_retargeted;
            stats.partition_entities_removed += pass.entities_removed;
        }
    }

    if options.experimental_coalesce_same_support_faces {
        for section in &mut exchange.data {
            let pass = face_coalesce::coalesce_same_support_faces(&mut section.entities);
            stats.face_coalesce_groups += pass.groups;
            stats.face_coalesce_faces_merged += pass.faces_merged;
            stats.face_coalesce_faces_removed += pass.faces_removed;
            stats.face_coalesce_internal_edges_removed += pass.internal_edges_removed;
            stats.face_coalesce_styles_removed += pass.styles_removed;
            stats.face_coalesce_entities_removed += pass.entities_removed;
        }
    } else if options.coalesce_same_support_planar_faces {
        for section in &mut exchange.data {
            let pass = face_coalesce::coalesce_same_support_planar_faces(&mut section.entities);
            stats.face_coalesce_groups += pass.groups;
            stats.face_coalesce_faces_merged += pass.faces_merged;
            stats.face_coalesce_faces_removed += pass.faces_removed;
            stats.face_coalesce_internal_edges_removed += pass.internal_edges_removed;
            stats.face_coalesce_styles_removed += pass.styles_removed;
            stats.face_coalesce_entities_removed += pass.entities_removed;
        }
    }
}

fn run_instancing(exchange: &mut Exchange, options: &Options, stats: &mut Stats) {
    if options.experimental_instance_z90_assembly {
        for section in &mut exchange.data {
            let pass = instances::instance_z90_solids_assembly(&mut section.entities);
            stats.instance_groups += pass.groups;
            stats.instanced_solids += pass.solids_replaced;
            stats.instance_entities_removed += pass.entities_removed;
            stats.instance_styles_replaced += pass.styles_replaced;
        }
    } else if options.experimental_instance_z90 {
        for section in &mut exchange.data {
            let pass = instances::instance_z90_solids(&mut section.entities);
            stats.instance_groups += pass.groups;
            stats.instanced_solids += pass.solids_replaced;
            stats.instance_entities_removed += pass.entities_removed;
            stats.instance_styles_replaced += pass.styles_replaced;
        }
    }

    if options.experimental_instance_planar_positive_features {
        for section in &mut exchange.data {
            let pass = planar_features::instance_planar_positive_features(&mut section.entities);
            stats.planar_feature_arrays += pass.arrays;
            stats.planar_feature_families += pass.families;
            stats.planar_feature_instances += pass.instances;
            stats.planar_feature_entities_removed += pass.entities_removed;
            stats.planar_feature_styles_replaced += pass.styles_replaced;
        }
    }

    if options.experimental_instance_spherical_caps {
        for section in &mut exchange.data {
            let pass = spherical_caps::instance_planar_spherical_caps(&mut section.entities);
            stats.spherical_cap_arrays += pass.arrays;
            stats.spherical_cap_instances += pass.instances;
            stats.spherical_cap_entities_removed += pass.entities_removed;
            stats.spherical_cap_styles_replaced += pass.styles_replaced;
        }
    }

    // Low-level translated support factoring also comes after whole-body and
    // feature recognition. It preserves face/topology entities and only shares
    // exact-parameterization PLANE/CYLINDRICAL_SURFACE supports by translation.
    if options.experimental_instance_translated_analytic_surfaces {
        for section in &mut exchange.data {
            let pass =
                surface_replicas::instance_translated_analytic_surfaces(&mut section.entities);
            stats.surface_replica_families += pass.families;
            stats.surface_replicas += pass.replicas;
            stats.surface_replica_planes += pass.plane_replicas;
            stats.surface_replica_cylinders += pass.cylinder_replicas;
            stats.surface_replica_transforms += pass.transforms;
            stats.surface_replica_entities_removed += pass.entities_removed;
            stats.surface_replica_max_transform_residual_mm = stats
                .surface_replica_max_transform_residual_mm
                .max(pass.max_transform_residual_mm);
        }
    }

    // Low-level curve factoring comes last. Higher-level body/feature repetition
    // must be recognized against the actual geometry first; otherwise a
    // CURVE_REPLICA decomposition can leak global source coordinates into a
    // later rigid-body signature and hide obvious whole-solid instances.
    if options.experimental_instance_translated_bspline_curves {
        for section in &mut exchange.data {
            let pass = curve_replicas::instance_translated_bspline_curves(&mut section.entities);
            stats.curve_replica_families += pass.families;
            stats.curve_replicas += pass.replicas;
            stats.curve_replica_direct_aliases += pass.direct_aliases;
            stats.curve_replica_transforms += pass.transforms;
            stats.curve_replica_entities_removed += pass.entities_removed;
            stats.curve_replica_max_residual_mm = stats
                .curve_replica_max_residual_mm
                .max(pass.max_residual_mm);
        }
    }
}

fn stabilize_output(exchange: &mut Exchange, options: &Options, stats: &mut Stats) {
    // Higher-level rewrites can remove the only non-topological users of a
    // geometric support and thereby make a locus alias newly safe. Re-run the
    // support pass after those rewrites so Compact reaches a fixed point in one
    // invocation instead of discovering the alias on clean #2.
    if options.experimental_intern_geometric_supports
        && (stats.partition_components_recovered > 0
            || stats.face_coalesce_groups > 0
            || stats.instance_groups > 0
            || stats.planar_feature_arrays > 0
            || stats.spherical_cap_arrays > 0
            || stats.curve_replicas > 0
            || stats.curve_replica_direct_aliases > 0
            || stats.surface_replicas > 0)
    {
        for section in &mut exchange.data {
            let pass = geometric_intern::intern_geometric_supports(&mut section.entities);
            stats.geometric_supports_merged += pass.supports_merged;
            stats.geometric_support_entities_removed += pass.entities_removed;
            stats.geometric_planes_merged += pass.planes_merged;
            stats.geometric_lines_merged += pass.lines_merged;
            stats.geometric_cylinders_merged += pass.cylinders_merged;
        }
    }

    // Experimental passes can create new placeholder-labelled entities.
    // Minify those before the post-rewrite intern pass so name normalization
    // cannot create fresh duplicates that only disappear on a second run.
    if options.minify_placeholder_names {
        for section in &mut exchange.data {
            stats.placeholder_names_minified += minify_placeholder_names(&mut section.entities);
        }
    }

    // Experimental passes create placements/directions and other support
    // values. Normalize them in the same invocation so aggressive output is a
    // fixed point rather than requiring a second safe cleanup pass.
    if (stats.straight_bspline_lines_recovered > 0
        || stats.v_extrusion_surfaces_recovered > 0
        || stats.geometric_supports_merged > 0
        || stats.partition_components_recovered > 0
        || stats.face_coalesce_groups > 0
        || stats.curve_replicas > 0
        || stats.curve_replica_direct_aliases > 0
        || stats.instance_groups > 0
        || stats.planar_feature_arrays > 0
        || stats.spherical_cap_arrays > 0)
        && options.intern_values
    {
        for section in &mut exchange.data {
            let pass = intern_section(&mut section.entities);
            stats.interned_entities += pass.total;
            for (k, v) in pass.by_type {
                *stats.interned_by_type.entry(k).or_insert(0) += v;
            }
        }
    }

    if options.dense_ids {
        for section in &mut exchange.data {
            dense_renumber(&mut section.entities);
        }
    }
}
