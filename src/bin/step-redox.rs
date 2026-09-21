use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Profile {
    /// Only explicitly requested experimental transformations.
    Manual,
    /// Conservative plain-BREP cleanup: avoid mapped/replica/assembly factoring.
    Compat,
    /// Aggressive compact STEP: enable proven semantic recovery and instancing.
    Compact,
}

#[derive(Parser, Debug)]
#[command(
    name = "step-redox",
    about = "Semantics-preserving STEP Part 21 cleanup"
)]
struct Cli {
    #[arg(
        long,
        value_enum,
        default_value_t = Profile::Manual,
        help = "Output policy: manual flags, broadly compatible B-rep, or compact STEP"
    )]
    profile: Profile,
    input: PathBuf,
    output: PathBuf,

    #[arg(long)]
    no_intern: bool,

    #[arg(long)]
    no_presentation_consolidation: bool,

    #[arg(
        long,
        help = "Experimental: replace strictly straight clamped B-spline edge curves with LINE"
    )]
    experimental_recover_straight_bspline_lines: bool,

    #[arg(
        long,
        help = "Experimental: canonicalize exact single-span clamped non-rational B-splines as BEZIER_CURVE"
    )]
    experimental_recover_exact_bezier_curves: bool,

    #[arg(
        long,
        alias = "experimental-recover-rational-v-extrusions",
        help = "Experimental: recover exact parameter-order-preserving B-spline V sweeps as SURFACE_OF_LINEAR_EXTRUSION"
    )]
    experimental_recover_v_extrusions: bool,

    #[arg(
        long,
        help = "Experimental: merge geometrically identical PLANE/LINE/CYLINDER supports at 1e-5 mm"
    )]
    experimental_intern_geometric_supports: bool,

    #[arg(
        long,
        help = "Experimental: recover bodies split by exact presentation-neutral planar partition faces"
    )]
    experimental_recover_partitioned_bodies: bool,

    #[arg(
        long,
        help = "Experimental: merge adjacent faces sharing the exact same support surface and sense"
    )]
    experimental_coalesce_same_support_faces: bool,

    #[arg(
        long,
        help = "Experimental: factor translation-equivalent 3-D B-spline curves as CURVE_REPLICA"
    )]
    experimental_instance_translated_bspline_curves: bool,

    #[arg(
        long,
        help = "Experimental: instance congruent top-level solids under Z quarter-turns"
    )]
    experimental_instance_z90: bool,

    #[arg(
        long,
        help = "Experimental: instance congruent top-level solids as STEP assembly occurrences"
    )]
    experimental_instance_z90_assembly: bool,

    #[arg(
        long,
        help = "Experimental: factor repeated positive features attached to planar closed-shell faces"
    )]
    experimental_instance_planar_positive_features: bool,

    #[arg(
        long,
        help = "Experimental: split planar-attached spherical-cap arrays and instance one closed feature"
    )]
    experimental_instance_spherical_caps: bool,

    #[arg(
        long,
        help = "Minify exact 'NONE' placeholder names on geometry/topology entities"
    )]
    minify_placeholder_names: bool,

    #[arg(long)]
    no_dense_ids: bool,

    #[arg(long, help = "Print machine-readable statistics")]
    json: bool,

    #[arg(long, value_name = "PATH", help = "Write detected regular instance patterns as JSON")]
    patterns_json: Option<PathBuf>,

    #[arg(long, value_name = "PATH", help = "Write detected periodic body grammars as JSON")]
    periodic_bodies_json: Option<PathBuf>,

    #[arg(long, value_name = "PATH", help = "Write recovered semantic count parameters as JSON")]
    count_parameters_json: Option<PathBuf>,

    #[arg(
        long,
        value_name = "PATH",
        help = "Write read-only periodic fused-solid chain analysis as JSON"
    )]
    periodic_chains_json: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let input =
        std::fs::read(&cli.input).with_context(|| format!("read {}", cli.input.display()))?;

    // Profiles are deliberately policy bundles, not different geometry engines.
    // Explicit flags can still opt into an extra transformation on top of a
    // profile for development/testing.
    let mut options = match cli.profile {
        Profile::Manual => step_redox::Options::default(),
        Profile::Compat => {
            step_redox::Options::for_profile(step_redox::OutputProfile::Compat)
        }
        Profile::Compact => {
            step_redox::Options::for_profile(step_redox::OutputProfile::Compact)
        }
    };
    options.intern_values = !cli.no_intern;
    options.consolidate_presentation = !cli.no_presentation_consolidation;
    options.dense_ids = !cli.no_dense_ids;
    options.experimental_recover_straight_bspline_lines |=
        cli.experimental_recover_straight_bspline_lines;
    options.experimental_recover_exact_bezier_curves |=
        cli.experimental_recover_exact_bezier_curves;
    options.experimental_recover_v_extrusions |= cli.experimental_recover_v_extrusions;
    options.experimental_intern_geometric_supports |= cli.experimental_intern_geometric_supports;
    options.experimental_recover_partitioned_bodies |=
        cli.experimental_recover_partitioned_bodies;
    options.experimental_coalesce_same_support_faces |=
        cli.experimental_coalesce_same_support_faces;
    options.experimental_instance_translated_bspline_curves |=
        cli.experimental_instance_translated_bspline_curves;
    options.experimental_instance_z90 |= cli.experimental_instance_z90;
    options.experimental_instance_z90_assembly |= cli.experimental_instance_z90_assembly;
    options.experimental_instance_planar_positive_features |=
        cli.experimental_instance_planar_positive_features;
    options.experimental_instance_spherical_caps |= cli.experimental_instance_spherical_caps;
    options.minify_placeholder_names |= cli.minify_placeholder_names;
    let cleaned = step_redox::clean_bytes(&input, &options)?;
    std::fs::write(&cli.output, &cleaned.bytes)
        .with_context(|| format!("write {}", cli.output.display()))?;
    if let Some(path) = &cli.patterns_json {
        let data = serde_json::to_vec_pretty(&cleaned.patterns)?;
        std::fs::write(path, data)
            .with_context(|| format!("write pattern report {}", path.display()))?;
    }
    if let Some(path) = &cli.periodic_bodies_json {
        let data = serde_json::to_vec_pretty(&cleaned.periodic_bodies)?;
        std::fs::write(path, data)
            .with_context(|| format!("write periodic body report {}", path.display()))?;
    }
    if let Some(path) = &cli.count_parameters_json {
        let data = serde_json::to_vec_pretty(&cleaned.count_parameters)?;
        std::fs::write(path, data)
            .with_context(|| format!("write count parameter report {}", path.display()))?;
    }
    if let Some(path) = &cli.periodic_chains_json {
        let chains = step_redox::detect_periodic_chains_bytes(&cleaned.bytes)?;
        let data = serde_json::to_vec_pretty(&chains)?;
        std::fs::write(path, data)
            .with_context(|| format!("write periodic chain report {}", path.display()))?;
    }

    if cli.json {
        eprintln!("{}", serde_json::to_string(&cleaned.stats)?);
    } else {
        eprintln!(
            "{} -> {} bytes ({:.2}%), entities {} -> {} (interned {}, consolidated {}, straight B-splines {}, direction groups {}, points removed {}, V extrusions {}, rational {}, profile curves {}, points removed {}, geometric supports {}, support entities removed {}, instance groups {}, solids {}, removed {}, planar arrays {}, families {}, instances {}, removed {}, spherical arrays {}, instances {}, removed {}, placeholder names {})",
            cleaned.stats.input_bytes,
            cleaned.stats.output_bytes,
            cleaned.stats.byte_ratio * 100.0,
            cleaned.stats.input_entities,
            cleaned.stats.output_entities,
            cleaned.stats.interned_entities,
            cleaned.stats.consolidated_entities,
            cleaned.stats.straight_bspline_lines_recovered,
            cleaned.stats.straight_bspline_direction_groups,
            cleaned.stats.straight_bspline_points_removed,
            cleaned.stats.v_extrusion_surfaces_recovered,
            cleaned.stats.v_extrusion_rational_surfaces_recovered,
            cleaned.stats.v_extrusion_profile_curves_created,
            cleaned.stats.v_extrusion_points_removed,
            cleaned.stats.geometric_supports_merged,
            cleaned.stats.geometric_support_entities_removed,
            cleaned.stats.instance_groups,
            cleaned.stats.instanced_solids,
            cleaned.stats.instance_entities_removed,
            cleaned.stats.planar_feature_arrays,
            cleaned.stats.planar_feature_families,
            cleaned.stats.planar_feature_instances,
            cleaned.stats.planar_feature_entities_removed,
            cleaned.stats.spherical_cap_arrays,
            cleaned.stats.spherical_cap_instances,
            cleaned.stats.spherical_cap_entities_removed,
            cleaned.stats.placeholder_names_minified,
        );
        if cleaned.stats.exact_bezier_curves_recovered > 0
            || cleaned.stats.partition_components_recovered > 0
            || cleaned.stats.face_coalesce_groups > 0
        {
            eprintln!(
                "  semantic recovery: exact Beziers {}, partition components {} ({} solids, {} interfaces, {} entities removed), co-surface groups {} ({} faces removed, {} entities removed)",
                cleaned.stats.exact_bezier_curves_recovered,
                cleaned.stats.partition_components_recovered,
                cleaned.stats.partition_solids_merged,
                cleaned.stats.partition_interfaces_removed,
                cleaned.stats.partition_entities_removed,
                cleaned.stats.face_coalesce_groups,
                cleaned.stats.face_coalesce_faces_removed,
                cleaned.stats.face_coalesce_entities_removed,
            );
        }
        for (ty, n) in &cleaned.stats.interned_by_type {
            eprintln!("  intern {ty}: {n}");
        }
        for (ty, n) in &cleaned.stats.consolidated_by_type {
            eprintln!("  consolidate {ty}: {n}");
        }
    }
    Ok(())
}
