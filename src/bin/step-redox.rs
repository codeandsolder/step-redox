use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "step-redox",
    about = "Semantics-preserving STEP Part 21 cleanup"
)]
struct Cli {
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
        help = "Experimental: instance congruent top-level solids under Z quarter-turns"
    )]
    experimental_instance_z90: bool,

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
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let input =
        std::fs::read(&cli.input).with_context(|| format!("read {}", cli.input.display()))?;
    let options = step_redox::Options {
        intern_values: !cli.no_intern,
        consolidate_presentation: !cli.no_presentation_consolidation,
        experimental_recover_straight_bspline_lines: cli
            .experimental_recover_straight_bspline_lines,
        experimental_instance_z90: cli.experimental_instance_z90,
        experimental_instance_planar_positive_features: cli
            .experimental_instance_planar_positive_features,
        experimental_instance_spherical_caps: cli.experimental_instance_spherical_caps,
        minify_placeholder_names: cli.minify_placeholder_names,
        dense_ids: !cli.no_dense_ids,
    };
    let cleaned = step_redox::clean_bytes(&input, &options)?;
    std::fs::write(&cli.output, &cleaned.bytes)
        .with_context(|| format!("write {}", cli.output.display()))?;

    if cli.json {
        eprintln!("{}", serde_json::to_string(&cleaned.stats)?);
    } else {
        eprintln!(
            "{} -> {} bytes ({:.2}%), entities {} -> {} (interned {}, consolidated {}, straight B-splines {}, direction groups {}, points removed {}, instance groups {}, solids {}, removed {}, planar arrays {}, families {}, instances {}, removed {}, spherical arrays {}, instances {}, removed {}, placeholder names {})",
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
        for (ty, n) in &cleaned.stats.interned_by_type {
            eprintln!("  intern {ty}: {n}");
        }
        for (ty, n) in &cleaned.stats.consolidated_by_type {
            eprintln!("  consolidate {ty}: {n}");
        }
    }
    Ok(())
}
