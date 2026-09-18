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
        dense_ids: !cli.no_dense_ids,
    };
    let cleaned = step_redox::clean_bytes(&input, &options)?;
    std::fs::write(&cli.output, &cleaned.bytes)
        .with_context(|| format!("write {}", cli.output.display()))?;

    if cli.json {
        eprintln!("{}", serde_json::to_string(&cleaned.stats)?);
    } else {
        eprintln!(
            "{} -> {} bytes ({:.2}%), entities {} -> {} (interned {}, consolidated {})",
            cleaned.stats.input_bytes,
            cleaned.stats.output_bytes,
            cleaned.stats.byte_ratio * 100.0,
            cleaned.stats.input_entities,
            cleaned.stats.output_entities,
            cleaned.stats.interned_entities,
            cleaned.stats.consolidated_entities,
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
