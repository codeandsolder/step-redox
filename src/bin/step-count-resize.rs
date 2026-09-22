use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum AnchorArg {
    Start,
    Center,
    End,
}

impl From<AnchorArg> for step_redox::CountAnchor {
    fn from(value: AnchorArg) -> Self {
        match value {
            AnchorArg::Start => step_redox::CountAnchor::Start,
            AnchorArg::Center => step_redox::CountAnchor::Center,
            AnchorArg::End => step_redox::CountAnchor::End,
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "step-count-resize",
    about = "Resize one recovered semantic count parameter, including body and coupled instance rows"
)]
struct Cli {
    input: PathBuf,
    output: PathBuf,

    #[arg(long, default_value_t = 0)]
    parameter: usize,

    #[arg(long)]
    sites: usize,

    #[arg(long, value_enum, default_value_t = AnchorArg::Start)]
    anchor: AnchorArg,

    #[arg(
        long,
        help = "Input is already normalized semantic STEP; skip the editable compact normalization pass"
    )]
    already_semantic: bool,

    #[arg(long, value_name = "PATH")]
    parameters_json: Option<PathBuf>,

    #[arg(long, value_name = "PATH")]
    patterns_json: Option<PathBuf>,

    #[arg(long, value_name = "PATH")]
    periodic_bodies_json: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let input =
        std::fs::read(&cli.input).with_context(|| format!("read {}", cli.input.display()))?;

    let semantic = if cli.already_semantic {
        input
    } else {
        let mut options = step_redox::Options::for_profile(step_redox::OutputProfile::Compact);
        // Whole-body/count recovery must run before low-level curve factoring.
        // Keep the editable semantic graph explicit for the graph mutator.
        options.experimental_instance_translated_bspline_curves = false;
        step_redox::clean_bytes(&input, &options)?.bytes
    };

    let edited = step_redox::resize_count_parameter_bytes_with_anchor(
        &semantic,
        cli.parameter,
        cli.sites,
        cli.anchor.into(),
    )?;

    std::fs::write(&cli.output, &edited.bytes)
        .with_context(|| format!("write {}", cli.output.display()))?;

    if let Some(path) = &cli.parameters_json {
        std::fs::write(path, serde_json::to_vec_pretty(&edited.count_parameters)?)
            .with_context(|| format!("write {}", path.display()))?;
    }
    if let Some(path) = &cli.patterns_json {
        std::fs::write(path, serde_json::to_vec_pretty(&edited.patterns)?)
            .with_context(|| format!("write {}", path.display()))?;
    }
    if let Some(path) = &cli.periodic_bodies_json {
        std::fs::write(path, serde_json::to_vec_pretty(&edited.periodic_bodies)?)
            .with_context(|| format!("write {}", path.display()))?;
    }

    eprintln!("{}", serde_json::to_string(&edited.resize)?);
    Ok(())
}
