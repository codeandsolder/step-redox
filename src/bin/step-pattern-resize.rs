use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Anchor {
    Start,
    Center,
    End,
}

impl From<Anchor> for step_redox::patterns::PatternAnchor {
    fn from(value: Anchor) -> Self {
        match value {
            Anchor::Start => Self::Start,
            Anchor::Center => Self::Center,
            Anchor::End => Self::End,
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "step-pattern-resize",
    about = "Resize one detected filled 1-D MAPPED_ITEM lattice in normalized STEP"
)]
struct Cli {
    input: PathBuf,
    output: PathBuf,

    #[arg(long, default_value_t = 0)]
    pattern: usize,

    #[arg(long)]
    count: usize,

    #[arg(long, value_enum, default_value_t = Anchor::Center)]
    anchor: Anchor,

    #[arg(long, value_name = "PATH")]
    patterns_json: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let input =
        std::fs::read(&cli.input).with_context(|| format!("read {}", cli.input.display()))?;
    let edited = step_redox::resize_linear_pattern_bytes(
        &input,
        cli.pattern,
        cli.count,
        cli.anchor.into(),
    )?;
    std::fs::write(&cli.output, &edited.bytes)
        .with_context(|| format!("write {}", cli.output.display()))?;

    if let Some(path) = &cli.patterns_json {
        std::fs::write(path, serde_json::to_vec_pretty(&edited.patterns)?)
            .with_context(|| format!("write {}", path.display()))?;
    }

    eprintln!("{}", serde_json::to_string(&edited.resize)?);
    Ok(())
}
