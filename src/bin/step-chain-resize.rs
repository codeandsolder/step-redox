use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "step-chain-resize",
    about = "Resize one proven fused-solid periodic chain"
)]
struct Cli {
    input: PathBuf,
    output: PathBuf,

    #[arg(long, default_value_t = 0)]
    chain: usize,

    #[arg(long)]
    sites: usize,

    #[arg(
        long,
        help = "Input is already normalized semantic STEP; skip compact normalization"
    )]
    already_semantic: bool,

    #[arg(long, value_name = "PATH")]
    chains_json: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let input =
        std::fs::read(&cli.input).with_context(|| format!("read {}", cli.input.display()))?;

    let semantic = if cli.already_semantic {
        input
    } else {
        step_redox::clean_bytes(
            &input,
            &step_redox::Options::for_profile(step_redox::OutputProfile::Compact),
        )?
        .bytes
    };

    let edited = step_redox::resize_periodic_chain_bytes(&semantic, cli.chain, cli.sites)?;

    std::fs::write(&cli.output, &edited.bytes)
        .with_context(|| format!("write {}", cli.output.display()))?;

    if let Some(path) = &cli.chains_json {
        std::fs::write(path, serde_json::to_vec_pretty(&edited.periodic_chains)?)
            .with_context(|| format!("write {}", path.display()))?;
    }

    eprintln!("{}", serde_json::to_string(&edited.resize)?);
    Ok(())
}
