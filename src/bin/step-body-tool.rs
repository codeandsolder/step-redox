use anyhow::Result;
use clap::{Parser, Subcommand};
use std::io::Write as _;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Scan and extract deduplicatable MANIFOLD_SOLID_BREP bodies")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Emit one JSON record per manifold solid; unsupported canonical identities fall back file-locally.
    Scan { input: PathBuf },
    /// Extract selected bodies as standalone STEP files.
    Extract {
        input: PathBuf,
        out_dir: PathBuf,
        /// Selection formatted as `DATA_SECTION:SOLID_ID:FINGERPRINT`.
        #[arg(long = "solid")]
        solids: Vec<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Scan { input } => {
            let json = step_redox::body_corpus::scan_json(&input)?;
            std::io::stdout().lock().write_all(json.as_bytes())?;
        }
        Command::Extract {
            input,
            out_dir,
            solids,
        } => step_redox::body_corpus::extract_selected(&input, &out_dir, &solids)?,
    }
    Ok(())
}
