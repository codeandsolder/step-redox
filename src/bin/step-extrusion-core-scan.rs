use anyhow::{Context, Result};
use std::path::PathBuf;

fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: step-extrusion-core-scan FILE.step")?;
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    serde_json::to_writer(
        std::io::stdout(),
        &step_redox::detect_extrusion_core_candidates_bytes(&bytes)?,
    )?;
    Ok(())
}
