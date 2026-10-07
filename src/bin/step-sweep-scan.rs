use anyhow::{Context, Result};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Serialize)]
struct SweepScan {
    closed_round: Vec<step_redox::solid_sweeps::RecoveredClosedRoundSweep>,
    open_rectangular: Vec<step_redox::solid_sweeps::RecoveredOpenRectangularSweep>,
}

fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: step-sweep-scan FILE.step"))?;
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    serde_json::to_writer(
        std::io::stdout(),
        &SweepScan {
            closed_round: step_redox::detect_closed_round_sweeps_bytes(&bytes)?,
            open_rectangular: step_redox::detect_open_rectangular_sweeps_bytes(&bytes)?,
        },
    )?;
    Ok(())
}
