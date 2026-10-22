use anyhow::{Context, Result, anyhow};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let first_path = args.next().map(PathBuf::from).ok_or_else(|| {
        anyhow!("usage: step-boundary-compare FIRST.step FIRST_SOLID SECOND.step SECOND_SOLID")
    })?;
    let first_solid_id = args
        .next()
        .ok_or_else(|| anyhow!("missing FIRST_SOLID"))?
        .to_string_lossy()
        .parse::<u64>()
        .context("parse FIRST_SOLID")?;
    let second_path = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("missing SECOND.step"))?;
    let second_solid_id = args
        .next()
        .ok_or_else(|| anyhow!("missing SECOND_SOLID"))?
        .to_string_lossy()
        .parse::<u64>()
        .context("parse SECOND_SOLID")?;
    if args.next().is_some() {
        return Err(anyhow!("unexpected extra arguments"));
    }

    let first =
        std::fs::read(&first_path).with_context(|| format!("read {}", first_path.display()))?;
    let second =
        std::fs::read(&second_path).with_context(|| format!("read {}", second_path.display()))?;
    let comparison = step_redox::compare_analytic_solid_boundaries_bytes(
        &first,
        first_solid_id,
        &second,
        second_solid_id,
    )?;
    serde_json::to_writer_pretty(std::io::stdout(), &comparison)?;
    println!();
    Ok(())
}
