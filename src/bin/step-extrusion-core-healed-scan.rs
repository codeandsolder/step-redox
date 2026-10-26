use anyhow::{Context, Result, bail};
use std::path::PathBuf;

fn main() -> Result<()> {
    let usage = "usage: step-extrusion-core-healed-scan INPUT.step SOLID_ID CAP_A CAP_B";
    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(args.next().context(usage)?);
    let solid_id = args
        .next()
        .context(usage)?
        .parse::<u64>()
        .context("parse SOLID_ID")?;
    let cap_a = args
        .next()
        .context(usage)?
        .parse::<u64>()
        .context("parse CAP_A")?;
    let cap_b = args
        .next()
        .context(usage)?
        .parse::<u64>()
        .context("parse CAP_B")?;
    if args.next().is_some() {
        bail!("{usage}");
    }
    let bytes = std::fs::read(&input).with_context(|| format!("read {}", input.display()))?;
    serde_json::to_writer(
        std::io::stdout(),
        &step_redox::recover_treatment_healed_extrusion_core_from_cap_faces_bytes(
            &bytes,
            solid_id,
            [cap_a, cap_b],
        )?,
    )?;
    Ok(())
}
