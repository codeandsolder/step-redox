use anyhow::{Context, Result, bail};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = env::args_os();
    let _exe = args.next();
    let input = PathBuf::from(
        args.next()
            .context("usage: step-complete-ir INPUT OUTPUT")?,
    );
    let output = PathBuf::from(
        args.next()
            .context("usage: step-complete-ir INPUT OUTPUT")?,
    );
    if args.next().is_some() {
        bail!("usage: step-complete-ir INPUT OUTPUT");
    }

    let bytes = std::fs::read(&input).with_context(|| format!("read {}", input.display()))?;
    let report = step_redox::complete_ir::recover_complete_ir_bytes(&bytes)?;
    let data = serde_json::to_vec_pretty(&report)?;
    std::fs::write(&output, data).with_context(|| format!("write {}", output.display()))?;
    Ok(())
}
