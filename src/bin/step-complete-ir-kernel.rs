#[cfg(feature = "cad-kernel-monstertruck")]
use anyhow::{Context, Result, bail};
#[cfg(feature = "cad-kernel-monstertruck")]
use std::path::PathBuf;

#[cfg(feature = "cad-kernel-monstertruck")]
fn main() -> Result<()> {
    use step_redox::cad_kernel::monstertruck::MonstertruckKernel;

    let usage = "usage: step-complete-ir-kernel INPUT OUTPUT";
    let mut args = std::env::args_os().skip(1);
    let input = PathBuf::from(args.next().context(usage)?);
    let output = PathBuf::from(args.next().context(usage)?);
    if args.next().is_some() {
        bail!("{usage}");
    }

    let bytes = std::fs::read(&input).with_context(|| format!("read {}", input.display()))?;
    let report = step_redox::complete_ir::recover_complete_ir_with_kernel_bytes(
        &bytes,
        &MonstertruckKernel,
    )?;
    let data = serde_json::to_vec_pretty(&report)?;
    std::fs::write(&output, data).with_context(|| format!("write {}", output.display()))?;
    Ok(())
}

#[cfg(not(feature = "cad-kernel-monstertruck"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("step-complete-ir-kernel requires --features cad-kernel-monstertruck")
}
