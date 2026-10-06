#[cfg(feature = "cad-kernel-monstertruck")]
use anyhow::Context;
use anyhow::{Result, bail};
#[cfg(feature = "cad-kernel-monstertruck")]
use std::path::PathBuf;

#[cfg(feature = "cad-kernel-monstertruck")]
fn main() -> Result<()> {
    use step_redox::cad_kernel::{CadKernel, monstertruck::MonstertruckKernel};

    let mut args = std::env::args_os().skip(1).map(PathBuf::from);
    let input = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: step-cad-rebuild INPUT.step OUTPUT.step"))?;
    let output = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: step-cad-rebuild INPUT.step OUTPUT.step"))?;
    if args.next().is_some() {
        bail!("usage: step-cad-rebuild INPUT.step OUTPUT.step");
    }

    let bytes = std::fs::read(&input).with_context(|| format!("read {}", input.display()))?;
    let report = step_redox::complete_ir::recover_complete_ir_bytes(&bytes)?;
    if report.body_count != 1 || report.constructively_recovered_solid_count != 1 {
        bail!(
            "expected exactly one constructively recovered body, got bodies={} constructive={}",
            report.body_count,
            report.constructively_recovered_solid_count
        );
    }

    let mut fragments = report
        .solid_extrusions
        .into_iter()
        .chain(report.solid_revolutions)
        .chain(report.radial_slot_revolutions)
        .chain(report.solid_sweeps)
        .collect::<Vec<_>>();
    if fragments.len() != 1 {
        bail!(
            "expected exactly one selected constructive fragment, got {}",
            fragments.len()
        );
    }
    let fragment = fragments.pop().expect("length checked");
    let kernel = MonstertruckKernel;
    let evaluated = kernel.evaluate(&fragment.model, fragment.root)?;
    let summary = kernel.summarize(&evaluated);
    if !summary.geometrically_consistent {
        bail!("Monstertruck rebuilt a geometrically inconsistent solid");
    }
    std::fs::write(&output, kernel.to_step(&evaluated)?)
        .with_context(|| format!("write {}", output.display()))?;
    Ok(())
}

#[cfg(not(feature = "cad-kernel-monstertruck"))]
fn main() -> Result<()> {
    bail!("step-cad-rebuild requires --features cad-kernel-monstertruck")
}
