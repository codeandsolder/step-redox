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
    let kernel = MonstertruckKernel::project_standard();
    let report = step_redox::complete_ir::recover_complete_ir_with_kernel_bytes(&bytes, &kernel)?;
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
        .chain(report.periodic_chains)
        .chain(report.boundary_feature_composites)
        .chain(report.edge_treatment_composites)
        .chain(report.coplanar_normalized_solids)
        .chain(report.extrusion_core_unions)
        .map(|recovered| recovered.fragment)
        .collect::<Vec<_>>();
    if fragments.len() != 1 {
        bail!(
            "expected exactly one selected constructive fragment, got {}",
            fragments.len()
        );
    }
    let fragment = fragments
        .pop()
        .context("selected fragment disappeared after cardinality check")?;
    let evaluated = kernel.evaluate(&fragment.model, fragment.root)?;
    let summary = kernel.summarize(&evaluated);
    if !summary.geometrically_consistent || summary.shells != 1 {
        bail!("Monstertruck did not rebuild one geometrically consistent solid: {summary:?}");
    }
    std::fs::write(&output, kernel.to_step(&evaluated)?)
        .with_context(|| format!("write {}", output.display()))?;
    Ok(())
}

#[cfg(not(feature = "cad-kernel-monstertruck"))]
fn main() -> Result<()> {
    bail!("step-cad-rebuild requires --features cad-kernel-monstertruck")
}
