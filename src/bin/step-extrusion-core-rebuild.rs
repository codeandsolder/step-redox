#[cfg(feature = "cad-kernel-monstertruck")]
use anyhow::{Context, Result, bail};
#[cfg(feature = "cad-kernel-monstertruck")]
use std::path::PathBuf;

#[cfg(feature = "cad-kernel-monstertruck")]
fn main() -> Result<()> {
    use step_redox::cad_kernel::{CadKernel, monstertruck::MonstertruckKernel};

    let usage = "usage: step-extrusion-core-rebuild INPUT.step SOLID_ID CAP_A CAP_B OUTPUT.step";
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
    let output = PathBuf::from(args.next().context(usage)?);
    if args.next().is_some() {
        bail!("{usage}");
    }

    let bytes = std::fs::read(&input).with_context(|| format!("read {}", input.display()))?;
    let core =
        step_redox::recover_extrusion_core_from_cap_faces_bytes(&bytes, solid_id, [cap_a, cap_b])?
            .with_context(|| {
                format!("no translated-cap core for solid #{solid_id} caps #{cap_a}/#{cap_b}")
            })?;
    let fragment = step_redox::cad_recovery::recover_solid_extrusion_fragment(&core)?;

    let kernel = MonstertruckKernel;
    let evaluated = kernel.evaluate(&fragment.model, fragment.root)?;
    let summary = kernel.summarize(&evaluated);
    if !summary.geometrically_consistent {
        bail!("Monstertruck rebuilt a geometrically inconsistent extrusion core");
    }
    std::fs::write(&output, kernel.to_step(&evaluated)?)
        .with_context(|| format!("write {}", output.display()))?;
    Ok(())
}

#[cfg(not(feature = "cad-kernel-monstertruck"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("step-extrusion-core-rebuild requires --features cad-kernel-monstertruck")
}
