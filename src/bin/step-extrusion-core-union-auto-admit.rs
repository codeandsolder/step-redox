#[cfg(feature = "cad-kernel-monstertruck")]
use anyhow::{Context, Result};
#[cfg(feature = "cad-kernel-monstertruck")]
use std::path::PathBuf;

#[cfg(feature = "cad-kernel-monstertruck")]
fn main() -> Result<()> {
    use step_redox::cad_kernel::{CadKernel, monstertruck::MonstertruckKernel};

    let usage = "usage: step-extrusion-core-union-auto-admit INPUT.step DATA_SECTION_INDEX SOLID_ID OUTPUT.step";
    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(args.next().context(usage)?);
    let data_section = args
        .next()
        .context(usage)?
        .parse::<usize>()
        .context("parse zero-based DATA_SECTION_INDEX")?;
    let solid_id = args
        .next()
        .context(usage)?
        .parse::<u64>()
        .context("parse SOLID_ID")?;
    let output = PathBuf::from(args.next().context(usage)?);
    if args.next().is_some() {
        anyhow::bail!("{usage}");
    }

    let source = std::fs::read(&input).with_context(|| format!("read {}", input.display()))?;
    let kernel = MonstertruckKernel::project_standard();
    let Some(fragment) =
        step_redox::cad_recovery::recover_auto_proven_extrusion_core_union_fragment(
            &source,
            data_section,
            solid_id,
            &kernel,
        )?
    else {
        anyhow::bail!("no unique bounded exact extrusion-core union was admitted");
    };

    let evaluated = kernel.evaluate(&fragment.model, fragment.root)?;
    std::fs::write(&output, kernel.to_step(&evaluated)?)
        .with_context(|| format!("write {}", output.display()))?;
    serde_json::to_writer_pretty(std::io::stdout(), &fragment.source)?;
    println!();
    Ok(())
}

#[cfg(not(feature = "cad-kernel-monstertruck"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!(
        "step-extrusion-core-union-auto-admit requires --features cad-kernel-monstertruck"
    )
}
