#[cfg(feature = "cad-kernel-monstertruck")]
use anyhow::{Context, Result, bail};
#[cfg(feature = "cad-kernel-monstertruck")]
use std::path::PathBuf;

#[cfg(feature = "cad-kernel-monstertruck")]
fn main() -> Result<()> {
    use step_redox::cad_kernel::{CadKernel, monstertruck::MonstertruckKernel};

    let usage = "usage: step-extrusion-core-union-admit INPUT.step SOLID_ID OUTPUT.step CAP_A CAP_B [CAP_A CAP_B ...]";
    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(args.next().context(usage)?);
    let solid_id = args
        .next()
        .context(usage)?
        .parse::<u64>()
        .context("parse SOLID_ID")?;
    let output = PathBuf::from(args.next().context(usage)?);
    let pair_args = args.collect::<Vec<_>>();
    if pair_args.len() < 4 || !pair_args.len().is_multiple_of(2) {
        bail!("{usage}");
    }
    let cap_pairs = pair_args
        .chunks_exact(2)
        .map(|pair| {
            Ok([
                pair[0]
                    .parse::<u64>()
                    .with_context(|| format!("parse CAP_A {}", pair[0]))?,
                pair[1]
                    .parse::<u64>()
                    .with_context(|| format!("parse CAP_B {}", pair[1]))?,
            ])
        })
        .collect::<Result<Vec<_>>>()?;

    let source = std::fs::read(&input).with_context(|| format!("read {}", input.display()))?;
    let kernel = MonstertruckKernel;
    let fragment = step_redox::cad_recovery::recover_proven_extrusion_core_union_fragment(
        &source, solid_id, &cap_pairs, &kernel,
    )?
    .context("extrusion-core union was not admitted")?;

    let evaluated = kernel.evaluate(&fragment.model, fragment.root)?;
    std::fs::write(&output, kernel.to_step(&evaluated)?)
        .with_context(|| format!("write {}", output.display()))?;
    serde_json::to_writer_pretty(std::io::stdout(), &fragment.source)?;
    println!();
    Ok(())
}

#[cfg(not(feature = "cad-kernel-monstertruck"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("step-extrusion-core-union-admit requires --features cad-kernel-monstertruck")
}
