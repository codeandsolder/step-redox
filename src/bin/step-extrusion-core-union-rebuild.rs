#[cfg(feature = "cad-kernel-monstertruck")]
use anyhow::{Context, Result, bail};
#[cfg(feature = "cad-kernel-monstertruck")]
use std::path::PathBuf;

#[cfg(feature = "cad-kernel-monstertruck")]
fn main() -> Result<()> {
    use step_redox::cad_kernel::{CadKernel, monstertruck::MonstertruckKernel};

    let usage = "usage: step-extrusion-core-union-rebuild INPUT.step SOLID_ID OUTPUT.step CAP_A CAP_B [CAP_A CAP_B ...]";
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

    let bytes = std::fs::read(&input).with_context(|| format!("read {}", input.display()))?;
    let mut fragments = Vec::with_capacity(pair_args.len() / 2);
    for pair in pair_args.chunks_exact(2) {
        let cap_a = pair[0]
            .parse::<u64>()
            .with_context(|| format!("parse CAP_A {}", pair[0]))?;
        let cap_b = pair[1]
            .parse::<u64>()
            .with_context(|| format!("parse CAP_B {}", pair[1]))?;
        let core = step_redox::recover_extrusion_core_from_cap_faces_bytes(
            &bytes,
            solid_id,
            [cap_a, cap_b],
        )?
        .with_context(|| {
            format!("no translated-cap core for solid #{solid_id} caps #{cap_a}/#{cap_b}")
        })?;
        fragments.push(step_redox::cad_recovery::recover_solid_extrusion_fragment(
            &core,
        )?);
    }

    let (model, root) = step_redox::cad_recovery::compose_union_model(&fragments)?;
    let kernel = MonstertruckKernel;
    let evaluated = kernel.evaluate(&model, root)?;
    let summary = kernel.summarize(&evaluated);
    if !summary.geometrically_consistent {
        bail!("Monstertruck built a geometrically inconsistent core union");
    }
    if summary.shells != 1 {
        bail!(
            "Monstertruck core union has {} boundary shells, expected 1",
            summary.shells
        );
    }
    std::fs::write(&output, kernel.to_step(&evaluated)?)
        .with_context(|| format!("write {}", output.display()))?;
    Ok(())
}

#[cfg(not(feature = "cad-kernel-monstertruck"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("step-extrusion-core-union-rebuild requires --features cad-kernel-monstertruck")
}
