use anyhow::{Context, Result, bail};
use std::path::PathBuf;
use truck_stepio::r#in::*;

fn shell_fingerprints(path: &PathBuf) -> Result<Vec<Vec<u8>>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let exchange = ruststep::parser::parse(&text).context("parse STEP")?;
    let section = exchange.data.first().context("missing DATA section")?;
    let table = Table::from_data_section(section);

    let mut fingerprints = Vec::with_capacity(table.shell.len());
    for shell in table.shell.values() {
        let compressed = table
            .to_compressed_shell(shell)
            .map_err(|e| anyhow::anyhow!("convert shell: {e}"))?;
        fingerprints.push(serde_json::to_vec(&compressed)?);
    }
    fingerprints.sort();
    Ok(fingerprints)
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let a = PathBuf::from(args.next().context("reference STEP path")?);
    let b = PathBuf::from(args.next().context("candidate STEP path")?);
    if args.next().is_some() {
        bail!("usage: compare_geometry REFERENCE.step CANDIDATE.step");
    }

    let fa = shell_fingerprints(&a)?;
    let fb = shell_fingerprints(&b)?;
    println!(
        "reference_shells={} candidate_shells={}",
        fa.len(),
        fb.len()
    );
    if fa != fb {
        bail!("compressed-shell geometry/topology differs");
    }
    println!("MATCH");
    Ok(())
}
