use anyhow::{Context, Result};
use truck_stepio::r#in::*;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("STEP path")?;
    let text = std::fs::read_to_string(&path)?;
    let exchange = ruststep::parser::parse(&text)?;
    let section = exchange.data.first().context("missing DATA section")?;
    let table = Table::from_data_section(section);

    let mut converted = 0usize;
    for shell in table.shell.values() {
        table
            .to_compressed_shell(shell)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        converted += 1;
    }
    println!("path={path} shells={converted}");
    Ok(())
}
