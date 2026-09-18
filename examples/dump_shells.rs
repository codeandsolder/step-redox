use anyhow::{Context, Result};
use truck_stepio::r#in::*;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("STEP path")?;
    let text = std::fs::read_to_string(path)?;
    let exchange = ruststep::parser::parse(&text)?;
    let table = Table::from_data_section(&exchange.data[0]);
    for (id, shell) in &table.shell {
        let c = table
            .to_compressed_shell(shell)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("#SHELL {id}");
        println!("{}", serde_json::to_string(&c)?);
    }
    Ok(())
}
