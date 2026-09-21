use anyhow::{Result, bail};
use step_redox::cad_ir::{CadModel, CadNode, Profile2d, emit_kcl};
use step_redox::cad_kernel::CadKernel;
use step_redox::cad_kernel::monstertruck::MonstertruckKernel;

fn validate_kcl(kcl_text: &str) -> Result<String> {
    #[cfg(feature = "kcl-conformance")]
    {
        let program = kcl_lib::Program::parse_no_errs(kcl_text)?;
        let recast = program.recast();
        let reparsed = kcl_lib::Program::parse_no_errs(&recast)?;
        if recast != reparsed.recast() {
            bail!("KCL parse/recast is not idempotent");
        }
        Ok(recast)
    }
    #[cfg(not(feature = "kcl-conformance"))]
    {
        Ok(kcl_text.to_owned())
    }
}

fn main() -> Result<()> {
    let mut model = CadModel::new();
    let profile = Profile2d::polygon(vec![[0.0, 0.0], [10.0, 0.0], [10.0, 6.0], [0.0, 6.0]])?;
    let body = model.add_node(CadNode::Extrude {
        profile,
        vector_mm: [0.0, 0.0, 2.0],
    });
    model.add_root(body)?;

    let kcl_text = validate_kcl(&emit_kcl(&model)?)?;
    let kernel = MonstertruckKernel;
    let evaluated = kernel.evaluate(&model, body)?;
    let summary = kernel.summarize(&evaluated);
    if !summary.geometrically_consistent {
        bail!("Monstertruck solid is inconsistent");
    }
    let step = kernel.to_step(&evaluated)?;

    std::fs::write("box.kcl", &kcl_text)?;
    std::fs::write("box.step", step)?;
    println!(
        "ok: kcl_bytes={} truck_faces={} step_bytes={}",
        kcl_text.len(),
        summary.faces,
        std::fs::metadata("box.step")?.len()
    );
    Ok(())
}
