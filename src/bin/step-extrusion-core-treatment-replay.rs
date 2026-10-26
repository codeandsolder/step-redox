#[cfg(feature = "cad-kernel-monstertruck")]
use anyhow::{Context, Result, bail};
#[cfg(feature = "cad-kernel-monstertruck")]
use std::path::PathBuf;

#[cfg(feature = "cad-kernel-monstertruck")]
fn main() -> Result<()> {
    use step_redox::cad_ir::{CadNode, EdgeSelector, EdgeTreatmentEdge, EdgeTreatmentProfile};
    use step_redox::cad_kernel::{CadKernel, monstertruck::MonstertruckKernel};
    use step_redox::edge_treatments::{EdgeTreatmentKind, EdgeTreatmentVirtualEdge};

    let usage =
        "usage: step-extrusion-core-treatment-replay INPUT.step SOLID_ID CAP_A CAP_B OUTPUT_PREFIX";
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
    let output_prefix = PathBuf::from(args.next().context(usage)?);
    if args.next().is_some() {
        bail!("{usage}");
    }

    let bytes = std::fs::read(&input).with_context(|| format!("read {}", input.display()))?;
    let candidate = step_redox::recover_treatment_healed_extrusion_core_from_cap_faces_bytes(
        &bytes,
        solid_id,
        [cap_a, cap_b],
    )?
    .with_context(|| {
        format!("no treatment-healed extrusion core for solid #{solid_id} caps #{cap_a}/#{cap_b}")
    })?;
    let evidence = step_redox::detect_edge_treatment_evidence_bytes(&bytes)?;
    let mut treatments = candidate
        .healed_treatment_face_ids
        .iter()
        .map(|face_id| {
            evidence
                .iter()
                .find(|item| {
                    item.solid_id == solid_id
                        && item.treatment_face_id == *face_id
                        && item.kind == EdgeTreatmentKind::PlanarChamfer
                        && item.carrier_pair_unique
                        && item.inferred_setbacks_mm.is_some()
                })
                .cloned()
                .with_context(|| {
                    format!("no replayable planar chamfer evidence for face #{face_id}")
                })
        })
        .collect::<Result<Vec<_>>>()?;
    treatments.sort_by_key(|item| item.treatment_face_id);
    if treatments.is_empty() || treatments.len() > 12 {
        bail!(
            "expected 1..=12 healed planar chamfers, got {}",
            treatments.len()
        );
    }

    let fragment = step_redox::cad_recovery::recover_solid_extrusion_fragment(&candidate.core)?;
    let kernel = MonstertruckKernel;
    let variants = 1_usize
        .checked_shl(u32::try_from(treatments.len()).context("too many treatment variants")?)
        .context("treatment variant count overflow")?;

    for mask in 0..variants {
        let mut model = fragment.model.clone();
        let mut root = fragment.root;
        for (index, treatment) in treatments.iter().enumerate() {
            let [mut first_setback_mm, mut second_setback_mm] =
                treatment.inferred_setbacks_mm.context("missing setbacks")?;
            if mask & (1_usize << index) != 0 {
                std::mem::swap(&mut first_setback_mm, &mut second_setback_mm);
            }
            let selector = match treatment.virtual_edge {
                EdgeTreatmentVirtualEdge::Segment { start_mm, end_mm } => {
                    EdgeSelector::Segment { start_mm, end_mm }
                }
                EdgeTreatmentVirtualEdge::Circle { .. } => {
                    bail!(
                        "asymmetric planar chamfer face #{} unexpectedly has a circular virtual edge",
                        treatment.treatment_face_id
                    )
                }
            };
            root = model.add_node(CadNode::EdgeTreatment {
                child: root,
                profile: EdgeTreatmentProfile::AsymmetricChamfer {
                    first_setback_mm,
                    second_setback_mm,
                },
                edges: vec![EdgeTreatmentEdge {
                    selector,
                    radius_mm: first_setback_mm.min(second_setback_mm),
                }],
            });
        }
        model.validate()?;
        let output = output_prefix.with_extension(format!(
            "mask-{mask:0width$b}.step",
            width = treatments.len()
        ));
        match kernel.evaluate(&model, root) {
            Ok(evaluated) => {
                let summary = kernel.summarize(&evaluated);
                if !summary.geometrically_consistent || summary.shells != 1 {
                    eprintln!(
                        "mask={mask:0width$b} invalid {summary:?}",
                        width = treatments.len()
                    );
                    continue;
                }
                std::fs::write(&output, kernel.to_step(&evaluated)?)
                    .with_context(|| format!("write {}", output.display()))?;
                eprintln!(
                    "mask={mask:0width$b} ok faces={} output={}",
                    summary.faces,
                    output.display(),
                    width = treatments.len()
                );
            }
            Err(error) => {
                eprintln!(
                    "mask={mask:0width$b} error={error:#}",
                    width = treatments.len()
                );
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "cad-kernel-monstertruck"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!(
        "step-extrusion-core-treatment-replay requires --features cad-kernel-monstertruck"
    )
}
