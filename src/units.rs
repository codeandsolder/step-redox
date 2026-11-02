use crate::step_document::{StepRecord, StepSectionView, StepValue};

const MAX_UNIT_RECURSION: usize = 8;

/// Resolve a STEP length unit to millimetres per source unit.
///
/// Supports direct SI length units and `CONVERSION_BASED_UNIT` chains whose
/// conversion factor is a `LENGTH_MEASURE_WITH_UNIT` / `MEASURE_WITH_UNIT`.
pub fn length_unit_scale_mm(unit_id: u64, section: &StepSectionView<'_>) -> Option<f64> {
    length_unit_scale_mm_inner(unit_id, section, 0)
}

fn length_unit_scale_mm_inner(
    unit_id: u64,
    section: &StepSectionView<'_>,
    depth: usize,
) -> Option<f64> {
    if depth > MAX_UNIT_RECURSION {
        return None;
    }

    let entity = section.entity(unit_id)?;
    entity.record_named("LENGTH_UNIT")?;

    if let Some(record) = entity.record_named("SI_UNIT") {
        return si_length_unit_scale_mm(record);
    }

    let params = entity
        .record_named("CONVERSION_BASED_UNIT")?
        .value()
        .list()?;
    if params.len() != 2 || params.get(0)?.string().is_none() {
        return None;
    }
    let factor_id = params.get(1)?.entity_ref()?;
    conversion_factor_scale_mm(factor_id, section, depth + 1)
}

fn conversion_factor_scale_mm(
    factor_id: u64,
    section: &StepSectionView<'_>,
    depth: usize,
) -> Option<f64> {
    let entity = section.entity(factor_id)?;
    let params = entity
        .record_named("LENGTH_MEASURE_WITH_UNIT")
        .or_else(|| entity.record_named("MEASURE_WITH_UNIT"))?
        .value()
        .list()?;
    if params.len() != 2 {
        return None;
    }

    let value = length_measure_value(params.get(0)?)?;
    let base_unit_id = params.get(1)?.entity_ref()?;
    let base_scale = length_unit_scale_mm_inner(base_unit_id, section, depth)?;
    let scale = value * base_scale;
    (scale.is_finite() && scale > 0.0).then_some(scale)
}

fn length_measure_value(parameter: StepValue<'_>) -> Option<f64> {
    let (keyword, parameter) = parameter.typed()?;
    if keyword != "LENGTH_MEASURE" {
        return None;
    }
    parameter.number()
}

fn si_length_unit_scale_mm(record: StepRecord<'_>) -> Option<f64> {
    let params = record.value().list()?;
    if params.len() != 2 {
        return None;
    }
    let prefix = params.get(0)?;
    let unit_name = params.get(1)?.enumeration()?;
    if unit_name != "METRE" {
        return None;
    }

    let metres = if prefix.is_not_provided() {
        1.0
    } else {
        match prefix.enumeration()? {
            "EXA" => 1.0e18,
            "PETA" => 1.0e15,
            "TERA" => 1.0e12,
            "GIGA" => 1.0e9,
            "MEGA" => 1.0e6,
            "KILO" => 1.0e3,
            "HECTO" => 1.0e2,
            "DECA" => 1.0e1,
            "DECI" => 1.0e-1,
            "CENTI" => 1.0e-2,
            "MILLI" => 1.0e-3,
            "MICRO" => 1.0e-6,
            "NANO" => 1.0e-9,
            "PICO" => 1.0e-12,
            "FEMTO" => 1.0e-15,
            "ATTO" => 1.0e-18,
            _ => return None,
        }
    };
    Some(metres * 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step_graph::build_index;

    fn parse_units(data: &str) -> anyhow::Result<ruststep::ast::Exchange> {
        let text = format!(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('x'),'1');\nFILE_NAME('a','b',(''),(''),'x','y','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n{data}\nENDSEC;\nEND-ISO-10303-21;\n"
        );
        Ok(ruststep::parser::parse(&text)?)
    }

    #[test]
    fn resolves_direct_and_conversion_based_length_units() -> anyhow::Result<()> {
        let exchange = parse_units(
            "#1=DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);\n\
             #2=(NAMED_UNIT(#1)LENGTH_UNIT()SI_UNIT(.MILLI.,.METRE.));\n\
             #3=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.0),#2);\n\
             #4=(CONVERSION_BASED_UNIT('MILLIMETRE',#3)LENGTH_UNIT()NAMED_UNIT(#1));\n\
             #5=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#2);\n\
             #6=(CONVERSION_BASED_UNIT('INCH',#5)LENGTH_UNIT()NAMED_UNIT(#1));",
        )?;
        let data = exchange
            .data
            .first()
            .ok_or_else(|| anyhow::anyhow!("parsed exchange contains no DATA section"))?;
        let index = build_index(&data.entities);
        let section = StepSectionView::new(&data.entities, &index);

        assert_eq!(length_unit_scale_mm(2, &section), Some(1.0));
        assert_eq!(length_unit_scale_mm(4, &section), Some(1.0));
        assert_eq!(length_unit_scale_mm(6, &section), Some(25.4));
        Ok(())
    }
}
