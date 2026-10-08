use crate::step_entities::parameter_number;
use crate::step_graph::entity_ref_value;
use ruststep::ast::{EntityInstance, Parameter, Record};
use std::collections::HashMap;

const MAX_UNIT_RECURSION: usize = 8;

/// Resolve a STEP length unit to millimetres per source unit.
///
/// Supports direct SI length units and `CONVERSION_BASED_UNIT` chains whose
/// conversion factor is a `LENGTH_MEASURE_WITH_UNIT` / `MEASURE_WITH_UNIT`.
pub fn length_unit_scale_mm(
    unit_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<f64> {
    length_unit_scale_mm_inner(unit_id, entities, index, 0)
}

fn length_unit_scale_mm_inner(
    unit_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    depth: usize,
) -> Option<f64> {
    if depth > MAX_UNIT_RECURSION {
        return None;
    }

    let entity = entities.get(*index.get(&unit_id)?)?;
    entity_record_named(entity, "LENGTH_UNIT")?;

    if let Some(record) = entity_record_named(entity, "SI_UNIT") {
        return si_length_unit_scale_mm(record);
    }

    let record = entity_record_named(entity, "CONVERSION_BASED_UNIT")?;
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let [Parameter::String(_name), conversion_factor] = params.as_slice() else {
        return None;
    };
    let factor_id = entity_ref_value(conversion_factor)?;
    conversion_factor_scale_mm(factor_id, entities, index, depth + 1)
}

fn conversion_factor_scale_mm(
    factor_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
    depth: usize,
) -> Option<f64> {
    let entity = entities.get(*index.get(&factor_id)?)?;
    let record = entity_record_named(entity, "LENGTH_MEASURE_WITH_UNIT")
        .or_else(|| entity_record_named(entity, "MEASURE_WITH_UNIT"))?;
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let [measure, unit] = params.as_slice() else {
        return None;
    };

    let value = length_measure_value(measure)?;
    let base_unit_id = entity_ref_value(unit)?;
    let base_scale = length_unit_scale_mm_inner(base_unit_id, entities, index, depth)?;
    let scale = value * base_scale;
    (scale.is_finite() && scale > 0.0).then_some(scale)
}

fn length_measure_value(parameter: &Parameter) -> Option<f64> {
    let Parameter::Typed { keyword, parameter } = parameter else {
        return None;
    };
    if keyword != "LENGTH_MEASURE" {
        return None;
    }
    parameter_number(parameter)
}

fn si_length_unit_scale_mm(record: &Record) -> Option<f64> {
    let Parameter::List(params) = &record.parameter else {
        return None;
    };
    let [prefix, Parameter::Enumeration(unit_name)] = params.as_slice() else {
        return None;
    };
    if unit_name != "METRE" {
        return None;
    }

    let metres = match prefix {
        Parameter::NotProvided => 1.0,
        Parameter::Enumeration(prefix) => match prefix.as_str() {
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
        },
        _ => return None,
    };
    Some(metres * 1000.0)
}

fn entity_record_named<'a>(entity: &'a EntityInstance, name: &str) -> Option<&'a Record> {
    match entity {
        EntityInstance::Simple { record, .. } => (record.name == name).then_some(record),
        EntityInstance::Complex { subsuper, .. } => {
            subsuper.0.iter().find(|record| record.name == name)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_units(data: &str) -> anyhow::Result<(Vec<EntityInstance>, HashMap<u64, usize>)> {
        let text = format!(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('x'),'1');\nFILE_NAME('a','b',(''),(''),'x','y','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n{data}\nENDSEC;\nEND-ISO-10303-21;\n"
        );
        let exchange = ruststep::parser::parse(&text)?;
        let entities = exchange
            .data
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("parsed exchange contains no DATA section"))?
            .entities;
        let index = entities
            .iter()
            .enumerate()
            .map(|(idx, entity)| {
                let id = match entity {
                    EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id,
                };
                (id, idx)
            })
            .collect();
        Ok((entities, index))
    }

    #[test]
    fn resolves_direct_and_conversion_based_length_units() -> anyhow::Result<()> {
        let (entities, index) = parse_units(
            "#1=DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);\n\
             #2=(NAMED_UNIT(#1)LENGTH_UNIT()SI_UNIT(.MILLI.,.METRE.));\n\
             #3=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.0),#2);\n\
             #4=(CONVERSION_BASED_UNIT('MILLIMETRE',#3)LENGTH_UNIT()NAMED_UNIT(#1));\n\
             #5=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#2);\n\
             #6=(CONVERSION_BASED_UNIT('INCH',#5)LENGTH_UNIT()NAMED_UNIT(#1));",
        )?;

        assert_eq!(length_unit_scale_mm(2, &entities, &index), Some(1.0));
        assert_eq!(length_unit_scale_mm(4, &entities, &index), Some(1.0));
        assert_eq!(length_unit_scale_mm(6, &entities, &index), Some(25.4));
        Ok(())
    }
}
