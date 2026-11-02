use ruststep::ast::{EntityInstance, Parameter, Record};
use std::collections::HashMap;

/// Parser-neutral read-only view over one STEP DATA section.
///
/// The current frontend is `step-p21`, but consumers of this module do not see
/// its AST types. This is the migration seam for replacing or reshaping the
/// parser later without dragging semantic-recovery code through that change.
pub(crate) struct StepSection<'a> {
    data_section: Option<usize>,
    entities: &'a [EntityInstance],
    index: &'a HashMap<u64, usize>,
}

impl<'a> StepSection<'a> {
    pub(crate) const fn new(
        data_section: usize,
        entities: &'a [EntityInstance],
        index: &'a HashMap<u64, usize>,
    ) -> Self {
        Self {
            data_section: Some(data_section),
            entities,
            index,
        }
    }

    pub(crate) const fn unlocated(
        entities: &'a [EntityInstance],
        index: &'a HashMap<u64, usize>,
    ) -> Self {
        Self {
            data_section: None,
            entities,
            index,
        }
    }

    pub(crate) fn entity(&self, id: u64) -> Option<StepEntity<'a>> {
        let ordinal = *self.index.get(&id)?;
        let entity = self.entities.get(ordinal)?;
        Some(StepEntity {
            source: SourceEntity {
                data_section: self.data_section,
                ordinal,
                id,
            },
            entity,
        })
    }
}

/// Stable source identity for provenance journals built above the parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SourceEntity {
    pub(crate) data_section: Option<usize>,
    pub(crate) ordinal: usize,
    pub(crate) id: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct StepEntity<'a> {
    source: SourceEntity,
    entity: &'a EntityInstance,
}

impl<'a> StepEntity<'a> {
    pub(crate) const fn source(self) -> SourceEntity {
        self.source
    }

    pub(crate) fn record_named(self, name: &str) -> Option<StepRecord<'a>> {
        match self.entity {
            EntityInstance::Simple { record, .. } => {
                (record.name == name).then_some(StepRecord(record))
            }
            EntityInstance::Complex { subsuper, .. } => subsuper
                .0
                .iter()
                .find(|record| record.name == name)
                .map(StepRecord),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct StepRecord<'a>(&'a Record);

impl<'a> StepRecord<'a> {
    pub(crate) fn value(self) -> StepValue<'a> {
        StepValue(&self.0.parameter)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct StepValue<'a>(&'a Parameter);

impl<'a> StepValue<'a> {
    pub(crate) fn list(self) -> Option<StepList<'a>> {
        match self.0 {
            Parameter::List(items) => Some(StepList(items)),
            _ => None,
        }
    }

    pub(crate) fn string(self) -> Option<&'a str> {
        match self.0 {
            Parameter::String(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn enumeration(self) -> Option<&'a str> {
        match self.0 {
            Parameter::Enumeration(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn entity_ref(self) -> Option<u64> {
        match self.0 {
            Parameter::Ref(ruststep::ast::Name::Entity(id)) => Some(*id),
            _ => None,
        }
    }

    pub(crate) fn typed(self) -> Option<(&'a str, StepValue<'a>)> {
        match self.0 {
            Parameter::Typed { keyword, parameter } => {
                Some((keyword, StepValue(parameter.as_ref())))
            }
            _ => None,
        }
    }

    pub(crate) fn number(self) -> Option<f64> {
        match self.0 {
            Parameter::Real(value) => Some(*value),
            Parameter::Integer(value) => crate::numeric::exact_i64_to_f64(*value),
            _ => None,
        }
    }

    pub(crate) const fn is_not_provided(self) -> bool {
        matches!(self.0, Parameter::NotProvided)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct StepList<'a>(&'a [Parameter]);

impl<'a> StepList<'a> {
    pub(crate) const fn len(self) -> usize {
        self.0.len()
    }

    pub(crate) fn get(self, index: usize) -> Option<StepValue<'a>> {
        self.0.get(index).map(StepValue)
    }

    pub(crate) fn iter(self) -> impl Iterator<Item = StepValue<'a>> {
        self.0.iter().map(StepValue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step_graph::build_index;

    #[test]
    fn hides_parser_ast_behind_source_stable_views() -> anyhow::Result<()> {
        let exchange = ruststep::parser::parse(
            "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('x'),'1');\nFILE_NAME('a','b',(''),(''),'x','y','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n#10=(NAMED_UNIT(#1)LENGTH_UNIT()SI_UNIT(.MILLI.,.METRE.));\nENDSEC;\nEND-ISO-10303-21;\n",
        )?;
        let section = exchange
            .data
            .first()
            .ok_or_else(|| anyhow::anyhow!("missing DATA section"))?;
        let index = build_index(&section.entities);
        let document = StepSection::new(3, &section.entities, &index);
        let entity = document
            .entity(10)
            .ok_or_else(|| anyhow::anyhow!("missing entity #10"))?;

        assert_eq!(
            entity.source(),
            SourceEntity {
                data_section: Some(3),
                ordinal: 0,
                id: 10,
            }
        );
        let si = entity
            .record_named("SI_UNIT")
            .ok_or_else(|| anyhow::anyhow!("missing SI_UNIT record"))?;
        let params = si
            .value()
            .list()
            .ok_or_else(|| anyhow::anyhow!("SI_UNIT parameters are not a list"))?;
        assert_eq!(params.len(), 2);
        assert_eq!(
            params.get(0).and_then(StepValue::enumeration),
            Some("MILLI")
        );
        assert_eq!(
            params.get(1).and_then(StepValue::enumeration),
            Some("METRE")
        );
        Ok(())
    }
}
