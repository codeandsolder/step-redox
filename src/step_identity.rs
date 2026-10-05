use ruststep::ast::{Name, Parameter};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Debug)]
pub(crate) enum IndexBucket {
    One(usize),
    Many(Vec<usize>),
}

impl IndexBucket {
    pub(crate) const fn one(index: usize) -> Self {
        Self::One(index)
    }

    pub(crate) fn find(&self, mut predicate: impl FnMut(usize) -> bool) -> Option<usize> {
        match self {
            Self::One(index) => predicate(*index).then_some(*index),
            Self::Many(indices) => indices.iter().copied().find(|&index| predicate(index)),
        }
    }

    pub(crate) fn push(&mut self, index: usize) {
        match self {
            Self::One(first) => *self = Self::Many(vec![*first, index]),
            Self::Many(indices) => indices.push(index),
        }
    }
}

pub(crate) fn hash_parameter(param: &Parameter, hasher: &mut impl Hasher) {
    hash_parameter_impl(param, None, hasher);
}

pub(crate) fn hash_parameter_with_alias(
    param: &Parameter,
    alias: &HashMap<u64, u64>,
    hasher: &mut impl Hasher,
) {
    hash_parameter_impl(param, Some(alias), hasher);
}

fn hash_parameter_impl(
    param: &Parameter,
    alias: Option<&HashMap<u64, u64>>,
    hasher: &mut impl Hasher,
) {
    match param {
        Parameter::Typed { keyword, parameter } => {
            0u8.hash(hasher);
            keyword.hash(hasher);
            hash_parameter_impl(parameter, alias, hasher);
        }
        Parameter::Integer(value) => {
            1u8.hash(hasher);
            value.hash(hasher);
        }
        Parameter::Real(value) => {
            2u8.hash(hasher);
            value.to_bits().hash(hasher);
        }
        Parameter::String(value) => {
            3u8.hash(hasher);
            value.hash(hasher);
        }
        Parameter::Enumeration(value) => {
            4u8.hash(hasher);
            value.hash(hasher);
        }
        Parameter::List(items) => {
            5u8.hash(hasher);
            items.len().hash(hasher);
            for item in items {
                hash_parameter_impl(item, alias, hasher);
            }
        }
        Parameter::Ref(Name::Entity(id)) => {
            6u8.hash(hasher);
            canonical_entity_id(alias, *id).hash(hasher);
        }
        Parameter::Ref(Name::Value(id)) => {
            7u8.hash(hasher);
            id.hash(hasher);
        }
        Parameter::Ref(Name::ConstantEntity(value)) => {
            8u8.hash(hasher);
            value.hash(hasher);
        }
        Parameter::Ref(Name::ConstantValue(value)) => {
            9u8.hash(hasher);
            value.hash(hasher);
        }
        Parameter::NotProvided => 10u8.hash(hasher),
        Parameter::Omitted => 11u8.hash(hasher),
    }
}

pub(crate) fn parameters_equivalent(left: &Parameter, right: &Parameter) -> bool {
    parameters_equivalent_impl(left, right, None)
}

pub(crate) fn parameters_equivalent_with_alias(
    left: &Parameter,
    right: &Parameter,
    alias: &HashMap<u64, u64>,
) -> bool {
    parameters_equivalent_impl(left, right, Some(alias))
}

fn parameters_equivalent_impl(
    left: &Parameter,
    right: &Parameter,
    alias: Option<&HashMap<u64, u64>>,
) -> bool {
    match (left, right) {
        (
            Parameter::Typed {
                keyword: left_keyword,
                parameter: left_parameter,
            },
            Parameter::Typed {
                keyword: right_keyword,
                parameter: right_parameter,
            },
        ) => {
            left_keyword == right_keyword
                && parameters_equivalent_impl(left_parameter, right_parameter, alias)
        }
        (Parameter::Integer(left), Parameter::Integer(right)) => left == right,
        (Parameter::Real(left), Parameter::Real(right)) => left.to_bits() == right.to_bits(),
        (Parameter::String(left), Parameter::String(right))
        | (Parameter::Enumeration(left), Parameter::Enumeration(right)) => left == right,
        (Parameter::List(left), Parameter::List(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| parameters_equivalent_impl(left, right, alias))
        }
        (Parameter::Ref(Name::Entity(left)), Parameter::Ref(Name::Entity(right))) => {
            canonical_entity_id(alias, *left) == canonical_entity_id(alias, *right)
        }
        (Parameter::Ref(Name::Value(left)), Parameter::Ref(Name::Value(right))) => left == right,
        (
            Parameter::Ref(Name::ConstantEntity(left)),
            Parameter::Ref(Name::ConstantEntity(right)),
        )
        | (Parameter::Ref(Name::ConstantValue(left)), Parameter::Ref(Name::ConstantValue(right))) => {
            left == right
        }
        (Parameter::NotProvided, Parameter::NotProvided)
        | (Parameter::Omitted, Parameter::Omitted) => true,
        _ => false,
    }
}

fn canonical_entity_id(alias: Option<&HashMap<u64, u64>>, id: u64) -> u64 {
    alias.map_or(id, |alias| resolve_alias(alias, id))
}

pub(crate) fn resolve_alias(alias: &HashMap<u64, u64>, mut id: u64) -> u64 {
    for _ in 0..64 {
        let Some(&next) = alias.get(&id) else {
            return id;
        };
        if next == id {
            return id;
        }
        id = next;
    }
    id
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn hash(param: &Parameter) -> u64 {
        let mut hasher = DefaultHasher::new();
        hash_parameter(param, &mut hasher);
        hasher.finish()
    }

    #[test]
    fn index_bucket_allocates_only_after_a_second_candidate() {
        let mut bucket = IndexBucket::one(4);
        assert_eq!(bucket.find(|index| index == 4), Some(4));
        assert_eq!(bucket.find(|index| index == 9), None);
        bucket.push(9);
        assert_eq!(bucket.find(|index| index == 9), Some(9));
    }

    #[test]
    fn structural_hash_tracks_structural_equality() {
        let left = Parameter::List(vec![
            Parameter::Integer(2),
            Parameter::Real(-0.0),
            Parameter::Enumeration("F".to_string()),
        ]);
        let same = left.clone();
        let different = Parameter::List(vec![
            Parameter::Integer(2),
            Parameter::Real(0.0),
            Parameter::Enumeration("F".to_string()),
        ]);

        assert!(parameters_equivalent(&left, &same));
        assert_eq!(hash(&left), hash(&same));
        assert!(!parameters_equivalent(&left, &different));
    }

    #[test]
    fn aliases_are_applied_consistently_to_hash_and_equality() {
        let left = Parameter::Ref(Name::Entity(7));
        let right = Parameter::Ref(Name::Entity(11));
        let alias = HashMap::from([(11, 7)]);

        let mut left_hash = DefaultHasher::new();
        let mut right_hash = DefaultHasher::new();
        hash_parameter_with_alias(&left, &alias, &mut left_hash);
        hash_parameter_with_alias(&right, &alias, &mut right_hash);

        assert!(parameters_equivalent_with_alias(&left, &right, &alias));
        assert_eq!(left_hash.finish(), right_hash.finish());
    }
}
