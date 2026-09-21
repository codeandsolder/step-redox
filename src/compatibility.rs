use ruststep::ast::EntityInstance;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Default)]
pub struct CompatibilityAudit {
    /// Constructs intentionally avoided by the broad-compatibility profile.
    pub structural_risk_entities: BTreeMap<String, usize>,
    pub structural_risk_total: usize,
    pub has_mapped_items: bool,
    pub has_curve_or_surface_replicas: bool,
    pub has_assembly_relationships: bool,
    /// True when none of the currently-audited structural compatibility risks
    /// remain. This is intentionally narrower than claiming support in every
    /// CAD importer.
    pub conservative_structure: bool,
}

const RISK_TYPES: &[&str] = &[
    "MAPPED_ITEM",
    "REPRESENTATION_MAP",
    "CURVE_REPLICA",
    "SURFACE_REPLICA",
    "NEXT_ASSEMBLY_USAGE_OCCURRENCE",
    "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION",
    "ITEM_DEFINED_TRANSFORMATION",
    "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION",
    "SHAPE_REPRESENTATION_RELATIONSHIP",
];

pub fn audit_entities(entities: &[EntityInstance]) -> CompatibilityAudit {
    let mut counts = BTreeMap::new();
    for entity in entities {
        match entity {
            EntityInstance::Simple { record, .. } => {
                if RISK_TYPES.contains(&record.name.as_str()) {
                    *counts.entry(record.name.clone()).or_insert(0) += 1;
                }
            }
            EntityInstance::Complex { subsuper, .. } => {
                for record in &subsuper.0 {
                    if RISK_TYPES.contains(&record.name.as_str()) {
                        *counts.entry(record.name.clone()).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    let get = |name: &str| counts.get(name).copied().unwrap_or(0);
    let structural_risk_total = counts.values().sum();
    let has_mapped_items = get("MAPPED_ITEM") > 0 || get("REPRESENTATION_MAP") > 0;
    let has_curve_or_surface_replicas =
        get("CURVE_REPLICA") > 0 || get("SURFACE_REPLICA") > 0;
    let has_assembly_relationships = [
        "NEXT_ASSEMBLY_USAGE_OCCURRENCE",
        "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION",
        "ITEM_DEFINED_TRANSFORMATION",
        "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION",
        "SHAPE_REPRESENTATION_RELATIONSHIP",
    ]
    .iter()
    .any(|name| get(name) > 0);

    CompatibilityAudit {
        structural_risk_entities: counts,
        structural_risk_total,
        has_mapped_items,
        has_curve_or_surface_replicas,
        has_assembly_relationships,
        conservative_structure: structural_risk_total == 0,
    }
}
