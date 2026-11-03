use crate::SourceEntity;
use anyhow::Result;
use serde::Serialize;

/// Parser-neutral source-load diagnostics exposed by a CAD-kernel adapter.
///
/// This boundary deliberately owns its vocabulary in Step Redox. Backend-native
/// report types must be translated before they cross it, so Complete IR does not
/// acquire a dependency on any particular kernel's STEP loader API.
pub trait SourceStepLoadInspector {
    /// Inspect how the kernel backend would ingest the original STEP source.
    ///
    /// # Errors
    /// Returns an error when the source cannot be decoded/parsed or when the
    /// backend's diagnostic load path itself fails unexpectedly.
    fn inspect_source_step(&self, input: &[u8]) -> Result<KernelSourceLoadEvidence>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelSourceLoadEvidence {
    pub sections: Vec<KernelSourceSectionLoadEvidence>,
    pub solids: Vec<KernelSourceSolidLoadEvidence>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelSourceSectionLoadEvidence {
    pub data_section: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub entity_losses: Vec<KernelEntityLoadLoss>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelEntityLoadLossKind {
    DeserializeRefused,
    UnexpectedParameterShape,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelEntityLoadLoss {
    pub kind: KernelEntityLoadLossKind,
    pub entity_type: String,
    pub tally: KernelLoadTally,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelSourceSolidLoadEvidence {
    pub source: SourceEntity,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub boundaries: Vec<KernelBoundaryLoadEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topology_provenance: Option<KernelTopologyProvenance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<KernelSourceSolidLoadFailure>,
}

/// Opaque kernel topology identity scoped to one imported solid.
///
/// This is deliberately a distinct namespace from STEP entity IDs. Values are
/// only comparable within the [`KernelSourceSolidLoadEvidence`] that owns them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct KernelTopologyId(u64);

impl KernelTopologyId {
    #[cfg(feature = "cad-kernel-monstertruck")]
    const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelTopologyEntityProvenance {
    pub source: SourceEntity,
    pub topology_id: KernelTopologyId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelFaceTopologyProvenance {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceEntity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shell_member: Option<SourceEntity>,
    pub topology_id: KernelTopologyId,
}

/// Source-to-kernel identity journal for topology retained during STEP import.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelTopologyProvenance {
    pub vertices: Vec<KernelTopologyEntityProvenance>,
    pub edges: Vec<KernelTopologyEntityProvenance>,
    pub faces: Vec<KernelFaceTopologyProvenance>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KernelSourceSolidLoadFailure {
    TableEntryUnavailable,
    ConversionFailed { detail: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelBoundaryRole {
    Outer,
    Void,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelBoundaryLoadEvidence {
    pub boundary_index: usize,
    pub role: KernelBoundaryRole,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub category_counts: Vec<KernelLoadCategoryCount>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub losses: Vec<KernelLoadLoss>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelLoadCategory {
    Face,
    Wire,
    EdgeUse,
    Edge,
    Vertex,
    Representation,
    Solid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelLoadCategoryCount {
    pub category: KernelLoadCategory,
    pub listed: usize,
    pub kept: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelLoadLossReason {
    FaceUnresolved,
    SurfaceUnresolved,
    SurfaceRefused,
    BoundUnresolved,
    DegenerateVertexLoop,
    BoundNotALoopWeImplement,
    EdgeUseUnresolved,
    EdgeUnresolved,
    EdgeGeometryMissing,
    EdgeCurveRefused,
    EdgeEndpointUnresolved,
    VertexUnresolved,
    RelatedRepresentationUnresolved,
    RepresentationItemNotASolid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelLoadLoss {
    pub category: KernelLoadCategory,
    pub reason: KernelLoadLossReason,
    pub tally: KernelLoadTally,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelLoadTally {
    pub count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_source: Option<SourceEntity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_detail: Option<String>,
}

#[cfg(feature = "cad-kernel-monstertruck")]
mod monstertruck_adapter {
    use super::{
        KernelBoundaryLoadEvidence, KernelBoundaryRole, KernelEntityLoadLoss,
        KernelEntityLoadLossKind, KernelFaceTopologyProvenance, KernelLoadCategory,
        KernelLoadCategoryCount, KernelLoadLoss, KernelLoadLossReason, KernelLoadTally,
        KernelSourceLoadEvidence, KernelSourceSectionLoadEvidence, KernelSourceSolidLoadEvidence,
        KernelSourceSolidLoadFailure, KernelTopologyEntityProvenance, KernelTopologyId,
        KernelTopologyProvenance, SourceEntity, SourceStepLoadInspector,
    };
    use crate::cad_kernel::monstertruck::MonstertruckKernel;
    use crate::step_document::StepSection;
    use crate::{step_graph, step_io::ParsedExchange};
    use anyhow::Result;
    use monstertruck_io::step::load::{
        EntityLoadReport, LossCategory, LossReason, LossTally, ShellLoadReport,
        StepTopologyProvenance, Table,
    };

    impl SourceStepLoadInspector for MonstertruckKernel {
        fn inspect_source_step(&self, input: &[u8]) -> Result<KernelSourceLoadEvidence> {
            let exchange = ParsedExchange::parse(input)?.exchange;
            let mut evidence = KernelSourceLoadEvidence {
                sections: Vec::with_capacity(exchange.data.len()),
                solids: Vec::new(),
            };

            for (data_section, section) in exchange.data.iter().enumerate() {
                let index = step_graph::build_index(&section.entities);
                let source = StepSection::new(data_section, &section.entities, &index);
                let table = Table::from_data_section(section);

                evidence.sections.push(KernelSourceSectionLoadEvidence {
                    data_section,
                    entity_losses: entity_losses(&table.entity_report, source),
                });

                for entity in source
                    .iter()
                    .filter(|entity| entity.record_named("MANIFOLD_SOLID_BREP").is_some())
                {
                    let source_entity = entity.source();
                    let Some(solid) = table.manifold_solid_brep.get(&source_entity.id) else {
                        evidence.solids.push(KernelSourceSolidLoadEvidence {
                            source: source_entity,
                            boundaries: Vec::new(),
                            topology_provenance: None,
                            failure: Some(KernelSourceSolidLoadFailure::TableEntryUnavailable),
                        });
                        continue;
                    };

                    match table.to_compressed_trimmed_solid_import(solid) {
                        Ok(import) => evidence.solids.push(KernelSourceSolidLoadEvidence {
                            source: source_entity,
                            boundaries: import
                                .shell_reports
                                .iter()
                                .enumerate()
                                .map(|(boundary_index, report)| KernelBoundaryLoadEvidence {
                                    boundary_index,
                                    role: if boundary_index == 0 {
                                        KernelBoundaryRole::Outer
                                    } else {
                                        KernelBoundaryRole::Void
                                    },
                                    category_counts: category_counts(report),
                                    losses: shell_losses(report, source),
                                })
                                .collect(),
                            topology_provenance: Some(map_topology_provenance(
                                &import.provenance,
                                source,
                            )?),
                            failure: None,
                        }),
                        Err(error) => evidence.solids.push(KernelSourceSolidLoadEvidence {
                            source: source_entity,
                            boundaries: Vec::new(),
                            topology_provenance: None,
                            failure: Some(KernelSourceSolidLoadFailure::ConversionFailed {
                                detail: error.to_string(),
                            }),
                        }),
                    }
                }
            }

            evidence.solids.sort_by_key(|solid| {
                (
                    solid.source.data_section,
                    solid.source.ordinal,
                    solid.source.id,
                )
            });
            Ok(evidence)
        }
    }

    fn map_topology_provenance(
        provenance: &StepTopologyProvenance,
        source: StepSection<'_>,
    ) -> Result<KernelTopologyProvenance> {
        let map_entity = |entry: &monstertruck_io::step::load::StepEntityProvenance| -> Result<KernelTopologyEntityProvenance> {
            let source_id = entry.source_id;
            let stable_id = entry.stable_id;
            let source = source
                .entity(source_id)
                .ok_or_else(|| {
                    anyhow::anyhow!("STEP provenance references missing entity #{source_id}")
                })?
                .source();
            Ok(KernelTopologyEntityProvenance {
                source,
                topology_id: KernelTopologyId::from_raw(stable_id.raw()),
            })
        };

        let vertices = provenance
            .vertices
            .iter()
            .map(&map_entity)
            .collect::<Result<Vec<_>>>()?;
        let edges = provenance
            .edges
            .iter()
            .map(&map_entity)
            .collect::<Result<Vec<_>>>()?;
        let faces = provenance
            .faces
            .iter()
            .map(|entry| {
                let map_optional = |id: Option<u64>| -> Result<Option<SourceEntity>> {
                    id.map(|source_id| {
                        source
                            .entity(source_id)
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "STEP face provenance references missing entity #{source_id}"
                                )
                            })
                            .map(crate::step_document::StepEntity::source)
                    })
                    .transpose()
                };
                Ok(KernelFaceTopologyProvenance {
                    source: map_optional(entry.source.face_surface_id)?,
                    shell_member: map_optional(entry.source.shell_member_id)?,
                    topology_id: KernelTopologyId::from_raw(entry.stable_id.raw()),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(KernelTopologyProvenance {
            vertices,
            edges,
            faces,
        })
    }

    fn entity_losses(
        report: &EntityLoadReport,
        source: StepSection<'_>,
    ) -> Vec<KernelEntityLoadLoss> {
        report
            .refused()
            .map(|(entity_type, tally)| KernelEntityLoadLoss {
                kind: KernelEntityLoadLossKind::DeserializeRefused,
                entity_type: entity_type.to_owned(),
                tally: map_tally(tally, source),
            })
            .chain(
                report
                    .shape_unexpected()
                    .map(|(entity_type, tally)| KernelEntityLoadLoss {
                        kind: KernelEntityLoadLossKind::UnexpectedParameterShape,
                        entity_type: entity_type.to_owned(),
                        tally: map_tally(tally, source),
                    }),
            )
            .collect()
    }

    fn category_counts(report: &ShellLoadReport) -> Vec<KernelLoadCategoryCount> {
        LossCategory::ALL
            .into_iter()
            .filter_map(|category| {
                let count = report.count(category);
                (count.listed != 0 || count.kept != 0).then_some(KernelLoadCategoryCount {
                    category: map_category(category),
                    listed: count.listed,
                    kept: count.kept,
                })
            })
            .collect()
    }

    fn shell_losses(report: &ShellLoadReport, source: StepSection<'_>) -> Vec<KernelLoadLoss> {
        report
            .losses()
            .map(|(category, reason, tally)| KernelLoadLoss {
                category: map_category(category),
                reason: map_reason(reason),
                tally: map_tally(tally, source),
            })
            .collect()
    }

    fn map_tally(tally: &LossTally, source: StepSection<'_>) -> KernelLoadTally {
        KernelLoadTally {
            count: tally.count,
            first_source: tally
                .first_id
                .and_then(|id| source.entity(id))
                .map(crate::step_document::StepEntity::source),
            first_detail: tally.first_detail.clone(),
        }
    }

    const fn map_category(category: LossCategory) -> KernelLoadCategory {
        match category {
            LossCategory::Face => KernelLoadCategory::Face,
            LossCategory::Wire => KernelLoadCategory::Wire,
            LossCategory::EdgeUse => KernelLoadCategory::EdgeUse,
            LossCategory::Edge => KernelLoadCategory::Edge,
            LossCategory::Vertex => KernelLoadCategory::Vertex,
            LossCategory::Representation => KernelLoadCategory::Representation,
            LossCategory::Solid => KernelLoadCategory::Solid,
        }
    }

    const fn map_reason(reason: LossReason) -> KernelLoadLossReason {
        match reason {
            LossReason::FaceUnresolved => KernelLoadLossReason::FaceUnresolved,
            LossReason::SurfaceUnresolved => KernelLoadLossReason::SurfaceUnresolved,
            LossReason::SurfaceRefused => KernelLoadLossReason::SurfaceRefused,
            LossReason::BoundUnresolved => KernelLoadLossReason::BoundUnresolved,
            LossReason::DegenerateVertexLoop => KernelLoadLossReason::DegenerateVertexLoop,
            LossReason::BoundNotALoopWeImplement => KernelLoadLossReason::BoundNotALoopWeImplement,
            LossReason::EdgeUseUnresolved => KernelLoadLossReason::EdgeUseUnresolved,
            LossReason::EdgeUnresolved => KernelLoadLossReason::EdgeUnresolved,
            LossReason::EdgeGeometryMissing => KernelLoadLossReason::EdgeGeometryMissing,
            LossReason::EdgeCurveRefused => KernelLoadLossReason::EdgeCurveRefused,
            LossReason::EdgeEndpointUnresolved => KernelLoadLossReason::EdgeEndpointUnresolved,
            LossReason::VertexUnresolved => KernelLoadLossReason::VertexUnresolved,
            LossReason::RelatedRepresentationUnresolved => {
                KernelLoadLossReason::RelatedRepresentationUnresolved
            }
            LossReason::RepresentationItemNotASolid => {
                KernelLoadLossReason::RepresentationItemNotASolid
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn maps_table_swallow_to_parser_neutral_source() -> Result<()> {
            let input = b"ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('x'),'1');\nFILE_NAME('a','b',(''),(''),'x','y','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n#11=ORIENTED_EDGE('bad');\nENDSEC;\nEND-ISO-10303-21;\n";
            let evidence = MonstertruckKernel::project_standard().inspect_source_step(input)?;
            assert_eq!(evidence.sections.len(), 1);
            assert_eq!(evidence.sections[0].entity_losses.len(), 1);
            let loss = &evidence.sections[0].entity_losses[0];
            assert_eq!(
                loss.kind,
                KernelEntityLoadLossKind::UnexpectedParameterShape
            );
            assert_eq!(loss.entity_type, "ORIENTED_EDGE");
            assert_eq!(loss.tally.count, 1);
            assert_eq!(loss.tally.first_source.map(|source| source.id), Some(11));
            assert_eq!(
                loss.tally.first_source.map(|source| source.data_section),
                Some(0)
            );
            Ok(())
        }

        #[test]
        fn distinguishes_unretained_source_solid_from_conversion_failure() -> Result<()> {
            let unretained = b"ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('x'),'1');\nFILE_NAME('a','b',(''),(''),'x','y','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n#11=MANIFOLD_SOLID_BREP('bad');\nENDSEC;\nEND-ISO-10303-21;\n";
            let evidence =
                MonstertruckKernel::project_standard().inspect_source_step(unretained)?;
            assert_eq!(evidence.solids.len(), 1);
            assert_eq!(
                evidence.solids[0].failure,
                Some(KernelSourceSolidLoadFailure::TableEntryUnavailable)
            );
            assert!(evidence.solids[0].topology_provenance.is_none());

            let conversion_failure = b"ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('x'),'1');\nFILE_NAME('a','b',(''),(''),'x','y','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n#11=MANIFOLD_SOLID_BREP('bad',#12);\nENDSEC;\nEND-ISO-10303-21;\n";
            let evidence =
                MonstertruckKernel::project_standard().inspect_source_step(conversion_failure)?;
            assert_eq!(evidence.solids.len(), 1);
            let Some(KernelSourceSolidLoadFailure::ConversionFailed { detail }) =
                evidence.solids[0].failure.as_ref()
            else {
                anyhow::bail!("source solid conversion failure was not typed separately");
            };
            assert_ne!(detail, "");
            assert!(evidence.solids[0].topology_provenance.is_none());
            Ok(())
        }

        #[test]
        fn reports_source_solid_boundaries_positionally() -> Result<()> {
            let input = include_bytes!("../validation/fixtures/production_radial_slot.step");
            let evidence = MonstertruckKernel::project_standard().inspect_source_step(input)?;
            assert_ne!(evidence.solids.len(), 0);
            for solid in &evidence.solids {
                assert!(solid.failure.is_none());
                assert_ne!(solid.boundaries.len(), 0);
                assert_eq!(solid.boundaries[0].boundary_index, 0);
                assert_eq!(solid.boundaries[0].role, KernelBoundaryRole::Outer);
                let provenance = solid
                    .topology_provenance
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("successful source solid omitted provenance"))?;
                assert_ne!(provenance.vertices.len(), 0);
                assert_ne!(provenance.edges.len(), 0);
                assert_ne!(provenance.faces.len(), 0);
                assert!(provenance.vertices.iter().all(|entry| {
                    entry.source.data_section == solid.source.data_section
                        && entry.topology_id.raw() != 0
                }));
                assert!(provenance.edges.iter().all(|entry| {
                    entry.source.data_section == solid.source.data_section
                        && entry.topology_id.raw() != 0
                }));
                assert!(provenance.faces.iter().all(|entry| {
                    entry.topology_id.raw() != 0
                        && entry
                            .source
                            .is_none_or(|source| source.data_section == solid.source.data_section)
                        && entry
                            .shell_member
                            .is_none_or(|source| source.data_section == solid.source.data_section)
                }));
            }
            Ok(())
        }
    }
}
