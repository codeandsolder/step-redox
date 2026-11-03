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
    #[cfg(any(test, feature = "cad-kernel-monstertruck"))]
    pub(crate) const fn from_raw(raw: u64) -> Self {
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

impl KernelTopologyProvenance {
    fn unique_id(ids: impl Iterator<Item = KernelTopologyId>) -> Option<KernelTopologyId> {
        let mut ids = ids.collect::<std::collections::BTreeSet<_>>().into_iter();
        let first = ids.next()?;
        ids.next().is_none().then_some(first)
    }

    pub(crate) fn vertex_for_source_id(
        &self,
        data_section: usize,
        source_id: u64,
    ) -> Option<KernelTopologyId> {
        Self::unique_id(
            self.vertices
                .iter()
                .filter(|entry| {
                    entry.source.data_section == data_section && entry.source.id == source_id
                })
                .map(|entry| entry.topology_id),
        )
    }

    pub(crate) fn edge_for_source_id(
        &self,
        data_section: usize,
        source_id: u64,
    ) -> Option<KernelTopologyId> {
        Self::unique_id(
            self.edges
                .iter()
                .filter(|entry| {
                    entry.source.data_section == data_section && entry.source.id == source_id
                })
                .map(|entry| entry.topology_id),
        )
    }

    pub(crate) fn face_for_source_id(
        &self,
        data_section: usize,
        source_id: u64,
    ) -> Option<KernelTopologyId> {
        let shell_member = Self::unique_id(
            self.faces
                .iter()
                .filter(|entry| {
                    entry.shell_member.is_some_and(|source| {
                        source.data_section == data_section && source.id == source_id
                    })
                })
                .map(|entry| entry.topology_id),
        );
        shell_member.or_else(|| {
            Self::unique_id(
                self.faces
                    .iter()
                    .filter(|entry| {
                        entry.source.is_some_and(|source| {
                            source.data_section == data_section && source.id == source_id
                        })
                    })
                    .map(|entry| entry.topology_id),
            )
        })
    }
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
    use crate::kernel_topology::{
        KernelFaceBoundaryReplacement, KernelFaceSplice, KernelPatchCurve, KernelPatchEdgeRef,
        KernelPatchVertexRef, KernelTopologyPatch, SourceTopologyEditor,
    };
    use crate::step_document::StepSection;
    use crate::{step_graph, step_io::ParsedExchange};
    use anyhow::Result;
    use monstertruck_io::step::load::step_geometry::{
        Conic3D, Curve3D, Line, Matrix4, Point3, Processor, TrimmedCurve, UnitCircle, Vector4,
    };
    use monstertruck_io::step::load::{
        EntityLoadReport, LossCategory, LossReason, LossTally, ShellLoadReport,
        StepTopologyProvenance, Table,
    };
    use monstertruck_io::step::save::{self, CompleteStepDisplay};
    use monstertruck_topology::StableId;
    use monstertruck_topology::compressed_surgery::{
        CompressedFaceBoundaryReplacement, CompressedFaceSplice, CompressedTrimmedTopologyPatch,
        NewCompressedEdge, PatchEdgeRef, PatchEdgeUse, PatchVertexRef, RetainedCompressedEdge,
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

    impl SourceTopologyEditor for MonstertruckKernel {
        fn edit_source_topology(
            &self,
            input: &[u8],
            source: SourceEntity,
            patches: &[KernelTopologyPatch],
        ) -> Result<String> {
            let exchange = ParsedExchange::parse(input)?.exchange;
            let section = exchange.data.get(source.data_section).ok_or_else(|| {
                anyhow::anyhow!(
                    "source solid references missing DATA section {}",
                    source.data_section
                )
            })?;
            let index = step_graph::build_index(&section.entities);
            let section_source = StepSection::new(source.data_section, &section.entities, &index);
            let actual_source = section_source
                .entity(source.id)
                .ok_or_else(|| anyhow::anyhow!("source solid #{} is missing", source.id))?
                .source();
            if actual_source != source {
                anyhow::bail!(
                    "source solid identity changed: requested {source:?}, found {actual_source:?}"
                );
            }

            let table = Table::from_data_section(section);
            let solid = table.manifold_solid_brep.get(&source.id).ok_or_else(|| {
                anyhow::anyhow!("source solid #{} is unavailable to Monstertruck", source.id)
            })?;
            let mut imported = table.to_compressed_trimmed_solid_import(solid)?;
            for patch in patches {
                imported
                    .solid
                    .apply_topology_patch(&translate_patch(patch))?;
            }

            Ok(CompleteStepDisplay::new(
                save::StepModel::from(&imported.solid),
                save::StepHeaderDescriptor {
                    organization_system: "step-redox Monstertruck source-topology surgery".into(),
                    ..Default::default()
                },
            )
            .to_string())
        }
    }

    fn translate_patch(
        patch: &KernelTopologyPatch,
    ) -> CompressedTrimmedTopologyPatch<Point3, Curve3D> {
        CompressedTrimmedTopologyPatch {
            remove_faces: patch
                .remove_faces
                .iter()
                .map(|id| StableId::new(id.raw()))
                .collect(),
            new_vertices: patch.new_vertices.iter().copied().map(point3).collect(),
            new_edges: patch
                .new_edges
                .iter()
                .map(|edge| NewCompressedEdge {
                    vertices: (
                        translate_vertex_ref(edge.vertices.0),
                        translate_vertex_ref(edge.vertices.1),
                    ),
                    curve: translate_curve(&edge.curve),
                })
                .collect(),
            rewrite_edges: patch
                .rewrite_edges
                .iter()
                .map(|edge| RetainedCompressedEdge {
                    edge: StableId::new(edge.edge.raw()),
                    vertices: (
                        translate_vertex_ref(edge.vertices.0),
                        translate_vertex_ref(edge.vertices.1),
                    ),
                    curve: translate_curve(&edge.curve),
                })
                .collect(),
            face_splices: patch
                .face_splices
                .iter()
                .map(translate_face_splice)
                .collect(),
            face_boundary_replacements: patch
                .face_boundary_replacements
                .iter()
                .map(translate_face_boundary_replacement)
                .collect(),
            require_unused_edges: patch
                .require_unused_edges
                .iter()
                .map(|id| StableId::new(id.raw()))
                .collect(),
        }
    }

    const fn translate_vertex_ref(reference: KernelPatchVertexRef) -> PatchVertexRef {
        match reference {
            KernelPatchVertexRef::Existing(id) => PatchVertexRef::Existing(StableId::new(id.raw())),
            KernelPatchVertexRef::New(index) => PatchVertexRef::New(index),
        }
    }

    const fn translate_edge_ref(reference: KernelPatchEdgeRef) -> PatchEdgeRef {
        match reference {
            KernelPatchEdgeRef::Existing(id) => PatchEdgeRef::Existing(StableId::new(id.raw())),
            KernelPatchEdgeRef::New(index) => PatchEdgeRef::New(index),
        }
    }

    fn translate_face_boundary_replacement(
        replacement: &KernelFaceBoundaryReplacement,
    ) -> CompressedFaceBoundaryReplacement {
        CompressedFaceBoundaryReplacement {
            face: StableId::new(replacement.face.raw()),
            boundaries: replacement
                .boundaries
                .iter()
                .map(|wire| {
                    wire.iter()
                        .map(|edge_use| PatchEdgeUse {
                            edge: translate_edge_ref(edge_use.edge),
                            orientation: edge_use.orientation,
                        })
                        .collect()
                })
                .collect(),
        }
    }

    fn translate_face_splice(splice: &KernelFaceSplice) -> CompressedFaceSplice {
        CompressedFaceSplice {
            face: StableId::new(splice.face.raw()),
            remove_edges: splice
                .remove_edges
                .iter()
                .map(|id| StableId::new(id.raw()))
                .collect(),
            replacement: splice
                .replacement
                .iter()
                .map(|edge_use| PatchEdgeUse {
                    edge: translate_edge_ref(edge_use.edge),
                    orientation: edge_use.orientation,
                })
                .collect(),
        }
    }

    fn translate_curve(curve: &KernelPatchCurve) -> Curve3D {
        match *curve {
            KernelPatchCurve::Line { start_mm, end_mm } => {
                Curve3D::Line(Line(point3(start_mm), point3(end_mm)))
            }
            KernelPatchCurve::CircleArc {
                center_mm,
                normal,
                x_direction,
                radius_mm,
                start_angle_rad,
                end_angle_rad,
            } => {
                let y_direction = [
                    normal[1].mul_add(x_direction[2], -normal[2] * x_direction[1]),
                    normal[2].mul_add(x_direction[0], -normal[0] * x_direction[2]),
                    normal[0].mul_add(x_direction[1], -normal[1] * x_direction[0]),
                ];
                let transform = Matrix4::from_cols(
                    Vector4::new(
                        x_direction[0] * radius_mm,
                        x_direction[1] * radius_mm,
                        x_direction[2] * radius_mm,
                        0.0,
                    ),
                    Vector4::new(
                        y_direction[0] * radius_mm,
                        y_direction[1] * radius_mm,
                        y_direction[2] * radius_mm,
                        0.0,
                    ),
                    Vector4::new(normal[0], normal[1], normal[2], 0.0),
                    Vector4::new(center_mm[0], center_mm[1], center_mm[2], 1.0),
                );
                Curve3D::Conic(Conic3D::Ellipse(Processor::with_transform(
                    TrimmedCurve::new(
                        UnitCircle::<Point3>::new(),
                        (start_angle_rad, end_angle_rad),
                    ),
                    transform,
                )))
            }
        }
    }

    const fn point3(point: [f64; 3]) -> Point3 {
        Point3::new(point[0], point[1], point[2])
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
        fn topology_editor_roundtrips_exact_selected_source_solid() -> Result<()> {
            let input = include_bytes!("../validation/fixtures/production_radial_slot.step");
            let kernel = MonstertruckKernel::project_standard();
            let evidence = kernel.inspect_source_step(input)?;
            let source = evidence
                .solids
                .first()
                .ok_or_else(|| anyhow::anyhow!("fixture has no source solid"))?
                .source;
            let edited = kernel.edit_source_topology(input, source, &[])?;
            let exchange = ParsedExchange::parse(edited.as_bytes())?.exchange;
            assert_eq!(exchange.data.len(), 1);
            let section = exchange
                .data
                .first()
                .ok_or_else(|| anyhow::anyhow!("edited STEP omitted DATA section"))?;
            let table = Table::from_data_section(section);
            assert_ne!(table.manifold_solid_brep.len(), 0);
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
