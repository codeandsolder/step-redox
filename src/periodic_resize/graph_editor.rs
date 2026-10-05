use super::{entity_ref, entity_ref_list, list_params, numeric_value};
use crate::instances::StyleRef;
use crate::math3::{add, cross, dot, norm, scale};
use crate::step_graph::{
    ReferenceGraph, build_index, entity_id, entity_ref_value, simple_record, simple_record_mut,
    visit_entity_refs,
};
use anyhow::{Result, anyhow, bail};
use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy)]
pub(super) struct Traversal {
    edge: u64,
    start: u64,
    end: u64,
}

#[derive(Debug, Clone)]
pub(super) struct SourceLoop {
    outer: bool,
    traversal: Vec<Traversal>,
}

pub(super) struct GraphEditor<'a> {
    entities: &'a mut Vec<EntityInstance>,
    index: HashMap<u64, usize>,
    next_id: u64,
    initial_len: usize,
}

impl<'a> GraphEditor<'a> {
    pub(super) fn new(entities: &'a mut Vec<EntityInstance>) -> Self {
        let index = build_index(entities);
        let next_id = entities.iter().map(entity_id).max().unwrap_or(0) + 1;
        let initial_len = entities.len();
        Self {
            entities,
            index,
            next_id,
            initial_len,
        }
    }

    pub(super) fn added_entities(&self) -> usize {
        self.entities.len().saturating_sub(self.initial_len)
    }

    pub(super) fn entity_delta(&self) -> isize {
        self.entities.len() as isize - self.initial_len as isize
    }

    pub(super) fn entity_type(&self, id: u64) -> Option<&str> {
        let idx = *self.index.get(&id)?;
        match &self.entities[idx] {
            EntityInstance::Simple { record, .. } => Some(record.name.as_str()),
            EntityInstance::Complex { .. } => Some("COMPLEX"),
        }
    }

    pub(super) fn entity(&self, id: u64) -> Option<&EntityInstance> {
        self.index.get(&id).map(|&idx| &self.entities[idx])
    }

    pub(super) fn simple_record(&self, id: u64) -> Option<&Record> {
        let idx = *self.index.get(&id)?;
        simple_record(&self.entities[idx])
    }

    pub(super) fn simple_record_mut(&mut self, id: u64) -> Option<&mut Record> {
        let idx = *self.index.get(&id)?;
        simple_record_mut(&mut self.entities[idx])
    }

    pub(super) fn push_simple(&mut self, name: &str, params: Vec<Parameter>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let idx = self.entities.len();
        self.entities.push(EntityInstance::Simple {
            id,
            record: Record {
                name: name.to_string(),
                parameter: Parameter::List(params),
            },
        });
        self.index.insert(id, idx);
        id
    }

    pub(super) fn shell_faces(&self, shell: u64) -> Result<Vec<u64>> {
        let record = self
            .simple_record(shell)
            .ok_or_else(|| anyhow!("missing shell #{shell}"))?;
        if record.name != "CLOSED_SHELL" {
            bail!("#{shell} is not CLOSED_SHELL");
        }
        let params =
            list_params(record).ok_or_else(|| anyhow!("shell parameters are not a list"))?;
        params
            .get(1)
            .and_then(entity_ref_list)
            .ok_or_else(|| anyhow!("shell #{shell} has no face aggregate"))
    }

    pub(super) fn set_shell_faces(&mut self, shell: u64, faces: &[u64]) -> Result<()> {
        let record = self
            .simple_record_mut(shell)
            .ok_or_else(|| anyhow!("missing shell #{shell}"))?;
        let Parameter::List(params) = &mut record.parameter else {
            bail!("shell #{shell} parameters are not a list");
        };
        if params.len() < 2 {
            bail!("shell #{shell} has incomplete parameters");
        }
        params[1] = Parameter::List(faces.iter().copied().map(entity_ref).collect());
        Ok(())
    }

    pub(super) fn descendant_closure(&self, seeds: &HashSet<u64>) -> Result<HashSet<u64>> {
        let mut descendants = HashSet::with_capacity(seeds.len());
        let mut stack = seeds.iter().copied().collect::<Vec<_>>();
        while let Some(id) = stack.pop() {
            if !descendants.insert(id) {
                continue;
            }
            let idx = *self
                .index
                .get(&id)
                .ok_or_else(|| anyhow!("descendant graph references missing entity #{id}"))?;
            visit_entity_refs(&self.entities[idx], &mut |child| stack.push(child));
        }
        Ok(descendants)
    }

    /// Remove only descendants of explicitly replaced graph roots that have
    /// become unreachable after rewiring. Shared supports are preserved
    /// automatically because any inbound reference from outside the deletion
    /// set blocks collection.
    pub(super) fn prune_unreachable_descendants(&mut self, seeds: &HashSet<u64>) -> Result<usize> {
        let references = ReferenceGraph::new(self.entities);
        let refs = references.forward();
        let inbound = references.inbound();
        let mut candidate = HashSet::<u64>::new();
        let mut stack = seeds.iter().copied().collect::<Vec<_>>();
        while let Some(id) = stack.pop() {
            if !candidate.insert(id) {
                continue;
            }
            let Some(children) = refs.get(&id) else {
                bail!("prune seed graph references missing entity #{id}");
            };
            stack.extend(children.iter().copied());
        }

        let mut delete = HashSet::<u64>::new();
        loop {
            let mut changed = false;
            for &id in &candidate {
                if delete.contains(&id) {
                    continue;
                }
                let all_dead = inbound
                    .get(&id)
                    .is_none_or(|parents| parents.iter().all(|parent| delete.contains(parent)));
                if !all_dead {
                    continue;
                }

                let seed = seeds.contains(&id);
                let child_of_dead = inbound.get(&id).is_some_and(|parents| {
                    !parents.is_empty() && parents.iter().all(|parent| delete.contains(parent))
                });
                if seed || child_of_dead {
                    delete.insert(id);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let removed = delete.len();
        self.entities
            .retain(|entity| !delete.contains(&entity_id(entity)));
        self.index = build_index(self.entities);
        Ok(removed)
    }

    pub(super) fn presentation_members_mut(&mut self, parent: u64) -> Result<&mut Vec<Parameter>> {
        let record = self
            .simple_record_mut(parent)
            .ok_or_else(|| anyhow!("missing presentation container #{parent}"))?;
        let index = match record.name.as_str() {
            "PRESENTATION_LAYER_ASSIGNMENT" => 2,
            "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION" => 1,
            other => bail!("unsupported presentation container {other} #{parent}"),
        };
        let Parameter::List(params) = &mut record.parameter else {
            bail!("presentation container #{parent} params invalid");
        };
        let Some(Parameter::List(members)) = params.get_mut(index) else {
            bail!("presentation container #{parent} member aggregate invalid");
        };
        Ok(members)
    }

    pub(super) fn clone_face_styles(
        &mut self,
        styles_by_target: &HashMap<u64, Vec<StyleRef>>,
        style_parents: &HashMap<u64, Vec<u64>>,
        mapping: &HashMap<u64, u64>,
    ) -> Result<usize> {
        let mut targets = mapping
            .iter()
            .filter_map(|(&old, &new)| styles_by_target.contains_key(&old).then_some((old, new)))
            .collect::<Vec<_>>();
        targets.sort_unstable();

        let mut cloned = 0usize;
        for (old_target, new_target) in targets {
            let mut styles = styles_by_target[&old_target].clone();
            styles.sort_by_key(|style| style.id);
            for style in styles {
                let old_record = self
                    .simple_record(style.id)
                    .ok_or_else(|| anyhow!("missing STYLED_ITEM #{}", style.id))?
                    .clone();
                if old_record.name != "STYLED_ITEM" {
                    bail!("#{} is not STYLED_ITEM", style.id);
                }
                let Parameter::List(mut params) = old_record.parameter else {
                    bail!("STYLED_ITEM #{} params invalid", style.id);
                };
                if params.len() != 3 {
                    bail!("STYLED_ITEM #{} arity invalid", style.id);
                }
                params[2] = entity_ref(new_target);
                let new_style = self.push_simple("STYLED_ITEM", params);
                let parents = style_parents.get(&style.id).ok_or_else(|| {
                    anyhow!("missing presentation parents for STYLED_ITEM #{}", style.id)
                })?;
                for &parent in parents {
                    self.presentation_members_mut(parent)?
                        .push(entity_ref(new_style));
                }
                cloned += 1;
            }
        }
        Ok(cloned)
    }

    pub(super) fn remove_face_styles(
        &mut self,
        styles_by_target: &HashMap<u64, Vec<StyleRef>>,
        style_parents: &HashMap<u64, Vec<u64>>,
        targets: &HashSet<u64>,
    ) -> Result<usize> {
        let mut styles = targets
            .iter()
            .flat_map(|target| styles_by_target.get(target).into_iter().flatten())
            .cloned()
            .collect::<Vec<_>>();
        styles.sort_by_key(|style| style.id);
        styles.dedup_by_key(|style| style.id);

        let mut delete = HashSet::<u64>::new();
        for style in &styles {
            let parents = style_parents.get(&style.id).ok_or_else(|| {
                anyhow!("missing presentation parents for STYLED_ITEM #{}", style.id)
            })?;
            for &parent in parents {
                let members = self.presentation_members_mut(parent)?;
                let before = members.len();
                members.retain(|item| entity_ref_value(item) != Some(style.id));
                if members.len() == before {
                    bail!(
                        "presentation container #{parent} did not reference STYLED_ITEM #{}",
                        style.id
                    );
                }
            }
            delete.insert(style.id);
        }

        self.entities
            .retain(|entity| !delete.contains(&entity_id(entity)));
        self.index = build_index(self.entities);
        Ok(delete.len())
    }

    pub(super) fn relative_replica_transform_origins(
        &self,
        ids: &[u64],
        old_index: &HashMap<u64, usize>,
    ) -> Result<HashSet<u64>> {
        let cloned_ids = ids.iter().copied().collect::<HashSet<_>>();
        let mut origins = HashSet::<u64>::new();
        for &old in ids {
            let idx = *old_index
                .get(&old)
                .ok_or_else(|| anyhow!("clone graph missing entity #{old}"))?;
            let Some(record) = simple_record(&self.entities[idx]) else {
                continue;
            };
            if !matches!(record.name.as_str(), "CURVE_REPLICA" | "SURFACE_REPLICA") {
                continue;
            }
            let params = list_params(record)
                .ok_or_else(|| anyhow!("{} #{old} parameters are not a list", record.name))?;
            let transform = entity_ref_value(
                params
                    .get(2)
                    .ok_or_else(|| anyhow!("{} #{old} missing transformation", record.name))?,
            )
            .ok_or_else(|| anyhow!("{} #{old} transformation is not a reference", record.name))?;
            if !cloned_ids.contains(&transform) {
                bail!(
                    "{} #{old} cloned without its transformation #{transform}",
                    record.name
                );
            }
            let transform_index = *old_index
                .get(&transform)
                .ok_or_else(|| anyhow!("missing replica transformation #{transform}"))?;
            let transform_record = simple_record(&self.entities[transform_index])
                .ok_or_else(|| anyhow!("replica transformation #{transform} is complex"))?;
            if transform_record.name != "CARTESIAN_TRANSFORMATION_OPERATOR_3D" {
                bail!(
                    "replica transformation #{transform} uses unsupported {}",
                    transform_record.name
                );
            }
            let transform_params = list_params(transform_record)
                .ok_or_else(|| anyhow!("replica transformation #{transform} params invalid"))?;
            let origin = entity_ref_value(transform_params.get(5).ok_or_else(|| {
                anyhow!("replica transformation #{transform} missing local origin")
            })?)
            .ok_or_else(|| {
                anyhow!("replica transformation #{transform} local origin is not a ref")
            })?;
            if !cloned_ids.contains(&origin) {
                bail!("replica transformation #{transform} cloned without origin #{origin}");
            }
            origins.insert(origin);
        }
        self.validate_relative_replica_origins(&origins, &cloned_ids, old_index)?;
        Ok(origins)
    }

    pub(super) fn validate_relative_replica_origins(
        &self,
        origins: &HashSet<u64>,
        cloned_ids: &HashSet<u64>,
        old_index: &HashMap<u64, usize>,
    ) -> Result<()> {
        if origins.is_empty() {
            return Ok(());
        }

        // Only cloned parents can make a relative replica origin unsafe.  The
        // old implementation built forward + inbound maps for the entire STEP
        // section here, even though it immediately discarded every parent
        // outside `cloned_ids`.  Stay inside the closure we are about to clone.
        for &parent in cloned_ids {
            let parent_index = *old_index
                .get(&parent)
                .ok_or_else(|| anyhow!("missing cloned parent #{parent}"))?;
            let mut referenced_origin = None;
            visit_entity_refs(&self.entities[parent_index], &mut |child| {
                if referenced_origin.is_none() && origins.contains(&child) {
                    referenced_origin = Some(child);
                }
            });
            let Some(origin) = referenced_origin else {
                continue;
            };
            let Some(parent_record) = simple_record(&self.entities[parent_index]) else {
                bail!(
                    "replica transform origin #{origin} is shared with complex cloned parent #{parent}"
                );
            };
            if parent_record.name != "CARTESIAN_TRANSFORMATION_OPERATOR_3D" {
                bail!(
                    "replica transform origin #{origin} is also used by cloned {} #{parent}",
                    parent_record.name
                );
            }
        }
        Ok(())
    }

    pub(super) fn clone_descendants(
        &mut self,
        seeds: &HashSet<u64>,
        delta: [f64; 3],
    ) -> Result<HashMap<u64, u64>> {
        let mut ids = self
            .descendant_closure(seeds)?
            .into_iter()
            .collect::<Vec<_>>();
        ids.sort_unstable();
        let mut mapping = HashMap::with_capacity(ids.len());
        for &old in &ids {
            let new = self.next_id;
            self.next_id += 1;
            mapping.insert(old, new);
        }

        // No graph mutation happens until all clone payloads are materialized,
        // so the live index is a valid immutable snapshot.  Avoid cloning the
        // whole section-sized HashMap for every repeated unit.
        let relative_transform_origins =
            self.relative_replica_transform_origins(&ids, &self.index)?;

        let mut clones = Vec::with_capacity(ids.len());
        for &old in &ids {
            let idx = *self
                .index
                .get(&old)
                .ok_or_else(|| anyhow!("clone graph missing entity #{old}"))?;
            let mut entity = self.entities[idx].clone();
            set_entity_id(&mut entity, mapping[&old]);
            remap_entity_refs(&mut entity, &mapping);
            if !relative_transform_origins.contains(&old) {
                translate_cartesian_point(&mut entity, delta)?;
            }
            clones.push(entity);
        }

        for entity in clones {
            let id = entity_id(&entity);
            let idx = self.entities.len();
            self.entities.push(entity);
            self.index.insert(id, idx);
        }
        Ok(mapping)
    }

    pub(super) fn face_edges_ordered(&self, face: u64) -> Result<Vec<u64>> {
        let record = self
            .simple_record(face)
            .ok_or_else(|| anyhow!("missing face #{face}"))?;
        if record.name != "ADVANCED_FACE" {
            bail!("#{face} is not ADVANCED_FACE");
        }
        let params = list_params(record).ok_or_else(|| anyhow!("face params are not a list"))?;
        let bounds = params
            .get(1)
            .and_then(entity_ref_list)
            .ok_or_else(|| anyhow!("face #{face} has no bounds"))?;
        let mut out = Vec::new();
        for bound in bounds {
            let bound_record = self
                .simple_record(bound)
                .ok_or_else(|| anyhow!("missing face bound #{bound}"))?;
            if bound_record.name != "FACE_BOUND" && bound_record.name != "FACE_OUTER_BOUND" {
                bail!("#{bound} is not a face bound");
            }
            let bound_params =
                list_params(bound_record).ok_or_else(|| anyhow!("bound params are not a list"))?;
            let loop_id = entity_ref_value(
                bound_params
                    .get(1)
                    .ok_or_else(|| anyhow!("bound #{bound} missing loop"))?,
            )
            .ok_or_else(|| anyhow!("bound #{bound} loop is not a reference"))?;
            let loop_record = self
                .simple_record(loop_id)
                .ok_or_else(|| anyhow!("missing edge loop #{loop_id}"))?;
            if loop_record.name != "EDGE_LOOP" {
                bail!("#{loop_id} is not EDGE_LOOP");
            }
            let loop_params =
                list_params(loop_record).ok_or_else(|| anyhow!("loop params are not a list"))?;
            let oriented = loop_params
                .get(1)
                .and_then(entity_ref_list)
                .ok_or_else(|| anyhow!("loop #{loop_id} has no edge aggregate"))?;
            for oe in oriented {
                let oe_record = self
                    .simple_record(oe)
                    .ok_or_else(|| anyhow!("missing oriented edge #{oe}"))?;
                if oe_record.name != "ORIENTED_EDGE" {
                    bail!("#{oe} is not ORIENTED_EDGE");
                }
                let oe_params = list_params(oe_record)
                    .ok_or_else(|| anyhow!("oriented edge params invalid"))?;
                let edge = entity_ref_value(
                    oe_params
                        .get(3)
                        .ok_or_else(|| anyhow!("oriented edge #{oe} missing EDGE_CURVE"))?,
                )
                .ok_or_else(|| anyhow!("oriented edge #{oe} edge is not a ref"))?;
                out.push(edge);
            }
        }
        Ok(out)
    }

    pub(super) fn edge_faces(&self, faces: &HashSet<u64>) -> Result<HashMap<u64, Vec<u64>>> {
        let mut out = HashMap::<u64, Vec<u64>>::new();
        for &face in faces {
            let edges = self.face_edges_ordered(face)?;
            let mut unique = HashSet::new();
            for edge in edges {
                if unique.insert(edge) {
                    out.entry(edge).or_default().push(face);
                }
            }
        }
        Ok(out)
    }

    pub(super) fn edge_vertices(&self, edge: u64) -> Result<[u64; 2]> {
        let record = self
            .simple_record(edge)
            .ok_or_else(|| anyhow!("missing edge #{edge}"))?;
        if record.name != "EDGE_CURVE" {
            bail!("#{edge} is not EDGE_CURVE");
        }
        let params = list_params(record).ok_or_else(|| anyhow!("edge params are not a list"))?;
        let va = entity_ref_value(
            params
                .get(1)
                .ok_or_else(|| anyhow!("edge #{edge} missing start vertex"))?,
        )
        .ok_or_else(|| anyhow!("edge #{edge} start vertex is not a ref"))?;
        let vb = entity_ref_value(
            params
                .get(2)
                .ok_or_else(|| anyhow!("edge #{edge} missing end vertex"))?,
        )
        .ok_or_else(|| anyhow!("edge #{edge} end vertex is not a ref"))?;
        Ok([va, vb])
    }

    pub(super) fn vertex_coord(&self, vertex: u64) -> Result<[f64; 3]> {
        let record = self
            .simple_record(vertex)
            .ok_or_else(|| anyhow!("missing vertex #{vertex}"))?;
        if record.name != "VERTEX_POINT" {
            bail!("#{vertex} is not VERTEX_POINT");
        }
        let params = list_params(record).ok_or_else(|| anyhow!("vertex params are not a list"))?;
        let point = entity_ref_value(
            params
                .get(1)
                .ok_or_else(|| anyhow!("vertex #{vertex} missing point"))?,
        )
        .ok_or_else(|| anyhow!("vertex #{vertex} point is not a ref"))?;
        self.cartesian_point(point)
    }

    pub(super) fn cartesian_point(&self, point: u64) -> Result<[f64; 3]> {
        let record = self
            .simple_record(point)
            .ok_or_else(|| anyhow!("missing point #{point}"))?;
        if record.name != "CARTESIAN_POINT" {
            bail!("#{point} is not CARTESIAN_POINT");
        }
        let params = list_params(record).ok_or_else(|| anyhow!("point params are not a list"))?;
        let Parameter::List(coords) = params
            .get(1)
            .ok_or_else(|| anyhow!("point #{point} missing coordinates"))?
        else {
            bail!("point #{point} coordinates are not a list");
        };
        if coords.len() != 3 {
            bail!("point #{point} is not 3-D");
        }
        Ok([
            numeric_value(&coords[0]).ok_or_else(|| anyhow!("point coord is not numeric"))?,
            numeric_value(&coords[1]).ok_or_else(|| anyhow!("point coord is not numeric"))?,
            numeric_value(&coords[2]).ok_or_else(|| anyhow!("point coord is not numeric"))?,
        ])
    }

    pub(super) fn face_vertices(&self, face: u64) -> Result<HashSet<u64>> {
        let mut vertices = HashSet::new();
        for edge in self.face_edges_ordered(face)? {
            vertices.extend(self.edge_vertices(edge)?);
        }
        Ok(vertices)
    }

    pub(super) fn face_center(&self, face: u64) -> Result<[f64; 3]> {
        let vertices = self.face_vertices(face)?;
        if vertices.is_empty() {
            bail!("face #{face} has no vertices");
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for vertex in vertices {
            let p = self.vertex_coord(vertex)?;
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        Ok([
            f64::midpoint(lo[0], hi[0]),
            f64::midpoint(lo[1], hi[1]),
            f64::midpoint(lo[2], hi[2]),
        ])
    }

    pub(super) fn edge_center_span(&self, edge: u64, axis: [f64; 3]) -> Result<(f64, f64)> {
        let [va, vb] = self.edge_vertices(edge)?;
        let pa = dot(self.vertex_coord(va)?, axis);
        let pb = dot(self.vertex_coord(vb)?, axis);
        Ok((f64::midpoint(pa, pb), (pb - pa).abs()))
    }

    pub(super) fn make_edge_like(&mut self, prototype: u64, va: u64, vb: u64) -> Result<u64> {
        let record = self
            .simple_record(prototype)
            .ok_or_else(|| anyhow!("missing edge prototype #{prototype}"))?
            .clone();
        if record.name != "EDGE_CURVE" {
            bail!("edge prototype #{prototype} is not EDGE_CURVE");
        }
        let Parameter::List(mut params) = record.parameter else {
            bail!("edge prototype params are not a list");
        };
        if params.len() < 5 {
            bail!("edge prototype has incomplete parameters");
        }
        params[1] = entity_ref(va);
        params[2] = entity_ref(vb);
        Ok(self.push_simple("EDGE_CURVE", params))
    }

    pub(super) fn source_loops(&self, face: u64) -> Result<Vec<SourceLoop>> {
        let record = self
            .simple_record(face)
            .ok_or_else(|| anyhow!("missing face #{face}"))?;
        let params = list_params(record).ok_or_else(|| anyhow!("face params invalid"))?;
        let bounds = params
            .get(1)
            .and_then(entity_ref_list)
            .ok_or_else(|| anyhow!("face #{face} has no bounds"))?;
        let mut out = Vec::new();

        for bound in bounds {
            let bound_record = self
                .simple_record(bound)
                .ok_or_else(|| anyhow!("missing bound #{bound}"))?;
            let outer = bound_record.name == "FACE_OUTER_BOUND";
            let bparams =
                list_params(bound_record).ok_or_else(|| anyhow!("bound params invalid"))?;
            let loop_id = entity_ref_value(
                bparams
                    .get(1)
                    .ok_or_else(|| anyhow!("bound #{bound} missing loop"))?,
            )
            .ok_or_else(|| anyhow!("bound loop is not a ref"))?;
            let loop_record = self
                .simple_record(loop_id)
                .ok_or_else(|| anyhow!("missing loop #{loop_id}"))?;
            let lparams = list_params(loop_record).ok_or_else(|| anyhow!("loop params invalid"))?;
            let oes = lparams
                .get(1)
                .and_then(entity_ref_list)
                .ok_or_else(|| anyhow!("loop #{loop_id} has no edges"))?;
            let mut traversal = Vec::with_capacity(oes.len());
            for oe in oes {
                let oe_record = self
                    .simple_record(oe)
                    .ok_or_else(|| anyhow!("missing oriented edge #{oe}"))?;
                let params = list_params(oe_record).ok_or_else(|| anyhow!("OE params invalid"))?;
                let edge = entity_ref_value(
                    params
                        .get(3)
                        .ok_or_else(|| anyhow!("OE #{oe} missing EDGE_CURVE"))?,
                )
                .ok_or_else(|| anyhow!("OE edge is not a ref"))?;
                let forward = matches!(
                    params.get(4),
                    Some(Parameter::Enumeration(value)) if value == "T"
                );
                let [va, vb] = self.edge_vertices(edge)?;
                traversal.push(if forward {
                    Traversal {
                        edge,
                        start: va,
                        end: vb,
                    }
                } else {
                    Traversal {
                        edge,
                        start: vb,
                        end: va,
                    }
                });
            }
            out.push(SourceLoop { outer, traversal });
        }
        Ok(out)
    }

    pub(super) fn rebuild_face_bounds(
        &mut self,
        face: u64,
        target_edges: &HashSet<u64>,
        source_loops: &[SourceLoop],
    ) -> Result<Vec<u64>> {
        let old_bounds = {
            let record = self
                .simple_record(face)
                .ok_or_else(|| anyhow!("missing face #{face} before boundary rewrite"))?;
            let params = list_params(record)
                .ok_or_else(|| anyhow!("face #{face} parameters are not a list"))?;
            params
                .get(1)
                .and_then(entity_ref_list)
                .ok_or_else(|| anyhow!("face #{face} has no boundary aggregate"))?
        };
        let source_outer = source_loops
            .iter()
            .find(|loop_| loop_.outer)
            .ok_or_else(|| anyhow!("face #{face} has no source outer loop"))?;
        let ref_area = self.area_vector(&source_outer.traversal)?;
        if norm(ref_area) <= 1.0e-12 {
            bail!("face #{face} source outer loop has zero area");
        }
        let hole_sign = if let Some(hole) = source_loops.iter().find(|loop_| !loop_.outer) {
            if dot(self.area_vector(&hole.traversal)?, ref_area) > 0.0 {
                1.0
            } else {
                -1.0
            }
        } else {
            -1.0
        };

        let mut cycles = self.trace_cycles(target_edges)?;
        if cycles.is_empty() {
            bail!("face #{face} target boundary has no cycles");
        }
        let scored = cycles
            .iter()
            .map(|cycle| {
                self.area_vector(cycle)
                    .map(|area| dot(area, ref_area).abs())
            })
            .collect::<Result<Vec<_>>>()?;
        let outer_index = scored
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .ok_or_else(|| anyhow!("face #{face} target boundary has no scorable cycles"))?;

        let mut new_bounds = Vec::with_capacity(cycles.len());
        for (index, cycle) in cycles.iter_mut().enumerate() {
            let desired = if index == outer_index { 1.0 } else { hole_sign };
            let actual = dot(self.area_vector(cycle)?, ref_area);
            if actual == 0.0 {
                bail!("face #{face} target cycle has zero area");
            }
            if (actual > 0.0) != (desired > 0.0) {
                cycle.reverse();
                for step in cycle.iter_mut() {
                    std::mem::swap(&mut step.start, &mut step.end);
                }
            }

            let mut oriented = Vec::with_capacity(cycle.len());
            for step in cycle.iter() {
                let [va, vb] = self.edge_vertices(step.edge)?;
                let forward = if step.start == va && step.end == vb {
                    true
                } else if step.start == vb && step.end == va {
                    false
                } else {
                    bail!("face #{face} cycle endpoint mismatch");
                };
                let oe = self.push_simple(
                    "ORIENTED_EDGE",
                    vec![
                        Parameter::String(String::new()),
                        Parameter::Omitted,
                        Parameter::Omitted,
                        entity_ref(step.edge),
                        Parameter::Enumeration(if forward { "T" } else { "F" }.to_string()),
                    ],
                );
                oriented.push(oe);
            }
            let loop_id = self.push_simple(
                "EDGE_LOOP",
                vec![
                    Parameter::String(String::new()),
                    Parameter::List(oriented.into_iter().map(entity_ref).collect()),
                ],
            );
            let bound_name = if index == outer_index {
                "FACE_OUTER_BOUND"
            } else {
                "FACE_BOUND"
            };
            let bound = self.push_simple(
                bound_name,
                vec![
                    Parameter::String(String::new()),
                    entity_ref(loop_id),
                    Parameter::Enumeration("T".to_string()),
                ],
            );
            new_bounds.push(bound);
        }

        let record = self
            .simple_record_mut(face)
            .ok_or_else(|| anyhow!("missing face #{face} during rewrite"))?;
        let Parameter::List(params) = &mut record.parameter else {
            bail!("face #{face} parameters changed shape");
        };
        if params.len() < 4 {
            bail!("face #{face} has incomplete ADVANCED_FACE parameters");
        }
        params[1] = Parameter::List(new_bounds.into_iter().map(entity_ref).collect());
        Ok(old_bounds)
    }

    pub(super) fn trace_cycles(&self, edges: &HashSet<u64>) -> Result<Vec<Vec<Traversal>>> {
        let mut incident = HashMap::<u64, Vec<u64>>::new();
        for &edge in edges {
            let [a, b] = self.edge_vertices(edge)?;
            incident.entry(a).or_default().push(edge);
            incident.entry(b).or_default().push(edge);
        }
        for (&vertex, members) in &incident {
            if members.len() != 2 {
                bail!(
                    "target face boundary is not a disjoint set of cycles: vertex #{vertex} has degree {}",
                    members.len()
                );
            }
        }

        let mut unused = edges.clone();
        let mut cycles = Vec::new();
        while !unused.is_empty() {
            let edge0 = *unused
                .iter()
                .min()
                .ok_or_else(|| anyhow!("target boundary edge set unexpectedly empty"))?;
            let [start_vertex, _] = self.edge_vertices(edge0)?;
            let mut current_vertex = start_vertex;
            let mut current_edge = edge0;
            let mut cycle = Vec::new();

            loop {
                unused.remove(&current_edge);
                let [a, b] = self.edge_vertices(current_edge)?;
                let next_vertex = if current_vertex == a {
                    b
                } else if current_vertex == b {
                    a
                } else {
                    bail!("cycle traversal lost endpoint identity");
                };
                cycle.push(Traversal {
                    edge: current_edge,
                    start: current_vertex,
                    end: next_vertex,
                });
                current_vertex = next_vertex;
                if current_vertex == start_vertex {
                    break;
                }
                let candidates = incident
                    .get(&current_vertex)
                    .into_iter()
                    .flatten()
                    .filter(|edge| unused.contains(edge))
                    .copied()
                    .collect::<Vec<_>>();
                if candidates.len() != 1 {
                    bail!(
                        "cycle continuation at vertex #{current_vertex} is ambiguous: {candidates:?}"
                    );
                }
                current_edge = candidates[0];
            }
            cycles.push(cycle);
        }
        Ok(cycles)
    }

    pub(super) fn area_vector(&self, traversal: &[Traversal]) -> Result<[f64; 3]> {
        if traversal.len() < 3 {
            return Ok([0.0; 3]);
        }
        let points = traversal
            .iter()
            .map(|step| self.vertex_coord(step.start))
            .collect::<Result<Vec<_>>>()?;
        let mut acc = [0.0; 3];
        for index in 0..points.len() {
            let a = points[index];
            let b = points[(index + 1) % points.len()];
            acc = add(acc, cross(a, b));
        }
        Ok(scale(acc, 0.5))
    }
}

fn translate_cartesian_point(entity: &mut EntityInstance, delta: [f64; 3]) -> Result<()> {
    let EntityInstance::Simple { record, .. } = entity else {
        return Ok(());
    };
    if record.name != "CARTESIAN_POINT" {
        return Ok(());
    }
    let Parameter::List(params) = &mut record.parameter else {
        bail!("CARTESIAN_POINT parameters are not a list");
    };
    let Some(Parameter::List(coords)) = params.get_mut(1) else {
        bail!("CARTESIAN_POINT lacks coordinate aggregate");
    };
    if coords.len() != 3 {
        bail!("CARTESIAN_POINT is not 3-D");
    }
    for index in 0..3 {
        let value = numeric_value(&coords[index])
            .ok_or_else(|| anyhow!("CARTESIAN_POINT coordinate is not numeric"))?;
        coords[index] = Parameter::Real(value + delta[index]);
    }
    Ok(())
}

fn remap_entity_refs(entity: &mut EntityInstance, mapping: &HashMap<u64, u64>) {
    match entity {
        EntityInstance::Simple { record, .. } => remap_param_refs(&mut record.parameter, mapping),
        EntityInstance::Complex { subsuper, .. } => {
            for record in &mut subsuper.0 {
                remap_param_refs(&mut record.parameter, mapping);
            }
        }
    }
}

fn remap_param_refs(param: &mut Parameter, mapping: &HashMap<u64, u64>) {
    match param {
        Parameter::Ref(Name::Entity(id)) => {
            if let Some(new) = mapping.get(id) {
                *id = *new;
            }
        }
        Parameter::List(items) => {
            for item in items {
                remap_param_refs(item, mapping);
            }
        }
        Parameter::Typed { parameter, .. } => remap_param_refs(parameter, mapping),
        _ => {}
    }
}

const fn set_entity_id(entity: &mut EntityInstance, id: u64) {
    match entity {
        EntityInstance::Simple { id: current, .. }
        | EntityInstance::Complex { id: current, .. } => *current = id,
    }
}
