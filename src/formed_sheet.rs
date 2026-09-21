use ruststep::ast::{EntityInstance, Name, Parameter, Record};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};

const GEOM_TOL_MM: f64 = 1.0e-5;
const DIR_TOL: f64 = 1.0e-10;
const MIN_COAXIAL_PAIRS: usize = 3;
const MIN_PARALLEL_PLANE_PAIRS: usize = 2;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FormedSheetEvidence {
    pub solid_id: u64,
    pub thickness_mm: f64,
    pub cylindrical_faces: usize,
    pub paired_cylindrical_faces: usize,
    pub coaxial_radius_pairs: usize,
    pub parallel_plane_pairs: usize,
    pub paired_cylinder_face_ratio: f64,
    pub cylinder_pairs: Vec<SheetCylinderPair>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SheetCylinderPair {
    pub inner_face_id: u64,
    pub outer_face_id: u64,
    pub inner_radius_mm: f64,
    pub outer_radius_mm: f64,
    pub axis: [f64; 3],
}

#[derive(Debug, Clone)]
struct PlaneFace {
    origin: [f64; 3],
    normal: [f64; 3],
}

#[derive(Debug, Clone)]
struct CylinderFace {
    face_id: u64,
    origin: [f64; 3],
    axis: [f64; 3],
    radius_mm: f64,
}

pub fn detect_formed_sheet_evidence(entities: &[EntityInstance]) -> Vec<FormedSheetEvidence> {
    let index = build_index(entities);
    let mut out = Vec::new();

    for entity in entities {
        let Some(record) = simple_record(entity) else {
            continue;
        };
        if record.name != "MANIFOLD_SOLID_BREP" {
            continue;
        }
        let solid_id = entity_id(entity);
        let Some(face_ids) = solid_faces(solid_id, entities, &index) else {
            continue;
        };

        let mut planes = Vec::new();
        let mut cylinders = Vec::new();
        for face_id in face_ids {
            let Some((surface_id, surface)) = face_surface(face_id, entities, &index) else {
                continue;
            };
            match surface.name.as_str() {
                "PLANE" => {
                    if let Some((origin, normal)) = surface_axis(surface_id, entities, &index) {
                        planes.push(PlaneFace { origin, normal });
                    }
                }
                "CYLINDRICAL_SURFACE" => {
                    let Some((origin, axis)) = surface_axis(surface_id, entities, &index) else {
                        continue;
                    };
                    let Some(radius_mm) = surface_radius(surface) else {
                        continue;
                    };
                    if radius_mm.is_finite() && radius_mm > GEOM_TOL_MM {
                        cylinders.push(CylinderFace {
                            face_id,
                            origin,
                            axis,
                            radius_mm,
                        });
                    }
                }
                _ => {}
            }
        }

        if cylinders.len() < MIN_COAXIAL_PAIRS * 2 || planes.len() < 2 {
            continue;
        }

        let cylinder_candidates = coaxial_radius_differences(&cylinders);
        if cylinder_candidates.is_empty() {
            continue;
        }
        let plane_candidates = parallel_plane_separations(&planes);

        let mut thickness_scores = BTreeMap::<i64, (usize, usize, Vec<SheetCylinderPair>)>::new();
        for pair in cylinder_candidates {
            let thickness = pair.outer_radius_mm - pair.inner_radius_mm;
            let tick = quantize_mm(thickness);
            if tick <= 0 {
                continue;
            }
            let entry = thickness_scores.entry(tick).or_default();
            entry.0 += 1;
            entry.2.push(pair);
        }
        for separation in plane_candidates {
            let tick = quantize_mm(separation);
            if let Some(entry) = thickness_scores.get_mut(&tick) {
                entry.1 += 1;
            }
        }

        let Some((&tick, (coaxial_pairs, plane_pairs, pairs))) = thickness_scores
            .iter()
            .max_by_key(|(tick, (cyl, plane, _))| (*cyl, *plane, -tick.abs()))
        else {
            continue;
        };
        if *coaxial_pairs < MIN_COAXIAL_PAIRS || *plane_pairs < MIN_PARALLEL_PLANE_PAIRS {
            continue;
        }

        let paired_faces = pairs
            .iter()
            .flat_map(|pair| [pair.inner_face_id, pair.outer_face_id])
            .collect::<HashSet<_>>()
            .len();
        let ratio = paired_faces as f64 / cylinders.len() as f64;

        out.push(FormedSheetEvidence {
            solid_id,
            thickness_mm: tick as f64 * GEOM_TOL_MM,
            cylindrical_faces: cylinders.len(),
            paired_cylindrical_faces: paired_faces,
            coaxial_radius_pairs: *coaxial_pairs,
            parallel_plane_pairs: *plane_pairs,
            paired_cylinder_face_ratio: ratio,
            cylinder_pairs: pairs.clone(),
        });
    }

    out.sort_by(|a, b| {
        b.paired_cylinder_face_ratio
            .total_cmp(&a.paired_cylinder_face_ratio)
            .then_with(|| b.coaxial_radius_pairs.cmp(&a.coaxial_radius_pairs))
            .then_with(|| a.solid_id.cmp(&b.solid_id))
    });
    out
}

fn coaxial_radius_differences(cylinders: &[CylinderFace]) -> Vec<SheetCylinderPair> {
    let mut out = Vec::new();
    for (index, a) in cylinders.iter().enumerate() {
        for b in &cylinders[index + 1..] {
            if !parallel(a.axis, b.axis) {
                continue;
            }
            if axis_line_distance(a.origin, a.axis, b.origin) > GEOM_TOL_MM {
                continue;
            }
            let (inner, outer) = if a.radius_mm <= b.radius_mm {
                (a, b)
            } else {
                (b, a)
            };
            let thickness = outer.radius_mm - inner.radius_mm;
            if thickness <= GEOM_TOL_MM {
                continue;
            }
            out.push(SheetCylinderPair {
                inner_face_id: inner.face_id,
                outer_face_id: outer.face_id,
                inner_radius_mm: inner.radius_mm,
                outer_radius_mm: outer.radius_mm,
                axis: canonical_axis(inner.axis),
            });
        }
    }
    out
}

fn parallel_plane_separations(planes: &[PlaneFace]) -> Vec<f64> {
    let mut out = Vec::new();
    for (index, a) in planes.iter().enumerate() {
        for b in &planes[index + 1..] {
            if !parallel(a.normal, b.normal) {
                continue;
            }
            let separation = dot(a.normal, sub(b.origin, a.origin)).abs();
            if separation > GEOM_TOL_MM && separation.is_finite() {
                out.push(separation);
            }
        }
    }
    out
}

fn solid_faces(
    solid_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<Vec<u64>> {
    let solid = simple_record(entities.get(*index.get(&solid_id)?)?)?;
    let params = list_params(solid)?;
    let shell_id = params.get(1).and_then(entity_ref_value)?;
    let shell = simple_record(entities.get(*index.get(&shell_id)?)?)?;
    if shell.name != "CLOSED_SHELL" && shell.name != "OPEN_SHELL" {
        return None;
    }
    let shell_params = list_params(shell)?;
    let Parameter::List(items) = shell_params.get(1)? else {
        return None;
    };
    items.iter().map(entity_ref_value).collect()
}

fn face_surface<'a>(
    face_id: u64,
    entities: &'a [EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<(u64, &'a Record)> {
    let face = simple_record(entities.get(*index.get(&face_id)?)?)?;
    if face.name != "ADVANCED_FACE" {
        return None;
    }
    let params = list_params(face)?;
    let surface_id = params.get(2).and_then(entity_ref_value)?;
    let surface = simple_record(entities.get(*index.get(&surface_id)?)?)?;
    Some((surface_id, surface))
}

fn surface_axis(
    surface_id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<([f64; 3], [f64; 3])> {
    let surface = simple_record(entities.get(*index.get(&surface_id)?)?)?;
    let params = list_params(surface)?;
    let placement_id = params.get(1).and_then(entity_ref_value)?;
    let placement = simple_record(entities.get(*index.get(&placement_id)?)?)?;
    if placement.name != "AXIS2_PLACEMENT_3D" {
        return None;
    }
    let placement_params = list_params(placement)?;
    let origin_id = placement_params.get(1).and_then(entity_ref_value)?;
    let axis_id = placement_params.get(2).and_then(entity_ref_value)?;
    let origin = cartesian_point(origin_id, entities, index)?;
    let axis = direction(axis_id, entities, index)?;
    Some((origin, normalize(axis)?))
}

fn surface_radius(surface: &Record) -> Option<f64> {
    let params = list_params(surface)?;
    numeric_value(params.get(2)?)
}

fn cartesian_point(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let record = simple_record(entities.get(*index.get(&id)?)?)?;
    if record.name != "CARTESIAN_POINT" {
        return None;
    }
    let params = list_params(record)?;
    let Parameter::List(coords) = params.get(1)? else {
        return None;
    };
    if coords.len() != 3 {
        return None;
    }
    Some([
        numeric_value(&coords[0])?,
        numeric_value(&coords[1])?,
        numeric_value(&coords[2])?,
    ])
}

fn direction(
    id: u64,
    entities: &[EntityInstance],
    index: &HashMap<u64, usize>,
) -> Option<[f64; 3]> {
    let record = simple_record(entities.get(*index.get(&id)?)?)?;
    if record.name != "DIRECTION" {
        return None;
    }
    let params = list_params(record)?;
    let Parameter::List(coords) = params.get(1)? else {
        return None;
    };
    if coords.len() != 3 {
        return None;
    }
    Some([
        numeric_value(&coords[0])?,
        numeric_value(&coords[1])?,
        numeric_value(&coords[2])?,
    ])
}

fn parallel(a: [f64; 3], b: [f64; 3]) -> bool {
    (dot(a, b).abs() - 1.0).abs() <= DIR_TOL
}

fn axis_line_distance(origin: [f64; 3], axis: [f64; 3], other: [f64; 3]) -> f64 {
    norm(cross(sub(other, origin), axis))
}

fn canonical_axis(mut axis: [f64; 3]) -> [f64; 3] {
    for value in axis {
        if value.abs() > DIR_TOL {
            if value < 0.0 {
                axis = [-axis[0], -axis[1], -axis[2]];
            }
            break;
        }
    }
    axis
}

fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    let length = norm(vector);
    if !length.is_finite() || length <= DIR_TOL {
        return None;
    }
    Some([vector[0] / length, vector[1] / length, vector[2] / length])
}

fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn quantize_mm(value: f64) -> i64 {
    (value / GEOM_TOL_MM).round() as i64
}

fn build_index(entities: &[EntityInstance]) -> HashMap<u64, usize> {
    entities
        .iter()
        .enumerate()
        .map(|(index, entity)| (entity_id(entity), index))
        .collect()
}

fn entity_id(entity: &EntityInstance) -> u64 {
    match entity {
        EntityInstance::Simple { id, .. } | EntityInstance::Complex { id, .. } => *id,
    }
}

fn simple_record(entity: &EntityInstance) -> Option<&Record> {
    match entity {
        EntityInstance::Simple { record, .. } => Some(record),
        EntityInstance::Complex { .. } => None,
    }
}

fn list_params(record: &Record) -> Option<&Vec<Parameter>> {
    match &record.parameter {
        Parameter::List(params) => Some(params),
        _ => None,
    }
}

fn entity_ref_value(parameter: &Parameter) -> Option<u64> {
    match parameter {
        Parameter::Ref(Name::Entity(id)) => Some(*id),
        _ => None,
    }
}

fn numeric_value(parameter: &Parameter) -> Option<f64> {
    match parameter {
        Parameter::Real(value) => Some(*value),
        Parameter::Integer(value) => Some(*value as f64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coaxial_radius_pairs_recover_wall_thickness() {
        let cylinders = vec![
            CylinderFace {
                face_id: 1,
                origin: [0.0, 0.0, 0.0],
                axis: [0.0, 0.0, 1.0],
                radius_mm: 0.4,
            },
            CylinderFace {
                face_id: 2,
                origin: [0.0, 0.0, 8.0],
                axis: [0.0, 0.0, -1.0],
                radius_mm: 0.6,
            },
            CylinderFace {
                face_id: 3,
                origin: [5.0, 0.0, 0.0],
                axis: [0.0, 0.0, 1.0],
                radius_mm: 0.9,
            },
        ];
        let pairs = coaxial_radius_differences(&cylinders);
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].outer_radius_mm - pairs[0].inner_radius_mm - 0.2).abs() < 1.0e-12);
    }

    #[test]
    fn plane_separation_is_orientation_invariant() {
        let planes = vec![
            PlaneFace {
                origin: [0.0, 0.0, 0.0],
                normal: [1.0, 0.0, 0.0],
            },
            PlaneFace {
                origin: [0.2, 5.0, -3.0],
                normal: [-1.0, 0.0, 0.0],
            },
        ];
        let separations = parallel_plane_separations(&planes);
        assert_eq!(separations.len(), 1);
        assert!((separations[0] - 0.2).abs() < 1.0e-12);
    }
}
