use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RecoveredProfileCurve {
    Line {
        source_edge_ids: Vec<u64>,
        start_mm: [f64; 2],
        end_mm: [f64; 2],
    },
    CircleArc {
        source_edge_ids: Vec<u64>,
        center_mm: [f64; 2],
        radius_mm: f64,
        start_angle_rad: f64,
        end_angle_rad: f64,
    },
    Bezier {
        source_edge_ids: Vec<u64>,
        control_points_mm: Vec<[f64; 2]>,
    },
    BSpline {
        source_edge_ids: Vec<u64>,
        degree: usize,
        control_points_mm: Vec<[f64; 2]>,
        knots: Vec<f64>,
        weights: Option<Vec<f64>>,
    },
}

impl RecoveredProfileCurve {
    pub(crate) fn first_source_edge_id(&self) -> u64 {
        self.source_edge_ids().first().copied().unwrap_or(u64::MAX)
    }

    pub fn source_edge_ids(&self) -> &[u64] {
        match self {
            Self::Line {
                source_edge_ids, ..
            }
            | Self::CircleArc {
                source_edge_ids, ..
            }
            | Self::Bezier {
                source_edge_ids, ..
            }
            | Self::BSpline {
                source_edge_ids, ..
            } => source_edge_ids,
        }
    }

    pub(crate) fn complexity(&self) -> usize {
        match self {
            Self::Line { .. } => 1,
            Self::CircleArc { .. } => 2,
            Self::Bezier {
                control_points_mm, ..
            } => 2 + control_points_mm.len(),
            Self::BSpline {
                control_points_mm,
                knots,
                weights,
                ..
            } => 4 + control_points_mm.len() + knots.len() + weights.as_ref().map_or(0, Vec::len),
        }
    }

    pub(crate) fn start_point(&self) -> Option<[f64; 2]> {
        match self {
            Self::Line { start_mm, .. } => Some(*start_mm),
            Self::CircleArc {
                center_mm,
                radius_mm,
                start_angle_rad,
                ..
            } => Some([
                center_mm[0] + radius_mm * start_angle_rad.cos(),
                center_mm[1] + radius_mm * start_angle_rad.sin(),
            ]),
            Self::Bezier {
                control_points_mm, ..
            }
            | Self::BSpline {
                control_points_mm, ..
            } => control_points_mm.first().copied(),
        }
    }

    pub(crate) fn end_point(&self) -> Option<[f64; 2]> {
        match self {
            Self::Line { end_mm, .. } => Some(*end_mm),
            Self::CircleArc {
                center_mm,
                radius_mm,
                end_angle_rad,
                ..
            } => Some([
                center_mm[0] + radius_mm * end_angle_rad.cos(),
                center_mm[1] + radius_mm * end_angle_rad.sin(),
            ]),
            Self::Bezier {
                control_points_mm, ..
            }
            | Self::BSpline {
                control_points_mm, ..
            } => control_points_mm.last().copied(),
        }
    }

    pub(crate) fn reversed(&self) -> Self {
        match self {
            Self::Line {
                source_edge_ids,
                start_mm,
                end_mm,
            } => Self::Line {
                source_edge_ids: source_edge_ids.clone(),
                start_mm: *end_mm,
                end_mm: *start_mm,
            },
            Self::CircleArc {
                source_edge_ids,
                center_mm,
                radius_mm,
                start_angle_rad,
                end_angle_rad,
            } => Self::CircleArc {
                source_edge_ids: source_edge_ids.clone(),
                center_mm: *center_mm,
                radius_mm: *radius_mm,
                start_angle_rad: *end_angle_rad,
                end_angle_rad: *start_angle_rad,
            },
            Self::Bezier {
                source_edge_ids,
                control_points_mm,
            } => Self::Bezier {
                source_edge_ids: source_edge_ids.clone(),
                control_points_mm: control_points_mm.iter().rev().copied().collect(),
            },
            Self::BSpline {
                source_edge_ids,
                degree,
                control_points_mm,
                knots,
                weights,
            } => Self::BSpline {
                source_edge_ids: source_edge_ids.clone(),
                degree: *degree,
                control_points_mm: control_points_mm.iter().rev().copied().collect(),
                knots: knots.iter().rev().map(|knot| 1.0 - knot).collect(),
                weights: weights
                    .as_ref()
                    .map(|values| values.iter().rev().copied().collect()),
            },
        }
    }

    pub(crate) fn is_spline(&self) -> bool {
        matches!(self, Self::Bezier { .. } | Self::BSpline { .. })
    }
}
