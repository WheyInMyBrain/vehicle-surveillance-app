use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub type NodeId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HighwayType {
    Motorway,
    Trunk,
    Primary,
    Secondary,
    Tertiary,
    Residential,
    Unclassified,
    Service,
}

impl HighwayType {
    pub fn speed_mps(&self) -> f64 {
        match self {
            HighwayType::Motorway => 70.0 / 3.6,
            HighwayType::Trunk => 60.0 / 3.6,
            HighwayType::Primary => 50.0 / 3.6,
            HighwayType::Secondary => 40.0 / 3.6,
            HighwayType::Tertiary => 32.0 / 3.6,
            HighwayType::Residential => 20.0 / 3.6,
            HighwayType::Unclassified => 22.0 / 3.6,
            HighwayType::Service => 15.0 / 3.6,
        }
    }

    pub fn hierarchy_weight(&self) -> f64 {
        match self {
            HighwayType::Motorway => 0.65,
            HighwayType::Trunk => 0.70,
            HighwayType::Primary => 0.80,
            HighwayType::Secondary => 0.95,
            HighwayType::Tertiary => 1.10,
            HighwayType::Residential => 1.50,
            HighwayType::Unclassified => 1.30,
            HighwayType::Service => 2.00,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMeta {
    pub lat: f64,
    pub lon: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMeta {
    pub length_m: f64,
    pub highway_type: HighwayType,
    pub bearing_deg: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraSighting {
    pub vehicle_id: String,
    pub camera_id: String,
    pub edge_from: NodeId,
    pub edge_to: NodeId,
    pub timestamp: DateTime<Utc>,
    pub speed_kmh: f64,
}