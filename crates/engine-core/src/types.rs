use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub type NodeId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RoadSurface {
    Asphalt,
    Paved,
    Cobblestone,
    Unpaved,
    Unknown,
}

impl RoadSurface {
    pub fn friction_multiplier(&self) -> f64 {
        match self {
            RoadSurface::Asphalt => 1.00,
            RoadSurface::Paved => 0.95,
            RoadSurface::Cobblestone => 0.75,
            RoadSurface::Unpaved => 0.55,
            RoadSurface::Unknown => 0.90,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HighwayType {
    Motorway,
    Trunk,
    Primary,
    Secondary,
    Tertiary,
    Residential,
    LivingStreet,
    Unclassified,
    Service,
    Track,
}

impl HighwayType {
    pub fn speed_mps(&self) -> f64 {
        match self {
            HighwayType::Motorway => 70.0 / 3.6,
            HighwayType::Trunk => 60.0 / 3.6,
            HighwayType::Primary => 45.0 / 3.6,
            HighwayType::Secondary => 35.0 / 3.6,
            HighwayType::Tertiary => 25.0 / 3.6,
            HighwayType::Residential => 20.0 / 3.6,
            HighwayType::LivingStreet => 15.0 / 3.6,
            HighwayType::Unclassified => 22.0 / 3.6,
            HighwayType::Service => 12.0 / 3.6,
            HighwayType::Track => 10.0 / 3.6,
        }
    }

    pub fn hierarchy_weight(&self) -> f64 {
        match self {
            HighwayType::Motorway => 0.65,
            HighwayType::Trunk => 0.70,
            HighwayType::Primary => 0.80,
            HighwayType::Secondary => 0.95,
            HighwayType::Tertiary => 1.10,
            HighwayType::Residential => 1.45,
            HighwayType::LivingStreet => 1.80,
            HighwayType::Unclassified => 1.30,
            HighwayType::Service => 2.10,
            HighwayType::Track => 2.50,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMeta {
    pub lat: f64,
    pub lon: f64,
    pub has_traffic_signal: bool,
    pub has_speed_bump: bool,
    pub is_intersection: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMeta {
    pub name: Option<String>,
    pub length_m: f64,
    pub highway_type: HighwayType,
    pub maxspeed_kmh: Option<f64>,
    pub lanes: Option<u8>,
    pub surface: RoadSurface,
    pub bearing_deg: f64,
    pub is_oneway: bool,
    pub is_bridge: bool,
    pub is_tunnel: bool,
    pub layer: i8,
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