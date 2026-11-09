use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub type NodeId = u64;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RoadSurface {
    Asphalt = 0,
    Paved = 1,
    Cobblestone = 2,
    Unpaved = 3,
    Unknown = 4,
}

impl RoadSurface {
    #[inline(always)]
    pub fn friction_multiplier(&self) -> f32 {
        match self {
            RoadSurface::Asphalt => 1.00,
            RoadSurface::Paved => 0.95,
            RoadSurface::Cobblestone => 0.75,
            RoadSurface::Unpaved => 0.55,
            RoadSurface::Unknown => 0.90,
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HighwayType {
    Motorway = 1,
    Trunk = 2,
    Primary = 3,
    Secondary = 4,
    Tertiary = 5,
    Residential = 6,
    LivingStreet = 7,
    Unclassified = 8,
    Service = 9,
    Track = 10,
}

impl HighwayType {
    #[inline(always)]
    pub fn baseline_speed_kmh(&self) -> u8 {
        match self {
            HighwayType::Motorway => 80,
            HighwayType::Trunk => 60,
            HighwayType::Primary => 50,
            HighwayType::Secondary => 40,
            HighwayType::Tertiary => 35,
            HighwayType::Residential => 25,
            HighwayType::LivingStreet => 15,
            HighwayType::Unclassified => 30,
            HighwayType::Service => 15,
            HighwayType::Track => 10,
        }
    }

    #[inline(always)]
    pub fn speed_mps(&self) -> f32 {
        (self.baseline_speed_kmh() as f32) / 3.6
    }

    #[inline(always)]
    pub fn hierarchy_weight(&self) -> f32 {
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

pub mod edge_flags {
    pub const ONEWAY: u8  = 1 << 0;
    pub const BRIDGE: u8  = 1 << 1;
    pub const TUNNEL: u8  = 1 << 2;
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NodeMeta {
    pub id: NodeId,
    pub lat: f64,
    pub lon: f64,
    pub has_traffic_signal: bool,
    pub has_speed_bump: bool,
    pub is_intersection: bool,
}

/// Cache-optimized 32-byte Edge Structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMeta {
    pub name: Option<Box<str>>,
    pub length_m: f32,
    pub bearing_deg: f32,
    pub maxspeed_kmh: Option<u8>,
    pub lanes: Option<u8>,
    pub highway_type: HighwayType,
    pub surface: RoadSurface,
    pub layer: i8,
    pub flags: u8,
}

impl EdgeMeta {
    #[inline(always)]
    pub fn is_oneway(&self) -> bool {
        (self.flags & edge_flags::ONEWAY) != 0
    }

    #[inline(always)]
    pub fn is_bridge(&self) -> bool {
        (self.flags & edge_flags::BRIDGE) != 0
    }

    #[inline(always)]
    pub fn is_tunnel(&self) -> bool {
        (self.flags & edge_flags::TUNNEL) != 0
    }

    #[inline]
    pub fn new(
        name: Option<String>,
        length_m: f64,
        highway_type: HighwayType,
        maxspeed_kmh: Option<f64>,
        lanes: Option<u8>,
        surface: RoadSurface,
        bearing_deg: f64,
        is_oneway: bool,
        is_bridge: bool,
        is_tunnel: bool,
        layer: i8,
    ) -> Self {
        let mut flags = 0u8;
        if is_oneway { flags |= edge_flags::ONEWAY; }
        if is_bridge { flags |= edge_flags::BRIDGE; }
        if is_tunnel { flags |= edge_flags::TUNNEL; }

        Self {
            name: name.map(|s| s.into_boxed_str()),
            length_m: length_m as f32,
            bearing_deg: bearing_deg as f32,
            maxspeed_kmh: maxspeed_kmh.map(|v| v.round().clamp(5.0, 200.0) as u8),
            lanes,
            highway_type,
            surface,
            layer,
            flags,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraSighting {
    pub vehicle_id: String,
    pub camera_id: String,
    pub edge_from: NodeId,
    pub edge_to: NodeId,
    pub timestamp: DateTime<Utc>,
    pub speed_kmh: f32,
}