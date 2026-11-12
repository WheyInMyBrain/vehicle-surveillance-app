use crate::hardware::{MountType, PhysicalSiteProfile};
use road_network_core::NodeId;
use serde::{Deserialize, Serialize};

pub type DirectedEdge = (NodeId, NodeId);

#[inline(always)]
pub fn canonical_edge(u: NodeId, v: NodeId) -> DirectedEdge {
    if u < v {
        (u, v)
    } else {
        (v, u)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraPole {
    pub pole_id: String,
    pub edge_from: NodeId,
    pub edge_to: NodeId,
    pub lat: f64,
    pub lon: f64,
    pub bearing_deg: f32,
    pub camera_heads: u8,
    pub lanes_covered: u8,
    pub mount_type: MountType,
    pub is_two_way: bool,
    pub estimated_cost_inr: f64,
    pub profile: PhysicalSiteProfile,
}

#[derive(Debug, Clone)]
pub struct MacroCorridor {
    pub id: usize,
    pub start_junction: NodeId,
    pub end_junction: NodeId,
    pub total_length_m: f32,
    pub lanes_forward: u8,
    pub lanes_backward: u8,
    pub maxspeed_kmh: u8,
    pub is_two_way: bool,
    pub raw_edges: Vec<DirectedEdge>,
    pub representative_edge: DirectedEdge,
    pub midpoint: (f64, f64, f32), // (lat, lon, bearing)
}