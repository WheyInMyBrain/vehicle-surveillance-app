use road_network_core::NodeId;
use serde::{Deserialize, Serialize};

pub type DirectedEdge = (NodeId, NodeId);

#[inline(always)]
pub fn canonical_edge(u: NodeId, v: NodeId) -> DirectedEdge {
    if u < v { (u, v) } else { (v, u) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MountType {
    VergePole,
    OverheadGantry,
    IntersectionMast,
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
    pub is_perimeter_gate: bool,
    pub is_bridge_seal: bool,
    pub intercepted_trips: usize,
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

#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub max_dark_distance_m: f32,
    pub max_dark_hops: usize,
    pub max_unmonitored_forks: usize,
    pub min_hits_per_trip: u8,
    pub num_od_eval_trips: usize,
    pub adversarial_epochs: usize,
    pub lanes_per_camera: u8,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            max_dark_distance_m: 500.0,
            max_dark_hops: 4,
            max_unmonitored_forks: 2,
            min_hits_per_trip: 2,
            num_od_eval_trips: 10_000,
            adversarial_epochs: 40,
            lanes_per_camera: 2,
        }
    }
}