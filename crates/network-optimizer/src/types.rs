use engine_core::NodeId;
use serde::{Deserialize, Serialize};

pub type DirectedEdge = (NodeId, NodeId);

/// A canonical unordered edge key (min(u, v), max(u, v))
#[inline(always)]
pub fn canonical_edge(u: NodeId, v: NodeId) -> DirectedEdge {
    if u < v { (u, v) } else { (v, u) }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedCameraPole {
    pub camera_id: String,
    pub edge_from: NodeId,
    pub edge_to: NodeId,
    pub lat: f64,
    pub lon: f64,
    pub is_two_way: bool,
}

#[derive(Debug, Clone)]
pub struct MacroCorridor {
    pub id: usize,
    pub start_node: NodeId,
    pub end_node: NodeId,
    pub total_length_m: f64,
    /// Ordered chain of underlying raw directed edges in CityGraph
    pub raw_edges: Vec<DirectedEdge>,
    /// Intermediate pass-through nodes along this chain
    pub internal_nodes: Vec<NodeId>,
    pub is_two_way: bool,
    /// Pre-selected best physical segment to place a camera on (usually near the midpoint)
    pub representative_edge: DirectedEdge,
}