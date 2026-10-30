use engine_core::NodeId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedCameraPole {
    pub camera_id: String,
    pub edge_from: NodeId,
    pub edge_to: NodeId,
    pub lat: f64,
    pub lon: f64,
    pub total_intercepted_trips: usize,
    pub marginal_new_trips: usize,
    pub is_two_way: bool,
}

#[derive(Debug, Clone)]
pub enum OptimizationGoal {
    /// Fixed budget: maximize coverage with exactly K cameras
    Budget(usize),
    /// Target coverage: find minimum cameras to intercept X% of flows (e.g., 1.0 = 100%)
    TargetCoverage(f64),
}