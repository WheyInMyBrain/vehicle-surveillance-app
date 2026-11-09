pub mod flow;
pub mod time;

pub use flow::{calculate_traversal_cost, interpolate_flow, TraversalStatus};
pub use road_graph_loader::{EdgeDescriptor, LinkFlowSnapshot, RoadGraph};
pub use time::TemporalCoord;

use chrono::{DateTime, Utc};
use road_network_core::NodeId;

pub struct TrafficTensorEngine<'a> {
    graph: &'a RoadGraph,
}

impl<'a> TrafficTensorEngine<'a> {
    #[inline(always)]
    pub fn new(graph: &'a RoadGraph) -> Self {
        Self { graph }
    }

    /// Evaluates edge flow with smooth intra-bin linear interpolation at a specific timestamp
    #[inline]
    pub fn evaluate_edge_at(
        &self,
        edge_idx: usize,
        timestamp: &DateTime<Utc>,
    ) -> TraversalStatus {
        let coord = TemporalCoord::from_datetime(timestamp);
        let (next_day, next_bin) = coord.next_bin();

        let curr = self.graph.get_flow_by_index(edge_idx, coord.day, coord.bin);
        let next = self.graph.get_flow_by_index(edge_idx, next_day, next_bin);
        let desc = self.graph.edge_descriptor(edge_idx);

        calculate_traversal_cost(desc.length_m, desc.maxspeed_kmh, &curr, &next, &coord)
    }

    /// Convenience lookup for (u, v)
    #[inline]
    pub fn evaluate_traversal(
        &self,
        u: NodeId,
        v: NodeId,
        timestamp: &DateTime<Utc>,
    ) -> Option<TraversalStatus> {
        let edge_idx = self.graph.get_edge_index(u, v)?;
        Some(self.evaluate_edge_at(edge_idx, timestamp))
    }

    /// Minimum physical duration (in seconds) across an edge assuming free-flow/uncapped limit.
    #[inline]
    pub fn min_physical_duration(&self, edge_idx: usize, timestamp: &DateTime<Utc>) -> f32 {
        let coord = TemporalCoord::from_datetime(timestamp);
        let flow = self.graph.get_flow_by_index(edge_idx, coord.day, coord.bin);
        let desc = self.graph.edge_descriptor(edge_idx);

        let max_speed_kmh = (flow.free_flow_kmh.max(desc.maxspeed_kmh)) as f32;
        let speed_mps = (max_speed_kmh / 3.6).max(1.0);
        desc.length_m / speed_mps
    }

    /// Direct access to the physical edge descriptor
    #[inline(always)]
    pub fn edge_descriptor(&self, edge_idx: usize) -> &EdgeDescriptor {
        self.graph.edge_descriptor(edge_idx)
    }

    /// Direct access to the raw flow snapshot
    #[inline(always)]
    pub fn flow_snapshot_at(&self, edge_idx: usize, coord: &TemporalCoord) -> LinkFlowSnapshot {
        self.graph.get_flow_by_index(edge_idx, coord.day, coord.bin)
    }

    #[inline(always)]
    pub fn graph(&self) -> &RoadGraph {
        self.graph
    }
}