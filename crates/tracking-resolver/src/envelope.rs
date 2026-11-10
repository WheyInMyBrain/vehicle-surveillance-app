use ahash::{AHashMap, AHashSet};
use ordered_float::OrderedFloat;
use road_graph_loader::RoadGraph;
use road_network_core::geo::fast_planar_distance;
use road_network_core::NodeId;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::config::ResolverConfig;
use crate::kinematics::edge_velocity_ceiling_mps;

#[derive(Clone, PartialEq, Eq)]
struct IsochroneState {
    f_cost: OrderedFloat<f32>,
    g_cost: OrderedFloat<f32>,
    node: NodeId,
}

impl Ord for IsochroneState {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        other.f_cost.cmp(&self.f_cost)
    }
}

impl PartialOrd for IsochroneState {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
pub fn resolve_node_coordinate(graph: &RoadGraph, node: NodeId) -> Option<(f64, f64)> {
    let outgoing = graph.outgoing_edges(node);
    if let Some(&(_, edge_idx)) = outgoing.first() {
        let desc = graph.edge_descriptor(edge_idx);
        if !desc.geometry.is_empty() {
            return Some((desc.geometry[0][0], desc.geometry[0][1]));
        }
    }
    None
}

/// Computes the admissible space-time subgraph using forward isochrones and target distance heuristics
pub fn compute_bidirectional_isochrone_hull(
    graph: &RoadGraph,
    source_node: NodeId,
    target_node: NodeId,
    dt_seconds: f32,
    config: &ResolverConfig,
) -> AHashSet<usize> {
    let time_budget = (dt_seconds * config.isochrone_time_budget_multiplier).max(45.0);
    let target_coord = resolve_node_coordinate(graph, target_node);

    let mut min_g: AHashMap<NodeId, f32> = AHashMap::with_capacity(1024);
    let mut valid_edges: AHashSet<usize> = AHashSet::with_capacity(512);
    let mut pq = BinaryHeap::new();

    min_g.insert(source_node, 0.0);
    pq.push(IsochroneState {
        f_cost: OrderedFloat(0.0),
        g_cost: OrderedFloat(0.0),
        node: source_node,
    });

    while let Some(IsochroneState { g_cost, node, .. }) = pq.pop() {
        if g_cost.0 > time_budget {
            continue;
        }

        for &(next_node, edge_idx) in graph.outgoing_edges(node) {
            let desc = graph.edge_descriptor(edge_idx);
            let step_time = desc.length_m / edge_velocity_ceiling_mps(desc, config);
            let next_g = g_cost.0 + step_time;

            // Calculate admissible lower-bound remaining time to target node
            let h_time = match (target_coord, desc.geometry.last()) {
                (Some((lat_t, lon_t)), Some(pt)) => {
                    let dist_rem = fast_planar_distance(pt[0], pt[1], lat_t, lon_t) as f32;
                    dist_rem / config.max_corridor_speed_mps
                }
                _ => 0.0,
            };

            if next_g + h_time <= time_budget {
                valid_edges.insert(edge_idx);

                if next_g < *min_g.get(&next_node).unwrap_or(&f32::INFINITY) {
                    min_g.insert(next_node, next_g);
                    pq.push(IsochroneState {
                        f_cost: OrderedFloat(next_g + h_time),
                        g_cost: OrderedFloat(next_g),
                        node: next_node,
                    });
                }
            }
        }
    }

    valid_edges
}