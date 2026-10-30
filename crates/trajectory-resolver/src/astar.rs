use engine_core::{geo::haversine_distance, CityGraph, NodeId};
use ordered_float::OrderedFloat;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Clone, PartialEq, Eq)]
struct SearchState {
    f_score: OrderedFloat<f64>,
    g_cost: OrderedFloat<f64>,
    dist_m: OrderedFloat<f64>,
    time_sec: OrderedFloat<f64>,
    curr_node: NodeIndex,
    prev_node: Option<NodeIndex>,
    parent_entry: Option<usize>, // Points into arena for zero-copy path tracking
    signals: u32,
}

impl Ord for SearchState {
    fn cmp(&self, other: &Self) -> Ordering {
        other.f_score.cmp(&self.f_score)
    }
}

impl PartialOrd for SearchState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Compact node arena entry for zero-allocation backtracking
struct ArenaEntry {
    node: NodeIndex,
    parent: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct TrajectoryCandidate {
    pub probability: f64,
    pub path: Vec<NodeId>,
}

pub fn resolve_edge_gap_probabilistic(
    city: &CityGraph,
    exit_node_id: NodeId,
    entry_node_id: NodeId,
    dt_seconds: f64,
    max_candidates: usize,
) -> Vec<TrajectoryCandidate> {
    if exit_node_id == entry_node_id {
        return vec![TrajectoryCandidate {
            probability: 100.0,
            path: vec![exit_node_id],
        }];
    }

    let start_idx = match city.node_map.get(&exit_node_id) {
        Some(&idx) => idx,
        None => return vec![],
    };
    let target_idx = match city.node_map.get(&entry_node_id) {
        Some(&idx) => idx,
        None => return vec![],
    };

    let target_meta = &city.graph[target_idx];
    let mut pq = BinaryHeap::with_capacity(256);
    let mut arena: Vec<ArenaEntry> = Vec::with_capacity(1024);
    let mut raw_candidates: Vec<(f64, usize)> = Vec::with_capacity(max_candidates);
    let mut best_costs: HashMap<NodeIndex, f64> = HashMap::with_capacity(512);

    // Initial arena root
    arena.push(ArenaEntry {
        node: start_idx,
        parent: None,
    });
    let root_entry = 0;

    // 1. Initial outgoing edges
    for edge in city.graph.edges_directed(start_idx, Direction::Outgoing) {
        let next_idx = edge.target();
        let meta = edge.weight();
        let next_meta = &city.graph[next_idx];

        let travel_time = meta.length_m / meta.highway_type.speed_mps();
        let g_init = meta.length_m * meta.highway_type.hierarchy_weight();
        let h = haversine_distance(next_meta.lat, next_meta.lon, target_meta.lat, target_meta.lon);

        let entry_idx = arena.len();
        arena.push(ArenaEntry {
            node: next_idx,
            parent: Some(root_entry),
        });

        pq.push(SearchState {
            f_score: OrderedFloat(g_init + h * 0.8),
            g_cost: OrderedFloat(g_init),
            dist_m: OrderedFloat(meta.length_m),
            time_sec: OrderedFloat(travel_time),
            curr_node: next_idx,
            prev_node: Some(start_idx),
            parent_entry: Some(entry_idx),
            signals: 0,
        });
    }

    let mut expansions = 0;
    const MAX_EXPANSIONS: usize = 2000;

    while let Some(state) = pq.pop() {
        expansions += 1;
        if expansions > MAX_EXPANSIONS || raw_candidates.len() >= max_candidates {
            break;
        }

        if state.curr_node == target_idx {
            let idle_sec = (dt_seconds - state.time_sec.0).max(0.0);
            let max_idle = (state.signals as f64 * 70.0).max(30.0);
            let time_penalty = if state.time_sec.0 <= dt_seconds * 1.30 {
                (idle_sec / max_idle.max(1.0) * 50.0).min(120.0)
            } else {
                300.0
            };

            if let Some(entry) = state.parent_entry {
                raw_candidates.push((state.g_cost.0 + time_penalty, entry));
            }
            continue;
        }

        if let Some(&prior_cost) = best_costs.get(&state.curr_node) {
            // Keep a tiny margin so competitive parallel avenues aren't choked
            if prior_cost <= state.g_cost.0 - 25.0 {
                continue;
            }
        }
        best_costs.insert(state.curr_node, state.g_cost.0);

        let is_signal = city.graph.edges_directed(state.curr_node, Direction::Outgoing).count() >= 3;

        for edge in city.graph.edges_directed(state.curr_node, Direction::Outgoing) {
            let next_idx = edge.target();

            // Prevent immediate U-turn turnaround on the exact same road
            if Some(next_idx) == state.prev_node {
                continue;
            }

            let meta = edge.weight();
            let next_meta = &city.graph[next_idx];
            let step_cost = meta.length_m * meta.highway_type.hierarchy_weight();
            let new_g = state.g_cost.0 + step_cost;
            let new_dist = state.dist_m.0 + meta.length_m;
            let new_time = state.time_sec.0 + (meta.length_m / meta.highway_type.speed_mps());
            let new_signals = state.signals + if is_signal { 1 } else { 0 };

            let h = haversine_distance(next_meta.lat, next_meta.lon, target_meta.lat, target_meta.lon);

            let new_entry_idx = arena.len();
            arena.push(ArenaEntry {
                node: next_idx,
                parent: state.parent_entry,
            });

            pq.push(SearchState {
                f_score: OrderedFloat(new_g + h * 0.8),
                g_cost: OrderedFloat(new_g),
                dist_m: OrderedFloat(new_dist),
                time_sec: OrderedFloat(new_time),
                curr_node: next_idx,
                prev_node: Some(state.curr_node),
                parent_entry: Some(new_entry_idx),
                signals: new_signals,
            });
        }
    }

    if raw_candidates.is_empty() {
        return vec![TrajectoryCandidate {
            probability: 100.0,
            path: vec![exit_node_id, entry_node_id],
        }];
    }

    // Sort by best score
    raw_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let min_cost = raw_candidates[0].0;
    let temp = 100.0;

    let exps: Vec<f64> = raw_candidates
        .iter()
        .map(|(cost, _)| (-(cost - min_cost) / temp).exp())
        .collect();
    let sum_exps: f64 = exps.iter().sum();

    // Reconstruct full paths backwards from the arena only for chosen candidates
    let mut results = Vec::with_capacity(raw_candidates.len());
    for (i, (_, entry_idx)) in raw_candidates.into_iter().enumerate() {
        let mut path_rev = Vec::new();
        let mut curr_ptr = Some(entry_idx);

        while let Some(ptr) = curr_ptr {
            path_rev.push(city.reverse_node_map[&arena[ptr].node]);
            curr_ptr = arena[ptr].parent;
        }

        path_rev.reverse();
        let prob = ((exps[i] / sum_exps) * 1000.0).round() / 10.0;

        results.push(TrajectoryCandidate {
            probability: prob,
            path: path_rev,
        });
    }

    results
}