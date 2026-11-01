use crate::kinematics::{min_physical_traversal_time, structural_delay_capacity};
use ahash::AHashMap;
use engine_core::{geo::haversine_distance, CityGraph, NodeId};
use ordered_float::OrderedFloat;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, PartialEq, Eq)]
struct SearchState {
    f_score: OrderedFloat<f64>,
    min_time_sec: OrderedFloat<f64>,
    dist_m: OrderedFloat<f64>,
    delay_capacity: OrderedFloat<f64>,
    curr_node: NodeIndex,
    prev_node: Option<NodeIndex>,
    prev_bearing: Option<OrderedFloat<f64>>,
    parent_entry: Option<usize>,
    depth: usize,
}

impl Ord for SearchState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap on f_score
        other.f_score.cmp(&self.f_score)
    }
}

impl PartialOrd for SearchState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct ArenaEntry {
    node: NodeIndex,
    parent: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct TrajectoryCandidate {
    pub probability: f64,
    pub path: Vec<NodeId>,
    pub min_possible_seconds: f64,
    pub distance_m: f64,
}

pub fn resolve_edge_gap_probabilistic(
    city: &CityGraph,
    exit_node_id: NodeId,
    entry_node_id: NodeId,
    dt_seconds: f64,
    _speed_at_exit_kmh: f64,
    _speed_at_entry_kmh: f64,
    max_candidates: usize,
) -> Vec<TrajectoryCandidate> {
    let start_idx = match city.node_map.get(&exit_node_id) {
        Some(&idx) => idx,
        None => return vec![],
    };
    let target_idx = match city.node_map.get(&entry_node_id) {
        Some(&idx) => idx,
        None => return vec![],
    };

    // 1. Contiguous transition (Camera A exit IS Camera B entry)
    if exit_node_id == entry_node_id {
        return vec![TrajectoryCandidate {
            probability: 100.0,
            path: vec![exit_node_id],
            min_possible_seconds: 0.0,
            distance_m: 0.0,
        }];
    }

    // 2. Direct single-edge connection
    if let Some(edge) = city.graph.find_edge(start_idx, target_idx) {
        let meta = &city.graph[edge];
        let min_t = min_physical_traversal_time(meta, None);

        // If the car took at least the minimum physical time (with slight tolerance), direct edge is valid
        if dt_seconds >= min_t * 0.75 {
            return vec![TrajectoryCandidate {
                probability: 100.0,
                path: vec![exit_node_id, entry_node_id],
                min_possible_seconds: min_t,
                distance_m: meta.length_m,
            }];
        }
    }

    let target_meta = &city.graph[target_idx];
    let mut pq = BinaryHeap::with_capacity(1024);
    let mut arena: Vec<ArenaEntry> = Vec::with_capacity(8192);
    let mut raw_candidates: Vec<(f64, f64, f64, usize)> = Vec::with_capacity(max_candidates);
    let mut best_dist: AHashMap<NodeIndex, f64> = AHashMap::with_capacity(4096);

    arena.push(ArenaEntry {
        node: start_idx,
        parent: None,
    });
    let root_entry = 0;

    for edge in city.graph.edges_directed(start_idx, Direction::Outgoing) {
        let next_idx = edge.target();
        let meta = edge.weight();
        let next_meta = &city.graph[next_idx];
        let out_deg = city.graph.edges_directed(next_idx, Direction::Outgoing).count();

        let t_min_edge = min_physical_traversal_time(meta, None);
        let cap = structural_delay_capacity(next_meta, out_deg);
        let h_dist = haversine_distance(next_meta.lat, next_meta.lon, target_meta.lat, target_meta.lon);

        let entry_idx = arena.len();
        arena.push(ArenaEntry {
            node: next_idx,
            parent: Some(root_entry),
        });

        let g_init = meta.length_m;
        pq.push(SearchState {
            f_score: OrderedFloat(g_init + h_dist),
            min_time_sec: OrderedFloat(t_min_edge),
            dist_m: OrderedFloat(meta.length_m),
            delay_capacity: OrderedFloat(cap),
            curr_node: next_idx,
            prev_node: Some(start_idx),
            prev_bearing: Some(OrderedFloat(meta.bearing_deg)),
            parent_entry: Some(entry_idx),
            depth: 1,
        });
    }

    let mut expansions = 0;
    const MAX_EXPANSIONS: usize = 20_000;

    while let Some(state) = pq.pop() {
        expansions += 1;
        if expansions > MAX_EXPANSIONS {
            break;
        }

        // Causal lower-bound check:
        // A path is only impossible if its MINIMUM free-flow speed time exceeds dt.
        // It is NEVER pruned for being too fast (because vehicles stop, wait at lights, or crawl in traffic).
        if state.min_time_sec.0 > (dt_seconds * 1.30).max(60.0) {
            continue;
        }

        if state.curr_node == target_idx {
            let min_t = state.min_time_sec.0;
            let dist = state.dist_m.0;
            let excess_delay = (dt_seconds - min_t).max(0.0);

            // Likelihood scoring:
            // 1. Shorter physical distance is favored
            // 2. Excess delay should be supported by delay capacity (signals, bumps, intersections)
            let capacity = state.delay_capacity.0.max(1.0);
            let delay_plausibility = excess_delay / (capacity * 25.0);
            let score = dist + delay_plausibility * 8.0;

            if let Some(entry) = state.parent_entry {
                raw_candidates.push((score, min_t, dist, entry));
            }

            if raw_candidates.len() >= max_candidates {
                break;
            }
            continue;
        }

        if let Some(&prior_dist) = best_dist.get(&state.curr_node) {
            if prior_dist <= state.dist_m.0 {
                continue;
            }
        }
        best_dist.insert(state.curr_node, state.dist_m.0);

        for edge in city.graph.edges_directed(state.curr_node, Direction::Outgoing) {
            let next_idx = edge.target();

            // Prevent direct U-turn backtrack
            if Some(next_idx) == state.prev_node {
                continue;
            }

            let meta = edge.weight();
            let next_meta = &city.graph[next_idx];
            let out_deg = city.graph.edges_directed(next_idx, Direction::Outgoing).count();
            let pb = state.prev_bearing.map(|b| b.0);

            let t_min_step = min_physical_traversal_time(meta, pb);
            let new_min_t = state.min_time_sec.0 + t_min_step;

            // Physical causality check
            if new_min_t > (dt_seconds * 1.30).max(60.0) {
                continue;
            }

            let new_dist = state.dist_m.0 + meta.length_m;
            let new_cap = state.delay_capacity.0 + structural_delay_capacity(next_meta, out_deg);
            let h_dist = haversine_distance(next_meta.lat, next_meta.lon, target_meta.lat, target_meta.lon);

            let new_entry_idx = arena.len();
            arena.push(ArenaEntry {
                node: next_idx,
                parent: state.parent_entry,
            });

            pq.push(SearchState {
                f_score: OrderedFloat(new_dist + h_dist),
                min_time_sec: OrderedFloat(new_min_t),
                dist_m: OrderedFloat(new_dist),
                delay_capacity: OrderedFloat(new_cap),
                curr_node: next_idx,
                prev_node: Some(state.curr_node),
                prev_bearing: Some(OrderedFloat(meta.bearing_deg)),
                parent_entry: Some(new_entry_idx),
                depth: state.depth + 1,
            });
        }
    }

    // Fallback: If no candidate was found within bounds, perform standard Dijkstra/A*
    if raw_candidates.is_empty() {
        if let Some(path) = petgraph::algo::astar(
            &city.graph,
            start_idx,
            |finish| finish == target_idx,
            |e| e.weight().length_m,
            |n| {
                let m = &city.graph[n];
                haversine_distance(m.lat, m.lon, target_meta.lat, target_meta.lon)
            },
        ) {
            let node_ids: Vec<NodeId> =
                path.1.into_iter().map(|idx| city.reverse_node_map[&idx]).collect();
            return vec![TrajectoryCandidate {
                probability: 100.0,
                path: node_ids,
                min_possible_seconds: 0.0,
                distance_m: path.0,
            }];
        }
        return vec![];
    }

    // Softmax probabilities
    raw_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let min_score = raw_candidates[0].0;

    let exps: Vec<f64> = raw_candidates
        .iter()
        .map(|(score, _, _, _)| (-(score - min_score) / 50.0).exp())
        .collect();
    let sum_exps: f64 = exps.iter().sum();

    let mut results = Vec::with_capacity(raw_candidates.len());
    for (i, (_, min_t, dist, entry_idx)) in raw_candidates.into_iter().enumerate() {
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
            min_possible_seconds: min_t,
            distance_m: dist,
        });
    }

    results
}