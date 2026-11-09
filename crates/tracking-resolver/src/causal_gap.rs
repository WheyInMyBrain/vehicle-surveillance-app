use crate::kinematics::{min_physical_edge_duration, structural_delay_capacity};
use ahash::AHashMap;
use chrono::{DateTime, Duration, Utc};
use ordered_float::OrderedFloat;
use road_graph_loader::{RoadGraph, UnifiedHighwayTier};
use road_network_core::geo::fast_planar_distance;
use road_network_core::NodeId;
use road_traffic_tensor::{TrafficTensorEngine, TraversalStatus};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, PartialEq, Eq)]
struct CausalSearchState {
    f_score: OrderedFloat<f32>,
    min_time_sec: OrderedFloat<f32>,
    expected_tensor_sec: OrderedFloat<f32>,
    dist_m: OrderedFloat<f32>,
    delay_capacity_sec: OrderedFloat<f32>,
    dark_alley_dist_m: OrderedFloat<f32>,
    curr_node: NodeId,
    prev_node: Option<NodeId>,
    arena_entry: usize,
    depth: usize,
}

impl Ord for CausalSearchState {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        other.f_score.cmp(&self.f_score)
    }
}

impl PartialOrd for CausalSearchState {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct ArenaNode {
    node_id: NodeId,
    parent: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct TrajectoryCandidate {
    pub probability: f32,
    pub path: Vec<NodeId>,
    pub min_possible_seconds: f32,
    pub expected_tensor_seconds: f32,
    pub distance_m: f32,
    pub dark_alley_ratio: f32,
    pub transition_log_likelihood: f32,
}

pub struct CausalGapWorkspace {
    pq: BinaryHeap<CausalSearchState>,
    arena: Vec<ArenaNode>,
    best_dist: AHashMap<NodeId, f32>,
    raw_candidates: Vec<(f32, f32, f32, f32, f32, usize)>,
}

impl CausalGapWorkspace {
    pub fn new() -> Self {
        Self {
            pq: BinaryHeap::with_capacity(512),
            arena: Vec::with_capacity(2048),
            best_dist: AHashMap::with_capacity(1024),
            raw_candidates: Vec::with_capacity(8),
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.pq.clear();
        self.arena.clear();
        self.best_dist.clear();
        self.raw_candidates.clear();
    }
}

impl Default for CausalGapWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
pub fn resolve_node_lat_lon(graph: &RoadGraph, node: NodeId) -> Option<(f64, f64)> {
    let outgoing = graph.outgoing_edges(node);
    if let Some(&(_, edge_idx)) = outgoing.first() {
        let desc = graph.edge_descriptor(edge_idx);
        if !desc.geometry.is_empty() {
            return Some((desc.geometry[0][0], desc.geometry[0][1]));
        }
    }
    None
}

/// Retrospectively resolves unmonitored trajectory candidates between two sightings
pub fn resolve_camera_gap(
    graph: &RoadGraph,
    tensor: &TrafficTensorEngine,
    exit_node: NodeId,
    entry_node: NodeId,
    start_time: &DateTime<Utc>,
    dt_seconds: f32,
    max_candidates: usize,
    workspace: &mut CausalGapWorkspace,
) -> Vec<TrajectoryCandidate> {
    if exit_node == entry_node {
        return vec![TrajectoryCandidate {
            probability: 100.0,
            path: vec![exit_node],
            min_possible_seconds: 0.0,
            expected_tensor_seconds: 0.0,
            distance_m: 0.0,
            dark_alley_ratio: 0.0,
            transition_log_likelihood: 0.0,
        }];
    }

    // Direct single edge connection
    if let Some(edge_idx) = graph.get_edge_index(exit_node, entry_node) {
        let desc = graph.edge_descriptor(edge_idx);
        let snapshot = tensor.flow_snapshot_at(
            edge_idx,
            &road_traffic_tensor::TemporalCoord::from_datetime(start_time),
        );
        let min_t = min_physical_edge_duration(desc, snapshot.free_flow_kmh);
        let expected_t = desc.length_m / (snapshot.speed_kmh as f32 / 3.6).max(1.0);

        if dt_seconds >= min_t * 0.70 {
            let is_minor = matches!(
                desc.tier,
                UnifiedHighwayTier::Residential
                    | UnifiedHighwayTier::Service
                    | UnifiedHighwayTier::LivingStreet
            );
            let diff = (dt_seconds - expected_t).abs();
            let log_l = -0.5 * (diff / 12.0).powi(2);
            return vec![TrajectoryCandidate {
                probability: 100.0,
                path: vec![exit_node, entry_node],
                min_possible_seconds: min_t,
                expected_tensor_seconds: expected_t,
                distance_m: desc.length_m,
                dark_alley_ratio: if is_minor { 1.0 } else { 0.0 },
                transition_log_likelihood: log_l.max(-15.0),
            }];
        }
    }

    workspace.clear();
    let target_coord_opt = resolve_node_lat_lon(graph, entry_node);

    workspace.arena.push(ArenaNode {
        node_id: exit_node,
        parent: None,
    });
    let root_idx = 0;

    for &(next_node, edge_idx) in graph.outgoing_edges(exit_node) {
        let status = tensor.evaluate_edge_at(edge_idx, start_time);
        if let TraversalStatus::Open { speed_mps, jam_factor, .. } = status {
            let desc = graph.edge_descriptor(edge_idx);
            let h_dist = match (target_coord_opt, desc.geometry.last()) {
                (Some((lat2, lon2)), Some(p)) => fast_planar_distance(p[0], p[1], lat2, lon2) as f32,
                _ => 0.0,
            };

            let snapshot = tensor.flow_snapshot_at(
                edge_idx,
                &road_traffic_tensor::TemporalCoord::from_datetime(start_time),
            );
            let min_t = min_physical_edge_duration(desc, snapshot.free_flow_kmh);
            let expected_t = desc.length_m / speed_mps.max(1.0) * (1.0 + jam_factor * 0.06);
            let cap = structural_delay_capacity(graph.outgoing_edges(next_node).len());
            let is_minor = matches!(
                desc.tier,
                UnifiedHighwayTier::Residential
                    | UnifiedHighwayTier::Service
                    | UnifiedHighwayTier::LivingStreet
            );

            let entry_idx = workspace.arena.len();
            workspace.arena.push(ArenaNode {
                node_id: next_node,
                parent: Some(root_idx),
            });

            workspace.pq.push(CausalSearchState {
                f_score: OrderedFloat(desc.length_m + h_dist),
                min_time_sec: OrderedFloat(min_t),
                expected_tensor_sec: OrderedFloat(expected_t),
                dist_m: OrderedFloat(desc.length_m),
                delay_capacity_sec: OrderedFloat(cap),
                dark_alley_dist_m: OrderedFloat(if is_minor { desc.length_m } else { 0.0 }),
                curr_node: next_node,
                prev_node: Some(exit_node),
                arena_entry: entry_idx,
                depth: 1,
            });
        }
    }

    let mut expansions = 0;
    const MAX_EXPANSIONS: usize = 12_000;

    while let Some(state) = workspace.pq.pop() {
        expansions += 1;
        if expansions > MAX_EXPANSIONS {
            break;
        }

        // Physical lower bound constraint
        if state.min_time_sec.0 > (dt_seconds * 1.35).max(45.0) {
            continue;
        }

        if state.curr_node == entry_node {
            let min_t = state.min_time_sec.0;
            let expected_t = state.expected_tensor_sec.0;
            let dist = state.dist_m.0;
            let dark_ratio = if dist > 0.0 {
                state.dark_alley_dist_m.0 / dist
            } else {
                0.0
            };

            let excess_delay = (dt_seconds - min_t).max(0.0);
            let delay_ratio = excess_delay / (state.delay_capacity_sec.0 * 15.0).max(1.0);
            let score = dist + delay_ratio * 10.0;

            workspace.raw_candidates.push((
                score,
                min_t,
                expected_t,
                dist,
                dark_ratio,
                state.arena_entry,
            ));

            if workspace.raw_candidates.len() >= max_candidates {
                break;
            }
            continue;
        }

        if let Some(&prior_dist) = workspace.best_dist.get(&state.curr_node) {
            if prior_dist <= state.dist_m.0 {
                continue;
            }
        }
        workspace.best_dist.insert(state.curr_node, state.dist_m.0);

        let curr_time_at_node =
            *start_time + Duration::milliseconds((state.min_time_sec.0 * 1000.0) as i64);

        for &(next_node, edge_idx) in graph.outgoing_edges(state.curr_node) {
            if Some(next_node) == state.prev_node {
                continue;
            }

            let status = tensor.evaluate_edge_at(edge_idx, &curr_time_at_node);
            if let TraversalStatus::Open { speed_mps, jam_factor, .. } = status {
                let desc = graph.edge_descriptor(edge_idx);
                let snapshot = tensor.flow_snapshot_at(
                    edge_idx,
                    &road_traffic_tensor::TemporalCoord::from_datetime(&curr_time_at_node),
                );
                let step_min_t = min_physical_edge_duration(desc, snapshot.free_flow_kmh);
                let new_min_t = state.min_time_sec.0 + step_min_t;

                if new_min_t > (dt_seconds * 1.35).max(45.0) {
                    continue;
                }

                let step_expected_t =
                    desc.length_m / speed_mps.max(1.0) * (1.0 + jam_factor * 0.06);
                let h_dist = match (target_coord_opt, desc.geometry.last()) {
                    (Some((lat2, lon2)), Some(p)) => {
                        fast_planar_distance(p[0], p[1], lat2, lon2) as f32
                    }
                    _ => 0.0,
                };

                let new_dist = state.dist_m.0 + desc.length_m;
                let new_cap = state.delay_capacity_sec.0
                    + structural_delay_capacity(graph.outgoing_edges(next_node).len());
                let is_minor = matches!(
                    desc.tier,
                    UnifiedHighwayTier::Residential
                        | UnifiedHighwayTier::Service
                        | UnifiedHighwayTier::LivingStreet
                );
                let new_dark_dist = state.dark_alley_dist_m.0
                    + if is_minor { desc.length_m } else { 0.0 };

                let new_entry_idx = workspace.arena.len();
                workspace.arena.push(ArenaNode {
                    node_id: next_node,
                    parent: Some(state.arena_entry),
                });

                workspace.pq.push(CausalSearchState {
                    f_score: OrderedFloat(new_dist + h_dist),
                    min_time_sec: OrderedFloat(new_min_t),
                    expected_tensor_sec: OrderedFloat(
                        state.expected_tensor_sec.0 + step_expected_t,
                    ),
                    dist_m: OrderedFloat(new_dist),
                    delay_capacity_sec: OrderedFloat(new_cap),
                    dark_alley_dist_m: OrderedFloat(new_dark_dist),
                    curr_node: next_node,
                    prev_node: Some(state.curr_node),
                    arena_entry: new_entry_idx,
                    depth: state.depth + 1,
                });
            }
        }
    }

    if workspace.raw_candidates.is_empty() {
        return vec![];
    }

    workspace.raw_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let best_score = workspace.raw_candidates[0].0;

    let exps: Vec<f32> = workspace
        .raw_candidates
        .iter()
        .map(|(score, _, _, _, _, _)| (-(score - best_score) / 60.0).exp())
        .collect();
    let sum_exps: f32 = exps.iter().sum();

    let mut results = Vec::with_capacity(workspace.raw_candidates.len());
    for (i, &(_, min_t, expected_t, dist, dark_ratio, entry_idx)) in
        workspace.raw_candidates.iter().enumerate()
    {
        let mut path_rev = Vec::new();
        let mut curr_ptr = Some(entry_idx);

        while let Some(ptr) = curr_ptr {
            path_rev.push(workspace.arena[ptr].node_id);
            curr_ptr = workspace.arena[ptr].parent;
        }

        path_rev.reverse();
        let prob = ((exps[i] / sum_exps) * 100.0).round();

        // Calculate rigorous HMM log-likelihood: Gaussian transition + dark penalty
        let time_err = (dt_seconds - expected_t).abs();
        let sigma = (expected_t * 0.18 + 8.0).max(5.0);
        let log_likelihood = (-0.5 * (time_err / sigma).powi(2)) - (dark_ratio * 2.0);

        results.push(TrajectoryCandidate {
            probability: prob,
            path: path_rev,
            min_possible_seconds: min_t,
            expected_tensor_seconds: expected_t,
            distance_m: dist,
            dark_alley_ratio: dark_ratio,
            transition_log_likelihood: log_likelihood.max(-25.0),
        });
    }

    results
}