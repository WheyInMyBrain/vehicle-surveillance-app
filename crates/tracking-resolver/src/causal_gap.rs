use ahash::AHashSet;
use chrono::{DateTime, Utc};
use road_graph_loader::{RoadGraph, UnifiedHighwayTier};
use road_network_core::NodeId;
use road_traffic_tensor::{TemporalCoord, TrafficTensorEngine};

use crate::config::ResolverConfig;
use crate::consensus::{fuse_bayesian_consensus, EvaluatedTrajectory};
use crate::ensemble::generate_ensemble_candidates;
use crate::envelope::compute_bidirectional_isochrone_hull;
use crate::kinematics::{
    compute_path_min_physical_duration, min_physical_edge_duration, resolve_legal_maxspeed_kmh,
};
use crate::likelihood::{
    evaluate_dwell_mixture, hierarchy_transition_penalty, junction_deflection_deg,
    junction_delay_and_penalty, negative_camera_penalty,
};

#[derive(Debug, Clone)]
pub struct TrajectoryCandidate {
    pub probability: f32,
    pub path: Vec<NodeId>,
    pub min_possible_seconds: f32,
    pub expected_tensor_seconds: f32,
    pub distance_m: f32,
    pub dark_alley_ratio: f32,
    pub transition_log_likelihood: f32,
    pub models_agreed: Vec<&'static str>,
    pub is_dwell_detected: bool,
    pub excess_dwell_sec: f32,
}

pub struct CausalGapWorkspace {
    pub unobserved_camera_edges: AHashSet<usize>,
    pub config: ResolverConfig,
}

impl CausalGapWorkspace {
    pub fn new() -> Self {
        Self {
            unobserved_camera_edges: AHashSet::with_capacity(64),
            config: ResolverConfig::default(),
        }
    }

    pub fn with_config(config: ResolverConfig) -> Self {
        Self {
            unobserved_camera_edges: AHashSet::with_capacity(64),
            config,
        }
    }

    pub fn clear(&mut self) {
        self.unobserved_camera_edges.clear();
    }
}

impl Default for CausalGapWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

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
    let config = &workspace.config;

    // Trivial identical node fast-path
    if exit_node == entry_node {
        let is_dwell = dt_seconds > config.dwell_min_slack_sec;
        let excess_sec = if is_dwell { dt_seconds } else { 0.0 };
        return vec![TrajectoryCandidate {
            probability: 100.0,
            path: vec![exit_node],
            min_possible_seconds: 0.0,
            expected_tensor_seconds: 0.0,
            distance_m: 0.0,
            dark_alley_ratio: 0.0,
            transition_log_likelihood: if is_dwell { -1.0 } else { 0.0 },
            models_agreed: vec!["DirectTrivial"],
            is_dwell_detected: is_dwell,
            excess_dwell_sec: excess_sec,
        }];
    }

    let coord = TemporalCoord::from_datetime(start_time);
    let (prior_day, prior_bin) = coord.prior_completed_bin();

    // Direct single edge fast-path with dwell mixture evaluation
    if let Some(edge_idx) = graph.get_edge_index(exit_node, entry_node) {
        let desc = graph.edge_descriptor(edge_idx);
        let snapshot = tensor.graph().get_flow_by_index(edge_idx, prior_day, prior_bin);
        let min_t = min_physical_edge_duration(desc, snapshot.free_flow_kmh, config);
        let flow_kmh = (snapshot.speed_kmh as f32).min(resolve_legal_maxspeed_kmh(desc)).max(6.0);
        let exp_t = desc.length_m / (flow_kmh / 3.6);

        let variance = (exp_t * (1.2 - snapshot.confidence as f32)).powi(2) + 36.0;
        let (is_dwell, excess_sec, ln_p_time) =
            evaluate_dwell_mixture(dt_seconds, exp_t, variance.sqrt(), config);

        let is_minor = matches!(
            desc.tier,
            UnifiedHighwayTier::Residential
                | UnifiedHighwayTier::Service
                | UnifiedHighwayTier::LivingStreet
        );

        return vec![TrajectoryCandidate {
            probability: 100.0,
            path: vec![exit_node, entry_node],
            min_possible_seconds: min_t,
            expected_tensor_seconds: exp_t,
            distance_m: desc.length_m,
            dark_alley_ratio: if is_minor { 1.0 } else { 0.0 },
            transition_log_likelihood: ln_p_time.max(-30.0),
            models_agreed: vec!["DirectEdge"],
            is_dwell_detected: is_dwell,
            excess_dwell_sec: excess_sec,
        }];
    }

    // Layer 1: Isochrone Filter
    let mut valid_edges = compute_bidirectional_isochrone_hull(graph, exit_node, entry_node, dt_seconds, config);
    
    // Fallback: If isochrone hull is empty (e.g. extreme speeding anomaly), widen the search
    if valid_edges.is_empty() {
        valid_edges = compute_bidirectional_isochrone_hull(graph, exit_node, entry_node, dt_seconds * 3.0 + 60.0, config);
    }
    if valid_edges.is_empty() {
        return Vec::new();
    }

    let additional_k = (dt_seconds / config.k_expansion_interval_sec.max(1.0)).floor() as usize;
    let dynamic_k = (config.min_k_candidates + additional_k).min(config.max_k_candidates);

    // Layer 2: Ensemble Generation with dynamic K
    let ensemble_routes = generate_ensemble_candidates(
        graph,
        tensor,
        &valid_edges,
        exit_node,
        entry_node,
        prior_day,
        prior_bin,
        dynamic_k,
        config,
    );

    // Layer 3: Physical & Real-World Evaluation
    let mut intermediate_eval = Vec::with_capacity(ensemble_routes.len());

    for cand in ensemble_routes {
        let edge_descs: Vec<_> = cand.edge_indices.iter().map(|&idx| graph.edge_descriptor(idx)).collect();
        let min_physical = compute_path_min_physical_duration(&edge_descs, 8.0, 8.0, config);

        let mut total_distance = 0.0f32;
        let mut expected_time = 0.0f32;
        let mut variance = 36.0f32;
        let mut turn_penalty_total = 0.0f32;
        let mut hierarchy_penalty_total = 0.0f32;
        let mut dark_dist = 0.0f32;
        let mut unobserved_cameras = 0;

        for i in 0..edge_descs.len() {
            let desc = edge_descs[i];
            let edge_idx = cand.edge_indices[i];
            let snapshot = tensor.graph().get_flow_by_index(edge_idx, prior_day, prior_bin);
            let speed_mps = ((snapshot.speed_kmh as f32).min(resolve_legal_maxspeed_kmh(desc)).max(6.0)) / 3.6;

            total_distance += desc.length_m;
            let t_e = desc.length_m / speed_mps;
            expected_time += t_e;
            variance += (t_e * (1.2 - snapshot.confidence as f32)).powi(2);

            if matches!(
                desc.tier,
                UnifiedHighwayTier::Residential
                    | UnifiedHighwayTier::Service
                    | UnifiedHighwayTier::LivingStreet
            ) {
                dark_dist += desc.length_m;
            }

            if workspace.unobserved_camera_edges.contains(&edge_idx) {
                unobserved_cameras += 1;
            }

            if i < edge_descs.len() - 1 {
                let next_desc = edge_descs[i + 1];
                let deflection = junction_deflection_deg(desc, next_desc);
                let (j_delay, j_pen) = junction_delay_and_penalty(deflection, snapshot.jam_factor, config);
                expected_time += j_delay;
                turn_penalty_total += j_pen;
                hierarchy_penalty_total += hierarchy_transition_penalty(desc.tier, next_desc.tier);
            }
        }

        let (is_dwell, excess_sec, ln_p_time) =
            evaluate_dwell_mixture(dt_seconds, expected_time, variance.sqrt(), config);
        let camera_pen = negative_camera_penalty(unobserved_cameras, config);
        let dark_ratio = if total_distance > 0.0 { dark_dist / total_distance } else { 0.0 };

        // Soft penalization for speed limit violations instead of hard rejection,
        // so excessive speeding reaches evaluate_leg_anomaly.
        let speed_violation_pen = if dt_seconds < min_physical * 0.85 {
            -25.0
        } else {
            0.0
        };

        let log_likelihood = ln_p_time
            + turn_penalty_total
            + hierarchy_penalty_total
            + camera_pen
            + speed_violation_pen
            - (dark_ratio * 3.0);

        intermediate_eval.push((
            cand,
            log_likelihood,
            min_physical,
            expected_time,
            total_distance,
            dark_ratio,
            is_dwell,
            excess_sec,
        ));
    }

    // Layer 4: Bayesian Consensus
    let fused = fuse_bayesian_consensus(intermediate_eval, config);

    fused
        .into_iter()
        .take(max_candidates)
        .map(|r: EvaluatedTrajectory| TrajectoryCandidate {
            probability: r.probability_percent,
            path: r.candidate.node_path,
            min_possible_seconds: r.min_physical_sec,
            expected_tensor_seconds: r.expected_tensor_sec,
            distance_m: r.distance_m,
            dark_alley_ratio: r.dark_alley_ratio,
            transition_log_likelihood: r.log_likelihood.max(-50.0),
            models_agreed: r.candidate.models_agreed,
            is_dwell_detected: r.is_dwell_detected,
            excess_dwell_sec: r.excess_dwell_sec,
        })
        .collect()
}