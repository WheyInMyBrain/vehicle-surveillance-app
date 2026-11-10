use ahash::{AHashMap, AHashSet};
use ordered_float::OrderedFloat;
use road_graph_loader::{EdgeDescriptor, RoadGraph, UnifiedHighwayTier};
use road_network_core::NodeId;
use road_traffic_tensor::TrafficTensorEngine;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::config::ResolverConfig;
use crate::kinematics::{edge_velocity_ceiling_mps, resolve_legal_maxspeed_kmh};

#[derive(Clone, PartialEq, Eq)]
pub struct SearchState {
    pub cost: OrderedFloat<f32>,
    pub node: NodeId,
}

impl Ord for SearchState {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost)
    }
}
impl PartialOrd for SearchState {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
pub struct RouteCandidate {
    pub node_path: Vec<NodeId>,
    pub edge_indices: Vec<usize>,
    pub models_agreed: Vec<&'static str>,
}

#[inline(always)]
fn dijkstra_subcorridor_masked<F>(
    graph: &RoadGraph,
    valid_edges: &AHashSet<usize>,
    masked_edges: &AHashSet<usize>,
    start: NodeId,
    goal: NodeId,
    mut cost_fn: F,
) -> Option<(Vec<NodeId>, Vec<usize>)>
where
    F: FnMut(usize, &EdgeDescriptor, Option<&EdgeDescriptor>) -> f32,
{
    let mut dist: AHashMap<NodeId, f32> = AHashMap::with_capacity(128);
    let mut prev: AHashMap<NodeId, (NodeId, usize)> = AHashMap::with_capacity(128);
    let mut pq: BinaryHeap<SearchState> = BinaryHeap::with_capacity(128);

    dist.insert(start, 0.0);
    pq.push(SearchState {
        cost: OrderedFloat(0.0),
        node: start,
    });

    while let Some(SearchState { cost, node }) = pq.pop() {
        if node == goal {
            let mut nodes = Vec::with_capacity(24);
            let mut edges = Vec::with_capacity(24);
            let mut curr = goal;
            nodes.push(curr);

            while let Some(&(p_node, e_idx)) = prev.get(&curr) {
                edges.push(e_idx);
                curr = p_node;
                nodes.push(curr);
                if curr == start {
                    break;
                }
            }
            nodes.reverse();
            edges.reverse();
            return Some((nodes, edges));
        }

        if cost.0 > *dist.get(&node).unwrap_or(&f32::INFINITY) {
            continue;
        }

        for &(next_node, edge_idx) in graph.outgoing_edges(node) {
            // Fast O(1) checks without cloning valid_edges
            if !valid_edges.contains(&edge_idx) {
                continue;
            }
            if !masked_edges.is_empty() && masked_edges.contains(&edge_idx) {
                continue;
            }

            let desc = graph.edge_descriptor(edge_idx);
            let prev_desc = prev.get(&node).map(|&(_, e)| graph.edge_descriptor(e));
            let step = cost_fn(edge_idx, desc, prev_desc);
            let next_cost = cost.0 + step;

            let cur_dist = *dist.get(&next_node).unwrap_or(&f32::INFINITY);
            if next_cost < cur_dist {
                dist.insert(next_node, next_cost);
                prev.insert(next_node, (node, edge_idx));
                pq.push(SearchState {
                    cost: OrderedFloat(next_cost),
                    node: next_node,
                });
            }
        }
    }
    None
}

pub fn generate_k_paths<F>(
    graph: &RoadGraph,
    valid_edges: &AHashSet<usize>,
    start: NodeId,
    goal: NodeId,
    k: usize,
    cost_fn: F,
) -> Vec<(Vec<NodeId>, Vec<usize>)>
where
    F: Fn(usize, &EdgeDescriptor, Option<&EdgeDescriptor>) -> f32 + Copy,
{
    let mut paths = Vec::with_capacity(k);
    let empty_mask = AHashSet::new();

    if let Some(p0) = dijkstra_subcorridor_masked(
        graph,
        valid_edges,
        &empty_mask,
        start,
        goal,
        cost_fn,
    ) {
        paths.push(p0);
    } else {
        return paths;
    }

    if k <= 1 {
        return paths;
    }

    let mut masked_edges = AHashSet::with_capacity(8);

    for _ in 1..k {
        let prev_path = &paths[paths.len() - 1];
        let mut best_spur: Option<(Vec<NodeId>, Vec<usize>)> = None;

        for i in 0..prev_path.1.len() {
            let spur_node = prev_path.0[i];
            let root_nodes = &prev_path.0[0..=i];
            let root_edges = &prev_path.1[0..i];

            // Only mask the small set of spur edges (usually 1-3 integers)
            masked_edges.clear();
            for p in &paths {
                if p.1.len() > i && &p.1[0..i] == root_edges {
                    masked_edges.insert(p.1[i]);
                }
            }

            if let Some((spur_nodes, spur_edges)) = dijkstra_subcorridor_masked(
                graph,
                valid_edges,
                &masked_edges,
                spur_node,
                goal,
                cost_fn,
            ) {
                let mut full_nodes = Vec::with_capacity(root_nodes.len() + spur_nodes.len() - 1);
                full_nodes.extend_from_slice(root_nodes);
                full_nodes.extend_from_slice(&spur_nodes[1..]);

                let mut full_edges = Vec::with_capacity(root_edges.len() + spur_edges.len());
                full_edges.extend_from_slice(root_edges);
                full_edges.extend_from_slice(&spur_edges);

                if !paths.iter().any(|(n, _)| n == &full_nodes) {
                    best_spur = Some((full_nodes, full_edges));
                    break;
                }
            }
        }

        if let Some(spur) = best_spur {
            paths.push(spur);
        } else {
            break;
        }
    }
    paths
}

pub fn generate_ensemble_candidates(
    graph: &RoadGraph,
    tensor: &TrafficTensorEngine,
    valid_edges: &AHashSet<usize>,
    start: NodeId,
    goal: NodeId,
    prior_day: usize,
    prior_bin: usize,
    k_per_model: usize,
    config: &ResolverConfig,
) -> Vec<RouteCandidate> {
    let mut candidate_map: AHashMap<Vec<NodeId>, (Vec<usize>, Vec<&'static str>)> =
        AHashMap::with_capacity(k_per_model * 2);

    // 1. Model Alpha: Tensor Dynamic Impedance
    let paths_tensor = generate_k_paths(
        graph,
        valid_edges,
        start,
        goal,
        k_per_model,
        |edge_idx, desc, _| {
            let snapshot = tensor.graph().get_flow_by_index(edge_idx, prior_day, prior_bin);
            let speed_mps = ((snapshot.speed_kmh as f32)
                .min(resolve_legal_maxspeed_kmh(desc))
                .max(6.0))
                / 3.6;
            (desc.length_m / speed_mps) * (1.0 + (snapshot.jam_factor / 10.0).powi(2) * 1.5)
        },
    );

    for (nodes, edges) in paths_tensor {
        candidate_map
            .entry(nodes)
            .or_insert((edges, Vec::new()))
            .1
            .push("Model_A_TensorImpedance");
    }

    // 2. Model Beta: Human Cognitive Hierarchy
    let paths_hierarchy = generate_k_paths(
        graph,
        valid_edges,
        start,
        goal,
        k_per_model.min(2),
        |_, desc, _| {
            let mult = match desc.tier {
                UnifiedHighwayTier::Motorway => 1.0,
                UnifiedHighwayTier::Trunk => 1.1,
                UnifiedHighwayTier::Primary => 1.25,
                UnifiedHighwayTier::Secondary => 1.6,
                UnifiedHighwayTier::Tertiary => 2.2,
                UnifiedHighwayTier::Residential => 3.8,
                UnifiedHighwayTier::Service | UnifiedHighwayTier::LivingStreet => 6.0,
                _ => 2.5,
            };
            desc.length_m * mult
        },
    );

    for (nodes, edges) in paths_hierarchy {
        candidate_map
            .entry(nodes)
            .or_insert((edges, Vec::new()))
            .1
            .push("Model_B_HierarchyInertia");
    }

    // 3. Model Gamma: Lowest Energy / Curvature Free-Flow
    let k_kinematic = (k_per_model.saturating_sub(1)).max(2);
    let paths_kinematic = generate_k_paths(
        graph,
        valid_edges,
        start,
        goal,
        k_kinematic,
        |_, desc, _| desc.length_m / edge_velocity_ceiling_mps(desc, config),
    );

    for (nodes, edges) in paths_kinematic {
        candidate_map
            .entry(nodes)
            .or_insert((edges, Vec::new()))
            .1
            .push("Model_C_KinematicMinimalist");
    }

    candidate_map
        .into_iter()
        .map(|(node_path, (edge_indices, models_agreed))| RouteCandidate {
            node_path,
            edge_indices,
            models_agreed,
        })
        .collect()
}