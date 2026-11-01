use crate::cut::TopologicalCutReport;
use crate::skeleton::SkeletonGraph;
use crate::types::OptimizedCameraPole;
use ahash::{AHashMap, AHashSet};
use engine_core::{CityGraph, NodeId};
use std::collections::BinaryHeap;

#[derive(Debug, Clone)]
pub struct ZeroEscapePrunerConfig {
    /// Maximum allowed unmonitored continuous driving distance in meters (e.g. 450m - 600m)
    pub max_dark_distance_m: f64,
    /// Maximum allowed unmonitored consecutive junction hops (e.g. 4 - 6 junctions)
    pub max_dark_hops: usize,
    /// Enforce the K-Fork rule: an unmonitored junction cannot have >= max_unmonitored_forks unmonitored branches
    pub max_unmonitored_forks: usize,
}

impl Default for ZeroEscapePrunerConfig {
    fn default() -> Self {
        Self {
            max_dark_distance_m: 500.0,
            max_dark_hops: 5,
            max_unmonitored_forks: 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PruningSummary {
    pub initial_macro_corridors: usize,
    pub permanently_locked_gates: usize,
    pub final_active_cameras: usize,
    pub pruned_redundant_cameras: usize,
    pub execution_duration_ms: f64,
}

/// Reverse-prunes cameras starting from 100% blanket coverage, validating every removal
/// with localized unmonitored dark-path BFS searches to guarantee zero unmonitored escapes.
pub fn prune_to_zero_escape_network(
    city: &CityGraph,
    skeleton: &SkeletonGraph,
    cuts: &TopologicalCutReport,
    config: ZeroEscapePrunerConfig,
) -> (Vec<OptimizedCameraPole>, PruningSummary) {
    let t0 = std::time::Instant::now();

    // 1. PHASE 1: Blanket Deployment
    // Initially, EVERY macro-corridor has an active camera
    let mut active_cameras: AHashSet<usize> = (0..skeleton.corridors.len()).collect();

    // Pre-build directed corridor adjacency on the skeleton:
    // start_junction -> Vec<(corridor_id, end_junction, length_m)>
    let mut outgoing_corridors: AHashMap<NodeId, Vec<(usize, NodeId, f64)>> = AHashMap::new();

    for corridor in &skeleton.corridors {
        let u = corridor.start_node;
        let v = corridor.end_node;
        let len = corridor.total_length_m;

        outgoing_corridors.entry(u).or_default().push((corridor.id, v, len));

        if corridor.is_two_way {
            outgoing_corridors.entry(v).or_default().push((corridor.id, u, len));
        }
    }

    // 2. PHASE 2: Candidate Evaluation Ordering
    // Never prune mandatory locked gates (Tarjan bridges and perimeter portals)
    let locked_set = &cuts.mandatory_locked_corridors;

    // Sort candidate corridors: evaluate short local links first,
    // keeping longer primary avenues as structural backbones.
    let mut candidate_ids: Vec<usize> = (0..skeleton.corridors.len())
        .filter(|id| !locked_set.contains(id))
        .collect();

    candidate_ids.sort_by(|&a, &b| {
        let len_a = skeleton.corridors[a].total_length_m;
        let len_b = skeleton.corridors[b].total_length_m;
        len_a.partial_cmp(&len_b).unwrap()
    });

    // 3. PHASE 3: Localized Dark-Path BFS Elimination Loop
    let mut pruned_count = 0usize;

    for &corridor_id in &candidate_ids {
        // Temporarily simulate removal
        active_cameras.remove(&corridor_id);

        let corridor = &skeleton.corridors[corridor_id];
        let u = corridor.start_node;
        let v = corridor.end_node;

        // Check if removing this camera creates an escape path exceeding limits
        let forward_violates = dark_cluster_violates(
            v,
            corridor.total_length_m,
            &active_cameras,
            &outgoing_corridors,
            cuts,
            &config,
        );

        let backward_violates = if corridor.is_two_way {
            dark_cluster_violates(
                u,
                corridor.total_length_m,
                &active_cameras,
                &outgoing_corridors,
                cuts,
                &config,
            )
        } else {
            false
        };

        if forward_violates || backward_violates {
            // REJECTION: Critical structural seal. Restore camera immediately.
            active_cameras.insert(corridor_id);
        } else {
            // ACCEPTED: Redundant internal sensor. Keep pruned.
            pruned_count += 1;
        }
    }

    // 4. PHASE 4: Build Hardware Camera Pole Records
    let mut final_poles = Vec::with_capacity(active_cameras.len());

    for &corridor_id in &active_cameras {
        let corr = &skeleton.corridors[corridor_id];
        let (from_id, to_id) = corr.representative_edge;

        let from_idx = city.node_map[&from_id];
        let to_idx = city.node_map[&to_id];
        let u_meta = &city.graph[from_idx];
        let v_meta = &city.graph[to_idx];

        final_poles.push(OptimizedCameraPole {
            camera_id: format!("CAM_ZERO_ESC_{:05}", corridor_id),
            edge_from: from_id,
            edge_to: to_id,
            lat: (u_meta.lat + v_meta.lat) / 2.0,
            lon: (u_meta.lon + v_meta.lon) / 2.0,
            is_two_way: corr.is_two_way,
        });
    }

    let summary = PruningSummary {
        initial_macro_corridors: skeleton.corridors.len(),
        permanently_locked_gates: locked_set.len(),
        final_active_cameras: final_poles.len(),
        pruned_redundant_cameras: pruned_count,
        execution_duration_ms: t0.elapsed().as_secs_f64() * 1000.0,
    };

    (final_poles, summary)
}

#[derive(Copy, Clone, PartialEq)]
struct DState {
    dist: f64,
    node: NodeId,
    hops: usize,
}

impl Eq for DState {}

impl Ord for DState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.dist.partial_cmp(&self.dist).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for DState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn dark_cluster_violates(
    start_node: NodeId,
    initial_edge_len: f64,
    active_cameras: &AHashSet<usize>,
    outgoing_corridors: &AHashMap<NodeId, Vec<(usize, NodeId, f64)>>,
    cuts: &TopologicalCutReport,
    config: &ZeroEscapePrunerConfig,
) -> bool {
    let mut dist_map: AHashMap<NodeId, f64> = AHashMap::new();
    let mut heap = BinaryHeap::new();

    dist_map.insert(start_node, initial_edge_len);
    heap.push(DState {
        dist: initial_edge_len,
        node: start_node,
        hops: 1,
    });

    while let Some(DState { dist, node, hops }) = heap.pop() {
        if dist > *dist_map.get(&node).unwrap_or(&f64::INFINITY) {
            continue;
        }

        if dist > config.max_dark_distance_m {
            return true;
        }

        if hops > config.max_dark_hops {
            return true;
        }

        if cuts.perimeter_corridor_ids.iter().any(|&pid| !active_cameras.contains(&pid)) && dist > 150.0 {
            return true;
        }

        if let Some(neighbors) = outgoing_corridors.get(&node) {
            let mut dark_branches = 0usize;

            for &(corr_id, next_node, corr_len) in neighbors {
                if active_cameras.contains(&corr_id) {
                    continue;
                }

                dark_branches += 1;
                let next_dist = dist + corr_len;

                if next_dist > config.max_dark_distance_m {
                    return true;
                }

                if next_dist < *dist_map.get(&next_node).unwrap_or(&f64::INFINITY) {
                    dist_map.insert(next_node, next_dist);
                    heap.push(DState {
                        dist: next_dist,
                        node: next_node,
                        hops: hops + 1,
                    });
                }
            }

            if dark_branches >= config.max_unmonitored_forks && dist > 120.0 {
                return true;
            }
        }
    }

    false
}