use crate::od_sampler::DirectedEdge;
use crate::types::{OptimizationGoal, OptimizedCameraPole};
use ahash::{AHashMap, AHashSet};
use engine_core::CityGraph;

pub fn solve_flow_interception(
    city: &CityGraph,
    corridors: &[Vec<DirectedEdge>],
    goal: OptimizationGoal,
    require_multi_hit: bool,
) -> Vec<OptimizedCameraPole> {
    let total_trips = corridors.len();
    if total_trips == 0 {
        return vec![];
    }

    let target_hits_needed: u8 = if require_multi_hit { 2 } else { 1 };

    // 1. Invert paths: Canonical Undirected Edge -> Set of passing trip IDs
    let mut edge_to_trips: AHashMap<DirectedEdge, AHashSet<usize>> = AHashMap::new();

    for (trip_id, edge_path) in corridors.iter().enumerate() {
        for &(u, v) in edge_path {
            // Normalize edge key so two-way road segments share an entry
            let edge_key = if u < v { (u, v) } else { (v, u) };
            edge_to_trips
                .entry(edge_key)
                .or_insert_with(AHashSet::new)
                .insert(trip_id);
        }
    }

    let mut trip_hit_counts = vec![0u8; total_trips];
    let mut covered_trips: AHashSet<usize> = AHashSet::with_capacity(total_trips);
    let mut selected_poles = Vec::new();

    let target_coverage_count = match goal {
        OptimizationGoal::TargetCoverage(pct) => ((total_trips as f64) * pct.clamp(0.0, 1.0)).ceil() as usize,
        OptimizationGoal::Budget(_) => total_trips,
    };
    let budget_k = match goal {
        OptimizationGoal::Budget(k) => k,
        OptimizationGoal::TargetCoverage(_) => usize::MAX,
    };

    // 2. Greedy Submodular Set Cover Loop
    while selected_poles.len() < budget_k && covered_trips.len() < target_coverage_count {
        let mut best_edge = None;
        let mut best_gain = 0;
        let mut trips_helped = Vec::new();

        for (&edge, passing_trips) in edge_to_trips.iter() {
            let mut gain = 0;
            let mut helped = Vec::new();

            for &t_id in passing_trips {
                if trip_hit_counts[t_id] < target_hits_needed {
                    gain += 1;
                    helped.push(t_id);
                }
            }

            if gain > best_gain {
                best_gain = gain;
                best_edge = Some(edge);
                trips_helped = helped;
            }
        }

        let chosen_edge = match best_edge {
            Some(e) => e,
            None => break, // No edges left that improve coverage
        };

        let total_traffic = edge_to_trips.remove(&chosen_edge).unwrap().len();

        for t_id in trips_helped {
            trip_hit_counts[t_id] += 1;
            if trip_hit_counts[t_id] >= target_hits_needed {
                covered_trips.insert(t_id);
            }
        }

        let (u, v) = chosen_edge;
        let u_idx = city.node_map[&u];
        let v_idx = city.node_map[&v];
        let u_meta = &city.graph[u_idx];
        let v_meta = &city.graph[v_idx];

        let is_two_way = city.graph.contains_edge(u_idx, v_idx) && city.graph.contains_edge(v_idx, u_idx);

        selected_poles.push(OptimizedCameraPole {
            camera_id: format!("CAM_OPT_{:04}", selected_poles.len() + 1),
            edge_from: u,
            edge_to: v,
            lat: (u_meta.lat + v_meta.lat) / 2.0,
            lon: (u_meta.lon + v_meta.lon) / 2.0,
            total_intercepted_trips: total_traffic,
            marginal_new_trips: best_gain,
            is_two_way,
        });
    }

    selected_poles
}