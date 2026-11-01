use crate::cordon::find_perimeter_gateway_edges;
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

    let min_hits_per_trip: u8 = if require_multi_hit { 2 } else { 2 };
    let budget_k = match goal {
        OptimizationGoal::Budget(k) => k,
        OptimizationGoal::TargetCoverage(_) => 650, // High-fidelity density default
    };

    let mut selected_poles = Vec::new();
    let mut deployed_canonical: AHashSet<DirectedEdge> = AHashSet::new();

    // -------------------------------------------------------------
    // PHASE 1: Pre-seed Perimeter Gateways (20% of Budget)
    // -------------------------------------------------------------
    let gateway_budget = (budget_k as f64 * 0.20).ceil() as usize;
    let gateways = find_perimeter_gateway_edges(city, gateway_budget);

    for (u, v) in gateways {
        if selected_poles.len() >= budget_k {
            break;
        }
        let u_idx = match city.node_map.get(&u) {
            Some(&i) => i,
            None => continue,
        };
        let v_idx = match city.node_map.get(&v) {
            Some(&i) => i,
            None => continue,
        };

        let canonical = if u < v { (u, v) } else { (v, u) };
        if !deployed_canonical.insert(canonical) {
            continue;
        }

        let u_meta = &city.graph[u_idx];
        let v_meta = &city.graph[v_idx];
        let is_two_way = city.graph.contains_edge(u_idx, v_idx) && city.graph.contains_edge(v_idx, u_idx);

        selected_poles.push(OptimizedCameraPole {
            camera_id: format!("CAM_CORDON_{:04}", selected_poles.len() + 1),
            edge_from: u,
            edge_to: v,
            lat: (u_meta.lat + v_meta.lat) / 2.0,
            lon: (u_meta.lon + v_meta.lon) / 2.0,
            total_intercepted_trips: 0,
            marginal_new_trips: 0,
            is_two_way,
        });
    }

    // -------------------------------------------------------------
    // PHASE 2: Invert Corridors: Canonical Edge -> (trip_id, step_idx, total_len)
    // -------------------------------------------------------------
    let mut edge_to_steps: AHashMap<DirectedEdge, Vec<(usize, usize, usize)>> = AHashMap::new();

    for (trip_id, edge_path) in corridors.iter().enumerate() {
        let path_len = edge_path.len();
        for (step_idx, &(u, v)) in edge_path.iter().enumerate() {
            let u_idx = match city.node_map.get(&u) {
                Some(&i) => i,
                None => continue,
            };
            let v_idx = match city.node_map.get(&v) {
                Some(&i) => i,
                None => continue,
            };

            let canonical = if city.graph.contains_edge(u_idx, v_idx) {
                (u, v)
            } else {
                (v, u)
            };

            edge_to_steps
                .entry(canonical)
                .or_insert_with(Vec::new)
                .push((trip_id, step_idx, path_len));
        }
    }

    let mut trip_hit_counts = vec![0u8; total_trips];
    let mut trip_hit_positions: Vec<Vec<usize>> = vec![Vec::new(); total_trips];
    let mut covered_trips: AHashSet<usize> = AHashSet::with_capacity(total_trips);

    // Register any trips already intercepted by Phase 1 cordon poles
    for pole in &selected_poles {
        let canonical = if pole.edge_from < pole.edge_to {
            (pole.edge_from, pole.edge_to)
        } else {
            (pole.edge_to, pole.edge_from)
        };

        if let Some(steps) = edge_to_steps.get(&canonical) {
            for &(t_id, step_idx, _) in steps {
                trip_hit_counts[t_id] += 1;
                trip_hit_positions[t_id].push(step_idx);
                if trip_hit_counts[t_id] >= min_hits_per_trip {
                    covered_trips.insert(t_id);
                }
            }
        }
    }

    let target_coverage_count = match goal {
        OptimizationGoal::TargetCoverage(pct) => {
            ((total_trips as f64) * pct.clamp(0.0, 1.0)).ceil() as usize
        }
        OptimizationGoal::Budget(_) => total_trips,
    };

    // -------------------------------------------------------------
    // PHASE 3: Submodular Flow-Interception with Distance Clamping
    // -------------------------------------------------------------
    while selected_poles.len() < budget_k {
        let mut best_edge = None;
        let mut best_score: i64 = 0;
        let mut step_updates = Vec::new();

        for (&edge, steps) in edge_to_steps.iter() {
            let mut score: i64 = 0;
            let mut valid_hits = Vec::new();

            for &(t_id, step_idx, total_len) in steps {
                let hits = trip_hit_counts[t_id];

                // Perimeter weighting: first 3 or last 3 intersections of a trip
                let is_perimeter = step_idx <= 2 || (total_len > 3 && step_idx >= total_len - 3);

                if hits < min_hits_per_trip {
                    // Highest priority: unmonitored journeys
                    score += if is_perimeter { 25 } else { 12 };
                    valid_hits.push((t_id, step_idx));
                } else {
                    // Intermediate gap clamping: bonus if distance to existing camera >= 6 nodes (~350m)
                    let positions = &trip_hit_positions[t_id];
                    let min_dist_to_existing = positions
                        .iter()
                        .map(|&p| (p as isize - step_idx as isize).abs())
                        .min()
                        .unwrap_or(isize::MAX);

                    if min_dist_to_existing >= 6 {
                        score += 5;
                        valid_hits.push((t_id, step_idx));
                    }
                }
            }

            if score > best_score {
                best_score = score;
                best_edge = Some(edge);
                step_updates = valid_hits;
            }
        }

        let chosen_edge = match best_edge {
            Some(e) if best_score > 0 => e,
            _ => break, // No remaining edges offer marginal improvement
        };

        let total_traffic = edge_to_steps.remove(&chosen_edge).unwrap().len();

        for (t_id, step_idx) in step_updates {
            trip_hit_counts[t_id] += 1;
            trip_hit_positions[t_id].push(step_idx);

            if trip_hit_counts[t_id] >= min_hits_per_trip {
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
            marginal_new_trips: best_score as usize,
            is_two_way,
        });

        // If target coverage is met and we've placed enough cordon anchors, we can stop
        if covered_trips.len() >= target_coverage_count && selected_poles.len() >= 300 {
            break;
        }
    }

    selected_poles
}