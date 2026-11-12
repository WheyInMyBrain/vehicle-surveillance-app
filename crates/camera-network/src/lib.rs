pub mod audit;
pub mod config;
pub mod hardware;
pub mod skeleton;
pub mod types;

pub use audit::{run_audit, AuditReport};
pub use config::NetworkConfig;
pub use hardware::{evaluate_hardware, HardwareAssessment, MountType, PhysicalSiteProfile};
pub use skeleton::{build_skeleton, SkeletonGraph};
pub use types::{CameraPole, MacroCorridor};

use ahash::{AHashMap, AHashSet};
use road_graph_loader::RoadGraph;
use road_network_core::NodeId;

#[derive(Debug, Clone)]
pub struct DeploymentSummary {
    pub stage_name: String,
    pub poles: Vec<CameraPole>,
    pub total_poles: usize,
    pub total_heads: usize,
    pub total_cost_inr: f64,
    pub monitored_corridors: usize,
    pub execution_ms: f64,
    pub monitored_corridor_ids: AHashSet<usize>,
}

/// Baseline Stage 0: Exactly 1 camera pole per continuous road corridor.
pub fn deploy_stage_0_baseline(graph: &RoadGraph, config: &NetworkConfig) -> DeploymentSummary {
    let t0 = std::time::Instant::now();
    let skeleton = build_skeleton(graph);

    let all_corridor_ids: AHashSet<usize> = (0..skeleton.corridors.len()).collect();
    let mut poles = Vec::with_capacity(skeleton.corridors.len());
    let mut total_heads = 0usize;
    let mut total_cost = 0.0f64;

    for corr in &skeleton.corridors {
        let hw = evaluate_hardware(
            corr.lanes_forward,
            corr.lanes_backward,
            corr.is_two_way,
            false,
            &config.sensor,
            &config.bom,
        );

        total_heads += hw.camera_heads as usize;
        total_cost += hw.estimated_cost;

        poles.push(CameraPole {
            pole_id: format!("POLE_RAW_{:05}", corr.id),
            edge_from: corr.representative_edge.0,
            edge_to: corr.representative_edge.1,
            lat: corr.midpoint.0,
            lon: corr.midpoint.1,
            bearing_deg: corr.midpoint.2,
            camera_heads: hw.camera_heads,
            lanes_covered: hw.total_lanes,
            mount_type: hw.mount_type,
            is_two_way: corr.is_two_way,
            estimated_cost_inr: hw.estimated_cost,
            profile: hw.profile,
        });
    }

    DeploymentSummary {
        stage_name: "Stage 0: Unpruned Complete Baseline".to_string(),
        total_poles: poles.len(),
        total_heads,
        total_cost_inr: total_cost,
        monitored_corridors: all_corridor_ids.len(),
        execution_ms: t0.elapsed().as_secs_f64() * 1000.0,
        monitored_corridor_ids: all_corridor_ids,
        poles,
    }
}

/// Stage 1: Physical Site Co-Location
/// Collapses opposite directions of undivided roads and parallel divided carriageways
/// onto shared physical poles and gantries (Cash/Infrastructure optimization).
pub fn deploy_stage_1_colocated(graph: &RoadGraph, config: &NetworkConfig) -> DeploymentSummary {
    let t0 = std::time::Instant::now();
    let skeleton = build_skeleton(graph);

    let all_corridor_ids: AHashSet<usize> = (0..skeleton.corridors.len()).collect();
    let mut paired_with: AHashMap<usize, usize> = AHashMap::new();
    let mut claimed: AHashSet<usize> = AHashSet::new();

    // 1. Match exact-reverse corridors (opposite directions of undivided roads)
    let mut junction_pair_map: AHashMap<(NodeId, NodeId), usize> = AHashMap::new();
    for corr in &skeleton.corridors {
        let u = corr.start_junction.min(corr.end_junction);
        let v = corr.start_junction.max(corr.end_junction);
        if corr.start_junction != corr.end_junction {
            if let Some(&other_cid) = junction_pair_map.get(&(u, v)) {
                if !claimed.contains(&corr.id) && !claimed.contains(&other_cid) {
                    paired_with.insert(corr.id, other_cid);
                    paired_with.insert(other_cid, corr.id);
                    claimed.insert(corr.id);
                    claimed.insert(other_cid);
                }
            } else {
                junction_pair_map.insert((u, v), corr.id);
            }
        }
    }

    // 2. Match parallel divided carriageways (antiparallel within 35m lateral distance)
    let mut spatial_grid: AHashMap<(i32, i32), Vec<usize>> = AHashMap::new();
    let grid_size = 0.0005;

    for corr in &skeleton.corridors {
        if claimed.contains(&corr.id) {
            continue;
        }
        let gx = (corr.midpoint.0 / grid_size).floor() as i32;
        let gy = (corr.midpoint.1 / grid_size).floor() as i32;
        spatial_grid.entry((gx, gy)).or_default().push(corr.id);
    }

    for corr_a in &skeleton.corridors {
        if claimed.contains(&corr_a.id) {
            continue;
        }
        let gx = (corr_a.midpoint.0 / grid_size).floor() as i32;
        let gy = (corr_a.midpoint.1 / grid_size).floor() as i32;

        let mut best_pair = None;
        let mut min_dist = 35.0f32;

        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(neighbors) = spatial_grid.get(&(gx + dx, gy + dy)) {
                    for &other_cid in neighbors {
                        if other_cid == corr_a.id || claimed.contains(&other_cid) {
                            continue;
                        }
                        let corr_b = &skeleton.corridors[other_cid];

                        let angle_diff = (corr_a.midpoint.2 - corr_b.midpoint.2).abs();
                        let opp_diff = (angle_diff - 180.0).abs();
                        if opp_diff < 35.0 || opp_diff > 325.0 {
                            let dlat = ((corr_a.midpoint.0 - corr_b.midpoint.0).abs() * 111_000.0) as f32;
                            let dlon = ((corr_a.midpoint.1 - corr_b.midpoint.1).abs() * 105_000.0) as f32;
                            let dist = (dlat * dlat + dlon * dlon).sqrt();

                            if dist < min_dist {
                                min_dist = dist;
                                best_pair = Some(other_cid);
                            }
                        }
                    }
                }
            }
        }

        if let Some(other_cid) = best_pair {
            paired_with.insert(corr_a.id, other_cid);
            paired_with.insert(other_cid, corr_a.id);
            claimed.insert(corr_a.id);
            claimed.insert(other_cid);
        }
    }

    // 3. Build consolidated physical poles
    let mut poles = Vec::new();
    let mut total_heads = 0usize;
    let mut total_cost = 0.0f64;
    let mut processed: AHashSet<usize> = AHashSet::new();

    for corr_a in &skeleton.corridors {
        if processed.contains(&corr_a.id) {
            continue;
        }
        processed.insert(corr_a.id);

        if let Some(&other_cid) = paired_with.get(&corr_a.id) {
            processed.insert(other_cid);
            let corr_b = &skeleton.corridors[other_cid];

            let hw = evaluate_hardware(
                corr_a.lanes_forward,
                corr_b.lanes_forward,
                true,
                false,
                &config.sensor,
                &config.bom,
            );

            total_heads += hw.camera_heads as usize;
            total_cost += hw.estimated_cost;

            poles.push(CameraPole {
                pole_id: format!("POLE_COLOC_{:05}_{:05}", corr_a.id.min(other_cid), corr_a.id.max(other_cid)),
                edge_from: corr_a.representative_edge.0,
                edge_to: corr_a.representative_edge.1,
                lat: (corr_a.midpoint.0 + corr_b.midpoint.0) * 0.5,
                lon: (corr_a.midpoint.1 + corr_b.midpoint.1) * 0.5,
                bearing_deg: corr_a.midpoint.2,
                camera_heads: hw.camera_heads,
                lanes_covered: corr_a.lanes_forward + corr_b.lanes_forward,
                mount_type: hw.mount_type,
                is_two_way: true,
                estimated_cost_inr: hw.estimated_cost,
                profile: hw.profile,
            });
        } else {
            let hw = evaluate_hardware(
                corr_a.lanes_forward,
                corr_a.lanes_backward,
                corr_a.is_two_way,
                false,
                &config.sensor,
                &config.bom,
            );

            total_heads += hw.camera_heads as usize;
            total_cost += hw.estimated_cost;

            poles.push(CameraPole {
                pole_id: format!("POLE_SOLO_{:05}", corr_a.id),
                edge_from: corr_a.representative_edge.0,
                edge_to: corr_a.representative_edge.1,
                lat: corr_a.midpoint.0,
                lon: corr_a.midpoint.1,
                bearing_deg: corr_a.midpoint.2,
                camera_heads: hw.camera_heads,
                lanes_covered: hw.total_lanes,
                mount_type: hw.mount_type,
                is_two_way: corr_a.is_two_way,
                estimated_cost_inr: hw.estimated_cost,
                profile: hw.profile,
            });
        }
    }

    DeploymentSummary {
        stage_name: "Stage 1: Physical Site Co-Location".to_string(),
        total_poles: poles.len(),
        total_heads,
        total_cost_inr: total_cost,
        monitored_corridors: all_corridor_ids.len(),
        execution_ms: t0.elapsed().as_secs_f64() * 1000.0,
        monitored_corridor_ids: all_corridor_ids,
        poles,
    }
}

/// Stage 2: Linear Corridor Cadence Normalization
/// Chains consecutive corridors along unbroken roads (where connecting nodes have zero
/// other turn choices) and spaces cameras at regular 800m - 1200m intervals.
pub fn deploy_stage_2_linear_cadence(graph: &RoadGraph, config: &NetworkConfig) -> DeploymentSummary {
    let t0 = std::time::Instant::now();
    let skeleton = build_skeleton(graph);

    let target_cadence_m = 900.0f32; // Normal camera spacing along unbroken straight highways
    let mut active_corridors = AHashSet::new();

    // 1. Group corridors into unbroken linear road chains
    // A junction node is an intermediate continuation (not a fork) if both in-degree and out-degree are 1
    let mut visited_corridors = AHashSet::new();

    for start_cid in 0..skeleton.corridors.len() {
        if !visited_corridors.insert(start_cid) {
            continue;
        }

        let mut chain = vec![start_cid];
        let mut chain_length = skeleton.corridors[start_cid].total_length_m;

        // Walk forward along continuous linear continuation
        let mut curr_cid = start_cid;
        loop {
            let corr = &skeleton.corridors[curr_cid];
            let end_node = corr.end_junction;

            if let Some(outgoing) = skeleton.junction_outgoing.get(&end_node) {
                // If the end node has exactly 1 forward outgoing choice and 0 side branches
                if outgoing.len() == 1 {
                    let next_cid = outgoing[0].0;
                    if visited_corridors.insert(next_cid) {
                        chain.push(next_cid);
                        chain_length += skeleton.corridors[next_cid].total_length_m;
                        curr_cid = next_cid;
                        continue;
                    }
                }
            }
            break;
        }

        // 2. Apply Cadence Spacing along the chain:
        // - If chain <= 1000m: keep exactly 1 camera on the chain.
        // - If chain > 1000m: place cameras spaced at ~target_cadence_m intervals.
        if chain_length <= 1000.0 || chain.len() == 1 {
            // Pick representative corridor in the middle of the chain
            let rep_idx = chain.len() / 2;
            active_corridors.insert(chain[rep_idx]);
        } else {
            let mut accumulated_dist = 0.0f32;
            let mut last_cam_dist = 0.0f32;

            for &cid in &chain {
                let len = skeleton.corridors[cid].total_length_m;
                accumulated_dist += len;

                // Place camera if we have traversed past target cadence or if it's the chain entry/exit
                if (accumulated_dist - last_cam_dist) >= target_cadence_m || last_cam_dist == 0.0 {
                    active_corridors.insert(cid);
                    last_cam_dist = accumulated_dist;
                }
            }
            // Ensure the terminal end of a long expressway chain is guarded
            if let Some(&last_cid) = chain.last() {
                active_corridors.insert(last_cid);
            }
        }
    }

    // 3. Physical Site Co-Location over the cadence-normalized active set
    let mut paired_with: AHashMap<usize, usize> = AHashMap::new();
    let mut claimed: AHashSet<usize> = AHashSet::new();

    let mut junction_pair_map: AHashMap<(NodeId, NodeId), usize> = AHashMap::new();
    for &cid in &active_corridors {
        let corr = &skeleton.corridors[cid];
        let u = corr.start_junction.min(corr.end_junction);
        let v = corr.start_junction.max(corr.end_junction);
        if corr.start_junction != corr.end_junction {
            if let Some(&other_cid) = junction_pair_map.get(&(u, v)) {
                if !claimed.contains(&cid) && !claimed.contains(&other_cid) {
                    paired_with.insert(cid, other_cid);
                    paired_with.insert(other_cid, cid);
                    claimed.insert(cid);
                    claimed.insert(other_cid);
                }
            } else {
                junction_pair_map.insert((u, v), cid);
            }
        }
    }

    let mut spatial_grid: AHashMap<(i32, i32), Vec<usize>> = AHashMap::new();
    let grid_size = 0.0005;

    for &cid in &active_corridors {
        if claimed.contains(&cid) {
            continue;
        }
        let corr = &skeleton.corridors[cid];
        let gx = (corr.midpoint.0 / grid_size).floor() as i32;
        let gy = (corr.midpoint.1 / grid_size).floor() as i32;
        spatial_grid.entry((gx, gy)).or_default().push(cid);
    }

    for &cid in &active_corridors {
        if claimed.contains(&cid) {
            continue;
        }
        let corr_a = &skeleton.corridors[cid];
        let gx = (corr_a.midpoint.0 / grid_size).floor() as i32;
        let gy = (corr_a.midpoint.1 / grid_size).floor() as i32;

        let mut best_pair = None;
        let mut min_dist = 35.0f32;

        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(neighbors) = spatial_grid.get(&(gx + dx, gy + dy)) {
                    for &other_cid in neighbors {
                        if other_cid == cid || claimed.contains(&other_cid) {
                            continue;
                        }
                        let corr_b = &skeleton.corridors[other_cid];

                        let angle_diff = (corr_a.midpoint.2 - corr_b.midpoint.2).abs();
                        let opp_diff = (angle_diff - 180.0).abs();
                        if opp_diff < 35.0 || opp_diff > 325.0 {
                            let dlat = ((corr_a.midpoint.0 - corr_b.midpoint.0).abs() * 111_000.0) as f32;
                            let dlon = ((corr_a.midpoint.1 - corr_b.midpoint.1).abs() * 105_000.0) as f32;
                            let dist = (dlat * dlat + dlon * dlon).sqrt();

                            if dist < min_dist {
                                min_dist = dist;
                                best_pair = Some(other_cid);
                            }
                        }
                    }
                }
            }
        }

        if let Some(other_cid) = best_pair {
            paired_with.insert(cid, other_cid);
            paired_with.insert(other_cid, cid);
            claimed.insert(cid);
            claimed.insert(other_cid);
        }
    }

    let mut poles = Vec::new();
    let mut total_heads = 0usize;
    let mut total_cost = 0.0f64;
    let mut processed: AHashSet<usize> = AHashSet::new();

    for &cid in &active_corridors {
        if processed.contains(&cid) {
            continue;
        }
        processed.insert(cid);
        let corr_a = &skeleton.corridors[cid];

        if let Some(&other_cid) = paired_with.get(&cid) {
            processed.insert(other_cid);
            let corr_b = &skeleton.corridors[other_cid];

            let hw = evaluate_hardware(
                corr_a.lanes_forward,
                corr_b.lanes_forward,
                true,
                false,
                &config.sensor,
                &config.bom,
            );

            total_heads += hw.camera_heads as usize;
            total_cost += hw.estimated_cost;

            poles.push(CameraPole {
                pole_id: format!("POLE_S2_{:05}_{:05}", cid.min(other_cid), cid.max(other_cid)),
                edge_from: corr_a.representative_edge.0,
                edge_to: corr_a.representative_edge.1,
                lat: (corr_a.midpoint.0 + corr_b.midpoint.0) * 0.5,
                lon: (corr_a.midpoint.1 + corr_b.midpoint.1) * 0.5,
                bearing_deg: corr_a.midpoint.2,
                camera_heads: hw.camera_heads,
                lanes_covered: corr_a.lanes_forward + corr_b.lanes_forward,
                mount_type: hw.mount_type,
                is_two_way: true,
                estimated_cost_inr: hw.estimated_cost,
                profile: hw.profile,
            });
        } else {
            let hw = evaluate_hardware(
                corr_a.lanes_forward,
                corr_a.lanes_backward,
                corr_a.is_two_way,
                false,
                &config.sensor,
                &config.bom,
            );

            total_heads += hw.camera_heads as usize;
            total_cost += hw.estimated_cost;

            poles.push(CameraPole {
                pole_id: format!("POLE_S2_{:05}", cid),
                edge_from: corr_a.representative_edge.0,
                edge_to: corr_a.representative_edge.1,
                lat: corr_a.midpoint.0,
                lon: corr_a.midpoint.1,
                bearing_deg: corr_a.midpoint.2,
                camera_heads: hw.camera_heads,
                lanes_covered: hw.total_lanes,
                mount_type: hw.mount_type,
                is_two_way: corr_a.is_two_way,
                estimated_cost_inr: hw.estimated_cost,
                profile: hw.profile,
            });
        }
    }

    DeploymentSummary {
        stage_name: "Stage 2: Linear Cadence Normalization".to_string(),
        total_poles: poles.len(),
        total_heads,
        total_cost_inr: total_cost,
        monitored_corridors: active_corridors.len(),
        execution_ms: t0.elapsed().as_secs_f64() * 1000.0,
        monitored_corridor_ids: active_corridors,
        poles,
    }
}

/// Stage 3: True Dead-End Spur Elimination (Conservation of Flow)
/// Eliminates cameras inside short terminal leaves where the terminal node
/// has zero other connections in the raw graph, and the root entrance junction
/// is guaranteed to be monitored by surrounding arterial corridors.
pub fn deploy_stage_3_dead_ends(graph: &RoadGraph, config: &NetworkConfig) -> DeploymentSummary {
    let t0 = std::time::Instant::now();
    let skeleton = build_skeleton(graph);

    // 1. Raw graph node connectivity
    let mut raw_in_deg: AHashMap<NodeId, usize> = AHashMap::with_capacity(graph.num_nodes);
    let mut raw_out_deg: AHashMap<NodeId, usize> = AHashMap::with_capacity(graph.num_nodes);
    for edge_idx in 0..graph.num_edges {
        let desc = graph.edge_descriptor(edge_idx);
        *raw_out_deg.entry(desc.u).or_default() += 1;
        *raw_in_deg.entry(desc.v).or_default() += 1;
    }

    // 2. Count distinct undirected neighbors on the macro-skeleton
    let mut physical_neighbors: AHashMap<NodeId, AHashSet<NodeId>> = AHashMap::new();
    for corr in &skeleton.corridors {
        if corr.start_junction != corr.end_junction {
            physical_neighbors.entry(corr.start_junction).or_default().insert(corr.end_junction);
            physical_neighbors.entry(corr.end_junction).or_default().insert(corr.start_junction);
        }
    }

    // 3. Identify candidate cul-de-sacs with verified raw graph dead ends
    let mut candidate_dead_ends = AHashSet::new();

    for corr in &skeleton.corridors {
        let u_neighbors = physical_neighbors.get(&corr.start_junction).map(|s| s.len()).unwrap_or(0);
        let v_neighbors = physical_neighbors.get(&corr.end_junction).map(|s| s.len()).unwrap_or(0);

        // One end must have exactly 1 neighbor, the other must have >= 2
        let (leaf_node, root_node) = if u_neighbors == 1 && v_neighbors >= 2 {
            (corr.start_junction, corr.end_junction)
        } else if v_neighbors == 1 && u_neighbors >= 2 {
            (corr.end_junction, corr.start_junction)
        } else {
            continue;
        };

        // Strict verification: Leaf node in the raw graph must have at most 1 connection in each direction
        // (prevents pruning border gateways or one-way turnaround branches)
        let leaf_in = raw_in_deg.get(&leaf_node).copied().unwrap_or(0);
        let leaf_out = raw_out_deg.get(&leaf_node).copied().unwrap_or(0);

        if leaf_in <= 1 && leaf_out <= 1 && corr.total_length_m <= 350.0 && corr.maxspeed_kmh <= 35 && corr.lanes_forward == 1 {
            candidate_dead_ends.insert((corr.id, root_node));
        }
    }

    // 4. Throat Guard Verification:
    // A dead-end corridor can ONLY be pruned if its connecting root junction has
    // at least 2 other corridors that remain 100% active.
    let mut safe_pruned_corridors = AHashSet::new();
    let candidate_ids: AHashSet<usize> = candidate_dead_ends.iter().map(|&(cid, _)| cid).collect();

    for &(cid, root_node) in &candidate_dead_ends {
        if let Some(outgoing) = skeleton.junction_outgoing.get(&root_node) {
            // Count non-cul-de-sac exits radiating from the root
            let arterial_exits = outgoing.iter().filter(|(other_cid, _, _)| !candidate_ids.contains(other_cid)).count();
            if arterial_exits >= 2 {
                safe_pruned_corridors.insert(cid);
            }
        }
    }

    // 5. Active corridors: All corridors minus safe verified dead ends
    let active_corridors: AHashSet<usize> = (0..skeleton.corridors.len())
        .filter(|id| !safe_pruned_corridors.contains(id))
        .collect();

    // 6. Physical Site Co-Location over active set
    let mut paired_with: AHashMap<usize, usize> = AHashMap::new();
    let mut claimed: AHashSet<usize> = AHashSet::new();

    let mut junction_pair_map: AHashMap<(NodeId, NodeId), usize> = AHashMap::new();
    for &cid in &active_corridors {
        let corr = &skeleton.corridors[cid];
        let u = corr.start_junction.min(corr.end_junction);
        let v = corr.start_junction.max(corr.end_junction);
        if corr.start_junction != corr.end_junction {
            if let Some(&other_cid) = junction_pair_map.get(&(u, v)) {
                if !claimed.contains(&cid) && !claimed.contains(&other_cid) {
                    paired_with.insert(cid, other_cid);
                    paired_with.insert(other_cid, cid);
                    claimed.insert(cid);
                    claimed.insert(other_cid);
                }
            } else {
                junction_pair_map.insert((u, v), cid);
            }
        }
    }

    let mut spatial_grid: AHashMap<(i32, i32), Vec<usize>> = AHashMap::new();
    let grid_size = 0.0005;

    for &cid in &active_corridors {
        if claimed.contains(&cid) {
            continue;
        }
        let corr = &skeleton.corridors[cid];
        let gx = (corr.midpoint.0 / grid_size).floor() as i32;
        let gy = (corr.midpoint.1 / grid_size).floor() as i32;
        spatial_grid.entry((gx, gy)).or_default().push(cid);
    }

    for &cid in &active_corridors {
        if claimed.contains(&cid) {
            continue;
        }
        let corr_a = &skeleton.corridors[cid];
        let gx = (corr_a.midpoint.0 / grid_size).floor() as i32;
        let gy = (corr_a.midpoint.1 / grid_size).floor() as i32;

        let mut best_pair = None;
        let mut min_dist = 35.0f32;

        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(neighbors) = spatial_grid.get(&(gx + dx, gy + dy)) {
                    for &other_cid in neighbors {
                        if other_cid == cid || claimed.contains(&other_cid) {
                            continue;
                        }
                        let corr_b = &skeleton.corridors[other_cid];

                        let angle_diff = (corr_a.midpoint.2 - corr_b.midpoint.2).abs();
                        let opp_diff = (angle_diff - 180.0).abs();
                        if opp_diff < 35.0 || opp_diff > 325.0 {
                            let dlat = ((corr_a.midpoint.0 - corr_b.midpoint.0).abs() * 111_000.0) as f32;
                            let dlon = ((corr_a.midpoint.1 - corr_b.midpoint.1).abs() * 105_000.0) as f32;
                            let dist = (dlat * dlat + dlon * dlon).sqrt();

                            if dist < min_dist {
                                min_dist = dist;
                                best_pair = Some(other_cid);
                            }
                        }
                    }
                }
            }
        }

        if let Some(other_cid) = best_pair {
            paired_with.insert(cid, other_cid);
            paired_with.insert(other_cid, cid);
            claimed.insert(cid);
            claimed.insert(other_cid);
        }
    }

    let mut poles = Vec::new();
    let mut total_heads = 0usize;
    let mut total_cost = 0.0f64;
    let mut processed: AHashSet<usize> = AHashSet::new();

    for &cid in &active_corridors {
        if processed.contains(&cid) {
            continue;
        }
        processed.insert(cid);
        let corr_a = &skeleton.corridors[cid];

        if let Some(&other_cid) = paired_with.get(&cid) {
            processed.insert(other_cid);
            let corr_b = &skeleton.corridors[other_cid];

            let hw = evaluate_hardware(
                corr_a.lanes_forward,
                corr_b.lanes_forward,
                true,
                false,
                &config.sensor,
                &config.bom,
            );

            total_heads += hw.camera_heads as usize;
            total_cost += hw.estimated_cost;

            poles.push(CameraPole {
                pole_id: format!("POLE_S3_{:05}_{:05}", cid.min(other_cid), cid.max(other_cid)),
                edge_from: corr_a.representative_edge.0,
                edge_to: corr_a.representative_edge.1,
                lat: (corr_a.midpoint.0 + corr_b.midpoint.0) * 0.5,
                lon: (corr_a.midpoint.1 + corr_b.midpoint.1) * 0.5,
                bearing_deg: corr_a.midpoint.2,
                camera_heads: hw.camera_heads,
                lanes_covered: corr_a.lanes_forward + corr_b.lanes_forward,
                mount_type: hw.mount_type,
                is_two_way: true,
                estimated_cost_inr: hw.estimated_cost,
                profile: hw.profile,
            });
        } else {
            let hw = evaluate_hardware(
                corr_a.lanes_forward,
                corr_a.lanes_backward,
                corr_a.is_two_way,
                false,
                &config.sensor,
                &config.bom,
            );

            total_heads += hw.camera_heads as usize;
            total_cost += hw.estimated_cost;

            poles.push(CameraPole {
                pole_id: format!("POLE_S3_{:05}", cid),
                edge_from: corr_a.representative_edge.0,
                edge_to: corr_a.representative_edge.1,
                lat: corr_a.midpoint.0,
                lon: corr_a.midpoint.1,
                bearing_deg: corr_a.midpoint.2,
                camera_heads: hw.camera_heads,
                lanes_covered: hw.total_lanes,
                mount_type: hw.mount_type,
                is_two_way: corr_a.is_two_way,
                estimated_cost_inr: hw.estimated_cost,
                profile: hw.profile,
            });
        }
    }

    DeploymentSummary {
        stage_name: "Stage 3: Cul-de-Sac Prune".to_string(),
        total_poles: poles.len(),
        total_heads,
        total_cost_inr: total_cost,
        monitored_corridors: active_corridors.len(),
        execution_ms: t0.elapsed().as_secs_f64() * 1000.0,
        monitored_corridor_ids: active_corridors,
        poles,
    }
}

/// Stage 4: K-1 Decision-Fork Deductibility
/// At multi-way junctions (K >= 3), deductively eliminates exactly 1 outgoing camera.
/// Guarantees NO CASCADING DARK RUNS: The corridor chosen for deduction must lead into
/// a junction where 100% of outgoing branches are actively monitored.
pub fn deploy_stage_4_k_minus_1(graph: &RoadGraph, config: &NetworkConfig) -> DeploymentSummary {
    let t0 = std::time::Instant::now();
    let skeleton = build_skeleton(graph);

    // 1. Inherit the active corridor set from Stage 3
    let stage3_summary = deploy_stage_3_dead_ends(graph, config);
    let mut active_corridors = stage3_summary.monitored_corridor_ids;

    // 2. Identify candidate junctions for K-1 deduction (K >= 3 outgoing active branches)
    // To prevent dark chaining, track junctions that are already guarded or adjacent to deducted roads
    let mut deducted_corridors = AHashSet::new();
    let mut protected_junctions = AHashSet::new(); // Junctions whose branches MUST remain 100% monitored

    // Sort junctions deterministically by node ID to ensure reproducible results
    let mut junction_candidates: Vec<NodeId> = skeleton.junction_nodes.iter().copied().collect();
    junction_candidates.sort();

    for junc in junction_candidates {
        // If this junction was marked as protected (it receives an unmonitored branch), skip it!
        if protected_junctions.contains(&junc) {
            continue;
        }

        if let Some(outgoing) = skeleton.junction_outgoing.get(&junc) {
            // Find currently active outgoing corridors from this junction
            let active_outgoing: Vec<(usize, NodeId, f32)> = outgoing
                .iter()
                .filter(|(cid, _, _)| active_corridors.contains(cid))
                .copied()
                .collect();

            let k = active_outgoing.len();

            // Only apply deduction at genuine forks (K >= 3)
            if k >= 3 {
                // Find a safe corridor to deduct:
                // - Must be short (<= 600m)
                // - The downstream destination junction must NOT already be protected or deducted
                let safe_candidate = active_outgoing.iter().find(|(cid, next_junc, len)| {
                    *len <= 600.0
                        && !protected_junctions.contains(next_junc)
                        && !deducted_corridors.contains(cid)
                });

                if let Some(&(chosen_cid, downstream_junc, _)) = safe_candidate {
                    // DEDUCT this corridor's camera
                    deducted_corridors.insert(chosen_cid);
                    active_corridors.remove(&chosen_cid);

                    // LOCK BOTH JUNCTIONS:
                    // 1. Origin junction has used its 1 deduction quota
                    protected_junctions.insert(junc);
                    // 2. Downstream junction MUST have 100% of its branches monitored to trap the vehicle
                    protected_junctions.insert(downstream_junc);
                }
            }
        }
    }

    // 3. Physical Site Co-Location over the resulting active set
    let mut paired_with: AHashMap<usize, usize> = AHashMap::new();
    let mut claimed: AHashSet<usize> = AHashSet::new();

    let mut junction_pair_map: AHashMap<(NodeId, NodeId), usize> = AHashMap::new();
    for &cid in &active_corridors {
        let corr = &skeleton.corridors[cid];
        let u = corr.start_junction.min(corr.end_junction);
        let v = corr.start_junction.max(corr.end_junction);
        if corr.start_junction != corr.end_junction {
            if let Some(&other_cid) = junction_pair_map.get(&(u, v)) {
                if !claimed.contains(&cid) && !claimed.contains(&other_cid) {
                    paired_with.insert(cid, other_cid);
                    paired_with.insert(other_cid, cid);
                    claimed.insert(cid);
                    claimed.insert(other_cid);
                }
            } else {
                junction_pair_map.insert((u, v), cid);
            }
        }
    }

    let mut spatial_grid: AHashMap<(i32, i32), Vec<usize>> = AHashMap::new();
    let grid_size = 0.0005;

    for &cid in &active_corridors {
        if claimed.contains(&cid) {
            continue;
        }
        let corr = &skeleton.corridors[cid];
        let gx = (corr.midpoint.0 / grid_size).floor() as i32;
        let gy = (corr.midpoint.1 / grid_size).floor() as i32;
        spatial_grid.entry((gx, gy)).or_default().push(cid);
    }

    for &cid in &active_corridors {
        if claimed.contains(&cid) {
            continue;
        }
        let corr_a = &skeleton.corridors[cid];
        let gx = (corr_a.midpoint.0 / grid_size).floor() as i32;
        let gy = (corr_a.midpoint.1 / grid_size).floor() as i32;

        let mut best_pair = None;
        let mut min_dist = 35.0f32;

        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(neighbors) = spatial_grid.get(&(gx + dx, gy + dy)) {
                    for &other_cid in neighbors {
                        if other_cid == cid || claimed.contains(&other_cid) {
                            continue;
                        }
                        let corr_b = &skeleton.corridors[other_cid];

                        let angle_diff = (corr_a.midpoint.2 - corr_b.midpoint.2).abs();
                        let opp_diff = (angle_diff - 180.0).abs();
                        if opp_diff < 35.0 || opp_diff > 325.0 {
                            let dlat = ((corr_a.midpoint.0 - corr_b.midpoint.0).abs() * 111_000.0) as f32;
                            let dlon = ((corr_a.midpoint.1 - corr_b.midpoint.1).abs() * 105_000.0) as f32;
                            let dist = (dlat * dlat + dlon * dlon).sqrt();

                            if dist < min_dist {
                                min_dist = dist;
                                best_pair = Some(other_cid);
                            }
                        }
                    }
                }
            }
        }

        if let Some(other_cid) = best_pair {
            paired_with.insert(cid, other_cid);
            paired_with.insert(other_cid, cid);
            claimed.insert(cid);
            claimed.insert(other_cid);
        }
    }

    let mut poles = Vec::new();
    let mut total_heads = 0usize;
    let mut total_cost = 0.0f64;
    let mut processed: AHashSet<usize> = AHashSet::new();

    for &cid in &active_corridors {
        if processed.contains(&cid) {
            continue;
        }
        processed.insert(cid);
        let corr_a = &skeleton.corridors[cid];

        if let Some(&other_cid) = paired_with.get(&cid) {
            processed.insert(other_cid);
            let corr_b = &skeleton.corridors[other_cid];

            let hw = evaluate_hardware(
                corr_a.lanes_forward,
                corr_b.lanes_forward,
                true,
                false,
                &config.sensor,
                &config.bom,
            );

            total_heads += hw.camera_heads as usize;
            total_cost += hw.estimated_cost;

            poles.push(CameraPole {
                pole_id: format!("POLE_S4_{:05}_{:05}", cid.min(other_cid), cid.max(other_cid)),
                edge_from: corr_a.representative_edge.0,
                edge_to: corr_a.representative_edge.1,
                lat: (corr_a.midpoint.0 + corr_b.midpoint.0) * 0.5,
                lon: (corr_a.midpoint.1 + corr_b.midpoint.1) * 0.5,
                bearing_deg: corr_a.midpoint.2,
                camera_heads: hw.camera_heads,
                lanes_covered: corr_a.lanes_forward + corr_b.lanes_forward,
                mount_type: hw.mount_type,
                is_two_way: true,
                estimated_cost_inr: hw.estimated_cost,
                profile: hw.profile,
            });
        } else {
            let hw = evaluate_hardware(
                corr_a.lanes_forward,
                corr_a.lanes_backward,
                corr_a.is_two_way,
                false,
                &config.sensor,
                &config.bom,
            );

            total_heads += hw.camera_heads as usize;
            total_cost += hw.estimated_cost;

            poles.push(CameraPole {
                pole_id: format!("POLE_S4_{:05}", cid),
                edge_from: corr_a.representative_edge.0,
                edge_to: corr_a.representative_edge.1,
                lat: corr_a.midpoint.0,
                lon: corr_a.midpoint.1,
                bearing_deg: corr_a.midpoint.2,
                camera_heads: hw.camera_heads,
                lanes_covered: hw.total_lanes,
                mount_type: hw.mount_type,
                is_two_way: corr_a.is_two_way,
                estimated_cost_inr: hw.estimated_cost,
                profile: hw.profile,
            });
        }
    }

    DeploymentSummary {
        stage_name: "Stage 4: K-1 Decision Deductibility".to_string(),
        total_poles: poles.len(),
        total_heads,
        total_cost_inr: total_cost,
        monitored_corridors: active_corridors.len(),
        execution_ms: t0.elapsed().as_secs_f64() * 1000.0,
        monitored_corridor_ids: active_corridors,
        poles,
    }
}