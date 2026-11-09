pub mod adversarial;
pub mod cuts;
pub mod pruner;
pub mod scratchpad;
pub mod skeleton;
pub mod types;

pub use cuts::{find_topological_cuts, TopologicalCuts};
pub use skeleton::{build_skeleton_from_road_graph, SkeletonGraph};
pub use types::{canonical_edge, DirectedEdge, MacroCorridor, MountType, CameraPole, OptimizerConfig};

use adversarial::{analyze_traffic_flow, seal_adversarial_leaks};
use pruner::prune_redundant_corridors;
use road_graph_loader::RoadGraph;

#[derive(Debug, Clone)]
pub struct OptimizationSummary {
    pub camera_poles: Vec<CameraPole>,
    pub total_poles: usize,
    pub total_camera_heads: usize,
    pub corridors_monitored: usize,
    pub corridors_pruned: usize,
    pub permanently_locked_gates: usize,
    pub execution_ms: f64,
}

/// Complete 4-stage optimization pipeline executed directly over RoadGraph
pub fn optimize_camera_network(
    graph: &RoadGraph,
    config: &OptimizerConfig,
) -> OptimizationSummary {
    let t0 = std::time::Instant::now();

    // 1. Stage 1: Contraction & Topological Cuts
    let skeleton = build_skeleton_from_road_graph(graph);
    let cuts = find_topological_cuts(graph, &skeleton);

    // 2. Stage 2: Flow Sampling & Protected Arterial Identification
    let flow = analyze_traffic_flow(graph, &skeleton, config);

    // 3. Stage 3: Reverse Zero-Escape Pruning
    let mut pruning_res = prune_redundant_corridors(&skeleton, &cuts, &flow.protected_arterials, config);

    // 4. Stage 4: Adversarial Leak Sealing
    seal_adversarial_leaks(&mut pruning_res.active_corridors, &skeleton, &cuts, &flow.protected_arterials, config);

    // 5. Construct Physical Hardware Poles
    let mut poles = Vec::with_capacity(pruning_res.active_corridors.len());
    let mut total_camera_heads = 0usize;

    for &corr_id in &pruning_res.active_corridors {
        let corr = &skeleton.corridors[corr_id];
        let (lat, lon, bearing) = corr.midpoint;

        let fwd_cams = ((corr.lanes_forward as f32) / (config.lanes_per_camera as f32)).ceil() as u8;
        let bwd_cams = if corr.is_two_way {
            ((corr.lanes_backward as f32) / (config.lanes_per_camera as f32)).ceil() as u8
        } else {
            0
        };

        let total_heads = (fwd_cams + bwd_cams).max(1);
        let total_lanes = corr.lanes_forward + if corr.is_two_way { corr.lanes_backward } else { 0 };

        let mount = if total_lanes >= 4 {
            MountType::OverheadGantry
        } else if cuts.perimeter_gateway_ids.contains(&corr_id) {
            MountType::IntersectionMast
        } else {
            MountType::VergePole
        };

        total_camera_heads += total_heads as usize;

        poles.push(CameraPole {
            pole_id: format!("POLE_CAM_{:05}", corr_id),
            edge_from: corr.representative_edge.0,
            edge_to: corr.representative_edge.1,
            lat,
            lon,
            bearing_deg: bearing,
            camera_heads: total_heads,
            lanes_covered: total_lanes,
            mount_type: mount,
            is_two_way: corr.is_two_way,
            is_perimeter_gate: cuts.perimeter_gateway_ids.contains(&corr_id),
            is_bridge_seal: cuts.bridge_corridor_ids.contains(&corr_id),
            intercepted_trips: flow.corridor_hit_counts.get(&corr_id).copied().unwrap_or(0),
        });
    }

    OptimizationSummary {
        total_poles: poles.len(),
        total_camera_heads,
        corridors_monitored: pruning_res.active_corridors.len(),
        corridors_pruned: pruning_res.pruned_count,
        permanently_locked_gates: cuts.locked_corridors.len(),
        camera_poles: poles,
        execution_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}