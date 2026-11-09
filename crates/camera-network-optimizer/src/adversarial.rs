use crate::cuts::TopologicalCuts;
use crate::skeleton::SkeletonGraph;
use crate::types::OptimizerConfig;
use ahash::{AHashMap, AHashSet};
use rand::seq::IndexedRandom;
use rand::RngExt;
use road_graph_loader::RoadGraph;
use road_network_core::NodeId;

pub struct FlowAnalysis {
    pub protected_arterials: AHashSet<usize>,
    pub corridor_hit_counts: AHashMap<usize, usize>,
}

/// Simulates commuter trips and identifies high-volume corridors needing multi-hit coverage
pub fn analyze_traffic_flow(
    graph: &RoadGraph,
    skeleton: &SkeletonGraph,
    config: &OptimizerConfig,
) -> FlowAnalysis {
    let mut rng = rand::rng();
    let junctions: Vec<NodeId> = skeleton.junction_nodes.iter().copied().collect();
    let mut corridor_hit_counts: AHashMap<usize, usize> = AHashMap::new();

    if junctions.len() < 2 {
        return FlowAnalysis {
            protected_arterials: AHashSet::new(),
            corridor_hit_counts,
        };
    }

    // Sample OD trips and count corridor usage, weighting arterial corridors by speed limit
    for _ in 0..config.num_od_eval_trips {
        let mut curr = *junctions.choose(&mut rng).unwrap();
        let mut hops = 0;
        let max_hops = rng.random_range(6..18);

        while hops < max_hops {
            if let Some(outgoing) = skeleton.junction_outgoing.get(&curr) {
                if outgoing.is_empty() {
                    break;
                }

                // Prefer higher-speed arterial roads (70% probability)
                let &(corr_id, next_node, _) = if rng.random_bool(0.70) {
                    outgoing
                        .iter()
                        .max_by_key(|(cid, _, _)| {
                            let corr = &skeleton.corridors[*cid];
                            if let Some(edge_idx) = graph.get_edge_index(corr.representative_edge.0, corr.representative_edge.1) {
                                graph.edge_descriptor(edge_idx).maxspeed_kmh
                            } else {
                                corr.maxspeed_kmh
                            }
                        })
                        .unwrap()
                } else {
                    outgoing.choose(&mut rng).unwrap()
                };

                *corridor_hit_counts.entry(corr_id).or_default() += 1;
                curr = next_node;
                hops += 1;
            } else {
                break;
            }
        }
    }

    // Top 20% highest-traffic corridors are protected arterials
    let mut sorted_corridors: Vec<(usize, usize)> = corridor_hit_counts.iter().map(|(&k, &v)| (k, v)).collect();
    sorted_corridors.sort_by_key(|&(_, count)| std::cmp::Reverse(count));

    let protect_n = (sorted_corridors.len() as f32 * 0.20).ceil() as usize;
    let protected_arterials = sorted_corridors.iter().take(protect_n).map(|&(cid, _)| cid).collect();

    FlowAnalysis {
        protected_arterials,
        corridor_hit_counts,
    }
}

/// Adversarial Monte Carlo: sends evasive agents through dark pockets and seals leaks
pub fn seal_adversarial_leaks(
    active_corridors: &mut AHashSet<usize>,
    skeleton: &SkeletonGraph,
    _cuts: &TopologicalCuts,
    _protected_arterials: &AHashSet<usize>,
    config: &OptimizerConfig,
) {
    let mut rng = rand::rng();
    let junctions: Vec<NodeId> = skeleton.junction_nodes.iter().copied().collect();

    for _ in 0..config.adversarial_epochs {
        let mut leak_corridors = Vec::new();

        for _ in 0..60 {
            let start = *junctions.choose(&mut rng).unwrap();
            let mut curr = start;
            let mut dark_len = 0.0f32;
            let mut dark_path = Vec::new();

            for _ in 0..8 {
                if let Some(outgoing) = skeleton.junction_outgoing.get(&curr) {
                    let unmonitored: Vec<_> = outgoing
                        .iter()
                        .filter(|(cid, _, _)| !active_corridors.contains(cid))
                        .collect();

                    if let Some(&&(cid, nxt, len)) = unmonitored.choose(&mut rng) {
                        dark_len += len;
                        dark_path.push(cid);
                        curr = nxt;
                        if dark_len >= config.max_dark_distance_m {
                            leak_corridors.extend(dark_path);
                            break;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        if leak_corridors.is_empty() {
            break;
        }

        // Seal the most frequently abused evasion corridor
        if let Some(&sealed_corr) = leak_corridors.choose(&mut rng) {
            active_corridors.insert(sealed_corr);
        }
    }
}