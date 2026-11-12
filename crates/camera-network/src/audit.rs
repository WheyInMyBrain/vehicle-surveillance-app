use crate::skeleton::SkeletonGraph;
use ahash::AHashSet;
use rand::seq::IndexedRandom;
use rand::RngExt;
use road_network_core::NodeId;

#[derive(Debug, Clone)]
pub struct AuditReport {
    pub total_trips: usize,
    pub trip_interception_pct: f64,
    pub multi_hit_pct: f64,
    pub zero_hit_escapes: usize,
    pub max_transit_dark_run_m: f32,
    pub max_terminal_dark_run_m: f32,
    pub max_terminal_branching: usize,
}

pub fn run_audit(
    skeleton: &SkeletonGraph,
    monitored_corridors: &AHashSet<usize>,
    num_trips: usize,
) -> AuditReport {
    let mut rng = rand::rng();

    let valid_start_junctions: Vec<NodeId> = skeleton
        .junction_nodes
        .iter()
        .copied()
        .filter(|nid| {
            skeleton
                .junction_outgoing
                .get(nid)
                .map(|edges| !edges.is_empty())
                .unwrap_or(false)
        })
        .collect();

    if valid_start_junctions.is_empty() {
        return AuditReport {
            total_trips: 0,
            trip_interception_pct: 0.0,
            multi_hit_pct: 0.0,
            zero_hit_escapes: 0,
            max_transit_dark_run_m: 0.0,
            max_terminal_dark_run_m: 0.0,
            max_terminal_branching: 0,
        };
    }

    let mut single_hit = 0usize;
    let mut multi_hit = 0usize;
    let mut zero_hit = 0usize;
    let mut evaluated_trips = 0usize;

    let mut max_transit_dark_m = 0.0f32;
    let mut max_terminal_dark_m = 0.0f32;
    let mut max_terminal_branching = 1usize;

    for _ in 0..num_trips {
        let start = *valid_start_junctions.choose(&mut rng).unwrap();
        let mut curr = start;
        let mut visited_path: Vec<(usize, f32)> = Vec::new();
        let mut camera_hit_indices: Vec<usize> = Vec::new();
        let target_hops = rng.random_range(6..22);

        // Track visited nodes to prevent U-turn cycles in random walks
        let mut visited_nodes: AHashSet<NodeId> = AHashSet::new();
        visited_nodes.insert(curr);

        for hop_idx in 0..target_hops {
            if let Some(outgoing) = skeleton.junction_outgoing.get(&curr) {
                // Filter out immediate U-turns if other forward branches exist
                let forward_options: Vec<&(usize, NodeId, f32)> = outgoing
                    .iter()
                    .filter(|(_, nxt, _)| !visited_nodes.contains(nxt))
                    .collect();

                let &(cid, next_node, len) = if !forward_options.is_empty() {
                    forward_options.choose(&mut rng).unwrap()
                } else {
                    outgoing.choose(&mut rng).unwrap()
                };

                visited_path.push((cid, len));
                visited_nodes.insert(next_node);

                if monitored_corridors.contains(&cid) {
                    camera_hit_indices.push(hop_idx);
                }

                curr = next_node;
            } else {
                break;
            }
        }

        if visited_path.len() >= 3 {
            evaluated_trips += 1;
            let hits = camera_hit_indices.len();

            if hits >= 1 {
                single_hit += 1;

                if hits >= 2 {
                    multi_hit += 1;
                    for window in camera_hit_indices.windows(2) {
                        let from_idx = window[0];
                        let to_idx = window[1];
                        let transit_dist: f32 = visited_path[from_idx + 1..to_idx]
                            .iter()
                            .map(|&(_, len)| len)
                            .sum();
                        if transit_dist > max_transit_dark_m {
                            max_transit_dark_m = transit_dist;
                        }
                    }
                }

                let last_cam_idx = *camera_hit_indices.last().unwrap();
                let terminal_dist: f32 = visited_path[last_cam_idx + 1..]
                    .iter()
                    .map(|&(_, len)| len)
                    .sum();
                if terminal_dist > max_terminal_dark_m {
                    max_terminal_dark_m = terminal_dist;
                }

                if let Some(final_outgoing) = skeleton.junction_outgoing.get(&curr) {
                    let dark_branches = final_outgoing
                        .iter()
                        .filter(|(c, _, _)| !monitored_corridors.contains(c))
                        .count();
                    if dark_branches > max_terminal_branching {
                        max_terminal_branching = dark_branches;
                    }
                }
            } else {
                zero_hit += 1;
            }
        }
    }

    let total = evaluated_trips.max(1);

    AuditReport {
        total_trips: evaluated_trips,
        trip_interception_pct: (single_hit as f64 / total as f64) * 100.0,
        multi_hit_pct: (multi_hit as f64 / total as f64) * 100.0,
        zero_hit_escapes: zero_hit,
        max_transit_dark_run_m: max_transit_dark_m,
        max_terminal_dark_run_m: max_terminal_dark_m,
        max_terminal_branching,
    }
}