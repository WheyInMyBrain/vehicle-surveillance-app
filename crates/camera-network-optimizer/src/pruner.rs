use crate::cuts::TopologicalCuts;
use crate::scratchpad::DijkstraScratchpad;
use crate::skeleton::SkeletonGraph;
use crate::types::OptimizerConfig;
use ahash::AHashSet;
use road_network_core::NodeId;

pub struct PruningResult {
    pub active_corridors: AHashSet<usize>,
    pub pruned_count: usize,
}

pub fn prune_redundant_corridors(
    skeleton: &SkeletonGraph,
    cuts: &TopologicalCuts,
    protected_arterials: &AHashSet<usize>,
    config: &OptimizerConfig,
) -> PruningResult {
    let mut active = (0..skeleton.corridors.len()).collect::<AHashSet<usize>>();

    let junction_list: Vec<_> = skeleton.junction_nodes.iter().copied().collect();
    let mut scratchpad = DijkstraScratchpad::new(&junction_list);
    let mut total_pruned = 0usize;

    // --- PASS 1: Density & Length Greedy Elimination ---
    // Corridors that have more neighboring active corridors are evaluated first
    let mut candidates: Vec<usize> = (0..skeleton.corridors.len())
        .filter(|id| !cuts.locked_corridors.contains(id) && !protected_arterials.contains(id))
        .collect();

    // Sort by length ascending (short alleys collapse first)
    candidates.sort_by(|&a, &b| {
        skeleton.corridors[a].total_length_m
            .partial_cmp(&skeleton.corridors[b].total_length_m)
            .unwrap()
            .then_with(|| a.cmp(&b))
    });

    for &corr_id in &candidates {
        if try_prune_corridor(corr_id, &mut active, skeleton, &mut scratchpad, config) {
            total_pruned += 1;
        }
    }

    // --- PASS 2: Second-Chance Reverse Sweep ---
    // Now that the graph is sparser, re-test the non-locked active cameras in reverse order.
    // Often, cameras that had to be kept early in Pass 1 can now be safely dropped!
    let mut pass2_candidates: Vec<usize> = active
        .iter()
        .copied()
        .filter(|id| !cuts.locked_corridors.contains(id) && !protected_arterials.contains(id))
        .collect();

    pass2_candidates.sort_by(|&a, &b| {
        skeleton.corridors[b].total_length_m
            .partial_cmp(&skeleton.corridors[a].total_length_m)
            .unwrap()
            .then_with(|| b.cmp(&a))
    });

    for &corr_id in &pass2_candidates {
        if active.contains(&corr_id) {
            if try_prune_corridor(corr_id, &mut active, skeleton, &mut scratchpad, config) {
                total_pruned += 1;
            }
        }
    }

    PruningResult {
        active_corridors: active,
        pruned_count: total_pruned,
    }
}

#[inline(always)]
fn try_prune_corridor(
    corr_id: usize,
    active: &mut AHashSet<usize>,
    skeleton: &SkeletonGraph,
    scratchpad: &mut DijkstraScratchpad,
    config: &OptimizerConfig,
) -> bool {
    active.remove(&corr_id);
    let corr = &skeleton.corridors[corr_id];

    let forward_violates = test_dark_path_violation(
        corr.end_junction,
        corr.total_length_m,
        active,
        skeleton,
        scratchpad,
        config,
    );

    let backward_violates = if corr.is_two_way {
        test_dark_path_violation(
            corr.start_junction,
            corr.total_length_m,
            active,
            skeleton,
            scratchpad,
            config,
        )
    } else {
        false
    };

    if forward_violates || backward_violates {
        // Essential seal: restore camera
        active.insert(corr_id);
        false
    } else {
        // Redundant: successfully eliminated!
        true
    }
}

fn test_dark_path_violation(
    start_junction: NodeId,
    initial_dist: f32,
    active_cameras: &AHashSet<usize>,
    skeleton: &SkeletonGraph,
    scratchpad: &mut DijkstraScratchpad,
    config: &OptimizerConfig,
) -> bool {
    scratchpad.reset();
    scratchpad.push(start_junction, initial_dist, 1);

    while let Some((curr, dist, hops)) = scratchpad.pop_min() {
        if dist > config.max_dark_distance_m || hops > config.max_dark_hops {
            return true;
        }

        if let Some(edges) = skeleton.junction_outgoing.get(&curr) {
            let mut dark_branches = 0usize;

            for &(cid, nxt_junction, len) in edges {
                if active_cameras.contains(&cid) {
                    continue;
                }

                dark_branches += 1;
                let next_dist = dist + len;

                if next_dist > config.max_dark_distance_m {
                    return true;
                }

                scratchpad.push(nxt_junction, next_dist, hops + 1);
            }

            if dark_branches >= config.max_unmonitored_forks && dist > 150.0 {
                return true;
            }
        }
    }

    false
}