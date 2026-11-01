use crate::types::{DirectedEdge, OptimizedCameraPole};
use ahash::{AHashMap, AHashSet};
use engine_core::CityGraph;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use rand::seq::IndexedRandom;
use rand::RngExt;

pub struct AdversarialOptimizerConfig {
    pub initial_poles: usize,
    pub epochs: usize,
    pub max_dark_distance_m: f64,
    pub junction_entropy_weight: f64,
}

impl Default for AdversarialOptimizerConfig {
    fn default() -> Self {
        Self {
            initial_poles: 450,
            epochs: 60,
            max_dark_distance_m: 600.0,
            junction_entropy_weight: 20.0,
        }
    }
}

#[inline(always)]
fn to_canonical(u: engine_core::NodeId, v: engine_core::NodeId) -> DirectedEdge {
    if u < v { (u, v) } else { (v, u) }
}

fn find_longest_dark_path_from(
    city: &CityGraph,
    start: NodeIndex,
    monitored_canonical: &AHashSet<DirectedEdge>,
    max_search_depth_m: f64,
) -> (f64, Vec<DirectedEdge>) {
    let mut max_dist = 0.0;
    let mut longest_edge_path = Vec::new();
    let mut stack = vec![(start, 0.0, Vec::new(), vec![start])];

    while let Some((curr, dist, edge_path, path_nodes)) = stack.pop() {
        if dist > max_dist {
            max_dist = dist;
            longest_edge_path = edge_path.clone();
        }

        if dist >= max_search_depth_m || path_nodes.len() > 25 {
            continue;
        }

        for edge in city.graph.edges_directed(curr, Direction::Outgoing) {
            let next = edge.target();
            let u_id = city.reverse_node_map[&curr];
            let v_id = city.reverse_node_map[&next];
            let canonical = to_canonical(u_id, v_id);

            // Blocked if monitored
            if monitored_canonical.contains(&canonical) {
                continue;
            }

            if !path_nodes.contains(&next) {
                let step_len = edge.weight().length_m;
                let mut new_edges = edge_path.clone();
                new_edges.push(canonical);

                let mut new_nodes = path_nodes.clone();
                new_nodes.push(next);

                stack.push((next, dist + step_len, new_edges, new_nodes));
            }
        }
    }

    (max_dist, longest_edge_path)
}

fn count_unmonitored_outgoing_branches(
    city: &CityGraph,
    node: NodeIndex,
    monitored_canonical: &AHashSet<DirectedEdge>,
) -> Vec<DirectedEdge> {
    let mut dark_edges = Vec::new();
    for edge in city.graph.edges_directed(node, Direction::Outgoing) {
        let u_id = city.reverse_node_map[&node];
        let v_id = city.reverse_node_map[&edge.target()];
        let canonical = to_canonical(u_id, v_id);

        if !monitored_canonical.contains(&canonical) {
            dark_edges.push(canonical);
        }
    }
    dark_edges
}

pub fn train_adversarial_camera_network(
    city: &CityGraph,
    initial_seed_poles: Vec<OptimizedCameraPole>,
    config: AdversarialOptimizerConfig,
) -> Vec<OptimizedCameraPole> {
    let mut rng = rand::rng();

    let mut monitored_canonical: AHashSet<DirectedEdge> = AHashSet::new();
    let mut locked_arterials: AHashSet<DirectedEdge> = AHashSet::new();
    let mut camera_hits: AHashMap<DirectedEdge, usize> = AHashMap::new();

    // Lock top 80% highest-traffic arterial cameras
    let mut sorted_seed = initial_seed_poles.clone();
    sorted_seed.sort_by_key(|p| std::cmp::Reverse(p.total_intercepted_trips));
    let lock_count = (sorted_seed.len() as f64 * 0.80).ceil() as usize;

    for (idx, pole) in sorted_seed.into_iter().enumerate() {
        let canonical = to_canonical(pole.edge_from, pole.edge_to);
        monitored_canonical.insert(canonical);
        camera_hits.insert(canonical, pole.total_intercepted_trips.max(1));

        if idx < lock_count {
            locked_arterials.insert(canonical);
        }
    }

    // Only sample nodes that have outgoing drivable edges
    let drivable_nodes: Vec<NodeIndex> = city
        .graph
        .node_indices()
        .filter(|&n| city.graph.edges_directed(n, Direction::Outgoing).count() > 0)
        .collect();

    let mut best_network = monitored_canonical.clone();
    let mut lowest_max_escape = f64::INFINITY;

    for _epoch in 0..config.epochs {
        let test_nodes: Vec<NodeIndex> = (0..80)
            .filter_map(|_| drivable_nodes.choose(&mut rng).copied())
            .collect();

        let mut critical_escape_edges = Vec::new();
        let mut current_max_escape = 0.0f64;

        for &start in &test_nodes {
            let (escape_len, path) = find_longest_dark_path_from(
                city,
                start,
                &monitored_canonical,
                config.max_dark_distance_m * 1.3,
            );

            if escape_len > current_max_escape {
                current_max_escape = escape_len;
            }

            if escape_len >= config.max_dark_distance_m && !path.is_empty() {
                critical_escape_edges.extend(path);
            }
        }

        if current_max_escape < lowest_max_escape {
            lowest_max_escape = current_max_escape;
            best_network = monitored_canonical.clone();
        }

        if current_max_escape <= config.max_dark_distance_m {
            break;
        }

        // Branch disambiguation check
        let mut fork_edges_to_seal = Vec::new();
        for &node in &test_nodes {
            let dark_branches = count_unmonitored_outgoing_branches(city, node, &monitored_canonical);
            if dark_branches.len() >= 2 {
                if let Some(&choice) = dark_branches.choose(&mut rng) {
                    fork_edges_to_seal.push(choice);
                }
            }
        }

        // Mutate only non-arterial cameras
        let mut mutable_cameras: Vec<DirectedEdge> = monitored_canonical
            .iter()
            .copied()
            .filter(|e| !locked_arterials.contains(e))
            .collect();

        mutable_cameras.sort_by_key(|e| *camera_hits.get(e).unwrap_or(&0));

        let relocate_count = (mutable_cameras.len() as f64 * 0.12).ceil() as usize;

        for i in 0..relocate_count.min(mutable_cameras.len()) {
            let old_cam = mutable_cameras[i];
            monitored_canonical.remove(&old_cam);
            camera_hits.remove(&old_cam);

            let new_target = if rng.random_bool(0.70) && !critical_escape_edges.is_empty() {
                critical_escape_edges.choose(&mut rng).copied()
            } else if !fork_edges_to_seal.is_empty() {
                fork_edges_to_seal.choose(&mut rng).copied()
            } else {
                None
            };

            if let Some(target) = new_target {
                monitored_canonical.insert(target);
                camera_hits.insert(target, 5);
            }
        }
    }

    // Reconstruct valid directed hardware poles
    best_network
        .into_iter()
        .filter_map(|(u, v)| {
            let u_idx = *city.node_map.get(&u)?;
            let v_idx = *city.node_map.get(&v)?;

            let forward_exists = city.graph.contains_edge(u_idx, v_idx);
            let reverse_exists = city.graph.contains_edge(v_idx, u_idx);

            if !forward_exists && !reverse_exists {
                return None;
            }

            let is_two_way = forward_exists && reverse_exists;
            // Orient along the legal direction of travel
            let (from_id, to_id) = if forward_exists { (u, v) } else { (v, u) };

            let u_meta = &city.graph[u_idx];
            let v_meta = &city.graph[v_idx];

            Some(OptimizedCameraPole {
                camera_id: format!("CAM_DISAMB_{}_{}", from_id, to_id),
                edge_from: from_id,
                edge_to: to_id,
                lat: (u_meta.lat + v_meta.lat) / 2.0,
                lon: (u_meta.lon + v_meta.lon) / 2.0,
                total_intercepted_trips: 0,
                marginal_new_trips: 0,
                is_two_way,
            })
        })
        .collect()
}