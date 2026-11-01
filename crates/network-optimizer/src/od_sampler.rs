use engine_core::{CityGraph, NodeId};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use rand::seq::IndexedRandom;
use rand::RngExt;
use rayon::prelude::*;

pub type DirectedEdge = (NodeId, NodeId);

/// Generates a realistic hybrid traffic corridor mix:
/// - 50% arterial highway shortest paths (commuters, taxis)
/// - 50% stochastic chaotic wanders (local deliveries, bypass runs)
pub fn generate_synthetic_od_corridors(
    city: &CityGraph,
    num_trips: usize,
    min_intersections: usize,
) -> Vec<Vec<DirectedEdge>> {
    let all_node_indices: Vec<NodeIndex> = city.graph.node_indices().collect();
    if all_node_indices.len() < 2 {
        return vec![];
    }

    let arterial_count = num_trips / 2;
    let chaotic_count = num_trips - arterial_count;

    // 1. Arterial Commute Corridors
    let arterial_corridors: Vec<Vec<DirectedEdge>> = (0..arterial_count)
        .into_par_iter()
        .filter_map(|_| {
            let mut rng = rand::rng();
            let src = *all_node_indices.choose(&mut rng)?;
            let dst = *all_node_indices.choose(&mut rng)?;
            if src == dst {
                return None;
            }

            let path = find_hierarchy_path(city, src, dst)?;
            if path.len() >= min_intersections {
                Some(nodes_to_edges(city, &path))
            } else {
                None
            }
        })
        .collect();

    // 2. Chaotic Local Wanders (ensures secondary & residential streets are sampled)
    let chaotic_corridors: Vec<Vec<DirectedEdge>> = (0..chaotic_count)
        .into_par_iter()
        .filter_map(|_| {
            let mut rng = rand::rng();
            let mut curr = *all_node_indices.choose(&mut rng)?;
            let mut visited = ahash::AHashSet::with_capacity(min_intersections + 10);
            let mut node_path = Vec::with_capacity(min_intersections + 10);
            node_path.push(curr);
            visited.insert(curr);

            for _ in 0..(min_intersections + 15) {
                let edges: Vec<_> = city.graph.edges_directed(curr, Direction::Outgoing).collect();
                if edges.is_empty() {
                    break;
                }

                // Weighted random walk: 60% follow highway, 40% take any local exit
                let chosen_edge = if rng.random_bool(0.60) {
                    edges.iter().min_by(|a, b| {
                        a.weight()
                            .highway_type
                            .hierarchy_weight()
                            .partial_cmp(&b.weight().highway_type.hierarchy_weight())
                            .unwrap()
                    })?
                } else {
                    edges.choose(&mut rng)?
                };

                let next = chosen_edge.target();
                if visited.contains(&next) {
                    // Loop avoidance
                    continue;
                }

                visited.insert(next);
                node_path.push(next);
                curr = next;
            }

            if node_path.len() >= min_intersections {
                Some(nodes_to_edges(city, &node_path))
            } else {
                None
            }
        })
        .collect();

    let mut all_corridors = arterial_corridors;
    all_corridors.extend(chaotic_corridors);
    all_corridors
}

fn nodes_to_edges(city: &CityGraph, nodes: &[NodeIndex]) -> Vec<DirectedEdge> {
    let mut edges = Vec::with_capacity(nodes.len() - 1);
    for window in nodes.windows(2) {
        let u = city.reverse_node_map[&window[0]];
        let v = city.reverse_node_map[&window[1]];
        edges.push((u, v));
    }
    edges
}

fn find_hierarchy_path(
    city: &CityGraph,
    start: NodeIndex,
    goal: NodeIndex,
) -> Option<Vec<NodeIndex>> {
    use ordered_float::OrderedFloat;
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(Copy, Clone, Eq, PartialEq)]
    struct State {
        cost: OrderedFloat<f64>,
        node: NodeIndex,
    }

    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other.cost.cmp(&self.cost)
        }
    }
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut dist = ahash::AHashMap::new();
    let mut prev: ahash::AHashMap<NodeIndex, NodeIndex> = ahash::AHashMap::new();
    let mut heap = BinaryHeap::new();

    dist.insert(start, OrderedFloat(0.0));
    heap.push(State {
        cost: OrderedFloat(0.0),
        node: start,
    });

    while let Some(State { cost, node }) = heap.pop() {
        if node == goal {
            let mut path = vec![goal];
            let mut curr = goal;
            while let Some(&p) = prev.get(&curr) {
                path.push(p);
                curr = p;
            }
            path.reverse();
            return Some(path);
        }

        if cost > *dist.get(&node).unwrap_or(&OrderedFloat(f64::INFINITY)) {
            continue;
        }

        for edge in city.graph.edges_directed(node, Direction::Outgoing) {
            let next = edge.target();
            let meta = edge.weight();
            let edge_cost = meta.length_m * meta.highway_type.hierarchy_weight();
            let next_cost = cost + OrderedFloat(edge_cost);

            if next_cost < *dist.get(&next).unwrap_or(&OrderedFloat(f64::INFINITY)) {
                dist.insert(next, next_cost);
                prev.insert(next, node);
                heap.push(State {
                    cost: next_cost,
                    node: next,
                });
            }
        }
    }

    None
}