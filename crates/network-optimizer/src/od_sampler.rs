use engine_core::{CityGraph, NodeId};
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use rand::seq::IndexedRandom;
use rayon::prelude::*;

pub type DirectedEdge = (NodeId, NodeId);

/// Generates balanced synthetic traffic corridors across the road network in parallel.
pub fn generate_synthetic_od_corridors(
    city: &CityGraph,
    num_trips: usize,
    min_intersections: usize,
) -> Vec<Vec<DirectedEdge>> {
    let all_node_indices: Vec<NodeIndex> = city.graph.node_indices().collect();
    if all_node_indices.len() < 2 {
        return vec![];
    }

    // Parallel chunk generation via Rayon
    (0..num_trips)
        .into_par_iter()
        .filter_map(|_| {
            let mut rng = rand::rng();
            let src_idx = *all_node_indices.choose(&mut rng)?;
            let dst_idx = *all_node_indices.choose(&mut rng)?;

            if src_idx == dst_idx {
                return None;
            }

            // Dijkstra with road hierarchy penalties
            let node_path = find_hierarchy_path(city, src_idx, dst_idx)?;

            if node_path.len() >= min_intersections {
                let mut edge_path = Vec::with_capacity(node_path.len() - 1);
                for window in node_path.windows(2) {
                    let u_id = city.reverse_node_map[&window[0]];
                    let v_id = city.reverse_node_map[&window[1]];
                    edge_path.push((u_id, v_id));
                }
                Some(edge_path)
            } else {
                None
            }
        })
        .collect()
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