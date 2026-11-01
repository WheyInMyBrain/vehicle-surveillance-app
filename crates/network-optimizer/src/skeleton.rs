use crate::types::{canonical_edge, DirectedEdge, MacroCorridor};
use ahash::{AHashMap, AHashSet};
use engine_core::{CityGraph, NodeId};
use petgraph::visit::EdgeRef;
use petgraph::Direction;

pub struct SkeletonGraph {
    /// True topological junctions (intersections with degree != 2, dead ends, or signals)
    pub junction_nodes: AHashSet<NodeId>,
    /// All contracted macro-corridors connecting junctions
    pub corridors: Vec<MacroCorridor>,
    /// Lookup from raw canonical edge -> Corridor ID
    pub edge_to_corridor: AHashMap<DirectedEdge, usize>,
    /// Adjacency: Junction NodeId -> list of incident Corridor IDs
    pub junction_to_corridors: AHashMap<NodeId, Vec<usize>>,
    /// Fast directed lookup: (start_junction, end_junction) -> Corridor ID
    pub directed_junction_pair_to_corridor: AHashMap<(NodeId, NodeId), usize>,
}

impl SkeletonGraph {
    /// Unpacks a compressed path of junction hops into the full sequence of raw OSM geometry nodes.
    pub fn expand_macro_path_to_raw(&self, macro_path: &[NodeId]) -> Vec<NodeId> {
        if macro_path.len() <= 1 {
            return macro_path.to_vec();
        }

        let mut raw_expanded = Vec::with_capacity(macro_path.len() * 4);

        for window in macro_path.windows(2) {
            let u = window[0];
            let v = window[1];

            if u == v {
                continue;
            }

            if raw_expanded.last() != Some(&u) {
                raw_expanded.push(u);
            }

            // Case A: (u, v) is a direct macro-corridor between two junctions
            if let Some(&corr_id) = self.directed_junction_pair_to_corridor.get(&(u, v)) {
                let corr = &self.corridors[corr_id];
                if corr.start_node == u && corr.end_node == v {
                    for &internal in &corr.internal_nodes {
                        if raw_expanded.last() != Some(&internal) {
                            raw_expanded.push(internal);
                        }
                    }
                } else if corr.is_two_way && corr.start_node == v && corr.end_node == u {
                    for &internal in corr.internal_nodes.iter().rev() {
                        if raw_expanded.last() != Some(&internal) {
                            raw_expanded.push(internal);
                        }
                    }
                }
            }

            if raw_expanded.last() != Some(&v) {
                raw_expanded.push(v);
            }
        }

        raw_expanded
    }
}

/// Contracts all non-branching degree-2 paths in `CityGraph` into macro-corridors.
pub fn build_skeleton_graph(city: &CityGraph) -> SkeletonGraph {
    let mut junction_nodes = AHashSet::new();

    // 1. Identify all true topological junctions
    for node_idx in city.graph.node_indices() {
        let node_id = city.reverse_node_map[&node_idx];
        let meta = &city.graph[node_idx];

        let in_deg = city.graph.edges_directed(node_idx, Direction::Incoming).count();
        let out_deg = city.graph.edges_directed(node_idx, Direction::Outgoing).count();

        let is_pass_through = if meta.has_traffic_signal || meta.has_speed_bump {
            false
        } else if in_deg == 1 && out_deg == 1 {
            let in_neighbor = city.graph.edges_directed(node_idx, Direction::Incoming).next().unwrap().source();
            let out_neighbor = city.graph.edges_directed(node_idx, Direction::Outgoing).next().unwrap().target();
            in_neighbor != out_neighbor
        } else if in_deg == 2 && out_deg == 2 {
            let mut in_neighbors = AHashSet::new();
            for e in city.graph.edges_directed(node_idx, Direction::Incoming) {
                in_neighbors.insert(e.source());
            }
            let mut out_neighbors = AHashSet::new();
            for e in city.graph.edges_directed(node_idx, Direction::Outgoing) {
                out_neighbors.insert(e.target());
            }
            in_neighbors.len() == 2 && in_neighbors == out_neighbors
        } else {
            false
        };

        if !is_pass_through {
            junction_nodes.insert(node_id);
        }
    }

    // Isolated degree-2 cycles with 0 junctions: anchor at least one node
    for node_idx in city.graph.node_indices() {
        let node_id = city.reverse_node_map[&node_idx];
        if !junction_nodes.contains(&node_id) {
            let mut curr = node_idx;
            let mut cycle_detect = AHashSet::new();
            let mut found_junction = false;

            while cycle_detect.insert(curr) {
                let curr_id = city.reverse_node_map[&curr];
                if junction_nodes.contains(&curr_id) {
                    found_junction = true;
                    break;
                }
                if let Some(next_edge) = city.graph.edges_directed(curr, Direction::Outgoing).next() {
                    curr = next_edge.target();
                } else {
                    break;
                }
            }

            if !found_junction {
                junction_nodes.insert(node_id);
            }
        }
    }

    // 2. Trace chains from every junction along outgoing edges
    let mut corridors = Vec::new();
    let mut visited_directed_edges: AHashSet<DirectedEdge> = AHashSet::new();
    let mut edge_to_corridor: AHashMap<DirectedEdge, usize> = AHashMap::new();
    let mut junction_to_corridors: AHashMap<NodeId, Vec<usize>> = AHashMap::new();
    let mut directed_junction_pair_to_corridor: AHashMap<(NodeId, NodeId), usize> = AHashMap::new();

    for &start_junction_id in &junction_nodes {
        let start_idx = city.node_map[&start_junction_id];

        for edge_ref in city.graph.edges_directed(start_idx, Direction::Outgoing) {
            let first_target_idx = edge_ref.target();
            let first_target_id = city.reverse_node_map[&first_target_idx];
            let initial_edge = (start_junction_id, first_target_id);

            if !visited_directed_edges.insert(initial_edge) {
                continue;
            }

            let mut chain_raw_edges = vec![initial_edge];
            let mut internal_nodes = Vec::new();
            let mut total_length = edge_ref.weight().length_m;

            let mut curr_node_idx = first_target_idx;
            let mut curr_node_id = first_target_id;

            while !junction_nodes.contains(&curr_node_id) {
                internal_nodes.push(curr_node_id);

                let prev_u_id = chain_raw_edges.last().unwrap().0;

                let next_edge = city
                    .graph
                    .edges_directed(curr_node_idx, Direction::Outgoing)
                    .find(|e| city.reverse_node_map[&e.target()] != prev_u_id);

                match next_edge {
                    Some(e) => {
                        let next_target_idx = e.target();
                        let next_target_id = city.reverse_node_map[&next_target_idx];
                        let edge_step = (curr_node_id, next_target_id);

                        if !visited_directed_edges.insert(edge_step) {
                            break;
                        }

                        chain_raw_edges.push(edge_step);
                        total_length += e.weight().length_m;

                        curr_node_idx = next_target_idx;
                        curr_node_id = next_target_id;
                    }
                    None => break,
                }
            }

            let end_junction_id = curr_node_id;
            let corridor_id = corridors.len();

            let mid_idx = chain_raw_edges.len() / 2;
            let representative_edge = chain_raw_edges[mid_idx];

            let start_n_idx = city.node_map[&chain_raw_edges[0].0];
            let first_n_idx = city.node_map[&chain_raw_edges[0].1];
            let is_two_way = city.graph.contains_edge(start_n_idx, first_n_idx)
                && city.graph.contains_edge(first_n_idx, start_n_idx);

            for &edge in &chain_raw_edges {
                edge_to_corridor.insert(canonical_edge(edge.0, edge.1), corridor_id);
            }

            junction_to_corridors
                .entry(start_junction_id)
                .or_default()
                .push(corridor_id);
            junction_to_corridors
                .entry(end_junction_id)
                .or_default()
                .push(corridor_id);

            directed_junction_pair_to_corridor.insert((start_junction_id, end_junction_id), corridor_id);
            if is_two_way {
                directed_junction_pair_to_corridor.insert((end_junction_id, start_junction_id), corridor_id);
            }

            corridors.push(MacroCorridor {
                id: corridor_id,
                start_node: start_junction_id,
                end_node: end_junction_id,
                total_length_m: total_length,
                raw_edges: chain_raw_edges,
                internal_nodes,
                is_two_way,
                representative_edge,
            });
        }
    }

    SkeletonGraph {
        junction_nodes,
        corridors,
        edge_to_corridor,
        junction_to_corridors,
        directed_junction_pair_to_corridor,
    }
}