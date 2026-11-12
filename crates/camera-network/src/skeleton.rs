use crate::types::{canonical_edge, DirectedEdge, MacroCorridor};
use ahash::{AHashMap, AHashSet};
use road_graph_loader::RoadGraph;
use road_network_core::NodeId;

pub struct SkeletonGraph {
    pub junction_nodes: AHashSet<NodeId>,
    pub corridors: Vec<MacroCorridor>,
    pub edge_to_corridor: AHashMap<DirectedEdge, usize>,
    pub junction_outgoing: AHashMap<NodeId, Vec<(usize, NodeId, f32)>>,
}

pub fn build_skeleton(graph: &RoadGraph) -> SkeletonGraph {
    // 1. Build distinct physical road adjacency (undirected neighbor set)
    // If road goes from u to v, u and v are neighbors regardless of direction or lanes.
    let mut neighbors: AHashMap<NodeId, AHashSet<NodeId>> = AHashMap::with_capacity(graph.num_nodes);

    for edge_idx in 0..graph.num_edges {
        let desc = graph.edge_descriptor(edge_idx);
        if desc.u != desc.v {
            neighbors.entry(desc.u).or_default().insert(desc.v);
            neighbors.entry(desc.v).or_default().insert(desc.u);
        }
    }

    // 2. Identify True Decision Junctions:
    // A node is a junction IF AND ONLY IF:
    // - Degree == 1 (cul-de-sac / dead-end leaf)
    // - Degree >= 3 (T-junction, crossroad, multi-way fork)
    // Degree == 2 nodes are continuous road points (curves, speed changes, OSM way seams).
    let mut junction_nodes = AHashSet::with_capacity(graph.num_nodes / 6);
    for (&node, adj) in &neighbors {
        if adj.len() != 2 {
            junction_nodes.insert(node);
        }
    }

    // 3. Trace complete continuous corridors from Junction to Junction
    let mut corridors = Vec::new();
    let mut visited_directed_steps = AHashSet::new();
    let mut edge_to_corridor = AHashMap::new();
    let mut junction_outgoing: AHashMap<NodeId, Vec<(usize, NodeId, f32)>> = AHashMap::new();

    let junction_list: Vec<NodeId> = junction_nodes.iter().copied().collect();

    for start_id in junction_list {
        for &(first_target_id, initial_edge_idx) in graph.outgoing_edges(start_id) {
            let initial_step = (start_id, first_target_id);
            if !visited_directed_steps.insert(initial_step) {
                continue;
            }

            let initial_desc = graph.edge_descriptor(initial_edge_idx);
            let mut chain_edges = vec![initial_step];
            let mut total_len = initial_desc.length_m;
            let lanes_fwd = initial_desc.lanes.max(1);
            let speed_limit = initial_desc.maxspeed_kmh;

            let mut curr_id = first_target_id;

            // Follow the continuous road through all degree-2 curve nodes
            while !junction_nodes.contains(&curr_id) {
                let prev_node = chain_edges.last().unwrap().0;
                let outgoing = graph.outgoing_edges(curr_id);

                // Find the forward continuation (not turning back to the node we just came from)
                if let Some(&(next_id, edge_idx)) = outgoing.iter().find(|(nxt, _)| *nxt != prev_node) {
                    let next_step = (curr_id, next_id);
                    if !visited_directed_steps.insert(next_step) {
                        break;
                    }
                    let desc = graph.edge_descriptor(edge_idx);
                    chain_edges.push(next_step);
                    total_len += desc.length_m;
                    curr_id = next_id;
                } else {
                    break;
                }
            }

            let end_id = curr_id;
            let corr_id = corridors.len();

            let mid_idx = chain_edges.len() / 2;
            let rep_edge = chain_edges[mid_idx];
            let rep_edge_idx = graph
                .get_edge_index(rep_edge.0, rep_edge.1)
                .unwrap_or(initial_edge_idx);
            let rep_desc = graph.edge_descriptor(rep_edge_idx);

            let (p0, p1) = if rep_desc.geometry.len() >= 2 {
                (rep_desc.geometry[0], rep_desc.geometry[1])
            } else {
                ([19.06, 72.85], [19.061, 72.851])
            };

            let bearing = road_network_core::initial_bearing(p0[0], p0[1], p1[0], p1[1]) as f32;
            let midpoint = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5, bearing);

            let is_two_way = graph.get_edge_index(initial_step.1, initial_step.0).is_some();

            for &e in &chain_edges {
                edge_to_corridor.insert(canonical_edge(e.0, e.1), corr_id);
            }

            junction_outgoing
                .entry(start_id)
                .or_default()
                .push((corr_id, end_id, total_len));
            if is_two_way {
                junction_outgoing
                    .entry(end_id)
                    .or_default()
                    .push((corr_id, start_id, total_len));
            }

            corridors.push(MacroCorridor {
                id: corr_id,
                start_junction: start_id,
                end_junction: end_id,
                total_length_m: total_len,
                lanes_forward: lanes_fwd,
                lanes_backward: lanes_fwd,
                maxspeed_kmh: speed_limit,
                is_two_way,
                raw_edges: chain_edges,
                representative_edge: rep_edge,
                midpoint,
            });
        }
    }

    SkeletonGraph {
        junction_nodes,
        corridors,
        edge_to_corridor,
        junction_outgoing,
    }
}