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

pub fn build_skeleton_from_road_graph(graph: &RoadGraph) -> SkeletonGraph {
    let mut in_degrees: AHashMap<NodeId, usize> = AHashMap::with_capacity(graph.num_nodes);
    let mut out_degrees: AHashMap<NodeId, usize> = AHashMap::with_capacity(graph.num_nodes);

    for edge_idx in 0..graph.num_edges {
        let desc = graph.edge_descriptor(edge_idx);
        *out_degrees.entry(desc.u).or_default() += 1;
        *in_degrees.entry(desc.v).or_default() += 1;
    }

    let mut junction_nodes = AHashSet::with_capacity(graph.num_nodes / 3);

    for edge_idx in 0..graph.num_edges {
        let desc = graph.edge_descriptor(edge_idx);
        for &nid in &[desc.u, desc.v] {
            let in_d = in_degrees.get(&nid).copied().unwrap_or(0);
            let out_d = out_degrees.get(&nid).copied().unwrap_or(0);

            // True junctions are decision points: degree != 2, dead ends, or perimeter boundary nodes
            if in_d != 1 || out_d != 1 {
                junction_nodes.insert(nid);
            }
        }
    }

    let mut corridors = Vec::new();
    let mut visited_edges = AHashSet::new();
    let mut edge_to_corridor = AHashMap::new();
    let mut junction_outgoing: AHashMap<NodeId, Vec<(usize, NodeId, f32)>> = AHashMap::new();

    for &start_id in &junction_nodes {
        for &(first_target_id, initial_edge_idx) in graph.outgoing_edges(start_id) {
            let step = (start_id, first_target_id);
            if !visited_edges.insert(step) {
                continue;
            }

            let initial_desc = graph.edge_descriptor(initial_edge_idx);
            let mut chain_edges = vec![step];
            let mut total_len = initial_desc.length_m;
            let lanes_fwd = initial_desc.lanes.max(1);
            let speed_limit = initial_desc.maxspeed_kmh;

            let mut curr_id = first_target_id;

            while !junction_nodes.contains(&curr_id) {
                let prev_u = chain_edges.last().unwrap().0;
                let outgoing = graph.outgoing_edges(curr_id);

                if let Some(&(next_id, edge_idx)) = outgoing.iter().find(|(nxt, _)| *nxt != prev_u) {
                    let next_step = (curr_id, next_id);
                    if !visited_edges.insert(next_step) {
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
            let rep_edge_idx = graph.get_edge_index(rep_edge.0, rep_edge.1).unwrap_or(initial_edge_idx);
            let rep_desc = graph.edge_descriptor(rep_edge_idx);

            let (p0, p1) = if rep_desc.geometry.len() >= 2 {
                (rep_desc.geometry[0], rep_desc.geometry[1])
            } else {
                ([19.06, 72.85], [19.061, 72.851])
            };

            let bearing = road_network_core::initial_bearing(p0[0], p0[1], p1[0], p1[1]) as f32;
            let midpoint = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5, bearing);

            let is_two_way = graph.get_edge_index(step.1, step.0).is_some();

            for &e in &chain_edges {
                edge_to_corridor.insert(canonical_edge(e.0, e.1), corr_id);
            }

            junction_outgoing.entry(start_id).or_default().push((corr_id, end_id, total_len));
            if is_two_way {
                junction_outgoing.entry(end_id).or_default().push((corr_id, start_id, total_len));
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