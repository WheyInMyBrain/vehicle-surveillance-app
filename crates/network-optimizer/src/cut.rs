use crate::skeleton::SkeletonGraph;
use crate::types::{canonical_edge, DirectedEdge};
use ahash::{AHashMap, AHashSet};
use engine_core::{CityGraph, NodeId};

#[derive(Debug, Clone)]
pub struct TopologicalCutReport {
    /// Macro-Corridor IDs that are strict topological bridges (never prunable)
    pub bridge_corridor_ids: AHashSet<usize>,
    /// Underlying raw canonical directed edges corresponding to bridges
    pub bridge_raw_edges: AHashSet<DirectedEdge>,
    /// Macro-Corridor IDs that act as perimeter entry/exit gateways
    pub perimeter_corridor_ids: AHashSet<usize>,
    /// All permanently locked corridor IDs (Bridges + Perimeter Gateways)
    pub mandatory_locked_corridors: AHashSet<usize>,
}

/// Identifies all topological bridges and perimeter gates in the skeleton graph.
pub fn find_topological_cuts(city: &CityGraph, skeleton: &SkeletonGraph) -> TopologicalCutReport {
    // -----------------------------------------------------------------
    // 1. Tarjan's Bridge Detection on Undirected Skeleton Graph
    // -----------------------------------------------------------------
    // Build adjacency list for the skeleton graph: Junction NodeId -> Vec<(Neighbor NodeId, Corridor ID)>
    let mut adj: AHashMap<NodeId, Vec<(NodeId, usize)>> = AHashMap::new();

    for corridor in &skeleton.corridors {
        let u = corridor.start_node;
        let v = corridor.end_node;

        if u != v {
            adj.entry(u).or_default().push((v, corridor.id));
            adj.entry(v).or_default().push((u, corridor.id));
        }
    }

    let mut discovery_time: AHashMap<NodeId, usize> = AHashMap::new();
    let mut low_time: AHashMap<NodeId, usize> = AHashMap::new();
    let mut bridge_corridor_ids: AHashSet<usize> = AHashSet::new();
    let mut timer = 0usize;

    for &junction in &skeleton.junction_nodes {
        if !discovery_time.contains_key(&junction) {
            tarjan_dfs(
                junction,
                None,
                &adj,
                &mut timer,
                &mut discovery_time,
                &mut low_time,
                &mut bridge_corridor_ids,
            );
        }
    }

    // -----------------------------------------------------------------
    // 2. Identify Perimeter Ingress / Egress Gateways
    // -----------------------------------------------------------------
    // A corridor is a perimeter gateway if one of its terminals has no outgoing/incoming
    // connections back into the city graph (boundary interface).
    let mut perimeter_corridor_ids: AHashSet<usize> = AHashSet::new();

    for corridor in &skeleton.corridors {
        let u_idx = city.node_map.get(&corridor.start_node);
        let v_idx = city.node_map.get(&corridor.end_node);

        if let (Some(&ui), Some(&vi)) = (u_idx, v_idx) {
            let u_deg = city.graph.edges(ui).count();
            let v_deg = city.graph.edges(vi).count();

            // Degree 1 in the directed graph means a dead-end entry/exit boundary
            if u_deg <= 1 || v_deg <= 1 {
                perimeter_corridor_ids.insert(corridor.id);
            }
        }
    }

    // -----------------------------------------------------------------
    // 3. Map to Raw Canonical Edges for Downstream Simulator
    // -----------------------------------------------------------------
    let mut bridge_raw_edges = AHashSet::new();
    let mut mandatory_locked_corridors = AHashSet::new();

    for &cid in &bridge_corridor_ids {
        mandatory_locked_corridors.insert(cid);
        let corr = &skeleton.corridors[cid];
        let (u, v) = corr.representative_edge;
        bridge_raw_edges.insert(canonical_edge(u, v));
    }

    for &cid in &perimeter_corridor_ids {
        mandatory_locked_corridors.insert(cid);
    }

    TopologicalCutReport {
        bridge_corridor_ids,
        bridge_raw_edges,
        perimeter_corridor_ids,
        mandatory_locked_corridors,
    }
}

fn tarjan_dfs(
    curr: NodeId,
    parent_edge: Option<usize>,
    adj: &AHashMap<NodeId, Vec<(NodeId, usize)>>,
    timer: &mut usize,
    discovery: &mut AHashMap<NodeId, usize>,
    low: &mut AHashMap<NodeId, usize>,
    bridges: &mut AHashSet<usize>,
) {
    *timer += 1;
    discovery.insert(curr, *timer);
    low.insert(curr, *timer);

    if let Some(neighbors) = adj.get(&curr) {
        for &(next_node, corridor_id) in neighbors {
            if Some(corridor_id) == parent_edge {
                continue; // Do not traverse the same corridor backwards
            }

            if let Some(&next_disc) = discovery.get(&next_node) {
                // Back-edge found: update low link
                let curr_low = low.get_mut(&curr).unwrap();
                *curr_low = (*curr_low).min(next_disc);
            } else {
                // Forward tree edge
                tarjan_dfs(
                    next_node,
                    Some(corridor_id),
                    adj,
                    timer,
                    discovery,
                    low,
                    bridges,
                );

                let next_low = *low.get(&next_node).unwrap();
                let curr_low = low.get_mut(&curr).unwrap();
                *curr_low = (*curr_low).min(next_low);

                // Tarjan Bridge Condition:
                // If neighbor cannot reach an ancestor of current node, this corridor is a BRIDGE!
                let curr_disc = *discovery.get(&curr).unwrap();
                if next_low > curr_disc {
                    bridges.insert(corridor_id);
                }
            }
        }
    }
}