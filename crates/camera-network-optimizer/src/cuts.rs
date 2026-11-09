use crate::skeleton::SkeletonGraph;
use ahash::{AHashMap, AHashSet};
use road_graph_loader::RoadGraph;
use road_network_core::NodeId;

pub struct TopologicalCuts {
    pub bridge_corridor_ids: AHashSet<usize>,
    pub perimeter_gateway_ids: AHashSet<usize>,
    pub locked_corridors: AHashSet<usize>,
}

pub fn find_topological_cuts(graph: &RoadGraph, skeleton: &SkeletonGraph) -> TopologicalCuts {
    // 1. Tarjan's Bridge Finder on the undirected skeleton graph
    let mut adj: AHashMap<NodeId, Vec<(NodeId, usize)>> = AHashMap::new();
    for corr in &skeleton.corridors {
        if corr.start_junction != corr.end_junction {
            adj.entry(corr.start_junction).or_default().push((corr.end_junction, corr.id));
            adj.entry(corr.end_junction).or_default().push((corr.start_junction, corr.id));
        }
    }

    let mut discovery = AHashMap::new();
    let mut low = AHashMap::new();
    let mut bridge_ids = AHashSet::new();
    let mut timer = 0usize;

    for &junc in &skeleton.junction_nodes {
        if !discovery.contains_key(&junc) {
            tarjan_dfs(junc, None, &adj, &mut timer, &mut discovery, &mut low, &mut bridge_ids);
        }
    }

    // 2. Perimeter Gateways (boundary entry/exit conduits)
    let mut perimeter_ids = AHashSet::new();
    for corr in &skeleton.corridors {
        let u_out = graph.outgoing_edges(corr.start_junction).len();
        let v_out = graph.outgoing_edges(corr.end_junction).len();

        if u_out <= 1 || v_out <= 1 {
            perimeter_ids.insert(corr.id);
        }
    }

    let mut locked = AHashSet::with_capacity(bridge_ids.len() + perimeter_ids.len());
    locked.extend(&bridge_ids);
    locked.extend(&perimeter_ids);

    TopologicalCuts {
        bridge_corridor_ids: bridge_ids,
        perimeter_gateway_ids: perimeter_ids,
        locked_corridors: locked,
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
        for &(nxt, edge_id) in neighbors {
            if Some(edge_id) == parent_edge {
                continue;
            }
            if let Some(&d) = discovery.get(&nxt) {
                let curr_low = low.get_mut(&curr).unwrap();
                *curr_low = (*curr_low).min(d);
            } else {
                tarjan_dfs(nxt, Some(edge_id), adj, timer, discovery, low, bridges);
                let nxt_low = *low.get(&nxt).unwrap();
                let curr_low = low.get_mut(&curr).unwrap();
                *curr_low = (*curr_low).min(nxt_low);

                if nxt_low > *discovery.get(&curr).unwrap() {
                    bridges.insert(edge_id);
                }
            }
        }
    }
}