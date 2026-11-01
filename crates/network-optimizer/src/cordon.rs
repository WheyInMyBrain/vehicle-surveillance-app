use ahash::AHashSet;
use engine_core::{CityGraph, HighwayType, NodeId};
use petgraph::visit::EdgeRef;
use petgraph::Direction;

pub type DirectedEdge = (NodeId, NodeId);

/// Finds natural gateway edges where local residential traffic funnels into
/// tertiary, secondary, or primary roads.
pub fn find_perimeter_gateway_edges(city: &CityGraph, max_gateways: usize) -> Vec<DirectedEdge> {
    let mut gateway_edges = Vec::new();
    let mut visited_canonical = AHashSet::new();

    for edge in city.graph.edge_references() {
        if gateway_edges.len() >= max_gateways {
            break;
        }

        let u_idx = edge.source();
        let v_idx = edge.target();
        let u_type = edge.weight().highway_type;

        // Check if target node acts as an intersection connecting to higher-level road
        let mut connects_to_higher_class = false;
        let mut out_degree = 0;

        for out_edge in city.graph.edges_directed(v_idx, Direction::Outgoing) {
            out_degree += 1;
            let out_type = out_edge.weight().highway_type;
            if is_higher_class(out_type, u_type) {
                connects_to_higher_class = true;
            }
        }

        // Gateway condition: transition point from local street into an arterial hub
        if connects_to_higher_class && out_degree >= 3 {
            let u_id = city.reverse_node_map[&u_idx];
            let v_id = city.reverse_node_map[&v_idx];
            let canonical = if u_id < v_id { (u_id, v_id) } else { (v_id, u_id) };

            if visited_canonical.insert(canonical) {
                gateway_edges.push((u_id, v_id));
            }
        }
    }

    gateway_edges
}

fn is_higher_class(target: HighwayType, source: HighwayType) -> bool {
    let rank = |h: HighwayType| match h {
        HighwayType::Motorway => 5,
        HighwayType::Trunk => 4,
        HighwayType::Primary => 3,
        HighwayType::Secondary => 2,
        HighwayType::Tertiary => 1,
        _ => 0,
    };
    rank(target) > rank(source)
}