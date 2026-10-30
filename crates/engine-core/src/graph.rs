use crate::types::{EdgeMeta, NodeId, NodeMeta};
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;

pub struct CityGraph {
    pub graph: DiGraph<NodeMeta, EdgeMeta>,
    pub node_map: HashMap<NodeId, NodeIndex>,
    pub reverse_node_map: HashMap<NodeIndex, NodeId>,
}

impl CityGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
            reverse_node_map: HashMap::new(),
        }
    }

    pub fn add_or_get_node(&mut self, id: NodeId, meta: NodeMeta) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(&id) {
            idx
        } else {
            let idx = self.graph.add_node(meta);
            self.node_map.insert(id, idx);
            self.reverse_node_map.insert(idx, id);
            idx
        }
    }

    pub fn add_edge(&mut self, from: NodeId, to: NodeId, meta: EdgeMeta) {
        let from_idx = *self.node_map.get(&from).expect("From node not found");
        let to_idx = *self.node_map.get(&to).expect("To node not found");
        self.graph.add_edge(from_idx, to_idx, meta);
    }
}

impl Default for CityGraph {
    fn default() -> Self {
        Self::new()
    }
}