use crate::types::{EdgeMeta, NodeId, NodeMeta};
use ahash::AHashMap;
use petgraph::graph::{DiGraph, NodeIndex};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GraphError {
    #[error("Node ID {0} not found in the graph index")]
    NodeNotFound(NodeId),
}

#[derive(Debug, Clone)]
pub struct CityGraph {
    pub graph: DiGraph<NodeMeta, EdgeMeta>,
    pub node_map: AHashMap<NodeId, NodeIndex>,
}

impl CityGraph {
    #[inline]
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: AHashMap::with_capacity(32_768),
        }
    }

    #[inline]
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            graph: DiGraph::with_capacity(nodes, edges),
            node_map: AHashMap::with_capacity(nodes),
        }
    }

    #[inline]
    pub fn add_or_get_node(&mut self, id: NodeId, meta: NodeMeta) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(&id) {
            idx
        } else {
            let idx = self.graph.add_node(meta);
            self.node_map.insert(id, idx);
            idx
        }
    }

    #[inline]
    pub fn add_edge(&mut self, from: NodeId, to: NodeId, meta: EdgeMeta) -> Result<(), GraphError> {
        let from_idx = *self.node_map.get(&from).ok_or(GraphError::NodeNotFound(from))?;
        let to_idx = *self.node_map.get(&to).ok_or(GraphError::NodeNotFound(to))?;
        self.graph.add_edge(from_idx, to_idx, meta);
        Ok(())
    }

    /// O(1) reverse lookup directly from NodeMeta weight — eliminates reverse_node_map.
    #[inline(always)]
    pub fn get_node_id(&self, idx: NodeIndex) -> NodeId {
        self.graph[idx].id
    }

    #[inline(always)]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    #[inline(always)]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

impl Default for CityGraph {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}