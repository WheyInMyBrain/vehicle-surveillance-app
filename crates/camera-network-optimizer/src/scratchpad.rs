use ahash::AHashMap;
use road_network_core::NodeId;

pub struct DijkstraScratchpad {
    node_to_idx: AHashMap<NodeId, usize>,
    visited_gen: Vec<u32>,
    best_dist: Vec<f32>,
    current_gen: u32,
    queue_dist: Vec<f32>,
    queue_node: Vec<NodeId>,
    queue_hops: Vec<usize>,
}

impl DijkstraScratchpad {
    pub fn new(all_junctions: &[NodeId]) -> Self {
        let n = all_junctions.len();
        let mut node_to_idx = AHashMap::with_capacity(n);
        for (i, &nid) in all_junctions.iter().enumerate() {
            node_to_idx.insert(nid, i);
        }

        Self {
            node_to_idx,
            visited_gen: vec![0; n],
            best_dist: vec![f32::INFINITY; n],
            current_gen: 1,
            queue_dist: Vec::with_capacity(512),
            queue_node: Vec::with_capacity(512),
            queue_hops: Vec::with_capacity(512),
        }
    }

    #[inline(always)]
    pub fn reset(&mut self) {
        self.current_gen = self.current_gen.wrapping_add(1);
        if self.current_gen == 0 {
            self.visited_gen.fill(0);
            self.current_gen = 1;
        }
        self.queue_dist.clear();
        self.queue_node.clear();
        self.queue_hops.clear();
    }

    #[inline(always)]
    pub fn push(&mut self, node: NodeId, dist: f32, hops: usize) {
        if let Some(&idx) = self.node_to_idx.get(&node) {
            if self.visited_gen[idx] != self.current_gen || dist < self.best_dist[idx] {
                self.visited_gen[idx] = self.current_gen;
                self.best_dist[idx] = dist;

                self.queue_dist.push(dist);
                self.queue_node.push(node);
                self.queue_hops.push(hops);
            }
        }
    }

    #[inline(always)]
    pub fn pop_min(&mut self) -> Option<(NodeId, f32, usize)> {
        if self.queue_node.is_empty() {
            return None;
        }

        let mut min_idx = 0;
        let mut min_val = self.queue_dist[0];
        for i in 1..self.queue_dist.len() {
            if self.queue_dist[i] < min_val {
                min_val = self.queue_dist[i];
                min_idx = i;
            }
        }

        let node = self.queue_node.swap_remove(min_idx);
        let dist = self.queue_dist.swap_remove(min_idx);
        let hops = self.queue_hops.swap_remove(min_idx);

        Some((node, dist, hops))
    }
}