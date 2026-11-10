use road_network_core::NodeId;

/// A cache-aligned flat generational hash table allocated on the heap (Box<[T]>).
/// Zero stack-overflow risk, zero heap allocations during search loops.
#[derive(Clone)]
pub struct GenerationalDenseMap<T: Copy + Default> {
    keys: Box<[NodeId]>,
    values: Box<[T]>,
    generations: Box<[u32]>,
    mask: usize,
    current_gen: u32,
}

impl<T: Copy + Default> GenerationalDenseMap<T> {
    pub fn new() -> Self {
        Self::with_capacity(2048)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        // Enforce power-of-two for bitwise masking
        let cap = capacity.next_power_of_two().max(512);
        Self {
            keys: vec![0; cap].into_boxed_slice(),
            values: vec![T::default(); cap].into_boxed_slice(),
            generations: vec![0; cap].into_boxed_slice(),
            mask: cap - 1,
            current_gen: 1,
        }
    }

    #[inline(always)]
    pub fn next_generation(&mut self) {
        self.current_gen = self.current_gen.wrapping_add(1);
        if self.current_gen == 0 {
            self.generations.fill(0);
            self.current_gen = 1;
        }
    }

    #[inline(always)]
    fn hash(&self, key: NodeId) -> usize {
        let mut x = key as u64;
        x ^= x >> 16;
        x = x.wrapping_mul(0x45d9f3b);
        x ^= x >> 16;
        (x as usize) & self.mask
    }

    #[inline(always)]
    pub fn get(&self, key: NodeId) -> Option<T> {
        let mut idx = self.hash(key);
        for _ in 0..16 {
            if self.generations[idx] == self.current_gen {
                if self.keys[idx] == key {
                    return Some(self.values[idx]);
                }
            } else {
                return None;
            }
            idx = (idx + 1) & self.mask;
        }
        None
    }

    #[inline(always)]
    pub fn insert(&mut self, key: NodeId, val: T) {
        let mut idx = self.hash(key);
        for _ in 0..16 {
            if self.generations[idx] != self.current_gen || self.keys[idx] == key {
                self.keys[idx] = key;
                self.values[idx] = val;
                self.generations[idx] = self.current_gen;
                return;
            }
            idx = (idx + 1) & self.mask;
        }
        // Slot eviction fallback on saturated probes
        self.keys[idx] = key;
        self.values[idx] = val;
        self.generations[idx] = self.current_gen;
    }
}

impl<T: Copy + Default> Default for GenerationalDenseMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Generation-stamped edge set allocated on the heap.
#[derive(Clone)]
pub struct GenerationalEdgeSet {
    edges: Box<[usize]>,
    generations: Box<[u32]>,
    mask: usize,
    current_gen: u32,
}

impl GenerationalEdgeSet {
    pub fn new() -> Self {
        Self::with_capacity(4096)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let cap = capacity.next_power_of_two().max(1024);
        Self {
            edges: vec![0; cap].into_boxed_slice(),
            generations: vec![0; cap].into_boxed_slice(),
            mask: cap - 1,
            current_gen: 1,
        }
    }

    #[inline(always)]
    pub fn next_generation(&mut self) {
        self.current_gen = self.current_gen.wrapping_add(1);
        if self.current_gen == 0 {
            self.generations.fill(0);
            self.current_gen = 1;
        }
    }

    #[inline(always)]
    fn hash(&self, edge: usize) -> usize {
        let mut x = edge as u64;
        x ^= x >> 16;
        x = x.wrapping_mul(0x517cc1b727220a95);
        (x as usize) & self.mask
    }

    #[inline(always)]
    pub fn insert(&mut self, edge: usize) {
        let mut idx = self.hash(edge);
        for _ in 0..16 {
            if self.generations[idx] != self.current_gen || self.edges[idx] == edge {
                self.edges[idx] = edge;
                self.generations[idx] = self.current_gen;
                return;
            }
            idx = (idx + 1) & self.mask;
        }
        self.edges[idx] = edge;
        self.generations[idx] = self.current_gen;
    }

    #[inline(always)]
    pub fn contains(&self, edge: usize) -> bool {
        let mut idx = self.hash(edge);
        for _ in 0..16 {
            if self.generations[idx] == self.current_gen {
                if self.edges[idx] == edge {
                    return true;
                }
            } else {
                return false;
            }
            idx = (idx + 1) & self.mask;
        }
        false
    }
}

impl Default for GenerationalEdgeSet {
    fn default() -> Self {
        Self::new()
    }
}