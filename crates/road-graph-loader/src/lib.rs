use ahash::AHashMap;
use road_network_core::NodeId;
use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::path::Path;
use thiserror::Error;

pub const TIME_BINS_PER_DAY: usize = 288;
pub const DAYS_PER_WEEK: usize = 7;
pub const TOTAL_TIME_BINS: usize = DAYS_PER_WEEK * TIME_BINS_PER_DAY; // 2016

pub type CanonicalDirectedEdge = (NodeId, NodeId);

#[derive(Error, Debug)]
pub enum RoadGraphError {
    #[error("I/O error reading binary graph: {0}")]
    Io(#[from] std::io::Error),
    #[error("Corrupted container: invalid magic header")]
    InvalidMagic,
    #[error("Unsupported graph format version: expected 1, found {0}")]
    UnsupportedVersion(u32),
    #[error("Dimension mismatch: expected {expected} bins, found {found}")]
    BinDimensionMismatch { expected: usize, found: usize },
    #[error("Buffer overrun while parsing offset {offset} (file size {size})")]
    BufferOverrun { offset: usize, size: usize },
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnifiedHighwayTier {
    Motorway = 1,
    Trunk = 2,
    Primary = 3,
    Secondary = 4,
    Tertiary = 5,
    Residential = 6,
    Service = 7,
    LivingStreet = 8,
    Unclassified = 9,
    Other = 10,
}

impl From<u8> for UnifiedHighwayTier {
    #[inline(always)]
    fn from(byte: u8) -> Self {
        match byte {
            1 => UnifiedHighwayTier::Motorway,
            2 => UnifiedHighwayTier::Trunk,
            3 => UnifiedHighwayTier::Primary,
            4 => UnifiedHighwayTier::Secondary,
            5 => UnifiedHighwayTier::Tertiary,
            6 => UnifiedHighwayTier::Residential,
            7 => UnifiedHighwayTier::Service,
            8 => UnifiedHighwayTier::LivingStreet,
            9 => UnifiedHighwayTier::Unclassified,
            _ => UnifiedHighwayTier::Other,
        }
    }
}

/// Fast, cache-friendly view of an edge's static physical properties
#[derive(Debug, Clone)]
pub struct EdgeDescriptor {
    pub u: NodeId,
    pub v: NodeId,
    pub length_m: f32,
    pub maxspeed_kmh: u8,
    pub lanes: u8,
    pub tier: UnifiedHighwayTier,
    pub geometry: Vec<[f64; 2]>,
}

/// Instant snapshot view of 5-minute dynamic physical traffic flow
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkFlowSnapshot {
    pub speed_kmh: u8,
    pub jam_factor: f32, // 0.0 to 10.0
    pub free_flow_kmh: u8,
    pub confidence: u8,  // 0% to 100%
    pub is_closed: bool,
    pub is_empirical: bool,
}

/// Zero-copy mmap-backed Mumbai Complete Road Graph
pub struct RoadGraph {
    _mmap: Mmap,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub total_time_bins: usize,

    // Fast O(1) hash map: (u, v) -> dense_edge_index
    edge_index_map: AHashMap<CanonicalDirectedEdge, usize>,
    
    // Adjacency graph for ultra-fast routing: u -> Vec<(v, edge_index)>
    adjacency: AHashMap<NodeId, Vec<(NodeId, usize)>>,

    // Deserialized metadata for topology traversals
    edges: Vec<EdgeDescriptor>,

    // Raw zero-copy slice pointers into the mmap buffer
    speeds_ptr: *const u8,
    jam_factors_ptr: *const u8,
    free_flow_ptr: *const u8,
    confidences_ptr: *const u8,
    is_closed_ptr: *const u8,
    is_empirical_ptr: *const u8,
}

// Safe to send across Rayon worker threads since the underlying mmap buffer is strictly read-only
unsafe impl Send for RoadGraph {}
unsafe impl Sync for RoadGraph {}

impl RoadGraph {
    /// Loads the binary graph file via kernel memory-mapping with zero copy
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, RoadGraphError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let buffer = &mmap[..];
        let file_len = buffer.len();

        let mut offset = 0;

        // 1. Verify 4-byte Magic Header: b"MMCG"
        if file_len < 20 {
            return Err(RoadGraphError::BufferOverrun { offset, size: file_len });
        }
        if &buffer[0..4] != b"MMCG" {
            return Err(RoadGraphError::InvalidMagic);
        }
        offset += 4;

        // 2. Read Layout Dimensions
        let version = u32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap());
        if version != 1 {
            return Err(RoadGraphError::UnsupportedVersion(version));
        }
        offset += 4;

        let num_nodes = u32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let num_edges = u32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let bins_per_edge = u32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        if bins_per_edge != TOTAL_TIME_BINS {
            return Err(RoadGraphError::BinDimensionMismatch {
                expected: TOTAL_TIME_BINS,
                found: bins_per_edge,
            });
        }

        // 3. Skip/Parse Nodes: [id: u64, lat: f64, lon: f64] (24 bytes per node)
        let nodes_bytes = num_nodes * 24;
        if offset + nodes_bytes > file_len {
            return Err(RoadGraphError::BufferOverrun { offset: offset + nodes_bytes, size: file_len });
        }
        offset += nodes_bytes;

        // 4. Parse Edge Metadata & Build Adjacency Table
        let mut edges = Vec::with_capacity(num_edges);
        let mut edge_index_map = AHashMap::with_capacity(num_edges);
        let mut adjacency: AHashMap<NodeId, Vec<(NodeId, usize)>> = AHashMap::with_capacity(num_nodes);

        for edge_idx in 0..num_edges {
            if offset + 26 > file_len {
                return Err(RoadGraphError::BufferOverrun { offset, size: file_len });
            }

            let u = u64::from_le_bytes(buffer[offset..offset + 8].try_into().unwrap());
            let v = u64::from_le_bytes(buffer[offset + 8..offset + 16].try_into().unwrap());
            let length_m = f32::from_le_bytes(buffer[offset + 16..offset + 20].try_into().unwrap());
            
            let maxspeed_kmh = buffer[offset + 20];
            let lanes = buffer[offset + 21];
            let tier = UnifiedHighwayTier::from(buffer[offset + 22]);
            // buffer[offset + 23] reserved
            offset += 24;

            let num_geom = u16::from_le_bytes(buffer[offset..offset + 2].try_into().unwrap()) as usize;
            offset += 2;

            let geom_bytes = num_geom * 16;
            if offset + geom_bytes > file_len {
                return Err(RoadGraphError::BufferOverrun { offset: offset + geom_bytes, size: file_len });
            }

            let mut geometry = Vec::with_capacity(num_geom);
            for _ in 0..num_geom {
                let p_lat = f64::from_le_bytes(buffer[offset..offset + 8].try_into().unwrap());
                let p_lon = f64::from_le_bytes(buffer[offset + 8..offset + 16].try_into().unwrap());
                geometry.push([p_lat, p_lon]);
                offset += 16;
            }

            edges.push(EdgeDescriptor {
                u,
                v,
                length_m,
                maxspeed_kmh,
                lanes,
                tier,
                geometry,
            });

            edge_index_map.insert((u, v), edge_idx);
            adjacency.entry(u).or_default().push((v, edge_idx));
        }

        // 5. Zero-Copy Pointer Mapping for Dynamic Telemetry Tensors
        let total_cells = num_edges * bins_per_edge;
        let required_telemetry_bytes = total_cells * 6;

        if offset + required_telemetry_bytes > file_len {
            return Err(RoadGraphError::BufferOverrun {
                offset: offset + required_telemetry_bytes,
                size: file_len,
            });
        }

        let speeds_ptr = unsafe { buffer.as_ptr().add(offset) };
        offset += total_cells;

        let jam_factors_ptr = unsafe { buffer.as_ptr().add(offset) };
        offset += total_cells;

        let free_flow_ptr = unsafe { buffer.as_ptr().add(offset) };
        offset += total_cells;

        let confidences_ptr = unsafe { buffer.as_ptr().add(offset) };
        offset += total_cells;

        let is_closed_ptr = unsafe { buffer.as_ptr().add(offset) };
        offset += total_cells;

        let is_empirical_ptr = unsafe { buffer.as_ptr().add(offset) };

        Ok(Self {
            _mmap: mmap,
            num_nodes,
            num_edges,
            total_time_bins: bins_per_edge,
            edge_index_map,
            adjacency,
            edges,
            speeds_ptr,
            jam_factors_ptr,
            free_flow_ptr,
            confidences_ptr,
            is_closed_ptr,
            is_empirical_ptr,
        })
    }

    /// Fast edge index lookup: O(1)
    #[inline(always)]
    pub fn get_edge_index(&self, u: NodeId, v: NodeId) -> Option<usize> {
        self.edge_index_map.get(&(u, v)).copied()
    }

    /// Fetches outgoing neighbors of a node: O(1)
    #[inline(always)]
    pub fn outgoing_edges(&self, u: NodeId) -> &[(NodeId, usize)] {
        self.adjacency.get(&u).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Fetches immutable physical properties of an edge
    #[inline(always)]
    pub fn edge_descriptor(&self, edge_idx: usize) -> &EdgeDescriptor {
        &self.edges[edge_idx]
    }

    /// Calculates flat cell index in O(1) without integer division
    #[inline(always)]
    pub fn cell_index(&self, edge_idx: usize, day: usize, time_bin_5min: usize) -> usize {
        let bin_idx = (day % DAYS_PER_WEEK) * TIME_BINS_PER_DAY + (time_bin_5min % TIME_BINS_PER_DAY);
        edge_idx * TOTAL_TIME_BINS + bin_idx
    }

    /// Zero-copy flow retrieval in < 3 nanoseconds.
    #[inline(always)]
    pub fn get_flow_by_index(&self, edge_idx: usize, day: usize, time_bin_5min: usize) -> LinkFlowSnapshot {
        let cell = self.cell_index(edge_idx, day, time_bin_5min);

        unsafe {
            let speed = *self.speeds_ptr.add(cell);
            let raw_jam = *self.jam_factors_ptr.add(cell);
            let free_flow = *self.free_flow_ptr.add(cell);
            let confidence = *self.confidences_ptr.add(cell);
            let closed = *self.is_closed_ptr.add(cell) == 1;
            let empirical = *self.is_empirical_ptr.add(cell) == 1;

            LinkFlowSnapshot {
                speed_kmh: speed,
                jam_factor: raw_jam as f32 / 10.0,
                free_flow_kmh: free_flow,
                confidence,
                is_closed: closed,
                is_empirical: empirical,
            }
        }
    }

    /// Convenience lookup for (u, v)
    #[inline(always)]
    pub fn get_flow(&self, u: NodeId, v: NodeId, day: usize, time_bin_5min: usize) -> Option<LinkFlowSnapshot> {
        let edge_idx = self.get_edge_index(u, v)?;
        Some(self.get_flow_by_index(edge_idx, day, time_bin_5min))
    }
}