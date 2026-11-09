use ahash::AHashMap;
use road_network_core::RoadSurface;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OsmError {
    #[error("I/O or PBF parsing error: {0}")]
    PbfError(#[from] osmpbf::Error),
    #[error("No drivable roads found in the dataset")]
    EmptyDataset,
}

#[derive(Debug, Default, Clone)]
pub struct ParsingStatistics {
    pub ways_scanned: usize,
    pub drivable_ways_extracted: usize,
    pub nodes_retained: usize,
    pub total_directed_edges: usize,
    pub traffic_signals_found: usize,
    pub speed_bumps_found: usize,
    pub surface_counts: AHashMap<RoadSurface, usize>,
    pub explicit_maxspeed_tags: usize,
    pub untagged_maxspeed: usize,
    pub speed_tier_counts: AHashMap<String, usize>,
    pub lane_distribution: AHashMap<Option<u8>, usize>,
    pub elevated_flyover_edges: usize,
    pub underground_tunnel_edges: usize,
}