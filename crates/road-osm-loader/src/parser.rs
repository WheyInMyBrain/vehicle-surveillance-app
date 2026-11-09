use crate::builder::build_city_graph;
use crate::error::{OsmError, ParsingStatistics};
use crate::scanner::scan_pbf;
use road_network_core::CityGraph;
use std::path::Path;

pub fn load_osm_pbf<P: AsRef<Path>>(
    pbf_path: P,
    bbox: Option<(f64, f64, f64, f64)>,
) -> Result<(CityGraph, ParsingStatistics), OsmError> {
    let scan_result = scan_pbf(pbf_path, bbox)?;
    let (city, stats) = build_city_graph(scan_result);
    Ok((city, stats))
}