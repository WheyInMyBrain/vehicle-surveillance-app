use ahash::{AHashMap, AHashSet};
use petgraph::visit::EdgeRef;
use road_network_compiler::geometry::parse_geojson_coords;
use road_network_compiler::parquet_reader::stream_parquet_rows;
use road_network_compiler::spatial_index::{SpatialEdgeEntry, SpatialEdgeGrid};
use road_network_compiler::types::{
    UnifiedEdgeRecord, UnifiedHighwayTier, UnifiedMumbaiGraph, UnifiedNodeRecord, DAYS_PER_WEEK,
    TIME_BINS_PER_DAY, TOTAL_TIME_BINS,
};
use road_osm_loader::load_osm_pbf;
use serde_json::json;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();

    let pbf_path = root.join("data/road/western-zone-261028.osm.pbf");
    let parquet_path = root.join("data/road/mumbai_full_week.parquet");
    let out_bin_path = root.join("data/road/mumbai_complete_graph.bin");
    let out_meta_path = root.join("data/road/mumbai_complete_graph_meta.json");

    println!("==========================================================");
    println!("     MUMBAI COMPLETE UNIFIED ROAD GRAPH COMPILER          ");
    println!("==========================================================");

    let t0 = Instant::now();
    let bkc_bbox = Some((19.0400, 72.8400, 19.0900, 72.9000));
    let (city, _) = load_osm_pbf(&pbf_path, bkc_bbox).expect("Failed to load OSM PBF");
    println!("Loaded OSM Sub-graph in {:.2?} (Nodes: {}, Edges: {})", t0.elapsed(), city.graph.node_count(), city.graph.edge_count());

    let mut nodes = Vec::with_capacity(city.graph.node_count());
    for node_idx in city.graph.node_indices() {
        let node_id = city.reverse_node_map[&node_idx];
        let meta = &city.graph[node_idx];
        nodes.push(UnifiedNodeRecord {
            id: node_id,
            lat: meta.lat,
            lon: meta.lon,
        });
    }

    let mut edges = Vec::with_capacity(city.graph.edge_count());
    let mut spatial_edges = Vec::with_capacity(city.graph.edge_count());

    for (i, edge_ref) in city.graph.edge_references().enumerate() {
        let u_idx = edge_ref.source();
        let v_idx = edge_ref.target();
        let u_id = city.reverse_node_map[&u_idx];
        let v_id = city.reverse_node_map[&v_idx];

        let u_meta = &city.graph[u_idx];
        let v_meta = &city.graph[v_idx];
        let edge_meta = edge_ref.weight();

        let tier = UnifiedHighwayTier::from_str(&format!("{:?}", edge_meta.highway_type));
        let maxspeed = edge_meta
            .maxspeed_kmh
            .map(|v| v.round().clamp(5.0, 150.0) as u8)
            .unwrap_or_else(|| tier.baseline_speed_kmh());

        edges.push(UnifiedEdgeRecord {
            u: u_id,
            v: v_id,
            length_m: edge_meta.length_m as f32,
            maxspeed_kmh: maxspeed,
            lanes: edge_meta.lanes.unwrap_or(1) as u8,
            highway_tier: tier,
            geometry: vec![[u_meta.lat, u_meta.lon], [v_meta.lat, v_meta.lon]],
        });

        spatial_edges.push(SpatialEdgeEntry {
            edge_idx: i,
            bearing_deg: edge_meta.bearing_deg,
            start_lat: u_meta.lat,
            start_lon: u_meta.lon,
            end_lat: v_meta.lat,
            end_lon: v_meta.lon,
        });
    }

    let spatial_grid = SpatialEdgeGrid::new(spatial_edges, 0.005);
    println!("Built spatial lookup grid across {} edges", edges.len());

    let mut graph = UnifiedMumbaiGraph::new(nodes, edges);

    println!("\nStreaming complete parquet telemetry into unified graph...");
    let t_scan = Instant::now();
    let mut bkc_rows = 0usize;
    let mut link_cache: AHashMap<(u64, u64, u64, u64), Vec<usize>> = AHashMap::with_capacity(65_536);
    let mut scratch_matched = AHashSet::new();

    let total_rows = stream_parquet_rows(&parquet_path, 65_536, |row| {
        if row.start_lat < 19.0300 || row.start_lat > 19.1000 || row.start_lon < 72.8300 || row.start_lon > 72.9100 {
            return Ok(());
        }
        bkc_rows += 1;

        let key = (
            row.start_lat.to_bits(),
            row.start_lon.to_bits(),
            row.end_lat.to_bits(),
            row.end_lon.to_bits(),
        );

        let matched_indices: &[usize] = if let Some(cached) = link_cache.get(&key) {
            cached.as_slice()
        } else {
            scratch_matched.clear();
            let coords = parse_geojson_coords(row.polyline_geojson);

            if coords.len() >= 2 {
                for w in coords.windows(2) {
                    spatial_grid.find_matching_edges_for_segment(w[0].0, w[0].1, w[1].0, w[1].1, 25.0, 35.0, &mut scratch_matched);
                }
            } else {
                spatial_grid.find_matching_edges_for_segment(row.start_lat, row.start_lon, row.end_lat, row.end_lon, 25.0, 35.0, &mut scratch_matched);
            }

            let vec_matches: Vec<usize> = scratch_matched.iter().copied().collect();
            link_cache.insert(key, vec_matches);
            link_cache.get(&key).unwrap().as_slice()
        };

        if !matched_indices.is_empty() {
            let speed = row.speed_kmh.clamp(0.0, 255.0).round() as u8;
            let freeflow = row.free_flow_kmh.clamp(0.0, 255.0).round() as u8;
            let jam = (row.jam_factor * 10.0).clamp(0.0, 100.0).round() as u8;
            let conf = (row.confidence * 100.0).clamp(0.0, 100.0).round() as u8;
            let is_closed = if row.status == "closed" { 1u8 } else { 0u8 };

            for &edge_idx in matched_indices {
                let cell = graph.cell_index(edge_idx, row.day_of_week, row.time_bin_5min);
                graph.speeds_kmh[cell] = speed;
                graph.jam_factors[cell] = jam;
                graph.free_flow_kmh[cell] = freeflow;
                graph.confidences[cell] = conf;
                graph.is_closed[cell] = is_closed;
                graph.is_empirical[cell] = 1;
            }
        }

        Ok(())
    })?;

    println!("Scanned {} rows in {:.2?} (BKC rows: {})", total_rows, t_scan.elapsed(), bkc_rows);
    println!("Unique HERE link corridors cached: {}", link_cache.len());

    println!("\nSerializing complete unified graph to {:?}...", out_bin_path);
    let t_save = Instant::now();
    graph.save_to_disk(&out_bin_path)?;
    let file_sz_mb = (std::fs::metadata(&out_bin_path)?.len() as f64) / (1024.0 * 1024.0);
    println!("Saved complete unified graph ({:.2} MB) in {:.2?}", file_sz_mb, t_save.elapsed());

    let meta = json!({
        "format": "MMCG_v1",
        "description": "Mumbai Complete Unified Road Graph and Temporal Traffic Tensor",
        "num_nodes": graph.nodes.len(),
        "num_edges": graph.edges.len(),
        "days_per_week": DAYS_PER_WEEK,
        "bins_per_day": TIME_BINS_PER_DAY,
        "total_time_bins": TOTAL_TIME_BINS,
        "file_size_mb": file_sz_mb,
        "output_path": out_bin_path.to_string_lossy(),
    });
    let meta_file = File::create(&out_meta_path)?;
    serde_json::to_writer_pretty(meta_file, &meta)?;
    println!("Metadata written to {:?}", out_meta_path);
    println!("==========================================================\n");

    Ok(())
}