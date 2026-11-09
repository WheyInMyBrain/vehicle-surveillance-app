use ahash::{AHashMap, AHashSet};
use petgraph::visit::EdgeRef;
use road_network_compiler::geometry::parse_geojson_coords;
use road_network_compiler::parquet_reader::stream_parquet_rows;
use road_network_compiler::spatial_index::{SpatialEdgeEntry, SpatialEdgeGrid};
use road_network_compiler::types::TOTAL_TIME_BINS;
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
    let out_report_path = root.join("data/road/traffic_coverage_audit.json");

    println!("==========================================================");
    println!("  POLYLINE-AWARE OSM vs. HERE EMPIRICAL COVERAGE AUDIT     ");
    println!("==========================================================");

    let t0 = Instant::now();
    let bkc_bbox = Some((19.0400, 72.8400, 19.0900, 72.9000));
    let (city, _) = load_osm_pbf(&pbf_path, bkc_bbox).expect("Failed to load OSM PBF");
    println!("Loaded OSM Sub-graph in {:.2?} (Nodes: {}, Edges: {})", t0.elapsed(), city.graph.node_count(), city.graph.edge_count());

    let mut spatial_edges = Vec::with_capacity(city.graph.edge_count());
    let mut edge_classes = Vec::with_capacity(city.graph.edge_count());
    let mut class_total_counts: AHashMap<String, usize> = AHashMap::new();

    for (i, edge_ref) in city.graph.edge_references().enumerate() {
        let u_idx = edge_ref.source();
        let v_idx = edge_ref.target();
        let u_meta = &city.graph[u_idx];
        let v_meta = &city.graph[v_idx];
        let edge_meta = edge_ref.weight();

        let highway = format!("{:?}", edge_meta.highway_type);
        *class_total_counts.entry(highway.clone()).or_insert(0) += 1;
        edge_classes.push(highway);

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
    let mut edge_observations: AHashMap<usize, (AHashSet<usize>, usize)> = AHashMap::with_capacity(edge_classes.len());

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
            let global_bin = row.day_of_week * 288 + row.time_bin_5min;
            for &edge_idx in matched_indices {
                let entry = edge_observations.entry(edge_idx).or_insert_with(|| (AHashSet::new(), 0));
                entry.0.insert(global_bin);
                entry.1 += 1;
            }
        }

        Ok(())
    })?;

    println!("Scanned {} rows in {:.2?} (BKC rows: {})", total_rows, t_scan.elapsed(), bkc_rows);

    let total_osm_edges = edge_classes.len();
    let matched_osm_edges = edge_observations.len();
    let unmonitored_osm_edges = total_osm_edges.saturating_sub(matched_osm_edges);

    let mut class_covered: AHashMap<String, usize> = AHashMap::new();
    let mut class_continuous: AHashMap<String, usize> = AHashMap::new();
    let mut fully_covered = 0usize;
    let mut intermittent = 0usize;

    for (&edge_idx, (bins, _)) in &edge_observations {
        let cls = &edge_classes[edge_idx];
        *class_covered.entry(cls.clone()).or_insert(0) += 1;

        if bins.len() >= (TOTAL_TIME_BINS as f64 * 0.90) as usize {
            fully_covered += 1;
            *class_continuous.entry(cls.clone()).or_insert(0) += 1;
        } else {
            intermittent += 1;
        }
    }

    println!("\n==========================================================");
    println!("      POLYLINE-PROJECTED TOPOLOGICAL COVERAGE             ");
    println!("==========================================================");
    println!("Total OSM Directed Edges:         {}", total_osm_edges);
    println!("Matched Edges (≥1 Reading):       {} ({:.1}%)", matched_osm_edges, (matched_osm_edges as f64 / total_osm_edges as f64) * 100.0);
    println!("Unmonitored / Dark Edges:         {} ({:.1}%)", unmonitored_osm_edges, (unmonitored_osm_edges as f64 / total_osm_edges as f64) * 100.0);
    println!("Continuous Coverage (≥90% bins):  {} ({:.1}%)", fully_covered, (fully_covered as f64 / total_osm_edges as f64) * 100.0);
    println!("Intermittent / Sparse Bins:       {} ({:.1}%)", intermittent, (intermittent as f64 / total_osm_edges as f64) * 100.0);

    println!("\n-----------------------------------------------------------------------------");
    println!("{:<18} | {:<10} | {:<10} | {:<12} | {:<10}", "OSM Highway Tier", "Total OSM", "Monitored", "Coverage %", "Continuous");
    println!("-----------------------------------------------------------------------------");

    let mut sorted_classes: Vec<(String, usize)> = class_total_counts.into_iter().collect();
    sorted_classes.sort_by(|a, b| b.1.cmp(&a.1));

    for (class, total) in &sorted_classes {
        let mon = *class_covered.get(class).unwrap_or(&0);
        let cont = *class_continuous.get(class).unwrap_or(&0);
        let pct = (mon as f64 / *total as f64) * 100.0;
        println!("{:<18} | {:<10} | {:<10} | {:<11.1}% | {:<10}", class, total, mon, pct, cont);
    }
    println!("-----------------------------------------------------------------------------\n");

    let report = json!({
        "total_osm_edges": total_osm_edges,
        "matched_edges": matched_osm_edges,
        "unmonitored_edges": unmonitored_osm_edges,
        "continuous_edges_90pct": fully_covered,
        "intermittent_edges": intermittent,
    });
    let f = File::create(&out_report_path)?;
    serde_json::to_writer_pretty(f, &report)?;
    println!("Full audit report written to {:?}", out_report_path);

    Ok(())
}