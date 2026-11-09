use road_osm_loader::load_osm_pbf;
use std::path::PathBuf;
use std::time::Instant;

#[test]
fn test_mumbai_western_zone_pbf() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let pbf_path = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("data/road/western-zone-261028.osm.pbf"))
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("data/road/western-zone-261028.osm.pbf"));

    println!("\nResolved PBF Path: {:?}", pbf_path);
    assert!(pbf_path.exists(), "PBF file does not exist at {:?}", pbf_path);

    let bkc_bbox = Some((19.0400, 72.8400, 19.0900, 72.9000));

    let start = Instant::now();
    let (city, stats) = load_osm_pbf(&pbf_path, bkc_bbox)
        .unwrap_or_else(|e| panic!("Failed to load {:?}: {:?}", pbf_path, e));
    let elapsed = start.elapsed();

    println!("\n==============================================");
    println!("     MUMBAI BKC RAW OSM EXTRACTION REPORT      ");
    println!("==============================================");
    println!("Parse Duration:           {:.2?}", elapsed);
    println!("Ways Scanned:             {}", stats.ways_scanned);
    println!("Drivable Ways Extracted:  {}", stats.drivable_ways_extracted);
    println!("Graph Nodes:              {}", stats.nodes_retained);
    println!("Directed Edges:           {}", stats.total_directed_edges);
    println!("Traffic Signals Located:  {}", stats.traffic_signals_found);
    println!("Speed Breakers Located:   {}", stats.speed_bumps_found);
    println!("Flyover / Elevated Ways:  {}", stats.elevated_flyover_edges);
    println!("Underground Tunnels:      {}", stats.underground_tunnel_edges);

    println!("\n--- SPEED LIMIT TAGGING ---");
    println!("Explicitly Tagged:        {}", stats.explicit_maxspeed_tags);
    println!("Untagged:                 {}", stats.untagged_maxspeed);
    for (tier, count) in &stats.speed_tier_counts {
        println!("  • {:<24} {}", tier, count);
    }

    println!("\n--- SURFACE TAGGING ---");
    for (surface, count) in &stats.surface_counts {
        println!("  • {:<15?} : {} ways", surface, count);
    }

    println!("\n--- LANE TAGGING ---");
    let mut sorted_lanes: Vec<_> = stats.lane_distribution.iter().collect();
    sorted_lanes.sort_by_key(|&(k, _)| k.unwrap_or(0));
    for (lanes, count) in sorted_lanes {
        match lanes {
            Some(l) => println!("  • {} lane(s): {} ways", l, count),
            None => println!("  • Untagged : {} ways", count),
        }
    }
    println!("==============================================\n");

    assert!(stats.ways_scanned > 0);
    assert!(city.graph.node_count() > 0);
}