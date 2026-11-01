use network_optimizer::cut::find_topological_cuts;
use network_optimizer::pruner::{prune_to_zero_escape_network, ZeroEscapePrunerConfig};
use network_optimizer::skeleton::build_skeleton_graph;
use osm_loader::load_osm_pbf;
use std::path::PathBuf;

#[test]
fn test_mumbai_zero_escape_pruning() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let pbf_path = root.join("data/western-zone-261028.osm.pbf");

    let bkc_bbox = Some((19.0400, 72.8400, 19.0900, 72.9000));
    let (city, _) = load_osm_pbf(&pbf_path, bkc_bbox).expect("Failed to load PBF");

    // 1. Skeleton Contraction
    let skeleton = build_skeleton_graph(&city);

    // 2. Tarjan Cuts & Perimeter Gate Detection
    let cuts = find_topological_cuts(&city, &skeleton);

    // 3. Reverse Pruning with 500m Unmonitored Dark-Path BFS
    let config = ZeroEscapePrunerConfig {
        max_dark_distance_m: 500.0,
        max_dark_hops: 5,
        max_unmonitored_forks: 2,
    };

    let (camera_network, summary) = prune_to_zero_escape_network(&city, &skeleton, &cuts, config);

    let camera_density_reduction =
        (1.0 - (summary.final_active_cameras as f64 / city.graph.edge_count() as f64)) * 100.0;

    println!("\n==============================================");
    println!("     ZERO-ESCAPE PRUNED NETWORK REPORT        ");
    println!("==============================================");
    println!("Execution Duration:       {:.2} ms", summary.execution_duration_ms);
    println!("Raw OSM Edges (Blanket):  {}", city.graph.edge_count());
    println!("Contracted Macro-Edges:   {}", summary.initial_macro_corridors);
    println!("Permanently Locked Gates: {} (Tarjan Bridges + Portals)", summary.permanently_locked_gates);
    println!("Redundant Sensors Pruned: {}", summary.pruned_redundant_cameras);
    println!("Final Optimized Cameras:  {}", summary.final_active_cameras);
    println!("Hardware Cost Reduction:  {:.1}% vs raw edges", camera_density_reduction);
    println!("==============================================\n");

    assert!(!camera_network.is_empty(), "Camera network is empty");
    assert!(
        summary.final_active_cameras >= summary.permanently_locked_gates,
        "Pruner illegally removed mandatory gates"
    );
    assert!(
        summary.final_active_cameras < summary.initial_macro_corridors,
        "No pruning occurred"
    );
}