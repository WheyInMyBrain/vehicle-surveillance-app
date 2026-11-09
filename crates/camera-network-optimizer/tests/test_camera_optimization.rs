use camera_network_optimizer::{optimize_camera_network, OptimizerConfig};
use road_graph_loader::RoadGraph;
use std::path::PathBuf;

#[test]
fn test_mumbai_camera_optimization_pipeline() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let bin_path = root.join("data/road/mumbai_complete_graph.bin");

    if !bin_path.exists() {
        eprintln!("Binary graph not found at {:?}, skipping test.", bin_path);
        return;
    }

    let graph = RoadGraph::open(&bin_path).expect("Failed to load RoadGraph from .bin");
    let config = OptimizerConfig::default();

    let summary = optimize_camera_network(&graph, &config);

    println!("\n==========================================================");
    println!("     ZERO-ESCAPE CAMERA NETWORK OPTIMIZATION SUMMARY      ");
    println!("==========================================================");
    println!("Execution Time:             {:.2} ms", summary.execution_ms);
    println!("Total Macro-Corridors:      {}", summary.corridors_monitored + summary.corridors_pruned);
    println!("Permanently Locked Gates:   {}", summary.permanently_locked_gates);
    println!("Redundant Corridors Pruned: {}", summary.corridors_pruned);
    println!("Active Corridors Monitored: {}", summary.corridors_monitored);
    println!("Total Physical Poles:       {}", summary.total_poles);
    println!("Total Active ANPR Heads:    {}", summary.total_camera_heads);
    println!("Avg Heads per Pole:         {:.2}", summary.total_camera_heads as f64 / summary.total_poles as f64);
    println!("==========================================================\n");

    assert!(summary.total_poles > 0, "Pipeline must place camera poles");
    assert!(summary.total_camera_heads >= summary.total_poles);
}