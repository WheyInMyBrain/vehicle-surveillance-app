use camera_network_optimizer::{optimize_camera_network, OptimizerConfig};
use road_graph_loader::RoadGraph;
use road_traffic_tensor::TrafficTensorEngine;
use sim_trip_generator::{simulate_fleet, CameraSensorIndex, FleetSimulationConfig};
use std::path::PathBuf;

#[test]
fn test_massive_fleet_simulation() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let bin_path = root.join("data/road/mumbai_complete_graph.bin");

    if !bin_path.exists() {
        eprintln!("Binary graph not found at {:?}, skipping test.", bin_path);
        return;
    }

    let graph = RoadGraph::open(&bin_path).expect("Failed to mmap RoadGraph");
    let tensor = TrafficTensorEngine::new(&graph);

    let opt_config = OptimizerConfig::default();
    let opt = optimize_camera_network(&graph, &opt_config);
    let sensors = CameraSensorIndex::from_poles(opt.camera_poles);

    // Simulate 200 concurrent vehicles across Mumbai across different days/hours of the week
    let fleet_config = FleetSimulationConfig {
        num_vehicles: 200,
        min_distance_m: 4_000.0,
        max_distance_m: 7_000.0,
        camera_dropout_rate: 0.03, // 3% optical dropouts
        base_seed: 0xBEEF_CAFE_1234,
    };

    let result = simulate_fleet(&graph, &tensor, &sensors, fleet_config);

    println!("\n==========================================================");
    println!("     PARALLEL FLEET MULTI-VEHICLE SIMULATION AUDIT        ");
    println!("==========================================================");
    println!("Total Vehicles Simulated:      {}", result.total_vehicles_simulated);
    println!("Total Distance Driven:         {:.1} km", result.total_distance_km);
    println!("Total Streamed ANPR Sightings: {}", result.global_sightings_stream.len());
    println!("Wall-Clock Time:               {:.2} ms", result.simulation_duration_ms);
    println!("Avg Compute Time per Vehicle:  {:.2} µs", (result.simulation_duration_ms * 1000.0) / result.total_vehicles_simulated as f64);
    println!("==========================================================\n");

    assert_eq!(result.total_vehicles_simulated, 200);
    assert!(!result.global_sightings_stream.is_empty());

    // Verify chronological ordering of the output stream
    for window in result.global_sightings_stream.windows(2) {
        assert!(window[0].timestamp <= window[1].timestamp, "Stream must be strictly chronological");
    }
}