use camera_network_optimizer::{optimize_camera_network, OptimizerConfig};
use chrono::Utc;
use rand::rngs::StdRng;
use rand::SeedableRng;
use road_graph_loader::RoadGraph;
use road_traffic_tensor::TrafficTensorEngine;
use sim_trip_generator::{generate_vehicle_trip, CameraSensorIndex, TripMode};
use std::path::PathBuf;
use std::time::Instant;

#[test]
fn test_end_to_end_vehicle_simulation() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let bin_path = root.join("data/road/mumbai_complete_graph.bin");

    if !bin_path.exists() {
        eprintln!("Binary graph not found at {:?}, skipping test.", bin_path);
        return;
    }

    // 1. Zero-copy mmap
    let graph = RoadGraph::open(&bin_path).expect("Failed to mmap RoadGraph");
    let tensor = TrafficTensorEngine::new(&graph);

    // 2. Generate camera network
    let config = OptimizerConfig::default();
    let opt_summary = optimize_camera_network(&graph, &config);
    let sensors = CameraSensorIndex::from_poles(opt_summary.camera_poles);

    println!("\n==========================================================");
    println!("     END-TO-END VEHICLE AGENT & ANPR SIMULATION TEST      ");
    println!("==========================================================");
    println!("Deployed Physical Camera Sensors: {}", sensors.len());

    let mut rng = StdRng::seed_from_u64(42);
    let start_time = Utc::now();

    let t0 = Instant::now();
    let trip = generate_vehicle_trip(
        &graph,
        &tensor,
        &sensors,
        "MH02-DX-9999",
        start_time,
        TripMode::SmoothCruise { target_distance_m: 5_000.0 }, // 5 km continuous cruise
        0.02, // 2% camera dropout
        &mut rng,
    ).expect("Failed to generate simulated trip");

    let sim_duration = t0.elapsed();

    println!("Simulation Computed In:           {:?}", sim_duration);
    println!("Vehicle License Plate:            {}", trip.vehicle_id);
    println!("Ground Truth Nodes Traversed:     {}", trip.ground_truth_nodes.len());
    println!("Total Driving Distance:           {:.2} km", trip.total_distance_m / 1000.0);
    println!(
        "Elapsed Travel Time:              {:.1} min",
        (trip.end_time - trip.start_time).num_seconds() as f64 / 60.0
    );
    println!("ANPR Camera Hits Registered:      {}", trip.camera_sightings.len());

    for (i, sighting) in trip.camera_sightings.iter().enumerate().take(8) {
        println!(
            "  Hit #{}: Pole [{}] Edge ({} -> {}) @ {} | Speed: {:.1} km/h",
            i + 1,
            sighting.camera_id,
            sighting.edge_from,
            sighting.edge_to,
            sighting.timestamp.format("%H:%M:%S%.3f"),
            sighting.speed_kmh
        );
    }
    println!("==========================================================\n");

    assert!(trip.ground_truth_nodes.len() >= 6);
    assert!(trip.total_distance_m >= 5_000.0);
    assert!(!trip.camera_sightings.is_empty(), "Vehicle must trigger at least one camera gantry");
}