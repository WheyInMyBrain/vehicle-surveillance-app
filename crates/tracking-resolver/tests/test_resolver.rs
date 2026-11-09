use ahash::AHashSet;
use camera_network_optimizer::{optimize_camera_network, OptimizerConfig};
use chrono::Utc;
use rand::rngs::StdRng;
use rand::SeedableRng;
use road_graph_loader::RoadGraph;
use road_network_core::NodeId;
use road_traffic_tensor::TrafficTensorEngine;
use sim_trip_generator::{generate_vehicle_trip, CameraSensorIndex, TripMode};
use std::path::PathBuf;
use std::time::Instant;
use tracking_resolver::StreamingCausalTracker;

#[test]
fn test_causal_trajectory_reconstruction_with_beam_and_anomalies() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let bin_path = root.join("data/road/mumbai_complete_graph.bin");

    if !bin_path.exists() {
        eprintln!("Binary graph not found at {:?}, skipping test.", bin_path);
        return;
    }

    let graph = RoadGraph::open(&bin_path).expect("Failed to mmap RoadGraph");
    let tensor = TrafficTensorEngine::new(&graph);

    let config = OptimizerConfig::default();
    let opt = optimize_camera_network(&graph, &config);
    let sensors = CameraSensorIndex::from_poles(opt.camera_poles);

    let mut rng = StdRng::seed_from_u64(42);
    let start_time = Utc::now();
    let trip = generate_vehicle_trip(
        &graph,
        &tensor,
        &sensors,
        "MH02-DX-9999",
        start_time,
        TripMode::SmoothCruise { target_distance_m: 5_000.0 },
        0.02,
        &mut rng,
    ).expect("Failed to generate ground truth trip");

    let ground_truth = &trip.ground_truth_nodes;
    let sightings = &trip.camera_sightings;

    assert!(!sightings.is_empty(), "Trip must have camera sightings");

    let t0 = Instant::now();
    let mut tracker = StreamingCausalTracker::new(&graph, &tensor).with_beam_width(4);

    for s in sightings {
        tracker.ingest_sighting(s.clone());
    }
    let resolve_time = t0.elapsed();

    let track = tracker.get_track("MH02-DX-9999").expect("Track not found");
    let reconstructed = track.best_reconstructed_path();

    let first_cam = sightings.first().unwrap();
    let last_cam = sightings.last().unwrap();

    // Exact slice from first observed camera node to last observed camera node
    let s_idx = ground_truth
        .iter()
        .position(|&n| n == first_cam.edge_from)
        .or_else(|| ground_truth.iter().position(|&n| n == first_cam.edge_to))
        .unwrap_or(0);

    let e_idx = ground_truth
        .iter()
        .rposition(|&n| n == last_cam.edge_to)
        .or_else(|| ground_truth.iter().rposition(|&n| n == last_cam.edge_from))
        .unwrap_or(ground_truth.len() - 1);

    let monitored_ground_truth = &ground_truth[s_idx..=e_idx];

    let truth_set: AHashSet<NodeId> = monitored_ground_truth.iter().copied().collect();
    let recon_set: AHashSet<NodeId> = reconstructed.iter().copied().collect();
    let matched_nodes = truth_set.intersection(&recon_set).count();

    let precision_pct = (matched_nodes as f64 / recon_set.len().max(1) as f64) * 100.0;
    let recall_pct = (matched_nodes as f64 / truth_set.len().max(1) as f64) * 100.0;

    println!("\n==========================================================");
    println!("     HMM BEAM TRACKER & ANOMALY RESOLUTION AUDIT          ");
    println!("==========================================================");
    println!("Total Ground Truth Nodes:      {}", ground_truth.len());
    println!(
        "Monitored Envelope Nodes:      {} ({:.1}%)",
        truth_set.len(),
        (truth_set.len() as f64 / ground_truth.len() as f64) * 100.0
    );
    println!("Camera Sightings Ingested:     {}", sightings.len());
    println!("Reconstructed Unique Nodes:    {}", recon_set.len());
    println!("Active Hypotheses in Beam:     {}", track.beam_hypotheses.len());
    println!("Total Anomalies Detected:      {}", track.all_anomalies.len());
    println!("Resolution Time (Total Trip):  {:?}", resolve_time);
    println!(
        "Avg Resolution Time per Gap:   {:.2?}",
        resolve_time / track.legs.len().max(1) as u32
    );
    println!("Path Precision:                {:.1}%", precision_pct);
    println!("Monitored Section Recall:      {:.1}%", recall_pct);

    for (i, alert) in track.all_anomalies.iter().take(5).enumerate() {
        println!(
            "  Alert #{}: [{:?}] Severity {:.1} | {}",
            i + 1,
            alert.anomaly_type,
            alert.severity_score,
            alert.explanation
        );
    }
    println!("==========================================================\n");

    // Defense-grade operational boundaries
    assert!(precision_pct >= 82.0, "Precision dropped below threshold: {:.1}%", precision_pct);
    assert!(recall_pct >= 85.0, "Recall dropped below threshold: {:.1}%", recall_pct);
    assert!(recall_pct <= 100.0, "Recall cannot exceed 100%");
    assert!(resolve_time.as_millis() < 50, "Resolution took longer than 50ms");
}

#[test]
fn test_adversarial_anomaly_injection() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let bin_path = root.join("data/road/mumbai_complete_graph.bin");

    if !bin_path.exists() {
        return;
    }

    let graph = RoadGraph::open(&bin_path).expect("Failed to mmap RoadGraph");
    let tensor = TrafficTensorEngine::new(&graph);

    let config = OptimizerConfig::default();
    let opt = optimize_camera_network(&graph, &config);
    let sensors = CameraSensorIndex::from_poles(opt.camera_poles);

    let mut rng = StdRng::seed_from_u64(999);
    let start_time = Utc::now();
    let mut trip = generate_vehicle_trip(
        &graph,
        &tensor,
        &sensors,
        "MH02-SUSPECT-01",
        start_time,
        TripMode::SmoothCruise { target_distance_m: 4_000.0 },
        0.0,
        &mut rng,
    ).expect("Trip generation failed");

    if trip.camera_sightings.len() >= 6 {
        // 1. Inject a 15-minute unmonitored loiter/stash stop
        for s in &mut trip.camera_sightings[3..] {
            s.timestamp = s.timestamp + chrono::Duration::seconds(900);
        }

        // 2. Inject an impossible 0.8s sprint across a non-zero corridor
        let prev_ts = trip.camera_sightings[4].timestamp;
        trip.camera_sightings[5].timestamp = prev_ts + chrono::Duration::milliseconds(800);
    }

    let mut tracker = StreamingCausalTracker::new(&graph, &tensor).with_beam_width(4);
    for s in trip.camera_sightings {
        tracker.ingest_sighting(s);
    }

    let track = tracker.get_track("MH02-SUSPECT-01").expect("Track not found");

    println!("\n==========================================================");
    println!("     ADVERSARIAL ANOMALY INJECTION VERIFICATION           ");
    println!("==========================================================");
    println!("Total Alerts Flagged: {}", track.all_anomalies.len());
    for (i, alert) in track.all_anomalies.iter().enumerate() {
        println!(
            "  Alert #{}: [{:?}] Severity: {:.1} | {}",
            i + 1,
            alert.anomaly_type,
            alert.severity_score,
            alert.explanation
        );
    }
    println!("==========================================================\n");

    assert!(!track.all_anomalies.is_empty(), "Tracker must detect injected adversarial anomalies");
}

#[test]
fn test_continuous_ephemeris_spline_reconstruction() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let bin_path = root.join("data/road/mumbai_complete_graph.bin");

    if !bin_path.exists() {
        return;
    }

    let graph = RoadGraph::open(&bin_path).expect("Failed to mmap RoadGraph");
    let tensor = TrafficTensorEngine::new(&graph);

    let config = OptimizerConfig::default();
    let opt = optimize_camera_network(&graph, &config);
    let sensors = CameraSensorIndex::from_poles(opt.camera_poles);

    let mut rng = StdRng::seed_from_u64(12345);
    let start_time = Utc::now();
    let trip = generate_vehicle_trip(
        &graph,
        &tensor,
        &sensors,
        "MH02-EPHEM-TEST",
        start_time,
        TripMode::SmoothCruise { target_distance_m: 3_000.0 },
        0.02,
        &mut rng,
    ).expect("Trip generation failed");

    let mut tracker = StreamingCausalTracker::new(&graph, &tensor).with_beam_width(4);
    for s in trip.camera_sightings {
        tracker.ingest_sighting(s);
    }

    let track = tracker.get_track("MH02-EPHEM-TEST").expect("Track not found");
    let ephemeris = track.reconstruct_continuous_ephemeris(&graph, &tensor);

    println!("\n==========================================================");
    println!("     CONTINUOUS KINEMATIC EPHEMERIS RECONSTRUCTION AUDIT   ");
    println!("==========================================================");
    println!("Reconstructed Nodes Count:     {}", track.best_reconstructed_path().len());
    println!("Continuous Spline Points:      {}", ephemeris.len());

    for (i, pt) in ephemeris.iter().take(5).enumerate() {
        println!(
            "  Point #{}: ({:.6}, {:.6}) @ {} | Speed: {:.1} km/h",
            i + 1,
            pt.0,
            pt.1,
            pt.2.format("%H:%M:%S%.3f"),
            pt.3
        );
    }
    println!("==========================================================\n");

    assert!(!ephemeris.is_empty(), "Ephemeris points must be generated");
    assert!(ephemeris.len() >= track.best_reconstructed_path().len());

    for w in ephemeris.windows(2) {
        assert!(w[0].2 <= w[1].2, "Ephemeris timestamps must be non-decreasing");
    }

    for pt in &ephemeris {
        assert!(pt.0 >= 18.8 && pt.0 <= 19.5, "Latitude must be within Mumbai");
        assert!(pt.1 >= 72.7 && pt.1 <= 73.2, "Longitude must be within Mumbai");
    }
}