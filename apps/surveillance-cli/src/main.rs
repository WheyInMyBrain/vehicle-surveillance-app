use network_optimizer::{generate_synthetic_od_corridors, solve_flow_interception, OptimizationGoal};
use osm_loader::load_osm_pbf;
use std::fs::File;
use std::time::Instant;
use trajectory_resolver::StreamingTracker;
use trip_simulator::{generate_chaotic_trip, CameraNetwork};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==================================================");
    println!("   MUMBAI SURVEILLANCE & TRAJECTORY ENGINE        ");
    println!("==================================================");

    let pbf_path = "western-zone-261028.osm.pbf";
    let bkc_bbox = Some((19.0400, 72.8400, 19.0900, 72.9000));

    println!("\n[1/5] Ingesting OSM PBF...");
    let start_load = Instant::now();
    let city = load_osm_pbf(pbf_path, bkc_bbox)?;
    println!(
        "✔ City Graph initialized in {:.2?}! ({} nodes, {} edges)",
        start_load.elapsed(),
        city.graph.node_count(),
        city.graph.edge_count()
    );

    println!("\n[2/5] Synthesizing 10,000 OD Transit Corridors in Parallel...");
    let start_od = Instant::now();
    let corridors = generate_synthetic_od_corridors(&city, 10000, 10);
    println!("✔ Generated {} corridors in {:.2?}!", corridors.len(), start_od.elapsed());

    println!("\n[3/5] Solving Optimal Camera Placement (100% Interception Target)...");
    let start_opt = Instant::now();
    let optimal_cameras = solve_flow_interception(
        &city,
        &corridors,
        OptimizationGoal::TargetCoverage(1.0), // 100% target coverage
        false,                                 // require_multi_hit
    );
    println!(
        "✔ Solved in {:.2?}! Needed exactly {} cameras for 100% interception.",
        start_opt.elapsed(),
        optimal_cameras.len()
    );

    // Save camera registry to disk
    let registry_file = File::create("cameras_registry.json")?;
    serde_json::to_writer_pretty(registry_file, &optimal_cameras)?;
    println!("✔ Saved hardware registry to 'cameras_registry.json'");

    println!("\n[4/5] Deploying Physical Camera Sensors into CameraNetwork...");
    let mut camera_network = CameraNetwork::new();
    for cam in &optimal_cameras {
        camera_network.deploy_pole(
            cam.camera_id.clone(),
            cam.edge_from,
            cam.edge_to,
            cam.lat,
            cam.lon,
            cam.is_two_way,
        );
    }
    println!("✔ Deployed {} strategic poles into sensor network.", optimal_cameras.len());

    println!("\n[5/5] Running Live Tracking Simulation...");
    let trip = loop {
        if let Some(t) = generate_chaotic_trip(&city, &camera_network, 40, 0.35) {
            if t.camera_sightings.len() >= 4 {
                break t;
            }
        }
    };

    println!("✔ Vehicle trip simulated across {} nodes.", trip.ground_truth_nodes.len());
    println!("  - Interceptions captured: {}", trip.camera_sightings.len());

    let mut tracker = StreamingTracker::new(&city).with_candidate_limit(5);
    let start_tracking = Instant::now();
    for sighting in trip.camera_sightings {
        tracker.ingest_sighting(sighting);
    }
    let resolve_time = start_tracking.elapsed();

    let track = tracker
        .get_track("MH02CB1234")
        .expect("Vehicle tracking failed");

    println!("✔ Trajectory resolved in {:.2?}!", resolve_time);
    println!("  - Reconstructed Nodes : {}", track.primary_route.len());
    println!("  - Blind Zones Handled : {}", track.leg_hypotheses.len());

    println!("\n==================================================");
    println!("             END-TO-END PIPELINE VERIFIED         ");
    println!("==================================================");

    Ok(())
}