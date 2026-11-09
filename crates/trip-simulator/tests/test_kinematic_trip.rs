use network_optimizer::cut::find_topological_cuts;
use network_optimizer::pruner::{prune_to_zero_escape_network, ZeroEscapePrunerConfig};
use network_optimizer::skeleton::build_skeleton_graph;
use osm_loader::load_osm_pbf;
use std::path::PathBuf;
use std::time::Instant;
use trip_simulator::{generate_chaotic_trip, CameraNetwork, GtfsCorridorSpeeds, TripBudget};

#[test]
fn test_chaotic_trip_kinematics() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let pbf_path = root.join("data/road/western-zone-261028.osm.pbf");
    let gtfs_dir = root.join("data/gtfs");

    let bkc_bbox = Some((19.0400, 72.8400, 19.0900, 72.9000));

    let t0 = Instant::now();
    let (city, _) = load_osm_pbf(&pbf_path, bkc_bbox).expect("Failed to parse OSM");
    println!("Graph loaded in {:.2?}", t0.elapsed());

    let t1 = Instant::now();
    let gtfs_speeds = GtfsCorridorSpeeds::load_from_dir(&gtfs_dir, &city).ok();
    println!("GTFS grid-indexed in {:.2?}", t1.elapsed());

    // 1. Generate the Zero-Escape Camera Network using network-optimizer
    let t_opt = Instant::now();
    let skeleton = build_skeleton_graph(&city);
    let cuts = find_topological_cuts(&city, &skeleton);
    let config = ZeroEscapePrunerConfig {
        max_dark_distance_m: 500.0,
        max_dark_hops: 5,
        max_unmonitored_forks: 2,
    };
    let (optimized_poles, summary) = prune_to_zero_escape_network(&city, &skeleton, &cuts, config);
    println!(
        "Optimized {} cameras (pruned {} redundant) in {:.2?}",
        summary.final_active_cameras,
        summary.pruned_redundant_cameras,
        t_opt.elapsed()
    );

    // 2. Deploy the optimized poles into the simulation's CameraNetwork
    let mut cameras = CameraNetwork::new();
    for pole in &optimized_poles {
        cameras.deploy_pole(
            pole.camera_id.clone(),
            pole.edge_from,
            pole.edge_to,
            pole.lat,
            pole.lon,
            pole.is_two_way,
        );
    }

    // 3. Generate a 20-minute chaotic trip (1200 seconds budget)
    let t2 = Instant::now();
    let trip = generate_chaotic_trip(
        &city,
        &cameras,
        gtfs_speeds.as_ref(),
        TripBudget::DurationSecs(1200.0),
        0.0, // 0% camera dropout for baseline verification
    )
    .expect("Failed to generate trip");
    let sim_duration = t2.elapsed();

    let travel_secs = (trip.end_time - trip.start_time).num_seconds();
    let km = trip.total_distance_m / 1000.0;
    let avg_speed = km / (travel_secs as f64 / 3600.0);
    let total_sightings = trip.camera_sightings.len();

    println!("\n==============================================");
    println!("     REALISTIC MUMBAI COMMUTE SIMULATION      ");
    println!("==============================================");
    println!("Simulation Compute Time:  {:.2?} (instant)", sim_duration);
    println!("Total Path Nodes:         {}", trip.ground_truth_nodes.len());
    println!("Total Distance Traveled:  {:.2} km", km);
    println!(
        "Elapsed Travel Time:      {} min {} sec ({} total secs)",
        travel_secs / 60,
        travel_secs % 60,
        travel_secs
    );
    println!("Average Journey Speed:    {:.1} km/h", avg_speed);
    println!("Total Cameras Intercepted: {}", total_sightings);

    // Calculate average spacing between camera hits
    if total_sightings >= 2 {
        let avg_time_between_hits_secs = travel_secs as f64 / (total_sightings - 1) as f64;
        let avg_distance_between_hits_m = trip.total_distance_m / (total_sightings - 1) as f64;
        println!(
            "Avg Spacing Between Hits: {:.1} meters (~{:.0} seconds)",
            avg_distance_between_hits_m, avg_time_between_hits_secs
        );
    }

    println!("\n--- CAMERA SIGHTINGS CHRONOLOGY ---");
    for (i, s) in trip.camera_sightings.iter().enumerate() {
        println!(
            "  • Hit #{:02}: [{}] at {} | Directed Edge: ({} -> {}) | Speed: {:.1} km/h",
            i + 1,
            s.camera_id,
            s.timestamp.format("%H:%M:%S"),
            s.edge_from,
            s.edge_to,
            s.speed_kmh
        );
    }
    println!("==============================================\n");

    assert!(travel_secs >= 1200, "Trip was shorter than budgeted duration");
    assert!(
        total_sightings > 0,
        "Vehicle escaped undetected through the zero-escape network!"
    );
}