use ahash::{AHashMap, AHashSet};
use camera_network_optimizer::{optimize_camera_network, OptimizerConfig};
use chrono::Duration;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use road_graph_loader::RoadGraph;
use road_network_core::{CameraSighting, NodeId};
use road_traffic_tensor::TrafficTensorEngine;
use sim_trip_generator::{simulate_fleet, CameraSensorIndex, FleetSimulationConfig, SimulatedTrip};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tracking_resolver::{AnomalyType, StreamingCausalTracker};

struct VehicleScore {
    precision: f64,
    recall: f64,
    reconstructed_count: usize,
    truth_count: usize,
}

#[test]
fn test_massive_fleet_trajectory_reconstruction_and_anomalies() {
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

    // Default configuration: 1,000 vehicles (scale to 10,000 or 50,000 via FLEET_SIZE env var)
    let fleet_size = std::env::var("FLEET_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1000);

    let fleet_config = FleetSimulationConfig {
        num_vehicles: fleet_size,
        min_distance_m: 3_500.0,
        max_distance_m: 8_000.0,
        camera_dropout_rate: 0.03, // 3% random optical dropouts
        base_seed: 0xCAFE_BABE_9999,
    };

    println!("\n==========================================================");
    println!("  STAGE 1: GENERATING HIGH-FIDELITY VEHICLE FLEET ({})    ", fleet_size);
    println!("==========================================================");
    let t_gen = Instant::now();
    let mut fleet_res = simulate_fleet(&graph, &tensor, &sensors, fleet_config);
    let gen_dur = t_gen.elapsed();
    println!("Fleet simulated in: {:.2?} ({:.1} km driven)", gen_dur, fleet_res.total_distance_km);

    // STAGE 2: INJECT ADVERSARIAL ANOMALIES ACROSS 10% OF THE FLEET
    let mut rng = StdRng::seed_from_u64(0x1337_5EED);
    let mut ground_truth_map: AHashMap<String, SimulatedTrip> = AHashMap::new();
    let mut injected_loiters = 0;
    let mut injected_speeders = 0;

    for trip in &mut fleet_res.trips {
        if trip.camera_sightings.len() >= 6 {
            // 5% of fleet commits unmonitored stash/loiter halts
            if rng.random_bool(0.05) {
                let delay_secs = rng.random_range(180..600);
                for s in &mut trip.camera_sightings[3..] {
                    s.timestamp = s.timestamp + Duration::seconds(delay_secs);
                }
                injected_loiters += 1;
            }
            // 5% of fleet commits excessive speeding / getaways
            else if rng.random_bool(0.05) {
                let prev_ts = trip.camera_sightings[2].timestamp;
                trip.camera_sightings[3].timestamp = prev_ts + Duration::milliseconds(900);
                trip.camera_sightings[3].speed_kmh = rng.random_range(85.0..120.0);
                injected_speeders += 1;
            }
        }
        ground_truth_map.insert(trip.vehicle_id.clone(), trip.clone());
    }

    // Re-flatten and sort sightings chronologically to simulate municipal firehose
    let mut firehose: Vec<CameraSighting> = fleet_res
        .trips
        .iter()
        .flat_map(|t| t.camera_sightings.clone())
        .collect();
    firehose.par_sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    println!("Injected Stash Stops:          {}", injected_loiters);
    println!("Injected Speed Violations:     {}", injected_speeders);
    println!("Total Streamed Sightings:      {}", firehose.len());

    // STAGE 3: CONCURRENT PARTITION-SHARDED STREAMING TRACKING
    println!("\n==========================================================");
    println!("  STAGE 2: STREAMING RESOLUTION BENCHMARK                 ");
    println!("==========================================================");

    // Partition vehicles into 16 concurrent buckets to parallelize ingestion across CPU cores
    const NUM_SHARDS: usize = 16;
    let mut shards: Vec<Vec<CameraSighting>> = (0..NUM_SHARDS).map(|_| Vec::new()).collect();

    for sighting in firehose {
        let mut hasher = ahash::AHasher::default();
        std::hash::Hash::hash(&sighting.vehicle_id, &mut hasher);
        let shard_idx = (std::hash::Hasher::finish(&hasher) as usize) % NUM_SHARDS;
        shards[shard_idx].push(sighting);
    }

    let t_resolve = Instant::now();
    let total_legs_resolved = AtomicUsize::new(0);

    // Parallel streaming resolution
    let shard_results: Vec<(Vec<VehicleScore>, AHashMap<AnomalyType, usize>)> = shards
        .into_par_iter()
        .map(|shard_stream| {
            let mut tracker = StreamingCausalTracker::new(&graph, &tensor).with_beam_width(4);
            let mut leg_count = 0;

            for sighting in shard_stream {
                tracker.ingest_sighting(sighting);
                leg_count += 1;
            }
            total_legs_resolved.fetch_add(leg_count, Ordering::Relaxed);

            let mut scores = Vec::new();
            let mut anomaly_counts: AHashMap<AnomalyType, usize> = AHashMap::new();

            for (vid, gt_trip) in &ground_truth_map {
                if let Some(track) = tracker.get_track(vid) {
                    let recon = track.best_reconstructed_path();
                    if recon.is_empty() {
                        continue;
                    }

                    // Tally anomalies
                    for alert in &track.all_anomalies {
                        *anomaly_counts.entry(alert.anomaly_type).or_insert(0) += 1;
                    }

                    // Compute set intersection
                    let first_cam = gt_trip.camera_sightings.first().unwrap();
                    let last_cam = gt_trip.camera_sightings.last().unwrap();

                    let s_idx = gt_trip
                        .ground_truth_nodes
                        .iter()
                        .position(|&n| n == first_cam.edge_from || n == first_cam.edge_to)
                        .unwrap_or(0);
                    let e_idx = gt_trip
                        .ground_truth_nodes
                        .iter()
                        .rposition(|&n| n == last_cam.edge_to || n == last_cam.edge_from)
                        .unwrap_or(gt_trip.ground_truth_nodes.len() - 1);

                    let truth_nodes = &gt_trip.ground_truth_nodes[s_idx..=e_idx];
                    let truth_set: AHashSet<NodeId> = truth_nodes.iter().copied().collect();
                    let recon_set: AHashSet<NodeId> = recon.iter().copied().collect();

                    let hits = truth_set.intersection(&recon_set).count();
                    let prec = (hits as f64 / recon_set.len().max(1) as f64) * 100.0;
                    let rec = (hits as f64 / truth_set.len().max(1) as f64) * 100.0;

                    scores.push(VehicleScore {
                        precision: prec,
                        recall: rec,
                        reconstructed_count: recon_set.len(),
                        truth_count: truth_set.len(),
                    });
                }
            }

            (scores, anomaly_counts)
        })
        .collect();

    let resolve_dur = t_resolve.elapsed();

    // STAGE 4: AGGREGATE STATISTICAL AUDIT
    let mut all_precisions = Vec::new();
    let mut all_recalls = Vec::new();
    let mut total_anomalies_detected: AHashMap<AnomalyType, usize> = AHashMap::new();

    for (scores, anomalies) in shard_results {
        for s in scores {
            all_precisions.push(s.precision);
            all_recalls.push(s.recall);
        }
        for (k, v) in anomalies {
            *total_anomalies_detected.entry(k).or_insert(0) += v;
        }
    }

    all_precisions.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_recalls.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = all_precisions.len() as f64;
    let mean_prec = all_precisions.iter().sum::<f64>() / n;
    let mean_rec = all_recalls.iter().sum::<f64>() / n;
    let median_prec = all_precisions[all_precisions.len() / 2];
    let median_rec = all_recalls[all_recalls.len() / 2];

    let p95_prec = all_precisions[(all_precisions.len() as f64 * 0.05) as usize];
    let p95_rec = all_recalls[(all_recalls.len() as f64 * 0.05) as usize];

    let total_sightings = total_legs_resolved.load(Ordering::Relaxed);
    let throughput_sightings_sec = total_sightings as f64 / resolve_dur.as_secs_f64();
    let throughput_vehicles_sec = fleet_size as f64 / resolve_dur.as_secs_f64();

    println!("\n==========================================================");
    println!("     FLEET SCALE BENCHMARK & RECONSTRUCTION AUDIT         ");
    println!("==========================================================");
    println!("Fleet Size Evaluated:          {} vehicles", fleet_size);
    println!("Total Sightings Processed:     {} captures", total_sightings);
    println!("Total Wall-Clock Time:         {:.2?}", resolve_dur);
    println!("Throughput:                    {:.0} sightings / sec", throughput_sightings_sec);
    println!("Vehicle Processing Rate:       {:.0} full trips / sec", throughput_vehicles_sec);
    println!("Average Gap Resolution Time:   {:.2} µs / sighting", (resolve_dur.as_secs_f64() * 1_000_000.0) / total_sightings as f64);

    println!("\n--- PATH ACCURACY DISTRIBUTION ---");
    println!("Precision:  Mean = {:.1}% | Median = {:.1}% | 5th-Percentile = {:.1}%", mean_prec, median_prec, p95_prec);
    println!("Recall:     Mean = {:.1}% | Median = {:.1}% | 5th-Percentile = {:.1}%", mean_rec, median_rec, p95_rec);

    println!("\n--- ANOMALY & ADVERSARIAL DETECTION AUDIT ---");
    println!("Stash / Loitering Alerts:      {} detected (Injected: {})", total_anomalies_detected.get(&AnomalyType::LoiteringOrStashStop).unwrap_or(&0), injected_loiters);
    println!("Excessive Speed Alerts:        {} detected (Injected: {})", total_anomalies_detected.get(&AnomalyType::ExcessiveSpeed).unwrap_or(&0), injected_speeders);
    println!("Dark Corridor Diversions:      {} detected", total_anomalies_detected.get(&AnomalyType::DarkCorridorEvasion).unwrap_or(&0));
    println!("Sensor Clock Glitches:         {} detected", total_anomalies_detected.get(&AnomalyType::SensorClockAnomaly).unwrap_or(&0));
    println!("==========================================================\n");

    assert!(mean_prec >= 88.0, "Mean precision dropped below target");
    assert!(mean_rec >= 85.0, "Mean recall dropped below target");
    assert!(mean_rec <= 100.0, "Recall invariant violated");
    assert!(throughput_sightings_sec >= 10_000.0, "Throughput below target rate");
}