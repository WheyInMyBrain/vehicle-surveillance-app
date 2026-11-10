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
use tracking_resolver::{AnomalyType, ResolverConfig, StreamingCausalTracker};

#[allow(dead_code)]
struct VehicleScore {
    precision: f64,
    recall: f64,
    reconstructed_count: usize,
    truth_count: usize,
}

#[allow(dead_code)]
struct DropoutSweepResult {
    rate_pct: f64,
    total_sightings: usize,
    throughput_sightings_sec: f64,
    avg_latency_us: f64,
    mean_precision: f64,
    median_precision: f64,
    p95_precision: f64,
    mean_recall: f64,
    median_recall: f64,
    p95_recall: f64,
    loiter_alerts: usize,
    speed_alerts: usize,
}

fn execute_dropout_benchmark(
    graph: &RoadGraph,
    tensor: &TrafficTensorEngine,
    sensors: &CameraSensorIndex,
    camera_edge_set: &AHashSet<usize>,
    fleet_size: usize,
    dropout_rate: f64,
    seed: u64,
) -> DropoutSweepResult {
    let fleet_config = FleetSimulationConfig {
        num_vehicles: fleet_size,
        min_distance_m: 3_500.0,
        max_distance_m: 10_000.0,
        camera_dropout_rate: dropout_rate,
        base_seed: seed,
    };

    let mut fleet_res = simulate_fleet(graph, tensor, sensors, fleet_config);

    // Inject anomalies into 10% of qualifying fleet trips
    let mut rng = StdRng::seed_from_u64(0x1337_5EED ^ (seed as u64));
    let mut ground_truth_map: AHashMap<String, SimulatedTrip> = AHashMap::new();

    for trip in &mut fleet_res.trips {
        if trip.camera_sightings.len() >= 6 {
            if rng.random_bool(0.05) {
                let delay_secs = rng.random_range(180..600);
                for s in &mut trip.camera_sightings[3..] {
                    s.timestamp = s.timestamp + Duration::seconds(delay_secs);
                }
            } else if rng.random_bool(0.05) {
                let prev_ts = trip.camera_sightings[2].timestamp;
                trip.camera_sightings[3].timestamp = prev_ts + Duration::milliseconds(900);
                trip.camera_sightings[3].speed_kmh = rng.random_range(85.0..120.0);
            }
        }
        ground_truth_map.insert(trip.vehicle_id.clone(), trip.clone());
    }

    let mut firehose: Vec<CameraSighting> = fleet_res
        .trips
        .iter()
        .flat_map(|t| t.camera_sightings.clone())
        .collect();
    firehose.par_sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    const NUM_SHARDS: usize = 16;
    let mut shards: Vec<Vec<CameraSighting>> = (0..NUM_SHARDS).map(|_| Vec::new()).collect();
    let mut shard_vids: Vec<AHashSet<String>> = (0..NUM_SHARDS).map(|_| AHashSet::new()).collect();

    for sighting in firehose {
        let mut hasher = ahash::AHasher::default();
        std::hash::Hash::hash(&sighting.vehicle_id, &mut hasher);
        let shard_idx = (std::hash::Hasher::finish(&hasher) as usize) % NUM_SHARDS;
        shard_vids[shard_idx].insert(sighting.vehicle_id.clone());
        shards[shard_idx].push(sighting);
    }

    let t_resolve = Instant::now();
    let total_legs_resolved = AtomicUsize::new(0);
    let resolver_cfg = ResolverConfig::default();

    let shard_results: Vec<(Vec<VehicleScore>, AHashMap<AnomalyType, usize>)> = shards
        .into_par_iter()
        .zip(shard_vids.into_par_iter())
        .map(|(shard_stream, vids)| {
            let mut tracker = StreamingCausalTracker::new(graph, tensor)
                .with_beam_width(4)
                .with_config(resolver_cfg.clone())
                .with_unobserved_cameras(camera_edge_set.clone());

            let mut leg_count = 0;
            for sighting in shard_stream {
                tracker.ingest_sighting(sighting);
                leg_count += 1;
            }
            total_legs_resolved.fetch_add(leg_count, Ordering::Relaxed);

            let mut scores = Vec::with_capacity(vids.len());
            let mut anomaly_counts: AHashMap<AnomalyType, usize> = AHashMap::new();

            for vid in &vids {
                if let (Some(gt_trip), Some(track)) = (ground_truth_map.get(vid), tracker.get_track(vid)) {
                    let recon = track.best_reconstructed_path();
                    if recon.is_empty() {
                        continue;
                    }

                    for alert in &track.all_anomalies {
                        *anomaly_counts.entry(alert.anomaly_type).or_insert(0) += 1;
                    }

                    if let (Some(first_cam), Some(last_cam)) =
                        (gt_trip.camera_sightings.first(), gt_trip.camera_sightings.last())
                    {
                        let s_idx = gt_trip
                            .ground_truth_nodes
                            .iter()
                            .position(|&n| n == first_cam.edge_from || n == first_cam.edge_to)
                            .unwrap_or(0);
                        let e_idx = gt_trip
                            .ground_truth_nodes
                            .iter()
                            .rposition(|&n| n == last_cam.edge_to || n == last_cam.edge_from)
                            .unwrap_or(gt_trip.ground_truth_nodes.len().saturating_sub(1));

                        if s_idx <= e_idx && e_idx < gt_trip.ground_truth_nodes.len() {
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
                }
            }

            (scores, anomaly_counts)
        })
        .collect();

    let resolve_dur = t_resolve.elapsed();

    let mut all_precisions = Vec::new();
    let mut all_recalls = Vec::new();
    let mut total_anomalies: AHashMap<AnomalyType, usize> = AHashMap::new();

    for (scores, anomalies) in shard_results {
        for s in scores {
            all_precisions.push(s.precision);
            all_recalls.push(s.recall);
        }
        for (k, v) in anomalies {
            *total_anomalies.entry(k).or_insert(0) += v;
        }
    }

    all_precisions.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_recalls.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = all_precisions.len().max(1) as f64;
    let mean_prec = all_precisions.iter().sum::<f64>() / n;
    let mean_rec = all_recalls.iter().sum::<f64>() / n;
    let median_prec = all_precisions.get(all_precisions.len() / 2).copied().unwrap_or(0.0);
    let median_rec = all_recalls.get(all_recalls.len() / 2).copied().unwrap_or(0.0);
    let p95_prec = all_precisions.get((all_precisions.len() as f64 * 0.05) as usize).copied().unwrap_or(0.0);
    let p95_rec = all_recalls.get((all_recalls.len() as f64 * 0.05) as usize).copied().unwrap_or(0.0);

    let total_sightings = total_legs_resolved.load(Ordering::Relaxed);
    let throughput = total_sightings as f64 / resolve_dur.as_secs_f64().max(1e-6);
    let avg_latency = (resolve_dur.as_secs_f64() * 1_000_000.0) / total_sightings.max(1) as f64;

    DropoutSweepResult {
        rate_pct: dropout_rate * 100.0,
        total_sightings,
        throughput_sightings_sec: throughput,
        avg_latency_us: avg_latency,
        mean_precision: mean_prec,
        median_precision: median_prec,
        p95_precision: p95_prec,
        mean_recall: mean_rec,
        median_recall: median_rec,
        p95_recall: p95_rec,
        loiter_alerts: total_anomalies.get(&AnomalyType::LoiteringOrStashStop).copied().unwrap_or(0),
        speed_alerts: total_anomalies.get(&AnomalyType::ExcessiveSpeed).copied().unwrap_or(0),
    }
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

    let camera_edge_set: AHashSet<usize> = opt
        .camera_poles
        .iter()
        .flat_map(|p| graph.get_edge_index(p.edge_from, p.edge_to))
        .collect();

    let sensors = CameraSensorIndex::from_poles(opt.camera_poles);

    let default_fleet = if cfg!(debug_assertions) { 150 } else { 1000 };
    let fleet_size = std::env::var("FLEET_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(default_fleet);

    // Standard multi-condition operational sweep:
    // 0% (pristine), 3% (nominal), 10% (weather/glare), 25% (degraded network), 50% (sparse network)
    let dropout_rates = [0.00, 0.03, 0.10, 0.25, 0.50];

    println!("\n==========================================================================================");
    println!("     MULTI-CONDITION SENSOR DROPOUT RESILIENCE AUDIT (Fleet: {})", fleet_size);
    println!("==========================================================================================");

    let mut results = Vec::new();
    for (i, &rate) in dropout_rates.iter().enumerate() {
        let sweep = execute_dropout_benchmark(
            &graph,
            &tensor,
            &sensors,
            &camera_edge_set,
            fleet_size,
            rate,
            0xCAFE_BABE_9999 + (i as u64 * 100),
        );
        results.push(sweep);
    }

    println!("\n┌──────────┬───────────┬──────────────┬──────────────┬──────────────┬──────────────┬────────────┐");
    println!("│ Dropout  │ Captures  │ Throughput   │ Avg Latency  │ Precision    │ Recall       │ Loiter/Spd │");
    println!("│ Rate (%) │ Processed │ (sightings/s)│ (µs/gap)     │ (Mean/P95)   │ (Mean/P95)   │ Alerts     │");
    println!("├──────────┼───────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┤");

    for r in &results {
        println!(
            "│ {:>7.1}% │ {:>9} │ {:>10.0} │ {:>10.2} µs │ {:>5.1}%/{:>4.1}% │ {:>5.1}%/{:>4.1}% │ {:>4}/{:<5} │",
            r.rate_pct,
            r.total_sightings,
            r.throughput_sightings_sec,
            r.avg_latency_us,
            r.mean_precision,
            r.p95_precision,
            r.mean_recall,
            r.p95_recall,
            r.loiter_alerts,
            r.speed_alerts,
        );
    }
    println!("└──────────┴───────────┴──────────────┴──────────────┴──────────────┴──────────────┴────────────┘\n");

    // Operational Assertions Across All Conditions:
    for r in &results {
        if r.rate_pct <= 10.0 {
            assert!(
                r.mean_precision >= 95.0,
                "Precision dropped too low on <=10% dropout: {:.1}%",
                r.mean_precision
            );
            assert!(
                r.mean_recall >= 90.0,
                "Recall dropped too low on <=10% dropout: {:.1}%",
                r.mean_recall
            );
        } else if r.rate_pct <= 50.0 {
            // Even under 50% sensor dropouts, the model must maintain high arterial fidelity
            assert!(
                r.mean_precision >= 82.0,
                "Severe drop in precision on 50% dropout: {:.1}%",
                r.mean_precision
            );
            assert!(
                r.mean_recall >= 80.0,
                "Severe drop in recall on 50% dropout: {:.1}%",
                r.mean_recall
            );
        }
    }
}