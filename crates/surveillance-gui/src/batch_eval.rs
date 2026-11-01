use crate::metrics::{compute_lcss_accuracy_pct, compute_path_length_m};
use engine_core::CityGraph;
use rayon::prelude::*;
use std::time::Instant;
use trajectory_resolver::StreamingTracker;
use trip_simulator::{generate_chaotic_trip, CameraNetwork, GtfsCorridorSpeeds, TripBudget};

#[derive(Debug, Clone, Default)]
pub struct BatchMetricsReport {
    pub total_simulations: usize,
    pub successful_intercepts: usize,
    pub intercept_rate_pct: f64,
    pub execution_duration_sec: f64,

    // Accuracy Metrics (Monitored LCSS %)
    pub accuracy_mean: f64,
    pub accuracy_p25: f64,
    pub accuracy_p50: f64,
    pub accuracy_p75: f64,
    pub accuracy_p90: f64,
    pub accuracy_p99: f64,

    // Journey Capture Ratio (% of journey between first & last camera)
    pub capture_ratio_mean: f64,
    pub capture_ratio_p50: f64,
    pub capture_ratio_p90: f64,

    // Blind Zone Distances (m)
    pub avg_unobserved_gap_m: f64,
    pub p90_unobserved_gap_m: f64,

    // Latency (µs)
    pub latency_mean_us: f64,
    pub latency_p95_us: f64,
    pub latency_p99_us: f64,

    // Detailed Distributions
    pub accuracy_distribution: [usize; 10], // 10 buckets of 10%
    pub cdf_samples: Vec<(f32, f32)>,       // (accuracy_pct, cumulative_pct)
}

struct SingleRunData {
    monitored_accuracy: f64,
    capture_ratio: f64,
    unobserved_gap_m: f64,
    latency_us: f64,
}

pub fn run_monte_carlo_batch_eval(
    city: &CityGraph,
    cameras: &CameraNetwork,
    gtfs_speeds: Option<&GtfsCorridorSpeeds>,
    num_runs: usize,
    trip_duration_secs: f64,
    dropout_rate: f64,
    max_candidates: usize,
) -> BatchMetricsReport {
    let start_time = Instant::now();

    let runs: Vec<Option<SingleRunData>> = (0..num_runs)
        .into_par_iter()
        .map(|_| {
            let trip = generate_chaotic_trip(
                city,
                cameras,
                gtfs_speeds,
                TripBudget::DurationSecs(trip_duration_secs),
                dropout_rate,
            )?;

            if trip.camera_sightings.len() < 2 {
                return None;
            }

            let mut tracker = StreamingTracker::new(city).with_candidate_limit(max_candidates);
            let t0 = Instant::now();
            for sighting in trip.camera_sightings.clone() {
                tracker.ingest_sighting(sighting);
            }
            let latency_us = t0.elapsed().as_micros() as f64;

            let track = tracker.get_track("MH02CB1234")?;

            let first_cam = trip.camera_sightings.first().unwrap().edge_from;
            let last_cam = trip.camera_sightings.last().unwrap().edge_to;

            let s_idx = trip.ground_truth_nodes.iter().position(|&n| n == first_cam).unwrap_or(0);
            let e_idx = trip.ground_truth_nodes.iter().rposition(|&n| n == last_cam).unwrap_or(trip.ground_truth_nodes.len() - 1);

            let mon_gt = if s_idx <= e_idx {
                &trip.ground_truth_nodes[s_idx..=e_idx]
            } else {
                &trip.ground_truth_nodes[..]
            };

            let acc = compute_lcss_accuracy_pct(mon_gt, &track.reconstructed_path);
            let full_len = compute_path_length_m(city, &trip.ground_truth_nodes);
            let mon_len = compute_path_length_m(city, mon_gt);
            let cap_ratio = if full_len > 0.0 { (mon_len / full_len) * 100.0 } else { 0.0 };

            Some(SingleRunData {
                monitored_accuracy: acc,
                capture_ratio: cap_ratio,
                unobserved_gap_m: (full_len - mon_len).max(0.0),
                latency_us,
            })
        })
        .collect();

    let mut valid: Vec<SingleRunData> = runs.into_iter().flatten().collect();
    let n = valid.len();
    if n == 0 {
        return BatchMetricsReport::default();
    }

    valid.sort_by(|a, b| a.monitored_accuracy.partial_cmp(&b.monitored_accuracy).unwrap());

    let p = |idx_pct: usize| -> f64 {
        let i = ((idx_pct * n) / 100).min(n - 1);
        valid[i].monitored_accuracy
    };

    let mean_acc = valid.iter().map(|r| r.monitored_accuracy).sum::<f64>() / n as f64;

    let mut caps: Vec<f64> = valid.iter().map(|r| r.capture_ratio).collect();
    caps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_cap = caps.iter().sum::<f64>() / n as f64;
    let p50_cap = caps[n * 50 / 100];
    let p90_cap = caps[(n * 90 / 100).min(n - 1)];

    let mut gaps: Vec<f64> = valid.iter().map(|r| r.unobserved_gap_m).collect();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_gap = gaps.iter().sum::<f64>() / n as f64;
    let p90_gap = gaps[(n * 90 / 100).min(n - 1)];

    let mut lats: Vec<f64> = valid.iter().map(|r| r.latency_us).collect();
    lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_lat = lats.iter().sum::<f64>() / n as f64;
    let p95_lat = lats[(n * 95 / 100).min(n - 1)];
    let p99_lat = lats[(n * 99 / 100).min(n - 1)];

    let mut dist = [0usize; 10];
    for r in &valid {
        let bucket = ((r.monitored_accuracy / 10.0).floor() as usize).min(9);
        dist[bucket] += 1;
    }

    let cdf_steps = 50;
    let mut cdf = Vec::with_capacity(cdf_steps);
    for step in 0..=cdf_steps {
        let cutoff = (step as f64) * (100.0 / cdf_steps as f64);
        let count = valid.iter().take_while(|r| r.monitored_accuracy <= cutoff).count();
        let cum_pct = (count as f32 / n as f32) * 100.0;
        cdf.push((cutoff as f32, cum_pct));
    }

    BatchMetricsReport {
        total_simulations: num_runs,
        successful_intercepts: n,
        intercept_rate_pct: (n as f64 / num_runs as f64) * 100.0,
        execution_duration_sec: start_time.elapsed().as_secs_f64(),

        accuracy_mean: mean_acc,
        accuracy_p25: p(25),
        accuracy_p50: p(50),
        accuracy_p75: p(75),
        accuracy_p90: p(90),
        accuracy_p99: p(99),

        capture_ratio_mean: mean_cap,
        capture_ratio_p50: p50_cap,
        capture_ratio_p90: p90_cap,

        avg_unobserved_gap_m: mean_gap,
        p90_unobserved_gap_m: p90_gap,

        latency_mean_us: mean_lat,
        latency_p95_us: p95_lat,
        latency_p99_us: p99_lat,

        accuracy_distribution: dist,
        cdf_samples: cdf,
    }
}