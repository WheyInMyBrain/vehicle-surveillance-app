use crate::generator::{generate_vehicle_trip, SimulatedTrip, TripMode};
use crate::sensor_index::CameraSensorIndex;
use chrono::{Duration, TimeZone, Utc};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use road_graph_loader::RoadGraph;
use road_network_core::CameraSighting;
use road_traffic_tensor::TrafficTensorEngine;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct FleetSimulationConfig {
    pub num_vehicles: usize,
    pub min_distance_m: f64,
    pub max_distance_m: f64,
    pub camera_dropout_rate: f64,
    pub base_seed: u64,
}

impl Default for FleetSimulationConfig {
    fn default() -> Self {
        Self {
            num_vehicles: 1000,
            min_distance_m: 3_000.0,
            max_distance_m: 8_000.0,
            camera_dropout_rate: 0.02,
            base_seed: 0xDEAD_BEEF_CAFE,
        }
    }
}

pub struct FleetSimulationResult {
    pub trips: Vec<SimulatedTrip>,
    /// Chronologically merged ANPR camera sighting stream as observed by the municipal surveillance center
    pub global_sightings_stream: Vec<CameraSighting>,
    pub total_vehicles_simulated: usize,
    pub total_distance_km: f64,
    pub simulation_duration_ms: f64,
}

/// Concurrently simulates thousands of vehicle trips across Mumbai with random week timestamps
pub fn simulate_fleet(
    graph: &RoadGraph,
    tensor: &TrafficTensorEngine,
    sensors: &CameraSensorIndex,
    config: FleetSimulationConfig,
) -> FleetSimulationResult {
    let t0 = Instant::now();

    // Base week reference: Monday 00:00 UTC
    let base_monday = Utc.with_ymd_and_hms(2026, 11, 9, 0, 0, 0).unwrap();

    let trips: Vec<SimulatedTrip> = (0..config.num_vehicles)
        .into_par_iter()
        .filter_map(|i| {
            let mut rng = StdRng::seed_from_u64(config.base_seed.wrapping_add(i as u64));

            // Randomize start time anywhere in the 7-day 5-minute schedule
            let random_offset_secs = rng.random_range(0..(7 * 86400));
            let start_time = base_monday + Duration::seconds(random_offset_secs);

            let vid = format!("MH02-SIM-{:05}", i + 1);
            let dist = rng.random_range(config.min_distance_m..config.max_distance_m);

            generate_vehicle_trip(
                graph,
                tensor,
                sensors,
                &vid,
                start_time,
                TripMode::SmoothCruise { target_distance_m: dist },
                config.camera_dropout_rate,
                &mut rng,
            )
        })
        .collect();

    // Flatten all sightings and sort chronologically into a realistic incoming ANPR firehose
    let mut all_sightings: Vec<CameraSighting> = trips
        .iter()
        .flat_map(|t| t.camera_sightings.clone())
        .collect();

    all_sightings.par_sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    let total_dist = trips.iter().map(|t| t.total_distance_m).sum::<f64>() / 1000.0;

    FleetSimulationResult {
        total_vehicles_simulated: trips.len(),
        trips,
        global_sightings_stream: all_sightings,
        total_distance_km: total_dist,
        simulation_duration_ms: t0.elapsed().as_secs_f64() * 1000.0,
    }
}