use crate::camera::CameraNetwork;
use crate::gtfs::GtfsCorridorSpeeds;
use crate::kinematics::simulate_edge_traversal;
use crate::signals::apply_node_delays;
use chrono::{DateTime, Duration, Utc};
use engine_core::{CameraSighting, CityGraph, NodeId};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use rand::seq::IndexedRandom;
use rand::RngExt;

#[derive(Debug, Clone)]
pub struct SimulatedTrip {
    pub origin: NodeId,
    pub destination: NodeId,
    pub ground_truth_nodes: Vec<NodeId>,
    pub camera_sightings: Vec<CameraSighting>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub total_distance_m: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum TripBudget {
    Intersections(usize),       // e.g. 30 nodes (local run)
    DurationSecs(f64),          // e.g. 1800.0 (30 minute drive)
    DistanceMeters(f64),        // e.g. 10_000.0 (10 km cross-city commute)
}

pub fn generate_chaotic_trip(
    city: &CityGraph,
    cameras: &CameraNetwork,
    gtfs_speeds: Option<&GtfsCorridorSpeeds>,
    budget: TripBudget,
    camera_dropout: f64,
) -> Option<SimulatedTrip> {
    let mut rng = rand::rng();
    let all_node_ids: Vec<NodeId> = city.reverse_node_map.values().copied().collect();

    if all_node_ids.is_empty() {
        return None;
    }

    let origin_id = *all_node_ids.choose(&mut rng)?;
    let mut curr_node_idx = *city.node_map.get(&origin_id)?;

    let mut path_nodes = vec![origin_id];
    let start_time = Utc::now();
    let mut curr_time = start_time;
    let mut sightings = Vec::new();
    let mut prev_bearing: Option<f64> = None;
    let mut vehicle_current_speed_kmh: f64 = rng.random_range(20.0..35.0);
    let mut total_distance_m = 0.0;

    // Buffer to avoid immediate backtracking
    let mut recent_nodes = [0u64; 4];
    let mut recent_count = 0;

    loop {
        // Evaluate stopping conditions based on budget
        let elapsed_secs = (curr_time - start_time).num_milliseconds() as f64 / 1000.0;
        let should_stop = match budget {
            TripBudget::Intersections(min_steps) => path_nodes.len() >= min_steps,
            TripBudget::DurationSecs(target_secs) => elapsed_secs >= target_secs,
            TripBudget::DistanceMeters(target_m) => total_distance_m >= target_m,
        };

        if should_stop && path_nodes.len() >= 10 {
            break;
        }

        // Hard failsafe for dead-end containment
        if path_nodes.len() > 2500 {
            break;
        }

        let outgoing: Vec<_> = city
            .graph
            .edges_directed(curr_node_idx, Direction::Outgoing)
            .collect();

        if outgoing.is_empty() {
            break;
        }

        // Avoid recent loops
        let chosen_edge = if outgoing.len() > 1 && recent_count > 0 {
            let unvisited: Vec<_> = outgoing
                .iter()
                .filter(|e| {
                    let tid = city.reverse_node_map[&e.target()];
                    !recent_nodes[..recent_count].contains(&tid)
                })
                .copied()
                .collect();

            if !unvisited.is_empty() {
                *unvisited.choose(&mut rng)?
            } else {
                *outgoing.choose(&mut rng)?
            }
        } else {
            *outgoing.choose(&mut rng)?
        };

        let next_node_idx = chosen_edge.target();
        let u = city.reverse_node_map[&curr_node_idx];
        let v = city.reverse_node_map[&next_node_idx];
        let edge_meta = chosen_edge.weight();
        let next_node_meta = &city.graph[next_node_idx];

        total_distance_m += edge_meta.length_m;

        let has_camera = cameras.get_sensor(u, v).is_some();
        let gtfs_speed = gtfs_speeds.and_then(|g| g.get_speed(u, v));

        let traversal = simulate_edge_traversal(
            edge_meta,
            prev_bearing,
            gtfs_speed,
            vehicle_current_speed_kmh,
            has_camera,
            &mut rng,
        );

        curr_time = curr_time + Duration::milliseconds((traversal.total_transit_seconds * 1000.0) as i64);
        vehicle_current_speed_kmh = traversal.final_exit_speed_kmh;
        prev_bearing = Some(edge_meta.bearing_deg);

        if let Some(cam) = cameras.get_sensor(u, v) {
            if !rng.random_bool(camera_dropout) {
                sightings.push(CameraSighting {
                    vehicle_id: "MH02CB1234".to_string(),
                    camera_id: cam.camera_id.clone(),
                    edge_from: u,
                    edge_to: v,
                    timestamp: curr_time,
                    speed_kmh: traversal.camera_instantaneous_speed_kmh,
                });
            }
        }

        apply_node_delays(next_node_meta, v, &mut curr_time, &mut rng);

        // Update loop prevention ring
        recent_nodes.rotate_right(1);
        recent_nodes[0] = u;
        if recent_count < 4 {
            recent_count += 1;
        }

        path_nodes.push(v);
        curr_node_idx = next_node_idx;
    }

    if path_nodes.len() < 5 {
        return None;
    }

    let destination_id = *path_nodes.last().unwrap();

    Some(SimulatedTrip {
        origin: origin_id,
        destination: destination_id,
        ground_truth_nodes: path_nodes,
        camera_sightings: sightings,
        start_time,
        end_time: curr_time,
        total_distance_m,
    })
}