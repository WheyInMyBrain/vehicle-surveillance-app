use crate::camera::CameraNetwork;
use chrono::{Duration, Utc};
use engine_core::{CameraSighting, CityGraph, NodeId};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use rand::seq::IndexedRandom;
use rand::RngExt;

pub struct SimulatedTrip {
    pub origin: NodeId,
    pub destination: NodeId,
    pub ground_truth_nodes: Vec<NodeId>,
    pub camera_sightings: Vec<CameraSighting>,
}

/// Generates a meandering, non-optimal route using stochastic goal-directed expansion.
pub fn generate_chaotic_trip(
    city: &CityGraph,
    cameras: &CameraNetwork,
    min_intersections: usize,
    camera_dropout: f64,
) -> Option<SimulatedTrip> {
    let mut rng = rand::rng();
    let all_node_ids: Vec<NodeId> = city.reverse_node_map.values().copied().collect();

    if all_node_ids.len() < min_intersections {
        return None;
    }

    // Pick random origin
    let origin_id = *all_node_ids.choose(&mut rng)?;
    let mut curr_node_idx = *city.node_map.get(&origin_id)?;

    let mut path_nodes = vec![origin_id];
    let mut curr_time = Utc::now();
    let mut current_speed_kmh: f64 = rng.random_range(30.0..50.0);
    let mut sightings = Vec::new();

    let target_steps = rng.random_range(min_intersections..min_intersections + 30);

    for _ in 0..target_steps {
        let outgoing_edges: Vec<_> = city
            .graph
            .edges_directed(curr_node_idx, Direction::Outgoing)
            .collect();

        if outgoing_edges.is_empty() {
            break;
        }

        // Avoid immediate backtracking if multiple branches exist
        let chosen_edge = if outgoing_edges.len() > 1 && path_nodes.len() >= 2 {
            let prev_node_id = path_nodes[path_nodes.len() - 2];
            let filtered: Vec<_> = outgoing_edges
                .iter()
                .filter(|e| city.reverse_node_map[&e.target()] != prev_node_id)
                .copied()
                .collect();
            if !filtered.is_empty() {
                *filtered.choose(&mut rng)?
            } else {
                *outgoing_edges.choose(&mut rng)?
            }
        } else {
            *outgoing_edges.choose(&mut rng)?
        };

        let next_node_idx = chosen_edge.target();
        let u = city.reverse_node_map[&curr_node_idx];
        let v = city.reverse_node_map[&next_node_idx];
        let edge_meta = chosen_edge.weight();

        // Kinematics: speed variation & turn deceleration
        let speed_delta: f64 = rng.random_range(-5.0..5.0);
        current_speed_kmh = (current_speed_kmh + speed_delta).clamp(15.0, 65.0);
        let speed_mps = current_speed_kmh / 3.6;
        let transit_seconds = edge_meta.length_m / speed_mps;
        curr_time = curr_time + Duration::milliseconds((transit_seconds * 1000.0) as i64);

        // Signal idle delay at high-degree junctions
        if outgoing_edges.len() >= 3 && rng.random_bool(0.22) {
            let wait_secs: f64 = rng.random_range(20.0..75.0);
            curr_time = curr_time + Duration::milliseconds((wait_secs * 1000.0) as i64);
            current_speed_kmh = 18.0;
        }

        // Camera detection check
        if let Some(cam) = cameras.get_sensor(u, v) {
            if !rng.random_bool(camera_dropout) {
                let speed_noise: f64 = rng.random_range(-3.0..3.0);
                sightings.push(CameraSighting {
                    vehicle_id: "MH02CB1234".to_string(),
                    camera_id: cam.camera_id.clone(),
                    edge_from: u,
                    edge_to: v,
                    timestamp: curr_time,
                    speed_kmh: (current_speed_kmh + speed_noise).max(5.0),
                });
            }
        }

        path_nodes.push(v);
        curr_node_idx = next_node_idx;
    }

    if path_nodes.len() < min_intersections {
        return None;
    }

    let destination_id = *path_nodes.last().unwrap();

    Some(SimulatedTrip {
        origin: origin_id,
        destination: destination_id,
        ground_truth_nodes: path_nodes,
        camera_sightings: sightings,
    })
}