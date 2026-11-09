use crate::sensor_index::CameraSensorIndex;
use chrono::{DateTime, Duration, Utc};
use ordered_float::OrderedFloat;
use rand::seq::IndexedRandom;
use rand::{Rng, RngExt};
use road_graph_loader::{RoadGraph, UnifiedHighwayTier};
use road_network_core::geo::{angle_diff, initial_bearing};
use road_network_core::{CameraSighting, HighwayType, NodeId, RoadSurface};
use road_traffic_tensor::{TrafficTensorEngine, TraversalStatus};
use sim_kinematics::{
    apply_junction_delays, simulate_edge_kinematics, DriverProfile, KinematicEphemerisPoint,
};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone)]
pub struct SimulatedTrip {
    pub vehicle_id: String,
    pub driver_profile: DriverProfile,
    pub origin: NodeId,
    pub destination: NodeId,
    pub ground_truth_nodes: Vec<NodeId>,
    pub ground_truth_ephemeris: Vec<KinematicEphemerisPoint>,
    pub camera_sightings: Vec<CameraSighting>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub total_distance_m: f64,
    pub had_micro_stops: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum TripMode {
    GoalDirected { destination: NodeId },
    SmoothCruise { target_distance_m: f64 },
}

fn map_unified_tier(tier: UnifiedHighwayTier) -> HighwayType {
    match tier {
        UnifiedHighwayTier::Motorway => HighwayType::Motorway,
        UnifiedHighwayTier::Trunk => HighwayType::Trunk,
        UnifiedHighwayTier::Primary => HighwayType::Primary,
        UnifiedHighwayTier::Secondary => HighwayType::Secondary,
        UnifiedHighwayTier::Tertiary => HighwayType::Tertiary,
        UnifiedHighwayTier::Residential => HighwayType::Residential,
        UnifiedHighwayTier::Service => HighwayType::Service,
        UnifiedHighwayTier::LivingStreet => HighwayType::LivingStreet,
        UnifiedHighwayTier::Unclassified => HighwayType::Unclassified,
        UnifiedHighwayTier::Other => HighwayType::Track,
    }
}

pub fn compute_traffic_aware_path(
    graph: &RoadGraph,
    tensor: &TrafficTensorEngine,
    start: NodeId,
    goal: NodeId,
    start_time: &DateTime<Utc>,
) -> Option<Vec<(NodeId, usize)>> {
    #[derive(Copy, Clone, Eq, PartialEq)]
    struct State {
        cost: OrderedFloat<f32>,
        node: NodeId,
    }

    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            other.cost.cmp(&self.cost)
        }
    }
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut dist = ahash::AHashMap::new();
    let mut prev: ahash::AHashMap<NodeId, (NodeId, usize)> = ahash::AHashMap::new();
    let mut heap = BinaryHeap::new();

    dist.insert(start, OrderedFloat(0.0));
    heap.push(State {
        cost: OrderedFloat(0.0),
        node: start,
    });

    while let Some(State { cost, node }) = heap.pop() {
        if node == goal {
            let mut path = Vec::new();
            let mut curr = goal;
            while let Some(&(p_node, edge_idx)) = prev.get(&curr) {
                path.push((curr, edge_idx));
                curr = p_node;
            }
            path.reverse();
            return Some(path);
        }

        if cost > *dist.get(&node).unwrap_or(&OrderedFloat(f32::INFINITY)) {
            continue;
        }

        for &(next_node, edge_idx) in graph.outgoing_edges(node) {
            let traversal = tensor.evaluate_edge_at(edge_idx, start_time);
            let edge_duration = match traversal {
                TraversalStatus::Closed => f32::INFINITY,
                TraversalStatus::Open { duration_sec, jam_factor, .. } => {
                    duration_sec * (1.0 + (jam_factor / 10.0))
                }
            };

            if edge_duration.is_infinite() {
                continue;
            }

            let next_cost = cost + OrderedFloat(edge_duration);
            if next_cost < *dist.get(&next_node).unwrap_or(&OrderedFloat(f32::INFINITY)) {
                dist.insert(next_node, next_cost);
                prev.insert(next_node, (node, edge_idx));
                heap.push(State {
                    cost: next_cost,
                    node: next_node,
                });
            }
        }
    }

    None
}

/// Generates a realistic vehicle trip with physics, momentum, continuous ephemeris, and sensor randomness
pub fn generate_vehicle_trip<R: Rng + RngExt>(
    graph: &RoadGraph,
    tensor: &TrafficTensorEngine,
    sensors: &CameraSensorIndex,
    vehicle_id: &str,
    start_time: DateTime<Utc>,
    mode: TripMode,
    camera_dropout_rate: f64,
    rng: &mut R,
) -> Option<SimulatedTrip> {
    if graph.num_nodes == 0 || graph.num_edges == 0 {
        return None;
    }

    // 1. Assign driver profile
    let driver = if rng.random_bool(0.15) {
        DriverProfile::Aggressive
    } else if rng.random_bool(0.15) {
        DriverProfile::Cautious
    } else if rng.random_bool(0.20) {
        DriverProfile::DeliveryCourier
    } else {
        DriverProfile::Standard
    };

    // 2. Pick a well-connected arterial origin node
    let mut origin_id = 0u64;
    for _ in 0..500 {
        let edge_idx = rng.random_range(0..graph.num_edges);
        let desc = graph.edge_descriptor(edge_idx);
        let candidate = desc.u;
        let out = graph.outgoing_edges(candidate);
        if out.len() >= 2
            && matches!(
                desc.tier,
                UnifiedHighwayTier::Primary
                    | UnifiedHighwayTier::Secondary
                    | UnifiedHighwayTier::Tertiary
                    | UnifiedHighwayTier::Trunk
            )
        {
            origin_id = candidate;
            break;
        }
    }
    if origin_id == 0 {
        origin_id = graph.edge_descriptor(0).u;
    }

    let planned_path = match mode {
        TripMode::GoalDirected { destination } => {
            compute_traffic_aware_path(graph, tensor, origin_id, destination, &start_time)
        }
        TripMode::SmoothCruise { .. } => None,
    };

    let mut path_nodes = vec![origin_id];
    let mut ephemeris_full = Vec::with_capacity(4096);
    let mut sightings = Vec::new();
    let mut curr_node = origin_id;
    let mut curr_time = start_time;
    let mut total_distance_m = 0.0f64;
    let mut prev_bearing: Option<f32> = None;
    let mut vehicle_speed_kmh: f32 = rng.random_range(30.0..48.0);
    let mut planned_idx = 0;
    let mut had_micro_stops = false;

    let mut last_edge_from: Option<NodeId> = None;

    while path_nodes.len() < 50_000 {
        match mode {
            TripMode::GoalDirected { destination } => {
                if curr_node == destination && path_nodes.len() >= 2 {
                    break;
                }
            }
            TripMode::SmoothCruise { target_distance_m } => {
                if total_distance_m >= target_distance_m && path_nodes.len() >= 4 {
                    break;
                }
            }
        }

        let outgoing = graph.outgoing_edges(curr_node);
        if outgoing.is_empty() {
            if let TripMode::SmoothCruise { target_distance_m } = mode {
                if total_distance_m < target_distance_m {
                    if let Some(prev) = last_edge_from {
                        if let Some(rev_idx) = graph.get_edge_index(curr_node, prev) {
                            let desc = graph.edge_descriptor(rev_idx);
                            total_distance_m += desc.length_m as f64;
                            curr_node = prev;
                            path_nodes.push(prev);
                            last_edge_from = None;
                            prev_bearing = None;
                            continue;
                        }
                    }

                    let mut reconnected = false;
                    for _ in 0..100 {
                        let rand_e = rng.random_range(0..graph.num_edges);
                        let candidate = graph.edge_descriptor(rand_e).u;
                        if graph.outgoing_edges(candidate).len() >= 2 {
                            curr_node = candidate;
                            path_nodes.push(candidate);
                            last_edge_from = None;
                            prev_bearing = None;
                            reconnected = true;
                            break;
                        }
                    }
                    if reconnected {
                        continue;
                    }
                }
            }
            break;
        }

        let chosen_edge = if let Some(ref plan) = planned_path {
            if planned_idx < plan.len() {
                let step = plan[planned_idx];
                planned_idx += 1;
                step
            } else {
                break;
            }
        } else {
            let mut candidates = Vec::with_capacity(outgoing.len());

            for &(next_node, edge_idx) in outgoing {
                let is_immediate_u_turn = last_edge_from.map_or(false, |p| p == next_node);
                if is_immediate_u_turn && outgoing.len() > 1 {
                    continue;
                }

                let desc = graph.edge_descriptor(edge_idx);
                let edge_bearing = if desc.geometry.len() >= 2 {
                    initial_bearing(
                        desc.geometry[0][0],
                        desc.geometry[0][1],
                        desc.geometry[1][0],
                        desc.geometry[1][1],
                    ) as f32
                } else {
                    0.0
                };

                let deflection = if let Some(pb) = prev_bearing {
                    angle_diff(pb as f64, edge_bearing as f64) as f32
                } else {
                    0.0
                };

                let traversal = tensor.evaluate_edge_at(edge_idx, &curr_time);
                match traversal {
                    TraversalStatus::Closed => continue,
                    TraversalStatus::Open { speed_mps, jam_factor, .. } => {
                        let momentum_multiplier = if deflection <= 25.0 {
                            3.0
                        } else if deflection <= 65.0 {
                            1.5
                        } else if deflection <= 110.0 {
                            0.7
                        } else {
                            0.15
                        };

                        let tier_multiplier = match desc.tier {
                            UnifiedHighwayTier::Motorway => 3.2,
                            UnifiedHighwayTier::Trunk => 2.6,
                            UnifiedHighwayTier::Primary => 2.2,
                            UnifiedHighwayTier::Secondary => 1.5,
                            UnifiedHighwayTier::Tertiary => 1.0,
                            _ => 0.4,
                        };

                        let weight = speed_mps * (10.5 - jam_factor).max(1.0) * momentum_multiplier * tier_multiplier;
                        candidates.push((next_node, edge_idx, weight.max(0.01)));
                    }
                }
            }

            if candidates.is_empty() {
                let &(fallback_node, fallback_idx) = outgoing.choose(rng).unwrap();
                (fallback_node, fallback_idx)
            } else {
                let total_weight: f32 = candidates.iter().map(|c| c.2).sum();
                let mut pick = rng.random_range(0.0..total_weight);
                let mut chosen = (candidates[0].0, candidates[0].1);
                for c in &candidates {
                    if pick <= c.2 {
                        chosen = (c.0, c.1);
                        break;
                    }
                    pick -= c.2;
                }
                chosen
            }
        };

        let (next_node, edge_idx) = chosen_edge;
        let desc = graph.edge_descriptor(edge_idx);
        let highway_type = map_unified_tier(desc.tier);

        let bearing = if desc.geometry.len() >= 2 {
            initial_bearing(
                desc.geometry[0][0],
                desc.geometry[0][1],
                desc.geometry[1][0],
                desc.geometry[1][1],
            ) as f32
        } else {
            0.0
        };

        let status = tensor.evaluate_edge_at(edge_idx, &curr_time);
        let (flow_speed_kmh, jam_factor) = match status {
            TraversalStatus::Open { speed_mps, jam_factor, .. } => (speed_mps * 3.6, jam_factor),
            TraversalStatus::Closed => (15.0, 8.5),
        };

        let has_camera = sensors.get_camera(curr_node, next_node).is_some();

        // 4. Micro-step Kinematics with continuous spline breadcrumbs
        let step_res = simulate_edge_kinematics(
            desc.length_m,
            desc.maxspeed_kmh,
            highway_type,
            RoadSurface::Asphalt,
            &desc.geometry,
            prev_bearing,
            bearing,
            flow_speed_kmh,
            jam_factor,
            vehicle_speed_kmh,
            has_camera,
            driver,
            &curr_time,
            rng,
        );

        if step_res.experienced_micro_stop {
            had_micro_stops = true;
        }

        total_distance_m += desc.length_m as f64;
        curr_time = curr_time + Duration::milliseconds((step_res.transit_duration_secs * 1000.0) as i64);
        vehicle_speed_kmh = step_res.exit_speed_kmh;
        prev_bearing = Some(bearing);
        ephemeris_full.extend(step_res.ephemeris_trace);

        // 5. ANPR Sighting Registration with Physical Sensor Randomness
        if let Some(cam) = sensors.get_camera(curr_node, next_node) {
            let optical_blur_dropout = rng.random_bool(camera_dropout_rate);
            if !optical_blur_dropout {
                // SENSOR NOISE: NTP clock jitter (-150ms to +150ms)
                let ntp_jitter_ms = rng.random_range(-150..150);
                let capture_timestamp = curr_time + Duration::milliseconds(ntp_jitter_ms);

                // SENSOR NOISE: Doppler radar calibration noise (-2.0 km/h to +2.0 km/h)
                let radar_speed_noise = rng.random_range(-2.0..2.0);
                let measured_speed = (step_res.camera_capture_speed_kmh + radar_speed_noise).max(2.0);

                sightings.push(CameraSighting {
                    vehicle_id: vehicle_id.to_string(),
                    camera_id: cam.pole_id.clone(),
                    edge_from: curr_node,
                    edge_to: next_node,
                    timestamp: capture_timestamp,
                    speed_kmh: measured_speed,
                });
            }
        }

        // 6. Junction Delay
        let is_agg = driver == DriverProfile::Aggressive;
        apply_junction_delays(next_node, false, false, is_agg, &mut curr_time, rng);

        last_edge_from = Some(curr_node);
        path_nodes.push(next_node);
        curr_node = next_node;
    }

    if path_nodes.len() < 2 {
        return None;
    }

    let destination_id = *path_nodes.last().unwrap();

    Some(SimulatedTrip {
        vehicle_id: vehicle_id.to_string(),
        driver_profile: driver,
        origin: origin_id,
        destination: destination_id,
        ground_truth_nodes: path_nodes,
        ground_truth_ephemeris: ephemeris_full,
        camera_sightings: sightings,
        start_time,
        end_time: curr_time,
        total_distance_m,
        had_micro_stops,
    })
}