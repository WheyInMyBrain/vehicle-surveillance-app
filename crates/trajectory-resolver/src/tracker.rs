use crate::astar::{resolve_edge_gap_probabilistic, TrajectoryCandidate};
use engine_core::{CameraSighting, CityGraph, NodeId};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct LegHypothesis {
    pub from_camera: String,
    pub to_camera: String,
    pub alternatives: Vec<TrajectoryCandidate>,
}

pub struct VehicleTrackState {
    pub vehicle_id: String,
    pub last_sighting: CameraSighting,
    pub primary_route: Vec<NodeId>,
    /// Stores alternative corridors per unmonitored blind zone
    pub leg_hypotheses: Vec<LegHypothesis>,
}

pub struct StreamingTracker<'a> {
    city: &'a CityGraph,
    vehicles: HashMap<String, VehicleTrackState>,
    max_alternatives_per_leg: usize,
}

impl<'a> StreamingTracker<'a> {
    pub fn new(city: &'a CityGraph) -> Self {
        Self {
            city,
            vehicles: HashMap::new(),
            max_alternatives_per_leg: 5, // Allows up to 5 viable candidates
        }
    }

    pub fn with_candidate_limit(mut self, limit: usize) -> Self {
        self.max_alternatives_per_leg = limit;
        self
    }

    pub fn ingest_sighting(&mut self, sighting: CameraSighting) {
        let vid = sighting.vehicle_id.clone();

        if let Some(state) = self.vehicles.get_mut(&vid) {
            let dt = (sighting.timestamp - state.last_sighting.timestamp).num_seconds() as f64;
            
            let candidates = resolve_edge_gap_probabilistic(
                self.city,
                state.last_sighting.edge_to,
                sighting.edge_from,
                dt,
                self.max_alternatives_per_leg,
            );

            // Primary choice is index 0
            if let Some(best) = candidates.first() {
                for &node in &best.path {
                    if state.primary_route.last() != Some(&node) {
                        state.primary_route.push(node);
                    }
                }
                if state.primary_route.last() != Some(&sighting.edge_to) {
                    state.primary_route.push(sighting.edge_to);
                }
            }

            // If there was uncertainty (multiple corridors), record all alternatives
            if candidates.len() > 1 {
                state.leg_hypotheses.push(LegHypothesis {
                    from_camera: state.last_sighting.camera_id.clone(),
                    to_camera: sighting.camera_id.clone(),
                    alternatives: candidates,
                });
            }

            state.last_sighting = sighting;
        } else {
            self.vehicles.insert(
                vid.clone(),
                VehicleTrackState {
                    vehicle_id: vid,
                    primary_route: vec![sighting.edge_from, sighting.edge_to],
                    leg_hypotheses: Vec::new(),
                    last_sighting: sighting,
                },
            );
        }
    }

    pub fn get_track(&self, vehicle_id: &str) -> Option<&VehicleTrackState> {
        self.vehicles.get(vehicle_id)
    }
}