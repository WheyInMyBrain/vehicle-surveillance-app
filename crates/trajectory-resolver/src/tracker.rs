use crate::astar::{resolve_edge_gap_probabilistic, TrajectoryCandidate};
use ahash::AHashMap;
use engine_core::{CameraSighting, CityGraph, NodeId};

#[derive(Debug, Clone)]
pub struct LegHypothesis {
    pub from_camera: String,
    pub to_camera: String,
    pub dt_seconds: f64,
    pub alternatives: Vec<TrajectoryCandidate>,
}

#[derive(Debug, Clone)]
pub struct VehicleTrackState {
    pub vehicle_id: String,
    pub last_sighting: CameraSighting,
    /// The reconstructed ground-truth sequence of node IDs traversed by the car
    pub reconstructed_path: Vec<NodeId>,
    pub leg_hypotheses: Vec<LegHypothesis>,
    pub total_resolved_distance_m: f64,
}

pub struct StreamingTracker<'a> {
    city: &'a CityGraph,
    vehicles: AHashMap<String, VehicleTrackState>,
    max_alternatives_per_leg: usize,
}

impl<'a> StreamingTracker<'a> {
    pub fn new(city: &'a CityGraph) -> Self {
        Self {
            city,
            vehicles: AHashMap::new(),
            max_alternatives_per_leg: 4,
        }
    }

    pub fn with_candidate_limit(mut self, limit: usize) -> Self {
        self.max_alternatives_per_leg = limit;
        self
    }

    pub fn ingest_sighting(&mut self, sighting: CameraSighting) {
        let vid = sighting.vehicle_id.clone();

        if let Some(state) = self.vehicles.get_mut(&vid) {
            let dt = (sighting.timestamp - state.last_sighting.timestamp).num_milliseconds() as f64
                / 1000.0;
            let gap_start = state.last_sighting.edge_to;
            let gap_end = sighting.edge_from;

            let candidates = resolve_edge_gap_probabilistic(
                self.city,
                gap_start,
                gap_end,
                dt,
                state.last_sighting.speed_kmh,
                sighting.speed_kmh,
                self.max_alternatives_per_leg,
            );

            if let Some(best) = candidates.first() {
                for &node in &best.path {
                    if state.reconstructed_path.last() != Some(&node) {
                        state.reconstructed_path.push(node);
                    }
                }
                state.total_resolved_distance_m += best.distance_m;
            }

            // Append camera observation edge
            if state.reconstructed_path.last() != Some(&sighting.edge_from) {
                state.reconstructed_path.push(sighting.edge_from);
            }
            if state.reconstructed_path.last() != Some(&sighting.edge_to) {
                state.reconstructed_path.push(sighting.edge_to);
            }

            state.leg_hypotheses.push(LegHypothesis {
                from_camera: state.last_sighting.camera_id.clone(),
                to_camera: sighting.camera_id.clone(),
                dt_seconds: dt,
                alternatives: candidates,
            });

            state.last_sighting = sighting;
        } else {
            // First time seeing this vehicle
            self.vehicles.insert(
                vid.clone(),
                VehicleTrackState {
                    vehicle_id: vid,
                    reconstructed_path: vec![sighting.edge_from, sighting.edge_to],
                    leg_hypotheses: Vec::new(),
                    last_sighting: sighting,
                    total_resolved_distance_m: 0.0,
                },
            );
        }
    }

    pub fn get_track(&self, vehicle_id: &str) -> Option<&VehicleTrackState> {
        self.vehicles.get(vehicle_id)
    }

    pub fn tracked_vehicle_count(&self) -> usize {
        self.vehicles.len()
    }
}