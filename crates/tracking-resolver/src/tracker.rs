use crate::anomaly::{evaluate_leg_anomaly, AnomalyAlert};
use crate::causal_gap::{resolve_camera_gap, CausalGapWorkspace, TrajectoryCandidate};
use ahash::AHashMap;
use chrono::{DateTime, Duration, Utc};
use road_graph_loader::RoadGraph;
use road_network_core::geo::fast_planar_distance;
use road_network_core::{CameraSighting, NodeId};
use road_traffic_tensor::{TrafficTensorEngine, TraversalStatus};

#[derive(Debug, Clone)]
pub struct ResolvedLeg {
    pub from_camera: String,
    pub to_camera: String,
    pub dt_seconds: f32,
    pub candidates: Vec<TrajectoryCandidate>,
    pub anomalies: Vec<AnomalyAlert>,
}

#[derive(Debug, Clone)]
pub struct TrackHypothesis {
    pub nodes: Vec<NodeId>,
    pub total_log_likelihood: f32,
    pub total_distance_m: f64,
}

#[derive(Debug, Clone)]
pub struct VehicleTrack {
    pub vehicle_id: String,
    pub first_sighting: CameraSighting,
    pub last_sighting: CameraSighting,
    pub legs: Vec<ResolvedLeg>,
    pub beam_hypotheses: Vec<TrackHypothesis>,
    pub all_anomalies: Vec<AnomalyAlert>,
}

impl VehicleTrack {
    /// Returns the globally optimal reconstructed node path (top hypothesis from the Viterbi beam)
    #[inline]
    pub fn best_reconstructed_path(&self) -> &[NodeId] {
        self.beam_hypotheses
            .first()
            .map(|h| h.nodes.as_slice())
            .unwrap_or(&[])
    }

    #[inline]
    pub fn total_distance_m(&self) -> f64 {
        self.beam_hypotheses
            .first()
            .map(|h| h.total_distance_m)
            .unwrap_or(0.0)
    }

    /// Reconstructs continuous, timestamped kinematic ephemeris points (lat, lon, timestamp, speed_kmh)
    /// across the entire observed camera window using RoadGraph geometry and tensor flow baselines.
    pub fn reconstruct_continuous_ephemeris(
        &self,
        graph: &RoadGraph,
        tensor: &TrafficTensorEngine,
    ) -> Vec<(f64, f64, DateTime<Utc>, f32)> {
        let best_nodes = self.best_reconstructed_path();
        if best_nodes.len() < 2 {
            return Vec::new();
        }

        // 1. Resolve physical edge indices along the path
        let mut edges_along_path = Vec::with_capacity(best_nodes.len() - 1);
        for pair in best_nodes.windows(2) {
            let u = pair[0];
            let v = pair[1];
            if let Some(edge_idx) = graph.get_edge_index(u, v) {
                edges_along_path.push((u, v, edge_idx));
            }
        }

        if edges_along_path.is_empty() {
            return Vec::new();
        }

        // 2. Compute expected traversal times across edges using dynamic tensor speeds
        let mut expected_durations = Vec::with_capacity(edges_along_path.len());
        let mut total_expected_sec = 0.0f32;
        let start_time = self.first_sighting.timestamp;

        for &(_, _, edge_idx) in &edges_along_path {
            let desc = graph.edge_descriptor(edge_idx);
            let status = tensor.evaluate_edge_at(edge_idx, &start_time);
            let speed_mps = match status {
                TraversalStatus::Open { speed_mps, .. } => speed_mps.max(1.5),
                TraversalStatus::Closed => 2.0,
            };
            let dur = desc.length_m / speed_mps;
            expected_durations.push(dur);
            total_expected_sec += dur;
        }

        let total_observed_sec = (self.last_sighting.timestamp - self.first_sighting.timestamp)
            .num_milliseconds() as f32
            / 1000.0;
        let scale_factor = if total_expected_sec > 0.0 {
            (total_observed_sec / total_expected_sec).max(0.05)
        } else {
            1.0
        };

        // 3. Generate continuous spline breadcrumbs every ~15m
        let mut ephemeris = Vec::with_capacity(edges_along_path.len() * 8);
        let mut curr_edge_start_time = self.first_sighting.timestamp;

        for (i, &(_, _, edge_idx)) in edges_along_path.iter().enumerate() {
            let desc = graph.edge_descriptor(edge_idx);
            let edge_duration_sec = (expected_durations[i] * scale_factor).max(0.1);
            let edge_speed_kmh = (desc.length_m / edge_duration_sec) * 3.6;

            if desc.geometry.len() >= 2 {
                // Compute total polyline length from coordinates
                let mut seg_lengths = Vec::with_capacity(desc.geometry.len() - 1);
                let mut poly_total_len = 0.0;

                for w in desc.geometry.windows(2) {
                    let d = fast_planar_distance(w[0][0], w[0][1], w[1][0], w[1][1]);
                    seg_lengths.push(d);
                    poly_total_len += d;
                }

                if poly_total_len <= 0.0 {
                    poly_total_len = desc.length_m as f64;
                }

                const SLICE_STEP_M: f64 = 15.0;
                let num_slices = ((poly_total_len / SLICE_STEP_M).ceil() as usize).max(1);

                for s in 0..num_slices {
                    let fraction = s as f64 / num_slices as f64;
                    let target_dist = fraction * poly_total_len;

                    // Locate position along polyline segments
                    let mut accum = 0.0;
                    let mut pt_lat = desc.geometry[0][0];
                    let mut pt_lon = desc.geometry[0][1];

                    for (seg_idx, &seg_len) in seg_lengths.iter().enumerate() {
                        if accum + seg_len >= target_dist || seg_idx == seg_lengths.len() - 1 {
                            let remain = (target_dist - accum).max(0.0);
                            let sub_frac = if seg_len > 0.0 { remain / seg_len } else { 0.0 };
                            let p0 = desc.geometry[seg_idx];
                            let p1 = desc.geometry[seg_idx + 1];
                            pt_lat = p0[0] + (p1[0] - p0[0]) * sub_frac;
                            pt_lon = p0[1] + (p1[1] - p0[1]) * sub_frac;
                            break;
                        }
                        accum += seg_len;
                    }

                    let slice_offset_sec = edge_duration_sec * (fraction as f32);
                    let pt_time = curr_edge_start_time
                        + Duration::milliseconds((slice_offset_sec * 1000.0) as i64);

                    ephemeris.push((pt_lat, pt_lon, pt_time, edge_speed_kmh));
                }
            } else if !desc.geometry.is_empty() {
                let p = desc.geometry[0];
                ephemeris.push((p[0], p[1], curr_edge_start_time, edge_speed_kmh));
            }

            curr_edge_start_time = curr_edge_start_time
                + Duration::milliseconds((edge_duration_sec * 1000.0) as i64);
        }

        // Add terminal destination point
        if let Some(&(_, _, last_edge_idx)) = edges_along_path.last() {
            let last_desc = graph.edge_descriptor(last_edge_idx);
            if let Some(last_pt) = last_desc.geometry.last() {
                let last_speed = (last_desc.length_m
                    / (expected_durations.last().copied().unwrap_or(1.0) * scale_factor).max(0.1))
                    * 3.6;
                ephemeris.push((last_pt[0], last_pt[1], self.last_sighting.timestamp, last_speed));
            }
        }

        ephemeris
    }
}

pub struct StreamingCausalTracker<'a> {
    graph: &'a RoadGraph,
    tensor: &'a TrafficTensorEngine<'a>,
    tracks: AHashMap<String, VehicleTrack>,
    workspace: CausalGapWorkspace,
    beam_width: usize,
}

impl<'a> StreamingCausalTracker<'a> {
    pub fn new(graph: &'a RoadGraph, tensor: &'a TrafficTensorEngine<'a>) -> Self {
        Self {
            graph,
            tensor,
            tracks: AHashMap::new(),
            workspace: CausalGapWorkspace::new(),
            beam_width: 4,
        }
    }

    pub fn with_beam_width(mut self, width: usize) -> Self {
        self.beam_width = width.max(1);
        self
    }

    /// Ingests an ANPR sighting in real-time under the Blind Observer Law
    pub fn ingest_sighting(&mut self, sighting: CameraSighting) {
        let vid = sighting.vehicle_id.clone();

        if let Some(track) = self.tracks.get_mut(&vid) {
            let dt = (sighting.timestamp - track.last_sighting.timestamp).num_milliseconds()
                as f32
                / 1000.0;
            let gap_start = track.last_sighting.edge_to;
            let gap_end = sighting.edge_from;

            let candidates = resolve_camera_gap(
                self.graph,
                self.tensor,
                gap_start,
                gap_end,
                &track.last_sighting.timestamp,
                dt,
                self.beam_width,
                &mut self.workspace,
            );

            let leg_index = track.legs.len();
            let mut leg_anomalies = Vec::new();

            if let Some(best) = candidates.first() {
                if let Some(alert) = evaluate_leg_anomaly(
                    &track.last_sighting.camera_id,
                    &sighting.camera_id,
                    leg_index,
                    dt,
                    best.min_possible_seconds,
                    best.expected_tensor_seconds,
                    best.distance_m,
                    best.dark_alley_ratio,
                ) {
                    leg_anomalies.push(alert.clone());
                    track.all_anomalies.push(alert);
                }
            }

            let edge_speed_est = if let Some(edge_idx) =
                self.graph.get_edge_index(sighting.edge_from, sighting.edge_to)
            {
                let desc = self.graph.edge_descriptor(edge_idx);
                desc.maxspeed_kmh as f32
            } else {
                35.0
            };
            let speed_diff = (sighting.speed_kmh - edge_speed_est).abs();
            let emission_log_l = -0.5 * (speed_diff / 8.0).powi(2);

            let mut next_beam: Vec<TrackHypothesis> =
                Vec::with_capacity(track.beam_hypotheses.len() * candidates.len().max(1));

            if candidates.is_empty() {
                for hyp in &track.beam_hypotheses {
                    let mut extended = hyp.nodes.clone();
                    if extended.last() != Some(&sighting.edge_from) {
                        extended.push(sighting.edge_from);
                    }
                    if extended.last() != Some(&sighting.edge_to) {
                        extended.push(sighting.edge_to);
                    }
                    next_beam.push(TrackHypothesis {
                        nodes: extended,
                        total_log_likelihood: hyp.total_log_likelihood - 30.0,
                        total_distance_m: hyp.total_distance_m,
                    });
                }
            } else {
                for hyp in &track.beam_hypotheses {
                    for cand in &candidates {
                        let mut extended_nodes = hyp.nodes.clone();
                        for &n in &cand.path {
                            if extended_nodes.last() != Some(&n) {
                                extended_nodes.push(n);
                            }
                        }
                        if extended_nodes.last() != Some(&sighting.edge_from) {
                            extended_nodes.push(sighting.edge_from);
                        }
                        if extended_nodes.last() != Some(&sighting.edge_to) {
                            extended_nodes.push(sighting.edge_to);
                        }

                        next_beam.push(TrackHypothesis {
                            nodes: extended_nodes,
                            total_log_likelihood: hyp.total_log_likelihood
                                + cand.transition_log_likelihood
                                + emission_log_l,
                            total_distance_m: hyp.total_distance_m + cand.distance_m as f64,
                        });
                    }
                }
            }

            next_beam.sort_by(|a, b| {
                b.total_log_likelihood
                    .partial_cmp(&a.total_log_likelihood)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            next_beam.truncate(self.beam_width);

            track.beam_hypotheses = next_beam;
            track.legs.push(ResolvedLeg {
                from_camera: track.last_sighting.camera_id.clone(),
                to_camera: sighting.camera_id.clone(),
                dt_seconds: dt,
                candidates,
                anomalies: leg_anomalies,
            });

            track.last_sighting = sighting;
        } else {
            let init_nodes = vec![sighting.edge_from, sighting.edge_to];
            self.tracks.insert(
                vid.clone(),
                VehicleTrack {
                    vehicle_id: vid,
                    first_sighting: sighting.clone(),
                    last_sighting: sighting,
                    legs: Vec::new(),
                    beam_hypotheses: vec![TrackHypothesis {
                        nodes: init_nodes,
                        total_log_likelihood: 0.0,
                        total_distance_m: 0.0,
                    }],
                    all_anomalies: Vec::new(),
                },
            );
        }
    }

    #[inline(always)]
    pub fn get_track(&self, vehicle_id: &str) -> Option<&VehicleTrack> {
        self.tracks.get(vehicle_id)
    }

    #[inline(always)]
    pub fn active_tracks_count(&self) -> usize {
        self.tracks.len()
    }
}