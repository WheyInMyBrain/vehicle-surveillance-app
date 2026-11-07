//! Multi-Object ByteTrack Association Engine.
//!
//! # Safety & Determinism
//! - Fixed capacity stack arrays (JPL Rule 3).
//! - Zero runtime heap reallocations.
//! - Bounded loop executions (JPL Rule 1).

use crate::core::kalman::KalmanFilter;
use crate::core::lapjv::{lapjv_bounded, LapMatchResult, MAX_MATCH_DIM};
use crate::core::types::{TrackState, TrackedVehicle};
use crate::track::STrack;
use heapless::Vec as StackVec;
use vehicle_detector::{BoundingBox, DetectedVehicle};

pub const MAX_TRACKS: usize = 64;
pub const MAX_DETECTIONS: usize = 32;

#[derive(Debug, Clone, Copy)]
pub struct ByteTrackConfig {
    pub track_thresh: f32,
    pub high_thresh: f32,
    pub match_thresh: f32,
    pub max_time_lost: usize,
    pub min_hits: usize,
    pub class_mismatch_penalty: f32,
    pub motion_weight: f32,
}

impl Default for ByteTrackConfig {
    fn default() -> Self {
        Self {
            track_thresh: 0.25,
            high_thresh: 0.45,
            match_thresh: 0.75,
            max_time_lost: 40,
            min_hits: 2,
            class_mismatch_penalty: 0.50,
            motion_weight: 0.15,
        }
    }
}

pub struct ByteTracker {
    config: ByteTrackConfig,
    frame_id: usize,
    tracked_stracks: StackVec<STrack, MAX_TRACKS>,
    lost_stracks: StackVec<STrack, MAX_TRACKS>,
    unconfirmed_stracks: StackVec<STrack, MAX_TRACKS>,
    kf: KalmanFilter,

    // Pre-allocated static scratch buffers
    det_high_buf: StackVec<STrack, MAX_DETECTIONS>,
    det_low_buf: StackVec<STrack, MAX_DETECTIONS>,
    pool_tracks_buf: StackVec<STrack, MAX_TRACKS>,
    remaining_tracked_buf: StackVec<STrack, MAX_TRACKS>,
    remaining_dets_buf: StackVec<STrack, MAX_DETECTIONS>,
    cost_matrix_buf: [f32; MAX_MATCH_DIM * MAX_MATCH_DIM],
}

impl Default for ByteTracker {
    fn default() -> Self {
        Self::new(ByteTrackConfig::default())
    }
}

impl ByteTracker {
    pub fn new(config: ByteTrackConfig) -> Self {
        Self {
            config,
            frame_id: 0,
            tracked_stracks: StackVec::new(),
            lost_stracks: StackVec::new(),
            unconfirmed_stracks: StackVec::new(),
            kf: KalmanFilter::new(),

            det_high_buf: StackVec::new(),
            det_low_buf: StackVec::new(),
            pool_tracks_buf: StackVec::new(),
            remaining_tracked_buf: StackVec::new(),
            remaining_dets_buf: StackVec::new(),
            cost_matrix_buf: [0.0f32; MAX_MATCH_DIM * MAX_MATCH_DIM],
        }
    }

    /// Primary tracking update method.
    ///
    /// # Memory Guarantee
    /// Executes with **zero** heap allocations and returns bounded stack vector.
    pub fn update(
        &mut self,
        detections: &[DetectedVehicle],
    ) -> StackVec<TrackedVehicle, MAX_TRACKS> {
        self.frame_id += 1;

        // 1. Predict state for all active, unconfirmed, and lost tracks
        for track in &mut self.tracked_stracks {
            track.predict(&self.kf);
        }
        for track in &mut self.unconfirmed_stracks {
            track.predict(&self.kf);
        }
        for track in &mut self.lost_stracks {
            track.predict(&self.kf);
        }

        // 2. Partition detections into high and low confidence pools
        self.det_high_buf.clear();
        self.det_low_buf.clear();

        for d in detections {
            let strack = STrack::new(d, &self.kf);
            if d.confidence >= self.config.high_thresh {
                let _ = self.det_high_buf.push(strack);
            } else if d.confidence >= self.config.track_thresh {
                let _ = self.det_low_buf.push(strack);
            }
        }

        // 3. First association: (Confirmed + Lost tracks) vs high-conf detections
        self.pool_tracks_buf.clear();
        for t in &self.tracked_stracks {
            let _ = self.pool_tracks_buf.push(*t);
        }
        for t in &self.lost_stracks {
            let _ = self.pool_tracks_buf.push(*t);
        }
        self.tracked_stracks.clear();
        self.lost_stracks.clear();

        let match_res_a = associate_internal(
            &self.pool_tracks_buf,
            &self.det_high_buf,
            self.config.match_thresh,
            self.config.class_mismatch_penalty,
            self.config.motion_weight,
            &mut self.cost_matrix_buf,
        );

        for &(t_idx, d_idx) in &match_res_a.matches {
            if let (Some(mut track), Some(det)) = (
                self.pool_tracks_buf.get(t_idx).copied(),
                self.det_high_buf.get(d_idx),
            ) {
                if track.state == TrackState::Tracked {
                    track.update(det, self.frame_id, &self.kf);
                } else {
                    track.re_activate(det, self.frame_id, &self.kf);
                }
                let _ = self.tracked_stracks.push(track);
            }
        }

        // 4. Second association: Unmatched tracks vs low-conf detections
        self.remaining_tracked_buf.clear();
        for &idx in &match_res_a.unmatched_rows {
            if let Some(t) = self.pool_tracks_buf.get(idx) {
                let _ = self.remaining_tracked_buf.push(*t);
            }
        }

        let match_res_b = associate_internal(
            &self.remaining_tracked_buf,
            &self.det_low_buf,
            0.60,
            self.config.class_mismatch_penalty,
            self.config.motion_weight,
            &mut self.cost_matrix_buf,
        );

        for &(t_idx, d_idx) in &match_res_b.matches {
            if let (Some(mut track), Some(det)) = (
                self.remaining_tracked_buf.get(t_idx).copied(),
                self.det_low_buf.get(d_idx),
            ) {
                track.update(det, self.frame_id, &self.kf);
                let _ = self.tracked_stracks.push(track);
            }
        }

        for &t_idx in &match_res_b.unmatched_rows {
            if let Some(mut track) = self.remaining_tracked_buf.get(t_idx).copied() {
                if track.state != TrackState::Lost {
                    track.mark_lost();
                }
                let _ = self.lost_stracks.push(track);
            }
        }

        // 5. Third association: Unconfirmed tracks vs remaining high-conf detections
        self.remaining_dets_buf.clear();
        for &idx in &match_res_a.unmatched_cols {
            if let Some(d) = self.det_high_buf.get(idx) {
                let _ = self.remaining_dets_buf.push(*d);
            }
        }

        // Snapshot current unconfirmed pool
        let mut unconfirmed_snapshot: StackVec<STrack, MAX_TRACKS> = StackVec::new();
        for t in &self.unconfirmed_stracks {
            let _ = unconfirmed_snapshot.push(*t);
        }
        self.unconfirmed_stracks.clear();

        let match_res_c = associate_internal(
            &unconfirmed_snapshot,
            &self.remaining_dets_buf,
            0.70,
            self.config.class_mismatch_penalty,
            self.config.motion_weight,
            &mut self.cost_matrix_buf,
        );

        for &(t_idx, d_idx) in &match_res_c.matches {
            if let (Some(mut track), Some(det)) = (
                unconfirmed_snapshot.get(t_idx).copied(),
                self.remaining_dets_buf.get(d_idx),
            ) {
                track.update(det, self.frame_id, &self.kf);
                if track.tracklet_len >= self.config.min_hits {
                    track.activate(self.frame_id);
                    let _ = self.tracked_stracks.push(track);
                } else {
                    let _ = self.unconfirmed_stracks.push(track);
                }
            }
        }

        // 6. Initialize new tracks from remaining detections
        for &d_idx in &match_res_c.unmatched_cols {
            if let Some(mut track) = self.remaining_dets_buf.get(d_idx).copied() {
                if self.config.min_hits <= 1 {
                    track.activate(self.frame_id);
                    let _ = self.tracked_stracks.push(track);
                } else {
                    let _ = self.unconfirmed_stracks.push(track);
                }
            }
        }

        // 7. Prune expired lost tracks (in-place stack retain)
        let max_time_lost = self.config.max_time_lost;
        let cur_frame = self.frame_id;
        let mut i = 0;
        while i < self.lost_stracks.len() {
            let expired = {
                let t = &self.lost_stracks[i];
                cur_frame.saturating_sub(t.age - t.time_since_update) > max_time_lost
            };
            if expired {
                let _ = self.lost_stracks.swap_remove(i);
            } else {
                i += 1;
            }
        }

        // 8. Output confirmed and active tracks
        let mut outputs: StackVec<TrackedVehicle, MAX_TRACKS> = StackVec::new();
        for t in &self.tracked_stracks {
            if t.is_activated && t.state == TrackState::Tracked {
                let _ = outputs.push(TrackedVehicle {
                    track_id: t.track_id,
                    class: t.class,
                    confidence: t.confidence,
                    bbox: t.to_bbox(),
                    tracklet_len: t.tracklet_len,
                    age: t.age,
                });
            }
        }

        outputs
    }
}

/// Helper function to run cost calculation and bounded assignment with zero allocations.
fn associate_internal(
    tracks: &[STrack],
    dets: &[STrack],
    max_dist: f32,
    class_mismatch_penalty: f32,
    motion_weight: f32,
    cost_matrix: &mut [f32; MAX_MATCH_DIM * MAX_MATCH_DIM],
) -> LapMatchResult {
    let n_t = tracks.len().min(MAX_MATCH_DIM);
    let n_d = dets.len().min(MAX_MATCH_DIM);

    if n_t == 0 || n_d == 0 {
        return lapjv_bounded(cost_matrix, n_t, n_d, max_dist);
    }

    for (i, t) in tracks.iter().take(n_t).enumerate() {
        let b_t = t.to_bbox();
        let (vx, vy) = t.velocity();
        let speed = (vx * vx + vy * vy).sqrt();
        let (cx_t, cy_t) = b_t.center();

        for (j, d) in dets.iter().take(n_d).enumerate() {
            let b_d = d.to_bbox();
            let diou = calculate_diou(&b_t, &b_d);
            let mut cost = 1.0 - diou;

            // Class mismatch penalty
            if t.class != d.class {
                cost += class_mismatch_penalty;
            }

            // Directional cosine motion penalty
            if speed > 2.0 {
                let (cx_d, cy_d) = b_d.center();
                let dx = cx_d - cx_t;
                let dy = cy_d - cy_t;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist > 1.0 {
                    let dot = (vx * dx + vy * dy) / (speed * dist);
                    if dot < 0.0 {
                        cost += motion_weight * (-dot);
                    }
                }
            }

            cost_matrix[i * n_d + j] = cost;
        }
    }

    lapjv_bounded(cost_matrix, n_t, n_d, max_dist)
}

/// Complete Distance-IoU (DIoU) metric with singularity protection.
#[inline(always)]
pub fn calculate_diou(box_a: &BoundingBox, box_b: &BoundingBox) -> f32 {
    let x1 = if box_a.x1 > box_b.x1 { box_a.x1 } else { box_b.x1 };
    let y1 = if box_a.y1 > box_b.y1 { box_a.y1 } else { box_b.y1 };
    let x2 = if box_a.x2 < box_b.x2 { box_a.x2 } else { box_b.x2 };
    let y2 = if box_a.y2 < box_b.y2 { box_a.y2 } else { box_b.y2 };

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = box_a.area();
    let area_b = box_b.area();
    let union_area = area_a + area_b - inter_area;

    if union_area <= 1e-6 {
        return -1.0;
    }

    let iou = inter_area / union_area;

    let (cx_a, cy_a) = box_a.center();
    let (cx_b, cy_b) = box_b.center();
    let center_dist_sq = (cx_a - cx_b) * (cx_a - cx_b) + (cy_a - cy_b) * (cy_a - cy_b);

    let enc_x1 = if box_a.x1 < box_b.x1 { box_a.x1 } else { box_b.x1 };
    let enc_y1 = if box_a.y1 < box_b.y1 { box_a.y1 } else { box_b.y1 };
    let enc_x2 = if box_a.x2 > box_b.x2 { box_a.x2 } else { box_b.x2 };
    let enc_y2 = if box_a.y2 > box_b.y2 { box_a.y2 } else { box_b.y2 };

    let enc_w = enc_x2 - enc_x1;
    let enc_h = enc_y2 - enc_y1;
    let enc_diag_sq = (enc_w * enc_w + enc_h * enc_h).max(1e-5);

    iou - (center_dist_sq / enc_diag_sq)
}