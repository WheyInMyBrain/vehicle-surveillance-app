use crate::kalman::KalmanFilter;
use crate::lapjv::lapjv;
use crate::track::STrack;
use crate::types::{TrackState, TrackedVehicle};
use vehicle_detector::DetectedVehicle;

#[derive(Debug, Clone)]
pub struct ByteTrackConfig {
    pub track_thresh: f32,
    pub high_thresh: f32,
    pub match_thresh: f32,
    pub max_time_lost: usize,
    pub min_hits: usize,
    /// Cost penalty added when detector class disagrees with active track class
    pub class_mismatch_penalty: f32,
    /// Directional cosine motion penalty weight
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
    tracked_stracks: Vec<STrack>,
    lost_stracks: Vec<STrack>,
    unconfirmed_stracks: Vec<STrack>,
    removed_stracks: Vec<STrack>,
    kf: KalmanFilter,

    // Pre-allocated scratch buffers to guarantee 0 heap reallocations per frame
    det_high_buf: Vec<STrack>,
    det_low_buf: Vec<STrack>,
    pool_tracks_buf: Vec<STrack>,
    remaining_tracked_buf: Vec<STrack>,
    remaining_dets_buf: Vec<STrack>,
    cost_matrix_buf: Vec<f32>,
}

impl ByteTracker {
    pub fn new(config: ByteTrackConfig) -> Self {
        Self {
            config,
            frame_id: 0,
            tracked_stracks: Vec::with_capacity(64),
            lost_stracks: Vec::with_capacity(64),
            unconfirmed_stracks: Vec::with_capacity(32),
            removed_stracks: Vec::with_capacity(64),
            kf: KalmanFilter::new(),

            det_high_buf: Vec::with_capacity(64),
            det_low_buf: Vec::with_capacity(64),
            pool_tracks_buf: Vec::with_capacity(128),
            remaining_tracked_buf: Vec::with_capacity(64),
            remaining_dets_buf: Vec::with_capacity(64),
            cost_matrix_buf: Vec::with_capacity(2048),
        }
    }

    pub fn update(&mut self, detections: &[DetectedVehicle]) -> Vec<TrackedVehicle> {
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
                self.det_high_buf.push(strack);
            } else if d.confidence >= self.config.track_thresh {
                self.det_low_buf.push(strack);
            }
        }

        // 3. First association: Confirmed tracked pool vs high-conf detections
        self.pool_tracks_buf.clear();
        self.pool_tracks_buf.extend(self.tracked_stracks.drain(..));
        self.pool_tracks_buf.extend(self.lost_stracks.drain(..));

        let (matched_a, unmatched_tracks_a, unmatched_dets_a) = associate_internal(
            &self.pool_tracks_buf,
            &self.det_high_buf,
            self.config.match_thresh,
            self.config.class_mismatch_penalty,
            self.config.motion_weight,
            &mut self.cost_matrix_buf,
        );

        for (t_idx, d_idx) in matched_a {
            let mut track = self.pool_tracks_buf[t_idx].clone();
            let det = &self.det_high_buf[d_idx];
            if track.state == TrackState::Tracked {
                track.update(det, self.frame_id, &self.kf);
                self.tracked_stracks.push(track);
            } else {
                track.re_activate(det, self.frame_id, &self.kf);
                self.tracked_stracks.push(track);
            }
        }

        // 4. Second association: Unmatched tracked pool vs low-conf detections
        self.remaining_tracked_buf.clear();
        for &idx in &unmatched_tracks_a {
            self.remaining_tracked_buf.push(self.pool_tracks_buf[idx].clone());
        }

        let (matched_b, unmatched_tracks_b, _) = associate_internal(
            &self.remaining_tracked_buf,
            &self.det_low_buf,
            0.60,
            self.config.class_mismatch_penalty,
            self.config.motion_weight,
            &mut self.cost_matrix_buf,
        );

        for (t_idx, d_idx) in matched_b {
            let mut track = self.remaining_tracked_buf[t_idx].clone();
            track.update(&self.det_low_buf[d_idx], self.frame_id, &self.kf);
            self.tracked_stracks.push(track);
        }

        for &t_idx in &unmatched_tracks_b {
            let mut track = self.remaining_tracked_buf[t_idx].clone();
            if track.state != TrackState::Lost {
                track.mark_lost();
            }
            self.lost_stracks.push(track);
        }

        // 5. Third association: Unconfirmed tracks vs remaining high-conf detections
        self.remaining_dets_buf.clear();
        for &idx in &unmatched_dets_a {
            self.remaining_dets_buf.push(self.det_high_buf[idx].clone());
        }

        let unconfirmed = self.unconfirmed_stracks.drain(..).collect::<Vec<_>>();
        let (matched_c, unmatched_unconfirmed, unmatched_final_dets) = associate_internal(
            &unconfirmed,
            &self.remaining_dets_buf,
            0.70,
            self.config.class_mismatch_penalty,
            self.config.motion_weight,
            &mut self.cost_matrix_buf,
        );

        for (t_idx, d_idx) in matched_c {
            let mut track = unconfirmed[t_idx].clone();
            track.update(&self.remaining_dets_buf[d_idx], self.frame_id, &self.kf);
            if track.tracklet_len >= self.config.min_hits {
                track.activate(self.frame_id);
                self.tracked_stracks.push(track);
            } else {
                self.unconfirmed_stracks.push(track);
            }
        }

        for &t_idx in &unmatched_unconfirmed {
            let mut track = unconfirmed[t_idx].clone();
            track.mark_removed();
            self.removed_stracks.push(track);
        }

        // 6. Initialize new tracks in unconfirmed state
        for &d_idx in &unmatched_final_dets {
            let mut track = self.remaining_dets_buf[d_idx].clone();
            if self.config.min_hits <= 1 {
                track.activate(self.frame_id);
                self.tracked_stracks.push(track);
            } else {
                self.unconfirmed_stracks.push(track);
            }
        }

        // 7. Prune expired lost tracks
        self.lost_stracks.retain_mut(|track| {
            if self.frame_id.saturating_sub(track.age - track.time_since_update)
                > self.config.max_time_lost
            {
                track.mark_removed();
                self.removed_stracks.push(track.clone());
                false
            } else {
                true
            }
        });

        // 8. Return active confirmed vehicles
        self.tracked_stracks
            .iter()
            .filter(|t| t.is_activated && t.state == TrackState::Tracked)
            .map(|t| TrackedVehicle {
                track_id: t.track_id,
                class: t.class,
                confidence: t.confidence,
                bbox: t.to_bbox(),
                tracklet_len: t.tracklet_len,
                age: t.age,
            })
            .collect()
    }
}

/// Helper function to decouple borrowing of `cost_matrix_buf` from `self`
fn associate_internal(
    tracks: &[STrack],
    dets: &[STrack],
    max_dist: f32,
    class_mismatch_penalty: f32,
    motion_weight: f32,
    cost_matrix_buf: &mut Vec<f32>,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let n_t = tracks.len();
    let n_d = dets.len();

    if n_t == 0 || n_d == 0 {
        return (Vec::new(), (0..n_t).collect(), (0..n_d).collect());
    }

    cost_matrix_buf.clear();
    cost_matrix_buf.reserve(n_t * n_d);

    for t in tracks {
        let b_t = t.to_bbox();
        let (vx, vy) = t.velocity();
        let speed = (vx * vx + vy * vy).sqrt();
        let cx_t = (b_t[0] + b_t[2]) * 0.5;
        let cy_t = (b_t[1] + b_t[3]) * 0.5;

        for d in dets {
            let b_d = d.to_bbox();
            let diou = calculate_diou(&b_t, &b_d);
            let mut cost = 1.0 - diou;

            // 1. Class consistency penalty
            if t.class != d.class {
                cost += class_mismatch_penalty;
            }

            // 2. Directional motion cosine penalty
            if speed > 2.0 {
                let cx_d = (b_d[0] + b_d[2]) * 0.5;
                let cy_d = (b_d[1] + b_d[3]) * 0.5;
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

            cost_matrix_buf.push(cost);
        }
    }

    lapjv(cost_matrix_buf, n_t, n_d, max_dist)
}

/// Distance-IoU metric
#[inline]
pub fn calculate_diou(box_a: &[f32; 4], box_b: &[f32; 4]) -> f32 {
    let inter_x1 = box_a[0].max(box_b[0]);
    let inter_y1 = box_a[1].max(box_b[1]);
    let inter_x2 = box_a[2].min(box_b[2]);
    let inter_y2 = box_a[3].min(box_b[3]);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = (box_a[2] - box_a[0]).max(0.0) * (box_a[3] - box_a[1]).max(0.0);
    let area_b = (box_b[2] - box_b[0]).max(0.0) * (box_b[3] - box_b[1]).max(0.0);
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        return -1.0;
    }

    let iou = inter_area / union_area;

    let cx_a = (box_a[0] + box_a[2]) * 0.5;
    let cy_a = (box_a[1] + box_a[3]) * 0.5;
    let cx_b = (box_b[0] + box_b[2]) * 0.5;
    let cy_b = (box_b[1] + box_b[3]) * 0.5;

    let center_dist_sq = (cx_a - cx_b) * (cx_a - cx_b) + (cy_a - cy_b) * (cy_a - cy_b);

    let enc_x1 = box_a[0].min(box_b[0]);
    let enc_y1 = box_a[1].min(box_b[1]);
    let enc_x2 = box_a[2].max(box_b[2]);
    let enc_y2 = box_a[3].max(box_b[3]);

    let enc_w = enc_x2 - enc_x1;
    let enc_h = enc_y2 - enc_y1;
    let enc_diag_sq = (enc_w * enc_w + enc_h * enc_h).max(1e-5);

    iou - (center_dist_sq / enc_diag_sq)
}