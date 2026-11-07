//! Individual vehicle tracklet state machine.

use crate::core::kalman::KalmanFilter;
use crate::core::types::TrackState;
use core::sync::atomic::{AtomicU64, Ordering};
use vehicle_detector::{BoundingBox, DetectedVehicle, VehicleClass};

static GLOBAL_TRACK_ID: AtomicU64 = AtomicU64::new(1);

#[inline(always)]
pub fn get_next_track_id() -> u64 {
    GLOBAL_TRACK_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy)]
pub struct STrack {
    pub track_id: u64,
    pub class: VehicleClass,
    pub confidence: f32,
    pub state: TrackState,
    pub mean: [f32; 8],
    pub covariance: [f32; 64],
    pub is_activated: bool,
    pub tracklet_len: usize,
    pub age: usize,
    pub time_since_update: usize,
}

impl STrack {
    pub fn new(detection: &DetectedVehicle, kf: &KalmanFilter) -> Self {
        let z = KalmanFilter::bbox_to_z(&detection.bbox);
        let (mean, covariance) = kf.initiate(&z);

        Self {
            track_id: 0,
            class: detection.class,
            confidence: detection.confidence,
            state: TrackState::New,
            mean,
            covariance,
            is_activated: false,
            tracklet_len: 0,
            age: 0,
            time_since_update: 0,
        }
    }

    #[inline(always)]
    pub fn to_bbox(&self) -> BoundingBox {
        KalmanFilter::z_to_bbox(&self.mean)
    }

    #[inline(always)]
    pub fn velocity(&self) -> (f32, f32) {
        (self.mean[4], self.mean[5])
    }

    pub fn predict(&mut self, kf: &KalmanFilter) {
        if self.state != TrackState::Tracked {
            self.mean[7] = 0.0;
        }
        kf.predict(&mut self.mean, &mut self.covariance);
        self.age += 1;
        self.time_since_update += 1;
    }

    pub fn activate(&mut self, frame_id: usize) {
        self.track_id = get_next_track_id();
        self.state = TrackState::Tracked;
        if frame_id == 1 {
            self.is_activated = true;
        }
        self.tracklet_len = 1;
        self.time_since_update = 0;
    }

    pub fn re_activate(&mut self, new_track: &Self, _frame_id: usize, kf: &KalmanFilter) {
        let z = KalmanFilter::bbox_to_z(&new_track.to_bbox());
        kf.update(&mut self.mean, &mut self.covariance, &z);
        self.confidence = new_track.confidence;
        self.class = new_track.class;
        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.tracklet_len += 1;
        self.time_since_update = 0;
    }

    pub fn update(&mut self, new_track: &Self, _frame_id: usize, kf: &KalmanFilter) {
        let z = KalmanFilter::bbox_to_z(&new_track.to_bbox());
        kf.update(&mut self.mean, &mut self.covariance, &z);
        self.confidence = new_track.confidence;
        self.class = new_track.class;
        self.state = TrackState::Tracked;
        self.is_activated = true;
        self.tracklet_len += 1;
        self.time_since_update = 0;
    }

    #[inline(always)]
    pub fn mark_lost(&mut self) {
        self.state = TrackState::Lost;
    }

    #[inline(always)]
    pub fn mark_removed(&mut self) {
        self.state = TrackState::Removed;
    }
}