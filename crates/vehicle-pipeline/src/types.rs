//! Safety-critical domain representations of pipeline frame results and telemetry.

use core::time::Duration;
use heapless::Vec as StackVec;
use vehicle_detector::{BoundingBox, VehicleClass};
use vehicle_lpocr::OcrResult;
use vehicle_tracker::MAX_TRACKS;

/// Granular per-stage microsecond execution durations for deterministic telemetry.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct PipelineStageTimings {
    pub vehicle_detect_time: Duration,
    pub tracking_time: Duration,
    pub quality_triage_time: Duration,
    pub plate_detect_time: Duration,
    pub plate_ocr_time: Duration,
    pub annotation_time: Duration,
    pub total_frame_time: Duration,
}

impl PipelineStageTimings {
    #[inline(always)]
    pub fn fps(&self) -> f64 {
        let secs = self.total_frame_time.as_secs_f64();
        if secs > 1e-6 { 1.0 / secs } else { 0.0 }
    }
}

/// A finalized vehicle observation within a single frame.
#[derive(Debug, Clone, PartialEq)]
pub struct VehicleSnapshot {
    pub track_id: u64,
    pub class: VehicleClass,
    pub confidence: f32,
    pub bbox: BoundingBox,
    pub tracklet_len: usize,
    /// Absolute license plate bounding box in full frame coordinates (if localized)
    pub plate_bbox: Option<BoundingBox>,
    /// Confirmed optical character recognition result
    pub plate_ocr: Option<OcrResult>,
}

impl VehicleSnapshot {
    #[inline(always)]
    pub fn x1(&self) -> f32 { self.bbox.x1 }
    #[inline(always)]
    pub fn y1(&self) -> f32 { self.bbox.y1 }
    #[inline(always)]
    pub fn x2(&self) -> f32 { self.bbox.x2 }
    #[inline(always)]
    pub fn y2(&self) -> f32 { self.bbox.y2 }
}

/// The complete telemetry payload of an evaluated video or shutter frame.
#[derive(Debug, Clone)]
pub struct PipelineFrameResult {
    pub frame_idx: usize,
    pub active_track_count: usize,
    pub plates_read_count: usize,
    pub vehicles: StackVec<VehicleSnapshot, MAX_TRACKS>,
    pub timings: PipelineStageTimings,
}