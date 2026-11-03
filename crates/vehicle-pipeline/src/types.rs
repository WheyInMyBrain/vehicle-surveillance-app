use std::time::Duration;
use vehicle_detector::VehicleClass;
use vehicle_lpocr::OcrResult;

/// Granular per-stage latencies for a single frame
#[derive(Debug, Clone, Default)]
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
    pub fn fps(&self) -> f64 {
        let secs = self.total_frame_time.as_secs_f64();
        if secs > 0.0 { 1.0 / secs } else { 0.0 }
    }
}

/// A finalized vehicle observation within a processed frame
#[derive(Debug, Clone)]
pub struct VehicleSnapshot {
    pub track_id: u64,
    pub class: VehicleClass,
    pub confidence: f32,
    pub bbox: [f32; 4],
    pub tracklet_len: usize,
    /// Absolute license plate bbox in frame coordinates (if detected)
    pub plate_bbox: Option<[f32; 4]>,
    /// Best plate OCR text and confidence for this track so far
    pub plate_ocr: Option<OcrResult>,
}

/// The complete output of a processed frame or image
#[derive(Debug, Clone)]
pub struct PipelineFrameResult {
    pub frame_idx: usize,
    pub vehicles: Vec<VehicleSnapshot>,
    pub active_track_count: usize,
    pub plates_read_count: usize,
    pub timings: PipelineStageTimings,
}