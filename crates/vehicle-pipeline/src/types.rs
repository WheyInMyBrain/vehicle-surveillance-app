use vehicle_detector::VehicleClass;
use vehicle_lpocr::OcrResult;

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
}