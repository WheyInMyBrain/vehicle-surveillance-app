use image::DynamicImage;
use vehicle_detector::VehicleClass;

/// Detailed breakdown of optical metrics for a vehicle crop
#[derive(Debug, Clone, PartialEq)]
pub struct QualityBreakdown {
    /// Laplacian variance score (> 60 is sharp, < 30 is blurry)
    pub sharpness: f32,
    /// RMS contrast score (0.0 - 1.0)
    pub contrast: f32,
    /// Average luma intensity (0.0 - 255.0)
    pub mean_brightness: f32,
    /// Crop dimensions: [width, height]
    pub resolution: [u32; 2],
    /// Edge proximity penalty factor (1.0 = fully inside, < 0.5 = touching frame border)
    pub border_penalty: f32,
    /// Composite aggregated score (higher is better for OCR)
    pub composite_score: f32,
    /// Binary gate whether this crop passes minimal OCR threshold requirements
    pub is_ocr_viable: bool,
}

/// A stored golden candidate crop for a tracked vehicle
#[derive(Clone)]
pub struct GoldenCrop {
    pub frame_idx: usize,
    pub track_id: u64,
    pub class: VehicleClass,
    pub crop: DynamicImage,
    pub bbox: [f32; 4],
    pub metrics: QualityBreakdown,
}

impl std::fmt::Debug for GoldenCrop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoldenCrop")
            .field("frame_idx", &self.frame_idx)
            .field("track_id", &self.track_id)
            .field("class", &self.class.as_str())
            .field("composite_score", &self.metrics.composite_score)
            .field("resolution", &self.metrics.resolution)
            .finish()
    }
}