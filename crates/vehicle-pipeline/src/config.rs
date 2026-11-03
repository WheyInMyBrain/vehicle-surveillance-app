use std::path::PathBuf;
use vehicle_detector::VehicleDetectorConfig;
use vehicle_lpocr::PlateOcrConfig;
use vehicle_lpr::PlateDetectorConfig;
use vehicle_quality::QualityConfig;
use vehicle_tracker::ByteTrackConfig;

#[derive(Debug, Clone)]
pub struct SurveillancePipelineConfig {
    pub vehicle_model_path: PathBuf,
    pub plate_model_path: PathBuf,
    pub ocr_model_path: PathBuf,

    pub detector: VehicleDetectorConfig,
    pub tracker: ByteTrackConfig,
    pub quality: QualityConfig,
    pub plate_detector: PlateDetectorConfig,
    pub ocr: PlateOcrConfig,

    /// Number of golden crops to retain in memory per vehicle track
    pub golden_crops_per_track: usize,
    /// Minimum composite score required for candidate consideration
    pub min_quality_score: f32,
    /// Path to a true-type font for high-legibility rendering
    pub custom_font_path: Option<PathBuf>,
}

impl SurveillancePipelineConfig {
    pub fn new<P: Into<PathBuf>>(vehicle: P, plate: P, ocr: P) -> Self {
        Self {
            vehicle_model_path: vehicle.into(),
            plate_model_path: plate.into(),
            ocr_model_path: ocr.into(),

            detector: VehicleDetectorConfig {
                confidence_threshold: 0.25,
                nms_iou_threshold: 0.45,
                input_width: 640,
                input_height: 640,
                min_box_dimension: 20.0,
                num_threads: 4,
            },
            tracker: ByteTrackConfig {
                track_thresh: 0.25,
                high_thresh: 0.45,
                match_thresh: 0.75,
                max_time_lost: 40,
                min_hits: 2,
                ..Default::default()
            },
            quality: QualityConfig {
                min_sharpness: 25.0,
                min_resolution: 80,
                brightness_range: (30.0, 230.0),
                min_contrast: 0.10,
            },
            plate_detector: PlateDetectorConfig {
                confidence_threshold: 0.20,
                nms_iou_threshold: 0.40,
                input_width: 640,
                input_height: 640,
                min_plate_height: 6.0,
                min_plate_width: 14.0,
                num_threads: 4,
            },
            ocr: PlateOcrConfig::default(),

            golden_crops_per_track: 3,
            min_quality_score: 10.0,
            custom_font_path: None,
        }
    }
}