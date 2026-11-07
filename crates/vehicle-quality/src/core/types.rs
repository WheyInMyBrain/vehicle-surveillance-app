//! Safety-critical domain primitives for optical quality evaluation.

use vehicle_detector::{BoundingBox, VehicleClass};

/// Detailed breakdown of optical metrics for a vehicle region.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct QualityBreakdown {
    /// Laplacian variance score (> 60 indicates sharp edges, < 25 is motion-blurred).
    pub sharpness: f32,
    /// Root-Mean-Square (RMS) luma contrast (0.0 to 1.0).
    pub contrast: f32,
    /// Average luma intensity (0.0 to 255.0).
    pub mean_brightness: f32,
    /// Crop dimensions: `[width, height]`.
    pub resolution: [u32; 2],
    /// Edge proximity factor: 1.0 = comfortably inside frame; 0.25 = clipped by frame edge.
    pub border_penalty: f32,
    /// Multi-objective composite score for downstream plate detection/OCR.
    pub composite_score: f32,
    /// Strict binary viability gate.
    pub is_ocr_viable: bool,
}

impl Default for QualityBreakdown {
    fn default() -> Self {
        Self {
            sharpness: 0.0,
            contrast: 0.0,
            mean_brightness: 0.0,
            resolution: [0, 0],
            border_penalty: 0.0,
            composite_score: 0.0,
            is_ocr_viable: false,
        }
    }
}

/// Lightweight metadata representation of a candidate golden crop.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct GoldenCropMetadata {
    pub frame_idx: usize,
    pub track_id: u64,
    pub class: VehicleClass,
    pub bbox: BoundingBox,
    pub metrics: QualityBreakdown,
}