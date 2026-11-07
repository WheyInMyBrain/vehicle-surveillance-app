//! Safety-critical domain primitives for License Plate Recognition (LPR).

use image::{DynamicImage, GenericImageView};
use vehicle_detector::BoundingBox;

/// Maximum number of candidate plate proposals evaluated by LPR NMS.
pub const MAX_PLATE_PROPOSALS: usize = 64;
/// Maximum number of final detected license plates per vehicle crop.
pub const MAX_OUTPUT_PLATES: usize = 4;

/// Detected license plate bounding box with optical verification metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct DetectedPlate {
    pub confidence: f32,
    pub bbox: BoundingBox,
}

impl DetectedPlate {
    #[inline(always)]
    pub fn x1(&self) -> f32 { self.bbox.x1 }
    #[inline(always)]
    pub fn y1(&self) -> f32 { self.bbox.y1 }
    #[inline(always)]
    pub fn x2(&self) -> f32 { self.bbox.x2 }
    #[inline(always)]
    pub fn y2(&self) -> f32 { self.bbox.y2 }

    #[inline(always)]
    pub fn width(&self) -> f32 { self.bbox.width() }

    #[inline(always)]
    pub fn height(&self) -> f32 { self.bbox.height() }

    #[inline(always)]
    pub fn area(&self) -> f32 { self.bbox.area() }

    /// Aspect ratio (width / height) — Standard automotive plates range between 2.0 and 5.2.
    #[inline(always)]
    pub fn aspect_ratio(&self) -> f32 {
        let h = self.height();
        if h > 1e-4 { self.width() / h } else { 0.0 }
    }

    /// Safely isolates and crops the license plate out of the vehicle crop.
    pub fn crop_from(&self, vehicle_crop: &DynamicImage) -> DynamicImage {
        let (crop_w, crop_h) = vehicle_crop.dimensions();

        let x1 = (self.bbox.x1.max(0.0) as u32).min(crop_w);
        let y1 = (self.bbox.y1.max(0.0) as u32).min(crop_h);
        let x2 = (self.bbox.x2.max(0.0) as u32).min(crop_w);
        let y2 = (self.bbox.y2.max(0.0) as u32).min(crop_h);

        let w = x2.saturating_sub(x1).max(1);
        let h = y2.saturating_sub(y1).max(1);

        vehicle_crop.crop_imm(x1, y1, w, h)
    }
}