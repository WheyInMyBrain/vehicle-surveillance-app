//! Direct frame optical quality auditor.

use crate::core::metrics::{
    clamped_pixel_bounds, compute_laplacian_variance_direct, compute_luma_statistics_direct,
};
use crate::core::types::QualityBreakdown;
use vehicle_detector::BoundingBox;

#[derive(Debug, Clone, Copy)]
pub struct QualityConfig {
    pub min_sharpness: f32,
    pub min_resolution: u32,
    pub brightness_range: (f32, f32),
    pub min_contrast: f32,
    pub border_margin: f32,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            min_sharpness: 25.0,
            min_resolution: 90,
            brightness_range: (30.0, 230.0),
            min_contrast: 0.10,
            border_margin: 8.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QualityAuditor {
    config: QualityConfig,
}

impl Default for QualityAuditor {
    fn default() -> Self {
        Self::new(QualityConfig::default())
    }
}

impl QualityAuditor {
    #[inline(always)]
    pub const fn new(config: QualityConfig) -> Self {
        Self { config }
    }

    /// Evaluates optical characteristics of a vehicle directly from raw frame pixels.
    ///
    /// # Zero-Allocation Guarantee
    /// Executes completely on the stack with **zero** heap allocations.
    pub fn evaluate_direct(
        &self,
        raw_rgb: &[u8],
        frame_w: u32,
        frame_h: u32,
        bbox: &BoundingBox,
    ) -> QualityBreakdown {
        let (x1, y1, x2, y2) = clamped_pixel_bounds(bbox, frame_w, frame_h);
        let w = x2.saturating_sub(x1);
        let h = y2.saturating_sub(y1);

        // 1. Resolution Viability Gate
        if w < self.config.min_resolution || h < (self.config.min_resolution / 2) {
            return QualityBreakdown {
                sharpness: 0.0,
                contrast: 0.0,
                mean_brightness: 0.0,
                resolution: [w, h],
                border_penalty: 0.0,
                composite_score: 0.0,
                is_ocr_viable: false,
            };
        }

        // 2. Direct Luma & RMS Contrast Calculation
        let (mean_brightness, rms_contrast) =
            compute_luma_statistics_direct(raw_rgb, frame_w, frame_h, bbox);

        // 3. Direct Laplacian Variance Calculation
        let sharpness = compute_laplacian_variance_direct(raw_rgb, frame_w, frame_h, bbox);

        // 4. Border Proximity Penalty
        let margin = self.config.border_margin;
        let mut border_penalty = 1.0f32;
        if bbox.x1 <= margin
            || bbox.y1 <= margin
            || bbox.x2 >= (frame_w as f32 - margin)
            || bbox.y2 >= (frame_h as f32 - margin)
        {
            border_penalty = 0.25;
        }

        // 5. Illumination Gating
        let (min_b, max_b) = self.config.brightness_range;
        let brightness_factor = if mean_brightness < min_b {
            (mean_brightness / min_b).powi(2)
        } else if mean_brightness > max_b {
            let denom = 255.0 - max_b;
            if denom > 1e-4 {
                ((255.0 - mean_brightness) / denom).powi(2)
            } else {
                0.0
            }
        } else {
            1.0
        };

        // 6. Multi-Objective Composite Score
        // Area Weighting: sqrt(area / 18,000) capped at 4.5
        let area = (w * h) as f32;
        let area_score = (area / 18_000.0).sqrt().min(4.5);
        let normalized_sharpness = sharpness.min(1800.0).sqrt();

        let composite_score = area_score
            * normalized_sharpness
            * (rms_contrast * 2.0).min(1.5)
            * brightness_factor
            * border_penalty;

        let is_ocr_viable = sharpness >= self.config.min_sharpness
            && w >= self.config.min_resolution
            && rms_contrast >= self.config.min_contrast
            && mean_brightness >= min_b
            && mean_brightness <= max_b
            && border_penalty > 0.5;

        QualityBreakdown {
            sharpness,
            contrast: rms_contrast,
            mean_brightness,
            resolution: [w, h],
            border_penalty,
            composite_score,
            is_ocr_viable,
        }
    }
}