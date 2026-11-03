use crate::types::QualityBreakdown;
use image::{DynamicImage, GenericImageView};

#[derive(Debug, Clone)]
pub struct QualityConfig {
    pub min_sharpness: f32,
    pub min_resolution: u32,
    pub brightness_range: (f32, f32),
    pub min_contrast: f32,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            min_sharpness: 25.0,
            min_resolution: 90, // Require at least 90px minimum dimension in native frame
            brightness_range: (30.0, 230.0),
            min_contrast: 0.10,
        }
    }
}

pub struct QualityAuditor {
    config: QualityConfig,
}

impl QualityAuditor {
    pub fn new(config: QualityConfig) -> Self {
        Self { config }
    }

    pub fn evaluate(
        &self,
        crop: &DynamicImage,
        bbox: &[f32; 4],
        frame_w: u32,
        frame_h: u32,
    ) -> QualityBreakdown {
        let (w, h) = crop.dimensions();

        // 1. Reject small crops (at native 720p/1080p, readable plates need >= 90px vehicles)
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

        let gray = crop.to_luma8();
        let raw = gray.as_raw();
        let stride = w as usize;
        let total_pixels = (w * h) as f64;

        // 2. Mean Brightness & RMS Contrast
        let mut sum_brightness = 0.0f64;
        let mut sum_brightness_sq = 0.0f64;

        for &px in raw {
            let v = px as f64;
            sum_brightness += v;
            sum_brightness_sq += v * v;
        }

        let mean_brightness = (sum_brightness / total_pixels) as f32;
        let variance_brightness =
            (sum_brightness_sq / total_pixels) - (mean_brightness as f64).powi(2);
        let rms_contrast = (variance_brightness.max(0.0).sqrt() / 255.0) as f32;

        // 3. Discrete Laplacian Kernel
        let mut lap_sum = 0.0f64;
        let mut lap_sum_sq = 0.0f64;
        let mut lap_count = 0.0f64;

        let h_usize = h as usize;
        let w_usize = w as usize;

        for y in 1..(h_usize - 1) {
            let row_offset = y * stride;
            let top_offset = (y - 1) * stride;
            let bot_offset = (y + 1) * stride;

            for x in 1..(w_usize - 1) {
                let center = raw[row_offset + x] as f64;
                let top = raw[top_offset + x] as f64;
                let bottom = raw[bot_offset + x] as f64;
                let left = raw[row_offset + (x - 1)] as f64;
                let right = raw[row_offset + (x + 1)] as f64;

                let lap = top + bottom + left + right - 4.0 * center;
                lap_sum += lap;
                lap_sum_sq += lap * lap;
                lap_count += 1.0;
            }
        }

        let sharpness = if lap_count > 0.0 {
            let mean_lap = lap_sum / lap_count;
            ((lap_sum_sq / lap_count) - (mean_lap * mean_lap)).max(0.0) as f32
        } else {
            0.0
        };

        // 4. Border Proximity Penalty
        let margin_x = 8.0;
        let margin_y = 8.0;
        let mut border_penalty = 1.0f32;

        if bbox[0] <= margin_x
            || bbox[1] <= margin_y
            || bbox[2] >= (frame_w as f32 - margin_x)
            || bbox[3] >= (frame_h as f32 - margin_y)
        {
            border_penalty = 0.25;
        }

        // 5. Illumination Gating
        let (min_b, max_b) = self.config.brightness_range;
        let brightness_factor = if mean_brightness < min_b {
            (mean_brightness / min_b).powi(2)
        } else if mean_brightness > max_b {
            ((255.0 - mean_brightness) / (255.0 - max_b)).powi(2)
        } else {
            1.0
        };

        // 6. Quadratic Area Weighting
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