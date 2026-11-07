//! High-performance direct-slice optical metric extractors.
//!
//! # Guarantees
//! - Zero heap allocations.
//! - Closed-form single-pass pixel accumulation.
//! - Guarded against division-by-zero on degenerate crops.

use vehicle_detector::BoundingBox;

/// Standard ITU-R BT.601 integer luma coefficients for SIMD-friendly conversion:
/// Y = (R * 77 + G * 150 + B * 29) >> 8
#[inline(always)]
pub fn rgb_to_luma(r: u8, g: u8, b: u8) -> u8 {
    let luma = ((r as u32 * 77) + (g as u32 * 150) + (b as u32 * 29)) >> 8;
    luma as u8
}

/// Computes mean brightness and RMS contrast directly on a sub-rectangle of an RGB8 buffer.
///
/// Guaranteed zero heap allocations.
pub fn compute_luma_statistics_direct(
    raw_rgb: &[u8],
    frame_w: u32,
    frame_h: u32,
    bbox: &BoundingBox,
) -> (f32, f32) {
    let (x1, y1, x2, y2) = clamped_pixel_bounds(bbox, frame_w, frame_h);
    let crop_w = x2.saturating_sub(x1);
    let crop_h = y2.saturating_sub(y1);

    if crop_w == 0 || crop_h == 0 {
        return (0.0, 0.0);
    }

    let stride = (frame_w * 3) as usize;
    let total_pixels = (crop_w as f64) * (crop_h as f64);

    let mut sum_luma = 0.0f64;
    let mut sum_luma_sq = 0.0f64;

    for y in y1..y2 {
        let row_start = (y as usize) * stride + (x1 as usize) * 3;
        for x in 0..crop_w {
            let px_idx = row_start + (x as usize) * 3;
            if let (Some(&r), Some(&g), Some(&b)) = (
                raw_rgb.get(px_idx),
                raw_rgb.get(px_idx + 1),
                raw_rgb.get(px_idx + 2),
            ) {
                let l = rgb_to_luma(r, g, b) as f64;
                sum_luma += l;
                sum_luma_sq += l * l;
            }
        }
    }

    let mean_luma = (sum_luma / total_pixels) as f32;
    let variance = ((sum_luma_sq / total_pixels) - (mean_luma as f64).powi(2)).max(0.0);
    let rms_contrast = (variance.sqrt() / 255.0) as f32;

    (mean_luma, rms_contrast)
}

/// Computes Discrete Laplacian Variance (Sharpness) directly on the bounded RGB frame region.
///
/// Convolution kernel:
/// ```text
///  [ 0,  1,  0 ]
///  [ 1, -4,  1 ]
///  [ 0,  1,  0 ]
/// ```
pub fn compute_laplacian_variance_direct(
    raw_rgb: &[u8],
    frame_w: u32,
    frame_h: u32,
    bbox: &BoundingBox,
) -> f32 {
    let (x1, y1, x2, y2) = clamped_pixel_bounds(bbox, frame_w, frame_h);
    let crop_w = x2.saturating_sub(x1);
    let crop_h = y2.saturating_sub(y1);

    if crop_w < 3 || crop_h < 3 {
        return 0.0;
    }

    let stride = (frame_w * 3) as usize;
    let mut lap_sum = 0.0f64;
    let mut lap_sum_sq = 0.0f64;
    let mut count = 0.0f64;

    // Convolution operates inside borders [1..crop_h - 1] and [1..crop_w - 1]
    for dy in 1..(crop_h - 1) {
        let y = y1 + dy;
        let row_center = (y as usize) * stride;
        let row_top = ((y - 1) as usize) * stride;
        let row_bottom = ((y + 1) as usize) * stride;

        for dx in 1..(crop_w - 1) {
            let x = x1 + dx;
            let col = (x as usize) * 3;
            let col_left = ((x - 1) as usize) * 3;
            let col_right = ((x + 1) as usize) * 3;

            // Extract luma values directly
            let c = get_luma_at(raw_rgb, row_center + col);
            let t = get_luma_at(raw_rgb, row_top + col);
            let b = get_luma_at(raw_rgb, row_bottom + col);
            let l = get_luma_at(raw_rgb, row_center + col_left);
            let r = get_luma_at(raw_rgb, row_center + col_right);

            let lap = (t + b + l + r) - 4.0 * c;
            lap_sum += lap;
            lap_sum_sq += lap * lap;
            count += 1.0;
        }
    }

    if count > 0.0 {
        let mean = lap_sum / count;
        let variance = (lap_sum_sq / count) - (mean * mean);
        variance.max(0.0) as f32
    } else {
        0.0
    }
}

#[inline(always)]
fn get_luma_at(raw: &[u8], idx: usize) -> f64 {
    if idx + 2 < raw.len() {
        rgb_to_luma(raw[idx], raw[idx + 1], raw[idx + 2]) as f64
    } else {
        0.0
    }
}

#[inline(always)]
pub fn clamped_pixel_bounds(
    bbox: &BoundingBox,
    frame_w: u32,
    frame_h: u32,
) -> (u32, u32, u32, u32) {
    let x1 = (bbox.x1.max(0.0) as u32).min(frame_w);
    let y1 = (bbox.y1.max(0.0) as u32).min(frame_h);
    let x2 = (bbox.x2.max(0.0) as u32).min(frame_w).max(x1);
    let y2 = (bbox.y2.max(0.0) as u32).min(frame_h).max(y1);
    (x1, y1, x2, y2)
}