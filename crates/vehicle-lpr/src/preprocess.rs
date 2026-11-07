//! High-performance preprocessing for letterboxed plate detector NCHW tensor inputs.

use image::{DynamicImage, GenericImageView};
use ndarray::Array4;
use vehicle_detector::{BoundingBox, LetterboxTransform};

const YOLO_PAD_VALUE: f32 = 114.0 / 255.0;
const INV_255: f32 = 1.0 / 255.0;

/// Fast Intersection-over-Union calculation with mathematical singularity protection.
#[inline(always)]
pub fn calculate_iou_fast(a: &BoundingBox, a_area: f32, b: &BoundingBox, b_area: f32) -> f32 {
    let x1 = if a.x1 > b.x1 { a.x1 } else { b.x1 };
    let y1 = if a.y1 > b.y1 { a.y1 } else { b.y1 };
    let x2 = if a.x2 < b.x2 { a.x2 } else { b.x2 };
    let y2 = if a.y2 < b.y2 { a.y2 } else { b.y2 };

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let union_area = a_area + b_area - inter_area;
    if union_area <= 1e-6 {
        0.0
    } else {
        inter_area / union_area
    }
}

/// Prepares the pre-allocated tensor buffer with normalized pixel data without intermediate heap copies.
pub fn fill_letterbox_tensor_direct(
    img: &DynamicImage,
    target_w: u32,
    target_h: u32,
    tensor: &mut Array4<f32>,
) -> Result<LetterboxTransform, &'static str> {
    let (orig_w, orig_h) = img.dimensions();
    if orig_w == 0 || orig_h == 0 {
        return Err("Source image dimensions cannot be zero");
    }

    let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale).round() as u32;
    let new_h = (orig_h as f32 * scale).round() as u32;

    let pad_x = ((target_w - new_w) as f32 * 0.5).floor();
    let pad_y = ((target_h - new_h) as f32 * 0.5).floor();

    let tw = target_w as usize;
    let th = target_h as usize;
    let plane_size = tw * th;

    let slice = tensor
        .as_slice_mut()
        .ok_or("Tensor buffer must be contiguous memory")?;

    if slice.len() != 3 * plane_size {
        return Err("Tensor shape does not match target dimensions");
    }

    slice.fill(YOLO_PAD_VALUE);

    let (c0, rest) = slice.split_at_mut(plane_size);
    let (c1, c2) = rest.split_at_mut(plane_size);

    let scale_x = orig_w as f32 / new_w as f32;
    let scale_y = orig_h as f32 / new_h as f32;

    let pad_x_u = pad_x as usize;
    let pad_y_u = pad_y as usize;
    let new_h_u = new_h as usize;
    let new_w_u = new_w as usize;
    let orig_h_max = (orig_h - 1) as usize;
    let orig_w_max = (orig_w - 1) as usize;

    if let DynamicImage::ImageRgb8(rgb) = img {
        let raw = rgb.as_raw();
        let src_stride = (orig_w * 3) as usize;

        for dy in 0..new_h_u {
            let sy = ((dy as f32 * scale_y) as usize).min(orig_h_max);
            let src_row = sy * src_stride;
            let dst_row = (pad_y_u + dy) * tw + pad_x_u;

            for dx in 0..new_w_u {
                let sx = ((dx as f32 * scale_x) as usize).min(orig_w_max);
                let src_idx = src_row + sx * 3;
                let dst_idx = dst_row + dx;

                if let (Some(r), Some(g), Some(b)) = (raw.get(src_idx), raw.get(src_idx + 1), raw.get(src_idx + 2)) {
                    if let (Some(tc0), Some(tc1), Some(tc2)) = (c0.get_mut(dst_idx), c1.get_mut(dst_idx), c2.get_mut(dst_idx)) {
                        *tc0 = (*r as f32) * INV_255;
                        *tc1 = (*g as f32) * INV_255;
                        *tc2 = (*b as f32) * INV_255;
                    }
                }
            }
        }
    } else {
        let rgb = img.to_rgb8();
        let raw = rgb.as_raw();
        let src_stride = (orig_w * 3) as usize;

        for dy in 0..new_h_u {
            let sy = ((dy as f32 * scale_y) as usize).min(orig_h_max);
            let src_row = sy * src_stride;
            let dst_row = (pad_y_u + dy) * tw + pad_x_u;

            for dx in 0..new_w_u {
                let sx = ((dx as f32 * scale_x) as usize).min(orig_w_max);
                let src_idx = src_row + sx * 3;
                let dst_idx = dst_row + dx;

                if let (Some(r), Some(g), Some(b)) = (raw.get(src_idx), raw.get(src_idx + 1), raw.get(src_idx + 2)) {
                    if let (Some(tc0), Some(tc1), Some(tc2)) = (c0.get_mut(dst_idx), c1.get_mut(dst_idx), c2.get_mut(dst_idx)) {
                        *tc0 = (*r as f32) * INV_255;
                        *tc1 = (*g as f32) * INV_255;
                        *tc2 = (*b as f32) * INV_255;
                    }
                }
            }
        }
    }

    Ok(LetterboxTransform { scale, pad_x, pad_y })
}