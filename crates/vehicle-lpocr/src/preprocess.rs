//! In-place preprocessing for NHWC OCR tensor inputs.
//!
//! Directly maps source image pixels into target NHWC [1, H, W, 3] layout without heap allocations.

use image::{DynamicImage, GenericImageView};
use ndarray::Array4;

/// In-place bilinear sampling from DynamicImage into an NHWC [1, H, W, 3] tensor slice.
///
/// Guaranteed zero dynamic heap allocations.
pub fn prepare_ocr_tensor_direct(
    img: &DynamicImage,
    target_w: u32,
    target_h: u32,
    tensor: &mut Array4<u8>,
) -> Result<(), &'static str> {
    let (orig_w, orig_h) = img.dimensions();
    if orig_w == 0 || orig_h == 0 {
        return Err("Source crop dimensions cannot be zero");
    }

    let tw = target_w as usize;
    let th = target_h as usize;
    let expected_len = tw * th * 3;

    let slice = tensor
        .as_slice_mut()
        .ok_or("Tensor buffer must be contiguous memory")?;

    if slice.len() != expected_len {
        return Err("Tensor shape does not match target NHWC dimensions");
    }

    let scale_x = orig_w as f32 / target_w as f32;
    let scale_y = orig_h as f32 / target_h as f32;

    let orig_h_max = (orig_h - 1) as usize;
    let orig_w_max = (orig_w - 1) as usize;

    if let DynamicImage::ImageRgb8(rgb) = img {
        let raw = rgb.as_raw();
        let src_stride = (orig_w * 3) as usize;

        for dy in 0..th {
            let sy = ((dy as f32 * scale_y) as usize).min(orig_h_max);
            let src_row = sy * src_stride;
            let dst_row = dy * tw * 3;

            for dx in 0..tw {
                let sx = ((dx as f32 * scale_x) as usize).min(orig_w_max);
                let src_idx = src_row + sx * 3;
                let dst_idx = dst_row + dx * 3;

                if let Some(src_px) = raw.get(src_idx..src_idx + 3) {
                    if let Some(dst_px) = slice.get_mut(dst_idx..dst_idx + 3) {
                        dst_px.copy_from_slice(src_px);
                    }
                }
            }
        }
    } else {
        let rgb = img.to_rgb8();
        let raw = rgb.as_raw();
        let src_stride = (orig_w * 3) as usize;

        for dy in 0..th {
            let sy = ((dy as f32 * scale_y) as usize).min(orig_h_max);
            let src_row = sy * src_stride;
            let dst_row = dy * tw * 3;

            for dx in 0..tw {
                let sx = ((dx as f32 * scale_x) as usize).min(orig_w_max);
                let src_idx = src_row + sx * 3;
                let dst_idx = dst_row + dx * 3;

                if let Some(src_px) = raw.get(src_idx..src_idx + 3) {
                    if let Some(dst_px) = slice.get_mut(dst_idx..dst_idx + 3) {
                        dst_px.copy_from_slice(src_px);
                    }
                }
            }
        }
    }

    Ok(())
}