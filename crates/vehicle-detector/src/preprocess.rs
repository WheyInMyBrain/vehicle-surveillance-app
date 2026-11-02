use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::Array4;

#[derive(Debug, Clone, Copy)]
pub struct LetterboxTransform {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub target_w: u32,
    pub target_h: u32,
}

impl LetterboxTransform {
    #[inline(always)]
    pub fn unproject_bbox(&self, x1: f32, y1: f32, x2: f32, y2: f32) -> [f32; 4] {
        let inv_scale = 1.0 / self.scale;

        let ox1 = ((x1 - self.pad_x) * inv_scale).max(0.0);
        let oy1 = ((y1 - self.pad_y) * inv_scale).max(0.0);
        let ox2 = ((x2 - self.pad_x) * inv_scale).max(0.0);
        let oy2 = ((y2 - self.pad_y) * inv_scale).max(0.0);

        [ox1.min(ox2), oy1.min(oy2), ox1.max(ox2), oy1.max(oy2)]
    }
}

/// In-place letterbox filling without intermediate canvas allocations.
pub fn fill_letterbox_tensor(
    img: &DynamicImage,
    target_w: u32,
    target_h: u32,
    tensor: &mut Array4<f32>,
) -> LetterboxTransform {
    let (orig_w, orig_h) = img.dimensions();

    let scale = (target_w as f32 / orig_w as f32).min(target_h as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale).round() as u32;
    let new_h = (orig_h as f32 * scale).round() as u32;

    let pad_x = ((target_w - new_w) as f32 * 0.5).floor();
    let pad_y = ((target_h - new_h) as f32 * 0.5).floor();

    let pad_x_usize = pad_x as usize;
    let pad_y_usize = pad_y as usize;
    let new_w_usize = new_w as usize;
    let new_h_usize = new_h as usize;
    let tw = target_w as usize;
    let th = target_h as usize;

    // Reset tensor to YOLO padding value (~0.447)
    tensor.fill(114.0 / 255.0);

    // Fast resize (Nearest is 5-8x faster than Triangle; use Triangle only if sub-pixel distant objects degrade)
    let resized = img.resize_exact(new_w, new_h, FilterType::Nearest);
    let rgb = resized.to_rgb8();
    let raw_pixels = rgb.as_raw();

    // Direct slice layout access: Planar NCHW [1, 3, target_h, target_w]
    // Channel strides:
    let plane_size = tw * th;
    let slice = tensor.as_slice_mut().expect("Tensor must be contiguous");

    let (c0, rest) = slice.split_at_mut(plane_size);
    let (c1, c2) = rest.split_at_mut(plane_size);

    let inv_255 = 1.0 / 255.0;

    for row in 0..new_h_usize {
        let src_row_offset = row * new_w_usize * 3;
        let dst_row_offset = (pad_y_usize + row) * tw + pad_x_usize;

        for col in 0..new_w_usize {
            let src_px = src_row_offset + col * 3;
            let dst_idx = dst_row_offset + col;

            c0[dst_idx] = raw_pixels[src_px] as f32 * inv_255;
            c1[dst_idx] = raw_pixels[src_px + 1] as f32 * inv_255;
            c2[dst_idx] = raw_pixels[src_px + 2] as f32 * inv_255;
        }
    }

    LetterboxTransform {
        scale,
        pad_x,
        pad_y,
        target_w,
        target_h,
    }
}

#[inline(always)]
pub fn calculate_iou_fast(box_a: &[f32; 4], area_a: f32, box_b: &[f32; 4], area_b: f32) -> f32 {
    let x1 = box_a[0].max(box_b[0]);
    let y1 = box_a[1].max(box_b[1]);
    let x2 = box_a[2].min(box_b[2]);
    let y2 = box_a[3].min(box_b[3]);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let union_area = area_a + area_b - inter_area;
    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}