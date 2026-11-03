use image::{imageops::FilterType, DynamicImage};
use ndarray::Array4;

/// Prepares the NHWC [1, H, W, 3] tensor directly from the plate crop
pub fn prepare_ocr_tensor(
    img: &DynamicImage,
    target_w: u32,
    target_h: u32,
    tensor: &mut Array4<u8>,
) {
    let resized = img.resize_exact(target_w, target_h, FilterType::Triangle);
    let rgb = resized.to_rgb8();
    let raw = rgb.as_raw();

    // In NHWC layout, the raw RGB buffer ([H, W, 3]) matches the memory layout directly
    let slice = tensor.as_slice_mut().expect("Tensor must be contiguous");
    slice.copy_from_slice(raw);
}