use vehicle_detector::BoundingBox;
use vehicle_quality::{
    compute_laplacian_variance_direct, compute_luma_statistics_direct, rgb_to_luma, QualityAuditor,
};

#[test]
fn test_luma_formula_bounds() {
    // Pure black: 0
    assert_eq!(rgb_to_luma(0, 0, 0), 0);
    // Pure white: 255
    assert_eq!(rgb_to_luma(255, 255, 255), 255);
}

#[test]
fn test_flat_patch_zero_sharpness() {
    // A completely uniform 100x100 RGB patch has 0.0 Laplacian variance
    let w = 100u32;
    let h = 100u32;
    let flat_rgb = vec![128u8; (w * h * 3) as usize];
    let bbox = BoundingBox::new_clamped(10.0, 10.0, 80.0, 80.0);

    let sharpness = compute_laplacian_variance_direct(&flat_rgb, w, h, &bbox);
    assert_eq!(sharpness, 0.0, "Flat patch must yield zero Laplacian variance");

    let (mean_luma, contrast) = compute_luma_statistics_direct(&flat_rgb, w, h, &bbox);
    assert!((mean_luma - 128.0).abs() < 1.0);
    assert_eq!(contrast, 0.0);
}

#[test]
fn test_degenerate_bounding_box_handling() {
    let raw = vec![255u8; 64 * 64 * 3];
    let auditor = QualityAuditor::default();

    // Zero-width box
    let b_zero = BoundingBox::new_clamped(10.0, 10.0, 10.0, 50.0);
    let q = auditor.evaluate_direct(&raw, 64, 64, &b_zero);
    assert_eq!(q.is_ocr_viable, false);
    assert_eq!(q.composite_score, 0.0);
}