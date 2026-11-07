use vehicle_detector::BoundingBox;
use vehicle_lpr::{calculate_iou_fast, DetectedPlate};

#[test]
fn test_plate_aspect_ratio_stability() {
    let bbox = BoundingBox::new_clamped(10.0, 10.0, 110.0, 35.0); // 100x25 -> aspect 4.0
    let plate = DetectedPlate {
        confidence: 0.95,
        bbox,
    };

    assert!((plate.aspect_ratio() - 4.0).abs() < 1e-4);

    // Degenerate height must not divide by zero
    let zero_h_bbox = BoundingBox::new_clamped(10.0, 10.0, 110.0, 10.0);
    let zero_plate = DetectedPlate {
        confidence: 0.95,
        bbox: zero_h_bbox,
    };
    assert_eq!(zero_plate.aspect_ratio(), 0.0);
}

#[test]
fn test_plate_iou_exactness() {
    let b1 = BoundingBox::new_clamped(0.0, 0.0, 50.0, 20.0);
    let b2 = BoundingBox::new_clamped(0.0, 0.0, 50.0, 20.0);

    let iou = calculate_iou_fast(&b1, b1.area(), &b2, b2.area());
    assert!((iou - 1.0).abs() < 1e-5);
}