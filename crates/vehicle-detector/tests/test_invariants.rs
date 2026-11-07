use heapless::Vec as StackVec;
use vehicle_detector::{
    calculate_iou_fast, parse_model_proposals, run_bounded_nms,
    BoundingBox, DetectedVehicle, LetterboxTransform, VehicleClass,
    MAX_NMS_CANDIDATES, MAX_OUTPUT_DETECTIONS,
};

#[test]
fn test_bounding_box_invariants_nan_inf_handling() {
    // Aerospace invariant: NaN or Inf coordinates must never panic or break ordering
    let bbox_nan = BoundingBox::new_clamped(f32::NAN, 10.0, 50.0, f32::NAN);
    assert!(!bbox_nan.x1.is_nan());
    assert!(!bbox_nan.y2.is_nan());
    assert!(bbox_nan.x1 <= bbox_nan.x2);
    assert!(bbox_nan.y1 <= bbox_nan.y2);

    // Inverted coordinates must normalize without crashing
    let bbox_inv = BoundingBox::new_clamped(100.0, 200.0, 20.0, 30.0);
    assert_eq!(bbox_inv.x1, 20.0);
    assert_eq!(bbox_inv.x2, 100.0);
    assert_eq!(bbox_inv.y1, 30.0);
    assert_eq!(bbox_inv.y2, 200.0);
    assert_eq!(bbox_inv.width(), 80.0);
    assert_eq!(bbox_inv.height(), 170.0);
}

#[test]
fn test_iou_mathematical_bounds() {
    let b1 = BoundingBox::new_clamped(0.0, 0.0, 10.0, 10.0);
    let b2 = BoundingBox::new_clamped(0.0, 0.0, 10.0, 10.0);
    
    // Identical boxes must have IoU == 1.0
    let iou_identical = calculate_iou_fast(&b1, b1.area(), &b2, b2.area());
    assert!((iou_identical - 1.0).abs() < 1e-5);

    // Non-overlapping boxes must have IoU == 0.0
    let b_disjoint = BoundingBox::new_clamped(20.0, 20.0, 30.0, 30.0);
    let iou_disjoint = calculate_iou_fast(&b1, b1.area(), &b_disjoint, b_disjoint.area());
    assert_eq!(iou_disjoint, 0.0);

    // Degenerate zero-area boxes must not trigger divide-by-zero or return NaN
    let b_zero = BoundingBox::new_clamped(5.0, 5.0, 5.0, 5.0);
    let iou_zero = calculate_iou_fast(&b_zero, b_zero.area(), &b_zero, b_zero.area());
    assert!(!iou_zero.is_nan());
    assert_eq!(iou_zero, 0.0);
}

#[test]
fn test_nms_bounded_saturation() {
    let mut proposals: StackVec<DetectedVehicle, MAX_NMS_CANDIDATES> = StackVec::new();

    // Saturate proposals buffer with overlapping boxes
    for i in 0..MAX_NMS_CANDIDATES {
        let _ = proposals.push(DetectedVehicle {
            class: VehicleClass::Car,
            confidence: 0.5 + (i as f32 * 0.001),
            bbox: BoundingBox::new_clamped(0.0, 0.0, 50.0, 50.0),
        });
    }

    assert_eq!(proposals.len(), MAX_NMS_CANDIDATES);

    // NMS must suppress all duplicates down to a single box without heap allocation
    let suppressed = run_bounded_nms(&mut proposals, 0.45);
    assert_eq!(suppressed.len(), 1);
}

#[test]
fn test_nms_maximum_output_capacity() {
    let mut proposals: StackVec<DetectedVehicle, MAX_NMS_CANDIDATES> = StackVec::new();

    // Populate with 100 non-overlapping distinct boxes
    for i in 0..100 {
        let offset = (i as f32) * 100.0;
        let _ = proposals.push(DetectedVehicle {
            class: VehicleClass::Car,
            confidence: 0.90,
            bbox: BoundingBox::new_clamped(offset, 0.0, offset + 50.0, 50.0),
        });
    }

    let output = run_bounded_nms(&mut proposals, 0.45);
    // Must clamp deterministically to MAX_OUTPUT_DETECTIONS without panic
    assert_eq!(output.len(), MAX_OUTPUT_DETECTIONS);
}

#[test]
fn test_tensor_parser_adversarial_dims_and_truncation() {
    let transform = LetterboxTransform { scale: 1.0, pad_x: 0.0, pad_y: 0.0 };
    let mut out: StackVec<DetectedVehicle, MAX_NMS_CANDIDATES> = StackVec::new();

    // Test truncated buffer (data slice shorter than shape promises)
    let bad_dims = [1, 100, 6];
    let truncated_data = vec![0.0f32; 10]; // Requires 600 floats

    let res = parse_model_proposals(&bad_dims, &truncated_data, 0.25, 5.0, &transform, &mut out);
    // Must return Ok without panicking on out-of-bounds reads
    assert!(res.is_ok());

    // Test invalid batch dimension
    let bad_batch_dims = [2, 10, 6];
    let res_bad_batch = parse_model_proposals(&bad_batch_dims, &truncated_data, 0.25, 5.0, &transform, &mut out);
    assert!(res_bad_batch.is_err());
}