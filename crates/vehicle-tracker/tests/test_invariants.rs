
use vehicle_detector::{BoundingBox, DetectedVehicle, VehicleClass};
use vehicle_tracker::{
    calculate_diou, invert_4x4_analytic, lapjv_bounded, ByteTracker, KalmanFilter,
    MAX_MATCH_DIM,
};

#[test]
fn test_kalman_exact_analytic_matrix_inversion() {
    // Non-trivial symmetric diagonally-dominant covariance matrix
    #[rustfmt::skip]
    let m = [
        4.0, 1.0, 0.5, 0.2,
        1.0, 5.0, 0.3, 0.1,
        0.5, 0.3, 3.0, 0.4,
        0.2, 0.1, 0.4, 6.0,
    ];

    let inv = invert_4x4_analytic(&m);

    // Verify identity M * M_inv = I
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0.0f32;
            for k in 0..4 {
                sum += m[i * 4 + k] * inv[k * 4 + j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (sum - expected).abs() < 1e-4,
                "Matrix inversion failed at ({}, {}): got {}, expected {}",
                i, j, sum, expected
            );
        }
    }
}

#[test]
fn test_lapjv_bounded_bipartite_optimality() {
    // 3 tracks, 3 detections with distinct costs
    #[rustfmt::skip]
    let cost_matrix = [
        0.1, 0.9, 0.8,
        0.7, 0.2, 0.6,
        0.9, 0.5, 0.15,
    ];

    let res = lapjv_bounded(&cost_matrix, 3, 3, 0.5);

    // Must match: (0 -> 0), (1 -> 1), (2 -> 2)
    assert_eq!(res.matches.len(), 3);
    assert_eq!(res.matches[0], (0, 0));
    assert_eq!(res.matches[1], (1, 1));
    assert_eq!(res.matches[2], (2, 2));
    assert!(res.unmatched_rows.is_empty());
    assert!(res.unmatched_cols.is_empty());
}

#[test]
fn test_diou_mathematical_bounds() {
    let b1 = BoundingBox::new_clamped(10.0, 10.0, 50.0, 50.0);
    let b2 = BoundingBox::new_clamped(10.0, 10.0, 50.0, 50.0);

    // Identical boxes must have DIoU == 1.0
    let diou = calculate_diou(&b1, &b2);
    assert!((diou - 1.0).abs() < 1e-5);

    // Distant boxes must have DIoU < 0.0
    let b_far = BoundingBox::new_clamped(500.0, 500.0, 550.0, 550.0);
    let diou_far = calculate_diou(&b1, &b_far);
    assert!(diou_far < 0.0);
}

#[test]
fn test_tracker_zero_nan_saturation() {
    let mut tracker = ByteTracker::default();

    // Send a frame of 10 detections
    let dets = (0..10)
        .map(|i| DetectedVehicle {
            class: VehicleClass::Car,
            confidence: 0.90,
            bbox: BoundingBox::new_clamped(
                (i as f32) * 50.0,
                100.0,
                (i as f32) * 50.0 + 40.0,
                140.0,
            ),
        })
        .collect::<Vec<_>>();

    for _ in 0..10 {
        let tracked = tracker.update(&dets);
        for t in &tracked {
            assert!(!t.x1().is_nan());
            assert!(!t.y1().is_nan());
            assert!(t.track_id > 0);
        }
    }
}