use heapless::String as StackString;
use vehicle_detector::BoundingBox;
use vehicle_lpocr::OcrResult;
use vehicle_pipeline::TemporalPlateVoter;

#[test]
fn test_temporal_consensus_majority_voting() {
    let mut voter = TemporalPlateVoter::new(101);

    // Observation 1: "MH12AB123A" (confidence 0.65)
    let mut obs1 = OcrResult::default();
    obs1.text = StackString::try_from("MH12AB123A").unwrap();
    obs1.confidence = 0.65;
    for ch in obs1.text.chars() {
        let _ = obs1.char_confidences.push((ch, 0.65));
    }
    voter.accumulate(&obs1);

    // Observation 2: "MH12AB1234" (confidence 0.85)
    let mut obs2 = OcrResult::default();
    obs2.text = StackString::try_from("MH12AB1234").unwrap();
    obs2.confidence = 0.85;
    for ch in obs2.text.chars() {
        let _ = obs2.char_confidences.push((ch, 0.85));
    }
    voter.accumulate(&obs2);

    // Consensus resolution
    let resolved = voter.resolve();
    assert_eq!(resolved.as_str(), "MH12AB1234");
    assert!(voter.is_consensus_reached());
}

#[test]
fn test_spatial_overlap_calculation() {
    let b1 = BoundingBox::new_clamped(0.0, 0.0, 100.0, 100.0); // Area = 10,000
    let b2 = BoundingBox::new_clamped(50.0, 0.0, 150.0, 100.0); // Overlap = 50 * 100 = 5,000 (50%)

    let x1 = b1.x1.max(b2.x1);
    let y1 = b1.y1.max(b2.y1);
    let x2 = b1.x2.min(b2.x2);
    let y2 = b1.y2.min(b2.y2);
    let inter_area = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let overlap_ratio = inter_area / b1.area();

    assert!((overlap_ratio - 0.50).abs() < 1e-4);
}