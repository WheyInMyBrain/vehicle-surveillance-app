use vehicle_lpocr::{OcrResult, MAX_PLATE_CHARS};
use heapless::String as StackString;

#[test]
fn test_ocr_result_validity_bounds() {
    let mut res = OcrResult::default();
    assert_eq!(res.is_valid_plate(), false);

    // Below 4 characters -> Invalid
    res.text = StackString::try_from("MH1").unwrap();
    res.confidence = 0.95;
    assert_eq!(res.is_valid_plate(), false);

    // 4+ characters with high confidence -> Valid
    res.text = StackString::try_from("MH12AB1234").unwrap();
    res.confidence = 0.88;
    assert_eq!(res.is_valid_plate(), true);

    // Low confidence -> Invalid
    res.confidence = 0.20;
    assert_eq!(res.is_valid_plate(), false);
}

#[test]
fn test_stack_capacity_saturation() {
    let mut res = OcrResult::default();
    for _ in 0..MAX_PLATE_CHARS + 5 {
        if res.text.len() < MAX_PLATE_CHARS {
            let _ = res.text.push('A');
        }
    }
    assert_eq!(res.text.len(), MAX_PLATE_CHARS);
}