use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use image::{DynamicImage, RgbImage};
use vehicle_detector::{
    fill_letterbox_tensor_direct, parse_model_proposals, run_bounded_nms,
    DetectedVehicle, LetterboxTransform, VehicleDetector,
    VehicleDetectorConfig, MAX_NMS_CANDIDATES,
};
use heapless::Vec as StackVec;
use ndarray::Array4;
use std::path::PathBuf;

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

#[test]
fn test_core_pipeline_strictly_zero_allocations() {
    let dummy_frame = DynamicImage::ImageRgb8(RgbImage::new(640, 640));
    let mut tensor = Array4::<f32>::zeros((1, 3, 640, 640));

    // Warm up the image reference access and ndarray slice
    let _ = fill_letterbox_tensor_direct(&dummy_frame, 640, 640, &mut tensor);

    let dummy_dims = [1i64, 100, 6];
    let dummy_tensor_data = [0.0f32; 100 * 6];
    let mut proposals: StackVec<DetectedVehicle, MAX_NMS_CANDIDATES> = StackVec::new();
    let transform = LetterboxTransform { scale: 1.0, pad_x: 0.0, pad_y: 0.0 };

    // Reset allocation counter immediately before critical path
    ALLOC_COUNT.store(0, Ordering::SeqCst);

    // 1. In-place preprocessing (must be 0 allocations)
    let _ = fill_letterbox_tensor_direct(&dummy_frame, 640, 640, &mut tensor)
        .expect("Letterbox failed");
    let allocs_step1 = ALLOC_COUNT.load(Ordering::SeqCst);

    // 2. Proposal parsing on stack (must be 0 allocations)
    parse_model_proposals(
        &dummy_dims,
        &dummy_tensor_data,
        0.25,
        5.0,
        &transform,
        &mut proposals,
    )
    .expect("Parser failed");
    let allocs_step2 = ALLOC_COUNT.load(Ordering::SeqCst);

    // 3. Bounded NMS on stack (must be 0 allocations)
    let _ = run_bounded_nms(&mut proposals, 0.45);
    let allocs_step3 = ALLOC_COUNT.load(Ordering::SeqCst);

    println!(
        "Allocations breakdown -> Step1 (preprocess): {}, Step2 (parse): {}, Step3 (nms): {}",
        allocs_step1,
        allocs_step2 - allocs_step1,
        allocs_step3 - allocs_step2
    );

    assert_eq!(
        allocs_step3, 0,
        "Core preprocessing, parsing, and NMS must execute with zero heap allocations"
    );
}

#[test]
fn test_detector_inference_bounded_allocations() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let model_path = root.join("data/models/yolo26n.onnx");
    let model_path = if model_path.exists() {
        model_path
    } else {
        root.join("data/models/yolo26s.onnx")
    };

    if !model_path.exists() {
        return; // Skip if models not present
    }

    let config = VehicleDetectorConfig::default();
    let mut detector = VehicleDetector::new(&model_path, config).expect("Init failed");

    let dummy_frame = DynamicImage::ImageRgb8(RgbImage::new(640, 640));

    // Warm-up run
    let _ = detector.detect(&dummy_frame);

    // Measure frame 1
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    let _ = detector.detect(&dummy_frame);
    let frame1_allocs = ALLOC_COUNT.load(Ordering::SeqCst);

    // Measure frame 2
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    let _ = detector.detect(&dummy_frame);
    let frame2_allocs = ALLOC_COUNT.load(Ordering::SeqCst);

    println!("Frame 1 allocations: {}, Frame 2 allocations: {}", frame1_allocs, frame2_allocs);

    // Allocation count must be constant and deterministic
    assert_eq!(frame1_allocs, frame2_allocs, "Inference allocations must remain deterministic and bounded");
}