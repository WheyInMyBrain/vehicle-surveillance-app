use heapless::Vec as StackVec;
use image::{DynamicImage, RgbImage};
use ndarray::Array4;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use vehicle_lpr::{
    fill_letterbox_tensor_direct, DetectedPlate, MAX_PLATE_PROPOSALS,
};

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
fn test_lpr_preprocessing_strictly_zero_allocations() {
    let dummy_crop = DynamicImage::ImageRgb8(RgbImage::new(320, 240));
    let mut tensor = Array4::<f32>::zeros((1, 3, 640, 640));

    // Warm-up
    let _ = fill_letterbox_tensor_direct(&dummy_crop, 640, 640, &mut tensor);

    // Reset allocation counter before hot path
    ALLOC_COUNT.store(0, Ordering::SeqCst);

    let _ = fill_letterbox_tensor_direct(&dummy_crop, 640, 640, &mut tensor)
        .expect("Letterbox direct failed");

    let mut proposals: StackVec<DetectedPlate, MAX_PLATE_PROPOSALS> = StackVec::new();
    assert_eq!(proposals.len(), 0);

    let allocs = ALLOC_COUNT.load(Ordering::SeqCst);
    println!("Total heap allocations during LPR pre-processing: {}", allocs);

    assert_eq!(allocs, 0, "LPR preprocessing must execute with zero heap allocations");
}