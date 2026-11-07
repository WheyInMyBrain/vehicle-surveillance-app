use image::{DynamicImage, RgbImage};
use ndarray::Array4;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use vehicle_lpocr::prepare_ocr_tensor_direct;

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
fn test_ocr_tensor_preprocessing_strictly_zero_allocations() {
    let dummy_plate = DynamicImage::ImageRgb8(RgbImage::new(128, 64));
    let mut tensor = Array4::<u8>::zeros((1, 64, 128, 3));

    // Warm-up
    let _ = prepare_ocr_tensor_direct(&dummy_plate, 128, 64, &mut tensor);

    // Reset allocation counter before hot path
    ALLOC_COUNT.store(0, Ordering::SeqCst);

    for _ in 0..10 {
        let _ = prepare_ocr_tensor_direct(&dummy_plate, 128, 64, &mut tensor)
            .expect("OCR preprocess failed");
    }

    let allocs = ALLOC_COUNT.load(Ordering::SeqCst);
    println!("Total heap allocations across 10 OCR tensor preparation cycles: {}", allocs);

    assert_eq!(allocs, 0, "OCR tensor preparation must execute with zero heap allocations");
}