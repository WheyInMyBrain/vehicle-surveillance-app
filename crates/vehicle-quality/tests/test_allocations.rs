use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use vehicle_detector::BoundingBox;
use vehicle_quality::{QualityAuditor, QualityConfig};

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
fn test_quality_auditor_zero_allocations() {
    let w = 640u32;
    let h = 480u32;
    let raw_rgb = vec![128u8; (w * h * 3) as usize];
    let bbox = BoundingBox::new_clamped(50.0, 50.0, 200.0, 200.0);
    let auditor = QualityAuditor::new(QualityConfig::default());

    // Reset allocation counter before hot evaluation
    ALLOC_COUNT.store(0, Ordering::SeqCst);

    for _ in 0..100 {
        let breakdown = auditor.evaluate_direct(&raw_rgb, w, h, &bbox);
        assert!(!breakdown.sharpness.is_nan());
    }

    let allocs = ALLOC_COUNT.load(Ordering::SeqCst);
    println!("Total allocations across 100 quality evaluations: {}", allocs);

    assert_eq!(allocs, 0, "Direct optical evaluation must execute with strictly zero heap allocations");
}