use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use vehicle_detector::{BoundingBox, DetectedVehicle, VehicleClass};
use vehicle_tracker::ByteTracker;

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
fn test_tracker_update_strictly_zero_allocations() {
    let mut tracker = ByteTracker::default();

    let sample_dets = [
        DetectedVehicle {
            class: VehicleClass::Car,
            confidence: 0.95,
            bbox: BoundingBox::new_clamped(100.0, 100.0, 200.0, 200.0),
        },
        DetectedVehicle {
            class: VehicleClass::TwoWheeler,
            confidence: 0.85,
            bbox: BoundingBox::new_clamped(250.0, 120.0, 290.0, 180.0),
        },
    ];

    // Warm-up step
    let _ = tracker.update(&sample_dets);

    // Reset allocation counter before hot path
    ALLOC_COUNT.store(0, Ordering::SeqCst);

    // Run 100 tracker ticks
    for _ in 0..100 {
        let _ = tracker.update(&sample_dets);
    }

    let allocs = ALLOC_COUNT.load(Ordering::SeqCst);
    println!("Total heap allocations across 100 tracker frames: {}", allocs);

    assert_eq!(allocs, 0, "Tracker hot loop must execute with strictly 0 heap allocations");
}