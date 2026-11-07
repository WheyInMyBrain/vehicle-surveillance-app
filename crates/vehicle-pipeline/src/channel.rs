//! Lock-free, zero-allocation communication channel between Front-End tracking and Back-End OCR.

use heapless::spsc::{Consumer, Producer, Queue};
use image::DynamicImage;
use vehicle_detector::BoundingBox;

/// Maximum number of queued candidate crops waiting for plate localization
pub const DISPATCH_QUEUE_CAPACITY: usize = 8;

/// Dispatch descriptor routed from the tracking thread to the LPR worker
pub struct LprTask {
    pub frame_idx: usize,
    pub track_id: u64,
    /// Bounding box normalized in [0.0, 1.0] relative space
    pub norm_bbox: BoundingBox,
    /// Full resolution image frame (4K still or video frame buffer)
    pub high_res_frame: DynamicImage,
}

pub type LprProducer = Producer<'static, LprTask>;
pub type LprConsumer = Consumer<'static, LprTask>;

/// Initializes static lock-free channel pointers
pub fn create_lpr_channel(
    queue: &'static mut Queue<LprTask, DISPATCH_QUEUE_CAPACITY>,
) -> (LprProducer, LprConsumer) {
    queue.split()
}