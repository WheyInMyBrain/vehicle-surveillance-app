//! Aerospace-grade License Plate Recognition (LPR) Detection Crate.
//!
//! Provides deterministic, zero-allocation license plate detection and NMS.

pub mod detector;
pub mod preprocess;
pub mod types;

pub use detector::{PlateDetector, PlateDetectorConfig, PlateDetectorError};
pub use preprocess::{calculate_iou_fast, fill_letterbox_tensor_direct};
pub use types::{DetectedPlate, MAX_OUTPUT_PLATES, MAX_PLATE_PROPOSALS};