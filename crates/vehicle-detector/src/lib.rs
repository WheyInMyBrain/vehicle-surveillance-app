pub mod core;

#[cfg(feature = "std")]
pub mod detector;
#[cfg(feature = "std")]
pub mod preprocess;

pub use crate::core::nms::{calculate_iou_fast, run_bounded_nms, MAX_NMS_CANDIDATES, MAX_OUTPUT_DETECTIONS};
pub use crate::core::parser::{parse_model_proposals, LetterboxTransform};
pub use crate::core::types::{BoundingBox, DetectedVehicle, VehicleClass};

#[cfg(feature = "std")]
pub use detector::{DetectorError, VehicleDetector, VehicleDetectorConfig};
#[cfg(feature = "std")]
pub use preprocess::fill_letterbox_tensor_direct;