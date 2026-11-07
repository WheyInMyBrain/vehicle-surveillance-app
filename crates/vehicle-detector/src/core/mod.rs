pub mod nms;
pub mod parser;
pub mod types;

pub use nms::{calculate_iou_fast, run_bounded_nms, MAX_NMS_CANDIDATES, MAX_OUTPUT_DETECTIONS};
pub use parser::{parse_model_proposals, LetterboxTransform};
pub use types::{BoundingBox, DetectedVehicle, VehicleClass};