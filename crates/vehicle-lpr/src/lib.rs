pub mod detector;
pub mod preprocess;
pub mod types;

pub use detector::{PlateDetector, PlateDetectorConfig, PlateDetectorError};
pub use preprocess::{calculate_iou_fast, fill_letterbox_tensor, LetterboxTransform};
pub use types::DetectedPlate;