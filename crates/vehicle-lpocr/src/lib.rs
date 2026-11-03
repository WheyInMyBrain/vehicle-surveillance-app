pub mod ocr;
pub mod preprocess;
pub mod types;

pub use ocr::{PlateOcrConfig, PlateOcrEngine, OcrError};
pub use types::OcrResult;