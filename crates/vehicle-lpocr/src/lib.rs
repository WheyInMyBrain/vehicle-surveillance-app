//! Aerospace-grade License Plate Optical Character Recognition (LP-OCR) Crate.
//!
//! Provides deterministic, zero-allocation sequence recognition and CTC decoding.

pub mod ocr;
pub mod preprocess;
pub mod types;

pub use ocr::{get_or_init_ort_env, OcrError, PlateOcrConfig, PlateOcrEngine};
pub use preprocess::prepare_ocr_tensor_direct;
pub use types::{OcrResult, MAX_PLATE_CHARS};