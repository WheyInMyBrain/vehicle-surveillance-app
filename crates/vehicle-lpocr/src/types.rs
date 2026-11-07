//! Safety-critical domain primitives for License Plate OCR.

use core::fmt;
use heapless::{String as StackString, Vec as StackVec};

pub const MAX_PLATE_CHARS: usize = 16;

/// Bounded License Plate OCR prediction with character-level confidences.
#[derive(Clone, PartialEq)]
pub struct OcrResult {
    /// Extracted plate text with pad tokens removed (e.g. "MH12AB1234").
    pub text: StackString<MAX_PLATE_CHARS>,
    /// Average confidence across all decoded non-pad characters (0.0 to 1.0).
    pub confidence: f32,
    /// Individual character predictions and their respective Softmax confidences.
    pub char_confidences: StackVec<(char, f32), MAX_PLATE_CHARS>,
}

impl Default for OcrResult {
    fn default() -> Self {
        Self {
            text: StackString::new(),
            confidence: 0.0,
            char_confidences: StackVec::new(),
        }
    }
}

impl fmt::Debug for OcrResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OcrResult")
            .field("text", &self.text.as_str())
            .field("confidence", &self.confidence)
            .field("char_count", &self.char_confidences.len())
            .finish()
    }
}

impl OcrResult {
    /// Automotive license plates strictly adhere to standardized syntax lengths.
    #[inline(always)]
    pub fn is_valid_plate(&self) -> bool {
        self.text.len() >= 4 && self.confidence >= 0.30
    }

    #[inline(always)]
    pub fn as_str(&self) -> &str {
        self.text.as_str()
    }
}