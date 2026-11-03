#[derive(Debug, Clone, PartialEq)]
pub struct OcrResult {
    /// Extracted plate text with pad tokens removed (e.g. "MH12AB1234")
    pub text: String,
    /// Average confidence across all decoded non-pad characters (0.0 to 1.0)
    pub confidence: f32,
    /// Individual character predictions and their respective Softmax confidences
    pub char_confidences: Vec<(char, f32)>,
}

impl OcrResult {
    pub fn is_valid_plate(&self) -> bool {
        // Most automotive plates have between 4 and 10 characters
        self.text.len() >= 4 && self.confidence >= 0.30
    }
}