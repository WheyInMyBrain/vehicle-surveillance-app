//! Safety-critical temporal voting engine for multi-frame license plate consensus.
//!
//! # Guarantees
//! - Zero dynamic heap allocation.
//! - Bounded state arrays per vehicle track.
//! - Probability-weighted character frequency voting.

use vehicle_lpocr::{OcrResult, MAX_PLATE_CHARS};

pub const VOCAB_SIZE: usize = 37; // 0-9, A-Z, _
const ALPHABET: &[u8; VOCAB_SIZE] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";

/// Bounded multi-pass accumulator for an active vehicle track.
#[derive(Debug, Clone, Copy)]
pub struct TemporalPlateVoter {
    /// Track ID associated with this voter
    pub track_id: u64,
    /// Probability weight accumulator per character slot: [slot][vocab_idx]
    logits_table: [[f32; VOCAB_SIZE]; MAX_PLATE_CHARS],
    /// Number of valid OCR observations incorporated
    pub observation_count: usize,
    /// Highest single-frame confidence seen so far
    pub peak_single_conf: f32,
}

impl TemporalPlateVoter {
    pub const fn new(track_id: u64) -> Self {
        Self {
            track_id,
            logits_table: [[0.0f32; VOCAB_SIZE]; MAX_PLATE_CHARS],
            observation_count: 0,
            peak_single_conf: 0.0,
        }
    }

    pub fn reset(&mut self, track_id: u64) {
        self.track_id = track_id;
        self.logits_table = [[0.0f32; VOCAB_SIZE]; MAX_PLATE_CHARS];
        self.observation_count = 0;
        self.peak_single_conf = 0.0;
    }

    /// Incorporates a single-frame OCR observation into the temporal consensus.
    pub fn accumulate(&mut self, observation: &OcrResult) {
        if !observation.is_valid_plate() {
            return;
        }

        self.observation_count += 1;
        if observation.confidence > self.peak_single_conf {
            self.peak_single_conf = observation.confidence;
        }

        let weight = observation.confidence;

        for (slot, &(ch, conf)) in observation.char_confidences.iter().enumerate() {
            if slot >= MAX_PLATE_CHARS {
                break;
            }

            if let Some(vocab_idx) = char_to_vocab_idx(ch) {
                self.logits_table[slot][vocab_idx] += conf * weight;
            }
        }
    }

    /// Resolves the current consensus plate string and composite confidence.
    pub fn resolve(&self) -> OcrResult {
        let mut result = OcrResult::default();
        if self.observation_count == 0 {
            return result;
        }

        let mut total_score = 0.0f32;
        let mut resolved_chars = 0usize;

        for slot in 0..MAX_PLATE_CHARS {
            let mut max_score = 0.0f32;
            let mut best_vocab = VOCAB_SIZE - 1; // Default to pad '_'

            for vocab_idx in 0..VOCAB_SIZE {
                let score = self.logits_table[slot][vocab_idx];
                if score > max_score {
                    max_score = score;
                    best_vocab = vocab_idx;
                }
            }

            let ch = ALPHABET[best_vocab] as char;

            // Stop at pad character or empty slot
            if ch == '_' || max_score <= 1e-4 {
                break;
            }

            let slot_norm_confidence = (max_score / self.observation_count as f32).min(1.0);
            let _ = result.text.push(ch);
            let _ = result.char_confidences.push((ch, slot_norm_confidence));
            total_score += slot_norm_confidence;
            resolved_chars += 1;
        }

        result.confidence = if resolved_chars > 0 {
            total_score / resolved_chars as f32
        } else {
            0.0
        };

        result
    }

    /// Binary criteria confirming the plate is finalized and requires no further reads.
    #[inline(always)]
    pub fn is_consensus_reached(&self) -> bool {
        (self.observation_count >= 2 && self.resolve().confidence >= 0.50)
            || (self.observation_count >= 1 && self.peak_single_conf >= 0.85)
    }
}

#[inline(always)]
fn char_to_vocab_idx(c: char) -> Option<usize> {
    match c {
        '0'..='9' => Some((c as u8 - b'0') as usize),
        'A'..='Z' => Some((c as u8 - b'A' + 10) as usize),
        'a'..='z' => Some((c as u8 - b'a' + 10) as usize), // Handle lowercase normalization
        '_' => Some(36),
        _ => None,
    }
}