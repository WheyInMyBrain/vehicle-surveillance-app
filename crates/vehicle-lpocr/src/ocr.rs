use crate::preprocess::prepare_ocr_tensor;
use crate::types::OcrResult;
use image::DynamicImage;
use ndarray::Array4;
use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::{Arc, OnceLock};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OcrError {
    #[error("ONNX Runtime error: {0}")]
    Ort(String),
    #[error("Invalid OCR model output tensor shape: {0:?}")]
    InvalidOutputShape(Vec<i64>),
}

impl From<ort::Error> for OcrError {
    fn from(err: ort::Error) -> Self {
        OcrError::Ort(err.to_string())
    }
}

static ORT_ENV: OnceLock<Result<Arc<Environment>, String>> = OnceLock::new();

fn get_or_init_ort_env() -> Result<Arc<Environment>, OcrError> {
    let env_res = ORT_ENV.get_or_init(|| {
        ort::init()
            .with_name("surveillance_env")
            .build()
            .map(Arc::new)
            .map_err(|e| e.to_string())
    });

    match env_res {
        Ok(env) => Ok(Arc::clone(env)),
        Err(e) => Err(OcrError::Ort(e.clone())),
    }
}

#[derive(Debug, Clone)]
pub struct PlateOcrConfig {
    pub img_width: u32,
    pub img_height: u32,
    pub alphabet: &'static str,
    pub pad_char: char,
    pub max_plate_slots: usize,
    pub num_threads: usize,
}

impl Default for PlateOcrConfig {
    fn default() -> Self {
        Self {
            img_width: 128,
            img_height: 64,
            alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_",
            pad_char: '_',
            max_plate_slots: 10,
            num_threads: 2,
        }
    }
}

pub struct PlateOcrEngine {
    _env: Arc<Environment>,
    session: Session,
    config: PlateOcrConfig,
    alphabet_chars: Vec<char>,
    input_buffer: Array4<u8>,
}

impl PlateOcrEngine {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        config: PlateOcrConfig,
    ) -> Result<Self, OcrError> {
        let env = get_or_init_ort_env()?;
        Self::new_with_env(model_path, config, env)
    }

    pub fn new_with_env<P: AsRef<Path>>(
        model_path: P,
        config: PlateOcrConfig,
        env: Arc<Environment>,
    ) -> Result<Self, OcrError> {
        let session = Session::builder(&env)
            .map_err(|e| OcrError::Ort(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| OcrError::Ort(e.to_string()))?
            .with_intra_threads(config.num_threads)
            .map_err(|e| OcrError::Ort(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| OcrError::Ort(e.to_string()))?;

        // NHWC tensor: [1, 64, 128, 3]
        let input_buffer = Array4::<u8>::zeros((
            1,
            config.img_height as usize,
            config.img_width as usize,
            3,
        ));

        let alphabet_chars = config.alphabet.chars().collect();

        Ok(Self {
            _env: env,
            session,
            config,
            alphabet_chars,
            input_buffer,
        })
    }

    pub fn recognize(&mut self, plate_crop: &DynamicImage) -> Result<OcrResult, OcrError> {
        prepare_ocr_tensor(
            plate_crop,
            self.config.img_width,
            self.config.img_height,
            &mut self.input_buffer,
        );

        let input_val = Value::from_array(self.input_buffer.clone())
            .map_err(|e| OcrError::Ort(e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs![input_val])
            .map_err(|e| OcrError::Ort(e.to_string()))?;

        let (out_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| OcrError::Ort(e.to_string()))?;

        let dims: &[i64] = out_shape.as_ref();

        if dims.len() != 3 {
            return Err(OcrError::InvalidOutputShape(dims.to_vec()));
        }

        let num_slots = dims[1] as usize;
        let vocab_size = dims[2] as usize;

        let mut recognized_text = String::with_capacity(num_slots);
        let mut char_confidences = Vec::with_capacity(num_slots);
        let mut total_confidence = 0.0f32;
        let mut char_count = 0usize;

        for slot in 0..num_slots {
            let offset = slot * vocab_size;
            let slot_logits = &data[offset..offset + vocab_size];

            let mut max_logit = f32::NEG_INFINITY;
            for &logit in slot_logits {
                if logit > max_logit {
                    max_logit = logit;
                }
            }

            let mut sum_exp = 0.0f32;
            let mut best_idx = 0;
            let mut best_val = f32::NEG_INFINITY;

            for (idx, &logit) in slot_logits.iter().enumerate() {
                let exp = (logit - max_logit).exp();
                sum_exp += exp;
                if logit > best_val {
                    best_val = logit;
                    best_idx = idx;
                }
            }

            let prob = if sum_exp > 0.0 {
                (best_val - max_logit).exp() / sum_exp
            } else {
                0.0
            };

            let ch = self.alphabet_chars.get(best_idx).copied().unwrap_or(self.config.pad_char);

            if ch != self.config.pad_char {
                recognized_text.push(ch);
                char_confidences.push((ch, prob));
                total_confidence += prob;
                char_count += 1;
            }
        }

        let avg_confidence = if char_count > 0 {
            total_confidence / char_count as f32
        } else {
            0.0
        };

        Ok(OcrResult {
            text: recognized_text,
            confidence: avg_confidence,
            char_confidences,
        })
    }
}