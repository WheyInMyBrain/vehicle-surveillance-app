//! Concrete ONNX Runtime detector implementation with deterministic execution.

use crate::core::nms::{run_bounded_nms, MAX_NMS_CANDIDATES, MAX_OUTPUT_DETECTIONS};
use crate::core::parser::parse_model_proposals;
use crate::core::types::DetectedVehicle;
use crate::preprocess::fill_letterbox_tensor_direct;

use heapless::Vec as StackVec;
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
pub enum DetectorError {
    #[error("ONNX Runtime error: {0}")]
    Ort(String),
    #[error("Tensor error: {0}")]
    Tensor(&'static str),
    #[error("Invalid model output dimensions: {0:?}")]
    InvalidOutputShape(Vec<i64>),
}

impl<T> From<ort::Error<T>> for DetectorError {
    fn from(err: ort::Error<T>) -> Self {
        DetectorError::Ort(err.to_string())
    }
}

static ORT_ENV: OnceLock<Result<Arc<Environment>, String>> = OnceLock::new();

pub fn get_or_init_ort_env() -> Result<Arc<Environment>, DetectorError> {
    let env_res = ORT_ENV.get_or_init(|| {
        ort::init()
            .with_name("surveillance_detector")
            .build()
            .map(Arc::new)
            .map_err(|e| e.to_string())
    });

    match env_res {
        Ok(env) => Ok(Arc::clone(env)),
        Err(e) => Err(DetectorError::Ort(e.clone())),
    }
}

#[derive(Debug, Clone)]
pub struct VehicleDetectorConfig {
    pub confidence_threshold: f32,
    pub nms_iou_threshold: f32,
    pub input_width: u32,
    pub input_height: u32,
    pub min_box_dimension: f32,
    pub num_threads: usize,
}

impl Default for VehicleDetectorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.25,
            nms_iou_threshold: 0.45,
            input_width: 640,
            input_height: 640,
            min_box_dimension: 8.0,
            num_threads: 4,
        }
    }
}

pub struct VehicleDetector {
    _env: Arc<Environment>,
    session: Session,
    config: VehicleDetectorConfig,
    input_buffer: Array4<f32>,
    proposals_buffer: StackVec<DetectedVehicle, MAX_NMS_CANDIDATES>,
}

impl VehicleDetector {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        config: VehicleDetectorConfig,
    ) -> Result<Self, DetectorError> {
        let env = get_or_init_ort_env()?;
        Self::new_with_env(model_path, config, env)
    }

    pub fn new_with_env<P: AsRef<Path>>(
        model_path: P,
        config: VehicleDetectorConfig,
        env: Arc<Environment>,
    ) -> Result<Self, DetectorError> {
        let session = Session::builder(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(config.num_threads)?
            .commit_from_file(model_path)?;

        let h = config.input_height as usize;
        let w = config.input_width as usize;

        let input_buffer = Array4::<f32>::zeros((1, 3, h, w));

        Ok(Self {
            _env: env,
            session,
            config,
            input_buffer,
            proposals_buffer: StackVec::new(),
        })
    }

    pub fn environment(&self) -> Arc<Environment> {
        Arc::clone(&self._env)
    }

    pub fn config(&self) -> &VehicleDetectorConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: VehicleDetectorConfig) {
        if self.config.input_width != config.input_width
            || self.config.input_height != config.input_height
        {
            let h = config.input_height as usize;
            let w = config.input_width as usize;
            self.input_buffer = Array4::<f32>::zeros((1, 3, h, w));
        }
        self.config = config;
    }

    /// Performs vehicle detection with deterministic stack-allocated proposal and NMS logic.
    pub fn detect(
        &mut self,
        img: &DynamicImage,
    ) -> Result<StackVec<DetectedVehicle, MAX_OUTPUT_DETECTIONS>, DetectorError> {
        let transform = fill_letterbox_tensor_direct(
            img,
            self.config.input_width,
            self.config.input_height,
            &mut self.input_buffer,
        )
        .map_err(DetectorError::Tensor)?;

        // Transfer into value for ORT session runner
        let input_val = Value::from_array(self.input_buffer.clone())?;

        let outputs = self.session.run(ort::inputs![input_val])?;

        let (out_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| DetectorError::Ort(e.to_string()))?;

        let dims: &[i64] = out_shape.as_ref();

        parse_model_proposals(
            dims,
            data,
            self.config.confidence_threshold,
            self.config.min_box_dimension,
            &transform,
            &mut self.proposals_buffer,
        )
        .map_err(|_| DetectorError::InvalidOutputShape(dims.to_vec()))?;

        // Zero-allocation stack NMS
        let final_detections = run_bounded_nms(
            &mut self.proposals_buffer,
            self.config.nms_iou_threshold,
        );

        Ok(final_detections)
    }
}