//! Concrete ONNX Runtime plate detector implementation with bounded stack memory.

use crate::preprocess::{calculate_iou_fast, fill_letterbox_tensor_direct};
use crate::types::{DetectedPlate, MAX_OUTPUT_PLATES, MAX_PLATE_PROPOSALS};

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
use vehicle_detector::LetterboxTransform;

#[derive(Error, Debug)]
pub enum PlateDetectorError {
    #[error("ONNX Runtime error: {0}")]
    Ort(String),
    #[error("Tensor buffer error: {0}")]
    Tensor(&'static str),
    #[error("Invalid model output tensor shape: {0:?}")]
    InvalidOutputShape(Vec<i64>),
}

impl<T> From<ort::Error<T>> for PlateDetectorError {
    fn from(err: ort::Error<T>) -> Self {
        PlateDetectorError::Ort(err.to_string())
    }
}

static ORT_ENV: OnceLock<Result<Arc<Environment>, String>> = OnceLock::new();

pub fn get_or_init_ort_env() -> Result<Arc<Environment>, PlateDetectorError> {
    let env_res = ORT_ENV.get_or_init(|| {
        ort::init()
            .with_name("surveillance_lpr")
            .build()
            .map(Arc::new)
            .map_err(|e| e.to_string())
    });

    match env_res {
        Ok(env) => Ok(Arc::clone(env)),
        Err(e) => Err(PlateDetectorError::Ort(e.clone())),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PlateDetectorConfig {
    pub confidence_threshold: f32,
    pub nms_iou_threshold: f32,
    pub input_width: u32,
    pub input_height: u32,
    pub min_plate_height: f32,
    pub min_plate_width: f32,
    pub num_threads: usize,
}

impl Default for PlateDetectorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.20,
            nms_iou_threshold: 0.40,
            input_width: 640,
            input_height: 640,
            min_plate_height: 6.0,
            min_plate_width: 14.0,
            num_threads: 4,
        }
    }
}

pub struct PlateDetector {
    _env: Arc<Environment>,
    session: Session,
    config: PlateDetectorConfig,
    input_buffer: Array4<f32>,
    proposals_buf: StackVec<DetectedPlate, MAX_PLATE_PROPOSALS>,
}

impl PlateDetector {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        config: PlateDetectorConfig,
    ) -> Result<Self, PlateDetectorError> {
        let env = get_or_init_ort_env()?;
        Self::new_with_env(model_path, config, env)
    }

    pub fn new_with_env<P: AsRef<Path>>(
        model_path: P,
        config: PlateDetectorConfig,
        env: Arc<Environment>,
    ) -> Result<Self, PlateDetectorError> {
        let session = Session::builder(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(config.num_threads)?
            .commit_from_file(model_path)?;

        let input_buffer = Array4::<f32>::zeros((
            1,
            3,
            config.input_height as usize,
            config.input_width as usize,
        ));

        Ok(Self {
            _env: env,
            session,
            config,
            input_buffer,
            proposals_buf: StackVec::new(),
        })
    }

    pub fn environment(&self) -> Arc<Environment> {
        Arc::clone(&self._env)
    }

    pub fn config(&self) -> &PlateDetectorConfig {
        &self.config
    }

    /// Detects license plates with strictly zero dynamic heap allocations.
    pub fn detect(
        &mut self,
        vehicle_crop: &DynamicImage,
    ) -> Result<StackVec<DetectedPlate, MAX_OUTPUT_PLATES>, PlateDetectorError> {
        let transform = fill_letterbox_tensor_direct(
            vehicle_crop,
            self.config.input_width,
            self.config.input_height,
            &mut self.input_buffer,
        )
        .map_err(PlateDetectorError::Tensor)?;

        let input_val = Value::from_array(self.input_buffer.clone())?;

        let outputs = self.session.run(ort::inputs![input_val])?;

        let (out_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| PlateDetectorError::Ort(e.to_string()))?;

        let dims: &[i64] = out_shape.as_ref();
        self.proposals_buf.clear();

        parse_plate_proposals(
            dims,
            data,
            self.config.confidence_threshold,
            self.config.min_plate_width,
            self.config.min_plate_height,
            &transform,
            &mut self.proposals_buf,
        )
        .map_err(|_| PlateDetectorError::InvalidOutputShape(dims.to_vec()))?;

        if self.proposals_buf.is_empty() {
            return Ok(StackVec::new());
        }

        // Bounded Non-Maximum Suppression executed on stack
        let final_plates = run_bounded_plate_nms(&mut self.proposals_buf, self.config.nms_iou_threshold);

        Ok(final_plates)
    }
}

/// Decodes candidate plate proposals directly into fixed-size stack vector.
fn parse_plate_proposals(
    dims: &[i64],
    data: &[f32],
    conf_thresh: f32,
    min_w: f32,
    min_h: f32,
    transform: &LetterboxTransform,
    out: &mut StackVec<DetectedPlate, MAX_PLATE_PROPOSALS>,
) -> Result<(), &'static str> {
    if dims.len() != 3 {
        return Err("Plate output tensor must have 3 dimensions");
    }

    let d0 = dims[0] as usize;
    let d1 = dims[1] as usize;
    let d2 = dims[2] as usize;

    if d0 != 1 {
        return Err("Only batch size 1 is supported");
    }

    if d2 == 6 {
        // End-to-End NMS [1, N, 6] (x1, y1, x2, y2, conf, class_id)
        let num_boxes = d1;
        for i in 0..num_boxes {
            let offset = i * 6;
            if offset + 4 >= data.len() {
                break;
            }

            let conf = data[offset + 4];
            if conf >= conf_thresh {
                let bbox = transform.unproject_bbox(
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                );

                if bbox.width() >= min_w && bbox.height() >= min_h {
                    let _ = out.push(DetectedPlate { confidence: conf, bbox });
                    if out.is_full() {
                        break;
                    }
                }
            }
        }
    } else if d1 >= 5 && d1 < d2 {
        // Planar 1-Class head [1, 5, N] (cx, cy, w, h, conf)
        let num_proposals = d2;
        let c0 = 0 * num_proposals;
        let c1 = 1 * num_proposals;
        let c2 = 2 * num_proposals;
        let c3 = 3 * num_proposals;
        let c4 = 4 * num_proposals;

        for i in 0..num_proposals {
            if let Some(&conf) = data.get(c4 + i) {
                if conf >= conf_thresh {
                    let cx = *data.get(c0 + i).unwrap_or(&0.0);
                    let cy = *data.get(c1 + i).unwrap_or(&0.0);
                    let w = *data.get(c2 + i).unwrap_or(&0.0);
                    let h = *data.get(c3 + i).unwrap_or(&0.0);

                    let bbox = transform.unproject_bbox(
                        cx - w * 0.5,
                        cy - h * 0.5,
                        cx + w * 0.5,
                        cy + h * 0.5,
                    );

                    if bbox.width() >= min_w && bbox.height() >= min_h {
                        let _ = out.push(DetectedPlate { confidence: conf, bbox });
                        if out.is_full() {
                            break;
                        }
                    }
                }
            }
        }
    } else if d2 >= 5 && d1 >= d2 {
        // Transposed Head [1, N, 5]
        let num_proposals = d1;
        let num_features = d2;

        for i in 0..num_proposals {
            let offset = i * num_features;
            if let Some(&conf) = data.get(offset + 4) {
                if conf >= conf_thresh {
                    let cx = *data.get(offset).unwrap_or(&0.0);
                    let cy = *data.get(offset + 1).unwrap_or(&0.0);
                    let w = *data.get(offset + 2).unwrap_or(&0.0);
                    let h = *data.get(offset + 3).unwrap_or(&0.0);

                    let bbox = transform.unproject_bbox(
                        cx - w * 0.5,
                        cy - h * 0.5,
                        cx + w * 0.5,
                        cy + h * 0.5,
                    );

                    if bbox.width() >= min_w && bbox.height() >= min_h {
                        let _ = out.push(DetectedPlate { confidence: conf, bbox });
                        if out.is_full() {
                            break;
                        }
                    }
                }
            }
        }
    } else {
        return Err("Unrecognized model output dimensions");
    }

    Ok(())
}

/// Zero-allocation, stack-only Non-Maximum Suppression for plates.
fn run_bounded_plate_nms(
    proposals: &mut StackVec<DetectedPlate, MAX_PLATE_PROPOSALS>,
    iou_thresh: f32,
) -> StackVec<DetectedPlate, MAX_OUTPUT_PLATES> {
    let mut outputs: StackVec<DetectedPlate, MAX_OUTPUT_PLATES> = StackVec::new();
    let n = proposals.len();

    proposals.sort_unstable_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    let mut suppressed = [false; MAX_PLATE_PROPOSALS];
    let mut areas = [0.0f32; MAX_PLATE_PROPOSALS];

    for i in 0..n {
        if let Some(p) = proposals.get(i) {
            areas[i] = p.area();
        }
    }

    for i in 0..n {
        if suppressed[i] {
            continue;
        }

        let p_i = match proposals.get(i) {
            Some(p) => *p,
            None => break,
        };

        if outputs.push(p_i).is_err() {
            break;
        }

        let area_i = areas[i];

        for j in (i + 1)..n {
            if suppressed[j] {
                continue;
            }

            if let Some(p_j) = proposals.get(j) {
                let iou = calculate_iou_fast(&p_i.bbox, area_i, &p_j.bbox, areas[j]);
                if iou > iou_thresh {
                    suppressed[j] = true;
                }
            }
        }
    }

    outputs
}