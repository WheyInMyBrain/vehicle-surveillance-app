use crate::preprocess::{calculate_iou_fast, fill_letterbox_tensor};
use crate::types::DetectedPlate;
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
pub enum PlateDetectorError {
    #[error("ONNX Runtime error: {0}")]
    Ort(String),
    #[error("Invalid model output tensor shape: {0:?}")]
    InvalidOutputShape(Vec<i64>),
}

impl From<ort::Error> for PlateDetectorError {
    fn from(err: ort::Error) -> Self {
        PlateDetectorError::Ort(err.to_string())
    }
}

static ORT_ENV: OnceLock<Result<Arc<Environment>, String>> = OnceLock::new();

fn get_or_init_ort_env() -> Result<Arc<Environment>, PlateDetectorError> {
    let env_res = ORT_ENV.get_or_init(|| {
        ort::init()
            .with_name("surveillance_env")
            .build()
            .map(Arc::new)
            .map_err(|e| e.to_string())
    });

    match env_res {
        Ok(env) => Ok(Arc::clone(env)),
        Err(e) => Err(PlateDetectorError::Ort(e.clone())),
    }
}

#[derive(Debug, Clone)]
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

    proposals_buf: Vec<DetectedPlate>,
    areas_buf: Vec<f32>,
    suppressed_buf: Vec<bool>,
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
        let session = Session::builder(&env)
            .map_err(|e| PlateDetectorError::Ort(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| PlateDetectorError::Ort(e.to_string()))?
            .with_intra_threads(config.num_threads)
            .map_err(|e| PlateDetectorError::Ort(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| PlateDetectorError::Ort(e.to_string()))?;

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
            proposals_buf: Vec::with_capacity(32),
            areas_buf: Vec::with_capacity(32),
            suppressed_buf: Vec::with_capacity(32),
        })
    }

    pub fn environment(&self) -> Arc<Environment> {
        Arc::clone(&self._env)
    }

    pub fn detect(
        &mut self,
        vehicle_crop: &DynamicImage,
    ) -> Result<Vec<DetectedPlate>, PlateDetectorError> {
        let transform = fill_letterbox_tensor(
            vehicle_crop,
            self.config.input_width,
            self.config.input_height,
            &mut self.input_buffer,
        );

        let input_val = Value::from_array(self.input_buffer.clone())
            .map_err(|e| PlateDetectorError::Ort(e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs![input_val])
            .map_err(|e| PlateDetectorError::Ort(e.to_string()))?;

        let (out_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| PlateDetectorError::Ort(e.to_string()))?;

        let dims: &[i64] = out_shape.as_ref();
        self.proposals_buf.clear();

        let conf_thresh = self.config.confidence_threshold;
        let min_w = self.config.min_plate_width;
        let min_h = self.config.min_plate_height;

        // FORMAT 1: End-to-End NMS [1, N, 6] (x1, y1, x2, y2, conf, class_id)
        if dims.len() == 3 && dims[2] == 6 {
            let num_boxes = dims[1] as usize;
            for i in 0..num_boxes {
                let offset = i * 6;
                let conf = data[offset + 4];
                if conf < conf_thresh {
                    continue;
                }

                let bbox = transform.unproject_bbox(
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                );

                let w = bbox[2] - bbox[0];
                let h = bbox[3] - bbox[1];
                if w >= min_w && h >= min_h {
                    self.proposals_buf.push(DetectedPlate { confidence: conf, bbox });
                }
            }
        }
        // FORMAT 2: Planar 1-Class head [1, 5, N] (where 5 = [cx, cy, w, h, conf])
        else if dims.len() == 3 && dims[1] >= 5 {
            let num_proposals = dims[2] as usize;
            let c0 = 0 * num_proposals;
            let c1 = 1 * num_proposals;
            let c2 = 2 * num_proposals;
            let c3 = 3 * num_proposals;
            let c4 = 4 * num_proposals;

            for i in 0..num_proposals {
                let conf = data[c4 + i];
                if conf >= conf_thresh {
                    let cx = data[c0 + i];
                    let cy = data[c1 + i];
                    let w = data[c2 + i];
                    let h = data[c3 + i];

                    let bbox = transform.unproject_bbox(
                        cx - w * 0.5,
                        cy - h * 0.5,
                        cx + w * 0.5,
                        cy + h * 0.5,
                    );

                    let bw = bbox[2] - bbox[0];
                    let bh = bbox[3] - bbox[1];
                    if bw >= min_w && bh >= min_h {
                        self.proposals_buf.push(DetectedPlate { confidence: conf, bbox });
                    }
                }
            }
        }
        // FORMAT 3: Transposed Head [1, N, 5]
        else if dims.len() == 3 && dims[2] >= 5 {
            let num_proposals = dims[1] as usize;
            let num_features = dims[2] as usize;

            for i in 0..num_proposals {
                let offset = i * num_features;
                let conf = data[offset + 4];
                if conf >= conf_thresh {
                    let cx = data[offset];
                    let cy = data[offset + 1];
                    let w = data[offset + 2];
                    let h = data[offset + 3];

                    let bbox = transform.unproject_bbox(
                        cx - w * 0.5,
                        cy - h * 0.5,
                        cx + w * 0.5,
                        cy + h * 0.5,
                    );

                    let bw = bbox[2] - bbox[0];
                    let bh = bbox[3] - bbox[1];
                    if bw >= min_w && bh >= min_h {
                        self.proposals_buf.push(DetectedPlate { confidence: conf, bbox });
                    }
                }
            }
        } else {
            return Err(PlateDetectorError::InvalidOutputShape(dims.to_vec()));
        }

        if self.proposals_buf.is_empty() {
            return Ok(Vec::new());
        }

        self.proposals_buf.sort_unstable_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n = self.proposals_buf.len();
        self.areas_buf.clear();
        self.areas_buf.extend(self.proposals_buf.iter().map(|p| p.area()));

        self.suppressed_buf.clear();
        self.suppressed_buf.resize(n, false);

        let mut final_plates = Vec::with_capacity(n.min(4));

        for i in 0..n {
            if self.suppressed_buf[i] {
                continue;
            }

            let p_i = self.proposals_buf[i];
            let area_i = self.areas_buf[i];
            final_plates.push(p_i);

            for j in (i + 1)..n {
                if self.suppressed_buf[j] {
                    continue;
                }

                let iou = calculate_iou_fast(
                    &p_i.bbox,
                    area_i,
                    &self.proposals_buf[j].bbox,
                    self.areas_buf[j],
                );

                if iou > self.config.nms_iou_threshold {
                    self.suppressed_buf[j] = true;
                }
            }
        }

        Ok(final_plates)
    }
}