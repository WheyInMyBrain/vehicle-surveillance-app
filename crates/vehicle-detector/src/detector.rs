use crate::preprocess::{calculate_iou_fast, fill_letterbox_tensor};
use crate::types::{DetectedVehicle, VehicleClass};
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
    #[error("Invalid model output tensor shape: {0:?}")]
    InvalidOutputShape(Vec<i64>),
}

impl From<ort::Error> for DetectorError {
    fn from(err: ort::Error) -> Self {
        DetectorError::Ort(err.to_string())
    }
}

static ORT_ENV: OnceLock<Result<Arc<Environment>, String>> = OnceLock::new();

pub fn get_or_init_ort_env() -> Result<Arc<Environment>, DetectorError> {
    let env_res = ORT_ENV.get_or_init(|| {
        ort::init()
            .with_name("surveillance_env")
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

    proposals_buf: Vec<DetectedVehicle>,
    areas_buf: Vec<f32>,
    suppressed_buf: Vec<bool>,
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
        let session = Session::builder(&env)
            .map_err(|e| DetectorError::Ort(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| DetectorError::Ort(e.to_string()))?
            .with_intra_threads(config.num_threads)
            .map_err(|e| DetectorError::Ort(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| DetectorError::Ort(e.to_string()))?;

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
            proposals_buf: Vec::with_capacity(128),
            areas_buf: Vec::with_capacity(128),
            suppressed_buf: Vec::with_capacity(128),
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
            self.input_buffer = Array4::<f32>::zeros((
                1,
                3,
                config.input_height as usize,
                config.input_width as usize,
            ));
        }
        self.config = config;
    }

    pub fn detect(&mut self, img: &DynamicImage) -> Result<Vec<DetectedVehicle>, DetectorError> {
        let transform = fill_letterbox_tensor(
            img,
            self.config.input_width,
            self.config.input_height,
            &mut self.input_buffer,
        );

        let input_val = Value::from_array(self.input_buffer.clone())
            .map_err(|e| DetectorError::Ort(e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs![input_val])
            .map_err(|e| DetectorError::Ort(e.to_string()))?;

        let (out_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| DetectorError::Ort(e.to_string()))?;

        let dims: &[i64] = out_shape.as_ref();
        self.proposals_buf.clear();

        let min_dim = self.config.min_box_dimension;
        let conf_thresh = self.config.confidence_threshold;

        if dims.len() == 3 && dims[2] == 6 {
            let num_boxes = dims[1] as usize;
            for i in 0..num_boxes {
                let offset = i * 6;
                let conf = data[offset + 4];
                if conf < conf_thresh {
                    continue;
                }

                let class_id = data[offset + 5].round() as usize;
                if let Some(vehicle_class) = VehicleClass::from_coco_id(class_id) {
                    let bbox = transform.unproject_bbox(
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    );

                    if (bbox[2] - bbox[0]) >= min_dim && (bbox[3] - bbox[1]) >= min_dim {
                        self.proposals_buf.push(DetectedVehicle {
                            class: vehicle_class,
                            confidence: conf,
                            bbox,
                        });
                    }
                }
            }
        } else if dims.len() == 3 && dims[1] >= 84 {
            let num_features = dims[1] as usize;
            let num_proposals = dims[2] as usize;

            for i in 0..num_proposals {
                let mut best_class = 0;
                let mut max_conf = 0.0f32;

                for c in 4..num_features {
                    let conf = data[c * num_proposals + i];
                    if conf > max_conf {
                        max_conf = conf;
                        best_class = c - 4;
                    }
                }

                if max_conf >= conf_thresh {
                    if let Some(vehicle_class) = VehicleClass::from_coco_id(best_class) {
                        let cx = data[i];
                        let cy = data[num_proposals + i];
                        let w = data[2 * num_proposals + i];
                        let h = data[3 * num_proposals + i];

                        let bbox = transform.unproject_bbox(
                            cx - w * 0.5,
                            cy - h * 0.5,
                            cx + w * 0.5,
                            cy + h * 0.5,
                        );

                        if (bbox[2] - bbox[0]) >= min_dim && (bbox[3] - bbox[1]) >= min_dim {
                            self.proposals_buf.push(DetectedVehicle {
                                class: vehicle_class,
                                confidence: max_conf,
                                bbox,
                            });
                        }
                    }
                }
            }
        } else if dims.len() == 3 && dims[2] >= 84 {
            let num_proposals = dims[1] as usize;
            let num_features = dims[2] as usize;

            for i in 0..num_proposals {
                let row = i * num_features;
                let mut best_class = 0;
                let mut max_conf = 0.0f32;

                for c in 4..num_features {
                    let conf = data[row + c];
                    if conf > max_conf {
                        max_conf = conf;
                        best_class = c - 4;
                    }
                }

                if max_conf >= conf_thresh {
                    if let Some(vehicle_class) = VehicleClass::from_coco_id(best_class) {
                        let cx = data[row];
                        let cy = data[row + 1];
                        let w = data[row + 2];
                        let h = data[row + 3];

                        let bbox = transform.unproject_bbox(
                            cx - w * 0.5,
                            cy - h * 0.5,
                            cx + w * 0.5,
                            cy + h * 0.5,
                        );

                        if (bbox[2] - bbox[0]) >= min_dim && (bbox[3] - bbox[1]) >= min_dim {
                            self.proposals_buf.push(DetectedVehicle {
                                class: vehicle_class,
                                confidence: max_conf,
                                bbox,
                            });
                        }
                    }
                }
            }
        } else {
            return Err(DetectorError::InvalidOutputShape(dims.to_vec()));
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

        let mut final_detections = Vec::with_capacity(n.min(32));

        for i in 0..n {
            if self.suppressed_buf[i] {
                continue;
            }

            let p_i = self.proposals_buf[i];
            let area_i = self.areas_buf[i];
            final_detections.push(p_i);

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

        Ok(final_detections)
    }
}