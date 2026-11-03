use crate::annotate::FrameAnnotator;
use crate::config::SurveillancePipelineConfig;
use crate::types::{PipelineFrameResult, VehicleSnapshot};
use image::{DynamicImage, RgbImage};
use std::collections::HashMap;
use thiserror::Error;
use vehicle_detector::VehicleDetector;
use vehicle_lpocr::{OcrResult, PlateOcrEngine};
use vehicle_lpr::PlateDetector;
use vehicle_quality::{QualityAuditor, TrackQualityManager};
use vehicle_tracker::ByteTracker;

#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("Detector error: {0}")]
    Detector(#[from] vehicle_detector::DetectorError),
    #[error("Plate detector error: {0}")]
    Plate(#[from] vehicle_lpr::PlateDetectorError),
    #[error("OCR error: {0}")]
    Ocr(#[from] vehicle_lpocr::OcrError),
}

pub struct VehiclePipeline {
    _config: SurveillancePipelineConfig,
    frame_idx: usize,

    detector: VehicleDetector,
    tracker: ByteTracker,
    quality_manager: TrackQualityManager,
    plate_detector: PlateDetector,
    ocr_engine: PlateOcrEngine,
    annotator: FrameAnnotator,

    /// Remembers the best confirmed license plate and plate bbox per vehicle track_id
    track_plate_registry: HashMap<u64, (Option<[f32; 4]>, OcrResult)>,
}

impl VehiclePipeline {
    pub fn new(config: SurveillancePipelineConfig) -> Result<Self, PipelineError> {
        // 1. Vehicle detector initializes the process-wide ONNX environment
        let detector = VehicleDetector::new(&config.vehicle_model_path, config.detector.clone())?;
        let shared_env = detector.environment();

        // 2. Share ONNX environment with plate detector and OCR
        let plate_detector = PlateDetector::new_with_env(
            &config.plate_model_path,
            config.plate_detector.clone(),
            shared_env.clone(),
        )?;

        let ocr_engine = PlateOcrEngine::new_with_env(
            &config.ocr_model_path,
            config.ocr.clone(),
            shared_env,
        )?;

        let tracker = ByteTracker::new(config.tracker.clone());
        let quality_auditor = QualityAuditor::new(config.quality.clone());
        let quality_manager = TrackQualityManager::new(
            quality_auditor,
            config.golden_crops_per_track,
            config.min_quality_score,
        );

        let annotator = FrameAnnotator::new(config.custom_font_path.as_deref());

        Ok(Self {
            _config: config,
            frame_idx: 0,
            detector,
            tracker,
            quality_manager,
            plate_detector,
            ocr_engine,
            annotator,
            track_plate_registry: HashMap::new(),
        })
    }

    /// Process a continuous video stream frame
    pub fn process_frame(
        &mut self,
        frame: &DynamicImage,
    ) -> Result<(PipelineFrameResult, RgbImage), PipelineError> {
        self.frame_idx += 1;

        // Step 1: Detect Vehicles
        let detections = self.detector.detect(frame)?;

        // Step 2: Track Vehicles
        let tracks = self.tracker.update(&detections);

        let mut active_track_ids = Vec::with_capacity(tracks.len());

        // Step 3: Quality triage and plate inspection on candidate vehicle crops
        for trk in &tracks {
            active_track_ids.push(trk.track_id);

            // Submit frame to quality buffer
            self.quality_manager.process_frame(self.frame_idx, trk, frame);

            // If we don't already have a high-confidence read for this vehicle, inspect its golden crop
            let already_resolved = self
                .track_plate_registry
                .get(&trk.track_id)
                .map(|(_, ocr)| ocr.confidence > 0.70)
                .unwrap_or(false);

            if !already_resolved {
                if let Some(golden) = self.quality_manager.get_best_crop(trk.track_id) {
                    let plates = self.plate_detector.detect(&golden.crop)?;

                    for plate in plates {
                        let plate_crop = plate.crop_from(&golden.crop);
                        let ocr = self.ocr_engine.recognize(&plate_crop)?;

                        if ocr.is_valid_plate() {
                            // Translate plate bbox from golden vehicle crop space to native frame space
                            let gx1 = trk.bbox[0] + plate.x1();
                            let gy1 = trk.bbox[1] + plate.y1();
                            let gx2 = gx1 + plate.width();
                            let gy2 = gy1 + plate.height();
                            let global_plate_bbox = [gx1, gy1, gx2, gy2];

                            self.track_plate_registry
                                .insert(trk.track_id, (Some(global_plate_bbox), ocr));
                            break;
                        }
                    }
                }
            }
        }

        // Clean up stale track buffers periodically
        if self.frame_idx % 60 == 0 {
            self.quality_manager.retain_active_tracks(&active_track_ids);
        }

        // Step 4: Assemble frame snapshots
        let mut snapshots = Vec::with_capacity(tracks.len());

        for trk in &tracks {
            let (plate_bbox, plate_ocr) = self
                .track_plate_registry
                .get(&trk.track_id)
                .cloned()
                .map(|(b, o)| (b, Some(o)))
                .unwrap_or((None, None));

            snapshots.push(VehicleSnapshot {
                track_id: trk.track_id,
                class: trk.class,
                confidence: trk.confidence,
                bbox: trk.bbox,
                tracklet_len: trk.tracklet_len,
                plate_bbox,
                plate_ocr,
            });
        }

        // Step 5: Render Annotations
        let annotated_img = self.annotator.annotate(frame, &snapshots);

        let result = PipelineFrameResult {
            frame_idx: self.frame_idx,
            active_track_count: tracks.len(),
            plates_read_count: self.track_plate_registry.len(),
            vehicles: snapshots,
        };

        Ok((result, annotated_img))
    }

    /// Process a standalone static image (resets tracker state and processes in single-frame mode)
    pub fn process_image(
        &mut self,
        image: &DynamicImage,
    ) -> Result<(PipelineFrameResult, RgbImage), PipelineError> {

        // Step 1: Detect Vehicles
        let detections = self.detector.detect(image)?;

        let mut snapshots = Vec::with_capacity(detections.len());

        for (idx, det) in detections.into_iter().enumerate() {
            let track_id = (idx + 1) as u64;
            let vehicle_crop = det.crop_from(image);

            // Step 2: Detect Plate inside Vehicle Crop
            let plates = self.plate_detector.detect(&vehicle_crop)?;

            let mut plate_bbox = None;
            let mut plate_ocr = None;

            for plate in plates {
                let p_crop = plate.crop_from(&vehicle_crop);
                let ocr = self.ocr_engine.recognize(&p_crop)?;

                // Map coordinates to full image space
                let gx1 = det.bbox[0] + plate.x1();
                let gy1 = det.bbox[1] + plate.y1();
                let gx2 = gx1 + plate.width();
                let gy2 = gy1 + plate.height();

                plate_bbox = Some([gx1, gy1, gx2, gy2]);
                plate_ocr = Some(ocr);
                break;
            }

            snapshots.push(VehicleSnapshot {
                track_id,
                class: det.class,
                confidence: det.confidence,
                bbox: det.bbox,
                tracklet_len: 1,
                plate_bbox,
                plate_ocr,
            });
        }

        let annotated_img = self.annotator.annotate(image, &snapshots);

        let count = snapshots.iter().filter(|s| s.plate_ocr.is_some()).count();

        let result = PipelineFrameResult {
            frame_idx: 1,
            active_track_count: snapshots.len(),
            plates_read_count: count,
            vehicles: snapshots,
        };

        Ok((result, annotated_img))
    }
}