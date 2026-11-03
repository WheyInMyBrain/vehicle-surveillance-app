use crate::annotate::FrameAnnotator;
use crate::config::SurveillancePipelineConfig;
use crate::types::{PipelineFrameResult, PipelineStageTimings, VehicleSnapshot};
use image::{DynamicImage, RgbImage};
use std::collections::HashMap;
use std::time::Instant;
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

/// Tracks LPD attempt history and OCR resolution state per vehicle
#[derive(Debug, Clone)]
struct TrackLprState {
    plate_bbox: Option<[f32; 4]>,
    ocr_result: Option<OcrResult>,
    /// Number of times LPD has been executed on this vehicle
    attempts: usize,
    /// The frame index when LPD was last attempted
    last_attempt_frame: usize,
    /// Highest composite quality score tested so far
    best_tested_score: f32,
}

pub struct VehiclePipeline {
    config: SurveillancePipelineConfig,
    frame_idx: usize,

    detector: VehicleDetector,
    tracker: ByteTracker,
    quality_manager: TrackQualityManager,
    plate_detector: PlateDetector,
    ocr_engine: PlateOcrEngine,
    annotator: FrameAnnotator,

    /// State machine tracking plate detection attempts and results per track_id
    track_lpr_states: HashMap<u64, TrackLprState>,
}

impl VehiclePipeline {
    pub fn new(config: SurveillancePipelineConfig) -> Result<Self, PipelineError> {
        let detector = VehicleDetector::new(&config.vehicle_model_path, config.detector.clone())?;
        let shared_env = detector.environment();

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
            config,
            frame_idx: 0,
            detector,
            tracker,
            quality_manager,
            plate_detector,
            ocr_engine,
            annotator,
            track_lpr_states: HashMap::new(),
        })
    }

    pub fn config(&self) -> &SurveillancePipelineConfig {
        &self.config
    }

    /// Resets tracking, quality, and LPD states without reloading models
    pub fn reset_session(&mut self) {
        self.frame_idx = 0;
        self.tracker = ByteTracker::new(self.config.tracker.clone());
        let auditor = QualityAuditor::new(self.config.quality.clone());
        self.quality_manager = TrackQualityManager::new(
            auditor,
            self.config.golden_crops_per_track,
            self.config.min_quality_score,
        );
        self.track_lpr_states.clear();
    }

    /// Primary streaming interface: processes one frame as it arrives in real time
    pub fn process_frame(
        &mut self,
        frame: &DynamicImage,
    ) -> Result<(PipelineFrameResult, RgbImage), PipelineError> {
        let t_total_start = Instant::now();
        self.frame_idx += 1;
        let mut timings = PipelineStageTimings::default();

        // 1. Vehicle Detection
        let t_detect = Instant::now();
        let detections = self.detector.detect(frame)?;
        timings.vehicle_detect_time = t_detect.elapsed();

        // 2. Tracking
        let t_track = Instant::now();
        let tracks = self.tracker.update(&detections);
        timings.tracking_time = t_track.elapsed();

        let mut active_track_ids = Vec::with_capacity(tracks.len());

        // 3. Quality Triage & Throttled Gated LPD/OCR
        let t_quality = Instant::now();
        let mut plate_detect_accum = std::time::Duration::ZERO;
        let mut plate_ocr_accum = std::time::Duration::ZERO;

        // Gating thresholds
        const MAX_LPD_ATTEMPTS: usize = 4;
        const MIN_FRAME_INTERVAL: usize = 6;
        const MIN_SCORE_IMPROVEMENT_RATIO: f32 = 1.15; // 15% improvement

        for trk in &tracks {
            active_track_ids.push(trk.track_id);
            self.quality_manager.process_frame(self.frame_idx, trk, frame);

            let state = self.track_lpr_states.entry(trk.track_id).or_insert(TrackLprState {
                plate_bbox: None,
                ocr_result: None,
                attempts: 0,
                last_attempt_frame: 0,
                best_tested_score: 0.0,
            });

            // Gate 1: Check if plate is already confirmed with high confidence (>70%)
            let already_resolved = state
                .ocr_result
                .as_ref()
                .map(|o| o.confidence > 0.70)
                .unwrap_or(false);

            if already_resolved {
                continue;
            }

            // Gate 2: Cap total lifetime attempts per vehicle track
            if state.attempts >= MAX_LPD_ATTEMPTS {
                continue;
            }

            // Gate 3: Check candidate crop availability
            let golden_crop = match self.quality_manager.get_best_crop(trk.track_id) {
                Some(crop) => crop,
                None => continue,
            };

            let current_score = golden_crop.metrics.composite_score;

            // Gate 4: Cooldown and Quality Jump Evaluation
            // Run LPD if:
            //   - First attempt (attempts == 0), OR
            //   - Quality improved by at least 15% AND at least MIN_FRAME_INTERVAL frames elapsed
            let is_first_attempt = state.attempts == 0;
            let frames_since_last = self.frame_idx.saturating_sub(state.last_attempt_frame);
            let has_significant_quality_jump = current_score >= (state.best_tested_score * MIN_SCORE_IMPROVEMENT_RATIO);

            let should_run_lpd = is_first_attempt || (frames_since_last >= MIN_FRAME_INTERVAL && has_significant_quality_jump);

            if should_run_lpd {
                state.attempts += 1;
                state.last_attempt_frame = self.frame_idx;
                state.best_tested_score = current_score.max(state.best_tested_score);

                let t_p = Instant::now();
                let plates = self.plate_detector.detect(&golden_crop.crop)?;
                plate_detect_accum += t_p.elapsed();

                for plate in plates {
                    let plate_crop = plate.crop_from(&golden_crop.crop);
                    let t_o = Instant::now();
                    let ocr = self.ocr_engine.recognize(&plate_crop)?;
                    plate_ocr_accum += t_o.elapsed();

                    if ocr.is_valid_plate() {
                        let gx1 = trk.bbox[0] + plate.x1();
                        let gy1 = trk.bbox[1] + plate.y1();
                        let gx2 = gx1 + plate.width();
                        let gy2 = gy1 + plate.height();
                        let global_plate_bbox = [gx1, gy1, gx2, gy2];

                        state.plate_bbox = Some(global_plate_bbox);
                        state.ocr_result = Some(ocr);
                        break;
                    }
                }
            }
        }

        timings.quality_triage_time = t_quality.elapsed().saturating_sub(plate_detect_accum + plate_ocr_accum);
        timings.plate_detect_time = plate_detect_accum;
        timings.plate_ocr_time = plate_ocr_accum;

        // Periodic eviction of stale tracking state
        if self.frame_idx % 60 == 0 {
            self.quality_manager.retain_active_tracks(&active_track_ids);
            self.track_lpr_states.retain(|id, _| active_track_ids.contains(id));
        }

        // 4. Assemble Vehicle Snapshots
        let mut snapshots = Vec::with_capacity(tracks.len());
        for trk in &tracks {
            let (plate_bbox, plate_ocr) = self
                .track_lpr_states
                .get(&trk.track_id)
                .map(|s| (s.plate_bbox, s.ocr_result.clone()))
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

        // 5. Annotation
        let t_annot = Instant::now();
        let annotated_img = self.annotator.annotate(frame, &snapshots);
        timings.annotation_time = t_annot.elapsed();
        timings.total_frame_time = t_total_start.elapsed();

        let plates_read = self
            .track_lpr_states
            .values()
            .filter(|s| s.ocr_result.is_some())
            .count();

        let result = PipelineFrameResult {
            frame_idx: self.frame_idx,
            active_track_count: tracks.len(),
            plates_read_count: plates_read,
            vehicles: snapshots,
            timings,
        };

        Ok((result, annotated_img))
    }

    /// Specialized burst mode: processes a sequential burst of still images
    pub fn process_burst(
        &mut self,
        burst_frames: &[DynamicImage],
    ) -> Result<Vec<(PipelineFrameResult, RgbImage)>, PipelineError> {
        let mut results = Vec::with_capacity(burst_frames.len());
        for frame in burst_frames {
            let res = self.process_frame(frame)?;
            results.push(res);
        }
        Ok(results)
    }

    /// Single static image fallback
    pub fn process_image(
        &mut self,
        image: &DynamicImage,
    ) -> Result<(PipelineFrameResult, RgbImage), PipelineError> {
        let t_total_start = Instant::now();
        let mut timings = PipelineStageTimings::default();

        let t_det = Instant::now();
        let detections = self.detector.detect(image)?;
        timings.vehicle_detect_time = t_det.elapsed();

        let mut snapshots = Vec::with_capacity(detections.len());
        let mut plate_det_time = std::time::Duration::ZERO;
        let mut plate_ocr_time = std::time::Duration::ZERO;

        for (idx, det) in detections.into_iter().enumerate() {
            let track_id = (idx + 1) as u64;
            let vehicle_crop = det.crop_from(image);

            let t_p = Instant::now();
            let plates = self.plate_detector.detect(&vehicle_crop)?;
            plate_det_time += t_p.elapsed();

            let mut plate_bbox = None;
            let mut plate_ocr = None;

            for plate in plates {
                let p_crop = plate.crop_from(&vehicle_crop);
                let t_o = Instant::now();
                let ocr = self.ocr_engine.recognize(&p_crop)?;
                plate_ocr_time += t_o.elapsed();

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

        timings.plate_detect_time = plate_det_time;
        timings.plate_ocr_time = plate_ocr_time;

        let t_ann = Instant::now();
        let annotated_img = self.annotator.annotate(image, &snapshots);
        timings.annotation_time = t_ann.elapsed();
        timings.total_frame_time = t_total_start.elapsed();

        let count = snapshots.iter().filter(|s| s.plate_ocr.is_some()).count();

        let result = PipelineFrameResult {
            frame_idx: 1,
            active_track_count: snapshots.len(),
            plates_read_count: count,
            vehicles: snapshots,
            timings,
        };

        Ok((result, annotated_img))
    }
}