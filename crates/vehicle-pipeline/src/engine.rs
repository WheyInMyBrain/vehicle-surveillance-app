//! Complete multi-stage surveillance orchestration engine with Occlusion-Aware Triage and Dual-Stream Dispatch.

use crate::annotate::FrameAnnotator;
use crate::config::SurveillancePipelineConfig;
use crate::consensus::TemporalPlateVoter;
use crate::types::{PipelineFrameResult, PipelineStageTimings, VehicleSnapshot};
use heapless::Vec as StackVec;
use image::{DynamicImage, GenericImageView, RgbImage};
use std::time::Instant;
use thiserror::Error;
use vehicle_detector::{BoundingBox, VehicleDetector};
use vehicle_lpocr::PlateOcrEngine;
use vehicle_lpr::PlateDetector;
use vehicle_quality::{QualityAuditor, TrackQualityManager};
use vehicle_tracker::{ByteTracker, MAX_TRACKS};

#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("Detector error: {0}")]
    Detector(#[from] vehicle_detector::DetectorError),
    #[error("Plate detector error: {0}")]
    Plate(#[from] vehicle_lpr::PlateDetectorError),
    #[error("OCR error: {0}")]
    Ocr(#[from] vehicle_lpocr::OcrError),
}

/// Tracks LPD attempt history, occlusion status, and consensus state per track
#[derive(Debug, Clone)]
struct TrackLprState {
    track_id: u64,
    active: bool,
    plate_bbox: Option<BoundingBox>,
    voter: TemporalPlateVoter,
    attempts: usize,
    last_attempt_frame: usize,
    best_tested_score: f32,
    is_occluded: bool,
}

impl TrackLprState {
    const fn empty() -> Self {
        Self {
            track_id: 0,
            active: false,
            plate_bbox: None,
            voter: TemporalPlateVoter::new(0),
            attempts: 0,
            last_attempt_frame: 0,
            best_tested_score: 0.0,
            is_occluded: false,
        }
    }
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

    /// Heap-boxed bounded table for active vehicle LPR states to prevent stack-overflow on edge threads
    track_lpr_table: Box<[TrackLprState; MAX_TRACKS]>,
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

        let tracker = ByteTracker::new(config.tracker);
        let quality_auditor = QualityAuditor::new(config.quality);
        let quality_manager = TrackQualityManager::new(
            quality_auditor,
            config.min_quality_score,
        );

        let annotator = FrameAnnotator::new(config.custom_font_path.as_deref());

        let vec_table: Vec<TrackLprState> = (0..MAX_TRACKS).map(|_| TrackLprState::empty()).collect();
        let track_lpr_table = match vec_table.into_boxed_slice().try_into() {
            Ok(boxed_arr) => boxed_arr,
            Err(_) => unreachable!("Vector length exactly matches MAX_TRACKS"),
        };

        Ok(Self {
            config,
            frame_idx: 0,
            detector,
            tracker,
            quality_manager,
            plate_detector,
            ocr_engine,
            annotator,
            track_lpr_table,
        })
    }

    pub fn config(&self) -> &SurveillancePipelineConfig {
        &self.config
    }

    pub fn reset_session(&mut self) {
        self.frame_idx = 0;
        self.tracker = ByteTracker::new(self.config.tracker);
        let auditor = QualityAuditor::new(self.config.quality);
        self.quality_manager = TrackQualityManager::new(
            auditor,
            self.config.min_quality_score,
        );
        for slot in self.track_lpr_table.iter_mut() {
            *slot = TrackLprState::empty();
        }
    }

    /// Evaluates a low-resolution tracking frame alongside an optional full 4K high-res capture.
    pub fn process_frame_dual(
        &mut self,
        tracking_frame: &DynamicImage,
        high_res_frame: Option<&DynamicImage>,
    ) -> Result<(PipelineFrameResult, RgbImage), PipelineError> {
        let t_total_start = Instant::now();
        self.frame_idx += 1;
        let mut timings = PipelineStageTimings::default();

        let (low_w, low_h) = tracking_frame.dimensions();
        let target_inspection_frame = high_res_frame.unwrap_or(tracking_frame);
        let (insp_w, insp_h) = target_inspection_frame.dimensions();

        // 1. Front-End Vehicle Detection on Low-Res Proxy
        let t_detect = Instant::now();
        let detections = self.detector.detect(tracking_frame)?;
        timings.vehicle_detect_time = t_detect.elapsed();

        // 2. Tracking Association
        let t_track = Instant::now();
        let tracks = self.tracker.update(&detections);
        timings.tracking_time = t_track.elapsed();

        let mut active_track_ids: StackVec<u64, MAX_TRACKS> = StackVec::new();

        // 3. Occlusion Analysis (Filter overlapping / partially hidden vehicles)
        let n_tracks = tracks.len();
        let mut is_occluded_flags = [false; MAX_TRACKS];

        for i in 0..n_tracks {
            for j in 0..n_tracks {
                if i != j {
                    let overlap_ratio = calculate_box_overlap_ratio(&tracks[i].bbox, &tracks[j].bbox);
                    if overlap_ratio > 0.08 {
                        is_occluded_flags[i] = true;
                        break;
                    }
                }
            }
        }

        // 4. Quality Triage & Gated High-Res LPD/OCR
        let t_quality = Instant::now();
        let mut plate_detect_accum = core::time::Duration::ZERO;
        let mut plate_ocr_accum = core::time::Duration::ZERO;

        const MAX_LPD_ATTEMPTS: usize = 4;
        const MIN_FRAME_INTERVAL: usize = 5;

        for (idx, trk) in tracks.iter().enumerate() {
            let _ = active_track_ids.push(trk.track_id);

            // Scale bounding box to inspection image resolution for quality scoring
            let scale_x = insp_w as f32 / low_w as f32;
            let scale_y = insp_h as f32 / low_h as f32;
            let mut inspection_trk = *trk;
            inspection_trk.bbox = BoundingBox::new_clamped(
                trk.bbox.x1 * scale_x,
                trk.bbox.y1 * scale_y,
                trk.bbox.x2 * scale_x,
                trk.bbox.y2 * scale_y,
            );

            // Audit quality against the high-res inspection buffer
            self.quality_manager.process_frame(self.frame_idx, &inspection_trk, target_inspection_frame);

            let slot_idx = match self.get_or_assign_lpr_slot(trk.track_id) {
                Some(s) => s,
                None => continue,
            };

            let state = &mut self.track_lpr_table[slot_idx];
            state.is_occluded = is_occluded_flags[idx];

            if state.voter.is_consensus_reached() || state.is_occluded || state.attempts >= MAX_LPD_ATTEMPTS {
                continue;
            }

            let golden_crop = match self.quality_manager.get_best_crop(trk.track_id) {
                Some(crop) => crop,
                None => continue,
            };

            let current_score = golden_crop.metrics.composite_score;
            let is_first = state.attempts == 0;
            let frames_since_last = self.frame_idx.saturating_sub(state.last_attempt_frame);
            let score_jump = current_score >= (state.best_tested_score * 1.10);

            let should_run = is_first || (frames_since_last >= MIN_FRAME_INTERVAL && score_jump);

            if should_run {
                state.attempts += 1;
                state.last_attempt_frame = self.frame_idx;
                state.best_tested_score = current_score.max(state.best_tested_score);

                let norm_box = BoundingBox::new_clamped(
                    trk.bbox.x1 / low_w as f32,
                    trk.bbox.y1 / low_h as f32,
                    trk.bbox.x2 / low_w as f32,
                    trk.bbox.y2 / low_h as f32,
                );

                let high_res_vehicle_crop = crop_normalized_box(target_inspection_frame, &norm_box);

                let t_p = Instant::now();
                let plates = self.plate_detector.detect(&high_res_vehicle_crop)?;
                plate_detect_accum += t_p.elapsed();

                for plate in &plates {
                    let plate_crop = plate.crop_from(&high_res_vehicle_crop);
                    let t_o = Instant::now();
                    let ocr = self.ocr_engine.recognize(&plate_crop)?;
                    plate_ocr_accum += t_o.elapsed();

                    if ocr.is_valid_plate() {
                        let high_crop_w = norm_box.width() * insp_w as f32;
                        let high_crop_h = norm_box.height() * insp_h as f32;

                        let p_norm_x1 = norm_box.x1 + (plate.x1() / high_crop_w) * norm_box.width();
                        let p_norm_y1 = norm_box.y1 + (plate.y1() / high_crop_h) * norm_box.height();
                        let p_norm_x2 = p_norm_x1 + (plate.width() / high_crop_w) * norm_box.width();
                        let p_norm_y2 = p_norm_y1 + (plate.height() / high_crop_h) * norm_box.height();

                        state.plate_bbox = Some(BoundingBox::new_clamped(
                            p_norm_x1 * low_w as f32,
                            p_norm_y1 * low_h as f32,
                            p_norm_x2 * low_w as f32,
                            p_norm_y2 * low_h as f32,
                        ));

                        state.voter.accumulate(&ocr);
                        break;
                    }
                }
            }
        }

        timings.quality_triage_time = t_quality.elapsed().saturating_sub(plate_detect_accum + plate_ocr_accum);
        timings.plate_detect_time = plate_detect_accum;
        timings.plate_ocr_time = plate_ocr_accum;

        // Periodic eviction of stale tracks
        if self.frame_idx % 60 == 0 {
            self.quality_manager.retain_active_tracks(&active_track_ids);
            for slot in self.track_lpr_table.iter_mut() {
                if slot.active && !active_track_ids.contains(&slot.track_id) {
                    *slot = TrackLprState::empty();
                }
            }
        }

        // 5. Assemble Final Telemetry Snapshots
        let mut snapshots: StackVec<VehicleSnapshot, MAX_TRACKS> = StackVec::new();
        for trk in &tracks {
            let (plate_bbox, plate_ocr) = self
                .find_lpr_slot(trk.track_id)
                .map(|idx| {
                    let s = &self.track_lpr_table[idx];
                    let resolved_ocr = s.voter.resolve();
                    let final_ocr = if resolved_ocr.is_valid_plate() {
                        Some(resolved_ocr)
                    } else {
                        None
                    };
                    (s.plate_bbox, final_ocr)
                })
                .unwrap_or((None, None));

            let _ = snapshots.push(VehicleSnapshot {
                track_id: trk.track_id,
                class: trk.class,
                confidence: trk.confidence,
                bbox: trk.bbox,
                tracklet_len: trk.tracklet_len,
                plate_bbox,
                plate_ocr,
            });
        }

        // 6. Annotation
        let t_annot = Instant::now();
        let annotated_img = self.annotator.annotate(tracking_frame, &snapshots);
        timings.annotation_time = t_annot.elapsed();
        timings.total_frame_time = t_total_start.elapsed();

        let plates_read = self
            .track_lpr_table
            .iter()
            .filter(|s| s.active && s.voter.is_consensus_reached())
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

    /// Single-stream convenience wrapper
    #[inline(always)]
    pub fn process_frame(
        &mut self,
        frame: &DynamicImage,
    ) -> Result<(PipelineFrameResult, RgbImage), PipelineError> {
        self.process_frame_dual(frame, None)
    }

    /// Single static image processing with localized plate detection
    pub fn process_image(
        &mut self,
        image: &DynamicImage,
    ) -> Result<(PipelineFrameResult, RgbImage), PipelineError> {
        let t_total_start = Instant::now();
        let mut timings = PipelineStageTimings::default();

        let t_det = Instant::now();
        let detections = self.detector.detect(image)?;
        timings.vehicle_detect_time = t_det.elapsed();

        let mut snapshots: StackVec<VehicleSnapshot, MAX_TRACKS> = StackVec::new();
        let mut plate_det_time = core::time::Duration::ZERO;
        let mut plate_ocr_time = core::time::Duration::ZERO;

        for (idx, det) in detections.iter().enumerate() {
            let track_id = (idx + 1) as u64;
            let vehicle_crop = crop_vehicle_box(image, &det.bbox);

            let t_p = Instant::now();
            let plates = self.plate_detector.detect(&vehicle_crop)?;
            plate_det_time += t_p.elapsed();

            let mut plate_bbox = None;
            let mut plate_ocr = None;

            for plate in &plates {
                let p_crop = plate.crop_from(&vehicle_crop);
                let t_o = Instant::now();
                let ocr = self.ocr_engine.recognize(&p_crop)?;
                plate_ocr_time += t_o.elapsed();

                let gx1 = det.bbox.x1 + plate.x1();
                let gy1 = det.bbox.y1 + plate.y1();
                let gx2 = gx1 + plate.width();
                let gy2 = gy1 + plate.height();

                plate_bbox = Some(BoundingBox::new_clamped(gx1, gy1, gx2, gy2));
                plate_ocr = Some(ocr);
                break;
            }

            let _ = snapshots.push(VehicleSnapshot {
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

    #[inline(always)]
    fn find_lpr_slot(&self, track_id: u64) -> Option<usize> {
        for (i, slot) in self.track_lpr_table.iter().enumerate() {
            if slot.active && slot.track_id == track_id {
                return Some(i);
            }
        }
        None
    }

    fn get_or_assign_lpr_slot(&mut self, track_id: u64) -> Option<usize> {
        if let Some(idx) = self.find_lpr_slot(track_id) {
            return Some(idx);
        }

        for (i, slot) in self.track_lpr_table.iter_mut().enumerate() {
            if !slot.active {
                slot.active = true;
                slot.track_id = track_id;
                slot.plate_bbox = None;
                slot.voter.reset(track_id);
                slot.attempts = 0;
                slot.last_attempt_frame = 0;
                slot.best_tested_score = 0.0;
                slot.is_occluded = false;
                return Some(i);
            }
        }

        None
    }
}

/// Ratio of box_a area occluded by box_b
#[inline(always)]
fn calculate_box_overlap_ratio(box_a: &BoundingBox, box_b: &BoundingBox) -> f32 {
    let x1 = box_a.x1.max(box_b.x1);
    let y1 = box_a.y1.max(box_b.y1);
    let x2 = box_a.x2.min(box_b.x2);
    let y2 = box_a.y2.min(box_b.y2);

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = box_a.area();
    if area_a <= 1e-4 {
        0.0
    } else {
        inter_area / area_a
    }
}

#[inline]
fn crop_normalized_box(img: &DynamicImage, norm_box: &BoundingBox) -> DynamicImage {
    let (img_w, img_h) = img.dimensions();
    let x1 = (norm_box.x1 * img_w as f32).max(0.0) as u32;
    let y1 = (norm_box.y1 * img_h as f32).max(0.0) as u32;
    let x2 = (norm_box.x2 * img_w as f32).min(img_w as f32) as u32;
    let y2 = (norm_box.y2 * img_h as f32).min(img_h as f32) as u32;

    let w = x2.saturating_sub(x1).max(1);
    let h = y2.saturating_sub(y1).max(1);

    img.crop_imm(x1, y1, w, h)
}

#[inline]
fn crop_vehicle_box(img: &DynamicImage, bbox: &BoundingBox) -> DynamicImage {
    let (img_w, img_h) = img.dimensions();
    let x1 = (bbox.x1.max(0.0) as u32).min(img_w);
    let y1 = (bbox.y1.max(0.0) as u32).min(img_h);
    let x2 = (bbox.x2.max(0.0) as u32).min(img_w);
    let y2 = (bbox.y2.max(0.0) as u32).min(img_h);

    let w = x2.saturating_sub(x1).max(1);
    let h = y2.saturating_sub(y1).max(1);

    img.crop_imm(x1, y1, w, h)
}