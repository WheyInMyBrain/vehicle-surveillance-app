use crate::audit::QualityAuditor;
use crate::types::GoldenCrop;
use image::DynamicImage;
use std::collections::HashMap;
use vehicle_tracker::TrackedVehicle;

pub struct TrackQualityManager {
    auditor: QualityAuditor,
    top_k: usize,
    min_score_threshold: f32,
    track_buffers: HashMap<u64, Vec<GoldenCrop>>,
}

impl TrackQualityManager {
    pub fn new(auditor: QualityAuditor, top_k: usize, min_score_threshold: f32) -> Self {
        Self {
            auditor,
            top_k: top_k.max(1),
            min_score_threshold,
            track_buffers: HashMap::new(),
        }
    }

    pub fn process_frame(
        &mut self,
        frame_idx: usize,
        tracked_vehicle: &TrackedVehicle,
        full_frame: &DynamicImage,
    ) -> Option<GoldenCrop> {
        let crop = tracked_vehicle.crop_from(full_frame);
        let (fw, fh) = (full_frame.width(), full_frame.height());

        let metrics = self.auditor.evaluate(&crop, &tracked_vehicle.bbox, fw, fh);

        // Strict OCR viability gate + score threshold
        if !metrics.is_ocr_viable || metrics.composite_score < self.min_score_threshold {
            return None;
        }

        let candidate = GoldenCrop {
            frame_idx,
            track_id: tracked_vehicle.track_id,
            class: tracked_vehicle.class,
            crop,
            bbox: tracked_vehicle.bbox,
            metrics,
        };

        let buffer = self
            .track_buffers
            .entry(tracked_vehicle.track_id)
            .or_insert_with(|| Vec::with_capacity(self.top_k + 1));

        buffer.push(candidate.clone());
        buffer.sort_unstable_by(|a, b| {
            b.metrics
                .composite_score
                .partial_cmp(&a.metrics.composite_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if buffer.len() > self.top_k {
            buffer.truncate(self.top_k);
        }

        Some(candidate)
    }

    pub fn get_best_crop(&self, track_id: u64) -> Option<&GoldenCrop> {
        self.track_buffers.get(&track_id).and_then(|b| b.first())
    }

    pub fn get_top_k_crops(&self, track_id: u64) -> &[GoldenCrop] {
        self.track_buffers
            .get(&track_id)
            .map(|b| b.as_slice())
            .unwrap_or(&[])
    }

    pub fn finalize_track(&mut self, track_id: u64) -> Option<Vec<GoldenCrop>> {
        self.track_buffers.remove(&track_id)
    }

    pub fn retain_active_tracks(&mut self, active_track_ids: &[u64]) {
        self.track_buffers
            .retain(|id, _| active_track_ids.contains(id));
    }
}