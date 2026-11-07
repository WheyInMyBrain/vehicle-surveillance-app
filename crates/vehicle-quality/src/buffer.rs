//! Deterministic, bounded track quality manager.

use crate::audit::QualityAuditor;
use crate::core::types::{GoldenCropMetadata, QualityBreakdown};
use heapless::Vec as StackVec;
use image::{DynamicImage, GenericImageView};
use vehicle_detector::{BoundingBox, VehicleClass};
use vehicle_tracker::TrackedVehicle;

pub const MAX_TRACK_BUFFERS: usize = 32;
pub const MAX_TOP_K: usize = 3;

/// Full golden crop holding pixel memory only when retained.
#[derive(Clone)]
pub struct GoldenCrop {
    pub frame_idx: usize,
    pub track_id: u64,
    pub class: VehicleClass,
    pub crop: DynamicImage,
    pub bbox: BoundingBox,
    pub metrics: QualityBreakdown,
}

impl core::fmt::Debug for GoldenCrop {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GoldenCrop")
            .field("frame_idx", &self.frame_idx)
            .field("track_id", &self.track_id)
            .field("class", &self.class.as_str())
            .field("composite_score", &self.metrics.composite_score)
            .field("resolution", &self.metrics.resolution)
            .finish()
    }
}

/// Single bounded entry per active tracklet.
#[derive(Clone)]
struct TrackQualitySlot {
    track_id: u64,
    active: bool,
    candidates: StackVec<GoldenCrop, MAX_TOP_K>,
}

impl TrackQualitySlot {
    const fn new() -> Self {
        Self {
            track_id: 0,
            active: false,
            candidates: StackVec::new(),
        }
    }
}

pub struct TrackQualityManager {
    auditor: QualityAuditor,
    min_score_threshold: f32,
    slots: [TrackQualitySlot; MAX_TRACK_BUFFERS],
}

impl TrackQualityManager {
    pub fn new(auditor: QualityAuditor, min_score_threshold: f32) -> Self {
        const INIT_SLOT: TrackQualitySlot = TrackQualitySlot::new();
        Self {
            auditor,
            min_score_threshold,
            slots: [INIT_SLOT; MAX_TRACK_BUFFERS],
        }
    }

    /// Evaluates vehicle quality without crop extraction unless score qualifies.
    pub fn process_frame(
        &mut self,
        frame_idx: usize,
        tracked: &TrackedVehicle,
        full_frame: &DynamicImage,
    ) -> Option<GoldenCropMetadata> {
        let (fw, fh) = full_frame.dimensions();
        let rgb_frame = match full_frame {
            DynamicImage::ImageRgb8(rgb) => rgb.as_raw(),
            _ => return None,
        };

        // 1. Zero-Allocation direct evaluation
        let metrics = self
            .auditor
            .evaluate_direct(rgb_frame, fw, fh, &tracked.bbox);

        if !metrics.is_ocr_viable || metrics.composite_score < self.min_score_threshold {
            return None;
        }

        let slot_idx = self.get_or_assign_slot(tracked.track_id)?;
        let slot = &mut self.slots[slot_idx];

        // 2. Check if candidate qualifies for TOP-K before performing image crop
        let should_insert = if slot.candidates.len() < MAX_TOP_K {
            true
        } else if let Some(worst) = slot.candidates.last() {
            metrics.composite_score > worst.metrics.composite_score
        } else {
            false
        };

        if !should_insert {
            return None;
        }

        // 3. Extract crop only when confirmed in Top-K
        let crop = crop_bounding_box(full_frame, &tracked.bbox);
        let candidate = GoldenCrop {
            frame_idx,
            track_id: tracked.track_id,
            class: tracked.class,
            crop,
            bbox: tracked.bbox,
            metrics,
        };

        let metadata = GoldenCropMetadata {
            frame_idx,
            track_id: tracked.track_id,
            class: tracked.class,
            bbox: tracked.bbox,
            metrics,
        };

        if slot.candidates.is_full() {
            let _ = slot.candidates.pop();
        }
        let _ = slot.candidates.push(candidate);

        slot.candidates.sort_unstable_by(|a, b| {
            b.metrics
                .composite_score
                .partial_cmp(&a.metrics.composite_score)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        Some(metadata)
    }

    #[inline]
    pub fn get_best_crop(&self, track_id: u64) -> Option<&GoldenCrop> {
        self.find_slot(track_id)
            .and_then(|idx| self.slots[idx].candidates.first())
    }

    #[inline]
    pub fn get_top_k_crops(&self, track_id: u64) -> &[GoldenCrop] {
        if let Some(idx) = self.find_slot(track_id) {
            &self.slots[idx].candidates
        } else {
            &[]
        }
    }

    /// Retires a track and frees its slot.
    pub fn finalize_track(&mut self, track_id: u64) -> Option<StackVec<GoldenCrop, MAX_TOP_K>> {
        if let Some(idx) = self.find_slot(track_id) {
            let slot = &mut self.slots[idx];
            slot.active = false;
            let mut out = StackVec::new();
            while let Some(c) = slot.candidates.pop() {
                let _ = out.push(c);
            }
            out.reverse();
            Some(out)
        } else {
            None
        }
    }

    /// Garbage collection based on active tracker track IDs.
    pub fn retain_active_tracks(&mut self, active_track_ids: &[u64]) {
        for slot in &mut self.slots {
            if slot.active && !active_track_ids.contains(&slot.track_id) {
                slot.active = false;
                slot.candidates.clear();
            }
        }
    }

    #[inline(always)]
    fn find_slot(&self, track_id: u64) -> Option<usize> {
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.active && slot.track_id == track_id {
                return Some(i);
            }
        }
        None
    }

    fn get_or_assign_slot(&mut self, track_id: u64) -> Option<usize> {
        if let Some(idx) = self.find_slot(track_id) {
            return Some(idx);
        }

        // Find empty slot
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if !slot.active {
                slot.active = true;
                slot.track_id = track_id;
                slot.candidates.clear();
                return Some(i);
            }
        }

        None
    }
}

#[inline]
fn crop_bounding_box(img: &DynamicImage, bbox: &BoundingBox) -> DynamicImage {
    let (img_w, img_h) = img.dimensions();
    let x1 = (bbox.x1.max(0.0) as u32).min(img_w);
    let y1 = (bbox.y1.max(0.0) as u32).min(img_h);
    let x2 = (bbox.x2.max(0.0) as u32).min(img_w);
    let y2 = (bbox.y2.max(0.0) as u32).min(img_h);

    let w = x2.saturating_sub(x1).max(1);
    let h = y2.saturating_sub(y1).max(1);

    img.crop_imm(x1, y1, w, h)
}