use image::{DynamicImage, GenericImageView};
use vehicle_detector::VehicleClass;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    New,
    Tracked,
    Lost,
    Removed,
}

#[derive(Debug, Clone)]
pub struct TrackedVehicle {
    pub track_id: u64,
    pub class: VehicleClass,
    pub confidence: f32,
    pub bbox: [f32; 4], // [x1, y1, x2, y2]
    pub tracklet_len: usize,
    pub age: usize,
}

impl TrackedVehicle {
    #[inline(always)]
    pub fn x1(&self) -> f32 { self.bbox[0] }
    #[inline(always)]
    pub fn y1(&self) -> f32 { self.bbox[1] }
    #[inline(always)]
    pub fn x2(&self) -> f32 { self.bbox[2] }
    #[inline(always)]
    pub fn y2(&self) -> f32 { self.bbox[3] }

    #[inline(always)]
    pub fn width(&self) -> f32 {
        (self.bbox[2] - self.bbox[0]).max(0.0)
    }

    #[inline(always)]
    pub fn height(&self) -> f32 {
        (self.bbox[3] - self.bbox[1]).max(0.0)
    }

    #[inline(always)]
    pub fn center(&self) -> (f32, f32) {
        (
            self.bbox[0] + self.width() * 0.5,
            self.bbox[1] + self.height() * 0.5,
        )
    }

    /// Crops the tracked vehicle out of the frame
    pub fn crop_from(&self, img: &DynamicImage) -> DynamicImage {
        let (img_w, img_h) = img.dimensions();

        let x = self.bbox[0].clamp(0.0, img_w as f32).floor() as u32;
        let y = self.bbox[1].clamp(0.0, img_h as f32).floor() as u32;

        let max_w = (img_w.saturating_sub(x)) as f32;
        let max_h = (img_h.saturating_sub(y)) as f32;

        let w = self.width().min(max_w).round() as u32;
        let h = self.height().min(max_h).round() as u32;

        img.crop_imm(x, y, w.max(1), h.max(1))
    }
}