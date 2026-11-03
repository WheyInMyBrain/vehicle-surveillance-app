use image::{DynamicImage, GenericImageView};

/// Detected license plate bounding box scaled to the vehicle crop's coordinates
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectedPlate {
    pub confidence: f32,
    /// Absolute coordinates within the vehicle crop: [x1, y1, x2, y2]
    pub bbox: [f32; 4],
}

impl DetectedPlate {
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
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    /// Aspect ratio (width / height) — Standard automotive plates range between 2.0 and 5.2
    #[inline(always)]
    pub fn aspect_ratio(&self) -> f32 {
        let h = self.height();
        if h > 0.0 { self.width() / h } else { 0.0 }
    }

    /// Safely isolates and crops the license plate out of the vehicle crop
    pub fn crop_from(&self, vehicle_crop: &DynamicImage) -> DynamicImage {
        let (crop_w, crop_h) = vehicle_crop.dimensions();

        let x1 = self.bbox[0].max(0.0).min(crop_w as f32) as u32;
        let y1 = self.bbox[1].max(0.0).min(crop_h as f32) as u32;
        let x2 = self.bbox[2].max(0.0).min(crop_w as f32) as u32;
        let y2 = self.bbox[3].max(0.0).min(crop_h as f32) as u32;

        let w = x2.saturating_sub(x1).max(1);
        let h = y2.saturating_sub(y1).max(1);

        vehicle_crop.crop_imm(x1, y1, w, h)
    }
}