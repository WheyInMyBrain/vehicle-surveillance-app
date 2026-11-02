use image::{DynamicImage, GenericImageView};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum VehicleClass {
    Car = 2,
    Motorcycle = 3,
    Bus = 5,
    Truck = 7,
}

impl VehicleClass {
    #[inline]
    pub const fn from_coco_id(id: usize) -> Option<Self> {
        match id {
            2 => Some(Self::Car),
            3 => Some(Self::Motorcycle),
            5 => Some(Self::Bus),
            7 => Some(Self::Truck),
            _ => None,
        }
    }

    #[inline]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Car => "car",
            Self::Motorcycle => "motorcycle",
            Self::Bus => "bus",
            Self::Truck => "truck",
        }
    }
}

/// Bounding box representation in absolute coordinates: `[x1, y1, x2, y2]`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectedVehicle {
    pub class: VehicleClass,
    pub confidence: f32,
    pub bbox: [f32; 4],
}

impl DetectedVehicle {
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

    #[inline(always)]
    pub fn center(&self) -> (f32, f32) {
        (
            (self.bbox[0] + self.bbox[2]) * 0.5,
            (self.bbox[1] + self.bbox[3]) * 0.5,
        )
    }

    /// Crops the vehicle bounding box from the frame safely with zero panics.
    pub fn crop_from(&self, img: &DynamicImage) -> DynamicImage {
        let (img_w, img_h) = img.dimensions();

        let x1 = self.bbox[0].max(0.0).min(img_w as f32) as u32;
        let y1 = self.bbox[1].max(0.0).min(img_h as f32) as u32;
        let x2 = self.bbox[2].max(0.0).min(img_w as f32) as u32;
        let y2 = self.bbox[3].max(0.0).min(img_h as f32) as u32;

        let w = x2.saturating_sub(x1).max(1);
        let h = y2.saturating_sub(y1).max(1);

        img.crop_imm(x1, y1, w, h)
    }
}