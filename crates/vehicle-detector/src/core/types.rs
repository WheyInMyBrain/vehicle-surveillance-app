//! Safety-critical domain primitives for vehicle classification and localization.
//!
//! # Invariants
//! - All bounding box coordinates satisfy `x1 <= x2` and `y1 <= y2`.
//! - Coordinates are finite (`!is_nan()` and `!is_infinite()`).
//! - All types are `Copy` and have deterministic stack memory footprints.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum VehicleClass {
    Car = 0,
    TwoWheeler = 1,
    AutoRickshaw = 2,
    Bus = 3,
    Truck = 4,
}

impl VehicleClass {
    /// Safe conversion from custom fine-tuned 5-class Indian traffic models.
    ///
    /// Indices:
    /// - `0`: Car
    /// - `1`: TwoWheeler (motorcycle, scooter, bicycle, rider)
    /// - `2`: AutoRickshaw
    /// - `3`: Bus
    /// - `4`: Truck (heavy truck, dumper, tempo, lorry)
    #[inline(always)]
    pub const fn from_custom_5class_id(id: usize) -> Option<Self> {
        match id {
            0 => Some(Self::Car),
            1 => Some(Self::TwoWheeler),
            2 => Some(Self::AutoRickshaw),
            3 => Some(Self::Bus),
            4 => Some(Self::Truck),
            _ => None,
        }
    }

    /// Strict conversion from standard 80-class MS COCO models (e.g. baseline yolo26n.pt).
    ///
    /// Indices:
    /// - `1`: bicycle -> TwoWheeler
    /// - `2`: car -> Car
    /// - `3`: motorcycle -> TwoWheeler
    /// - `5`: bus -> Bus
    /// - `7`: truck -> Truck
    /// - All other COCO classes (person, traffic light, dog, etc.) -> None
    #[inline(always)]
    pub const fn from_coco_id(id: usize) -> Option<Self> {
        match id {
            1 | 3 => Some(Self::TwoWheeler),
            2 => Some(Self::Car),
            5 => Some(Self::Bus),
            7 => Some(Self::Truck),
            _ => None,
        }
    }

    /// Resolves the class based on the model's total output class count.
    ///
    /// If `total_classes <= 5`, routes to the custom Indian traffic schema.
    /// If `total_classes > 5` (typically 80 for COCO), routes to the COCO schema.
    #[inline(always)]
    pub fn from_model_output(class_id: usize, total_classes: usize) -> Option<Self> {
        if total_classes <= 5 {
            Self::from_custom_5class_id(class_id)
        } else {
            Self::from_coco_id(class_id)
        }
    }

    #[inline(always)]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Car => "car",
            Self::TwoWheeler => "two_wheeler",
            Self::AutoRickshaw => "auto_rickshaw",
            Self::Bus => "bus",
            Self::Truck => "truck",
        }
    }
}

/// Axis-aligned bounding box defined by `[x1, y1, x2, y2]`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BoundingBox {
    #[inline(always)]
    pub fn new_clamped(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        let x1 = if x1.is_nan() { 0.0 } else { x1 };
        let y1 = if y1.is_nan() { 0.0 } else { y1 };
        let x2 = if x2.is_nan() { 0.0 } else { x2 };
        let y2 = if y2.is_nan() { 0.0 } else { y2 };

        let (min_x, max_x) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
        let (min_y, max_y) = if y1 < y2 { (y1, y2) } else { (y2, y1) };

        Self {
            x1: min_x.max(0.0),
            y1: min_y.max(0.0),
            x2: max_x.max(0.0),
            y2: max_y.max(0.0),
        }
    }

    #[inline(always)]
    pub fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    #[inline(always)]
    pub fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    #[inline(always)]
    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    #[inline(always)]
    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    }

    #[inline(always)]
    pub fn as_array(&self) -> [f32; 4] {
        [self.x1, self.y1, self.x2, self.y2]
    }
}

/// Fully validated vehicle detection result.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct DetectedVehicle {
    pub class: VehicleClass,
    pub confidence: f32,
    pub bbox: BoundingBox,
}

impl DetectedVehicle {
    #[inline(always)]
    pub fn x1(&self) -> f32 { self.bbox.x1 }

    #[inline(always)]
    pub fn y1(&self) -> f32 { self.bbox.y1 }

    #[inline(always)]
    pub fn x2(&self) -> f32 { self.bbox.x2 }

    #[inline(always)]
    pub fn y2(&self) -> f32 { self.bbox.y2 }

    #[inline(always)]
    pub fn width(&self) -> f32 { self.bbox.width() }

    #[inline(always)]
    pub fn height(&self) -> f32 { self.bbox.height() }

    #[inline(always)]
    pub fn area(&self) -> f32 { self.bbox.area() }
}