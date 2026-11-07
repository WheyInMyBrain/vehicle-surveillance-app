//! Domain primitives for multi-object vehicle tracking.

use vehicle_detector::{BoundingBox, VehicleClass};

/// Discrete operational states of an aerospace tracklet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TrackState {
    /// Newly spawned tracklet awaiting temporal confirmation.
    New = 0,
    /// Confirmed actively tracking vehicle.
    Tracked = 1,
    /// Tracklet lost sight of target; executing kinematic dead-reckoning.
    Lost = 2,
    /// Terminated tracklet retired from active memory pool.
    Removed = 3,
}

/// Fully validated tracking output for downstream stages (Plate Detection & OCR).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TrackedVehicle {
    pub track_id: u64,
    pub class: VehicleClass,
    pub confidence: f32,
    pub bbox: BoundingBox,
    pub tracklet_len: usize,
    pub age: usize,
}

impl TrackedVehicle {
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
    pub fn center(&self) -> (f32, f32) { self.bbox.center() }

    #[inline(always)]
    pub fn to_array(&self) -> [f32; 4] { self.bbox.as_array() }
}