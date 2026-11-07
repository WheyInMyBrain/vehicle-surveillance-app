//! Aerospace-grade Multi-Object Vehicle Tracking Crate.
//!
//! Provides deterministic, zero-allocation ByteTrack object tracking.

pub mod battrack;
pub mod core;
pub mod track;

pub use battrack::{calculate_diou, ByteTrackConfig, ByteTracker, MAX_DETECTIONS, MAX_TRACKS};
pub use core::kalman::{invert_4x4_analytic, KalmanFilter};
pub use core::lapjv::{lapjv_bounded, LapMatchResult, MAX_MATCH_DIM};
pub use core::types::{TrackState, TrackedVehicle};
pub use track::STrack;