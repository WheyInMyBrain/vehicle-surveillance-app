//! Aerospace-grade Vehicle Optical Quality Audit Crate.
//!
//! Provides deterministic, zero-allocation optical quality auditing and top-k golden crop selection.

pub mod audit;
pub mod buffer;
pub mod core;

pub use audit::{QualityAuditor, QualityConfig};
pub use buffer::{GoldenCrop, TrackQualityManager, MAX_TOP_K, MAX_TRACK_BUFFERS};
pub use core::metrics::{
    clamped_pixel_bounds, compute_laplacian_variance_direct, compute_luma_statistics_direct,
    rgb_to_luma,
};
pub use core::types::{GoldenCropMetadata, QualityBreakdown};