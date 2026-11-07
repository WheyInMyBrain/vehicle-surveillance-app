//! Aerospace-grade End-to-End Surveillance & Telemetry Pipeline Crate.

pub mod annotate;
pub mod channel;
pub mod config;
pub mod consensus;
pub mod engine;
pub mod types;

pub use annotate::FrameAnnotator;
pub use channel::{LprConsumer, LprProducer, LprTask, DISPATCH_QUEUE_CAPACITY};
pub use config::SurveillancePipelineConfig;
pub use consensus::TemporalPlateVoter;
pub use engine::{PipelineError, VehiclePipeline};
pub use types::{PipelineFrameResult, PipelineStageTimings, VehicleSnapshot};

#[cfg(target_os = "macos")]
#[no_mangle]
pub extern "C" fn __isPlatformVersionAtLeast(
    _platform: u32,
    _major: u32,
    _minor: u32,
    _subminor: u32,
) -> i32 {
    1
}