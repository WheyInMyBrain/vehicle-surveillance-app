pub mod annotate;
pub mod config;
pub mod engine;
pub mod types;

pub use annotate::FrameAnnotator;
pub use config::SurveillancePipelineConfig;
pub use engine::{PipelineError, VehiclePipeline};
pub use types::{PipelineFrameResult, VehicleSnapshot};