pub mod anomaly;
pub mod causal_gap;
pub mod kinematics;
pub mod tracker;

pub use anomaly::{AnomalyAlert, AnomalyType};
pub use causal_gap::{resolve_camera_gap, CausalGapWorkspace, TrajectoryCandidate};
pub use kinematics::{min_physical_edge_duration, structural_delay_capacity};
pub use tracker::{ResolvedLeg, StreamingCausalTracker, TrackHypothesis, VehicleTrack};