pub mod anomaly;
pub mod causal_gap;
pub mod config;
pub mod consensus;
pub mod ensemble;
pub mod envelope;
pub mod kinematics;
pub mod likelihood;
pub mod tracker;

pub use anomaly::{AnomalyAlert, AnomalyType};
pub use causal_gap::{resolve_camera_gap, CausalGapWorkspace, TrajectoryCandidate};
pub use config::ResolverConfig;
pub use consensus::EvaluatedTrajectory;
pub use ensemble::RouteCandidate;
pub use kinematics::{
    compute_path_min_physical_duration, min_physical_edge_duration, resolve_legal_maxspeed_kmh,
    structural_delay_capacity,
};
pub use tracker::{ResolvedLeg, StreamingCausalTracker, TrackHypothesis, VehicleTrack};