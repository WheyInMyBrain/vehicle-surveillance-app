pub mod motion;
pub mod signals;

pub use motion::{
    calculate_turn_speed_cap, interpolate_edge_position, simulate_edge_kinematics,
    DriverProfile, KinematicEphemerisPoint, MicroStepResult,
};
pub use signals::{apply_junction_delays, SignalController};