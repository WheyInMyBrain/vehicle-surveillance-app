pub mod fleet;
pub mod generator;
pub mod sensor_index;

pub use fleet::{simulate_fleet, FleetSimulationConfig, FleetSimulationResult};
pub use generator::{
    compute_traffic_aware_path, generate_vehicle_trip, SimulatedTrip, TripMode,
};
pub use sensor_index::CameraSensorIndex;