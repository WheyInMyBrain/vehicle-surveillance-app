use chrono::{DateTime, Duration, Utc};
use engine_core::NodeMeta;
use rand::{Rng, RngExt};

pub struct SignalController {
    cycle_duration_secs: i64,
    green_duration_secs: i64,
    phase_offset_secs: i64,
}

impl SignalController {
    pub fn for_node<R: Rng>(node_id: u64, rng: &mut R) -> Self {
        let cycle = rng.random_range(70..110);
        let green_ratio = rng.random_range(0.45..0.60);
        let green = (cycle as f64 * green_ratio).round() as i64;
        let offset = (node_id % cycle as u64) as i64;

        Self {
            cycle_duration_secs: cycle,
            green_duration_secs: green,
            phase_offset_secs: offset,
        }
    }

    /// Computes the exact delay in seconds upon arriving at this signal at `arrival_time`.
    pub fn compute_wait_seconds(&self, arrival_time: DateTime<Utc>) -> i64 {
        let current_sec = arrival_time.timestamp();
        let phase = (current_sec + self.phase_offset_secs) % self.cycle_duration_secs;

        if phase < self.green_duration_secs {
            0 // Green light: proceed without stopping
        } else {
            // Red light: wait for the cycle to roll over to green
            self.cycle_duration_secs - phase
        }
    }
}

pub fn apply_node_delays<R: Rng>(
    node_meta: &NodeMeta,
    node_id: u64,
    current_time: &mut DateTime<Utc>,
    rng: &mut R,
) {
    // 1. Traffic signal phase calculation
    if node_meta.has_traffic_signal {
        let controller = SignalController::for_node(node_id, rng);
        let wait_secs = controller.compute_wait_seconds(*current_time);
        if wait_secs > 0 {
            *current_time = *current_time + Duration::seconds(wait_secs);
        }
    }

    // 2. Speed bump / table delay
    if node_meta.has_speed_bump {
        let bump_delay_ms = rng.random_range(3000..6000);
        *current_time = *current_time + Duration::milliseconds(bump_delay_ms);
    }
}