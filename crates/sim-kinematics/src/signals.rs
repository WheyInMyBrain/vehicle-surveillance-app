use chrono::{DateTime, Duration, Utc};
use rand::{Rng, RngExt};

pub struct SignalController {
    cycle_duration_secs: i64,
    green_duration_secs: i64,
    phase_offset_secs: i64,
}

impl SignalController {
    #[inline]
    pub fn for_node<R: Rng + RngExt>(node_id: u64, rng: &mut R) -> Self {
        let jitter = rng.random_range(-5..5);
        let cycle = (75 + ((node_id % 37) as i64) + jitter).clamp(50, 130);
        let green_ratio = rng.random_range(0.42..0.58);
        let green = (cycle as f64 * green_ratio).round() as i64;
        let offset = ((node_id ^ 0x5DEECE66D) % (cycle as u64)) as i64;

        Self {
            cycle_duration_secs: cycle,
            green_duration_secs: green,
            phase_offset_secs: offset,
        }
    }

    #[inline(always)]
    pub fn compute_wait_seconds(&self, arrival_time: &DateTime<Utc>) -> i64 {
        let current_sec = arrival_time.timestamp();
        let phase = (current_sec + self.phase_offset_secs).rem_euclid(self.cycle_duration_secs);

        if phase < self.green_duration_secs {
            0 // Green light: clear passage
        } else {
            self.cycle_duration_secs - phase // Red light: wait for green
        }
    }
}

/// Applies real intersection physical delays (turning queues, lights, speed tables)
#[inline]
pub fn apply_junction_delays<R: Rng + RngExt>(
    node_id: u64,
    has_traffic_signal: bool,
    has_speed_bump: bool,
    is_aggressive_driver: bool,
    current_time: &mut DateTime<Utc>,
    rng: &mut R,
) -> f64 {
    let mut total_delay_sec = 0.0;

    if has_traffic_signal {
        // Aggressive drivers occasionally run yellow/red lights if wait is short
        let runs_red = is_aggressive_driver && rng.random_bool(0.08);
        if !runs_red {
            let signal = SignalController::for_node(node_id, rng);
            let wait_secs = signal.compute_wait_seconds(current_time);
            if wait_secs > 0 {
                // Stochastic queue delay at the front of the light
                let queue_creep = rng.random_range(1..6);
                let actual_wait = wait_secs + queue_creep;
                *current_time = *current_time + Duration::seconds(actual_wait);
                total_delay_sec += actual_wait as f64;
            }
        }
    }

    if has_speed_bump {
        let bump_ms = if is_aggressive_driver {
            rng.random_range(1200..2500) // Hard brake over bump
        } else {
            rng.random_range(2800..5500) // Careful gentle crawl
        };
        *current_time = *current_time + Duration::milliseconds(bump_ms);
        total_delay_sec += bump_ms as f64 / 1000.0;
    }

    total_delay_sec
}