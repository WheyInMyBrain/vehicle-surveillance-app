use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolverConfig {
    // Physical Kinematic Limits
    pub max_lateral_accel_mps2: f32,
    pub max_longitudinal_accel_mps2: f32,
    pub max_braking_decel_mps2: f32,
    pub max_speed_tolerance_factor: f32,
    pub min_crawl_speed_mps: f32,

    // Isochrone Spatiotemporal Envelope
    pub isochrone_time_budget_multiplier: f32,
    pub max_corridor_speed_mps: f32,

    // Turn & Junction Friction (Left-Hand Drive)
    pub turn_straight_delay_sec: f32,
    pub turn_left_delay_base_sec: f32,
    pub turn_left_jam_multiplier: f32,
    pub turn_right_delay_base_sec: f32,
    pub turn_right_jam_multiplier: f32,
    pub turn_u_turn_delay_base_sec: f32,
    pub turn_u_turn_jam_multiplier: f32,

    // Negative Information & Dwell Mixture
    pub camera_miss_probability: f32,
    pub camera_miss_sublinear_exponent: f32, // Sub-linear scaling for correlated sensor dropouts (e.g. 0.5)
    pub max_camera_miss_penalty: f32,        // Soft floor to prevent over-penalizing monitored arterials
    pub dwell_sigma_threshold: f32,
    pub dwell_min_slack_sec: f32,

    // Dynamic Candidate Diversity (K proportional to Δt_obs)
    pub min_k_candidates: usize,
    pub max_k_candidates: usize,
    pub k_expansion_interval_sec: f32, // Add +1 candidate for every X seconds of gap time

    // Consensus Priors
    pub consensus_bonus_triple: f32,
    pub consensus_bonus_double: f32,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        Self {
            max_lateral_accel_mps2: 3.5,
            max_longitudinal_accel_mps2: 2.4,
            max_braking_decel_mps2: 3.8,
            max_speed_tolerance_factor: 1.35,
            min_crawl_speed_mps: 2.5,

            isochrone_time_budget_multiplier: 1.30,
            max_corridor_speed_mps: 33.33, // 120 km/h

            turn_straight_delay_sec: 0.5,
            turn_left_delay_base_sec: 3.0,
            turn_left_jam_multiplier: 0.4,
            turn_right_delay_base_sec: 14.0,
            turn_right_jam_multiplier: 2.5,
            turn_u_turn_delay_base_sec: 45.0,
            turn_u_turn_jam_multiplier: 4.0,

            camera_miss_probability: 0.04,
            camera_miss_sublinear_exponent: 0.5, // sqrt(unobserved_count)
            max_camera_miss_penalty: -8.0,       // Never penalize worse than -8.0 log-likelihood
            dwell_sigma_threshold: 3.5,
            dwell_min_slack_sec: 90.0,

            min_k_candidates: 2,
            max_k_candidates: 6,
            k_expansion_interval_sec: 60.0,

            consensus_bonus_triple: 2.6,
            consensus_bonus_double: 1.2,
        }
    }
}