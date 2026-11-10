use crate::config::ResolverConfig;
use road_graph_loader::{EdgeDescriptor, UnifiedHighwayTier};

#[inline]
pub fn junction_deflection_deg(e1: &EdgeDescriptor, e2: &EdgeDescriptor) -> f32 {
    if e1.geometry.len() < 2 || e2.geometry.len() < 2 {
        return 0.0;
    }
    let p1 = e1.geometry[e1.geometry.len() - 2];
    let p2 = e1.geometry[e1.geometry.len() - 1];
    let p3 = e2.geometry[0];
    let p4 = e2.geometry[1];

    let theta1 = (p2[0] - p1[0]).atan2(p2[1] - p1[1]).to_degrees() as f32;
    let theta2 = (p4[0] - p3[0]).atan2(p4[1] - p3[1]).to_degrees() as f32;

    let mut diff = (theta2 - theta1) % 360.0;
    if diff > 180.0 { diff -= 360.0; }
    if diff < -180.0 { diff += 360.0; }
    diff
}

/// Evaluates turn friction for Left-Hand-Drive networks
#[inline]
pub fn junction_delay_and_penalty(
    deflection: f32,
    jam_factor: f32,
    config: &ResolverConfig,
) -> (f32, f32) {
    let abs_d = deflection.abs();
    if abs_d <= 20.0 {
        (config.turn_straight_delay_sec, 0.0)
    } else if deflection > 20.0 && deflection <= 100.0 {
        (config.turn_left_delay_base_sec + config.turn_left_jam_multiplier * jam_factor, -0.2)
    } else if deflection >= -100.0 && deflection < -20.0 {
        (config.turn_right_delay_base_sec + config.turn_right_jam_multiplier * jam_factor, -1.0)
    } else {
        (config.turn_u_turn_delay_base_sec + config.turn_u_turn_jam_multiplier * jam_factor, -4.5)
    }
}

/// Computes penalty for unobserved cameras using dynamic sub-linear scaling and floor clamping.
/// This prevents high consecutive dropout rates from penalizing legitimate arterial corridors.
#[inline]
pub fn negative_camera_penalty(unobserved_count: usize, config: &ResolverConfig) -> f32 {
    if unobserved_count == 0 {
        return 0.0;
    }

    // Sub-linear scaling: accounts for environmental correlation across consecutive cameras
    let effective_misses = (unobserved_count as f32).powf(config.camera_miss_sublinear_exponent);
    let penalty = effective_misses * config.camera_miss_probability.ln();

    // Clamp against the soft floor (e.g. -8.0) so camera absences do not override structural road hierarchy
    penalty.max(config.max_camera_miss_penalty)
}

/// Evaluates two-component mixture for dwell / pause separation
#[inline]
pub fn evaluate_dwell_mixture(
    observed_dt: f32,
    expected_t: f32,
    sigma: f32,
    config: &ResolverConfig,
) -> (bool, f32, f32) {
    let excess = observed_dt - expected_t;
    let is_dwell = excess > (config.dwell_sigma_threshold * sigma).max(config.dwell_min_slack_sec);
    let time_residual = if is_dwell { 0.0 } else { excess };
    let ln_p_time = -0.5 * (2.0 * std::f32::consts::PI * sigma * sigma).ln()
        - (time_residual * time_residual) / (2.0 * sigma * sigma);

    (is_dwell, excess.max(0.0), ln_p_time)
}

/// Hierarchy penalty for stepping down arterial ranks into backstreets
#[inline]
pub fn hierarchy_transition_penalty(from_tier: UnifiedHighwayTier, to_tier: UnifiedHighwayTier) -> f32 {
    let rank = |t: UnifiedHighwayTier| match t {
        UnifiedHighwayTier::Motorway => 1,
        UnifiedHighwayTier::Trunk => 2,
        UnifiedHighwayTier::Primary => 3,
        UnifiedHighwayTier::Secondary => 4,
        UnifiedHighwayTier::Tertiary => 5,
        UnifiedHighwayTier::Residential => 6,
        UnifiedHighwayTier::Service | UnifiedHighwayTier::LivingStreet => 7,
        _ => 5,
    };
    let step = rank(to_tier) - rank(from_tier);
    if step > 0 {
        -(step as f32).powi(2) * 1.1
    } else {
        0.0
    }
}