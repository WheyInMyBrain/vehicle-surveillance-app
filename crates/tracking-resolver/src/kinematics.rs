use crate::config::ResolverConfig;
use road_graph_loader::{EdgeDescriptor, UnifiedHighwayTier};

/// Resolves legal statutory or OSM-tagged max speed in km/h
#[inline(always)]
pub fn resolve_legal_maxspeed_kmh(desc: &EdgeDescriptor) -> f32 {
    let statutory = match desc.tier {
        UnifiedHighwayTier::Motorway => 100.0,
        UnifiedHighwayTier::Trunk => 80.0,
        UnifiedHighwayTier::Primary => 60.0,
        UnifiedHighwayTier::Secondary => 50.0,
        UnifiedHighwayTier::Tertiary => 40.0,
        UnifiedHighwayTier::Residential => 30.0,
        UnifiedHighwayTier::LivingStreet | UnifiedHighwayTier::Service => 20.0,
        UnifiedHighwayTier::Unclassified => 30.0,
        UnifiedHighwayTier::Other => 25.0,
    };
    if desc.maxspeed_kmh > 0 {
        (desc.maxspeed_kmh as f32).max(10.0)
    } else {
        statutory
    }
}

/// Computes discrete curvature κ = 1/R across polyline vertices and returns max curvature
#[inline]
pub fn compute_max_curvature_inv_m(geometry: &[[f64; 2]]) -> f32 {
    if geometry.len() < 3 {
        return 0.0;
    }
    let mut max_k = 0.0f32;
    for w in geometry.windows(3) {
        let p1 = w[0];
        let p2 = w[1];
        let p3 = w[2];

        // Menger curvature calculation using meters conversion
        let a = ((p2[0] - p1[0]).hypot(p2[1] - p1[1]) * 111_139.0) as f32;
        let b = ((p3[0] - p2[0]).hypot(p3[1] - p2[1]) * 111_139.0) as f32;
        let c = ((p3[0] - p1[0]).hypot(p3[1] - p1[1]) * 111_139.0) as f32;

        let s = (a + b + c) * 0.5;
        let area_sq = s * (s - a) * (s - b) * (s - c);
        if area_sq > 1e-4 {
            let area = area_sq.sqrt();
            let k = (4.0 * area) / (a * b * c).max(1.0);
            if k > max_k {
                max_k = k;
            }
        }
    }
    max_k
}

/// Physical ceiling velocity clamped by curve curvature radius and statutory limits
#[inline(always)]
pub fn edge_velocity_ceiling_mps(desc: &EdgeDescriptor, config: &ResolverConfig) -> f32 {
    let v_legal_mps = (resolve_legal_maxspeed_kmh(desc) * config.max_speed_tolerance_factor) / 3.6;
    let max_k = compute_max_curvature_inv_m(&desc.geometry);

    let v_curve_mps = if max_k > 1e-4 {
        (config.max_lateral_accel_mps2 / max_k).sqrt()
    } else {
        v_legal_mps
    };

    v_legal_mps.min(v_curve_mps).max(config.min_crawl_speed_mps)
}

/// Absolute physical minimum transit duration across an individual edge
#[inline(always)]
pub fn min_physical_edge_duration(
    desc: &EdgeDescriptor,
    free_flow_kmh: u8,
    config: &ResolverConfig,
) -> f32 {
    let ceil_mps = edge_velocity_ceiling_mps(desc, config);
    let flow_mps = (free_flow_kmh as f32 * config.max_speed_tolerance_factor) / 3.6;
    let eff_mps = ceil_mps.max(flow_mps).max(config.min_crawl_speed_mps);
    desc.length_m / eff_mps
}

/// Bang-bang forward-backward velocity profile over a multi-edge path
pub fn compute_path_min_physical_duration(
    edge_descriptors: &[&EdgeDescriptor],
    v_start_mps: f32,
    v_end_mps: f32,
    config: &ResolverConfig,
) -> f32 {
    let n = edge_descriptors.len();
    if n == 0 {
        return 0.0;
    }

    let mut v_caps = Vec::with_capacity(n);
    let mut lengths = Vec::with_capacity(n);
    for desc in edge_descriptors {
        v_caps.push(edge_velocity_ceiling_mps(desc, config));
        lengths.push(desc.length_m);
    }

    // Forward pass: Acceleration constrained
    let mut v_fwd = vec![0.0f32; n];
    let mut v_curr = v_start_mps.min(v_caps[0]);
    for i in 0..n {
        v_curr = v_curr.min(v_caps[i]);
        v_fwd[i] = v_curr;
        v_curr = (v_curr.powi(2) + 2.0 * config.max_longitudinal_accel_mps2 * lengths[i]).sqrt();
    }

    // Backward pass: Braking constrained
    let mut v_bwd = vec![0.0f32; n];
    v_curr = v_end_mps.min(v_caps[n - 1]);
    for i in (0..n).rev() {
        v_curr = v_curr.min(v_caps[i]);
        v_bwd[i] = v_curr;
        v_curr = (v_curr.powi(2) + 2.0 * config.max_braking_decel_mps2 * lengths[i]).sqrt();
    }

    // Numerical integration along the physical boundary envelope
    let mut t_min = 0.0f32;
    for i in 0..n {
        let v_eff = v_fwd[i].min(v_bwd[i]).max(config.min_crawl_speed_mps);
        t_min += lengths[i] / v_eff;
    }
    t_min
}

/// Structural delay capacity of an intermediate intersection
#[inline(always)]
pub fn structural_delay_capacity(out_degree: usize) -> f32 {
    if out_degree >= 4 {
        15.0
    } else if out_degree == 3 {
        8.0
    } else {
        2.0
    }
}