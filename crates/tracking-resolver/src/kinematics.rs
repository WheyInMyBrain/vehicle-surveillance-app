use road_graph_loader::{EdgeDescriptor, UnifiedHighwayTier};

/// Absolute physical minimum transit time across an edge.
/// Any trip completing the edge in less than this requires breaking physical laws.
#[inline(always)]
pub fn min_physical_edge_duration(desc: &EdgeDescriptor, free_flow_kmh: u8) -> f32 {
    let statutory_max = match desc.tier {
        UnifiedHighwayTier::Motorway => 100.0,
        UnifiedHighwayTier::Trunk => 80.0,
        UnifiedHighwayTier::Primary => 60.0,
        UnifiedHighwayTier::Secondary => 50.0,
        UnifiedHighwayTier::Tertiary => 40.0,
        UnifiedHighwayTier::Residential => 30.0,
        UnifiedHighwayTier::LivingStreet => 20.0,
        UnifiedHighwayTier::Service => 20.0,
        _ => 25.0,
    };

    let max_speed_kmh = (desc.maxspeed_kmh as f32)
        .max(free_flow_kmh as f32)
        .min(statutory_max)
        .max(10.0);

    let speed_mps = max_speed_kmh / 3.6;
    desc.length_m / speed_mps
}

/// Structural delay capacity of an intermediate intersection.
/// Explains why a vehicle took extra time without hallucinating detours.
#[inline(always)]
pub fn structural_delay_capacity(out_degree: usize) -> f32 {
    // Basic node: ~2s variance. Major junction (3+ branches): yielding, pedestrian & signal delays
    if out_degree >= 4 {
        15.0
    } else if out_degree == 3 {
        8.0
    } else {
        2.0
    }
}