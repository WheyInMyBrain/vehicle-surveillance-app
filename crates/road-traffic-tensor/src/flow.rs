use crate::time::TemporalCoord;
use road_graph_loader::LinkFlowSnapshot;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TraversalStatus {
    Open {
        speed_mps: f32,
        duration_sec: f32,
        jam_factor: f32,
        is_empirical: bool,
    },
    Closed,
}

#[inline(always)]
pub fn interpolate_flow(
    curr: &LinkFlowSnapshot,
    next: &LinkFlowSnapshot,
    fraction: f32,
) -> (f32, f32) {
    let speed_kmh = (curr.speed_kmh as f32) * (1.0 - fraction) + (next.speed_kmh as f32) * fraction;
    let jam_factor = curr.jam_factor * (1.0 - fraction) + next.jam_factor * fraction;
    (speed_kmh, jam_factor)
}

#[inline(always)]
pub fn calculate_traversal_cost(
    length_m: f32,
    maxspeed_kmh: u8,
    curr: &LinkFlowSnapshot,
    next: &LinkFlowSnapshot,
    coord: &TemporalCoord,
) -> TraversalStatus {
    if curr.is_closed {
        return TraversalStatus::Closed;
    }

    let (dyn_speed_kmh, jam_factor) = interpolate_flow(curr, next, coord.fraction);

    // Dynamic speed clamped to legal speed limit and minimum crawl (1.5 m/s ~ 5.4 km/h)
    let speed_kmh = dyn_speed_kmh.min(maxspeed_kmh as f32).max(5.4);
    let speed_mps = speed_kmh / 3.6;
    let duration_sec = length_m / speed_mps;

    TraversalStatus::Open {
        speed_mps,
        duration_sec,
        jam_factor,
        is_empirical: curr.is_empirical,
    }
}