use engine_core::{EdgeMeta, HighwayType, RoadSurface};
use rand::{Rng, RngExt};

#[inline(always)]
pub fn get_statutory_speed_kmh(highway: HighwayType) -> f64 {
    match highway {
        HighwayType::Motorway => 70.0,
        HighwayType::Trunk => 60.0,
        HighwayType::Primary => 45.0,
        HighwayType::Secondary => 35.0,
        HighwayType::Tertiary => 25.0,
        HighwayType::Residential => 20.0,
        HighwayType::LivingStreet => 15.0,
        HighwayType::Unclassified => 22.0,
        HighwayType::Service => 12.0,
        HighwayType::Track => 10.0,
    }
}

#[inline(always)]
pub fn calculate_turn_speed_cap(prev_bearing: Option<f64>, current_bearing: f64) -> f64 {
    if let Some(pb) = prev_bearing {
        let diff = (current_bearing - pb + 180.0) % 360.0 - 180.0;
        let angle = (if diff < -180.0 { diff + 360.0 } else { diff }).abs();

        if angle > 90.0 {
            12.0
        } else if angle > 45.0 {
            22.0
        } else if angle > 20.0 {
            35.0
        } else {
            120.0
        }
    } else {
        120.0
    }
}

#[derive(Debug, Clone)]
pub struct EdgeTraversalResult {
    pub total_transit_seconds: f64,
    pub camera_instantaneous_speed_kmh: f64,
    pub final_exit_speed_kmh: f64,
}

pub fn simulate_edge_traversal<R: Rng>(
    edge: &EdgeMeta,
    prev_bearing: Option<f64>,
    gtfs_calibrated_speed: Option<f64>,
    initial_speed_kmh: f64,
    has_camera: bool,
    rng: &mut R,
) -> EdgeTraversalResult {
    let target_ceiling_kmh = if let Some(gtfs_speed) = gtfs_calibrated_speed {
        gtfs_speed
    } else if let Some(osm_max) = edge.maxspeed_kmh {
        osm_max
    } else {
        get_statutory_speed_kmh(edge.highway_type)
    };

    let surface_factor = match edge.surface {
        RoadSurface::Asphalt => 1.00,
        RoadSurface::Paved => 0.95,
        RoadSurface::Cobblestone => 0.75,
        RoadSurface::Unpaved => 0.55,
        RoadSurface::Unknown => match edge.highway_type {
            HighwayType::Motorway | HighwayType::Trunk | HighwayType::Primary => 1.00,
            HighwayType::Secondary | HighwayType::Tertiary => 0.95,
            _ => 0.85,
        },
    };

    let lane_factor = match edge.lanes {
        Some(l) if l >= 3 => 1.00,
        Some(2) => 0.92,
        Some(1) => 0.75,
        _ => 0.78,
    };

    let turn_cap = calculate_turn_speed_cap(prev_bearing, edge.bearing_deg);
    let effective_max_kmh = (target_ceiling_kmh * surface_factor * lane_factor)
        .min(turn_cap)
        .clamp(8.0, 90.0);

    const STEP_LEN_M: f64 = 40.0;
    let num_steps = ((edge.length_m / STEP_LEN_M).ceil() as usize).max(1);
    let sub_dist_m = edge.length_m / num_steps as f64;
    // Multiply by 3.6 to convert km/h to seconds directly: sub_dist_m / (kmh / 3.6) = (sub_dist_m * 3.6) / kmh
    let dist_conv = sub_dist_m * 3.6;

    let mut current_speed_kmh = initial_speed_kmh.min(effective_max_kmh).max(10.0);
    let mut total_time_secs = 0.0;
    let mut camera_speed_kmh = current_speed_kmh;
    let camera_trigger_step = num_steps / 2;

    let is_minor_road = matches!(
        edge.highway_type,
        HighwayType::Residential | HighwayType::LivingStreet | HighwayType::Service | HighwayType::Tertiary
    );

    for step in 0..num_steps {
        // Markov speed drift
        let speed_jitter: f64 = rng.random_range(-6.0..6.0);
        current_speed_kmh = (current_speed_kmh + speed_jitter).clamp(5.0, effective_max_kmh);

        // Stochastic halts and traffic crawling
        if is_minor_road && rng.random_bool(0.035) {
            total_time_secs += rng.random_range(12.0..35.0);
            current_speed_kmh = 8.0;
        } else if rng.random_bool(0.08) {
            current_speed_kmh = rng.random_range(8.0..18.0);
        }

        if has_camera && step == camera_trigger_step {
            let noise: f64 = rng.random_range(-2.0..2.0);
            camera_speed_kmh = (current_speed_kmh + noise).max(5.0);
        }

        total_time_secs += dist_conv / current_speed_kmh;
    }

    EdgeTraversalResult {
        total_transit_seconds: total_time_secs,
        camera_instantaneous_speed_kmh: camera_speed_kmh,
        final_exit_speed_kmh: current_speed_kmh,
    }
}