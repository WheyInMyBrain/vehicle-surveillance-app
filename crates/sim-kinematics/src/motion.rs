use chrono::{DateTime, Duration, Utc};
use rand::{Rng, RngExt};
use road_network_core::geo::{angle_diff, initial_bearing};
use road_network_core::{HighwayType, RoadSurface};
use serde::{Deserialize, Serialize};

/// High-frequency spatiotemporal state point along the vehicle trajectory
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KinematicEphemerisPoint {
    pub timestamp: DateTime<Utc>,
    pub lat: f64,
    pub lon: f64,
    pub speed_kmh: f32,
    pub bearing_deg: f32,
    pub acceleration_mps2: f32,
}

/// Dynamic behavioral profile of the driver
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DriverProfile {
    Cautious,
    Standard,
    Aggressive,
    DeliveryCourier,
}

impl DriverProfile {
    #[inline]
    pub fn speed_compliance_multiplier(&self) -> f32 {
        match self {
            DriverProfile::Cautious => 0.88,
            DriverProfile::Standard => 1.00,
            DriverProfile::Aggressive => 1.25,
            DriverProfile::DeliveryCourier => 0.95,
        }
    }

    #[inline]
    pub fn max_accel_mps2(&self) -> f32 {
        match self {
            DriverProfile::Cautious => 1.4,
            DriverProfile::Standard => 2.2,
            DriverProfile::Aggressive => 3.8,
            DriverProfile::DeliveryCourier => 2.0,
        }
    }

    #[inline]
    pub fn max_braking_mps2(&self) -> f32 {
        match self {
            DriverProfile::Cautious => 2.0,
            DriverProfile::Standard => 3.5,
            DriverProfile::Aggressive => 5.5,
            DriverProfile::DeliveryCourier => 3.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MicroStepResult {
    pub transit_duration_secs: f64,
    pub exit_speed_kmh: f32,
    pub camera_capture_speed_kmh: f32,
    pub ephemeris_trace: Vec<KinematicEphemerisPoint>,
    pub experienced_micro_stop: bool,
}

/// Calculate turn speed cap based on bearing deflection
#[inline(always)]
pub fn calculate_turn_speed_cap(prev_bearing: Option<f32>, current_bearing: f32) -> f32 {
    if let Some(pb) = prev_bearing {
        let diff = angle_diff(pb as f64, current_bearing as f64) as f32;
        if diff > 105.0 {
            10.0
        } else if diff > 75.0 {
            16.0
        } else if diff > 45.0 {
            24.0
        } else if diff > 20.0 {
            38.0
        } else {
            120.0
        }
    } else {
        120.0
    }
}

/// Spline/linear interpolation of coordinates along polyline geometry
#[inline]
pub fn interpolate_edge_position(geometry: &[[f64; 2]], fraction: f32) -> (f64, f64, f32) {
    if geometry.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    if geometry.len() == 1 || fraction <= 0.0 {
        return (geometry[0][0], geometry[0][1], 0.0);
    }
    if fraction >= 1.0 {
        let last = geometry[geometry.len() - 1];
        let prev = geometry[geometry.len() - 2];
        let b = initial_bearing(prev[0], prev[1], last[0], last[1]) as f32;
        return (last[0], last[1], b);
    }

    let total_segs = geometry.len() - 1;
    let seg_float = fraction * (total_segs as f32);
    let seg_idx = (seg_float.floor() as usize).min(total_segs - 1);
    let sub_frac = (seg_float - seg_idx as f32).clamp(0.0, 1.0) as f64;

    let p0 = geometry[seg_idx];
    let p1 = geometry[seg_idx + 1];

    let lat = p0[0] + (p1[0] - p0[0]) * sub_frac;
    let lon = p0[1] + (p1[1] - p0[1]) * sub_frac;
    let bearing = initial_bearing(p0[0], p0[1], p1[0], p1[1]) as f32;

    (lat, lon, bearing)
}

/// Physical simulation across an edge actively driven by dynamic jam factors & tensor speeds
pub fn simulate_edge_kinematics<R: Rng + RngExt>(
    length_m: f32,
    maxspeed_kmh: u8,
    highway_tier: HighwayType,
    surface: RoadSurface,
    geometry: &[[f64; 2]],
    prev_bearing: Option<f32>,
    current_bearing: f32,
    tensor_flow_speed_kmh: f32,
    jam_factor: f32,
    initial_speed_kmh: f32,
    has_camera: bool,
    driver: DriverProfile,
    start_time: &DateTime<Utc>,
    rng: &mut R,
) -> MicroStepResult {
    let friction = surface.friction_multiplier();
    let turn_cap = calculate_turn_speed_cap(prev_bearing, current_bearing);
    let compliance = driver.speed_compliance_multiplier();

    // Jam Factor Impact on Speed Ceiling:
    // When jam_factor approaches 10, speed drops towards minimum crawl
    let congestion_dampener = ((10.0 - jam_factor.clamp(0.0, 10.0)) / 10.0).powf(0.85).max(0.12);

    let target_ceiling_kmh = ((tensor_flow_speed_kmh * compliance * congestion_dampener)
        .min(maxspeed_kmh as f32 * compliance)
        .min(turn_cap)
        * friction)
        .clamp(4.0, 130.0);

    const TARGET_SLICE_M: f32 = 15.0;
    let num_slices = ((length_m / TARGET_SLICE_M).ceil() as usize).max(1);
    let slice_dist_m = length_m / (num_slices as f32);

    let mut current_speed_kmh = initial_speed_kmh.min(target_ceiling_kmh).max(4.0);
    let mut total_time_secs = 0.0f64;
    let mut camera_capture_speed = current_speed_kmh;
    let camera_trigger_slice = num_slices / 2;
    let mut ephemeris_trace = Vec::with_capacity(num_slices + 2);
    let mut experienced_micro_stop = false;

    let is_minor = matches!(
        highway_tier,
        HighwayType::Residential | HighwayType::LivingStreet | HighwayType::Service | HighwayType::Track
    );

    for slice in 0..num_slices {
        let frac_start = (slice as f32) / (num_slices as f32);
        let (lat, lon, seg_bearing) = interpolate_edge_position(geometry, frac_start);

        // 1. Gridlock and Accordion Shockwaves Driven by jam_factor
        if jam_factor >= 8.0 && rng.random_bool(0.16) {
            let halt_secs = rng.random_range(12.0..35.0);
            total_time_secs += halt_secs as f64;
            current_speed_kmh = 0.0;
            experienced_micro_stop = true;

            let halt_ts = *start_time + Duration::milliseconds((total_time_secs * 1000.0) as i64);
            ephemeris_trace.push(KinematicEphemerisPoint {
                timestamp: halt_ts,
                lat,
                lon,
                speed_kmh: 0.0,
                bearing_deg: seg_bearing,
                acceleration_mps2: 0.0,
            });
        } else if jam_factor >= 4.5 && rng.random_bool(0.12) {
            current_speed_kmh = rng.random_range(4.0..10.0);
            total_time_secs += rng.random_range(2.0..6.0) as f64;
        } else if is_minor && rng.random_bool(0.015) {
            let halt_secs = rng.random_range(15.0..45.0);
            total_time_secs += halt_secs as f64;
            current_speed_kmh = 0.0;
            experienced_micro_stop = true;

            let halt_ts = *start_time + Duration::milliseconds((total_time_secs * 1000.0) as i64);
            ephemeris_trace.push(KinematicEphemerisPoint {
                timestamp: halt_ts,
                lat,
                lon,
                speed_kmh: 0.0,
                bearing_deg: seg_bearing,
                acceleration_mps2: 0.0,
            });
        }

        // 2. Markov Acceleration / Deceleration
        let accel_jitter = if current_speed_kmh < target_ceiling_kmh * 0.85 {
            rng.random_range(0.8..3.2)
        } else if current_speed_kmh > target_ceiling_kmh {
            rng.random_range(-4.2..-1.2)
        } else {
            let spread = (1.5 + (jam_factor * 0.3)).min(4.0);
            rng.random_range(-spread..spread)
        };

        let old_speed_mps = (current_speed_kmh / 3.6).max(0.5);
        current_speed_kmh = (current_speed_kmh + accel_jitter).clamp(3.5, target_ceiling_kmh);
        let new_speed_mps = (current_speed_kmh / 3.6).max(0.5);

        let slice_avg_mps = (old_speed_mps + new_speed_mps) * 0.5;
        let slice_duration_sec = (slice_dist_m / slice_avg_mps) as f64;
        let slice_accel_mps2 = (new_speed_mps - old_speed_mps) / (slice_duration_sec as f32).max(0.1);

        total_time_secs += slice_duration_sec;
        let point_timestamp = *start_time + Duration::milliseconds((total_time_secs * 1000.0) as i64);

        // 3. ANPR Snapshot
        if has_camera && slice == camera_trigger_slice {
            let radar_noise = rng.random_range(-1.2..1.2);
            camera_capture_speed = (current_speed_kmh + radar_noise).max(3.0);
        }

        ephemeris_trace.push(KinematicEphemerisPoint {
            timestamp: point_timestamp,
            lat,
            lon,
            speed_kmh: current_speed_kmh,
            bearing_deg: seg_bearing,
            acceleration_mps2: slice_accel_mps2,
        });
    }

    MicroStepResult {
        transit_duration_secs: total_time_secs,
        exit_speed_kmh: current_speed_kmh,
        camera_capture_speed_kmh: camera_capture_speed,
        ephemeris_trace,
        experienced_micro_stop,
    }
}