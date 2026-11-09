use std::f64::consts::PI;

pub const EARTH_RADIUS_M: f64 = 6_371_000.0;
const DEG_TO_RAD: f64 = PI / 180.0;
const RAD_TO_DEG: f64 = 180.0 / PI;

/// Haversine distance using sine identities and minimized trig evaluations.
/// Accurate over planetary distances.
#[inline]
pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let phi1 = lat1 * DEG_TO_RAD;
    let phi2 = lat2 * DEG_TO_RAD;
    let dphi = (lat2 - lat1) * DEG_TO_RAD;
    let dlambda = (lon2 - lon1) * DEG_TO_RAD;

    let sin_half_dphi = (dphi * 0.5).sin();
    let sin_half_dlambda = (dlambda * 0.5).sin();

    let a = sin_half_dphi * sin_half_dphi + phi1.cos() * phi2.cos() * sin_half_dlambda * sin_half_dlambda;
    let c = 2.0 * a.clamp(0.0, 1.0).sqrt().asin();

    EARTH_RADIUS_M * c
}

/// Fast equirectangular flat-earth approximation for small scales (e.g. city edges < 20 km).
/// 4x to 6x faster than Haversine with < 0.05% error margin within Mumbai latitude.
#[inline(always)]
pub fn fast_planar_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let mean_lat = ((lat1 + lat2) * 0.5) * DEG_TO_RAD;
    let cos_lat = mean_lat.cos();

    let m_per_deg_lat = 111_132.92;
    let m_per_deg_lon = 111_412.84 * cos_lat;

    let dy = (lat2 - lat1) * m_per_deg_lat;
    let dx = (lon2 - lon1) * m_per_deg_lon;

    (dx * dx + dy * dy).sqrt()
}

/// Initial bearing calculation in degrees [0.0, 360.0).
#[inline]
pub fn initial_bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let phi1 = lat1 * DEG_TO_RAD;
    let phi2 = lat2 * DEG_TO_RAD;
    let dlambda = (lon2 - lon1) * DEG_TO_RAD;

    let y = dlambda.sin() * phi2.cos();
    let x = phi1.cos() * phi2.sin() - phi1.sin() * phi2.cos() * dlambda.cos();

    let brng = y.atan2(x) * RAD_TO_DEG;
    (brng + 360.0) % 360.0
}

/// Smallest angular difference between two bearings in degrees [0.0, 180.0].
#[inline(always)]
pub fn angle_diff(a: f64, b: f64) -> f64 {
    let diff = (a - b).abs() % 360.0;
    if diff > 180.0 {
        360.0 - diff
    } else {
        diff
    }
}