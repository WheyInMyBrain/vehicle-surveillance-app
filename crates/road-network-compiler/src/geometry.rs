#[inline(always)]
pub fn calculate_bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let d_lon = (lon2 - lon1).to_radians();

    let y = d_lon.sin() * lat2_rad.cos();
    let x = lat1_rad.cos() * lat2_rad.sin() - lat1_rad.sin() * lat2_rad.cos() * d_lon.cos();
    let brng = y.atan2(x).to_degrees();
    (brng + 360.0) % 360.0
}

#[inline(always)]
pub fn angular_diff(a: f64, b: f64) -> f64 {
    let diff = (a - b).abs() % 360.0;
    if diff > 180.0 {
        360.0 - diff
    } else {
        diff
    }
}

/// Computes perpendicular distance from point P to line segment AB in meters
#[inline(always)]
pub fn point_to_segment_dist_m(
    p_lat: f64,
    p_lon: f64,
    a_lat: f64,
    a_lon: f64,
    b_lat: f64,
    b_lon: f64,
) -> f64 {
    let mid_lat = (a_lat + b_lat) / 2.0;
    let cos_lat = mid_lat.to_radians().cos();
    let m_per_lat = 111_132.92;
    let m_per_lon = 111_412.84 * cos_lat;

    let px = (p_lon - a_lon) * m_per_lon;
    let py = (p_lat - a_lat) * m_per_lat;
    let bx = (b_lon - a_lon) * m_per_lon;
    let by = (b_lat - a_lat) * m_per_lat;

    let len_sq = bx * bx + by * by;
    if len_sq < 1e-4 {
        return (px * px + py * py).sqrt();
    }

    let t = ((px * bx + py * by) / len_sq).clamp(0.0, 1.0);
    let dx = px - t * bx;
    let dy = py - t * by;
    (dx * dx + dy * dy).sqrt()
}

/// Zero-copy/fast coordinate parser for GeoJSON LineString coordinates: `[[lon1, lat1], ...]`
pub fn parse_geojson_coords(raw: &str) -> Vec<(f64, f64)> {
    let mut coords = Vec::with_capacity(8);
    let bytes = raw.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == b'[' {
            i += 1;
            while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t' || bytes[i] == b'\n') {
                i += 1;
            }
            if i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'-') {
                let start_lon = i;
                while i < bytes.len() && bytes[i] != b',' && bytes[i] != b']' {
                    i += 1;
                }
                let lon: f64 = std::str::from_utf8(&bytes[start_lon..i])
                    .unwrap_or("0")
                    .trim()
                    .parse()
                    .unwrap_or(0.0);

                if i < bytes.len() && bytes[i] == b',' {
                    i += 1;
                }
                while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
                    i += 1;
                }
                let start_lat = i;
                while i < bytes.len() && bytes[i] != b']' {
                    i += 1;
                }
                let lat: f64 = std::str::from_utf8(&bytes[start_lat..i])
                    .unwrap_or("0")
                    .trim()
                    .parse()
                    .unwrap_or(0.0);

                if lat != 0.0 && lon != 0.0 {
                    coords.push((lat, lon));
                }
            }
        }
        i += 1;
    }
    coords
}