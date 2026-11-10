use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalyType {
    LoiteringOrStashStop,
    ExcessiveSpeed,
    DarkCorridorEvasion,
    SensorClockAnomaly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyAlert {
    pub anomaly_type: AnomalyType,
    pub leg_index: usize,
    pub from_camera: String,
    pub to_camera: String,
    pub severity_score: f32, // 0.0 to 100.0
    pub explanation: String,
}

pub fn evaluate_leg_anomaly(
    from_camera: &str,
    to_camera: &str,
    leg_index: usize,
    observed_dt_sec: f32,
    min_physical_sec: f32,
    _expected_tensor_sec: f32,
    distance_m: f32,
    dark_alley_ratio: f32,
    is_dwell_detected: bool,
    excess_dwell_sec: f32,
) -> Option<AnomalyAlert> {
    // 1. Clock or Supersonic Anomaly
    if observed_dt_sec <= 0.5 && distance_m > 30.0 {
        return Some(AnomalyAlert {
            anomaly_type: AnomalyType::SensorClockAnomaly,
            leg_index,
            from_camera: from_camera.to_string(),
            to_camera: to_camera.to_string(),
            severity_score: 100.0,
            explanation: format!(
                "Impossible transit time ({:.2}s for {:.1}m) between poles",
                observed_dt_sec, distance_m
            ),
        });
    }

    // 2. Physical Speed Violation
    let avg_speed_kmh = (distance_m / observed_dt_sec.max(0.1)) * 3.6;
    if observed_dt_sec < min_physical_sec * 0.85 {
        return Some(AnomalyAlert {
            anomaly_type: AnomalyType::ExcessiveSpeed,
            leg_index,
            from_camera: from_camera.to_string(),
            to_camera: to_camera.to_string(),
            severity_score: ((min_physical_sec / observed_dt_sec.max(0.1)) * 50.0).clamp(50.0, 100.0),
            explanation: format!(
                "Speeding violation: transit speed {:.1} km/h violates curvature-clamped physical ceiling",
                avg_speed_kmh
            ),
        });
    }

    // 3. Loitering / Stash Stop Anomaly
    if is_dwell_detected && excess_dwell_sec > 120.0 {
        let excess_min = excess_dwell_sec / 60.0;
        return Some(AnomalyAlert {
            anomaly_type: AnomalyType::LoiteringOrStashStop,
            leg_index,
            from_camera: from_camera.to_string(),
            to_camera: to_camera.to_string(),
            severity_score: (excess_min * 12.0).clamp(35.0, 99.0),
            explanation: format!(
                "Unexplained halt/stash: {:.1} min pause decoupled from traffic tensor",
                excess_min
            ),
        });
    }

    // 4. Deliberate Dark Corridor Evasion
    if dark_alley_ratio > 0.65 && distance_m > 800.0 {
        return Some(AnomalyAlert {
            anomaly_type: AnomalyType::DarkCorridorEvasion,
            leg_index,
            from_camera: from_camera.to_string(),
            to_camera: to_camera.to_string(),
            severity_score: dark_alley_ratio * 80.0,
            explanation: format!(
                "{:.0}% of route diverted through dark unmonitored backalleys",
                dark_alley_ratio * 100.0
            ),
        });
    }

    None
}