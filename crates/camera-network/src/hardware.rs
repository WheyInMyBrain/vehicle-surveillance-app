use crate::config::{CameraSensorConfig, HardwareBOMConfig};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MountType {
    VergePole,
    OverheadGantry,
    IntersectionMast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicalSiteProfile {
    BidirectionalSinglePole,
    DirectionalCarriagewayGantry,
    ChokepointGatewayStation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpticalMetrics {
    pub mid_field_ppm_min: f32,
    pub mid_field_ppm_max: f32,
    pub sweet_spot_corridor_width_m: f32,
    pub optical_shadow_length_m: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareAssessment {
    pub profile: PhysicalSiteProfile,
    pub mount_type: MountType,
    pub camera_heads: u8,
    pub radar_units: u8,
    pub total_lanes: u8,
    pub estimated_cost: f64,
    pub optical: OpticalMetrics,
}

pub fn evaluate_hardware(
    lanes_forward: u8,
    lanes_backward: u8,
    is_two_way: bool,
    is_chokepoint: bool,
    sensor: &CameraSensorConfig,
    bom: &HardwareBOMConfig,
) -> HardwareAssessment {
    let pitch_rad = sensor.camera_pitch_deg.to_radians();
    let delta_h = (3.8f32.min(sensor.mounting_height_m - 0.5) - 0.5).max(0.5);
    let optical_shadow_length_m = delta_h / pitch_rad.tan();

    let mid_field_ppm_max = (sensor.sensor_resolution_px.0 as f32)
        / (sensor.ocr_sweet_spot_min_m * (sensor.hfov_deg.to_radians() * 0.5).tan() * 2.0);
    let mid_field_ppm_min = (sensor.sensor_resolution_px.0 as f32)
        / (sensor.ocr_sweet_spot_max_m * (sensor.hfov_deg.to_radians() * 0.5).tan() * 2.0);
    let sweet_spot_corridor_width_m =
        sensor.ocr_sweet_spot_min_m * (sensor.hfov_deg.to_radians() * 0.5).tan() * 2.0;

    let optical = OpticalMetrics {
        mid_field_ppm_min,
        mid_field_ppm_max,
        sweet_spot_corridor_width_m,
        optical_shadow_length_m,
    };

    let total_lanes = lanes_forward + if is_two_way { lanes_backward } else { 0 };

    if is_chokepoint {
        let fwd_cams = ((lanes_forward as f32) / (sensor.lanes_per_camera as f32)).ceil() as u8;
        let bwd_cams = if is_two_way {
            ((lanes_backward as f32) / (sensor.lanes_per_camera as f32)).ceil() as u8
        } else {
            0
        };
        let camera_heads = (fwd_cams + bwd_cams).max(2) * 2; // Dual-head redundancy for chokepoints
        let radar_units = 2;
        let mount_type = MountType::OverheadGantry;

        let estimated_cost = bom.cost_overhead_gantry
            + (camera_heads as f64 * bom.cost_camera_head)
            + (radar_units as f64 * bom.cost_radar_unit)
            + bom.cost_edge_compute
            + bom.cost_installation_civil_works;

        HardwareAssessment {
            profile: PhysicalSiteProfile::ChokepointGatewayStation,
            mount_type,
            camera_heads,
            radar_units,
            total_lanes,
            estimated_cost,
            optical,
        }
    } else if total_lanes >= 4 {
        let fwd_cams = ((lanes_forward as f32) / (sensor.lanes_per_camera as f32)).ceil() as u8;
        let bwd_cams = if is_two_way {
            ((lanes_backward as f32) / (sensor.lanes_per_camera as f32)).ceil() as u8
        } else {
            0
        };
        let camera_heads = (fwd_cams + bwd_cams).max(2);
        let radar_units = if is_two_way { 2 } else { 1 };
        let mount_type = MountType::OverheadGantry;

        let estimated_cost = bom.cost_overhead_gantry
            + (camera_heads as f64 * bom.cost_camera_head)
            + (radar_units as f64 * bom.cost_radar_unit)
            + bom.cost_edge_compute
            + bom.cost_installation_civil_works;

        HardwareAssessment {
            profile: PhysicalSiteProfile::DirectionalCarriagewayGantry,
            mount_type,
            camera_heads,
            radar_units,
            total_lanes,
            estimated_cost,
            optical,
        }
    } else {
        let camera_heads = if is_two_way { 2 } else { 1 };
        let radar_units = 1;
        let mount_type = MountType::VergePole;

        let estimated_cost = bom.cost_verge_pole
            + (camera_heads as f64 * bom.cost_camera_head)
            + (radar_units as f64 * bom.cost_radar_unit)
            + bom.cost_edge_compute
            + bom.cost_installation_civil_works;

        HardwareAssessment {
            profile: PhysicalSiteProfile::BidirectionalSinglePole,
            mount_type,
            camera_heads,
            radar_units,
            total_lanes,
            estimated_cost,
            optical,
        }
    }
}