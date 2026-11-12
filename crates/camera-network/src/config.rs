use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraSensorConfig {
    pub sensor_resolution_px: (u32, u32),
    pub focal_length_mm: f32,
    pub hfov_deg: f32,
    pub vfov_deg: f32,
    pub mounting_height_m: f32,
    pub camera_pitch_deg: f32,
    pub ocr_sweet_spot_min_m: f32,
    pub ocr_sweet_spot_max_m: f32,
    pub max_corridor_width_per_camera_m: f32,
    pub lane_width_m: f32,
    pub lanes_per_camera: u8,
}

impl Default for CameraSensorConfig {
    fn default() -> Self {
        Self {
            sensor_resolution_px: (3840, 2160),
            focal_length_mm: 12.0,
            hfov_deg: 32.0,
            vfov_deg: 18.0,
            mounting_height_m: 5.8,
            camera_pitch_deg: 23.0,
            ocr_sweet_spot_min_m: 16.0,
            ocr_sweet_spot_max_m: 28.0,
            max_corridor_width_per_camera_m: 8.5,
            lane_width_m: 3.5,
            lanes_per_camera: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareBOMConfig {
    pub cost_verge_pole: f64,
    pub cost_overhead_gantry: f64,
    pub cost_intersection_mast: f64,
    pub cost_camera_head: f64,
    pub cost_radar_unit: f64,
    pub cost_edge_compute: f64,
    pub cost_installation_civil_works: f64,
}

impl Default for HardwareBOMConfig {
    fn default() -> Self {
        Self {
            cost_verge_pole: 45_000.0,
            cost_overhead_gantry: 250_000.0,
            cost_intersection_mast: 80_000.0,
            cost_camera_head: 35_000.0,
            cost_radar_unit: 25_000.0,
            cost_edge_compute: 20_000.0,
            cost_installation_civil_works: 30_000.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub max_dark_distance_m: f32,
    pub max_dark_hops: usize,
    pub audit_trips: usize,
    pub lanes_per_camera: u8,
    pub sensor: CameraSensorConfig,
    pub bom: HardwareBOMConfig,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            max_dark_distance_m: 500.0,
            max_dark_hops: 6,
            audit_trips: 10_000,
            lanes_per_camera: 2,
            sensor: CameraSensorConfig::default(),
            bom: HardwareBOMConfig::default(),
        }
    }
}