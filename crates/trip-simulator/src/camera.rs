use ahash::AHashMap;
use engine_core::NodeId;

#[derive(Debug, Clone)]
pub struct PhysicalCameraPole {
    pub camera_id: String,
    pub lat: f64,
    pub lon: f64,
}

/// Fast lookup mapping directed edges (u, v) to physical camera sensors.
pub struct CameraNetwork {
    pub edge_to_camera: AHashMap<(NodeId, NodeId), PhysicalCameraPole>,
}

impl CameraNetwork {
    pub fn new() -> Self {
        Self {
            edge_to_camera: AHashMap::new(),
        }
    }

    /// Deploys a camera pole. If is_two_way is true, registers both (u, v) and (v, u).
    pub fn deploy_pole(
        &mut self,
        camera_id: String,
        u: NodeId,
        v: NodeId,
        lat: f64,
        lon: f64,
        is_two_way: bool,
    ) {
        let pole = PhysicalCameraPole {
            camera_id,
            lat,
            lon,
        };
        self.edge_to_camera.insert((u, v), pole.clone());
        if is_two_way {
            self.edge_to_camera.insert((v, u), pole);
        }
    }

    pub fn get_sensor(&self, u: NodeId, v: NodeId) -> Option<&PhysicalCameraPole> {
        self.edge_to_camera.get(&(u, v))
    }
}

impl Default for CameraNetwork {
    fn default() -> Self {
        Self::new()
    }
}