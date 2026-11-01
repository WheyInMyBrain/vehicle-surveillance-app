use ahash::AHashMap;
use engine_core::NodeId;

#[derive(Debug, Clone)]
pub struct PhysicalCameraPole {
    pub camera_id: String,
    pub lat: f64,
    pub lon: f64,
}

/// Fast lookup mapping directed edges (u, v) to physical camera sensors.
#[derive(Debug, Clone)]
pub struct CameraNetwork {
    pub edge_to_camera: AHashMap<(NodeId, NodeId), PhysicalCameraPole>,
}

impl CameraNetwork {
    pub fn new() -> Self {
        Self {
            edge_to_camera: AHashMap::new(),
        }
    }

    /// Deploys a camera pole on an exact directed edge (u, v).
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

    /// Deploys a camera pole across all sub-edges of a contracted macro-corridor.
    /// Guarantees that passing ANY segment of this non-branching road triggers the camera.
    pub fn deploy_corridor_pole(
        &mut self,
        camera_id: String,
        raw_edges: &[(NodeId, NodeId)],
        lat: f64,
        lon: f64,
        is_two_way: bool,
    ) {
        let pole = PhysicalCameraPole {
            camera_id,
            lat,
            lon,
        };
        for &(u, v) in raw_edges {
            self.edge_to_camera.insert((u, v), pole.clone());
            if is_two_way {
                self.edge_to_camera.insert((v, u), pole.clone());
            }
        }
    }

    #[inline(always)]
    pub fn get_sensor(&self, u: NodeId, v: NodeId) -> Option<&PhysicalCameraPole> {
        self.edge_to_camera.get(&(u, v))
    }

    pub fn total_monitored_edges(&self) -> usize {
        self.edge_to_camera.len()
    }
}

impl Default for CameraNetwork {
    fn default() -> Self {
        Self::new()
    }
}