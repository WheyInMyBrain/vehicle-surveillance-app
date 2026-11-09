use ahash::AHashMap;
use camera_network_optimizer::CameraPole;
use road_network_core::NodeId;

#[derive(Debug, Clone)]
pub struct CameraSensorIndex {
    edge_to_camera: AHashMap<(NodeId, NodeId), CameraPole>,
}

impl CameraSensorIndex {
    pub fn from_poles(poles: Vec<CameraPole>) -> Self {
        let mut edge_to_camera = AHashMap::with_capacity(poles.len() * 2);

        for pole in poles {
            edge_to_camera.insert((pole.edge_from, pole.edge_to), pole.clone());
            if pole.is_two_way {
                edge_to_camera.insert((pole.edge_to, pole.edge_from), pole);
            }
        }

        Self { edge_to_camera }
    }

    #[inline(always)]
    pub fn get_camera(&self, u: NodeId, v: NodeId) -> Option<&CameraPole> {
        self.edge_to_camera.get(&(u, v))
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.edge_to_camera.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.edge_to_camera.is_empty()
    }
}