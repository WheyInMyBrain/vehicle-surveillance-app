use chrono::Utc;
use engine_core::{CameraSighting, CityGraph, EdgeMeta, HighwayType, NodeMeta};
use trajectory_resolver::StreamingTracker;

fn build_toy_city() -> CityGraph {
    let mut city = CityGraph::new();
    city.add_or_get_node(1, NodeMeta { lat: 19.070, lon: 72.870 });
    city.add_or_get_node(2, NodeMeta { lat: 19.072, lon: 72.870 });
    city.add_or_get_node(3, NodeMeta { lat: 19.074, lon: 72.870 });

    let edge_spec = EdgeMeta {
        length_m: 200.0,
        highway_type: HighwayType::Primary,
        bearing_deg: 0.0,
    };

    city.add_edge(1, 2, edge_spec.clone());
    city.add_edge(2, 3, edge_spec);
    city
}

#[test]
fn test_streaming_ingestion() {
    let city = build_toy_city();
    let mut tracker = StreamingTracker::new(&city);

    let t0 = Utc::now();
    let s1 = CameraSighting {
        vehicle_id: "MH02CB1234".into(),
        camera_id: "CAM_01".into(),
        edge_from: 1,
        edge_to: 2,
        timestamp: t0,
        speed_kmh: 40.0,
    };

    tracker.ingest_sighting(s1);
    let track = tracker.get_track("MH02CB1234").expect("Vehicle not registered");
    assert_eq!(track.primary_route, vec![1, 2]);
}