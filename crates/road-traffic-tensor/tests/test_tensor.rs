use chrono::TimeZone;
use road_graph_loader::RoadGraph;
use road_traffic_tensor::{TemporalCoord, TrafficTensorEngine, TraversalStatus};
use std::path::PathBuf;

#[test]
fn test_temporal_flow_engine() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let bin_path = root.join("data/road/mumbai_complete_graph.bin");

    if !bin_path.exists() {
        return;
    }

    let graph = RoadGraph::open(&bin_path).expect("Failed to open road graph");
    let engine = TrafficTensorEngine::new(&graph);

    // Tuesday at 8:42:30 AM UTC
    let dt = chrono::Utc.with_ymd_and_hms(2026, 11, 10, 8, 42, 30).unwrap();
    let coord = TemporalCoord::from_datetime(&dt);

    assert_eq!(coord.day, 1); // Tuesday
    assert_eq!(coord.bin, 104); // 8:40 AM - 8:45 AM
    assert!((coord.fraction - 0.5).abs() < 1e-4); // Exactly halfway through the 5-min bin

    // Evaluate edge #0
    let status = engine.evaluate_edge_at(0, &dt);
    match status {
        TraversalStatus::Open { speed_mps, duration_sec, jam_factor, .. } => {
            assert!(speed_mps > 1.0);
            assert!(duration_sec > 0.0);
            assert!(jam_factor >= 0.0 && jam_factor <= 10.0);
        }
        TraversalStatus::Closed => {}
    }
}