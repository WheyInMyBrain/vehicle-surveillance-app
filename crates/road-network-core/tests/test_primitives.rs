use road_network_core::{
    fast_planar_distance, haversine_distance, initial_bearing, CityGraph, EdgeMeta, HighwayType,
    NodeMeta, RoadSurface,
};

#[test]
fn test_geo_precision_and_speed() {
    // BKC Junction coordinates
    let lat1 = 19.06318;
    let lon1 = 72.82460;
    let lat2 = 19.06355;
    let lon2 = 72.82467;

    let dist_haversine = haversine_distance(lat1, lon1, lat2, lon2);
    let dist_planar = fast_planar_distance(lat1, lon1, lat2, lon2);
    let bearing = initial_bearing(lat1, lon1, lat2, lon2);

    assert!(dist_haversine > 40.0 && dist_haversine < 45.0);
    assert!((dist_haversine - dist_planar).abs() < 0.05, "Planar error exceeds 5cm tolerance");
    assert!(bearing >= 0.0 && bearing <= 360.0);
}

#[test]
fn test_compact_graph_construction() {
    let mut city = CityGraph::new();

    let u_idx = city.add_or_get_node(
        1001,
        NodeMeta {
            id: 1001,
            lat: 19.06318,
            lon: 72.82460,
            has_traffic_signal: true,
            has_speed_bump: false,
            is_intersection: true,
        },
    );

    let v_idx = city.add_or_get_node(
        1002,
        NodeMeta {
            id: 1002,
            lat: 19.06355,
            lon: 72.82467,
            has_traffic_signal: false,
            has_speed_bump: false,
            is_intersection: false,
        },
    );

    let edge = EdgeMeta::new(
        Some("BKC Avenue".to_string()),
        41.5,
        HighwayType::Primary,
        Some(50.0),
        Some(3),
        RoadSurface::Asphalt,
        10.5,
        true,
        false,
        false,
        0,
    );

    assert!(city.add_edge(1001, 1002, edge).is_ok());
    assert_eq!(city.get_node_id(u_idx), 1001);
    assert_eq!(city.get_node_id(v_idx), 1002);
    assert_eq!(city.node_count(), 2);
    assert_eq!(city.edge_count(), 1);
}