use crate::error::ParsingStatistics;
use crate::scanner::ScanResult;
use ahash::AHashMap;
use road_network_core::{
    geo::{haversine_distance, initial_bearing},
    CityGraph, EdgeMeta, NodeId, NodeMeta,
};

pub fn build_city_graph(scan: ScanResult) -> (CityGraph, ParsingStatistics) {
    let mut city = CityGraph::new();
    let mut edges_added = 0usize;
    let raw_ways_len = scan.raw_ways.len();

    let mut surface_counts = AHashMap::new();
    let mut speed_tier_counts = AHashMap::new();
    let mut lane_distribution = AHashMap::new();
    let mut explicit_maxspeed = 0usize;
    let mut untagged_maxspeed = 0usize;
    let mut flyovers = 0usize;
    let mut tunnels = 0usize;

    // Detect true graph intersections (nodes shared by multiple distinct roads)
    let mut node_degree: AHashMap<NodeId, usize> = AHashMap::with_capacity(scan.node_metadata.len());
    for way in &scan.raw_ways {
        for &nid in &way.node_refs {
            *node_degree.entry(nid).or_insert(0) += 1;
        }
    }

    for way in scan.raw_ways {
        if let Some(spd) = way.maxspeed_kmh {
            explicit_maxspeed += 1;
            let speed_bin = if spd <= 20.0 {
                "≤ 20 km/h".to_string()
            } else if spd <= 40.0 {
                "21 - 40 km/h".to_string()
            } else if spd <= 60.0 {
                "41 - 60 km/h".to_string()
            } else {
                "> 60 km/h".to_string()
            };
            *speed_tier_counts.entry(speed_bin).or_insert(0) += 1;
        } else {
            untagged_maxspeed += 1;
        }

        *surface_counts.entry(way.surface).or_insert(0) += 1;
        let effective_lanes = way.lanes_forward.or(way.lanes_backward);
        *lane_distribution.entry(effective_lanes).or_insert(0) += 1;

        if way.is_bridge || way.layer > 0 {
            flyovers += 1;
        }
        if way.is_tunnel || way.layer < 0 {
            tunnels += 1;
        }

        for window in way.node_refs.windows(2) {
            let u_id = window[0];
            let v_id = window[1];

            let u_node = match scan.node_metadata.get(&u_id) {
                Some(c) => c,
                None => continue,
            };
            let v_node = match scan.node_metadata.get(&v_id) {
                Some(c) => c,
                None => continue,
            };

            let dist_m = haversine_distance(u_node.lat, u_node.lon, v_node.lat, v_node.lon);
            // Ignore degenerate micro-jitter edges under 30 cm
            if dist_m < 0.3 {
                continue;
            }

            let u_is_junction = *node_degree.get(&u_id).unwrap_or(&0) >= 3;
            let v_is_junction = *node_degree.get(&v_id).unwrap_or(&0) >= 3;

            city.add_or_get_node(
                u_id,
                NodeMeta {
                    id: u_id,
                    lat: u_node.lat,
                    lon: u_node.lon,
                    has_traffic_signal: u_node.has_signal,
                    has_speed_bump: u_node.has_bump,
                    is_intersection: u_is_junction,
                },
            );

            city.add_or_get_node(
                v_id,
                NodeMeta {
                    id: v_id,
                    lat: v_node.lat,
                    lon: v_node.lon,
                    has_traffic_signal: v_node.has_signal,
                    has_speed_bump: v_node.has_bump,
                    is_intersection: v_is_junction,
                },
            );

            let forward_bearing = initial_bearing(u_node.lat, u_node.lon, v_node.lat, v_node.lon);
            let reverse_bearing = (forward_bearing + 180.0) % 360.0;

            if way.reversed_oneway {
                let edge = EdgeMeta::new(
                    way.name.clone(),
                    dist_m,
                    way.highway,
                    way.maxspeed_kmh,
                    way.lanes_backward,
                    way.surface,
                    reverse_bearing,
                    true,
                    way.is_bridge,
                    way.is_tunnel,
                    way.layer,
                );
                if city.add_edge(v_id, u_id, edge).is_ok() {
                    edges_added += 1;
                }
            } else if way.oneway {
                let edge = EdgeMeta::new(
                    way.name.clone(),
                    dist_m,
                    way.highway,
                    way.maxspeed_kmh,
                    way.lanes_forward,
                    way.surface,
                    forward_bearing,
                    true,
                    way.is_bridge,
                    way.is_tunnel,
                    way.layer,
                );
                if city.add_edge(u_id, v_id, edge).is_ok() {
                    edges_added += 1;
                }
            } else {
                let fwd_edge = EdgeMeta::new(
                    way.name.clone(),
                    dist_m,
                    way.highway,
                    way.maxspeed_kmh,
                    way.lanes_forward,
                    way.surface,
                    forward_bearing,
                    false,
                    way.is_bridge,
                    way.is_tunnel,
                    way.layer,
                );
                let rev_edge = EdgeMeta::new(
                    way.name.clone(),
                    dist_m,
                    way.highway,
                    way.maxspeed_kmh,
                    way.lanes_backward,
                    way.surface,
                    reverse_bearing,
                    false,
                    way.is_bridge,
                    way.is_tunnel,
                    way.layer,
                );

                if city.add_edge(u_id, v_id, fwd_edge).is_ok() {
                    edges_added += 1;
                }
                if city.add_edge(v_id, u_id, rev_edge).is_ok() {
                    edges_added += 1;
                }
            }
        }
    }

    let stats = ParsingStatistics {
        ways_scanned: scan.ways_scanned,
        drivable_ways_extracted: raw_ways_len,
        nodes_retained: city.graph.node_count(),
        total_directed_edges: edges_added,
        traffic_signals_found: scan.signals_count,
        speed_bumps_found: scan.bumps_count,
        surface_counts,
        explicit_maxspeed_tags: explicit_maxspeed,
        untagged_maxspeed,
        speed_tier_counts,
        lane_distribution,
        elevated_flyover_edges: flyovers,
        underground_tunnel_edges: tunnels,
    };

    (city, stats)
}