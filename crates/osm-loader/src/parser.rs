use ahash::AHashMap;
use engine_core::{geo::haversine_distance, CityGraph, EdgeMeta, HighwayType, NodeId, NodeMeta};
use osmpbf::{Element, ElementReader};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OsmError {
    #[error("I/O or PBF parsing error: {0}")]
    PbfError(#[from] osmpbf::Error),
    #[error("Node not found during way assembly: {0}")]
    MissingNode(NodeId),
}

/// Helper to convert OSM highway string tags into our HighwayType enum.
fn parse_highway_tag(tag: &str) -> Option<HighwayType> {
    match tag {
        "motorway" | "motorway_link" => Some(HighwayType::Motorway),
        "trunk" | "trunk_link" => Some(HighwayType::Trunk),
        "primary" | "primary_link" => Some(HighwayType::Primary),
        "secondary" | "secondary_link" => Some(HighwayType::Secondary),
        "tertiary" | "tertiary_link" => Some(HighwayType::Tertiary),
        "residential" | "living_street" => Some(HighwayType::Residential),
        "unclassified" => Some(HighwayType::Unclassified),
        "service" => Some(HighwayType::Service),
        _ => None,
    }
}

/// Computes compass bearing in degrees [0, 360) from point 1 to point 2.
fn compute_bearing(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let phi1 = lat1.to_radians();
    let phi2 = lat2.to_radians();
    let delta_lambda = (lon2 - lon1).to_radians();

    let y = delta_lambda.sin() * phi2.cos();
    let x = phi1.cos() * phi2.sin() - phi1.sin() * phi2.cos() * delta_lambda.cos();
    let theta = y.atan2(x).to_degrees();

    (theta + 360.0) % 360.0
}

/// Loads drivable Mumbai roads from a .osm.pbf file.
/// Optional bounding box filter (min_lat, min_lon, max_lat, max_lon) to isolate target districts.
pub fn load_osm_pbf<P: AsRef<Path>>(
    pbf_path: P,
    bbox: Option<(f64, f64, f64, f64)>,
) -> Result<CityGraph, OsmError> {
    let reader = ElementReader::from_path(pbf_path)?;

    // Temporary storage for node coordinates using fast non-cryptographic AHashMap
    let mut coords: AHashMap<NodeId, (f64, f64)> = AHashMap::with_capacity(1_000_000);
    
    // Extracted candidate ways: (nodes_vec, highway_type, oneway)
    let mut drivable_ways: Vec<(Vec<NodeId>, HighwayType, bool)> = Vec::with_capacity(50_000);

    // Pass 1: Streaming scan over elements
    reader.for_each(|element| {
        match element {
            Element::Node(node) => {
                let id = node.id() as u64;
                let lat = node.lat();
                let lon = node.lon();

                let inside = match bbox {
                    Some((min_lat, min_lon, max_lat, max_lon)) => {
                        lat >= min_lat && lat <= max_lat && lon >= min_lon && lon <= max_lon
                    }
                    None => true,
                };

                if inside {
                    coords.insert(id, (lat, lon));
                }
            }
            Element::DenseNode(dense) => {
                let id = dense.id as u64;
                let lat = dense.lat();
                let lon = dense.lon();

                let inside = match bbox {
                    Some((min_lat, min_lon, max_lat, max_lon)) => {
                        lat >= min_lat && lat <= max_lat && lon >= min_lon && lon <= max_lon
                    }
                    None => true,
                };

                if inside {
                    coords.insert(id, (lat, lon));
                }
            }
            Element::Way(way) => {
                let mut highway_type = None;
                let mut oneway = false;

                for (key, val) in way.tags() {
                    if key == "highway" {
                        highway_type = parse_highway_tag(val);
                    } else if key == "oneway" && (val == "yes" || val == "1" || val == "true") {
                        oneway = true;
                    }
                }

                if let Some(hw) = highway_type {
                    let node_ids: Vec<NodeId> = way.refs().map(|id| id as u64).collect();
                    if node_ids.len() >= 2 {
                        drivable_ways.push((node_ids, hw, oneway));
                    }
                }
            }
            Element::Relation(_) => {}
        }
    })?;

    // Pass 2: Graph Assembly into PetGraph
    let mut city = CityGraph::new();

    for (node_ids, hw_type, oneway) in drivable_ways {
        // Only keep segments whose coordinates exist within the parsed bounding box
        for window in node_ids.windows(2) {
            let u_id = window[0];
            let v_id = window[1];

            if let (Some(&(u_lat, u_lon)), Some(&(v_lat, v_lon))) = (coords.get(&u_id), coords.get(&v_id)) {
                let length_m = haversine_distance(u_lat, u_lon, v_lat, v_lon);
                if length_m < 1.0 {
                    continue; // Skip zero-length duplicates
                }

                city.add_or_get_node(u_id, NodeMeta { lat: u_lat, lon: u_lon });
                city.add_or_get_node(v_id, NodeMeta { lat: v_lat, lon: v_lon });

                let forward_bearing = compute_bearing(u_lat, u_lon, v_lat, v_lon);
                city.add_edge(
                    u_id,
                    v_id,
                    EdgeMeta {
                        length_m,
                        highway_type: hw_type,
                        bearing_deg: forward_bearing,
                    },
                );

                // Two-way street: add reverse edge
                if !oneway {
                    let reverse_bearing = compute_bearing(v_lat, v_lon, u_lat, u_lon);
                    city.add_edge(
                        v_id,
                        u_id,
                        EdgeMeta {
                            length_m,
                            highway_type: hw_type,
                            bearing_deg: reverse_bearing,
                        },
                    );
                }
            }
        }
    }

    Ok(city)
}