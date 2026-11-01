use crate::error::OsmError;
use crate::tags::{is_access_denied, parse_highway_tag, parse_speed, parse_surface};
use ahash::{AHashMap, AHashSet};
use engine_core::{HighwayType, NodeId, RoadSurface};
use osmpbf::{Element, ElementReader};
use std::path::Path;

pub struct RawWay {
    pub name: Option<String>,
    pub node_refs: Vec<NodeId>,
    pub highway: HighwayType,
    pub maxspeed_kmh: Option<f64>,
    pub lanes_forward: Option<u8>,
    pub lanes_backward: Option<u8>,
    pub surface: RoadSurface,
    pub oneway: bool,
    pub reversed_oneway: bool,
    pub is_bridge: bool,
    pub is_tunnel: bool,
    pub layer: i8,
}

pub struct RawNodeData {
    pub lat: f64,
    pub lon: f64,
    pub has_signal: bool,
    pub has_bump: bool,
}

pub struct ScanResult {
    pub ways_scanned: usize,
    pub raw_ways: Vec<RawWay>,
    pub node_metadata: AHashMap<NodeId, RawNodeData>,
    pub signals_count: usize,
    pub bumps_count: usize,
}

pub fn scan_pbf<P: AsRef<Path>>(
    path: P,
    bbox: Option<(f64, f64, f64, f64)>,
) -> Result<ScanResult, OsmError> {
    let mut ways_scanned = 0usize;
    let mut needed_node_ids = AHashSet::new();
    let mut raw_ways = Vec::new();

    // Pass 1: Scan Ways
    let reader = ElementReader::from_path(path.as_ref())?;
    reader.for_each(|element| {
        if let Element::Way(way) = element {
            ways_scanned += 1;

            let mut highway_type: Option<HighwayType> = None;
            let mut name: Option<String> = None;
            let mut maxspeed_val = None;
            let mut lanes_total = None;
            let mut lanes_forward_val = None;
            let mut lanes_backward_val = None;
            let mut surface_val = None;
            let mut oneway_tag = None;
            let mut is_roundabout = false;
            let mut is_bridge = false;
            let mut is_tunnel = false;
            let mut layer: i8 = 0;
            let mut access_denied = false;

            for (key, val) in way.tags() {
                match key {
                    "highway" => highway_type = parse_highway_tag(val),
                    "name" => {
                        let trimmed = val.trim();
                        if !trimmed.is_empty() {
                            name = Some(trimmed.to_string());
                        }
                    }
                    "maxspeed" => maxspeed_val = Some(val),
                    "lanes" => lanes_total = val.parse::<u8>().ok(),
                    "lanes:forward" => lanes_forward_val = val.parse::<u8>().ok(),
                    "lanes:backward" => lanes_backward_val = val.parse::<u8>().ok(),
                    "surface" => surface_val = Some(val),
                    "oneway" => oneway_tag = Some(val),
                    "junction" if val == "roundabout" || val == "circular" => is_roundabout = true,
                    "bridge" if val == "yes" || val == "viaduct" => is_bridge = true,
                    "tunnel" if val == "yes" || val == "culvert" => is_tunnel = true,
                    "layer" => {
                        if let Ok(l) = val.parse::<i8>() {
                            layer = l;
                        }
                    }
                    "access" | "motor_vehicle" | "vehicle" => {
                        if is_access_denied(val) {
                            access_denied = true;
                        }
                    }
                    _ => {}
                }
            }

            if access_denied {
                return;
            }

            if let Some(hway) = highway_type {
                let maxspeed = maxspeed_val.and_then(parse_speed);
                let surface = surface_val.map(parse_surface).unwrap_or(RoadSurface::Unknown);

                let lanes_fwd = lanes_forward_val.or(lanes_total);
                let lanes_bwd = lanes_backward_val.or(lanes_total);

                let is_motorway = matches!(hway, HighwayType::Motorway);
                let oneway = match oneway_tag {
                    Some("yes") | Some("1") | Some("true") => true,
                    Some("-1") => false,
                    Some("no") | Some("0") | Some("false") => false,
                    _ => is_motorway || is_roundabout,
                };
                let reversed_oneway = oneway_tag == Some("-1");

                let refs: Vec<NodeId> = way.refs().map(|id| id as NodeId).collect();
                if refs.len() >= 2 {
                    for &id in &refs {
                        needed_node_ids.insert(id);
                    }

                    raw_ways.push(RawWay {
                        name,
                        node_refs: refs,
                        highway: hway,
                        maxspeed_kmh: maxspeed,
                        lanes_forward: lanes_fwd,
                        lanes_backward: lanes_bwd,
                        surface,
                        oneway,
                        reversed_oneway,
                        is_bridge,
                        is_tunnel,
                        layer,
                    });
                }
            }
        }
    })?;

    // Pass 2: Scan Nodes
    let reader = ElementReader::from_path(path.as_ref())?;
    let mut node_metadata: AHashMap<NodeId, RawNodeData> = AHashMap::with_capacity(needed_node_ids.len());
    let mut signals_count = 0usize;
    let mut bumps_count = 0usize;

    reader.for_each(|element| match element {
        Element::Node(node) => {
            let id = node.id() as NodeId;
            if needed_node_ids.contains(&id) {
                let lat = node.lat();
                let lon = node.lon();

                let inside = match bbox {
                    Some((min_lat, min_lon, max_lat, max_lon)) => {
                        lat >= min_lat && lat <= max_lat && lon >= min_lon && lon <= max_lon
                    }
                    None => true,
                };

                if inside {
                    let mut has_signal = false;
                    let mut has_bump = false;
                    for (k, v) in node.tags() {
                        if k == "highway" && v == "traffic_signals" {
                            has_signal = true;
                            signals_count += 1;
                        }
                        if k == "traffic_calming" && (v == "bump" || v == "table" || v == "hump") {
                            has_bump = true;
                            bumps_count += 1;
                        }
                    }
                    node_metadata.insert(id, RawNodeData { lat, lon, has_signal, has_bump });
                }
            }
        }
        Element::DenseNode(node) => {
            let id = node.id as NodeId;
            if needed_node_ids.contains(&id) {
                let lat = node.lat();
                let lon = node.lon();

                let inside = match bbox {
                    Some((min_lat, min_lon, max_lat, max_lon)) => {
                        lat >= min_lat && lat <= max_lat && lon >= min_lon && lon <= max_lon
                    }
                    None => true,
                };

                if inside {
                    let mut has_signal = false;
                    let mut has_bump = false;
                    for (k, v) in node.tags() {
                        if k == "highway" && v == "traffic_signals" {
                            has_signal = true;
                            signals_count += 1;
                        }
                        if k == "traffic_calming" && (v == "bump" || v == "table" || v == "hump") {
                            has_bump = true;
                            bumps_count += 1;
                        }
                    }
                    node_metadata.insert(id, RawNodeData { lat, lon, has_signal, has_bump });
                }
            }
        }
        _ => {}
    })?;

    Ok(ScanResult {
        ways_scanned,
        raw_ways,
        node_metadata,
        signals_count,
        bumps_count,
    })
}