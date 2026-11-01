use ahash::AHashMap;
use engine_core::{geo::haversine_distance, CityGraph, NodeId};
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct GtfsCorridorSpeeds {
    pub edge_speeds_kmh: AHashMap<(NodeId, NodeId), f64>,
}

impl GtfsCorridorSpeeds {
    pub fn empty() -> Self {
        Self {
            edge_speeds_kmh: AHashMap::new(),
        }
    }

    pub fn get_speed(&self, u: NodeId, v: NodeId) -> Option<f64> {
        self.edge_speeds_kmh.get(&(u, v)).copied()
    }

    pub fn load_from_dir<P: AsRef<Path>>(
        gtfs_dir: P,
        city: &CityGraph,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = gtfs_dir.as_ref();
        let stops_path = dir.join("stops.txt");
        let stop_times_path = dir.join("stop_times.txt");

        if !stops_path.exists() || !stop_times_path.exists() {
            return Ok(Self::empty());
        }

        // 1. Build a spatial grid for fast nearest-neighbor lookups (~200m per cell)
        const CELL_SIZE: f64 = 0.002;
        let mut spatial_grid: AHashMap<(i32, i32), Vec<(NodeId, f64, f64)>> = AHashMap::new();

        for &id in city.node_map.keys() {
            let idx = city.node_map[&id];
            let meta = &city.graph[idx];
            let cell = ((meta.lat / CELL_SIZE).floor() as i32, (meta.lon / CELL_SIZE).floor() as i32);
            spatial_grid.entry(cell).or_default().push((id, meta.lat, meta.lon));
        }

        // 2. Ingest stops.txt using 3x3 local neighborhood queries
        let mut stop_to_node: AHashMap<String, NodeId> = AHashMap::new();
        let mut rdr = csv::Reader::from_reader(File::open(stops_path)?);

        for result in rdr.records() {
            let record = match result {
                Ok(r) => r,
                Err(_) => continue,
            };

            let stop_id = record.get(0).unwrap_or("").trim().to_string();
            let lat: f64 = record.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            let lon: f64 = record.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);

            if lat == 0.0 || lon == 0.0 {
                continue;
            }

            let cell_x = (lat / CELL_SIZE).floor() as i32;
            let cell_y = (lon / CELL_SIZE).floor() as i32;

            let mut nearest = None;
            let mut min_d = 100.0;

            for dx in -1..=1 {
                for dy in -1..=1 {
                    if let Some(nodes) = spatial_grid.get(&(cell_x + dx, cell_y + dy)) {
                        for &(nid, nlat, nlon) in nodes {
                            let d = haversine_distance(lat, lon, nlat, nlon);
                            if d < min_d {
                                min_d = d;
                                nearest = Some(nid);
                            }
                        }
                    }
                }
            }

            if let Some(nid) = nearest {
                stop_to_node.insert(stop_id, nid);
            }
        }

        // 3. Process stop_times.txt sequentially
        let mut edge_speed_samples: AHashMap<(NodeId, NodeId), Vec<f64>> = AHashMap::new();
        let mut times_rdr = csv::Reader::from_reader(File::open(stop_times_path)?);

        let mut prev_trip_id = String::new();
        let mut prev_stop_node: Option<NodeId> = None;
        let mut prev_dep_secs: Option<i64> = None;

        for result in times_rdr.records() {
            let record = match result {
                Ok(r) => r,
                Err(_) => continue,
            };

            let trip_id = record.get(0).unwrap_or("").trim().to_string();
            let arr_str = record.get(1).unwrap_or("").trim();
            let dep_str = record.get(2).unwrap_or("").trim();
            let stop_id = record.get(3).unwrap_or("").trim();

            let curr_node = stop_to_node.get(stop_id).copied();
            let curr_arr_secs = parse_gtfs_time_to_seconds(arr_str);
            let curr_dep_secs = parse_gtfs_time_to_seconds(dep_str);

            if trip_id == prev_trip_id {
                if let (Some(u), Some(v), Some(t_dep), Some(t_arr)) =
                    (prev_stop_node, curr_node, prev_dep_secs, curr_arr_secs)
                {
                    if u != v && t_arr > t_dep {
                        let delta_secs = (t_arr - t_dep) as f64;
                        if delta_secs >= 10.0 {
                            if let (Some(&ui), Some(&vi)) = (city.node_map.get(&u), city.node_map.get(&v)) {
                                let dist_m = haversine_distance(
                                    city.graph[ui].lat,
                                    city.graph[ui].lon,
                                    city.graph[vi].lat,
                                    city.graph[vi].lon,
                                );
                                let speed_kmh = (dist_m / delta_secs) * 3.6;

                                if (5.0..=80.0).contains(&speed_kmh) {
                                    edge_speed_samples
                                        .entry((u, v))
                                        .or_default()
                                        .push(speed_kmh);
                                }
                            }
                        }
                    }
                }
            }

            prev_trip_id = trip_id;
            prev_stop_node = curr_node;
            prev_dep_secs = curr_dep_secs;
        }

        // 4. Compute median speed per corridor
        let mut final_speeds = AHashMap::new();
        for (edge, mut speeds) in edge_speed_samples {
            speeds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = speeds[speeds.len() / 2];
            final_speeds.insert(edge, median);
        }

        Ok(Self {
            edge_speeds_kmh: final_speeds,
        })
    }
}

#[inline(always)]
fn parse_gtfs_time_to_seconds(time_str: &str) -> Option<i64> {
    let mut parts = time_str.split(':');
    let h: i64 = parts.next()?.parse().ok()?;
    let m: i64 = parts.next()?.parse().ok()?;
    let s: i64 = parts.next()?.parse().ok()?;
    Some(h * 3600 + m * 60 + s)
}