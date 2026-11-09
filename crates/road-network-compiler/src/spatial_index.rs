use crate::geometry::{angular_diff, calculate_bearing, point_to_segment_dist_m};
use ahash::{AHashMap, AHashSet};

#[derive(Debug, Clone)]
pub struct SpatialEdgeEntry {
    pub edge_idx: usize,
    pub bearing_deg: f64,
    pub start_lat: f64,
    pub start_lon: f64,
    pub end_lat: f64,
    pub end_lon: f64,
}

pub struct SpatialEdgeGrid {
    cell_size_deg: f64,
    grid: AHashMap<(i32, i32), Vec<SpatialEdgeEntry>>,
}

impl SpatialEdgeGrid {
    pub fn new(edges: Vec<SpatialEdgeEntry>, cell_size_deg: f64) -> Self {
        let mut grid: AHashMap<(i32, i32), Vec<SpatialEdgeEntry>> = AHashMap::new();

        for edge in edges {
            let mid_lat = (edge.start_lat + edge.end_lat) / 2.0;
            let mid_lon = (edge.start_lon + edge.end_lon) / 2.0;
            let cx = (mid_lon / cell_size_deg).floor() as i32;
            let cy = (mid_lat / cell_size_deg).floor() as i32;

            for dx in -1..=1 {
                for dy in -1..=1 {
                    grid.entry((cx + dx, cy + dy)).or_default().push(edge.clone());
                }
            }
        }

        Self { cell_size_deg, grid }
    }

    pub fn find_matching_edges_for_segment(
        &self,
        a_lat: f64,
        a_lon: f64,
        b_lat: f64,
        b_lon: f64,
        max_dist_m: f64,
        max_angle_deg: f64,
        out_matches: &mut AHashSet<usize>,
    ) {
        let seg_bearing = calculate_bearing(a_lat, a_lon, b_lat, b_lon);
        let mid_lat = (a_lat + b_lat) / 2.0;
        let mid_lon = (a_lon + b_lon) / 2.0;
        let cx = (mid_lon / self.cell_size_deg).floor() as i32;
        let cy = (mid_lat / self.cell_size_deg).floor() as i32;

        if let Some(candidates) = self.grid.get(&(cx, cy)) {
            for c in candidates {
                if angular_diff(seg_bearing, c.bearing_deg) > max_angle_deg {
                    continue;
                }
                let d_start = point_to_segment_dist_m(c.start_lat, c.start_lon, a_lat, a_lon, b_lat, b_lon);
                let d_end = point_to_segment_dist_m(c.end_lat, c.end_lon, a_lat, a_lon, b_lat, b_lon);

                if d_start <= max_dist_m && d_end <= max_dist_m {
                    out_matches.insert(c.edge_idx);
                }
            }
        }
    }
}