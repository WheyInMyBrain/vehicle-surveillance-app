use engine_core::{EdgeMeta, HighwayType, NodeMeta, RoadSurface};

/// Computes the physical fastest theoretical time to traverse an edge.
/// Any trip taking less than this would require breaking physical speed limits.
#[inline(always)]
pub fn min_physical_traversal_time(edge: &EdgeMeta, prev_bearing: Option<f64>) -> f64 {
    // Upper legal limit from OSM or generic physical road class maximum
    let max_legal_kmh = edge.maxspeed_kmh.unwrap_or(match edge.highway_type {
        HighwayType::Motorway => 80.0,
        HighwayType::Trunk => 70.0,
        HighwayType::Primary => 50.0,
        HighwayType::Secondary => 40.0,
        HighwayType::Tertiary => 35.0,
        HighwayType::Residential => 25.0,
        HighwayType::LivingStreet => 15.0,
        HighwayType::Unclassified => 25.0,
        HighwayType::Service => 15.0,
        HighwayType::Track => 15.0,
    });

    // Surface physical speed cap (e.g., cannot physically do 80 km/h on unpaved dirt)
    let surface_cap = match edge.surface {
        RoadSurface::Asphalt | RoadSurface::Paved => 120.0,
        RoadSurface::Cobblestone => 40.0,
        RoadSurface::Unpaved => 30.0,
        RoadSurface::Unknown => 60.0,
    };

    // Angular turn geometry restriction
    let turn_cap = if let Some(pb) = prev_bearing {
        let diff = (edge.bearing_deg - pb + 180.0) % 360.0 - 180.0;
        let angle = (if diff < -180.0 { diff + 360.0 } else { diff }).abs();
        if angle > 90.0 {
            15.0 // Sharp turn
        } else if angle > 45.0 {
            25.0
        } else if angle > 20.0 {
            40.0
        } else {
            120.0
        }
    } else {
        120.0
    };

    let max_possible_speed_kmh = max_legal_kmh.min(surface_cap).min(turn_cap).max(10.0);
    let speed_mps = max_possible_speed_kmh / 3.6;

    edge.length_m / speed_mps
}

/// Evaluates how much delay a node can naturally explain/absorb
/// based on structural features (signals, speed breakers, junction complexity)
/// WITHOUT assuming fixed durations.
#[inline(always)]
pub fn structural_delay_capacity(node_meta: &NodeMeta, out_degree: usize) -> f64 {
    let mut capacity_weight = 1.0;

    if node_meta.has_traffic_signal {
        // A traffic signal has massive variance capacity (green = 0s, multiple reds = 180s+)
        capacity_weight += 5.0;
    }
    if node_meta.has_speed_bump {
        capacity_weight += 1.5;
    }
    if out_degree >= 3 {
        // Multi-way intersection creates natural yielding and pedestrian friction
        capacity_weight += 2.0;
    }

    capacity_weight
}