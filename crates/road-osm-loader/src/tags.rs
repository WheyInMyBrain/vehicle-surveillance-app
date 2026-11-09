use road_network_core::{HighwayType, RoadSurface};

#[inline(always)]
pub fn parse_highway_tag(val: &str) -> Option<HighwayType> {
    match val {
        "motorway" | "motorway_link" => Some(HighwayType::Motorway),
        "trunk" | "trunk_link" => Some(HighwayType::Trunk),
        "primary" | "primary_link" => Some(HighwayType::Primary),
        "secondary" | "secondary_link" => Some(HighwayType::Secondary),
        "tertiary" | "tertiary_link" => Some(HighwayType::Tertiary),
        "residential" => Some(HighwayType::Residential),
        "living_street" => Some(HighwayType::LivingStreet),
        "unclassified" => Some(HighwayType::Unclassified),
        "service" => Some(HighwayType::Service),
        "track" => Some(HighwayType::Track),
        _ => None,
    }
}

#[inline(always)]
pub fn parse_speed(raw: &str) -> Option<f64> {
    let mut num_end = 0;
    let bytes = raw.as_bytes();
    while num_end < bytes.len() && (bytes[num_end].is_ascii_digit() || bytes[num_end] == b'.') {
        num_end += 1;
    }
    if num_end == 0 {
        return None;
    }
    std::str::from_utf8(&bytes[..num_end]).ok()?.parse::<f64>().ok()
}

#[inline(always)]
pub fn parse_surface(val: &str) -> RoadSurface {
    match val {
        "asphalt" | "concrete" => RoadSurface::Asphalt,
        "paved" => RoadSurface::Paved,
        "cobblestone" | "sett" => RoadSurface::Cobblestone,
        "unpaved" | "compacted" | "gravel" | "dirt" | "ground" | "sand" => RoadSurface::Unpaved,
        _ => RoadSurface::Unknown,
    }
}

#[inline(always)]
pub fn is_access_denied(val: &str) -> bool {
    matches!(val, "no" | "private" | "agricultural" | "forestry" | "customers")
}