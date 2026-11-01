use engine_core::{HighwayType, RoadSurface};

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

pub fn parse_speed(raw: &str) -> Option<f64> {
    let clean: String = raw
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    clean.parse::<f64>().ok() // No arbitrary clamp(10.0, 120.0)
}

pub fn parse_surface(val: &str) -> RoadSurface {
    match val.to_ascii_lowercase().as_str() {
        "asphalt" | "concrete" => RoadSurface::Asphalt,
        "paved" => RoadSurface::Paved,
        "cobblestone" | "sett" => RoadSurface::Cobblestone,
        "unpaved" | "compacted" | "gravel" | "dirt" | "ground" | "sand" => RoadSurface::Unpaved,
        _ => RoadSurface::Unknown,
    }
}

pub fn is_access_denied(val: &str) -> bool {
    matches!(val, "no" | "private" | "agricultural" | "forestry")
}