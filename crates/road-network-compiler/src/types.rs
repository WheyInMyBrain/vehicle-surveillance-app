use ahash::AHashMap;
use road_network_core::NodeId;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

pub const TIME_BINS_PER_DAY: usize = 288;
pub const DAYS_PER_WEEK: usize = 7;
pub const TOTAL_TIME_BINS: usize = DAYS_PER_WEEK * TIME_BINS_PER_DAY; // 2016

pub type CanonicalDirectedEdge = (NodeId, NodeId);

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnifiedHighwayTier {
    Motorway = 1,
    Trunk = 2,
    Primary = 3,
    Secondary = 4,
    Tertiary = 5,
    Residential = 6,
    Service = 7,
    LivingStreet = 8,
    Unclassified = 9,
    Other = 10,
}

impl UnifiedHighwayTier {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "motorway" => UnifiedHighwayTier::Motorway,
            "trunk" => UnifiedHighwayTier::Trunk,
            "primary" => UnifiedHighwayTier::Primary,
            "secondary" => UnifiedHighwayTier::Secondary,
            "tertiary" => UnifiedHighwayTier::Tertiary,
            "residential" => UnifiedHighwayTier::Residential,
            "service" => UnifiedHighwayTier::Service,
            "livingstreet" => UnifiedHighwayTier::LivingStreet,
            "unclassified" => UnifiedHighwayTier::Unclassified,
            _ => UnifiedHighwayTier::Other,
        }
    }

    pub fn baseline_speed_kmh(&self) -> u8 {
        match self {
            UnifiedHighwayTier::Motorway => 80,
            UnifiedHighwayTier::Trunk => 60,
            UnifiedHighwayTier::Primary => 50,
            UnifiedHighwayTier::Secondary => 40,
            UnifiedHighwayTier::Tertiary => 35,
            UnifiedHighwayTier::Residential => 25,
            UnifiedHighwayTier::LivingStreet => 15,
            UnifiedHighwayTier::Service => 15,
            UnifiedHighwayTier::Unclassified => 30,
            UnifiedHighwayTier::Other => 25,
        }
    }
}

#[derive(Debug, Clone)]
pub struct UnifiedNodeRecord {
    pub id: NodeId,
    pub lat: f64,
    pub lon: f64,
}

#[derive(Debug, Clone)]
pub struct UnifiedEdgeRecord {
    pub u: NodeId,
    pub v: NodeId,
    pub length_m: f32,
    pub maxspeed_kmh: u8,
    pub lanes: u8,
    pub highway_tier: UnifiedHighwayTier,
    pub geometry: Vec<[f64; 2]>,
}

#[derive(Debug, Clone)]
pub struct UnifiedMumbaiGraph {
    pub nodes: Vec<UnifiedNodeRecord>,
    pub edges: Vec<UnifiedEdgeRecord>,
    pub edge_to_idx: AHashMap<CanonicalDirectedEdge, usize>,

    // Dynamic telemetry tables: size = edges.len() * 2016
    pub speeds_kmh: Vec<u8>,
    pub jam_factors: Vec<u8>,     // Scaled 0..100 (where 100 = 10.0 jam)
    pub free_flow_kmh: Vec<u8>,
    pub confidences: Vec<u8>,     // 0..100%
    pub is_closed: Vec<u8>,       // 0 = open, 1 = closed
    pub is_empirical: Vec<u8>,    // 1 = observed probe telemetry, 0 = baseline derived
}

impl UnifiedMumbaiGraph {
    pub fn new(nodes: Vec<UnifiedNodeRecord>, edges: Vec<UnifiedEdgeRecord>) -> Self {
        let total_cells = edges.len() * TOTAL_TIME_BINS;
        let mut edge_to_idx = AHashMap::with_capacity(edges.len());

        let mut speeds_kmh = vec![0u8; total_cells];
        let mut free_flow_kmh = vec![0u8; total_cells];

        for (i, e) in edges.iter().enumerate() {
            edge_to_idx.insert((e.u, e.v), i);
            let base_speed = e.highway_tier.baseline_speed_kmh();
            let base_offset = i * TOTAL_TIME_BINS;
            for t in 0..TOTAL_TIME_BINS {
                speeds_kmh[base_offset + t] = base_speed;
                free_flow_kmh[base_offset + t] = base_speed;
            }
        }

        Self {
            nodes,
            edges,
            edge_to_idx,
            speeds_kmh,
            jam_factors: vec![0u8; total_cells],
            free_flow_kmh,
            confidences: vec![0u8; total_cells],
            is_closed: vec![0u8; total_cells],
            is_empirical: vec![0u8; total_cells],
        }
    }

    #[inline(always)]
    pub fn cell_index(&self, edge_idx: usize, day: usize, time_bin_5min: usize) -> usize {
        let bin_idx = (day % DAYS_PER_WEEK) * TIME_BINS_PER_DAY + (time_bin_5min % TIME_BINS_PER_DAY);
        edge_idx * TOTAL_TIME_BINS + bin_idx
    }

    pub fn save_to_disk<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Header: [Magic: b"MMCG", Version: 1, Nodes: u32, Edges: u32, Bins: u32]
        writer.write_all(b"MMCG")?;
        writer.write_all(&1u32.to_le_bytes())?;
        writer.write_all(&(self.nodes.len() as u32).to_le_bytes())?;
        writer.write_all(&(self.edges.len() as u32).to_le_bytes())?;
        writer.write_all(&(TOTAL_TIME_BINS as u32).to_le_bytes())?;

        // Nodes Table
        for n in &self.nodes {
            writer.write_all(&n.id.to_le_bytes())?;
            writer.write_all(&n.lat.to_le_bytes())?;
            writer.write_all(&n.lon.to_le_bytes())?;
        }

        // Edges Table
        for e in &self.edges {
            writer.write_all(&e.u.to_le_bytes())?;
            writer.write_all(&e.v.to_le_bytes())?;
            writer.write_all(&e.length_m.to_le_bytes())?;
            writer.write_all(&[e.maxspeed_kmh, e.lanes, e.highway_tier as u8, 0u8])?;

            writer.write_all(&(e.geometry.len() as u16).to_le_bytes())?;
            for pt in &e.geometry {
                writer.write_all(&pt[0].to_le_bytes())?;
                writer.write_all(&pt[1].to_le_bytes())?;
            }
        }

        // Dense Dynamic Telemetry Tensors
        writer.write_all(&self.speeds_kmh)?;
        writer.write_all(&self.jam_factors)?;
        writer.write_all(&self.free_flow_kmh)?;
        writer.write_all(&self.confidences)?;
        writer.write_all(&self.is_closed)?;
        writer.write_all(&self.is_empirical)?;

        writer.flush()?;
        Ok(())
    }

    pub fn load_from_disk<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"MMCG" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid MMCG header magic",
            ));
        }

        let mut u32_buf = [0u8; 4];
        reader.read_exact(&mut u32_buf)?; // Version
        reader.read_exact(&mut u32_buf)?;
        let num_nodes = u32::from_le_bytes(u32_buf) as usize;
        reader.read_exact(&mut u32_buf)?;
        let num_edges = u32::from_le_bytes(u32_buf) as usize;
        reader.read_exact(&mut u32_buf)?;
        let bins_per_edge = u32::from_le_bytes(u32_buf) as usize;

        let mut nodes = Vec::with_capacity(num_nodes);
        let mut u64_buf = [0u8; 8];
        let mut f64_buf = [0u8; 8];

        for _ in 0..num_nodes {
            reader.read_exact(&mut u64_buf)?;
            let id = u64::from_le_bytes(u64_buf);
            reader.read_exact(&mut f64_buf)?;
            let lat = f64::from_le_bytes(f64_buf);
            reader.read_exact(&mut f64_buf)?;
            let lon = f64::from_le_bytes(f64_buf);
            nodes.push(UnifiedNodeRecord { id, lat, lon });
        }

        let mut edges = Vec::with_capacity(num_edges);
        let mut edge_to_idx = AHashMap::with_capacity(num_edges);
        let mut f32_buf = [0u8; 4];
        let mut u16_buf = [0u8; 2];
        let mut flag_buf = [0u8; 4];

        for i in 0..num_edges {
            reader.read_exact(&mut u64_buf)?;
            let u = u64::from_le_bytes(u64_buf);
            reader.read_exact(&mut u64_buf)?;
            let v = u64::from_le_bytes(u64_buf);
            reader.read_exact(&mut f32_buf)?;
            let length_m = f32::from_le_bytes(f32_buf);
            reader.read_exact(&mut flag_buf)?;

            let maxspeed_kmh = flag_buf[0];
            let lanes = flag_buf[1];
            let tier_byte = flag_buf[2];

            let highway_tier = match tier_byte {
                1 => UnifiedHighwayTier::Motorway,
                2 => UnifiedHighwayTier::Trunk,
                3 => UnifiedHighwayTier::Primary,
                4 => UnifiedHighwayTier::Secondary,
                5 => UnifiedHighwayTier::Tertiary,
                6 => UnifiedHighwayTier::Residential,
                7 => UnifiedHighwayTier::Service,
                8 => UnifiedHighwayTier::LivingStreet,
                9 => UnifiedHighwayTier::Unclassified,
                _ => UnifiedHighwayTier::Other,
            };

            reader.read_exact(&mut u16_buf)?;
            let num_geom = u16::from_le_bytes(u16_buf) as usize;
            let mut geometry = Vec::with_capacity(num_geom);
            for _ in 0..num_geom {
                reader.read_exact(&mut f64_buf)?;
                let p_lat = f64::from_le_bytes(f64_buf);
                reader.read_exact(&mut f64_buf)?;
                let p_lon = f64::from_le_bytes(f64_buf);
                geometry.push([p_lat, p_lon]);
            }

            edges.push(UnifiedEdgeRecord {
                u,
                v,
                length_m,
                maxspeed_kmh,
                lanes,
                highway_tier,
                geometry,
            });
            edge_to_idx.insert((u, v), i);
        }

        let total_cells = num_edges * bins_per_edge;
        let mut speeds_kmh = vec![0u8; total_cells];
        let mut jam_factors = vec![0u8; total_cells];
        let mut free_flow_kmh = vec![0u8; total_cells];
        let mut confidences = vec![0u8; total_cells];
        let mut is_closed = vec![0u8; total_cells];
        let mut is_empirical = vec![0u8; total_cells];

        reader.read_exact(&mut speeds_kmh)?;
        reader.read_exact(&mut jam_factors)?;
        reader.read_exact(&mut free_flow_kmh)?;
        reader.read_exact(&mut confidences)?;
        reader.read_exact(&mut is_closed)?;
        reader.read_exact(&mut is_empirical)?;

        Ok(Self {
            nodes,
            edges,
            edge_to_idx,
            speeds_kmh,
            jam_factors,
            free_flow_kmh,
            confidences,
            is_closed,
            is_empirical,
        })
    }
}