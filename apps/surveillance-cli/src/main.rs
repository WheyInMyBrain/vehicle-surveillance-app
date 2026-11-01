use osm_loader::load_osm_pbf;
use std::path::PathBuf;
use std::time::Instant;
use surveillance_gui::launch_interactive_workbench;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==================================================");
    println!("   MUMBAI SURVEILLANCE & TRAJECTORY ENGINE        ");
    println!("==================================================");

    let pbf_path = PathBuf::from("data/western-zone-261028.osm.pbf");

    println!("Target PBF Path: {:?}", pbf_path);

    let bkc_bbox = Some((19.0400, 72.8400, 19.0900, 72.9000));

    println!("\n[1/2] Loading OSM Mumbai PBF Network...");
    let start_load = Instant::now();
    let (city, stats) = load_osm_pbf(&pbf_path, bkc_bbox)?;
    let city_ref = Box::leak(Box::new(city));

    println!(
        "✔ City Graph initialized in {:.2?}! ({} intersections, {} road segments)",
        start_load.elapsed(),
        city_ref.graph.node_count(),
        city_ref.graph.edge_count()
    );
    println!(
        "   • Signals: {} | Speed Bumps: {} | Drivable Ways: {}",
        stats.traffic_signals_found, stats.speed_bumps_found, stats.drivable_ways_extracted
    );

    println!("\n[2/2] Launching Interactive Desktop Workbench...");
    launch_interactive_workbench(city_ref)?;

    Ok(())
}