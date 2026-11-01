use network_optimizer::cut::find_topological_cuts;
use network_optimizer::skeleton::build_skeleton_graph;
use osm_loader::load_osm_pbf;
use std::path::PathBuf;
use std::time::Instant;

#[test]
fn test_mumbai_topological_cuts() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let pbf_path = root.join("data/western-zone-261028.osm.pbf");

    let bkc_bbox = Some((19.0400, 72.8400, 19.0900, 72.9000));
    let (city, _) = load_osm_pbf(&pbf_path, bkc_bbox).expect("Failed to load PBF");

    let skeleton = build_skeleton_graph(&city);

    let t0 = Instant::now();
    let cuts = find_topological_cuts(&city, &skeleton);
    let elapsed = t0.elapsed();

    println!("\n==============================================");
    println!("     TOPOLOGICAL BRIDGES & GATES REPORT       ");
    println!("==============================================");
    println!("Bridge Analysis Time:     {:.2?}", elapsed);
    println!("Total Macro-Corridors:    {}", skeleton.corridors.len());
    println!("Strict Topological Bridges: {}", cuts.bridge_corridor_ids.len());
    println!("Perimeter Gateways:       {}", cuts.perimeter_corridor_ids.len());
    println!("Total Mandatory Locked:   {} ({:.1}% of corridors)", 
        cuts.mandatory_locked_corridors.len(),
        (cuts.mandatory_locked_corridors.len() as f64 / skeleton.corridors.len() as f64) * 100.0
    );

    // Inspect sample bridges
    println!("\n--- SAMPLE CRITICAL BRIDGES (PERMANENTLY SEALED) ---");
    for &cid in cuts.bridge_corridor_ids.iter().take(5) {
        let corr = &skeleton.corridors[cid];
        println!("  • Bridge Corridor #{}: Length: {:.1}m | Between Junctions ({} -> {})",
            cid, corr.total_length_m, corr.start_node, corr.end_node
        );
    }
    println!("==============================================\n");

    assert!(!cuts.bridge_corridor_ids.is_empty(), "No bridges found in the graph");
    assert!(!cuts.mandatory_locked_corridors.is_empty(), "No mandatory locks found");
}