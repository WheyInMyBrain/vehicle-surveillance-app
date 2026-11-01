use osm_loader::load_osm_pbf;
use std::path::PathBuf;
use std::time::Instant;
use network_optimizer::skeleton::build_skeleton_graph;

#[test]
fn test_mumbai_skeleton_contraction() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let pbf_path = root.join("data/western-zone-261028.osm.pbf");

    let bkc_bbox = Some((19.0400, 72.8400, 19.0900, 72.9000));
    let (city, _) = load_osm_pbf(&pbf_path, bkc_bbox).expect("Failed to load PBF");

    let t0 = Instant::now();
    let skeleton = build_skeleton_graph(&city);
    let elapsed = t0.elapsed();

    let raw_node_count = city.graph.node_count();
    let raw_edge_count = city.graph.edge_count();
    let junction_count = skeleton.junction_nodes.len();
    let corridor_count = skeleton.corridors.len();

    let node_reduction = (1.0 - (junction_count as f64 / raw_node_count as f64)) * 100.0;
    let edge_reduction = (1.0 - (corridor_count as f64 / raw_edge_count as f64)) * 100.0;

    println!("\n==============================================");
    println!("     SKELETON DEGREE-2 CONTRACTION REPORT      ");
    println!("==============================================");
    println!("Contraction Duration:   {:.2?}", elapsed);
    println!("Raw OSM Graph Nodes:    {}", raw_node_count);
    println!("Topological Junctions:  {} ({:.1}% reduction)", junction_count, node_reduction);
    println!("Raw Directed Edges:     {}", raw_edge_count);
    println!("Macro-Corridors:        {} ({:.1}% search space reduction)", corridor_count, edge_reduction);

    let max_chain = skeleton.corridors.iter().map(|c| c.raw_edges.len()).max().unwrap_or(0);
    let max_length = skeleton.corridors.iter().map(|c| c.total_length_m).fold(0.0f64, f64::max);
    let avg_length = skeleton.corridors.iter().map(|c| c.total_length_m).sum::<f64>() / corridor_count as f64;

    println!("Max Edges in a Chain:   {}", max_chain);
    println!("Longest Macro-Corridor: {:.1} m", max_length);
    println!("Avg Corridor Length:    {:.1} m", avg_length);
    println!("==============================================\n");

    assert!(junction_count < raw_node_count, "Contraction did not reduce node count");
    assert!(corridor_count < raw_edge_count, "Contraction did not reduce edge count");
}