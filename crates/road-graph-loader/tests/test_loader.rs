use road_graph_loader::{RoadGraph, DAYS_PER_WEEK, TIME_BINS_PER_DAY};
use std::path::PathBuf;
use std::time::Instant;

#[test]
fn test_mmap_load_benchmark() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let bin_path = root.join("data/road/mumbai_complete_graph.bin");

    if !bin_path.exists() {
        eprintln!("File not found: {:?}. Run compiler first.", bin_path);
        return;
    }

    println!("\n========================================================");
    println!("     MMAP ZERO-COPY GRAPH LOADER TEST        ");
    println!("========================================================");

    let t0 = Instant::now();
    let graph = RoadGraph::open(&bin_path).expect("Failed to mmap graph");
    let load_time = t0.elapsed();

    println!("Graph mmap opened in:         {:?}", load_time);
    println!("Nodes in memory:              {}", graph.num_nodes);
    println!("Edges indexed:                {}", graph.num_edges);
    println!("Total time intervals/edge:    {}", graph.total_time_bins);

    assert!(load_time.as_millis() < 50, "Mmap loader took longer than 50ms!");

    // Benchmark 1,000,000 random flow queries
    let t_query = Instant::now();
    let iterations = 1_000_000;
    let mut sum_speed = 0u64;

    for i in 0..iterations {
        let edge_idx = i % graph.num_edges;
        let day = (i / 100) % DAYS_PER_WEEK;
        let bin = i % TIME_BINS_PER_DAY;

        let flow = graph.get_flow_by_index(edge_idx, day, bin);
        sum_speed += flow.speed_kmh as u64;
    }

    let query_duration = t_query.elapsed();
    let ns_per_lookup = query_duration.as_nanos() as f64 / iterations as f64;

    println!("Executed {} queries in: {:?}", iterations, query_duration);
    println!("Average lookup latency:       {:.2} ns / query", ns_per_lookup);
    println!("Query speed checksum:         {}", sum_speed);
    println!("========================================================\n");

    assert!(ns_per_lookup < 25.0, "Latency must be sub-25ns per lookup");
}