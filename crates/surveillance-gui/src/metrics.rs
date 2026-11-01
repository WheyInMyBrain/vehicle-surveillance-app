use engine_core::{geo::haversine_distance, CityGraph, NodeId};

pub fn compute_path_length_m(city: &CityGraph, path: &[NodeId]) -> f64 {
    let mut total = 0.0;
    for w in path.windows(2) {
        if let (Some(&u_idx), Some(&v_idx)) = (city.node_map.get(&w[0]), city.node_map.get(&w[1])) {
            if let Some(edge) = city.graph.find_edge(u_idx, v_idx).or_else(|| city.graph.find_edge(v_idx, u_idx)) {
                total += city.graph[edge].length_m;
            } else {
                let u_m = &city.graph[u_idx];
                let v_m = &city.graph[v_idx];
                total += haversine_distance(u_m.lat, u_m.lon, v_m.lat, v_m.lon);
            }
        }
    }
    total
}

/// Strict Longest Common Subsequence (LCSS) preserving temporal direction
pub fn compute_lcss_accuracy_pct(ground_truth: &[NodeId], reconstructed: &[NodeId]) -> f64 {
    let n = ground_truth.len();
    let m = reconstructed.len();
    if n == 0 || m == 0 {
        return 0.0;
    }
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..n {
        for j in 0..m {
            if ground_truth[i] == reconstructed[j] {
                dp[i + 1][j + 1] = dp[i][j] + 1;
            } else {
                dp[i + 1][j + 1] = dp[i][j + 1].max(dp[i + 1][j]);
            }
        }
    }
    ((dp[n][m] as f64 / n as f64) * 100.0).clamp(0.0, 100.0)
}

/// Maximum physical drift between reconstructed route and ground truth in meters
pub fn compute_max_spatial_drift_m(city: &CityGraph, ground_truth: &[NodeId], reconstructed: &[NodeId]) -> f64 {
    let mut max_drift = 0.0f64;
    for &r_nid in reconstructed {
        if let Some(&r_idx) = city.node_map.get(&r_nid) {
            let r_m = &city.graph[r_idx];
            let mut min_d = f64::INFINITY;
            for &gt_nid in ground_truth {
                if let Some(&gt_idx) = city.node_map.get(&gt_nid) {
                    let gt_m = &city.graph[gt_idx];
                    let d = haversine_distance(r_m.lat, r_m.lon, gt_m.lat, gt_m.lon);
                    if d < min_d {
                        min_d = d;
                    }
                }
            }
            if min_d.is_finite() && min_d > max_drift {
                max_drift = min_d;
            }
        }
    }
    max_drift
}