use camera_network::{
    build_skeleton, deploy_stage_0_baseline, deploy_stage_1_colocated,
    deploy_stage_2_linear_cadence, deploy_stage_3_dead_ends, deploy_stage_4_k_minus_1,
    run_audit, CameraPole, NetworkConfig,
};
use road_graph_loader::RoadGraph;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct MapLayer {
    pub name: String,
    pub color_hex: String,
    pub circle_radius: f32,
    pub opacity: f32,
    pub poles: Vec<CameraPole>,
}

fn export_test_map(
    graph: &RoadGraph,
    layers: &[MapLayer],
    output_path: &Path,
) -> std::io::Result<()> {
    let mut file = File::create(output_path)?;

    let mut road_segments_js = String::new();
    road_segments_js.push_str("const roadSegments = [\n");
    let max_sample = graph.num_edges.min(12_000);
    for edge_idx in (0..graph.num_edges).step_by((graph.num_edges / max_sample).max(1)) {
        let desc = graph.edge_descriptor(edge_idx);
        if desc.geometry.len() >= 2 {
            road_segments_js.push_str("  [");
            for pt in &desc.geometry {
                road_segments_js.push_str(&format!("[{:.5}, {:.5}],", pt[0], pt[1]));
            }
            road_segments_js.push_str("],\n");
        }
    }
    road_segments_js.push_str("];\n");

    let mut layers_js = String::new();
    layers_js.push_str("const mapLayers = [\n");
    for layer in layers {
        layers_js.push_str("  {\n");
        layers_js.push_str(&format!("    name: \"{}\",\n", layer.name));
        layers_js.push_str(&format!("    color: \"{}\",\n", layer.color_hex));
        layers_js.push_str(&format!("    radius: {},\n", layer.circle_radius));
        layers_js.push_str(&format!("    opacity: {},\n", layer.opacity));
        layers_js.push_str("    points: [\n");
        for pole in &layer.poles {
            layers_js.push_str(&format!(
                "      [{:.5}, {:.5}, \"{}\", {}, \"{:?}\"],\n",
                pole.lat, pole.lon, pole.pole_id, pole.camera_heads, pole.mount_type
            ));
        }
        layers_js.push_str("    ]\n");
        layers_js.push_str("  },\n");
    }
    layers_js.push_str("];\n");

    let html_content = format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Mumbai Camera Progression Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; }}
    #map {{ width: 100%; height: 100%; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script>
    const map = L.map('map').setView([19.0760, 72.8777], 11);

    L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    {road_segments_js}
    {layers_js}

    const roadGroup = L.layerGroup();
    roadSegments.forEach(pts => {{
      L.polyline(pts, {{ color: '#888888', weight: 1, opacity: 0.4 }}).addTo(roadGroup);
    }});
    roadGroup.addTo(map);

    const overlayControls = {{ "Road Network": roadGroup }};

    mapLayers.forEach((layerData, idx) => {{
      const group = L.layerGroup();
      layerData.points.forEach(pt => {{
        const [lat, lon, id, heads, mount] = pt;
        L.circleMarker([lat, lon], {{
          radius: layerData.radius,
          color: layerData.color,
          fillColor: layerData.color,
          fillOpacity: layerData.opacity,
          weight: 1
        }}).bindPopup(`<b>${{id}}</b><br>Layer: ${{layerData.name}}<br>Heads: ${{heads}}<br>Mount: ${{mount}}`).addTo(group);
      }});

      if (idx === mapLayers.length - 1) {{
        group.addTo(map);
      }}
      overlayControls[`<span style='color:${{layerData.color}}'>●</span> ${{layerData.name}} (${{layerData.points.length}})`] = group;
    }});

    L.control.layers(null, overlayControls, {{ collapsed: false }}).addTo(map);
  </script>
</body>
</html>
"#
    );

    file.write_all(html_content.as_bytes())?;
    Ok(())
}

#[test]
fn test_camera_network_progression() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let bin_path = root.join("data/road/mumbai_complete_graph.bin");

    if !bin_path.exists() {
        eprintln!("Graph not found at {:?}, skipping benchmark.", bin_path);
        return;
    }

    let graph = RoadGraph::open(&bin_path).expect("Failed to load RoadGraph from .bin");
    let config = NetworkConfig::default();
    let skeleton = build_skeleton(&graph);

    // Stage 0: Complete Unpruned Baseline
    let s0 = deploy_stage_0_baseline(&graph, &config);
    let a0 = run_audit(&skeleton, &s0.monitored_corridor_ids, config.audit_trips);

    // Stage 1: Physical Site Co-Location
    let s1 = deploy_stage_1_colocated(&graph, &config);
    let a1 = run_audit(&skeleton, &s1.monitored_corridor_ids, config.audit_trips);

    // Stage 2: Linear Cadence Normalization
    let s2 = deploy_stage_2_linear_cadence(&graph, &config);
    let a2 = run_audit(&skeleton, &s2.monitored_corridor_ids, config.audit_trips);

    // Stage 3: True Cul-de-Sac Pruning
    let s3 = deploy_stage_3_dead_ends(&graph, &config);
    let a3 = run_audit(&skeleton, &s3.monitored_corridor_ids, config.audit_trips);

    // Stage 4: K-1 Decision Deductibility
    let s4 = deploy_stage_4_k_minus_1(&graph, &config);
    let a4 = run_audit(&skeleton, &s4.monitored_corridor_ids, config.audit_trips);

    println!("\n=============================================================================================================");
    println!("                                CAMERA NETWORK OPTIMIZATION BENCHMARK                                        ");
    println!("=============================================================================================================");
    println!(
        "{:<35} | {:<7} | {:<8} | {:<12} | {:<8} | {:<8} | {:<11} | {:<10} | {:<6}",
        "Optimization Stage", "Poles", "Heads", "Cost (INR)", "Trip Cov", "Multi-Hit", "Transit Gap", "Term Dark", "Forks"
    );
    println!("-------------------------------------------------------------------------------------------------------------");
    println!(
        "{:<35} | {:<7} | {:<8} | ₹{:<11.0} | {:<7.2}% | {:<7.2}% | {:<10.1}m | {:<9.1}m | {:<6}",
        s0.stage_name, s0.total_poles, s0.total_heads, s0.total_cost_inr,
        a0.trip_interception_pct, a0.multi_hit_pct, a0.max_transit_dark_run_m, a0.max_terminal_dark_run_m, a0.max_terminal_branching
    );
    println!(
        "{:<35} | {:<7} | {:<8} | ₹{:<11.0} | {:<7.2}% | {:<7.2}% | {:<10.1}m | {:<9.1}m | {:<6}",
        s1.stage_name, s1.total_poles, s1.total_heads, s1.total_cost_inr,
        a1.trip_interception_pct, a1.multi_hit_pct, a1.max_transit_dark_run_m, a1.max_terminal_dark_run_m, a1.max_terminal_branching
    );
    println!(
        "{:<35} | {:<7} | {:<8} | ₹{:<11.0} | {:<7.2}% | {:<7.2}% | {:<10.1}m | {:<9.1}m | {:<6}",
        s2.stage_name, s2.total_poles, s2.total_heads, s2.total_cost_inr,
        a2.trip_interception_pct, a2.multi_hit_pct, a2.max_transit_dark_run_m, a2.max_terminal_dark_run_m, a2.max_terminal_branching
    );
    println!(
        "{:<35} | {:<7} | {:<8} | ₹{:<11.0} | {:<7.2}% | {:<7.2}% | {:<10.1}m | {:<9.1}m | {:<6}",
        s3.stage_name, s3.total_poles, s3.total_heads, s3.total_cost_inr,
        a3.trip_interception_pct, a3.multi_hit_pct, a3.max_transit_dark_run_m, a3.max_terminal_dark_run_m, a3.max_terminal_branching
    );
    println!(
        "{:<35} | {:<7} | {:<8} | ₹{:<11.0} | {:<7.2}% | {:<7.2}% | {:<10.1}m | {:<9.1}m | {:<6}",
        s4.stage_name, s4.total_poles, s4.total_heads, s4.total_cost_inr,
        a4.trip_interception_pct, a4.multi_hit_pct, a4.max_transit_dark_run_m, a4.max_terminal_dark_run_m, a4.max_terminal_branching
    );
    println!("=============================================================================================================\n");

    let map_path = root.join("data/mumbai_camera_progression_map.html");
    let layers = vec![
        MapLayer {
            name: "Stage 0: Unpruned Baseline".to_string(),
            color_hex: "#111111".to_string(),
            circle_radius: 2.0,
            opacity: 0.15,
            poles: s0.poles.clone(),
        },
        MapLayer {
            name: "Stage 1: Co-Located Sites".to_string(),
            color_hex: "#e63946".to_string(),
            circle_radius: 2.4,
            opacity: 0.35,
            poles: s1.poles.clone(),
        },
        MapLayer {
            name: "Stage 2: Cadence Normalized".to_string(),
            color_hex: "#f4a261".to_string(),
            circle_radius: 2.7,
            opacity: 0.55,
            poles: s2.poles.clone(),
        },
        MapLayer {
            name: "Stage 3: Cul-de-Sac Pruned".to_string(),
            color_hex: "#2a9d8f".to_string(),
            circle_radius: 3.0,
            opacity: 0.75,
            poles: s3.poles.clone(),
        },
        MapLayer {
            name: "Stage 4: K-1 Deductibility".to_string(),
            color_hex: "#9b5de5".to_string(), // Purple
            circle_radius: 3.4,
            opacity: 0.95,
            poles: s4.poles.clone(),
        },
    ];

    export_test_map(&graph, &layers, &map_path).expect("Failed to export progression map");
    println!("Interactive map generated at: {:?}", map_path);
}