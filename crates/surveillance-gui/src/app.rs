use crate::batch_dialog::render_batch_report_window;
use crate::batch_eval::{run_monte_carlo_batch_eval, BatchMetricsReport};
use crate::canvas::{render_ground_truth_canvas, render_reconstructed_canvas, ViewportBounds};
use crate::metrics::{compute_lcss_accuracy_pct, compute_max_spatial_drift_m, compute_path_length_m};
use eframe::egui::{self, Color32, Ui};
use engine_core::CityGraph;
use network_optimizer::cut::find_topological_cuts;
use network_optimizer::pruner::{prune_to_zero_escape_network, ZeroEscapePrunerConfig};
use network_optimizer::skeleton::build_skeleton_graph;
use network_optimizer::types::OptimizedCameraPole;
use std::path::PathBuf;
use std::time::Instant;
use trajectory_resolver::{StreamingTracker, VehicleTrackState};
use trip_simulator::{generate_chaotic_trip, CameraNetwork, GtfsCorridorSpeeds, SimulatedTrip, TripBudget};

#[derive(Clone)]
pub struct SimulationOutput {
    pub trip: SimulatedTrip,
    pub track: VehicleTrackState,
    pub full_trip_len_m: f64,
    pub monitored_gt_len_m: f64,
    pub reconstructed_len_m: f64,
    pub full_trip_similarity_pct: f64,
    pub monitored_similarity_pct: f64,
    pub max_spatial_drift_m: f64,
    pub execution_time_us: u128,
}

pub struct InteractiveWorkbench {
    pub city: &'static CityGraph,
    pub gtfs_speeds: Option<GtfsCorridorSpeeds>,

    // Optimizer Parameters (Now fully configurable in GUI)
    pub max_dark_distance_m: f64,
    pub max_dark_hops: usize,
    pub max_unmonitored_forks: usize,

    // Trip & Simulation Controls
    pub trip_duration_mins: f64,
    pub camera_dropout_pct: f32,
    pub max_candidate_hypotheses: usize,

    // State
    pub camera_network: CameraNetwork,
    pub deployed_poles: Vec<OptimizedCameraPole>,
    pub latest_simulation: Option<SimulationOutput>,
    pub status_message: String,
    pub bounds: ViewportBounds,

    // Batch Monte Carlo State
    pub batch_runs: usize,
    pub batch_report: Option<BatchMetricsReport>,
    pub show_batch_window: bool,
}

impl InteractiveWorkbench {
    pub fn new(city: &'static CityGraph) -> Self {
        // Resolve GTFS data directory if present
        let gtfs_path = PathBuf::from("data/gtfs");
        let gtfs_speeds = GtfsCorridorSpeeds::load_from_dir(&gtfs_path, city).ok();

        let mut app = Self {
            city,
            gtfs_speeds,

            // Zero-Escape Defaults
            max_dark_distance_m: 500.0,
            max_dark_hops: 5,
            max_unmonitored_forks: 2,

            // Kinematic Trip Defaults
            trip_duration_mins: 20.0, // 20-minute realistic commute
            camera_dropout_pct: 0.05, // 5% real-world sensor dropout
            max_candidate_hypotheses: 4,

            camera_network: CameraNetwork::new(),
            deployed_poles: Vec::new(),
            latest_simulation: None,
            status_message: "Initializing topology and cameras...".to_string(),

            bounds: ViewportBounds {
                min_lon: 72.84,
                max_lon: 72.90,
                min_lat: 19.04,
                max_lat: 19.09,
            },

            batch_runs: 5_000,
            batch_report: None,
            show_batch_window: false,
        };

        app.recalculate_cameras();
        app.run_vehicle_simulation();
        app
    }

    pub fn recalculate_cameras(&mut self) {
        let start = Instant::now();

        // 1. Skeleton Degree-2 Contraction
        let skeleton = build_skeleton_graph(self.city);

        // 2. Tarjan Cut-Bridges & Perimeter Portals
        let cuts = find_topological_cuts(self.city, &skeleton);

        // 3. Reverse Pruning with Continuous Dark-Path BFS
        let config = ZeroEscapePrunerConfig {
            max_dark_distance_m: self.max_dark_distance_m,
            max_dark_hops: self.max_dark_hops,
            max_unmonitored_forks: self.max_unmonitored_forks,
        };

        let (poles, summary) = prune_to_zero_escape_network(self.city, &skeleton, &cuts, config);

        // 4. Register cameras into physical sensor network
        let mut net = CameraNetwork::new();
        for cam in &poles {
            net.deploy_pole(
                cam.camera_id.clone(),
                cam.edge_from,
                cam.edge_to,
                cam.lat,
                cam.lon,
                cam.is_two_way,
            );
        }

        self.deployed_poles = poles;
        self.camera_network = net;
        self.status_message = format!(
            "Zero-Escape: {} poles (pruned {} redundant) in {:.1}ms.",
            summary.final_active_cameras,
            summary.pruned_redundant_cameras,
            start.elapsed().as_secs_f64() * 1000.0
        );
    }

    pub fn run_vehicle_simulation(&mut self) {
        if self.deployed_poles.is_empty() {
            self.recalculate_cameras();
        }

        let budget_secs = self.trip_duration_mins * 60.0;
        let mut trip_opt = None;

        for _ in 0..20 {
            if let Some(t) = generate_chaotic_trip(
                self.city,
                &self.camera_network,
                self.gtfs_speeds.as_ref(),
                TripBudget::DurationSecs(budget_secs),
                self.camera_dropout_pct as f64,
            ) {
                if t.camera_sightings.len() >= 2 {
                    trip_opt = Some(t);
                    break;
                }
            }
        }

        let trip = match trip_opt {
            Some(t) => t,
            None => {
                self.status_message = "Trip generation failed to meet constraints.".to_string();
                return;
            }
        };

        // Stream into the unbiased trajectory resolver
        let mut tracker =
            StreamingTracker::new(self.city).with_candidate_limit(self.max_candidate_hypotheses);
        let t0 = Instant::now();
        for sighting in trip.camera_sightings.clone() {
            tracker.ingest_sighting(sighting);
        }
        let solve_us = t0.elapsed().as_micros();
        let track = tracker
            .get_track("MH02CB1234")
            .cloned()
            .expect("Tracking resolution failed");

        self.bounds = ViewportBounds::from_trip_and_track(self.city, &trip, &track);

        let first_cam = trip.camera_sightings.first().unwrap().edge_from;
        let last_cam = trip.camera_sightings.last().unwrap().edge_to;
        let s_idx = trip.ground_truth_nodes.iter().position(|&n| n == first_cam).unwrap_or(0);
        let e_idx = trip.ground_truth_nodes.iter().rposition(|&n| n == last_cam).unwrap_or(trip.ground_truth_nodes.len() - 1);

        let mon_gt = if s_idx <= e_idx {
            &trip.ground_truth_nodes[s_idx..=e_idx]
        } else {
            &trip.ground_truth_nodes[..]
        };

        let full_sim = compute_lcss_accuracy_pct(&trip.ground_truth_nodes, &track.reconstructed_path);
        let mon_sim = compute_lcss_accuracy_pct(mon_gt, &track.reconstructed_path);
        let full_len = compute_path_length_m(self.city, &trip.ground_truth_nodes);
        let mon_len = compute_path_length_m(self.city, mon_gt);
        let recon_len = compute_path_length_m(self.city, &track.reconstructed_path);
        let drift = compute_max_spatial_drift_m(self.city, &trip.ground_truth_nodes, &track.reconstructed_path);

        self.latest_simulation = Some(SimulationOutput {
            trip,
            track,
            full_trip_len_m: full_len,
            monitored_gt_len_m: mon_len,
            reconstructed_len_m: recon_len,
            full_trip_similarity_pct: full_sim,
            monitored_similarity_pct: mon_sim,
            max_spatial_drift_m: drift,
            execution_time_us: solve_us,
        });

        self.status_message = format!("Resolved in {} µs (Precision: {:.1}%)", solve_us, mon_sim);
    }
}

impl eframe::App for InteractiveWorkbench {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let total_avail = ui.available_size();
        let bottom_height = 55.0;
        let main_height = total_avail.y - bottom_height - 12.0;
        let right_panel_width = 340.0;
        let left_canvas_width = (total_avail.x - right_panel_width - 24.0) / 2.0;

        // 1. DUAL CANVASES + CONTROLS
        ui.horizontal(|ui| {
            render_ground_truth_canvas(
                ui,
                self.city,
                &self.deployed_poles,
                self.latest_simulation.as_ref().map(|s| &s.trip),
                &self.bounds,
                left_canvas_width,
                main_height - 30.0,
            );

            ui.separator();

            render_reconstructed_canvas(
                ui,
                self.city,
                self.latest_simulation.as_ref().map(|s| &s.track),
                &self.bounds,
                left_canvas_width,
                main_height - 30.0,
            );

            ui.separator();

            // CONTROL SIDEBAR
            ui.vertical(|ui| {
                ui.set_width(right_panel_width);
                ui.heading("Workbench Controls");
                ui.label(egui::RichText::new("Unbiased Zero-Escape Surveillance Engine").weak());
                ui.separator();

                ui.collapsing("1. Zero-Escape Network Optimizer", |ui| {
                    ui.label(egui::RichText::new("Topological Constraints:").strong());
                    ui.add(
                        egui::Slider::new(&mut self.max_dark_distance_m, 200.0..=1200.0)
                            .text("Max Dark Dist (m)")
                            .step_by(50.0),
                    );
                    ui.add(egui::Slider::new(&mut self.max_dark_hops, 2..=10).text("Max Dark Hops"));
                    ui.add(egui::Slider::new(&mut self.max_unmonitored_forks, 1..=4).text("Fork Ambiguity Limit"));

                    ui.add_space(4.0);
                    if ui.button("⟳ Re-optimize Network Topology").clicked() {
                        self.recalculate_cameras();
                    }
                    ui.label(format!("Active Poles: {}", self.deployed_poles.len()));
                });

                ui.add_space(6.0);
                ui.collapsing("2. Realistic Trip Kinematics", |ui| {
                    ui.add(
                        egui::Slider::new(&mut self.trip_duration_mins, 5.0..=60.0)
                            .text("Commute Duration (min)")
                            .step_by(5.0),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.camera_dropout_pct, 0.0..=0.50)
                            .text("Hardware Dropout")
                            .custom_formatter(|n, _| format!("{:.0}%", n * 100.0)),
                    );
                    ui.add(egui::Slider::new(&mut self.max_candidate_hypotheses, 1..=8).text("Max Hypotheses"));
                });

                ui.add_space(8.0);
                ui.collapsing("3. High-Volume Monte Carlo (Batch)", |ui| {
                    ui.add(
                        egui::Slider::new(&mut self.batch_runs, 500..=50_000)
                            .text("Runs")
                            .logarithmic(true),
                    );
                    if ui
                        .add_sized(
                            [right_panel_width - 10.0, 32.0],
                            egui::Button::new(
                                egui::RichText::new("🔬 Run Monte Carlo Evaluation")
                                    .color(Color32::WHITE)
                                    .strong(),
                            )
                            .fill(Color32::from_rgb(130, 40, 190)),
                        )
                        .clicked()
                    {
                        let rep = run_monte_carlo_batch_eval(
                            self.city,
                            &self.camera_network,
                            self.gtfs_speeds.as_ref(),
                            self.batch_runs,
                            self.trip_duration_mins * 60.0,
                            self.camera_dropout_pct as f64,
                            self.max_candidate_hypotheses,
                        );
                        self.batch_report = Some(rep);
                        self.show_batch_window = true;
                        self.status_message = format!("Completed {} runs successfully.", self.batch_runs);
                    }
                });

                ui.add_space(10.0);
                if ui
                    .add_sized(
                        [right_panel_width - 10.0, 36.0],
                        egui::Button::new(
                            egui::RichText::new("▶ Run Vehicle Simulation")
                                .color(Color32::WHITE)
                                .strong(),
                        )
                        .fill(Color32::from_rgb(0, 120, 215)),
                    )
                    .clicked()
                {
                    self.run_vehicle_simulation();
                }

                ui.add_space(4.0);
                if ui
                    .add_sized(
                        [right_panel_width - 10.0, 28.0],
                        egui::Button::new("⚡ Full Pipeline (Cameras + Trip)"),
                    )
                    .clicked()
                {
                    self.recalculate_cameras();
                    self.run_vehicle_simulation();
                }

                ui.add_space(8.0);
                ui.separator();
                ui.label(egui::RichText::new("Status:").strong());
                ui.colored_label(Color32::from_rgb(40, 140, 60), &self.status_message);
            });
        });

        ui.separator();

        // 2. BOTTOM ANALYTICS BAR
        ui.horizontal_wrapped(|ui| {
            if let Some(sim) = &self.latest_simulation {
                ui.label(egui::RichText::new("METRICS:").strong());
                let col = if sim.monitored_similarity_pct >= 85.0 {
                    Color32::from_rgb(0, 150, 60)
                } else {
                    Color32::from_rgb(220, 120, 0)
                };
                ui.colored_label(
                    col,
                    egui::RichText::new(format!("Monitored Precision: {:.1}%", sim.monitored_similarity_pct))
                        .strong(),
                );
                ui.separator();
                ui.label(format!("Full Journey Match: {:.1}%", sim.full_trip_similarity_pct));
                ui.separator();
                ui.label(format!("True Trip: {:.0}m", sim.full_trip_len_m));
                ui.separator();
                ui.label(format!("Inferred: {:.0}m", sim.reconstructed_len_m));
                ui.separator();
                ui.label(format!("Max Spatial Drift: {:.1}m", sim.max_spatial_drift_m));
                ui.separator();
                ui.label(format!("Camera Hits: {}", sim.trip.camera_sightings.len()));
                ui.separator();
                ui.colored_label(Color32::from_rgb(0, 120, 215), format!("Solve Time: {} µs", sim.execution_time_us));
            } else {
                ui.label("No active simulation.");
            }
        });

        // 3. BATCH MODAL DIALOG
        if self.show_batch_window {
            if let Some(rep) = &self.batch_report {
                render_batch_report_window(ui.ctx(), &mut self.show_batch_window, rep);
            }
        }
    }
}

pub fn launch_interactive_workbench(city: &'static CityGraph) -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1550.0, 920.0])
            .with_title("Mumbai ALPR Surveillance Engine & Optimization Workbench"),
        ..Default::default()
    };

    eframe::run_native(
        "Mumbai Surveillance Workbench",
        native_options,
        Box::new(|_cc| Ok(Box::new(InteractiveWorkbench::new(city)))),
    )
}