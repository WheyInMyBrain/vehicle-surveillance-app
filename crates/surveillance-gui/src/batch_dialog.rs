use crate::batch_eval::BatchMetricsReport;
use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};

pub fn render_batch_report_window(ctx: &egui::Context, open: &mut bool, report: &BatchMetricsReport) {
    egui::Window::new("🔬 High-Volume Monte Carlo Analytical Dashboard")
        .open(open)
        .resizable(true)
        .default_width(780.0)
        .default_height(620.0)
        .show(ctx, |ui| {
            ui.heading("Surveillance Engine Statistical Fidelity");
            ui.label(
                egui::RichText::new(format!(
                    "Evaluated {} simulations in {:.2}s ({:.0} runs/sec) across all threads.",
                    report.total_simulations,
                    report.execution_duration_sec,
                    report.total_simulations as f64 / report.execution_duration_sec.max(0.001)
                ))
                .weak(),
            );
            ui.separator();

            // --- 1. CORE KPI TILES ---
            ui.columns(4, |cols| {
                cols[0].group(|ui| {
                    ui.label("Intercept Rate");
                    ui.heading(format!("{:.1}%", report.intercept_rate_pct));
                    ui.label(egui::RichText::new(format!("{} trips intercepted", report.successful_intercepts)).weak().size(11.0));
                });
                cols[1].group(|ui| {
                    ui.label("Median Accuracy (p50)");
                    ui.heading(format!("{:.1}%", report.accuracy_p50));
                    ui.label(egui::RichText::new(format!("Mean: {:.1}%", report.accuracy_mean)).weak().size(11.0));
                });
                cols[2].group(|ui| {
                    ui.label("Journey Capture");
                    ui.heading(format!("{:.1}%", report.capture_ratio_p50));
                    ui.label(egui::RichText::new(format!("p90: {:.1}%", report.capture_ratio_p90)).weak().size(11.0));
                });
                cols[3].group(|ui| {
                    ui.label("Mean Solve Latency");
                    ui.heading(format!("{:.1} µs", report.latency_mean_us));
                    ui.label(egui::RichText::new(format!("p99: {:.1} µs", report.latency_p99_us)).weak().size(11.0));
                });
            });

            ui.add_space(8.0);
            ui.separator();

            // --- 2. VECTOR GRAPHS (HISTOGRAM & CDF) ---
            ui.columns(2, |cols| {
                // COLUMN A: ACCURACY HISTOGRAM
                cols[0].vertical(|ui| {
                    ui.label(egui::RichText::new("Sequence Accuracy Distribution").strong());
                    let (resp, painter) = ui.allocate_painter(Vec2::new(350.0, 160.0), egui::Sense::hover());
                    let rect = resp.rect;
                    painter.rect_filled(rect, 4.0, Color32::from_rgb(245, 246, 248));

                    let max_count = *report.accuracy_distribution.iter().max().unwrap_or(&1).max(&1) as f32;
                    let bar_w = rect.width() / 10.0;

                    for i in 0..10 {
                        let count = report.accuracy_distribution[i] as f32;
                        let h = (count / max_count) * (rect.height() - 25.0);
                        let bar_rect = Rect::from_min_max(
                            Pos2::new(rect.min.x + (i as f32) * bar_w + 2.0, rect.max.y - 18.0 - h),
                            Pos2::new(rect.min.x + ((i + 1) as f32) * bar_w - 2.0, rect.max.y - 18.0),
                        );
                        let color = if i >= 8 {
                            Color32::from_rgb(0, 160, 75)
                        } else if i >= 6 {
                            Color32::from_rgb(0, 130, 220)
                        } else if i >= 4 {
                            Color32::from_rgb(230, 140, 0)
                        } else {
                            Color32::from_rgb(215, 45, 45)
                        };
                        painter.rect_filled(bar_rect, 2.0, color);
                    }

                    // Base axis label
                    painter.text(
                        Pos2::new(rect.min.x + 8.0, rect.max.y - 14.0),
                        egui::Align2::LEFT_TOP,
                        "0%",
                        egui::FontId::proportional(11.0),
                        Color32::DARK_GRAY,
                    );
                    painter.text(
                        Pos2::new(rect.max.x - 8.0, rect.max.y - 14.0),
                        egui::Align2::RIGHT_TOP,
                        "100%",
                        egui::FontId::proportional(11.0),
                        Color32::DARK_GRAY,
                    );
                });

                // COLUMN B: CUMULATIVE DISTRIBUTION FUNCTION (CDF)
                cols[1].vertical(|ui| {
                    ui.label(egui::RichText::new("Cumulative Distribution Function (CDF)").strong());
                    let (resp, painter) = ui.allocate_painter(Vec2::new(350.0, 160.0), egui::Sense::hover());
                    let rect = resp.rect;
                    painter.rect_filled(rect, 4.0, Color32::from_rgb(245, 246, 248));

                    // Grid lines
                    let stroke = Stroke::new(0.8, Color32::from_rgb(220, 222, 226));
                    painter.line_segment([Pos2::new(rect.min.x, rect.min.y + rect.height() * 0.5), Pos2::new(rect.max.x, rect.min.y + rect.height() * 0.5)], stroke);

                    let to_pos = |acc: f32, cum: f32| -> Pos2 {
                        let x = rect.min.x + (acc / 100.0) * (rect.width() - 20.0) + 10.0;
                        let y = rect.max.y - (cum / 100.0) * (rect.height() - 30.0) - 15.0;
                        Pos2::new(x, y)
                    };

                    let pts: Vec<Pos2> = report.cdf_samples.iter().map(|&(a, c)| to_pos(a, c)).collect();
                    for w in pts.windows(2) {
                        painter.line_segment([w[0], w[1]], Stroke::new(2.5, Color32::from_rgb(120, 40, 200)));
                    }

                    painter.text(Pos2::new(rect.min.x + 10.0, rect.min.y + 6.0), egui::Align2::LEFT_TOP, "100%", egui::FontId::proportional(10.0), Color32::DARK_GRAY);
                    painter.text(Pos2::new(rect.min.x + 10.0, rect.max.y - 14.0), egui::Align2::LEFT_TOP, "0%", egui::FontId::proportional(10.0), Color32::DARK_GRAY);
                });
            });

            ui.add_space(8.0);
            ui.separator();

            // --- 3. PERCENTILE BREAKDOWN TABLE ---
            ui.label(egui::RichText::new("Detailed Percentile Breakdown").strong());
            egui::Grid::new("batch_percentiles_grid")
                .striped(true)
                .min_col_width(120.0)
                .show(ui, |ui| {
                    ui.strong("Metric");
                    ui.strong("p25");
                    ui.strong("p50 (Median)");
                    ui.strong("p75");
                    ui.strong("p90");
                    ui.strong("p99");
                    ui.end_row();

                    ui.label("Monitored Accuracy");
                    ui.label(format!("{:.1}%", report.accuracy_p25));
                    ui.label(format!("{:.1}%", report.accuracy_p50));
                    ui.label(format!("{:.1}%", report.accuracy_p75));
                    ui.label(format!("{:.1}%", report.accuracy_p90));
                    ui.label(format!("{:.1}%", report.accuracy_p99));
                    ui.end_row();

                    ui.label("Journey Capture Ratio");
                    ui.label("-");
                    ui.label(format!("{:.1}%", report.capture_ratio_p50));
                    ui.label("-");
                    ui.label(format!("{:.1}%", report.capture_ratio_p90));
                    ui.label("-");
                    ui.end_row();

                    ui.label("Unobserved Blind Gap");
                    ui.label("-");
                    ui.label(format!("{:.0}m", report.avg_unobserved_gap_m));
                    ui.label("-");
                    ui.label(format!("{:.0}m", report.p90_unobserved_gap_m));
                    ui.label("-");
                    ui.end_row();
                });
        });
}