use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};
use engine_core::CityGraph;
use network_optimizer::OptimizedCameraPole;
use petgraph::visit::EdgeRef;
use trajectory_resolver::VehicleTrackState;
use trip_simulator::SimulatedTrip;

#[derive(Clone, Copy)]
pub struct ViewportBounds {
    pub min_lon: f64,
    pub max_lon: f64,
    pub min_lat: f64,
    pub max_lat: f64,
}

impl ViewportBounds {
    pub fn from_trip_and_track(city: &CityGraph, trip: &SimulatedTrip, track: &VehicleTrackState) -> Self {
        let mut min_lon = f64::INFINITY;
        let mut max_lon = f64::NEG_INFINITY;
        let mut min_lat = f64::INFINITY;
        let mut max_lat = f64::NEG_INFINITY;

        for &nid in &trip.ground_truth_nodes {
            if let Some(&idx) = city.node_map.get(&nid) {
                let m = &city.graph[idx];
                min_lon = min_lon.min(m.lon);
                max_lon = max_lon.max(m.lon);
                min_lat = min_lat.min(m.lat);
                max_lat = max_lat.max(m.lat);
            }
        }
        for &nid in &track.reconstructed_path {
            if let Some(&idx) = city.node_map.get(&nid) {
                let m = &city.graph[idx];
                min_lon = min_lon.min(m.lon);
                max_lon = max_lon.max(m.lon);
                min_lat = min_lat.min(m.lat);
                max_lat = max_lat.max(m.lat);
            }
        }

        let d_lon = (max_lon - min_lon).max(0.003) * 0.20;
        let d_lat = (max_lat - min_lat).max(0.003) * 0.20;

        Self {
            min_lon: min_lon - d_lon,
            max_lon: max_lon + d_lon,
            min_lat: min_lat - d_lat,
            max_lat: max_lat + d_lat,
        }
    }

    pub fn project(&self, lat: f64, lon: f64, rect: Rect) -> Pos2 {
        let x_norm = (lon - self.min_lon) / (self.max_lon - self.min_lon);
        let y_norm = (lat - self.min_lat) / (self.max_lat - self.min_lat);
        Pos2::new(
            rect.min.x + (x_norm as f32) * rect.width(),
            rect.max.y - (y_norm as f32) * rect.height(),
        )
    }

    pub fn render_road_network(&self, city: &CityGraph, painter: &egui::Painter, rect: Rect) {
        for edge in city.graph.edge_references() {
            let u = &city.graph[edge.source()];
            let v = &city.graph[edge.target()];

            if (u.lon < self.min_lon && v.lon < self.min_lon)
                || (u.lon > self.max_lon && v.lon > self.max_lon)
                || (u.lat < self.min_lat && v.lat < self.min_lat)
                || (u.lat > self.max_lat && v.lat > self.max_lat)
            {
                continue;
            }

            let p1 = self.project(u.lat, u.lon, rect);
            let p2 = self.project(v.lat, v.lon, rect);
            painter.line_segment([p1, p2], Stroke::new(1.0, Color32::from_rgb(220, 222, 225)));
        }
    }
}

pub fn render_ground_truth_canvas(
    ui: &mut egui::Ui,
    city: &CityGraph,
    poles: &[OptimizedCameraPole],
    trip: Option<&SimulatedTrip>,
    bounds: &ViewportBounds,
    width: f32,
    height: f32,
) {
    ui.vertical(|ui| {
        ui.set_width(width);
        ui.heading(
            egui::RichText::new("REAL WORLD: Ground Truth & Telemetry")
                .color(Color32::from_rgb(0, 150, 70))
                .strong(),
        );

        let (resp, painter) = ui.allocate_painter(Vec2::new(width, height), egui::Sense::hover());
        let rect = resp.rect;
        painter.rect_filled(rect, 4.0, Color32::from_rgb(250, 250, 252));

        bounds.render_road_network(city, &painter, rect);

        for pole in poles {
            if pole.lon >= bounds.min_lon && pole.lon <= bounds.max_lon
                && pole.lat >= bounds.min_lat && pole.lat <= bounds.max_lat
            {
                let pt = bounds.project(pole.lat, pole.lon, rect);
                painter.circle_filled(pt, 3.5, Color32::from_rgb(170, 175, 180));
            }
        }

        if let Some(t) = trip {
            let mut prev_pt = None;
            for &nid in &t.ground_truth_nodes {
                if let Some(&idx) = city.node_map.get(&nid) {
                    let pt = bounds.project(city.graph[idx].lat, city.graph[idx].lon, rect);
                    if let Some(p) = prev_pt {
                        painter.line_segment([p, pt], Stroke::new(4.0, Color32::from_rgb(0, 180, 70)));
                    }
                    prev_pt = Some(pt);
                }
            }

            for s in &t.camera_sightings {
                if let Some(&idx) = city.node_map.get(&s.edge_from) {
                    let pt = bounds.project(city.graph[idx].lat, city.graph[idx].lon, rect);
                    painter.circle_filled(pt, 6.0, Color32::from_rgb(0, 220, 80));
                    painter.circle_stroke(pt, 6.0, Stroke::new(1.8, Color32::BLACK));
                }
            }
        }
    });
}

pub fn render_reconstructed_canvas(
    ui: &mut egui::Ui,
    city: &CityGraph,
    track: Option<&VehicleTrackState>,
    bounds: &ViewportBounds,
    width: f32,
    height: f32,
) {
    ui.vertical(|ui| {
        ui.set_width(width);
        ui.heading(
            egui::RichText::new("SURVEILLANCE ENGINE: Inferred Trajectory")
                .color(Color32::from_rgb(20, 100, 215))
                .strong(),
        );

        let (resp, painter) = ui.allocate_painter(Vec2::new(width, height), egui::Sense::hover());
        let rect = resp.rect;
        painter.rect_filled(rect, 4.0, Color32::from_rgb(250, 250, 252));

        bounds.render_road_network(city, &painter, rect);

        if let Some(tr) = track {
            for leg in &tr.leg_hypotheses {
                for alt in leg.alternatives.iter().skip(1) {
                    let mut prev = None;
                    for &nid in &alt.path {
                        if let Some(&idx) = city.node_map.get(&nid) {
                            let pt = bounds.project(city.graph[idx].lat, city.graph[idx].lon, rect);
                            if let Some(p) = prev {
                                painter.line_segment([p, pt], Stroke::new(2.4, Color32::from_rgb(245, 160, 0)));
                            }
                            prev = Some(pt);
                        }
                    }
                }
            }

            let mut prev = None;
            for &nid in &tr.reconstructed_path {
                if let Some(&idx) = city.node_map.get(&nid) {
                    let pt = bounds.project(city.graph[idx].lat, city.graph[idx].lon, rect);
                    if let Some(p) = prev {
                        painter.line_segment([p, pt], Stroke::new(3.8, Color32::from_rgb(220, 30, 45)));
                    }
                    prev = Some(pt);
                }
            }
        }
    });
}