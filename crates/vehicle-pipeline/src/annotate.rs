//! Optional visualizer for development debugging and test verification.

#[cfg(feature = "annotate")]
use ab_glyph::FontArc;
use image::{DynamicImage, Rgb, RgbImage};
#[cfg(feature = "annotate")]
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
#[cfg(feature = "annotate")]
use imageproc::rect::Rect;
use std::path::Path;
use crate::types::VehicleSnapshot;

pub struct FrameAnnotator {
    #[cfg(feature = "annotate")]
    font: Option<FontArc>,
}

impl FrameAnnotator {
    pub fn new(_font_path: Option<&Path>) -> Self {
        #[cfg(feature = "annotate")]
        {
            let font = _font_path
                .and_then(|p| std::fs::read(p).ok())
                .and_then(|b| FontArc::try_from_vec(b).ok())
                .or_else(Self::discover_system_font);

            Self { font }
        }
        #[cfg(not(feature = "annotate"))]
        {
            Self {}
        }
    }

    #[cfg(feature = "annotate")]
    fn discover_system_font() -> Option<FontArc> {
        let candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        ];

        for path in candidates {
            if let Ok(bytes) = std::fs::read(path) {
                if let Ok(font) = FontArc::try_from_vec(bytes) {
                    return Some(font);
                }
            }
        }
        None
    }

    /// Renders overlays onto frame. If feature is disabled, returns unannotated RGB image with zero overhead.
    pub fn annotate(&self, frame: &DynamicImage, snapshots: &[VehicleSnapshot]) -> RgbImage {
        let mut annotated = frame.to_rgb8();

        #[cfg(feature = "annotate")]
        {
            for v in snapshots {
                let x = v.bbox.x1.max(0.0) as i32;
                let y = v.bbox.y1.max(0.0) as i32;
                let w = v.bbox.width().max(1.0) as u32;
                let h = v.bbox.height().max(1.0) as u32;

                // 1. Vehicle Bounding Box (Vibrant Green)
                let v_rect = Rect::at(x, y).of_size(w, h);
                draw_hollow_rect_mut(&mut annotated, v_rect, Rgb([0, 255, 128]));

                // 2. Vehicle Tag Banner
                if let Some(ref f) = self.font {
                    let v_label = format!("#{} {}", v.track_id, v.class.as_str());
                    let label_w = (v_label.len() as u32 * 9) + 6;
                    let tag_y = (y - 18).max(0);
                    draw_filled_rect_mut(&mut annotated, Rect::at(x, tag_y).of_size(label_w, 18), Rgb([0, 0, 0]));
                    draw_text_mut(&mut annotated, Rgb([0, 255, 128]), x + 3, tag_y + 1, 14.0, f, &v_label);
                }

                // 3. Plate Bounding Box (Bright Yellow)
                if let Some(p_box) = v.plate_bbox {
                    let px = p_box.x1.max(0.0) as i32;
                    let py = p_box.y1.max(0.0) as i32;
                    let pw = p_box.width().max(1.0) as u32;
                    let ph = p_box.height().max(1.0) as u32;

                    let p_rect = Rect::at(px, py).of_size(pw, ph);
                    draw_hollow_rect_mut(&mut annotated, p_rect, Rgb([255, 235, 0]));

                    if let (Some(ref ocr), Some(ref f)) = (&v.plate_ocr, &self.font) {
                        let plate_label = format!("{} ({:.0}%)", ocr.as_str(), ocr.confidence * 100.0);
                        let banner_w = (plate_label.len() as u32 * 10) + 8;
                        let text_y = (py - 20).max(0);

                        draw_filled_rect_mut(&mut annotated, Rect::at(px, text_y).of_size(banner_w, 19), Rgb([0, 0, 0]));
                        draw_text_mut(&mut annotated, Rgb([255, 235, 0]), px + 4, text_y + 1, 15.0, f, &plate_label);
                    }
                }
            }
        }

        annotated
    }
}