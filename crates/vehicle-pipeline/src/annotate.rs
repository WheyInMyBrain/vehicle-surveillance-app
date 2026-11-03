use ab_glyph::FontArc;
use image::{DynamicImage, Rgb, RgbImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use std::fs::read;
use std::path::Path;
use crate::types::VehicleSnapshot;

pub struct FrameAnnotator {
    font: Option<FontArc>,
}

impl FrameAnnotator {
    pub fn new(font_path: Option<&Path>) -> Self {
        let font = font_path
            .and_then(|p| read(p).ok())
            .and_then(|b| FontArc::try_from_vec(b).ok())
            .or_else(Self::discover_system_font);

        Self { font }
    }

    fn discover_system_font() -> Option<FontArc> {
        let candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        ];

        for path in candidates {
            if let Ok(bytes) = read(path) {
                if let Ok(font) = FontArc::try_from_vec(bytes) {
                    return Some(font);
                }
            }
        }
        None
    }

    pub fn annotate(&self, frame: &DynamicImage, snapshots: &[VehicleSnapshot]) -> RgbImage {
        let mut annotated = frame.to_rgb8();

        for v in snapshots {
            let x = v.bbox[0].max(0.0) as i32;
            let y = v.bbox[1].max(0.0) as i32;
            let w = (v.bbox[2] - v.bbox[0]).max(1.0) as u32;
            let h = (v.bbox[3] - v.bbox[1]).max(1.0) as u32;

            // 1. Vehicle Bounding Box (Vibrant Green - 2px thick)
            let v_rect = Rect::at(x, y).of_size(w, h);
            draw_hollow_rect_mut(&mut annotated, v_rect, Rgb([0, 255, 128]));
            draw_hollow_rect_mut(&mut annotated, Rect::at(x + 1, y + 1).of_size(w.saturating_sub(2).max(1), h.saturating_sub(2).max(1)), Rgb([0, 255, 128]));

            // 2. Vehicle Tag Banner
            if let Some(ref f) = self.font {
                let v_label = format!("#{} {}", v.track_id, v.class.as_str());
                let label_w = (v_label.len() as u32 * 9) + 6;
                let tag_y = (y - 18).max(0);
                draw_filled_rect_mut(&mut annotated, Rect::at(x, tag_y).of_size(label_w, 18), Rgb([0, 0, 0]));
                draw_text_mut(&mut annotated, Rgb([0, 255, 128]), x + 3, tag_y + 1, 14.0, f, &v_label);
            }

            // 3. License Plate Bounding Box (Bright Yellow)
            if let Some(p_box) = v.plate_bbox {
                let px = p_box[0].max(0.0) as i32;
                let py = p_box[1].max(0.0) as i32;
                let pw = (p_box[2] - p_box[0]).max(1.0) as u32;
                let ph = (p_box[3] - p_box[1]).max(1.0) as u32;

                let p_rect = Rect::at(px, py).of_size(pw, ph);
                draw_hollow_rect_mut(&mut annotated, p_rect, Rgb([255, 235, 0]));
                draw_hollow_rect_mut(&mut annotated, Rect::at(px + 1, py + 1).of_size(pw.saturating_sub(2).max(1), ph.saturating_sub(2).max(1)), Rgb([255, 235, 0]));

                // 4. Plate OCR Text Banner
                if let (Some(ref ocr), Some(ref f)) = (&v.plate_ocr, &self.font) {
                    let plate_label = format!("{} ({:.0}%)", ocr.text, ocr.confidence * 100.0);
                    let banner_w = (plate_label.len() as u32 * 10) + 8;
                    let text_y = (py - 20).max(0);

                    draw_filled_rect_mut(&mut annotated, Rect::at(px, text_y).of_size(banner_w, 19), Rgb([0, 0, 0]));
                    draw_text_mut(&mut annotated, Rgb([255, 235, 0]), px + 4, text_y + 1, 15.0, f, &plate_label);
                }
            }
        }

        annotated
    }
}