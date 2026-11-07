//! Bounds-checked tensor parser for YOLO object detection models.

use super::nms::MAX_NMS_CANDIDATES;
use super::types::{BoundingBox, DetectedVehicle, VehicleClass};
use heapless::Vec as StackVec;

/// Geometric inverse transform descriptor from letterbox coordinates to source image.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxTransform {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

impl LetterboxTransform {
    #[inline(always)]
    pub fn unproject_bbox(&self, x1: f32, y1: f32, x2: f32, y2: f32) -> BoundingBox {
        let inv_scale = if self.scale > 1e-6 { 1.0 / self.scale } else { 1.0 };
        let ox1 = ((x1 - self.pad_x) * inv_scale).max(0.0);
        let oy1 = ((y1 - self.pad_y) * inv_scale).max(0.0);
        let ox2 = ((x2 - self.pad_x) * inv_scale).max(0.0);
        let oy2 = ((y2 - self.pad_y) * inv_scale).max(0.0);

        BoundingBox::new_clamped(ox1, oy1, ox2, oy2)
    }
}

/// Decodes raw tensor float buffer into candidate vehicle detections.
///
/// Supports:
/// 1. End-to-end NMS baked output: `[1, num_boxes, 6]`
/// 2. Columnar / Transposed YOLO: `[1, channels, proposals]` (e.g. `[1, 9, 8400]` or `[1, 84, 8400]`)
/// 3. Standard Planar YOLO: `[1, proposals, channels]` (e.g. `[1, 8400, 9]` or `[1, 8400, 84]`)
///
/// Guaranteed panic-free: uses safe slices, bounded loops, and explicit capacity checking.
pub fn parse_model_proposals(
    dims: &[i64],
    data: &[f32],
    conf_thresh: f32,
    min_dim: f32,
    transform: &LetterboxTransform,
    out: &mut StackVec<DetectedVehicle, MAX_NMS_CANDIDATES>,
) -> Result<(), &'static str> {
    out.clear();

    if dims.len() != 3 {
        return Err("Model output must have 3 dimensions");
    }

    let d0 = dims[0] as usize;
    let d1 = dims[1] as usize;
    let d2 = dims[2] as usize;

    if d0 != 1 {
        return Err("Only batch size 1 is supported");
    }

    if d2 == 6 {
        // End-to-end NMS baked: [1, num_boxes, 6] -> [x1, y1, x2, y2, conf, class_id]
        let num_boxes = d1;
        for i in 0..num_boxes {
            let offset = i * 6;
            if offset + 5 >= data.len() {
                break;
            }

            let conf = data[offset + 4];
            if conf < conf_thresh {
                continue;
            }

            let raw_cls = data[offset + 5].max(0.0).round() as usize;
            
            // Check custom 5-class first (0..=4), then COCO fallback
            let vehicle_class = VehicleClass::from_custom_5class_id(raw_cls)
                .or_else(|| VehicleClass::from_coco_id(raw_cls));

            if let Some(cls) = vehicle_class {
                let bbox = transform.unproject_bbox(
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                );

                if bbox.width() >= min_dim && bbox.height() >= min_dim {
                    let _ = out.push(DetectedVehicle {
                        class: cls,
                        confidence: conf,
                        bbox,
                    });
                    if out.is_full() {
                        break;
                    }
                }
            }
        }
    } else if d1 < d2 {
        // Columnar / Transposed format [1, channels, proposals]
        // Base YOLO26n COCO: [1, 84, 8400] -> channels = 84, total_classes = 80
        // Fine-tuned 5-class: [1, 9, 8400]  -> channels = 9,  total_classes = 5
        let channels = d1;
        let proposals = d2;
        let total_classes = channels.saturating_sub(4);

        if total_classes == 0 {
            return Err("Tensor channels must be greater than 4");
        }

        for i in 0..proposals {
            let mut best_class = 0;
            let mut max_conf = 0.0f32;

            for c in 4..channels {
                let idx = c * proposals + i;
                if let Some(&conf) = data.get(idx) {
                    if conf > max_conf {
                        max_conf = conf;
                        best_class = c - 4;
                    }
                }
            }

            if max_conf >= conf_thresh {
                if let Some(vehicle_class) = VehicleClass::from_model_output(best_class, total_classes) {
                    let cx = *data.get(i).unwrap_or(&0.0);
                    let cy = *data.get(proposals + i).unwrap_or(&0.0);
                    let w = *data.get(2 * proposals + i).unwrap_or(&0.0);
                    let h = *data.get(3 * proposals + i).unwrap_or(&0.0);

                    let bbox = transform.unproject_bbox(
                        cx - w * 0.5,
                        cy - h * 0.5,
                        cx + w * 0.5,
                        cy + h * 0.5,
                    );

                    if bbox.width() >= min_dim && bbox.height() >= min_dim {
                        let _ = out.push(DetectedVehicle {
                            class: vehicle_class,
                            confidence: max_conf,
                            bbox,
                        });
                        if out.is_full() {
                            break;
                        }
                    }
                }
            }
        }
    } else {
        // Planar format [1, proposals, channels]
        // e.g. [1, 8400, 84] or [1, 8400, 9]
        let proposals = d1;
        let channels = d2;
        let total_classes = channels.saturating_sub(4);

        if total_classes == 0 {
            return Err("Tensor channels must be greater than 4");
        }

        for i in 0..proposals {
            let row = i * channels;
            let mut best_class = 0;
            let mut max_conf = 0.0f32;

            for c in 4..channels {
                if let Some(&conf) = data.get(row + c) {
                    if conf > max_conf {
                        max_conf = conf;
                        best_class = c - 4;
                    }
                }
            }

            if max_conf >= conf_thresh {
                if let Some(vehicle_class) = VehicleClass::from_model_output(best_class, total_classes) {
                    let cx = *data.get(row).unwrap_or(&0.0);
                    let cy = *data.get(row + 1).unwrap_or(&0.0);
                    let w = *data.get(row + 2).unwrap_or(&0.0);
                    let h = *data.get(row + 3).unwrap_or(&0.0);

                    let bbox = transform.unproject_bbox(
                        cx - w * 0.5,
                        cy - h * 0.5,
                        cx + w * 0.5,
                        cy + h * 0.5,
                    );

                    if bbox.width() >= min_dim && bbox.height() >= min_dim {
                        let _ = out.push(DetectedVehicle {
                            class: vehicle_class,
                            confidence: max_conf,
                            bbox,
                        });
                        if out.is_full() {
                            break;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}