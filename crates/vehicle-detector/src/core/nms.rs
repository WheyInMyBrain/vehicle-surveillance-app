//! Non-Maximum Suppression (NMS) with compile-time bounded capacity.
//!
//! Adheres to JPL Rules: bounded loops, zero heap allocation, deterministic execution time.

use super::types::{BoundingBox, DetectedVehicle};
use heapless::Vec as StackVec;

/// Maximum number of candidate proposals evaluated by NMS.
pub const MAX_NMS_CANDIDATES: usize = 256;
/// Maximum number of suppressed output detections.
pub const MAX_OUTPUT_DETECTIONS: usize = 32;

/// Fast Intersection-over-Union calculation using SIMD-friendly primitives.
///
/// # Guarantees
/// - Returns `0.0` for invalid/negative dimensions or zero union.
/// - Never divides by zero or returns NaN.
#[inline(always)]
pub fn calculate_iou_fast(a: &BoundingBox, a_area: f32, b: &BoundingBox, b_area: f32) -> f32 {
    let x1 = if a.x1 > b.x1 { a.x1 } else { b.x1 };
    let y1 = if a.y1 > b.y1 { a.y1 } else { b.y1 };
    let x2 = if a.x2 < b.x2 { a.x2 } else { b.x2 };
    let y2 = if a.y2 < b.y2 { a.y2 } else { b.y2 };

    let inter_w = (x2 - x1).max(0.0);
    let inter_h = (y2 - y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let union_area = a_area + b_area - inter_area;
    if union_area <= 1e-6 {
        0.0
    } else {
        inter_area / union_area
    }
}

/// Executes Non-Maximum Suppression entirely on the stack.
///
/// # Complexity
/// - Space: O(1) - Zero heap allocations.
/// - Time: Bounded by `O(MAX_NMS_CANDIDATES * MAX_OUTPUT_DETECTIONS)`.
///
/// # Panics
/// Strictly panic-free. Loop counters and slice accesses are provably bounded.
pub fn run_bounded_nms(
    proposals: &mut StackVec<DetectedVehicle, MAX_NMS_CANDIDATES>,
    iou_threshold: f32,
) -> StackVec<DetectedVehicle, MAX_OUTPUT_DETECTIONS> {
    let mut outputs: StackVec<DetectedVehicle, MAX_OUTPUT_DETECTIONS> = StackVec::new();
    let n = proposals.len();
    if n == 0 {
        return outputs;
    }

    // Sort descending by confidence
    proposals.sort_unstable_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    let mut suppressed = [false; MAX_NMS_CANDIDATES];
    let mut areas = [0.0f32; MAX_NMS_CANDIDATES];

    // Precalculate areas up to active length
    for i in 0..n {
        if let Some(prop) = proposals.get(i) {
            areas[i] = prop.area();
        }
    }

    for i in 0..n {
        if suppressed[i] {
            continue;
        }

        let p_i = match proposals.get(i) {
            Some(v) => *v,
            None => break,
        };

        if outputs.push(p_i).is_err() {
            // Reached maximum allowed outputs
            break;
        }

        let area_i = areas[i];

        for j in (i + 1)..n {
            if suppressed[j] {
                continue;
            }

            if let Some(p_j) = proposals.get(j) {
                let iou = calculate_iou_fast(&p_i.bbox, area_i, &p_j.bbox, areas[j]);
                if iou > iou_threshold {
                    suppressed[j] = true;
                }
            }
        }
    }

    outputs
}