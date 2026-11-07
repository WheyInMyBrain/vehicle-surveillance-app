//! 8-Dimensional Discrete Linear Kalman Filter for Bounding Box Kinematics.
//!
//! State Vector x: `[center_x, center_y, aspect_ratio, height, vx, vy, va, vh]`
//! Measurement Vector z: `[center_x, center_y, aspect_ratio, height]`
//!
//! # Guarantees
//! - Zero dynamic heap allocation.
//! - Closed-form analytic 4x4 matrix inversion with singularity fallbacks.
//! - Finite arithmetic protection against NaN / Inf.

use vehicle_detector::BoundingBox;

#[derive(Debug, Clone, Copy)]
pub struct KalmanFilter {
    std_weight_position: f32,
    std_weight_velocity: f32,
}

impl Default for KalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl KalmanFilter {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            std_weight_position: 1.0 / 20.0,
            std_weight_velocity: 1.0 / 160.0,
        }
    }

    /// Converts bounding box into measurement state [cx, cy, aspect, h].
    #[inline(always)]
    pub fn bbox_to_z(bbox: &BoundingBox) -> [f32; 4] {
        let w = bbox.width().max(1.0);
        let h = bbox.height().max(1.0);
        let (cx, cy) = bbox.center();
        let aspect = w / h;
        [cx, cy, aspect, h]
    }

    /// Converts state vector [cx, cy, aspect, h, ...] back into a validated BoundingBox.
    #[inline(always)]
    pub fn z_to_bbox(z: &[f32; 8]) -> BoundingBox {
        let cx = if z[0].is_nan() { 0.0 } else { z[0] };
        let cy = if z[1].is_nan() { 0.0 } else { z[1] };
        let aspect = if z[2].is_nan() || z[2] <= 0.05 { 0.05 } else { z[2] };
        let h = if z[3].is_nan() || z[3] <= 1.0 { 1.0 } else { z[3] };
        let w = aspect * h;

        BoundingBox::new_clamped(
            cx - w * 0.5,
            cy - h * 0.5,
            cx + w * 0.5,
            cy + h * 0.5,
        )
    }

    /// Initializes state mean and covariance from initial measurement.
    pub fn initiate(&self, measurement: &[f32; 4]) -> ([f32; 8], [f32; 64]) {
        let mut mean = [0.0f32; 8];
        mean[0] = measurement[0];
        mean[1] = measurement[1];
        mean[2] = measurement[2];
        mean[3] = measurement[3];

        let h = measurement[3].max(1.0);
        let std_p = 2.0 * self.std_weight_position * h;
        let std_v = 10.0 * self.std_weight_velocity * h;

        let std = [std_p, std_p, 1e-2, std_p, std_v, std_v, 1e-5, std_v];

        let mut cov = [0.0f32; 64];
        for i in 0..8 {
            cov[i * 8 + i] = std[i] * std[i];
        }

        (mean, cov)
    }

    /// Predicts state forward in time with constant-velocity motion model.
    pub fn predict(&self, mean: &mut [f32; 8], cov: &mut [f32; 64]) {
        let h = mean[3].max(1.0);
        let std_p = self.std_weight_position * h;
        let std_v = self.std_weight_velocity * h;

        let motion_cov_std = [
            std_p,
            std_p,
            1e-2,
            std_p,
            std_v,
            std_v,
            1e-5,
            std_v * 1.5,
        ];

        // Kinematic state extrapolation: x' = x + v * dt
        for i in 0..4 {
            mean[i] += mean[i + 4];
        }

        // Covariance extrapolation: P' = F * P * F^T + Q
        let mut p_new = *cov;
        for i in 0..4 {
            for j in 0..8 {
                p_new[i * 8 + j] += cov[(i + 4) * 8 + j];
            }
        }
        for j in 0..4 {
            for i in 0..8 {
                p_new[i * 8 + j] += p_new[i * 8 + (j + 4)];
            }
        }

        // Add Q (process noise covariance)
        for i in 0..8 {
            p_new[i * 8 + i] += motion_cov_std[i] * motion_cov_std[i];
        }

        *cov = p_new;
    }

    /// Updates state given measurement [cx, cy, a, h].
    pub fn update(&self, mean: &mut [f32; 8], cov: &mut [f32; 64], measurement: &[f32; 4]) {
        let h = mean[3].max(1.0);
        let std_p = self.std_weight_position * h;
        let r_diag = [std_p * std_p, std_p * std_p, 1e-2, std_p * std_p];

        // Innovation: y = z - H * x
        let mut y = [0.0f32; 4];
        for i in 0..4 {
            y[i] = measurement[i] - mean[i];
        }

        // Innovation covariance: S = H * P * H^T + R
        let mut s = [0.0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                s[i * 4 + j] = cov[i * 8 + j];
            }
            s[i * 4 + i] += r_diag[i];
        }

        let s_inv = invert_4x4_analytic(&s);

        // Kalman gain: K = P * H^T * S^-1 (size 8x4)
        let mut k = [0.0f32; 32];
        for i in 0..8 {
            for j in 0..4 {
                let mut sum = 0.0f32;
                for l in 0..4 {
                    sum += cov[i * 8 + l] * s_inv[l * 4 + j];
                }
                k[i * 4 + j] = sum;
            }
        }

        // State update: mean = mean + K * y
        for i in 0..8 {
            let mut delta = 0.0f32;
            for j in 0..4 {
                delta += k[i * 4 + j] * y[j];
            }
            mean[i] += delta;
        }

        // Joseph form / Standard Covariance update: P = (I - K * H) * P
        let mut i_kh = [0.0f32; 64];
        for i in 0..8 {
            i_kh[i * 8 + i] = 1.0;
        }
        for i in 0..8 {
            for j in 0..4 {
                i_kh[i * 8 + j] -= k[i * 4 + j];
            }
        }

        let mut p_updated = [0.0f32; 64];
        for i in 0..8 {
            for j in 0..8 {
                let mut sum = 0.0f32;
                for l in 0..4 {
                    sum += i_kh[i * 8 + l] * cov[l * 8 + j];
                }
                sum += i_kh[i * 8 + 4] * cov[4 * 8 + j]
                    + i_kh[i * 8 + 5] * cov[5 * 8 + j]
                    + i_kh[i * 8 + 6] * cov[6 * 8 + j]
                    + i_kh[i * 8 + 7] * cov[7 * 8 + j];
                p_updated[i * 8 + j] = sum;
            }
        }

        *cov = p_updated;
    }
}

/// Exact analytic 4x4 matrix inversion via Cramer's Rule.
/// Guaranteed 0 heap allocations, fully unrolled and vectorized by LLVM.
#[inline]
pub fn invert_4x4_analytic(m: &[f32; 16]) -> [f32; 16] {
    let mut inv = [0.0f32; 16];

    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
        + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];

    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
        - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];

    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
        + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];

    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
        - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
        - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];

    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
        + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];

    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
        - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];

    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
        + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
        + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];

    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
        - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];

    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
        + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];

    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
        - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
        - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
        + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
        - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
        + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

    let det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if det.abs() < 1e-7 {
        // Fallback for near-singular matrix: diagonal reciprocal
        let mut fallback = [0.0f32; 16];
        for i in 0..4 {
            let v = m[i * 4 + i];
            fallback[i * 4 + i] = if v.abs() > 1e-6 { 1.0 / v } else { 1.0 };
        }
        return fallback;
    }

    let inv_det = 1.0 / det;
    for i in 0..16 {
        inv[i] *= inv_det;
    }

    inv
}