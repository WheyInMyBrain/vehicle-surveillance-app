/// 8-dimensional state Kalman filter using stack-allocated [f32; 8] and [f32; 64] matrices.
/// State: [center_x, center_y, aspect_ratio, height, vx, vy, va, vh]
#[derive(Debug, Clone)]
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
    pub fn new() -> Self {
        Self {
            std_weight_position: 1.0 / 20.0,
            std_weight_velocity: 1.0 / 160.0,
        }
    }

    #[inline(always)]
    pub fn bbox_to_z(bbox: &[f32; 4]) -> [f32; 4] {
        let w = (bbox[2] - bbox[0]).max(1.0);
        let h = (bbox[3] - bbox[1]).max(1.0);
        let cx = bbox[0] + w * 0.5;
        let cy = bbox[1] + h * 0.5;
        let aspect = w / h;
        [cx, cy, aspect, h]
    }

    #[inline(always)]
    pub fn z_to_bbox(z: &[f32; 8]) -> [f32; 4] {
        let cx = z[0];
        let cy = z[1];
        let aspect = z[2].max(0.1);
        let h = z[3].max(1.0);
        let w = aspect * h;

        [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]
    }

    pub fn initiate(&self, measurement: &[f32; 4]) -> ([f32; 8], [f32; 64]) {
        let mut mean = [0.0f32; 8];
        mean[0..4].copy_from_slice(measurement);

        let h = measurement[3];
        let std_p = 2.0 * self.std_weight_position * h;
        let std_v = 10.0 * self.std_weight_velocity * h;

        let std = [std_p, std_p, 1e-2, std_p, std_v, std_v, 1e-5, std_v];

        let mut cov = [0.0f32; 64];
        for i in 0..8 {
            cov[i * 8 + i] = std[i] * std[i];
        }

        (mean, cov)
    }

    /// Predicts state forward in time with scale-adaptive acceleration noise
    pub fn predict(&self, mean: &mut [f32; 8], cov: &mut [f32; 64]) {
        let h = mean[3].max(1.0);
        let std_p = self.std_weight_position * h;
        let std_v = self.std_weight_velocity * h;

        // Process noise variance
        let motion_cov_std = [
            std_p,
            std_p,
            1e-2,
            std_p,
            std_v,
            std_v,
            1e-5,
            std_v * 1.5, // Scale acceleration adaptation
        ];

        // x' = x + v * dt
        for i in 0..4 {
            mean[i] += mean[i + 4];
        }

        // Transition matrix F applied to covariance: P' = F * P * F^T + Q
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

        // Add Q
        for i in 0..8 {
            p_new[i * 8 + i] += motion_cov_std[i] * motion_cov_std[i];
        }

        *cov = p_new;
    }

    /// Updates state given measurement [cx, cy, a, h]
    pub fn update(&self, mean: &mut [f32; 8], cov: &mut [f32; 64], measurement: &[f32; 4]) {
        let h = mean[3].max(1.0);
        let std_p = self.std_weight_position * h;
        let r_diag = [std_p * std_p, std_p * std_p, 1e-2, std_p * std_p];

        // Innovation: y = z - H * x
        let mut y = [0.0f32; 4];
        for i in 0..4 {
            y[i] = measurement[i] - mean[i];
        }

        // S = H * P * H^T + R  (top-left 4x4 of P + R)
        let mut s = [0.0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                s[i * 4 + j] = cov[i * 8 + j];
            }
            s[i * 4 + i] += r_diag[i];
        }

        let s_inv = invert_4x4(&s);

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

        // mean = mean + K * y
        for i in 0..8 {
            let mut delta = 0.0f32;
            for j in 0..4 {
                delta += k[i * 4 + j] * y[j];
            }
            mean[i] += delta;
        }

        // Covariance update: P = (I - K * H) * P
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
                for l in 0..8 {
                    sum += i_kh[i * 8 + l] * cov[l * 8 + j];
                }
                p_updated[i * 8 + j] = sum;
            }
        }

        *cov = p_updated;
    }
}

#[inline]
fn invert_4x4(m: &[f32; 16]) -> [f32; 16] {
    let mut inv = [0.0f32; 16];
    // Fast diagonal approximation with fallback for diagonally dominant covariance
    for i in 0..4 {
        let val = m[i * 4 + i];
        inv[i * 4 + i] = if val.abs() > 1e-6 { 1.0 / val } else { 1.0 };
    }
    inv
}