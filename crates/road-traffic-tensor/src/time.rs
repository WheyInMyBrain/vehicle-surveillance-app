use chrono::{DateTime, Datelike, Duration, Timelike, Utc};
use road_graph_loader::{DAYS_PER_WEEK, TIME_BINS_PER_DAY};

/// Canonical temporal coordinate inside the weekly traffic profile
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemporalCoord {
    /// 0 = Monday, ..., 6 = Sunday
    pub day: usize,
    /// 0..287 (5-minute bucket)
    pub bin: usize,
    /// Fractional progress into the 5-minute interval [0.0, 1.0) for smooth lerp
    pub fraction: f32,
}

impl TemporalCoord {
    #[inline]
    pub fn from_datetime(dt: &DateTime<Utc>) -> Self {
        // Monday = 0, Sunday = 6
        let day = (dt.weekday().num_days_from_monday() as usize) % DAYS_PER_WEEK;
        let sec_of_day = dt.hour() * 3600 + dt.minute() * 60 + dt.second();
        
        let total_5min_sec = (sec_of_day % 86400) as f32;
        let bin_float = total_5min_sec / 300.0;
        
        let bin = (bin_float.floor() as usize).min(TIME_BINS_PER_DAY - 1);
        let fraction = (bin_float - (bin as f32)).clamp(0.0, 1.0);

        Self { day, bin, fraction }
    }

    /// Computes temporal coordinate offset by a telemetry latency lag (e.g. 5 minutes back)
    #[inline]
    pub fn from_datetime_with_lag(dt: &DateTime<Utc>, lag: Duration) -> Self {
        let lagged_dt = *dt - lag;
        Self::from_datetime(&lagged_dt)
    }

    /// Retrospectively snaps strictly to the prior closed 5-minute bin boundary
    #[inline]
    pub fn prior_completed_bin(&self) -> (usize, usize) {
        if self.bin > 0 {
            (self.day, self.bin - 1)
        } else {
            // Roll back to previous day's last bin (Sunday -> Saturday, Monday -> Sunday)
            let prev_day = (self.day + DAYS_PER_WEEK - 1) % DAYS_PER_WEEK;
            (prev_day, TIME_BINS_PER_DAY - 1)
        }
    }

    #[inline]
    pub fn next_bin(&self) -> (usize, usize) {
        if self.bin + 1 < TIME_BINS_PER_DAY {
            (self.day, self.bin + 1)
        } else {
            ((self.day + 1) % DAYS_PER_WEEK, 0)
        }
    }
}