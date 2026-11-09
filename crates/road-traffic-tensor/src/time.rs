use chrono::{DateTime, Datelike, Timelike, Utc};
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
        let fraction = bin_float - (bin as f32);

        Self { day, bin, fraction }
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