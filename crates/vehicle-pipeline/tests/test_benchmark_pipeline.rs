use image::{DynamicImage, RgbImage};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use vehicle_pipeline::{SurveillancePipelineConfig, VehiclePipeline};

/// Cross-platform Resident Set Size (RSS) probe in Megabytes.
/// Uses getrusage(RUSAGE_SELF) on Darwin/Linux - memory-safe and zero-crash.
fn get_current_rss_mb() -> f64 {
    #[cfg(target_os = "macos")]
    {
        use std::mem::MaybeUninit;
        unsafe {
            let mut usage = MaybeUninit::<libc::rusage>::uninit();
            if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) == 0 {
                let usage = usage.assume_init();
                // On macOS Darwin, ru_maxrss is returned in bytes
                (usage.ru_maxrss as f64) / (1024.0 * 1024.0)
            } else {
                0.0
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(statm) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(rss_pages) = statm.split_whitespace().nth(1) {
                if let Ok(pages) = rss_pages.parse::<f64>() {
                    return (pages * 4096.0) / (1024.0 * 1024.0);
                }
            }
        }
        0.0
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0.0
    }
}

fn get_video_dimensions<P: AsRef<Path>>(video_path: P) -> (u32, u32) {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            video_path.as_ref().to_str().unwrap(),
        ])
        .output()
        .expect("Failed to execute ffprobe.");

    let out_str = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = out_str.trim().split('x').collect();
    if parts.len() == 2 {
        let w = parts[0].parse::<u32>().unwrap_or(1280);
        let h = parts[1].parse::<u32>().unwrap_or(720);
        (w, h)
    } else {
        (1280, 720)
    }
}

#[derive(Default)]
struct AggregatedStats {
    total_frames: usize,
    total_time: Duration,
    detect_time: Duration,
    track_time: Duration,
    quality_time: Duration,
    lpr_time: Duration,
    ocr_time: Duration,
    annot_time: Duration,
    latencies_ms: Vec<f64>,
    initial_rss_mb: f64,
    peak_rss_mb: f64,
}

impl AggregatedStats {
    fn record(&mut self, timings: &vehicle_pipeline::types::PipelineStageTimings) {
        self.total_frames += 1;
        self.total_time += timings.total_frame_time;
        self.detect_time += timings.vehicle_detect_time;
        self.track_time += timings.tracking_time;
        self.quality_time += timings.quality_triage_time;
        self.lpr_time += timings.plate_detect_time;
        self.ocr_time += timings.plate_ocr_time;
        self.annot_time += timings.annotation_time;
        self.latencies_ms.push(timings.total_frame_time.as_secs_f64() * 1000.0);

        // Sample RSS periodically to avoid syscall overhead on every frame
        if self.total_frames % 60 == 0 {
            let current_rss = get_current_rss_mb();
            if current_rss > self.peak_rss_mb {
                self.peak_rss_mb = current_rss;
            }
        }
    }

    fn print_summary(&mut self, test_title: &str) {
        if self.total_frames == 0 {
            return;
        }

        self.latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = self.latencies_ms.len();
        let avg_latency = self.latencies_ms.iter().sum::<f64>() / n as f64;
        let p50 = self.latencies_ms[n / 2];
        let p95 = self.latencies_ms[(n as f64 * 0.95) as usize];
        let p99 = self.latencies_ms[(n as f64 * 0.99).min((n - 1) as f64) as usize];
        let max = self.latencies_ms[n - 1];
        let fps = n as f64 / self.total_time.as_secs_f64().max(0.001);

        println!("\n================================================================================");
        println!(" BENCHMARK: {}", test_title);
        println!("================================================================================");
        println!(" Frames Processed : {}", n);
        println!(" Effective Speed  : {:>6.1} FPS", fps);
        println!(" Latency P50      : {:>6.2} ms", p50);
        println!(" Latency P95      : {:>6.2} ms", p95);
        println!(" Latency P99      : {:>6.2} ms", p99);
        println!(" Latency Max      : {:>6.2} ms", max);
        println!(" Latency Mean     : {:>6.2} ms", avg_latency);
        println!(" Memory Baseline  : {:>6.2} MB", self.initial_rss_mb);
        println!(
            " Peak RSS Memory  : {:>6.2} MB (Growth: +{:.2} MB)",
            self.peak_rss_mb,
            (self.peak_rss_mb - self.initial_rss_mb).max(0.0)
        );
        println!("--------------------------------------------------------------------------------");
        println!(" STAGE LATENCY BREAKDOWN (Average per Frame):");
        let d_ms = (self.detect_time.as_secs_f64() * 1000.0) / n as f64;
        let t_ms = (self.track_time.as_secs_f64() * 1000.0) / n as f64;
        let q_ms = (self.quality_time.as_secs_f64() * 1000.0) / n as f64;
        let l_ms = (self.lpr_time.as_secs_f64() * 1000.0) / n as f64;
        let o_ms = (self.ocr_time.as_secs_f64() * 1000.0) / n as f64;
        let a_ms = (self.annot_time.as_secs_f64() * 1000.0) / n as f64;

        println!("  - Vehicle Detector (YOLO26) : {:>5.2} ms ({:>4.1}%)", d_ms, (d_ms / avg_latency) * 100.0);
        println!("  - ByteTrack (Association)   : {:>5.2} ms ({:>4.1}%)", t_ms, (t_ms / avg_latency) * 100.0);
        println!("  - Quality Triage (Laplacian): {:>5.2} ms ({:>4.1}%)", q_ms, (q_ms / avg_latency) * 100.0);
        println!("  - Plate Detector (YOLOv11)  : {:>5.2} ms ({:>4.1}%)", l_ms, (l_ms / avg_latency) * 100.0);
        println!("  - Plate OCR (CCT Global)    : {:>5.2} ms ({:>4.1}%)", o_ms, (o_ms / avg_latency) * 100.0);
        println!("  - Image Annotation Drawing  : {:>5.2} ms ({:>4.1}%)", a_ms, (a_ms / avg_latency) * 100.0);
        println!("================================================================================\n");
    }
}

#[test]
fn test_real_world_streaming_benchmark() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();

    let data_tests = root.join("data/tests");
    let vehicle_model = root.join("data/models/yolo26n.onnx");
    let vehicle_model = if vehicle_model.exists() {
        vehicle_model
    } else {
        root.join("data/models/yolo26s.onnx")
    };
    let plate_model = root.join("data/models/license-plate-finetune-v1n.onnx");
    let ocr_model = root.join("data/models/cct_s_v2_global.onnx");

    assert!(vehicle_model.exists(), "Missing vehicle detector model");
    assert!(plate_model.exists(), "Missing license plate detector model");
    assert!(ocr_model.exists(), "Missing OCR model");

    let config = SurveillancePipelineConfig::new(&vehicle_model, &plate_model, &ocr_model);
    let mut pipeline = VehiclePipeline::new(config).expect("Failed to initialize pipeline");

    let video_files = ["1.mp4", "2.mp4", "3.mp4"];

    for video_name in &video_files {
        let video_path = data_tests.join(video_name);
        if !video_path.exists() {
            continue;
        }

        pipeline.reset_session();

        let (w, h) = get_video_dimensions(&video_path);
        let frame_size = (w * h * 3) as usize;

        let mut decoder = Command::new("ffmpeg")
            .args([
                "-i", video_path.to_str().unwrap(),
                "-f", "image2pipe",
                "-pix_fmt", "rgb24",
                "-vcodec", "rawvideo",
                "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("Failed to spawn ffmpeg decoder");

        let mut stdout = decoder.stdout.take().expect("Failed to get stdout");
        let mut frame_buf = vec![0u8; frame_size];
        let mut stats = AggregatedStats::default();
        stats.initial_rss_mb = get_current_rss_mb();
        stats.peak_rss_mb = stats.initial_rss_mb;

        println!("Streaming & Benchmarking {} ({}x{}) frame-by-frame...", video_name, w, h);

        while stdout.read_exact(&mut frame_buf).is_ok() {
            let rgb_img = match RgbImage::from_raw(w, h, std::mem::take(&mut frame_buf)) {
                Some(im) => im,
                None => break,
            };
            let dyn_img = DynamicImage::ImageRgb8(rgb_img);

            let (result, _) = pipeline.process_frame(&dyn_img).expect("Frame failed");
            stats.record(&result.timings);

            if let DynamicImage::ImageRgb8(im) = dyn_img {
                frame_buf = im.into_raw();
            }
        }

        let final_rss = get_current_rss_mb();
        if final_rss > stats.peak_rss_mb {
            stats.peak_rss_mb = final_rss;
        }

        let _ = decoder.wait();
        stats.print_summary(&format!("Streaming Video Frame-by-Frame [{}]", video_name));
    }
}