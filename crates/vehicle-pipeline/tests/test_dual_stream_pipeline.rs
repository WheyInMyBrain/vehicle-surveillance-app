use image::{DynamicImage, RgbImage};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use vehicle_pipeline::{SurveillancePipelineConfig, VehiclePipeline};

fn get_current_rss_mb() -> f64 {
    #[cfg(target_os = "macos")]
    {
        use std::mem::MaybeUninit;
        unsafe {
            let mut usage = MaybeUninit::<libc::rusage>::uninit();
            if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) == 0 {
                let usage = usage.assume_init();
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
struct DualStreamStats {
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

impl DualStreamStats {
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
        println!(" DUAL-STREAM TELEMETRY: {}", test_title);
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

        println!("  - Proxy Vehicle Detector (YOLO26): {:>5.2} ms ({:>4.1}%)", d_ms, (d_ms / avg_latency) * 100.0);
        println!("  - ByteTrack (Association)         : {:>5.2} ms ({:>4.1}%)", t_ms, (t_ms / avg_latency) * 100.0);
        println!("  - High-Res Quality Triage (Laplace): {:>5.2} ms ({:>4.1}%)", q_ms, (q_ms / avg_latency) * 100.0);
        println!("  - 4K Plate Detector (YOLOv11)     : {:>5.2} ms ({:>4.1}%)", l_ms, (l_ms / avg_latency) * 100.0);
        println!("  - 4K Plate OCR (CCT Global)       : {:>5.2} ms ({:>4.1}%)", o_ms, (o_ms / avg_latency) * 100.0);
        println!("  - Image Annotation Drawing        : {:>5.2} ms ({:>4.1}%)", a_ms, (a_ms / avg_latency) * 100.0);
        println!("================================================================================\n");
    }
}

#[test]
fn test_simulated_dual_stream_pipeline() {
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

    let video_path = data_tests.join("3.mp4");
    if !video_path.exists() {
        println!("Test video 3.mp4 not found, skipping dual-stream test.");
        return;
    }

    let (native_w, native_h) = get_video_dimensions(&video_path);
    let proxy_w = 640u32;
    let proxy_h = 360u32;

    let config = SurveillancePipelineConfig::new(&vehicle_model, &plate_model, &ocr_model);
    let mut pipeline = VehiclePipeline::new(config).expect("Failed to initialize pipeline");

    let high_res_frame_bytes = (native_w * native_h * 3) as usize;
    let proxy_frame_bytes = (proxy_w * proxy_h * 3) as usize;

    // Decoder 1: Native High-Res
    let mut high_decoder = Command::new("ffmpeg")
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
        .expect("Failed to spawn native decoder");

    // Decoder 2: Hardware-simulated sub-sampled proxy stream
    let mut proxy_decoder = Command::new("ffmpeg")
        .args([
            "-i", video_path.to_str().unwrap(),
            "-vf", &format!("scale={}:{}", proxy_w, proxy_h),
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo",
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to spawn proxy decoder");

    let mut high_stdout = high_decoder.stdout.take().expect("Failed to get high-res stdout");
    let mut proxy_stdout = proxy_decoder.stdout.take().expect("Failed to get proxy stdout");

    let mut high_buf = vec![0u8; high_res_frame_bytes];
    let mut proxy_buf = vec![0u8; proxy_frame_bytes];

    let mut stats = DualStreamStats::default();
    stats.initial_rss_mb = get_current_rss_mb();
    stats.peak_rss_mb = stats.initial_rss_mb;

    println!("\nStreaming Dual Feeds concurrently (Proxy: 640x360 | Inspection: {}x{})...", native_w, native_h);

    while high_stdout.read_exact(&mut high_buf).is_ok()
        && proxy_stdout.read_exact(&mut proxy_buf).is_ok()
    {
        let high_rgb = match RgbImage::from_raw(native_w, native_h, std::mem::take(&mut high_buf)) {
            Some(im) => im,
            None => break,
        };
        let proxy_rgb = match RgbImage::from_raw(proxy_w, proxy_h, std::mem::take(&mut proxy_buf)) {
            Some(im) => im,
            None => break,
        };

        let high_img = DynamicImage::ImageRgb8(high_rgb);
        let proxy_img = DynamicImage::ImageRgb8(proxy_rgb);

        let (result, _) = pipeline
            .process_frame_dual(&proxy_img, Some(&high_img))
            .expect("Dual-stream step failed");

        stats.record(&result.timings);

        if let DynamicImage::ImageRgb8(im) = high_img {
            high_buf = im.into_raw();
        }
        if let DynamicImage::ImageRgb8(im) = proxy_img {
            proxy_buf = im.into_raw();
        }
    }

    let final_rss = get_current_rss_mb();
    if final_rss > stats.peak_rss_mb {
        stats.peak_rss_mb = final_rss;
    }

    let _ = high_decoder.wait();
    let _ = proxy_decoder.wait();

    stats.print_summary("Simulated Dual-Stream 640p Proxy -> 1080p Crop");
}