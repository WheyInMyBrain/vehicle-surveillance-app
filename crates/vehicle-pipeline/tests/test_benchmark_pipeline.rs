use image::{DynamicImage, RgbImage};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use vehicle_pipeline::{SurveillancePipelineConfig, VehiclePipeline};

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
        println!(" BENCHMARK REPORT: {}", test_title);
        println!("================================================================================");
        println!(" Frames Processed : {}", n);
        println!(" Effective Speed  : {:>6.1} FPS", fps);
        println!(" Latency P50      : {:>6.2} ms", p50);
        println!(" Latency P95      : {:>6.2} ms", p95);
        println!(" Latency P99      : {:>6.2} ms", p99);
        println!(" Latency Max      : {:>6.2} ms", max);
        println!(" Latency Mean     : {:>6.2} ms", avg_latency);
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
    let plate_model = root.join("data/models/license-plate-finetune-v1n.onnx");
    let ocr_model = root.join("data/models/cct_s_v2_global.onnx");

    assert!(vehicle_model.exists(), "Missing yolo26n.onnx");
    assert!(plate_model.exists(), "Missing license-plate-finetune-v1n.onnx");
    assert!(ocr_model.exists(), "Missing cct_s_v2_global.onnx");

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

        println!("Streaming & Benchmarking {} ({}x{}) frame-by-frame...", video_name, w, h);

        while stdout.read_exact(&mut frame_buf).is_ok() {
            let rgb_img = RgbImage::from_raw(w, h, frame_buf.clone()).unwrap();
            let dyn_img = DynamicImage::ImageRgb8(rgb_img);

            let (result, _) = pipeline.process_frame(&dyn_img).expect("Frame failed");
            stats.record(&result.timings);
        }

        let _ = decoder.wait();
        stats.print_summary(&format!("Streaming Video Frame-by-Frame [{}]", video_name));
    }
}

#[test]
fn test_burst_shutter_benchmark() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();

    let data_tests = root.join("data/tests");
    let vehicle_model = root.join("data/models/yolo26n.onnx");
    let plate_model = root.join("data/models/license-plate-finetune-v1n.onnx");
    let ocr_model = root.join("data/models/cct_s_v2_global.onnx");

    let config = SurveillancePipelineConfig::new(&vehicle_model, &plate_model, &ocr_model);
    let mut pipeline = VehiclePipeline::new(config).expect("Failed to initialize pipeline");

    let video_path = data_tests.join("1.mp4");
    if !video_path.exists() {
        return;
    }

    let (w, h) = get_video_dimensions(&video_path);
    let frame_size = (w * h * 3) as usize;

    // Simulate 5 bursts of 8 rapid shutter stills each
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
        .expect("Failed to spawn ffmpeg");

    let mut stdout = decoder.stdout.take().unwrap();
    let mut frame_buf = vec![0u8; frame_size];
    let mut burst_queue = Vec::with_capacity(8);
    let mut stats = AggregatedStats::default();

    println!("\n==================================================");
    println!(" SIMULATING RAPID-SHUTTER MULTI-STILL BURSTS ");
    println!("==================================================");

    let mut bursts_processed = 0;

    while stdout.read_exact(&mut frame_buf).is_ok() {
        let rgb_img = RgbImage::from_raw(w, h, frame_buf.clone()).unwrap();
        burst_queue.push(DynamicImage::ImageRgb8(rgb_img));

        if burst_queue.len() == 8 {
            pipeline.reset_session();
            let t_burst_start = Instant::now();

            let burst_results = pipeline.process_burst(&burst_queue).expect("Burst failed");
            let elapsed = t_burst_start.elapsed();

            for (res, _) in &burst_results {
                stats.record(&res.timings);
            }

            bursts_processed += 1;
            println!(
                "  Burst #{}: Processed 8 stills in {:.2?} ({:.1} FPS burst rate) | Identified {} plates",
                bursts_processed,
                elapsed,
                8.0 / elapsed.as_secs_f64(),
                burst_results.last().unwrap().0.plates_read_count
            );

            burst_queue.clear();
            if bursts_processed >= 5 {
                break;
            }
        }
    }

    let _ = decoder.kill();
    stats.print_summary("Camera Hardware Multi-Still Burst Capture (8-Frame Bursts)");
}