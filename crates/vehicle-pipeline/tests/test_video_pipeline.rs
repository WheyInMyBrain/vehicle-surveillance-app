use image::{DynamicImage, RgbImage};
use std::fs::create_dir_all;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;
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

#[test]
fn test_video_stream_pipeline() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();

    let data_tests = root.join("data/tests");
    let out_dir = data_tests.join("pipeline_output/videos");
    create_dir_all(&out_dir).expect("Failed to create video output dir");

    let vehicle_model = root.join("data/models/yolo26n.onnx");
    let vehicle_model = if vehicle_model.exists() {
        vehicle_model
    } else {
        root.join("data/models/yolo26s.onnx")
    };
    let plate_model = root.join("data/models/license-plate-finetune-v1n.onnx");
    let ocr_model = root.join("data/models/cct_s_v2_global.onnx");

    let video_files = ["1.mp4", "2.mp4", "3.mp4"];

    println!("\n==================================================");
    println!(" RUNNING VIDEO STREAM PIPELINE BENCHMARK ");
    println!("==================================================");

    for video_name in &video_files {
        let video_path = data_tests.join(video_name);
        if !video_path.exists() {
            println!(" [!] Video {:?} not found, skipping...", video_name);
            continue;
        }

        let (native_w, native_h) = get_video_dimensions(&video_path);
        let frame_size = (native_w * native_h * 3) as usize;
        let stem = video_path.file_stem().unwrap().to_string_lossy();
        let out_video = out_dir.join(format!("{}_annotated.mp4", stem));

        println!("\n--------------------------------------------------");
        println!("Streaming: {} ({}x{}) -> {:?}", video_name, native_w, native_h, out_video);

        let config = SurveillancePipelineConfig::new(&vehicle_model, &plate_model, &ocr_model);
        let mut pipeline = VehiclePipeline::new(config).expect("Failed to initialize pipeline");

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

        let mut encoder = Command::new("ffmpeg")
            .args([
                "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", &format!("{}x{}", native_w, native_h),
                "-r", "25",
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                out_video.to_str().unwrap(),
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("Failed to spawn ffmpeg encoder");

        let mut decoder_stdout = decoder.stdout.take().expect("Failed to get stdout");
        let mut encoder_stdin = encoder.stdin.take().expect("Failed to get stdin");

        let mut frame_buf = vec![0u8; frame_size];
        let mut frame_idx = 0usize;
        let t_start = Instant::now();

        while decoder_stdout.read_exact(&mut frame_buf).is_ok() {
            frame_idx += 1;

            let rgb_img = match RgbImage::from_raw(native_w, native_h, std::mem::take(&mut frame_buf)) {
                Some(im) => im,
                None => break,
            };
            let dyn_img = DynamicImage::ImageRgb8(rgb_img);

            let (result, annotated) = pipeline.process_frame(&dyn_img).expect("Frame processing failed");

            encoder_stdin.write_all(annotated.as_raw()).expect("Failed to write to encoder");

            if frame_idx % 50 == 0 {
                print!("\r  [{}] Frames: {:>4} | Active Tracks: {:>2} | Total Plates Read: {:>2}", video_name, frame_idx, result.active_track_count, result.plates_read_count);
                std::io::stdout().flush().unwrap();
            }

            if let DynamicImage::ImageRgb8(im) = dyn_img {
                frame_buf = im.into_raw();
            }
        }

        drop(encoder_stdin);
        let _ = decoder.wait();
        let _ = encoder.wait();

        let elapsed = t_start.elapsed();
        println!(
            "\nDone {}: {} frames in {:.2?} ({:.1} FPS)",
            video_name,
            frame_idx,
            elapsed,
            frame_idx as f64 / elapsed.as_secs_f64()
        );
    }
}