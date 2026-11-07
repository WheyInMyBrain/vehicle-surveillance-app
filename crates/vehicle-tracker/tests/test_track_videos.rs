use ab_glyph::FontArc;
use image::{DynamicImage, Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use std::collections::HashSet;
use std::fs::{create_dir_all, read};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;
use vehicle_detector::{VehicleDetector, VehicleDetectorConfig};
use vehicle_tracker::{ByteTrackConfig, ByteTracker};

fn get_available_font() -> Option<FontArc> {
    let font_candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ];

    for path in font_candidates {
        if let Ok(bytes) = read(path) {
            if let Ok(font) = FontArc::try_from_vec(bytes) {
                return Some(font);
            }
        }
    }
    None
}

fn track_id_to_color(track_id: u64) -> Rgb<u8> {
    let golden_ratio = 0.618_033_988_749_895;
    let hue = ((track_id as f64 * golden_ratio) % 1.0) as f32 * 360.0;
    let c = 0.95 * 0.85;
    let x = c * (1.0 - (((hue / 60.0) % 2.0) - 1.0).abs());
    let m = 0.95 - c;

    let (r, g, b) = match (hue as u32) / 60 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    Rgb([
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ])
}

#[test]
fn test_track_videos() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();

    let data_tests = root.join("data/tests");
    let out_base_dir = data_tests.join("vehicle_tracks");

    let model_path = root.join("data/models/yolo26n.onnx");
    let model_path = if model_path.exists() {
        model_path
    } else {
        root.join("data/models/yolo26s.onnx")
    };

    assert!(model_path.exists(), "Model not found at {:?}", model_path);

    let target_width = 640u32;
    let target_height = 360u32;
    let frame_size = (target_width * target_height * 3) as usize;

    let video_files = ["1.mp4", "2.mp4", "3.mp4"];
    let font = get_available_font();

    println!("\n==================================================");
    println!(" HIGH-SPEED BYTE-TRACK VIDEO BENCHMARK ");
    println!(" Processing Resolution: {}x{} (Fast Testing)", target_width, target_height);
    println!("==================================================");

    // 1. Lower the detector threshold slightly for video streams
    let det_config = VehicleDetectorConfig {
        confidence_threshold: 0.18,  // Lowered from 0.22 so approaching vehicles get detected early
        nms_iou_threshold: 0.50,
        input_width: 640,
        input_height: 640,
        min_box_dimension: 6.0,      // Allow small/distant detections on 360p stream
        num_threads: 4,
    };
    let mut detector = VehicleDetector::new(&model_path, det_config)
        .expect("Failed to initialize VehicleDetector");

    // 2. Align ByteTrack thresholds with the fine-tuned model's confidence distribution
    let base_track_config = ByteTrackConfig {
        track_thresh: 0.18,          // Minimum confidence to be considered for tracking
        high_thresh: 0.30,           // Lowered from 0.45 so 0.34 confidence detections CAN spawn tracks
        match_thresh: 0.65,          // Slightly relaxed IoU match threshold for fast-moving targets
        max_time_lost: 45,           // Tolerate temporary occlusions
        min_hits: 2,                 // Confirm after 2 consecutive frames
        class_mismatch_penalty: 0.10, // Relaxed from 0.50 so class flicker doesn't drop the track
        motion_weight: 0.10,
        ..Default::default()
    };

    for video_name in &video_files {
        let video_path = data_tests.join(video_name);
        if !video_path.exists() {
            println!(" [!] Video not found at {:?}, skipping...", video_path);
            continue;
        }

        let stem = video_path.file_stem().unwrap().to_string_lossy();
        let video_out_dir = out_base_dir.join(&*stem);
        create_dir_all(&video_out_dir).expect("Failed to create track output directory");

        let out_video_path = video_out_dir.join(format!("{}_tracked.mp4", stem));

        let mut tracker = ByteTracker::new(base_track_config.clone());

        let mut decoder_child = Command::new("ffmpeg")
            .args([
                "-i",
                video_path.to_str().unwrap(),
                "-vf",
                &format!("scale={}:{}", target_width, target_height),
                "-f",
                "image2pipe",
                "-pix_fmt",
                "rgb24",
                "-vcodec",
                "rawvideo",
                "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("Failed to spawn ffmpeg decoder");

        let mut encoder_child = Command::new("ffmpeg")
            .args([
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                &format!("{}x{}", target_width, target_height),
                "-r",
                "25",
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "ultrafast",
                out_video_path.to_str().unwrap(),
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("Failed to spawn ffmpeg encoder");

        let mut decoder_stdout = decoder_child.stdout.take().expect("Failed to get stdout");
        let mut encoder_stdin = encoder_child.stdin.take().expect("Failed to get stdin");

        let mut frame_buf = vec![0u8; frame_size];
        let mut frame_idx = 0usize;
        let mut unique_tracks = HashSet::new();

        let t_start = Instant::now();

        while decoder_stdout.read_exact(&mut frame_buf).is_ok() {
            frame_idx += 1;

            let rgb_img = RgbImage::from_raw(target_width, target_height, frame_buf.clone())
                .expect("Failed to build image");
            let dyn_img = DynamicImage::ImageRgb8(rgb_img);

            let detections = detector.detect(&dyn_img).expect("Inference error");
            let tracks = tracker.update(&detections);

            for trk in &tracks {
                unique_tracks.insert(trk.track_id);
            }

            let mut annotated = dyn_img.to_rgb8();

            for trk in &tracks {
                let color = track_id_to_color(trk.track_id);
                let x = trk.x1().max(0.0) as i32;
                let y = trk.y1().max(0.0) as i32;
                let w = trk.width().max(1.0) as u32;
                let h = trk.height().max(1.0) as u32;

                let rect = Rect::at(x, y).of_size(w, h);
                draw_hollow_rect_mut(&mut annotated, rect, color);

                if let Some(ref f) = font {
                    let label = format!("#{} {}", trk.track_id, trk.class.as_str());
                    draw_text_mut(&mut annotated, color, (x + 2).max(0), (y - 14).max(0), 14.0, f, &label);
                }
            }

            encoder_stdin
                .write_all(annotated.as_raw())
                .expect("Failed to write to encoder");

            if frame_idx % 50 == 0 {
                print!("\r  [{}] Encoded frames: {} | Active tracks: {}", video_name, frame_idx, tracks.len());
                std::io::stdout().flush().unwrap();
            }
        }

        drop(encoder_stdin);
        let _ = decoder_child.wait();
        let _ = encoder_child.wait();

        let total_time = t_start.elapsed();
        let fps = frame_idx as f64 / total_time.as_secs_f64().max(0.001);

        println!(
            "\n  Done {}: {} frames in {:.2?} ({:.1} FPS)",
            video_name, frame_idx, total_time, fps
        );
    }
}