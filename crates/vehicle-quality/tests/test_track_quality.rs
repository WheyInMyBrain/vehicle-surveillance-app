use image::{DynamicImage, RgbImage};
use std::collections::HashSet;
use std::fs::create_dir_all;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;
use vehicle_detector::{VehicleDetector, VehicleDetectorConfig};
use vehicle_quality::{QualityAuditor, QualityConfig, TrackQualityManager};
use vehicle_tracker::{ByteTrackConfig, ByteTracker};

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
        .expect("Failed to execute ffprobe. Ensure ffmpeg and ffprobe are in PATH.");

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
fn test_video_golden_crop_selection() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();

    let data_tests = root.join("data/tests");
    let out_base_dir = data_tests.join("vehicle_golden_crops");

    let model_path = root.join("data/models/yolo26n.onnx");
    let model_path = if model_path.exists() {
        model_path
    } else {
        root.join("data/models/yolo26s.onnx")
    };
    assert!(model_path.exists(), "Detector model not found at {:?}", model_path);

    let video_files = ["1.mp4", "2.mp4", "3.mp4"];

    println!("\n==================================================");
    println!(" FULL-PIPELINE QUALITY & GOLDEN-CROP EXTRACTION ");
    println!("==================================================");

    let det_config = VehicleDetectorConfig {
        confidence_threshold: 0.25,
        nms_iou_threshold: 0.45,
        input_width: 640,
        input_height: 640,
        min_box_dimension: 20.0,
        num_threads: 4,
    };
    let mut detector = VehicleDetector::new(&model_path, det_config)
        .expect("Failed to initialize VehicleDetector");

    for video_name in &video_files {
        let video_path = data_tests.join(video_name);
        if !video_path.exists() {
            println!(" [!] Video not found at {:?}, skipping...", video_path);
            continue;
        }

        let (native_w, native_h) = get_video_dimensions(&video_path);
        let frame_size = (native_w * native_h * 3) as usize;

        let stem = video_path.file_stem().unwrap().to_string_lossy();
        let crops_out_dir = out_base_dir.join(&*stem);
        create_dir_all(&crops_out_dir).expect("Failed to create golden crop dir");

        println!(
            "\nProcessing: {} (Native Resolution: {}x{})",
            video_name, native_w, native_h
        );

        let track_config = ByteTrackConfig {
            track_thresh: 0.25,
            high_thresh: 0.45,
            match_thresh: 0.75,
            max_time_lost: 40,
            min_hits: 2,
            ..Default::default()
        };
        let mut tracker = ByteTracker::new(track_config);

        let quality_config = QualityConfig {
            min_sharpness: 25.0,
            min_resolution: 90,
            brightness_range: (30.0, 230.0),
            min_contrast: 0.10,
        };
        let auditor = QualityAuditor::new(quality_config);
        let mut quality_manager = TrackQualityManager::new(auditor, 3, 12.0);

        let mut decoder_child = Command::new("ffmpeg")
            .args([
                "-i",
                video_path.to_str().unwrap(),
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

        let mut decoder_stdout = decoder_child.stdout.take().expect("Failed to get stdout");
        let mut frame_buf = vec![0u8; frame_size];
        let mut frame_idx = 0usize;
        let mut unique_tracks = HashSet::new();

        let t_start = Instant::now();

        while decoder_stdout.read_exact(&mut frame_buf).is_ok() {
            frame_idx += 1;

            let rgb_img = RgbImage::from_raw(native_w, native_h, frame_buf.clone())
                .expect("Failed to build image");
            let dyn_img = DynamicImage::ImageRgb8(rgb_img);

            let detections = detector.detect(&dyn_img).expect("Inference error");
            let tracks = tracker.update(&detections);

            for trk in &tracks {
                unique_tracks.insert(trk.track_id);
                quality_manager.process_frame(frame_idx, trk, &dyn_img);
            }
        }

        let _ = decoder_child.wait();
        let elapsed = t_start.elapsed();

        println!(
            "Done {}: {} frames in {:.2?} ({:.1} FPS)",
            video_name,
            frame_idx,
            elapsed,
            frame_idx as f64 / elapsed.as_secs_f64()
        );

        let mut exported_count = 0;
        for track_id in &unique_tracks {
            let top_crops = quality_manager.get_top_k_crops(*track_id);
            for (rank, golden) in top_crops.iter().enumerate() {
                let file_name = format!(
                    "track_{:04}_rank{}_frame_{:05}_score_{:.1}.jpg",
                    golden.track_id,
                    rank + 1,
                    golden.frame_idx,
                    golden.metrics.composite_score
                );
                let _ = golden.crop.save(crops_out_dir.join(file_name));
                exported_count += 1;
            }

            if let Some(best) = top_crops.first() {
                println!(
                    "  -> Track #{:<3} [{}]: Score: {:>5.1} | Res: {:>3}x{:<3} | Sharpness: {:>6.1} | Frame #{}",
                    best.track_id,
                    best.class.as_str(),
                    best.metrics.composite_score,
                    best.metrics.resolution[0],
                    best.metrics.resolution[1],
                    best.metrics.sharpness,
                    best.frame_idx
                );
            }
        }

        println!("Saved {} high-res golden crops to: {:?}", exported_count, crops_out_dir);
    }
}