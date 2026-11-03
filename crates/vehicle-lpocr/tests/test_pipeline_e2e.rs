use image::{DynamicImage, RgbImage};
use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;
use vehicle_detector::{VehicleDetector, VehicleDetectorConfig};
use vehicle_lpocr::{PlateOcrConfig, PlateOcrEngine};
use vehicle_lpr::{PlateDetector, PlateDetectorConfig};
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
fn test_complete_surveillance_pipeline() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();

    let data_tests = root.join("data/tests");
    let vehicle_model = root.join("data/models/yolo26n.onnx");
    let plate_model = root.join("data/models/license-plate-finetune-v1n.onnx");
    let ocr_model = root.join("data/models/cct_s_v2_global.onnx");

    assert!(vehicle_model.exists(), "Missing yolo26n.onnx");
    assert!(plate_model.exists(), "Missing license-plate-finetune-v1n.onnx");
    assert!(ocr_model.exists(), "Missing cct_s_v2_global.onnx");

    let video_files = ["1.mp4", "2.mp4", "3.mp4"];

    println!("\n================================================================================");
    println!(" FULL SURVEILLANCE PIPELINE: DETECT -> TRACK -> QUALITY -> LPR -> OCR ");
    println!("================================================================================");

    // 1. Initialize Vehicle Detector
    let det_config = VehicleDetectorConfig {
        confidence_threshold: 0.25,
        nms_iou_threshold: 0.45,
        input_width: 640,
        input_height: 640,
        min_box_dimension: 20.0,
        num_threads: 4,
    };
    let mut vehicle_detector = VehicleDetector::new(&vehicle_model, det_config)
        .expect("Failed to initialize VehicleDetector");

    // Share common ONNX environment across all models in the process
    let shared_env = vehicle_detector.environment();

    // 2. Initialize Plate Detector
    let plate_config = PlateDetectorConfig {
        confidence_threshold: 0.20,
        nms_iou_threshold: 0.40,
        input_width: 640,
        input_height: 640,
        min_plate_height: 6.0,
        min_plate_width: 14.0,
        num_threads: 4,
    };
    let mut plate_detector = PlateDetector::new_with_env(&plate_model, plate_config, shared_env.clone())
        .expect("Failed to initialize PlateDetector");

    // 3. Initialize CCT Plate OCR
    let ocr_config = PlateOcrConfig::default();
    let mut ocr_engine = PlateOcrEngine::new_with_env(&ocr_model, ocr_config, shared_env)
        .expect("Failed to initialize PlateOcrEngine");

    for video_name in &video_files {
        let video_path = data_tests.join(video_name);
        if !video_path.exists() {
            println!(" [!] Video {:?} not found, skipping...", video_name);
            continue;
        }

        let (native_w, native_h) = get_video_dimensions(&video_path);
        let frame_size = (native_w * native_h * 3) as usize;

        println!("\n--------------------------------------------------------------------------------");
        println!("Analyzing Stream: {} ({}x{})", video_name, native_w, native_h);

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
        let mut quality_manager = TrackQualityManager::new(auditor, 3, 10.0);

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
                .expect("Failed to construct image");
            let dyn_img = DynamicImage::ImageRgb8(rgb_img);

            let detections = vehicle_detector.detect(&dyn_img).expect("Vehicle detector failed");
            let tracks = tracker.update(&detections);

            for trk in &tracks {
                unique_tracks.insert(trk.track_id);
                quality_manager.process_frame(frame_idx, trk, &dyn_img);
            }
        }

        let _ = decoder_child.wait();
        let stream_elapsed = t_start.elapsed();

        println!(
            "Tracked {} frames in {:.2?} ({:.1} FPS) | Identified {} vehicles",
            frame_idx,
            stream_elapsed,
            frame_idx as f64 / stream_elapsed.as_secs_f64(),
            unique_tracks.len()
        );

        let mut plates_read = 0;

        for track_id in &unique_tracks {
            let golden_crops = quality_manager.get_top_k_crops(*track_id);

            for (_rank, golden) in golden_crops.iter().enumerate() {
                let plates = plate_detector.detect(&golden.crop).expect("Plate detection failed");

                for plate in plates {
                    let plate_crop = plate.crop_from(&golden.crop);
                    let ocr_result = ocr_engine.recognize(&plate_crop).expect("OCR inference failed");

                    if ocr_result.is_valid_plate() {
                        plates_read += 1;
                        println!(
                            "  [Vehicle #{:<3} ({})] Frame #{:<4} | Plate: {:<10} | Confidence: {:>5.1}%",
                            golden.track_id,
                            golden.class.as_str(),
                            golden.frame_idx,
                            ocr_result.text,
                            ocr_result.confidence * 100.0
                        );
                        break;
                    }
                }
            }
        }

        println!("Completed {}: Successfully read {} license plates.", video_name, plates_read);
    }
}