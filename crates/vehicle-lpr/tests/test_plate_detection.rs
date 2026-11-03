use ab_glyph::FontArc;
use image::{DynamicImage, Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use std::collections::HashSet;
use std::fs::{create_dir_all, read};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;
use vehicle_detector::{VehicleDetector, VehicleDetectorConfig};
use vehicle_lpr::{PlateDetector, PlateDetectorConfig};
use vehicle_quality::{QualityAuditor, QualityConfig, TrackQualityManager};
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
fn test_end_to_end_plate_detection() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();

    let data_tests = root.join("data/tests");
    let out_base_dir = data_tests.join("license_plates");

    let vehicle_model = root.join("data/models/yolo26n.onnx");
    let vehicle_model = if vehicle_model.exists() {
        vehicle_model
    } else {
        root.join("data/models/yolo26s.onnx")
    };
    let plate_model = root.join("data/models/license-plate-finetune-v1n.onnx");

    assert!(vehicle_model.exists(), "Vehicle detector not found at {:?}", vehicle_model);
    assert!(plate_model.exists(), "Plate detector not found at {:?}", plate_model);

    let video_files = ["1.mp4", "2.mp4", "3.mp4"];
    let font = get_available_font();

    println!("\n==================================================");
    println!(" END-TO-END PIPELINE: DETECT -> TRACK -> QUALITY -> LPR ");
    println!("==================================================");

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

    let plate_config = PlateDetectorConfig {
        confidence_threshold: 0.20,
        nms_iou_threshold: 0.40,
        input_width: 640,
        input_height: 640,
        min_plate_height: 6.0,
        min_plate_width: 14.0,
        num_threads: 4,
    };

    // Share the exact same ONNX Runtime environment
    let mut plate_detector = PlateDetector::new_with_env(
        &plate_model,
        plate_config,
        vehicle_detector.environment(),
    )
    .expect("Failed to initialize PlateDetector");

    for video_name in &video_files {
        let video_path = data_tests.join(video_name);
        if !video_path.exists() {
            println!(" [!] Video not found at {:?}, skipping...", video_path);
            continue;
        }

        let stem = video_path.file_stem().unwrap().to_string_lossy();
        let video_out_dir = out_base_dir.join(&*stem);
        let plate_crops_dir = video_out_dir.join("crops");
        create_dir_all(&plate_crops_dir).expect("Failed to create output dir");

        let (native_w, native_h) = get_video_dimensions(&video_path);
        let frame_size = (native_w * native_h * 3) as usize;

        println!("\n--------------------------------------------------");
        println!("Processing Stream: {} ({}x{})", video_name, native_w, native_h);

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
            min_resolution: 90, // Require minimum 90px vehicle crop for readable plate
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
                .expect("Failed to build image");
            let dyn_img = DynamicImage::ImageRgb8(rgb_img);

            let detections = vehicle_detector.detect(&dyn_img).expect("Detector error");
            let tracks = tracker.update(&detections);

            for trk in &tracks {
                unique_tracks.insert(trk.track_id);
                quality_manager.process_frame(frame_idx, trk, &dyn_img);
            }
        }

        let _ = decoder_child.wait();
        let elapsed = t_start.elapsed();

        println!(
            "Decoded & tracked {} frames in {:.2?} ({:.1} FPS)",
            frame_idx,
            elapsed,
            frame_idx as f64 / elapsed.as_secs_f64()
        );
        println!("Running Plate Detector on candidate vehicle crops...");

        let mut total_plates = 0;

        for track_id in &unique_tracks {
            let golden_crops = quality_manager.get_top_k_crops(*track_id);
            for (crop_rank, golden) in golden_crops.iter().enumerate() {
                let plates = plate_detector.detect(&golden.crop).expect("LPR inference error");

                if !plates.is_empty() {
                    let mut annotated: RgbImage = golden.crop.to_rgb8();

                    for (p_idx, plate) in plates.iter().enumerate() {
                        total_plates += 1;

                        println!(
                            "  [Track #{:<3} Rank {}] Frame #{}: Plate found (Conf: {:.1}%, Res: {:.0}x{:.0})",
                            golden.track_id,
                            crop_rank + 1,
                            golden.frame_idx,
                            plate.confidence * 100.0,
                            plate.width(),
                            plate.height()
                        );

                        let x = plate.x1().max(0.0) as i32;
                        let y = plate.y1().max(0.0) as i32;
                        let w = plate.width().max(1.0) as u32;
                        let h = plate.height().max(1.0) as u32;

                        let rect = Rect::at(x, y).of_size(w, h);
                        draw_hollow_rect_mut(&mut annotated, rect, Rgb([255, 230, 0]));

                        if let Some(ref f) = font {
                            let label = format!("{:.0}%", plate.confidence * 100.0);
                            draw_text_mut(&mut annotated, Rgb([255, 230, 0]), (x + 2).max(0), (y - 12).max(0), 12.0, f, &label);
                        }

                        let plate_crop = plate.crop_from(&golden.crop);
                        let plate_file = format!(
                            "track_{:04}_rank{}_plate_{}.jpg",
                            golden.track_id, crop_rank + 1, p_idx + 1
                        );
                        let _ = plate_crop.save(plate_crops_dir.join(plate_file));
                    }

                    let annotated_file = format!(
                        "vehicle_track_{:04}_rank{}.jpg",
                        golden.track_id, crop_rank + 1
                    );
                    let _ = annotated.save(video_out_dir.join(annotated_file));
                }
            }
        }

        println!("Finished {}: Detected {} total plates.", video_name, total_plates);
        println!("Plates saved to: {:?}", plate_crops_dir);
    }
}