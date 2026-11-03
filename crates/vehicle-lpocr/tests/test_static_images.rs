use ab_glyph::FontArc;
use image::{open, Rgb, RgbImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use std::fs::{create_dir_all, read};
use std::path::{Path, PathBuf};
use std::time::Instant;
use vehicle_detector::{VehicleDetector, VehicleDetectorConfig};
use vehicle_lpocr::{PlateOcrConfig, PlateOcrEngine};
use vehicle_lpr::{PlateDetector, PlateDetectorConfig};
use vehicle_quality::{QualityAuditor, QualityConfig};

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

#[test]
fn test_static_car_images_pipeline() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();

    let data_tests = root.join("data/tests");
    let out_dir = data_tests.join("annotated_images");
    create_dir_all(&out_dir).expect("Failed to create annotated output directory");

    let vehicle_model = root.join("data/models/yolo26n.onnx");
    let vehicle_model = if vehicle_model.exists() {
        vehicle_model
    } else {
        root.join("data/models/yolo26s.onnx")
    };
    let plate_model = root.join("data/models/license-plate-finetune-v1n.onnx");
    let ocr_model = root.join("data/models/cct_s_v2_global.onnx");

    assert!(vehicle_model.exists(), "Missing vehicle detector model at {:?}", vehicle_model);
    assert!(plate_model.exists(), "Missing plate detector model at {:?}", plate_model);
    assert!(ocr_model.exists(), "Missing OCR model at {:?}", ocr_model);

    let font = get_available_font();

    println!("\n================================================================================");
    println!(" STATIC IMAGE PIPELINE: DETECT VEHICLE -> QUALITY -> DETECT PLATE -> OCR ");
    println!("================================================================================");

    // 1. Initialize Vehicle Detector
    let det_config = VehicleDetectorConfig {
        confidence_threshold: 0.25,
        nms_iou_threshold: 0.45,
        input_width: 640,
        input_height: 640,
        min_box_dimension: 24.0,
        num_threads: 4,
    };
    let mut vehicle_detector = VehicleDetector::new(&vehicle_model, det_config)
        .expect("Failed to initialize VehicleDetector");

    let shared_env = vehicle_detector.environment();

    // 2. Initialize Plate Detector (Sharing same ORT env)
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

    // 3. Initialize CCT Plate OCR (NHWC format)
    let ocr_config = PlateOcrConfig::default();
    let mut ocr_engine = PlateOcrEngine::new_with_env(&ocr_model, ocr_config, shared_env)
        .expect("Failed to initialize PlateOcrEngine");

    // 4. Initialize Quality Auditor (Metrics inspector)
    let quality_auditor = QualityAuditor::new(QualityConfig::default());

    let image_names = ["1.jpg", "2.jpg", "3.jpg"];

    for img_name in &image_names {
        let img_path = data_tests.join(img_name);
        if !img_path.exists() {
            println!(" [!] Image {:?} not found, skipping...", img_path);
            continue;
        }

        println!("\n--------------------------------------------------------------------------------");
        println!("Processing: {}", img_name);

        let dyn_img = open(&img_path).expect("Failed to open image");
        let (img_w, img_h) = (dyn_img.width(), dyn_img.height());
        println!("Resolution: {}x{}", img_w, img_h);

        let t_total = Instant::now();

        // Step A: Detect Vehicles
        let t0 = Instant::now();
        let vehicles = vehicle_detector.detect(&dyn_img).expect("Vehicle inference failed");
        println!("  1. Vehicle Detector: Found {} vehicles in {:.2?}", vehicles.len(), t0.elapsed());

        let mut annotated: RgbImage = dyn_img.to_rgb8();

        for (v_idx, vehicle) in vehicles.iter().enumerate() {
            let vx1 = vehicle.x1().max(0.0);
            let vy1 = vehicle.y1().max(0.0);
            let vw = vehicle.width();
            let vh = vehicle.height();

            // Extract vehicle crop
            let vehicle_crop = vehicle.crop_from(&dyn_img);

            // Step B: Run Quality Assessment
            let q_metrics = quality_auditor.evaluate(&vehicle_crop, &vehicle.bbox, img_w, img_h);
            println!(
                "     Vehicle #{:<2} [{}]: Res: {:.0}x{:.0} | Sharpness: {:>5.1} | Contrast: {:.2}",
                v_idx + 1,
                vehicle.class.as_str(),
                vw,
                vh,
                q_metrics.sharpness,
                q_metrics.contrast
            );

            // Draw Vehicle Bounding Box (Green)
            let v_rect = Rect::at(vx1 as i32, vy1 as i32).of_size(vw.max(1.0) as u32, vh.max(1.0) as u32);
            draw_hollow_rect_mut(&mut annotated, v_rect, Rgb([0, 255, 100]));
            draw_hollow_rect_mut(&mut annotated, Rect::at((vx1 + 1.0) as i32, (vy1 + 1.0) as i32).of_size(vw.max(2.0) as u32 - 1, vh.max(2.0) as u32 - 1), Rgb([0, 255, 100]));

            if let Some(ref f) = font {
                let v_label = format!("{}: {:.0}%", vehicle.class.as_str(), vehicle.confidence * 100.0);
                draw_text_mut(&mut annotated, Rgb([0, 255, 100]), (vx1 as i32 + 4).max(0), (vy1 as i32 - 16).max(0), 16.0, f, &v_label);
            }

            // Step C: Plate Detection inside the vehicle crop
            let t_plate = Instant::now();
            let plates = plate_detector.detect(&vehicle_crop).expect("Plate detection failed");

            for (p_idx, plate) in plates.iter().enumerate() {
                // Step D: Run OCR on the plate crop
                let plate_crop = plate.crop_from(&vehicle_crop);
                let ocr_result = ocr_engine.recognize(&plate_crop).expect("OCR failed");

                // Translate local vehicle crop coordinates back to full image space
                let gx1 = vx1 + plate.x1();
                let gy1 = vy1 + plate.y1();
                let pw = plate.width();
                let ph = plate.height();

                println!(
                    "        Plate #{:<2}: Plate Conf: {:>4.1}% | Text: {:<10} | OCR Conf: {:>4.1}% (Found in {:.2?})",
                    p_idx + 1,
                    plate.confidence * 100.0,
                    if ocr_result.text.is_empty() { "<unreadable>" } else { &ocr_result.text },
                    ocr_result.confidence * 100.0,
                    t_plate.elapsed()
                );

                // Draw License Plate Bounding Box (Bright Yellow, thick)
                let p_rect = Rect::at(gx1 as i32, gy1 as i32).of_size(pw.max(1.0) as u32, ph.max(1.0) as u32);
                draw_hollow_rect_mut(&mut annotated, p_rect, Rgb([255, 235, 0]));
                draw_hollow_rect_mut(&mut annotated, Rect::at((gx1 + 1.0) as i32, (gy1 + 1.0) as i32).of_size(pw.max(2.0) as u32 - 1, ph.max(2.0) as u32 - 1), Rgb([255, 235, 0]));

                // Draw Text Banner with Black Background for high legibility
                if let Some(ref f) = font {
                    let display_text = if !ocr_result.text.is_empty() {
                        format!("{} ({:.0}%)", ocr_result.text, ocr_result.confidence * 100.0)
                    } else {
                        format!("Plate: {:.0}%", plate.confidence * 100.0)
                    };

                    let text_x = (gx1 as i32).max(2);
                    let text_y = (gy1 as i32 - 20).max(2);

                    // Background box behind plate text
                    let banner_w = (display_text.len() as u32 * 10) + 8;
                    draw_filled_rect_mut(&mut annotated, Rect::at(text_x, text_y).of_size(banner_w, 18), Rgb([0, 0, 0]));
                    draw_text_mut(&mut annotated, Rgb([255, 235, 0]), text_x + 4, text_y + 1, 15.0, f, &display_text);
                }
            }
        }

        let out_file = out_dir.join(format!("{}_annotated.jpg", Path::new(img_name).file_stem().unwrap().to_string_lossy()));
        annotated.save(&out_file).expect("Failed to save annotated image");

        println!("  -> Completed in {:.2?}. Output saved to: {:?}", t_total.elapsed(), out_file);
    }

    println!("\n================================================================================");
    println!(" All annotated test images exported to: {:?}", out_dir);
    println!("================================================================================\n");
}