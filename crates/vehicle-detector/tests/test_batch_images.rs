use ab_glyph::FontArc;
use image::{open, Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use std::fs::{create_dir_all, read, read_dir};
use std::path::PathBuf;
use std::time::Instant;
use vehicle_detector::{VehicleClass, VehicleDetector, VehicleDetectorConfig};

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
fn test_detect_mumbai_test_images() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();

    let test_dir = root.join("data/tests");
    let out_dir = root.join("data/tests/vehicle_detections");
    let crops_dir = out_dir.join("crops");

    create_dir_all(&crops_dir).expect("Failed to create detection output directories");

    let model_path = root.join("data/models/yolo26n.onnx");
    let model_path = if model_path.exists() {
        model_path
    } else {
        root.join("data/models/yolo26s.onnx")
    };

    assert!(model_path.exists(), "Model not found at {:?}", model_path);

    let config = VehicleDetectorConfig {
        confidence_threshold: 0.30,
        nms_iou_threshold: 0.45,
        input_width: 640,
        input_height: 640,
        min_box_dimension: 30.0,
        num_threads: 4,
    };

    let mut detector = VehicleDetector::new(&model_path, config)
        .expect("Failed to initialize ONNX vehicle detector");

    let mut image_files: Vec<PathBuf> = read_dir(&test_dir)
        .expect("Cannot read test directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            if let Some(ext) = p.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                ext_str == "jpg" || ext_str == "jpeg" || ext_str == "png"
            } else {
                false
            }
        })
        .collect();

    image_files.sort();
    assert!(!image_files.is_empty(), "No test images found in {:?}", test_dir);

    let font = get_available_font();
    let mut total_vehicles = 0;
    let mut total_infer_time = std::time::Duration::ZERO;

    for img_path in &image_files {
        let file_name = img_path.file_name().unwrap().to_string_lossy();
        let stem = img_path.file_stem().unwrap().to_string_lossy();

        let raw_img = match open(img_path) {
            Ok(im) => im,
            Err(e) => {
                println!(" [!] Skipping {}: {}", file_name, e);
                continue;
            }
        };

        let t_infer = Instant::now();
        let detections = detector.detect(&raw_img).expect("Inference error");
        let elapsed = t_infer.elapsed();

        total_infer_time += elapsed;
        total_vehicles += detections.len();

        println!(
            "Scene: {:<12} | Inferred in {:>7.2?} | Vehicles: {}",
            file_name,
            elapsed,
            detections.len()
        );

        let mut annotated: RgbImage = raw_img.to_rgb8();

        for (i, d) in detections.iter().enumerate() {
            println!(
                "  - Box #{}: [{}] Conf: {:.1}% | BBox: [{:.0}, {:.0}, {:.0}, {:.0}]",
                i + 1,
                d.class.as_str(),
                d.confidence * 100.0,
                d.x1(),
                d.y1(),
                d.x2(),
                d.y2()
            );

            let x = d.x1().max(0.0) as i32;
            let y = d.y1().max(0.0) as i32;
            let w = d.width().max(1.0) as u32;
            let h = d.height().max(1.0) as u32;

            let color = match d.class {
                VehicleClass::Car => Rgb([0, 220, 50]),
                VehicleClass::Truck => Rgb([0, 140, 255]),
                VehicleClass::Bus => Rgb([255, 160, 0]),
                VehicleClass::Motorcycle => Rgb([220, 30, 220]),
            };

            for thickness in 0..3 {
                let rect = Rect::at(x - thickness, y - thickness)
                    .of_size(w + (thickness as u32 * 2), h + (thickness as u32 * 2));
                draw_hollow_rect_mut(&mut annotated, rect, color);
            }

            if let Some(ref f) = font {
                let label = format!("{}: {:.0}%", d.class.as_str(), d.confidence * 100.0);
                draw_text_mut(&mut annotated, color, (x + 4).max(0), (y - 18).max(0), 18.0, f, &label);
            }

            // Export vehicle crop for downstream crates
            let crop = d.crop_from(&raw_img);
            let crop_path = crops_dir.join(format!("{}_crop_{}_{}.jpg", stem, i + 1, d.class.as_str()));
            let _ = crop.save(&crop_path);
        }

        let out_img_path = out_dir.join(format!("{}_annotated.jpg", stem));
        annotated.save(&out_img_path).expect("Failed to save annotated image");
    }

    println!("\nProcessed scenes in {:.2?}", total_infer_time);
    println!("Total Valid Vehicles Detected: {}", total_vehicles);
    assert!(total_vehicles > 0, "No vehicles detected across test images");
}