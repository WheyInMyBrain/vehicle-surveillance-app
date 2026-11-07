use image::open;
use std::fs::create_dir_all;
use std::path::PathBuf;
use std::time::Instant;
use vehicle_pipeline::{SurveillancePipelineConfig, VehiclePipeline};

#[test]
fn test_static_images_pipeline() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir.parent().unwrap().parent().unwrap();

    let data_tests = root.join("data/tests");
    let out_dir = data_tests.join("pipeline_output/images");
    create_dir_all(&out_dir).expect("Failed to create image output dir");

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

    println!("\n==================================================");
    println!(" RUNNING STATIC IMAGE PIPELINE BENCHMARK ");
    println!("==================================================");

    let config = SurveillancePipelineConfig::new(&vehicle_model, &plate_model, &ocr_model);
    let mut pipeline = VehiclePipeline::new(config).expect("Failed to initialize pipeline");

    let image_names = ["1.jpg", "2.jpg", "3.jpg"];

    for img_name in &image_names {
        let img_path = data_tests.join(img_name);
        if !img_path.exists() {
            println!(" [!] Image {:?} not found, skipping...", img_path);
            continue;
        }

        let dyn_img = open(&img_path).expect("Failed to read image");
        let t0 = Instant::now();

        let (result, annotated) = pipeline.process_image(&dyn_img).expect("Pipeline execution failed");
        let elapsed = t0.elapsed();

        println!(
            "\nProcessed: {} in {:.2?} | Vehicles: {} | Plates Read: {}",
            img_name,
            elapsed,
            result.active_track_count,
            result.plates_read_count
        );

        for v in &result.vehicles {
            let ocr_str = v.plate_ocr.as_ref().map(|o| format!("{} ({:.1}%)", o.as_str(), o.confidence * 100.0)).unwrap_or_else(|| "None".into());
            println!("  -> Vehicle #{:<2} [{}]: Plate: {}", v.track_id, v.class.as_str(), ocr_str);
        }

        let out_path = out_dir.join(format!("{}_pipeline.jpg", PathBuf::from(img_name).file_stem().unwrap().to_string_lossy()));
        annotated.save(&out_path).expect("Failed to save annotated image");
        println!("Saved annotated frame to: {:?}", out_path);
    }
}