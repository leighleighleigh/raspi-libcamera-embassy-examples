use std::io::Write;
use std::io::{BufRead, BufReader, BufWriter};
use std::{env, io::Seek};

use embassy_executor::Spawner;
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::channel::Channel;
use embassy_sync::pubsub::PubSubChannel;
use embassy_time::{Duration, Ticker, Timer};

mod model;
use model::{Multiples, YoloV8};

use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_core as candle;
use candle_nn::{Module, VarBuilder};
use candle_transformers::object_detection::{Bbox, KeyPoint, non_maximum_suppression};

// Use V4L2 to get images
use image::DynamicImage;
use opencv::prelude::*;
use opencv::videoio;
use opencv::imgcodecs::imencode;
use opencv::core::Vector;
use opencv::imgcodecs::IMWRITE_JPEG_QUALITY;

use std::iter::Iterator;
use log::{Level, error, info};
use rerun::{MemoryLimit, RecordingStream};

pub static IMAGES_CHANNEL: PubSubChannel<CriticalSectionRawMutex, image::RgbImage, 2, 3, 1> =
    PubSubChannel::new();

// Model architecture from https://github.com/ultralytics/ultralytics/issues/189
// https://github.com/tinygrad/tinygrad/blob/master/examples/yolov8.py
pub type Detections = Vec<Bbox<usize>>;

pub fn report_detect(
    pred: &Tensor,
    confidence_threshold: f32,
    nms_threshold: f32,
) -> Result<Detections> {
    let pred = pred.to_device(&Device::Cpu)?;
    let (pred_size, npreds) = pred.dims2()?;
    let nclasses = pred_size - 4;
    // The bounding boxes grouped by (maximum) class index.
    let mut bboxes: Vec<Vec<Bbox<Vec<KeyPoint>>>> = (0..nclasses).map(|_| vec![]).collect();
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        let pred = Vec::<f32>::try_from(pred.i((.., index))?)?;
        let confidence = *pred[4..].iter().max_by(|x, y| x.total_cmp(y)).unwrap();
        if confidence > confidence_threshold {
            let mut class_index = 0;
            for i in 0..nclasses {
                if pred[4 + i] > pred[4 + class_index] {
                    class_index = i
                }
            }
            if pred[class_index + 4] > 0. {
                let bbox = Bbox {
                    xmin: pred[0] - pred[2] / 2.,
                    ymin: pred[1] - pred[3] / 2.,
                    xmax: pred[0] + pred[2] / 2.,
                    ymax: pred[1] + pred[3] / 2.,
                    confidence,
                    data: vec![],
                };
                bboxes[class_index].push(bbox)
            }
        }
    }

    non_maximum_suppression(&mut bboxes, nms_threshold);

    let dets: Detections = bboxes
        .iter()
        .enumerate()
        .flat_map(|(c, bbx)| {
            bbx.iter().map(move |b| Bbox {
                xmin: b.xmin,
                ymin: b.ymin,
                xmax: b.xmax,
                ymax: b.ymax,
                confidence: b.confidence,
                data: c.clone(),
            })
        })
        .collect();

    Ok(dets)
}

// pub fn load_model() -> anyhow::Result<std::path::PathBuf> {
//     let path = {
//         let api = hf_hub::api::sync::Api::new()?;
//         let api = api.model("lmz/candle-yolo-v8".to_string());
//         let size = 'n';
//         api.get(&format!("yolov8{size}.safetensors"))?
//     };
//     Ok(path)
// }

pub trait Task: Module + Sized {
    fn load(vb: VarBuilder, multiples: Multiples) -> Result<Self>;
    fn report(pred: &Tensor, confidence_threshold: f32, nms_threshold: f32) -> Result<Detections>;
}

impl Task for YoloV8 {
    fn load(vb: VarBuilder, multiples: Multiples) -> Result<Self> {
        // try load normal yolo with 80 classes
        match YoloV8::load(vb.clone(), multiples, /* num_classes=*/ 80) {
            Ok(m) => Ok(m),
            Err(_) => {
                // or try load 2 class, or three class
                return YoloV8::load(vb, multiples, /* num_classes=*/ 2);
            }
        }
    }

    fn report(pred: &Tensor, confidence_threshold: f32, nms_threshold: f32) -> Result<Detections> {
        report_detect(pred, confidence_threshold, nms_threshold)
    }
}

// TODO: pass something which is just an interable image stream,
// sourced from anywhere (files on disk, webcam, libcamera, GPU, etc)
#[embassy_executor::task]
async fn task_camera(rec: RecordingStream) {
    // Initialize the camera
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY).unwrap(); // 0 is the default camera
	let opened = videoio::VideoCapture::is_opened(&cam).unwrap();
	if !opened {
		panic!("Unable to open default camera!");
	}
   
    let tx = IMAGES_CHANNEL.publisher().unwrap();

    // Reduce jpeg quality for streaming
	let encode_params = Vector::from_slice(&[IMWRITE_JPEG_QUALITY, 70]);
	let mut buffer = Mat::default();
	let mut frame = Vector::default();

    loop {
		cam.read(&mut buffer).unwrap();

		if buffer.size().unwrap().width <= 0 {
            continue;
		}

        // Read frame from the camera & encode it
		imencode(".jpg", &buffer, &mut frame, &encode_params).unwrap();

        // // ONLY FOR PIXEL FORMAT MJPEG
        let imgbuf = std::io::Cursor::new(frame.as_slice());
        let buffered_image = image::ImageReader::new(imgbuf)
            .with_guessed_format()
            .unwrap()
            .decode()
            .expect("Decoded image")
            .to_rgb8();

        tx.publish_immediate(buffered_image);

        Timer::after(Duration::from_millis(10)).await;
    }
}

#[embassy_executor::task]
async fn task_log_images(rec: RecordingStream) {
    let mut framenum = 0;
    let mut rx = IMAGES_CHANNEL.subscriber().unwrap();

    loop {
        // wait for an image to be available
        let img_rgb: image::RgbImage = rx.next_message_pure().await;
        let (width, height) = (img_rgb.width(), img_rgb.height());

        rec.set_time_sequence("frame", framenum);
        rec.log(
            "image_rgb",
            &rerun::Image::from_rgb24(img_rgb.clone().into_vec(), [width, height]),
        )
        .unwrap();
        framenum += 1;
        info!("Logged image frame {}", framenum);
    }
}

#[embassy_executor::task]
async fn task_run_yolo(rec: RecordingStream) {
    let mut framenum = 0;
    let mut rx = IMAGES_CHANNEL.subscriber().unwrap();

    let device = Device::Cpu;
    // Create the model and load the weights from the file.
    let multiples = Multiples::n();
    let model: std::path::PathBuf = "best.safetensors".into();
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device).expect("Mapped memory")
    };
    let model = <YoloV8 as Task>::load(vb, multiples).expect("Loaded model");

    loop {
        // run every half second
        Timer::after(Duration::from_millis(500)).await;

        // wait for an image to be available
        let img_rgb: image::RgbImage = rx.next_message_pure().await;

        // Create a DynamicImage from the img_rgb buffer.
        let original_image = DynamicImage::ImageRgb8(img_rgb);

        let (width, height) = {
            let w = original_image.width() as usize;
            let h = original_image.height() as usize;
            if w < h {
                let w = w * 640 / h;
                // Sizes have to be divisible by 32.
                (w / 32 * 32, 640)
            } else {
                let h = h * 640 / w;
                (640, h / 32 * 32)
            }
        };

        let image_t = {
            let img = original_image.resize_exact(
                width as u32,
                height as u32,
                image::imageops::FilterType::Nearest,
            );
            let data = img.to_rgb8().into_raw();
            Tensor::from_vec(
                data,
                (img.height() as usize, img.width() as usize, 3),
                &device,
            )
            .unwrap()
            .permute((2, 0, 1))
            .unwrap()
        };

        let image_t =
            (image_t.unsqueeze(0).unwrap().to_dtype(DType::F32).unwrap() * (1. / 255.)).unwrap();
        let predictions = model.forward(&image_t).unwrap().squeeze(0).unwrap();
        let bboxes = <YoloV8 as Task>::report(
            &predictions,
            0.5,  // args.confidence_threshold,
            0.45, // args.nms_threshold,
        )
        .unwrap();

        // scale xs and ys by the image size, compared to the original image size
        let xscale = original_image.width() as f32 / width as f32;
        let yscale = original_image.height() as f32 / height as f32;

        // collect the mins and sizes into a list, which will be logged simultaneously
        let boxes_mins_size_labels: Vec<(f32, f32, f32, f32, String)> = bboxes
            .iter()
            .map(|b| {
                let class_name = match b.data {
                    0 => format!("Empty Box ({:.1})", b.confidence),
                    1 => format!("Ball ({:.1})", b.confidence),
                    _ => format!("{:?}", b.data),
                };
                // print!
                println!("{:?} - {}", b, class_name);
                // convert xmin,xmax,ymin,ymax to x,y,w,h
                let x = b.xmin * xscale;
                let y = b.ymin * yscale;
                let w = (b.xmax - b.xmin) * xscale;
                let h = (b.ymax - b.ymin) * yscale;
                (x, y, w, h, class_name)
            })
            .collect();

        framenum += 1;
        info!("Inferenced frame {}", framenum);

        rec.set_time_sequence("frame", framenum);
        // log to rerun
        rec.log(
            "image_rgb/detections",
            &rerun::Boxes2D::from_mins_and_sizes(
                boxes_mins_size_labels
                    .iter()
                    .map(|(x, y, _, _, _)| (*x, *y)),
                boxes_mins_size_labels
                    .iter()
                    .map(|(_, _, w, h, _)| (*w, *h)),
            )
            .with_labels(
                boxes_mins_size_labels
                    .iter()
                    .map(|(_, _, _, _, label)| rerun::datatypes::Utf8::from(label.as_str()))
                    .collect::<Vec<_>>(),
            ),
        )
        .unwrap();
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp_nanos()
        .init();

    let rec = rerun::RecordingStreamBuilder::new("rerun_example_minimal")
        .serve_grpc_opts("0.0.0.0", 9876, MemoryLimit::from_fraction_of_total(0.25))
        .unwrap();

    spawner.spawn(task_camera(rec.clone())).unwrap();
    spawner.spawn(task_log_images(rec.clone())).unwrap();
    spawner.spawn(task_run_yolo(rec.clone())).unwrap();
}
