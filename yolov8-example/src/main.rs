
use embassy_executor::Spawner;
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::pubsub::PubSubChannel;
use embassy_time::{Duration, Timer};

mod model;
use model::{Multiples, YoloV8};

use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_core as candle;
use candle_nn::{Module, VarBuilder};
use candle_transformers::object_detection::{Bbox, KeyPoint, non_maximum_suppression};

// Use V4L2 to get images
use image::DynamicImage;
use libcamera::{
    camera_manager::CameraManager,
    geometry::Size, logging::LoggingLevel,
    stream::{StreamConfigurationRef, StreamRole},
    utils::Immutable,
};
use std::iter::Iterator;

use libcamera::{
    camera::CameraConfigurationStatus,
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    pixel_format::PixelFormat,
    request::ReuseFlag,
};

use log::info;
use rerun::{MemoryLimit, RecordingStream};

pub static IMAGES_CHANNEL: PubSubChannel<CriticalSectionRawMutex, image::RgbImage, 2, 3, 1> =
    PubSubChannel::new();

// drm-fourcc does not have MJPEG type yet, construct it from raw fourcc identifier
pub const PIXEL_FORMAT: PixelFormat =
    PixelFormat::new(u32::from_le_bytes([b'M', b'J', b'P', b'G']), 0);
// raspi camera only supports YUYV directly
// pub const PIXEL_FORMAT: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'Y', b'U', b'Y', b'V']), 0);

// Change the output format as desired
//const IMAGE_FILE_SUFFIX: &str = "png";
pub const IMAGE_FILE_SUFFIX: &str = "jpg";

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
    let mgr = CameraManager::new().unwrap();
    mgr.log_set_level("Camera", LoggingLevel::Error);
    let cameras = mgr.cameras();
    let cam = cameras.get(1).expect("No cameras found");
    let mut cam = cam.acquire().expect("Unable to acquire camera");
    //.generate_configuration(&[StreamRole::VideoRecording])
    let mut cfgs = cam
        .generate_configuration(&[StreamRole::VideoRecording])
        .unwrap();
    cfgs.get_mut(0).unwrap().set_pixel_format(PIXEL_FORMAT);

    match cfgs.validate() {
        CameraConfigurationStatus::Valid => info!("Camera configuration valid!"),
        CameraConfigurationStatus::Adjusted => {
            info!("Camera configuration was adjusted: {:#?}", cfgs)
        }
        CameraConfigurationStatus::Invalid => panic!("Error validating camera configuration"),
    }

    // Ensure that pixel format was unchanged
    assert_eq!(
        cfgs.get(0).unwrap().get_pixel_format(),
        PIXEL_FORMAT,
        "Selected pixel format is not supported by the camera"
    );

    let cfg_size: Size = Size {
        width: 1920,
        height: 1080,
    };
    let mut mut_cfg: StreamConfigurationRef = cfgs.get_mut(0).unwrap();
    mut_cfg.set_size(cfg_size);
    //mut_cfg.set_buffer_count(2);

    cam.configure(&mut cfgs)
        .expect("Unable to configure camera");

    let mut alloc = FrameBufferAllocator::new(&cam);

    // Allocate frame buffers for the stream
    let cfg: Immutable<StreamConfigurationRef> = cfgs.get(0).unwrap();

    let image_size = cfg.value().get_size();
    let height = image_size.height;
    let width = image_size.width;
    let stride = cfg.value().get_stride();

    let stream = cfg.stream().unwrap();
    let buffers = alloc.alloc(&stream).unwrap();

    // Convert FrameBuffer to MemoryMappedFrameBuffer, which allows reading &[u8]
    let buffers = buffers
        .into_iter()
        .map(|buf| MemoryMappedFrameBuffer::new(buf).unwrap())
        .collect::<Vec<_>>();

    // Create capture requests and attach buffers
    let reqs = buffers
        .into_iter()
        .map(|buf| {
            let mut req = cam.create_request(None).unwrap();
            req.add_buffer(&stream, buf).unwrap();
            req
        })
        .collect::<Vec<_>>();

    // Completed capture requests are returned as a callback
    let (tx, rx) = std::sync::mpsc::channel();

    cam.on_request_completed(move |req| {
        tx.send(req).unwrap();
    });

    cam.start(None).unwrap();

    // Enqueue all requests to the camera
    for req in reqs {
        println!("Request queued for execution: {req:#?}");
        cam.queue_request(req).unwrap();
    }

    // TODO: Convert from raw YUYV pixels data, into BGR data, then encode as JPEG.
    let target_channels: u32 = 3;
    let img_rgb = vec![0u8; width as usize * height as usize * target_channels as usize];
    let tx = IMAGES_CHANNEL.publisher().unwrap();

    loop {
        // Multiple requests can be queued at a time, but for this example we just want a single frame.
        // cam.queue_request(reqs.pop().unwrap()).unwrap();
        let mut rreq = rx
            .recv_timeout(std::time::Duration::from_millis(5000).into())
            .expect("Camera request failed");
        let framebuffer: &MemoryMappedFrameBuffer<FrameBuffer> = rreq.buffer(&stream).unwrap();

        // NOTE: MJPEG format has only one data plane containing encoded jpeg data with all the headers
        let planes = framebuffer.data();
        let img_data = planes.get(0).unwrap();
        // Actual data will be smalled than framebuffer size, its length can be obtained from metadata.
        let data_len = framebuffer
            .metadata()
            .unwrap()
            .planes()
            .get(0)
            .unwrap()
            .bytes_used as usize;

        // ONLY FOR PIXEL FORMAT YUV
        /*
        // Convert the raw YUYV422 packed pixel data into RGB8
        let src_yuyv422: YuvPackedImage<u8> = YuvPackedImage {
            yuy: &img_data[..data_len],
            yuy_stride: stride,
            width,
            height,
        };
        src_yuyv422
            .check_constraints()
            .expect("YUYV422 data formed correctly.");
        yuyv422_to_rgb(
            &src_yuyv422,
            &mut img_rgb,
            width * target_channels,
            YuvRange::Limited,
            YuvStandardMatrix::Bt601,
        )
        .unwrap();
        let buffered_image = image::RgbImage::from_vec(width, height, img_rgb.clone()).expect("Built image from buffer");
        */

        // ONLY FOR PIXEL FORMAT MJPEG
        let imgbuf = std::io::Cursor::new(&img_data[..data_len]);
        let buffered_image = image::ImageReader::new(imgbuf)
            .with_guessed_format()
            .unwrap()
            .decode()
            .expect("Decoded image")
            .to_rgb8();

        tx.publish_immediate(buffered_image);

        // Push request back onto queue and go again after a second
        rreq.reuse(ReuseFlag::REUSE_BUFFERS);
        // reqs.push(req);
        cam.queue_request(rreq).unwrap();

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
        .filter_level(log::LevelFilter::Debug)
        .format_timestamp_nanos()
        .init();

    let rec = rerun::RecordingStreamBuilder::new("rerun_example_minimal")
        .serve_grpc_opts("0.0.0.0", 9876, MemoryLimit::from_fraction_of_total(0.25))
        .unwrap();

    spawner.spawn(task_camera(rec.clone())).unwrap();
    spawner.spawn(task_log_images(rec.clone())).unwrap();
    spawner.spawn(task_run_yolo(rec.clone())).unwrap();
}
