use std::io::Write;
use std::io::{BufRead, BufReader, BufWriter};
use std::{env, io::Seek};

mod model;
use model::{Multiples, YoloV8};

use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_core as candle;
use candle_nn::{Module, VarBuilder};
use candle_transformers::object_detection::{Bbox, KeyPoint, non_maximum_suppression};

// Use V4L2 to get images
use image::DynamicImage;
use libcamera::{
    camera::ActiveCamera,
    camera_manager::CameraManager,
    geometry::Size,
    logging::LoggingLevel,
    stream::{Stream, StreamConfigurationRef, StreamRole},
    utils::Immutable,
};
use std::iter::Iterator;
use tempfile::{NamedTempFile, tempfile};

use libcamera::{
    camera::CameraConfigurationStatus,
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    pixel_format::PixelFormat,
    properties,
    request::ReuseFlag,
};

use log::{Level, error, info};
use yuvutils_rs::{YuvPackedImage, YuvRange, YuvStandardMatrix, yuyv422_to_rgb};
use rerun::{RecordingStream,MemoryLimit};

// drm-fourcc does not have MJPEG type yet, construct it from raw fourcc identifier
//const PIXEL_FORMAT: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'M', b'J', b'P', b'G']), 0);
// raspi camera only supports YUYV directly
pub const PIXEL_FORMAT: PixelFormat =
    PixelFormat::new(u32::from_le_bytes([b'Y', b'U', b'Y', b'V']), 0);

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

pub fn load_model() -> anyhow::Result<std::path::PathBuf> {
    let path = {
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model("lmz/candle-yolo-v8".to_string());
        let size = 'n';
        api.get(&format!("yolov8{size}.safetensors"))?
    };
    Ok(path)
}

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
pub fn run<T: Task>(rec: RecordingStream) -> anyhow::Result<()> {
    let device = Device::Cpu;

    // Create the model and load the weights from the file.
    let multiples = Multiples::n();

    // let model = load_model()?;
    let model: std::path::PathBuf = "best.safetensors".into();

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? };
    let model = T::load(vb, multiples)?;

    let mgr = CameraManager::new().unwrap();
    mgr.log_set_level("Camera", LoggingLevel::Error);
    let cameras = mgr.cameras();
    let cam = cameras.get(0).expect("No cameras found");
    let mut cam = cam.acquire().expect("Unable to acquire camera");
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
        width: 1280,
        height: 720,
    };
    let mut mut_cfg: StreamConfigurationRef = cfgs.get_mut(0).unwrap();
    mut_cfg.set_size(cfg_size);

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
    let mut reqs = buffers
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

    // TODO: Convert from raw YUYV pixels data, into BGR data, then encode as JPEG.
    let target_channels: u32 = 3;
    let mut img_rgb = vec![0u8; width as usize * height as usize * target_channels as usize];
    let mut framenum = 0;

    loop {
        // Multiple requests can be queued at a time, but for this example we just want a single frame.
        cam.queue_request(reqs.pop().unwrap()).unwrap();
        let mut req = rx
            .recv_timeout(std::time::Duration::from_millis(5000).into())
            .expect("Camera request failed");
        let framebuffer: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();

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

        // Push request back onto queue and go again after a second
        req.reuse(ReuseFlag::REUSE_BUFFERS);
        reqs.push(req);

        // // this works to read files, but is inefficient.
        // let mut file = tempfile().expect("Created temporary image file.");
        // image::write_buffer_with_format(file, &img_rgb, width, height, image::ExtendedColorType::Rgb8, image::ImageFormat::Jpeg).expect("Wrote JPEG to buffer.");
        rec.set_time_sequence("frame",framenum);
        rec.log(
            "image_rgb",
            &rerun::Image::from_rgb24(img_rgb.clone(), [width, height]),
        )
        .unwrap();
        framenum += 1;

        // Create a DynamicImage from the img_rgb buffer.
        let buffered_image = image::RgbImage::from_vec(width, height, img_rgb.clone()).expect("Built image from buffer");
        let original_image = DynamicImage::ImageRgb8(buffered_image);

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
                image::imageops::FilterType::CatmullRom,
            );
            let data = img.to_rgb8().into_raw();
            Tensor::from_vec(
                data,
                (img.height() as usize, img.width() as usize, 3),
                &device,
            )?
            .permute((2, 0, 1))?
        };

        let image_t = (image_t.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;
        let predictions = model.forward(&image_t)?.squeeze(0)?;
        let bboxes = T::report(
            &predictions,
            0.5, // args.confidence_threshold,
            0.45, // args.nms_threshold,
        )?;

        // scale xs and ys by the image size, compared to the original image size
        let xscale = original_image.width() as f32 / width as f32;
        let yscale = original_image.height() as f32 / height as f32;

        for b in bboxes {
            let class_name = match b.data {
                0 => &format!("Empty Box ({:.1})",b.confidence),
                1 => &format!("Ball ({:.1})",b.confidence),
                _ => &format!("{:?}", b.data),
            };
            println!("{:?} - {}", b, class_name);

            // convert xmin,xmax,ymin,ymax to x,y,w,h
            let x = b.xmin * xscale;
            let y = b.ymin * yscale;
            let w = (b.xmax - b.xmin) * xscale;
            let h = (b.ymax - b.ymin) * yscale;

            // log to rerun
            rec.log(
                "image_rgb/detections",
                &rerun::Boxes2D::from_mins_and_sizes([(x,y)], [(w,h)]).with_labels([rerun::datatypes::Utf8::from(class_name.as_str())])
            )?;
        }
    }
}

pub fn main() -> anyhow::Result<()> {
    let rec = rerun::RecordingStreamBuilder::new("rerun_example_minimal").serve_grpc_opts("0.0.0.0", 9876, MemoryLimit::from_fraction_of_total(0.25)).unwrap();
    run::<YoloV8>(rec.clone())?;
    Ok(())
}
