use embassy_executor::Spawner;
use embassy_time::{Duration,Timer};
use log::*;
use libcamera::{camera_manager::CameraManager, logging::LoggingLevel, stream::StreamRole};

use libcamera::{
    camera::CameraConfigurationStatus,
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    pixel_format::PixelFormat,
    request::ReuseFlag,
    properties,
};

// drm-fourcc does not have MJPEG type yet, construct it from raw fourcc identifier
//const PIXEL_FORMAT: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'M', b'J', b'P', b'G']), 0);
const PIXEL_FORMAT: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'Y', b'U', b'Y', b'V']), 0);
const IMAGE_FILE_SUFFIX: &str = "raw";

#[embassy_executor::task]
async fn task_camera_capture(filename : String) {
    let mgr = CameraManager::new().unwrap();
    mgr.log_set_level("Camera", LoggingLevel::Error);

    let cameras = mgr.cameras();

    let cam = cameras.get(0).expect("No cameras found");

    info!(
        "Using camera: {}",
        *cam.properties().get::<properties::Model>().unwrap()
    );

    let mut cam = cam.acquire().expect("Unable to acquire camera");

    // This will generate default configuration for each specified role
    //let mut cfgs = cam.generate_configuration(&[StreamRole::Raw]).unwrap();
    let mut cfgs = cam.generate_configuration(&[StreamRole::StillCapture]).unwrap();
    //let mut cfgs = cam.generate_configuration(&[StreamRole::VideoRecording]).unwrap();
    //let mut cfgs = cam.generate_configuration(&[StreamRole::ViewFinder]).unwrap();

    // Use MJPEG format so we can write resulting frame directly into jpeg file
    cfgs.get_mut(0).unwrap().set_pixel_format(PIXEL_FORMAT);

    info!("Generated config: {:#?}", cfgs);

    match cfgs.validate() {
        CameraConfigurationStatus::Valid => info!("Camera configuration valid!"),
        CameraConfigurationStatus::Adjusted => info!("Camera configuration was adjusted: {:#?}", cfgs),
        CameraConfigurationStatus::Invalid => panic!("Error validating camera configuration"),
    }

    // Ensure that pixel format was unchanged
    assert_eq!(
        cfgs.get(0).unwrap().get_pixel_format(),
        PIXEL_FORMAT,
        "Selected pixel format is not supported by the camera"
    );

    cam.configure(&mut cfgs).expect("Unable to configure camera");

    let mut alloc = FrameBufferAllocator::new(&cam);

    // Allocate frame buffers for the stream
    let cfg = cfgs.get(0).unwrap();
    let stream = cfg.stream().unwrap();
    let buffers = alloc.alloc(&stream).unwrap();
    info!("Allocated {} buffers", buffers.len());

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

    
    loop {
        // Multiple requests can be queued at a time, but for this example we just want a single frame.
        cam.queue_request(reqs.pop().unwrap()).unwrap();

        info!("Waiting for camera request execution");
        let mut req = rx.recv_timeout(Duration::from_secs(1).into()).expect("Camera request failed");
        info!("Camera request {:?} completed!", req);
        info!("Metadata: {:#?}", req.metadata());
        // Get framebuffer for our stream
        let framebuffer: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();
        info!("FrameBuffer metadata: {:#?}", framebuffer.metadata());

        // NOTE: MJPEG format has only one data plane containing encoded jpeg data with all the headers
        let planes = framebuffer.data();
        //info!("{:?}", planes);
        let img_data = planes.get(0).unwrap();
        // Actual data will be smalled than framebuffer size, its length can be obtained from metadata.
        let data_len = framebuffer.metadata().unwrap().planes().get(0).unwrap().bytes_used as usize; 
        
        // TODO: Convert from raw YUYV pixels data, into RGB pixels, then into JPEG.
        //       The ezk-image crate looks decent for this.
        /*
        let (width, height) = (1920, 1080);
        let rgb_image = vec![0u8; PixelFormat::RGB.buffer_size(width, height)];
        let source = Image::from_buffer(
            PixelFormat::RGB,
            &rgb_image[..], // RGB only has one plane
            None, // No need to define strides if there's no padding between rows
            width,
            height,
            ColorInfo::RGB(RgbColorInfo {
                transfer: ColorTransfer::Linear,
                primaries: ColorPrimaries::BT709,
            }),
        ).unwrap();
        let mut destination = Image::blank(
            PixelFormat::NV12, // We're converting to NV12
            width,
            height,
            ColorInfo::YUV(YuvColorInfo {
                space: ColorSpace::BT709,
                transfer: ColorTransfer::Linear,
                primaries: ColorPrimaries::BT709,
                full_range: false,
            }),
        );
        convert_multi_thread(
            &source,
            &mut destination,
        ).unwrap();
        */

        let fname = format!("{}_{}.{}",&filename,req.sequence(),IMAGE_FILE_SUFFIX);
        std::fs::write(&fname, &img_data[..data_len]).unwrap();
        info!("Written {} bytes to {}", data_len, &fname);

        // Push request back onto queue and go again after a second
        Timer::after_millis(1000).await;

        req.reuse(ReuseFlag::REUSE_BUFFERS);
        reqs.push(req);

    }
}


#[embassy_executor::task]
async fn task_ticker() {
    loop {
        info!("tick");
        Timer::after_millis(500).await;
    }
}

#[embassy_executor::task]
async fn task_beeper() {
    loop {
        info!("beep");
        Timer::after_millis(2000).await;
    }
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let filename = std::env::args().nth(1).expect("Usage ./app <filename.jpg>");

    env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .format_timestamp_nanos()
        .init();

    spawner.spawn(task_ticker()).unwrap();
    spawner.spawn(task_beeper()).unwrap();
    spawner.spawn(task_camera_capture(filename)).unwrap();
}
