use embassy_executor::Spawner;
use embassy_time::{Duration,Timer,Ticker};
use log::*;
use libcamera::{camera_manager::CameraManager, geometry::Size, logging::LoggingLevel, stream::{Stream, StreamConfigurationRef, StreamRole}, utils::Immutable};

use libcamera::{
    camera::CameraConfigurationStatus,
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    pixel_format::PixelFormat,
    request::ReuseFlag,
    properties,
};

use yuvutils_rs::{yuyv422_to_rgb,YuvPackedImage,YuvStandardMatrix,YuvRange};

// drm-fourcc does not have MJPEG type yet, construct it from raw fourcc identifier
//const PIXEL_FORMAT: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'M', b'J', b'P', b'G']), 0);
// raspi camera only supports YUYV directly
const PIXEL_FORMAT: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'Y', b'U', b'Y', b'V']), 0);

// Change the output format as desired
const IMAGE_FILE_SUFFIX: &str = "png";
//const IMAGE_FILE_SUFFIX: &str = "jpg";

#[embassy_executor::task]
async fn task_camera_capture(filename : String) {
    let mgr = CameraManager::new().unwrap();
    mgr.log_set_level("Camera", LoggingLevel::Error);

    let cameras = mgr.cameras();
    let cam = cameras.get(0).expect("No cameras found");

    //info!(
    //    "Using camera: {}",
    //    *cam.properties().get::<properties::Model>().unwrap()
    //);

    let mut cam = cam.acquire().expect("Unable to acquire camera");

    // This will generate default configuration for each specified role
    //let mut cfgs = cam.generate_configuration(&[StreamRole::Raw]).unwrap();
    let mut cfgs = cam.generate_configuration(&[StreamRole::StillCapture]).unwrap();
    //let mut cfgs = cam.generate_configuration(&[StreamRole::VideoRecording]).unwrap();
    //let mut cfgs = cam.generate_configuration(&[StreamRole::ViewFinder]).unwrap();

    // Use MJPEG format so we can write resulting frame directly into jpeg file
    cfgs.get_mut(0).unwrap().set_pixel_format(PIXEL_FORMAT);

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

    /* Set the pixel size
    {
        //let cfg_size : Size = Size { width: 1920, height: 1080 };
        let cfg_size : Size = Size { width: 1296, height: 972 };
        let mut mut_cfg : StreamConfigurationRef = cfgs.get_mut(0).unwrap();
        mut_cfg.set_size(cfg_size);
    }
    */
    
    info!("Generated config: {:#?}", cfgs);

    // Apply the final config
    cam.configure(&mut cfgs).expect("Unable to configure camera");

    let mut alloc = FrameBufferAllocator::new(&cam);

    // Allocate frame buffers for the stream
    let cfg: Immutable<StreamConfigurationRef> = cfgs.get(0).unwrap();

    let image_size = cfg.value().get_size();
    let height = image_size.height;
    let width = image_size.width;
    let stride = cfg.value().get_stride();

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

    // TODO: Convert from raw YUYV pixels data, into BGR data, then encode as JPEG.
    let target_channels : u32 = 3;
    let mut img_rgb = vec![0u8; width as usize * height as usize * target_channels as usize];

    let mut tick = Ticker::every(Duration::from_millis(1000));
    
    loop {
        // Multiple requests can be queued at a time, but for this example we just want a single frame.
        cam.queue_request(reqs.pop().unwrap()).unwrap();
        info!("Waiting for camera request execution");
        let mut req = rx.recv_timeout(Duration::from_millis(1000).into()).expect("Camera request failed");
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

        // Convert the raw YUYV422 packed pixel data into RGB8
        let src_yuyv422 : YuvPackedImage<u8> = YuvPackedImage{ yuy: &img_data[..data_len], yuy_stride: stride, width, height };
        src_yuyv422.check_constraints().expect("YUYV422 data formed correctly.");
        yuyv422_to_rgb(&src_yuyv422, &mut img_rgb, width * target_channels, YuvRange::Limited, YuvStandardMatrix::Bt601).unwrap();
        
        // encode the RGB buffer into a regular image file, depending on IMAGE_FILE_SUFFIX.
        let fname = format!("{}_{}.{}",&filename,req.sequence(),IMAGE_FILE_SUFFIX);
        image::save_buffer(&fname, &img_rgb, width, height, image::ExtendedColorType::Rgb8).unwrap();

        info!("Written {} bytes to {}", img_rgb.len(), &fname);

        req.reuse(ReuseFlag::REUSE_BUFFERS);
        reqs.push(req);

        // Push request back onto queue and go again after a second
        tick.next().await;
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
