#include "camera_source.hpp"
#include <iostream>
#include <stdexcept>

#ifdef PLATFORM_RPI5
#include "core/rpicam_app.hpp"
#include "core/options.hpp"

class CameraSource::Impl {
public:
    RPiCamApp app;
    Options options;
};
#else
class CameraSource::Impl {};
#endif

CameraSource::CameraSource(int index) 
    : camera_index(index), pimpl(std::make_unique<Impl>()) {}

CameraSource::~CameraSource() { 
    release(); 
}

void CameraSource::start() {
#ifdef PLATFORM_RPI5
    pimpl->options.camera = camera_index;
    pimpl->options.width = 1920;
    pimpl->options.height = 1080;
    pimpl->options.framerate = 30.0;
    pimpl->options.verbose = 0;
    pimpl->app.ParseOptions(pimpl->options);
    pimpl->app.OpenCamera();
    pimpl->app.ConfigureVideo();
    pimpl->app.StartCamera();
#else
    throw std::runtime_error("Fatal Error: Raspberry Pi 5 Camera Hardware streams are unavailable in Desktop cross-compiles!");
#endif
}

bool CameraSource::read_frame(cv::Mat& frame) {
#ifdef PLATFORM_RPI5
    RPiCamApp::Msg msg = pimpl->app.Wait();
    if (msg.type != RPiCamApp::MsgType::Frame) return false;
    
    CompletedRequestPtr request = std::get<CompletedRequestPtr>(msg.payload);
    libcamera::Span<uint8_t> buffer = pimpl->app.GetStreamBuffer(request);
    
    cv::Mat raw_yuv(1080 * 1.5, 1920, CV_8UC1, buffer.data());
    cv::cvtColor(raw_yuv, frame, cv::COLOR_YUV2BGR_I420);
    return true;
#else
    (void)frame;
    return false;
#endif
}

void CameraSource::release() {
#ifdef PLATFORM_RPI5
    if (pimpl) {
        pimpl->app.StopCamera();
        pimpl->app.CloseCamera();
    }
#endif
}
