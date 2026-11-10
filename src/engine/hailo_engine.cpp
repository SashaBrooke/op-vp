#include "hailo_engine.hpp"
#include <iostream>

class HailoEngine::Impl {};

HailoEngine::HailoEngine(std::string path) 
    : model_path(path.empty() ? "yolov8n_npu.hef" : path), pimpl(std::make_unique<Impl>()) {}
HailoEngine::~HailoEngine() = default;

void HailoEngine::initialize() { std::cout << "[Engine] Initializing Raspberry Pi 5 NPU Context.\n"; }
std::vector<TrackedObject> HailoEngine::track_frame(const cv::Mat& /*frame*/) { return {}; }
