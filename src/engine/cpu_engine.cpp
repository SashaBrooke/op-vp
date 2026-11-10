#include "cpu_engine.hpp"
#include <iostream>

CPUEngine::CPUEngine(std::string path) : model_path(path.empty() ? "models/yolov8n.onnx" : path) {}
void CPUEngine::initialize() { std::cout << "[Engine] Initializing Fallback CPU Context.\n"; }
std::vector<TrackedObject> CPUEngine::track_frame(const cv::Mat& /*frame*/) { return {}; }
