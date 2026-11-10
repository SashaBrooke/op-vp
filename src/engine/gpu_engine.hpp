#pragma once
#include "base_engine.hpp"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <string>

class GPUEngine : public BaseInferenceEngine {
private:
    std::string model_path;
    
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;

public:
    explicit GPUEngine(std::string path);
    void initialize() override;
    std::vector<TrackedObject> track_frame(const cv::Mat& frame) override;
};
