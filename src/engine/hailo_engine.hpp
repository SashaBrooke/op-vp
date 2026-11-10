#pragma once
#include "base_engine.hpp"
#include <memory>

class HailoEngine : public BaseInferenceEngine {
private:
    std::string model_path;
    class Impl;
    std::unique_ptr<Impl> pimpl;
public:
    explicit HailoEngine(std::string path);
    ~HailoEngine() override;
    void initialize() override;
    std::vector<TrackedObject> track_frame(const cv::Mat& frame) override;
};
