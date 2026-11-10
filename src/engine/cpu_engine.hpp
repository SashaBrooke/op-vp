#pragma once
#include "base_engine.hpp"

class CPUEngine : public BaseInferenceEngine {
private:
    std::string model_path;
public:
    explicit CPUEngine(std::string path);
    void initialize() override;
    std::vector<TrackedObject> track_frame(const cv::Mat& frame) override;
};
