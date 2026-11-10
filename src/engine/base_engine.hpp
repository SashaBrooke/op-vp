#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct TrackedObject {
    cv::Rect box;
    int track_id;
    std::string label;
};

class BaseInferenceEngine {
public:
    virtual ~BaseInferenceEngine() = default;
    virtual void initialize() = 0;
    virtual std::vector<TrackedObject> track_frame(const cv::Mat& frame) = 0;
};
