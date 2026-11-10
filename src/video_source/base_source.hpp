#pragma once
#include <opencv2/opencv.hpp>

class BaseVideoSource {
public:
    virtual ~BaseVideoSource() = default;
    virtual void start() = 0;
    virtual bool read_frame(cv::Mat& frame) = 0;
    virtual void release() = 0;
};
