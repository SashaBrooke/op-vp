#pragma once
#include "base_source.hpp"
#include <memory>

class CameraSource : public BaseVideoSource {
private:
    int camera_index;
    class Impl;
    std::unique_ptr<Impl> pimpl;

public:
    explicit CameraSource(int index = 0);
    ~CameraSource() override;
    void start() override;
    bool read_frame(cv::Mat& frame) override;
    void release() override;
};
