#pragma once

#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "base_source.hpp"

class FileSource : public BaseVideoSource {
public:
    explicit FileSource(std::string path);
    ~FileSource() override = default;

    void start() override;
    bool read_frame(cv::Mat& frame) override;
    void release() override;

private:
    std::string file_path;
    cv::VideoCapture cap;
    
    double video_fps;
    int64_t frame_duration_us;
    std::chrono::steady_clock::time_point last_frame_time;
};
