#include "file_source.hpp"
#include <stdexcept>
#include <iostream>
#include <thread>
#include <chrono>

FileSource::FileSource(std::string path) : file_path(std::move(path)), video_fps(30.0), frame_duration_us(33333) {}

void FileSource::start() {
    cap.open(file_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Failed to parse video file: " + file_path);
    }

    // Automatically extract native FPS from file metadata
    video_fps = cap.get(cv::CAP_PROP_FPS);

    if (video_fps <= 0.0 || video_fps > 240.0) {
        std::cout << "[Source Warning] Invalid video metadata FPS (" << video_fps << "). Defaulting to 30.0 FPS.\n";
        video_fps = 30.0;
    } else {
        std::cout << "[Source] Detected Video Metadata Stream Rate: " << video_fps << " FPS\n";
    }

    frame_duration_us = static_cast<int64_t>(1000000.0 / video_fps);
    last_frame_time = std::chrono::steady_clock::now();
}

bool FileSource::read_frame(cv::Mat& frame) {
    // Pace processing to match native video playback speed
    auto now = std::chrono::steady_clock::now();
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(now - last_frame_time).count();

    if (elapsed_us < frame_duration_us) {
        auto sleep_duration = std::chrono::microseconds(frame_duration_us - elapsed_us);
        std::this_thread::sleep_for(sleep_duration);
    }
    
    last_frame_time = std::chrono::steady_clock::now();
    return cap.read(frame);
}

void FileSource::release() { 
    if (cap.isOpened()) {
        cap.release(); 
    }
}
