#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <condition_variable>
#include <iomanip> 
#include <vector>

#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>

#include "base_engine.hpp"
#include "base_source.hpp"
#include "engine_factory.hpp"
#include "video_source_factory.hpp"

namespace po = boost::program_options;

template <typename T>
class SafeQueue {
private:
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cv;
    size_t max_size = 60; 
public:
    void push(T value) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this]() { return queue.size() < max_size; });
        queue.push(std::move(value));
        lock.unlock();
        cv.notify_one();
    }
    bool pop(T& value, std::atomic<bool>& running) {
        std::unique_lock<std::mutex> lock(mutex);
        while (queue.empty() && running) {
            if (cv.wait_for(lock, std::chrono::milliseconds(50)) == std::cv_status::timeout && !running) {
                return false;
            }
        }
        if (queue.empty()) return false;
        value = std::move(queue.front());
        queue.pop();
        lock.unlock();
        cv.notify_one();
        return true;
    }
};

SafeQueue<cv::Mat> output_queue;
std::atomic<bool> pipeline_running(true);
std::mutex io_queue_mutex;
std::queue<std::vector<int>> serial_tx_queue;

void serial_io_worker() {
    while (pipeline_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void network_io_worker() {
    while (pipeline_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void video_writer_worker() {
    cv::VideoWriter video_writer;
    cv::Mat frame;
    while (output_queue.pop(frame, pipeline_running)) {
        if (!video_writer.isOpened() && !frame.empty()) {
            video_writer.open("output/output_tracked.mp4", cv::VideoWriter::fourcc('m','p','4','v'), 30.0, frame.size());
        }
        if (video_writer.isOpened()) {
            video_writer.write(frame);
        }
    }
    if (video_writer.isOpened()) video_writer.release();
}

int main(int argc, char* argv[]) {
    std::string engine_param;
    std::string source_param;
    std::string input_path;
    std::string model_path;
    int cam_idx = 0;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Produce help message")
        ("engine", po::value<std::string>(&engine_param)->default_value("gpu"), "Inference engine hardware choice: gpu, cpu, hailo")
        ("source", po::value<std::string>(&source_param)->default_value("file"), "Video ingestion choice: file, camera")
        ("input", po::value<std::string>(&input_path)->default_value("media/car_occlusion.mp4"), "Target file path if source is 'file'")
        ("model", po::value<std::string>(&model_path)->default_value(""), "Custom model weights override path")
        ("cam_idx", po::value<int>(&cam_idx)->default_value(0), "System camera hardware index mapping");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "CLI Parsing Error: " << e.what() << "\n";
        return 1;
    }

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    EngineType selected_engine = EngineType::GPU;
    if (engine_param == "cpu") selected_engine = EngineType::CPU;
    else if (engine_param == "hailo") selected_engine = EngineType::HAILO;

    SourceType selected_source = SourceType::FILE;
    if (source_param == "camera") selected_source = SourceType::CAMERA;

    std::cout << "[Pipeline] Booting production C++ tracker engine...\n";
    std::cout << "[Pipeline] Hardware Backend: " << engine_param << "\n";
    
    auto engine = EngineFactory::create_engine(selected_engine, model_path);
    auto video_source = VideoSourceFactory::create_source(selected_source, input_path, cam_idx);

    engine->initialize();
    video_source->start();

    std::thread serial_thread(serial_io_worker);
    std::thread network_thread(network_io_worker);
    std::thread writer_thread(video_writer_worker);

    try {
        cv::Mat frame;
        int frame_count = 0;
        
        // Metrics counters
        double total_inference_time = 0.0;
        double total_drawing_time = 0.0;
        double total_handoff_time = 0.0;

        // Moving window containers for live rolling FPS calculation
        std::vector<std::chrono::high_resolution_clock::time_point> timestamps;
        const size_t fps_window_size = 30; 
        double live_fps = 0.0;

        auto start_time = std::chrono::high_resolution_clock::now();

        std::cout << "--------------------------------------------------------------------\n";
        std::cout << "Frame | Inference  | Drawing/Post | Queue Handoff | Live Rolling FPS\n";
        std::cout << "--------------------------------------------------------------------\n";



        while (video_source->read_frame(frame) && !frame.empty()) {
            auto current_frame_time = std::chrono::high_resolution_clock::now();
            frame_count++;
            
            // 1. Calculate Live Rolling FPS based on a moving time delta array window
            timestamps.push_back(current_frame_time);
            if (timestamps.size() > fps_window_size) {
                timestamps.erase(timestamps.begin());
            }
            if (timestamps.size() > 1) {
                auto duration_seconds = std::chrono::duration<double>(timestamps.back() - timestamps.front()).count();
                live_fps = (timestamps.size() - 1) / duration_seconds;
            }

            // --- INFERENCE ENGINE WORK STEP ---
            auto t_inf_start = std::chrono::high_resolution_clock::now();
            auto targets = engine->track_frame(frame);
            auto t_inf_end = std::chrono::high_resolution_clock::now();

            // --- DRAWING & METRICS STEP ---
            auto t_draw_start = std::chrono::high_resolution_clock::now();
            if (!targets.empty()) {
                std::lock_guard<std::mutex> lock(io_queue_mutex);
                std::vector<int> target_count_vec;
                target_count_vec.push_back(static_cast<int>(targets.size()));
                serial_tx_queue.push(target_count_vec);
            }

            for (const auto& target : targets) {
                cv::rectangle(frame, target.box, cv::Scalar(0, 255, 0), 2);
            }
            auto t_draw_end = std::chrono::high_resolution_clock::now();

            // --- ASYNC DISK DISPATCH HANDOFF STEP ---
            auto t_queue_start = std::chrono::high_resolution_clock::now();
            output_queue.push(frame.clone());
            auto t_queue_end = std::chrono::high_resolution_clock::now();

            double inf_ms = std::chrono::duration<double, std::milli>(t_inf_end - t_inf_start).count();
            double draw_ms = std::chrono::duration<double, std::milli>(t_draw_end - t_draw_start).count();
            double queue_ms = std::chrono::duration<double, std::milli>(t_queue_end - t_queue_start).count();

            total_inference_time += inf_ms;
            total_drawing_time += draw_ms;
            total_handoff_time += queue_ms;

            std::string inf_str   = cv::format("%.2fms", inf_ms);
            std::string draw_str  = cv::format("%.2fms", draw_ms);
            std::string queue_str = cv::format("%.2fms", queue_ms);
            std::string fps_str   = cv::format("%.1f FPS", live_fps);

            // Output real-time metrics
            std::cout << std::setw(5) << frame_count << " | "
                      << std::setw(10) << inf_str << " | "
                      << std::setw(12) << draw_str << " | "
                      << std::setw(13) << queue_str << " | "
                      << std::setw(16) << fps_str << "\r" << std::flush;
        }
        
        pipeline_running = false;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end_time - start_time;
        
        std::cout << "\n--------------------------------------------------------------------\n";
        std::cout << "[Pipeline Metrics Summary Table]\n";
        std::cout << "\t- Avg Inference Engine Step : " << (total_inference_time / frame_count) << " ms\n";
        std::cout << "\t- Avg Drawing / BBox Step   : " << (total_drawing_time / frame_count) << " ms\n";
        std::cout << "\t- Avg Async Queue Handoff   : " << (total_handoff_time / frame_count) << " ms\n";
        std::cout << "--------------------------------------------------------------------\n";
        std::cout << "[Pipeline Complete] Overall Pipeline Speed: " << (frame_count / diff.count()) << " FPS\n";
        
    } catch (const std::exception& e) {
        pipeline_running = false;
        std::cerr << "\n[Runtime Failure] " << e.what() << "\n";
    }

    video_source->release();
    if (serial_thread.joinable()) serial_thread.join();
    if (network_thread.joinable()) network_thread.join();
    if (writer_thread.joinable()) writer_thread.join();
    return 0;
}
