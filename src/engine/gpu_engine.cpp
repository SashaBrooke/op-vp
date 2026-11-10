#include "gpu_engine.hpp"
#include <iostream>
#include <vector>
#include <numeric>

GPUEngine::GPUEngine(std::string path) 
    : model_path(path.empty() ? "models/yolov8n.onnx" : path) {}

void GPUEngine::initialize() {
    std::cout << "[Engine] Initializing Production ONNX Runtime C++ GPU Engine...\n";
    
    // 1. Initialize the internal core API environment instance
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OPVP_Inference");
    
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 2. Direct Hardware Layer Binding: Force execution onto CUDA Execution Providers
    try {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0; // Target your primary RTX GPU
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "[Engine] Linked successfully to native CUDA Hardware Lanes.\n";
    } catch (const std::exception& e) {
        std::cerr << "[Engine Warning] CUDA provider failed. Using CPU fallback: " << e.what() << "\n";
    }

    // 3. Instantiate the complete compiled model session
    session = std::make_unique<Ort::Session>(*env, model_path.c_str(), session_options);

    // 4. Map static network input/output string tokens
    input_node_names = {"images"};
    output_node_names = {"output0"};
}

std::vector<TrackedObject> GPUEngine::track_frame(const cv::Mat& frame) {
    std::vector<TrackedObject> tracked_items;
    if (frame.empty()) return tracked_items;

    // --- STAGE 1: HIGH-SPEED IMAGE PRE-PROCESSING ---
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // HWC layout array converted to continuous planar CHW tensor formats
    std::vector<float> input_tensor_values(1 * 3 * 640 * 640);
    cv::Mat float_mat;
    resized.convertTo(float_mat, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> chw_channels(3);
    for (int i = 0; i < 3; ++i) {
        chw_channels[i] = cv::Mat(640, 640, CV_32FC1, &input_tensor_values[i * 640 * 640]);
    }
    cv::split(float_mat, chw_channels);

    // --- STAGE 2: EXECUTE GPU TENSOR INFERENCE ---
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    std::array<int64_t, 4> input_shape = {1, 3, 640, 640};
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size()
    );

    auto output_tensors = session->Run(
        Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1
    );

    // --- STAGE 3: COORDINATE METRIC PARSING ---
    auto& output_tensor = output_tensors[0];
    
    float* raw_output = output_tensor.GetTensorMutableData<float>();
    auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    
    int dimensions = output_shape[1]; // 84 (4 metrics + 80 labels)
    int proposals = output_shape[2];  // 8400 boxes

    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    float x_factor = static_cast<float>(frame.cols) / 640.0f;
    float y_factor = static_cast<float>(frame.rows) / 640.0f;

    // Transpose data parsing manually out of sequential data slices
    for (int i = 0; i < proposals; ++i) {
        float max_score = 0.0f;
        int class_id = -1;

        for (int j = 4; j < dimensions; ++j) {
            float score = raw_output[j * proposals + i];
            if (score > max_score) {
                max_score = score;
                class_id = j - 4;
            }
        }

        if (max_score > 0.45f) {
            float cx = raw_output[0 * proposals + i];
            float cy = raw_output[1 * proposals + i];
            float w  = raw_output[2 * proposals + i];
            float h  = raw_output[3 * proposals + i];

            int left = static_cast<int>((cx - 0.5f * w) * x_factor);
            int top = static_cast<int>((cy - 0.5f * h) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            bboxes.emplace_back(left, top, width, height);
            confidences.push_back(max_score);
            class_ids.push_back(class_id);
        }
    }

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(bboxes, confidences, 0.45f, 0.50f, nms_indices);

    static const std::vector<std::string> coco_labels = {"person", "bicycle", "car", "motorcycle", "bus", "truck"};

    for (int idx : nms_indices) {
        TrackedObject obj;
        obj.box = bboxes[idx];
        obj.track_id = 1000 + idx;
        int id = class_ids[idx];
        obj.label = (id < static_cast<int>(coco_labels.size())) ? coco_labels[id] : "object";
        tracked_items.push_back(obj);
    }

    return tracked_items;
}
