#include "engine_factory.hpp"
#include "base_engine.hpp"
#include "cpu_engine.hpp"
#include "gpu_engine.hpp"
#include "hailo_engine.hpp"
#include <stdexcept>

std::unique_ptr<BaseInferenceEngine> EngineFactory::create_engine(EngineType type, const std::string& model_path) {
    switch (type) {
        case EngineType::GPU:   return std::make_unique<GPUEngine>(model_path);
        case EngineType::CPU:   return std::make_unique<CPUEngine>(model_path);
        case EngineType::HAILO: return std::make_unique<HailoEngine>(model_path);
        default: throw std::invalid_argument("Unsupported inference engine backend.");
    }
}
