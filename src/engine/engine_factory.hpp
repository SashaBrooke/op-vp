#pragma once
#include <memory>
#include <string>

// Forward declaration of the base class
class BaseInferenceEngine;

enum class EngineType { GPU, CPU, HAILO };

class EngineFactory {
public:
    static std::unique_ptr<BaseInferenceEngine> create_engine(EngineType type, const std::string& model_path = "");
};
