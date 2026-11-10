#pragma once
#include <memory>
#include <string>

// Forward declaration of the base class
class BaseVideoSource;

enum class SourceType { FILE, CAMERA };

class VideoSourceFactory {
public:
    static std::unique_ptr<BaseVideoSource> create_source(SourceType type, const std::string& path = "", int idx = 0);
};
