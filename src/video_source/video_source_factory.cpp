#include "video_source_factory.hpp"
#include "base_source.hpp"
#include "file_source.hpp"
#include "camera_source.hpp"

std::unique_ptr<BaseVideoSource> VideoSourceFactory::create_source(SourceType type, const std::string& path, int idx) {
    if (type == SourceType::FILE) {
        return std::make_unique<FileSource>(path);
    } else {
        return std::make_unique<CameraSource>(idx);
    }
}
