#pragma once

#include "image.hpp"
#include "ort-session.hpp"

#include <string>
#include <vector>

namespace baddiecam::ai {

class YuNetDetector {
public:
    bool load(const std::wstring& path, InferenceBackend backend, std::string& error);
    bool detect(const RgbImage& frame, std::vector<RectF>& faces, std::string& error);
    [[nodiscard]] bool using_directml() const noexcept { return session_.using_directml(); }

private:
    static constexpr int kInputSize = 640;
    OrtSession session_;
    float confidence_threshold_ = 0.74f;
    float nms_threshold_ = 0.30f;
};

} // namespace baddiecam::ai
