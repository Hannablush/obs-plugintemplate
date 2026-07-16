#pragma once

#include "image.hpp"
#include "ort-session.hpp"

#include <string>

namespace baddiecam::ai {

class FaceParser {
public:
    bool load(const std::wstring& path, InferenceBackend backend, std::string& error);
    bool parse(const RgbImage& frame, RectF face, int mask_width, int mask_height,
               RgbaMask& mask, RgbaMask& glam_mask, std::string& error);
    [[nodiscard]] bool using_directml() const noexcept { return session_.using_directml(); }

private:
    static constexpr int kInputSize = 512;
    OrtSession session_;
};

} // namespace baddiecam::ai
