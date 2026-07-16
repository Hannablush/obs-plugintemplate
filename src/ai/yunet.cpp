#include "yunet.hpp"

#include <algorithm>
#include <cmath>
#include <map>

namespace baddiecam::ai {

bool YuNetDetector::load(const std::wstring& path, InferenceBackend backend, std::string& error)
{
    return session_.load(path, backend, error);
}

static const TensorOutput* find_output(const std::vector<TensorOutput>& outputs, const std::string& name)
{
    for (const auto& out : outputs)
        if (out.name == name)
            return &out;
    return nullptr;
}

bool YuNetDetector::detect(const RgbImage& frame, std::vector<RectF>& faces, std::string& error)
{
    faces.clear();
    if (!frame.valid()) {
        error = "Detector received an invalid frame.";
        return false;
    }
    LetterboxTransform transform;
    const RgbImage input_image = letterbox_square(frame, kInputSize, transform);
    std::vector<float> input(static_cast<std::size_t>(3 * kInputSize * kInputSize));
    const std::size_t plane = static_cast<std::size_t>(kInputSize * kInputSize);
    for (int y = 0; y < kInputSize; ++y) {
        for (int x = 0; x < kInputSize; ++x) {
            const std::size_t src = static_cast<std::size_t>((y * kInputSize + x) * 3);
            const std::size_t dst = static_cast<std::size_t>(y * kInputSize + x);
            // YuNet's OpenCV preprocessing is BGR with scale 1.0.
            input[dst + plane * 0] = static_cast<float>(input_image.pixels[src + 2]);
            input[dst + plane * 1] = static_cast<float>(input_image.pixels[src + 1]);
            input[dst + plane * 2] = static_cast<float>(input_image.pixels[src + 0]);
        }
    }

    std::vector<TensorOutput> outputs;
    if (!session_.run(input, {1, 3, kInputSize, kInputSize}, outputs, error))
        return false;

    std::vector<RectF> candidates;
    for (const int stride : {8, 16, 32}) {
        const auto* cls = find_output(outputs, "cls_" + std::to_string(stride));
        const auto* obj = find_output(outputs, "obj_" + std::to_string(stride));
        const auto* bbox = find_output(outputs, "bbox_" + std::to_string(stride));
        if (!cls || !obj || !bbox || bbox->data.size() % 4 != 0)
            continue;
        const std::size_t count = bbox->data.size() / 4;
        if (cls->data.size() < count || obj->data.size() < count)
            continue;
        const int grid = static_cast<int>(std::lround(std::sqrt(static_cast<double>(count))));
        if (grid <= 0 || static_cast<std::size_t>(grid * grid) != count)
            continue;
        for (std::size_t i = 0; i < count; ++i) {
            const float score = std::sqrt(std::max(0.0f, cls->data[i]) * std::max(0.0f, obj->data[i]));
            if (score < confidence_threshold_)
                continue;
            const int row = static_cast<int>(i) / grid;
            const int col = static_cast<int>(i) % grid;
            const float dx = bbox->data[i * 4 + 0];
            const float dy = bbox->data[i * 4 + 1];
            const float dw = std::min(8.0f, bbox->data[i * 4 + 2]);
            const float dh = std::min(8.0f, bbox->data[i * 4 + 3]);
            const float cx = (static_cast<float>(col) + dx) * static_cast<float>(stride);
            const float cy = (static_cast<float>(row) + dy) * static_cast<float>(stride);
            const float w = std::exp(dw) * static_cast<float>(stride);
            const float h = std::exp(dh) * static_cast<float>(stride);
            RectF rect{cx - w * 0.5f, cy - h * 0.5f, w, h, score};
            rect = undo_letterbox(rect, transform, frame.width, frame.height);
            if (rect.w >= 18.0f && rect.h >= 18.0f)
                candidates.push_back(rect);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const RectF& a, const RectF& b) { return a.score > b.score; });
    std::vector<bool> suppressed(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i])
            continue;
        faces.push_back(candidates[i]);
        for (std::size_t j = i + 1; j < candidates.size(); ++j)
            if (!suppressed[j] && iou(candidates[i], candidates[j]) >= nms_threshold_)
                suppressed[j] = true;
        if (faces.size() >= 8)
            break;
    }
    return true;
}

} // namespace baddiecam::ai
