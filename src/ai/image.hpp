#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace baddiecam::ai {

struct RectF {
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    float score = 0.0f;

    [[nodiscard]] float cx() const noexcept { return x + w * 0.5f; }
    [[nodiscard]] float cy() const noexcept { return y + h * 0.5f; }
    [[nodiscard]] float area() const noexcept { return std::max(0.0f, w) * std::max(0.0f, h); }
};

struct RgbImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;

    [[nodiscard]] bool valid() const noexcept {
        return width > 0 && height > 0 && pixels.size() == static_cast<std::size_t>(width * height * 3);
    }
};

struct RgbaMask {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;
    std::uint64_t sequence = 0;

    [[nodiscard]] bool valid() const noexcept {
        return width > 0 && height > 0 && pixels.size() == static_cast<std::size_t>(width * height * 4);
    }
};

struct LetterboxTransform {
    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    int output_size = 0;
};

float clampf(float v, float lo, float hi) noexcept;
float iou(const RectF& a, const RectF& b) noexcept;
RectF clamp_rect(RectF rect, int width, int height) noexcept;
RectF expand_to_square(RectF rect, float width_scale, float height_scale, int width, int height) noexcept;

RgbImage resize_bilinear(const RgbImage& src, int out_width, int out_height);
RgbImage letterbox_square(const RgbImage& src, int output_size, LetterboxTransform& transform);
RgbImage crop_resize(const RgbImage& src, RectF crop, int out_width, int out_height);
RectF undo_letterbox(RectF rect, const LetterboxTransform& transform, int original_width, int original_height) noexcept;

void box_blur_channel(std::vector<std::uint8_t>& rgba, int width, int height, int channel, int radius);
void dilate_channel(std::vector<std::uint8_t>& rgba, int width, int height, int channel, int radius);

} // namespace baddiecam::ai
