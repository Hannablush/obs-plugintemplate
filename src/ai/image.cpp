#include "image.hpp"

#include <limits>

namespace baddiecam::ai {

float clampf(float v, float lo, float hi) noexcept
{
    return std::max(lo, std::min(hi, v));
}

float iou(const RectF& a, const RectF& b) noexcept
{
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.w, b.x + b.w);
    const float y2 = std::min(a.y + a.h, b.y + b.h);
    const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float union_area = a.area() + b.area() - intersection;
    return union_area > 1e-6f ? intersection / union_area : 0.0f;
}

RectF clamp_rect(RectF rect, int width, int height) noexcept
{
    const float max_x = static_cast<float>(std::max(0, width));
    const float max_y = static_cast<float>(std::max(0, height));
    const float x1 = clampf(rect.x, 0.0f, max_x);
    const float y1 = clampf(rect.y, 0.0f, max_y);
    const float x2 = clampf(rect.x + rect.w, 0.0f, max_x);
    const float y2 = clampf(rect.y + rect.h, 0.0f, max_y);
    rect.x = x1;
    rect.y = y1;
    rect.w = std::max(0.0f, x2 - x1);
    rect.h = std::max(0.0f, y2 - y1);
    return rect;
}

RectF expand_to_square(RectF rect, float width_scale, float height_scale, int width, int height) noexcept
{
    const float cx = rect.cx();
    const float cy = rect.cy() - rect.h * 0.04f;
    const float expanded_w = rect.w * width_scale;
    const float expanded_h = rect.h * height_scale;
    const float side = std::max(expanded_w, expanded_h);
    return clamp_rect({cx - side * 0.5f, cy - side * 0.5f, side, side, rect.score}, width, height);
}

static std::uint8_t bilinear_channel(const RgbImage& src, float x, float y, int c)
{
    x = clampf(x, 0.0f, static_cast<float>(src.width - 1));
    y = clampf(y, 0.0f, static_cast<float>(src.height - 1));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, src.width - 1);
    const int y1 = std::min(y0 + 1, src.height - 1);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const auto at = [&](int px, int py) -> float {
        return static_cast<float>(src.pixels[(py * src.width + px) * 3 + c]);
    };
    const float top = at(x0, y0) * (1.0f - fx) + at(x1, y0) * fx;
    const float bottom = at(x0, y1) * (1.0f - fx) + at(x1, y1) * fx;
    return static_cast<std::uint8_t>(std::lround(top * (1.0f - fy) + bottom * fy));
}

RgbImage resize_bilinear(const RgbImage& src, int out_width, int out_height)
{
    RgbImage out;
    if (!src.valid() || out_width <= 0 || out_height <= 0)
        return out;
    out.width = out_width;
    out.height = out_height;
    out.pixels.resize(static_cast<std::size_t>(out_width * out_height * 3));
    const float sx = static_cast<float>(src.width) / static_cast<float>(out_width);
    const float sy = static_cast<float>(src.height) / static_cast<float>(out_height);
    for (int y = 0; y < out_height; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5f) * sy - 0.5f;
        for (int x = 0; x < out_width; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5f) * sx - 0.5f;
            for (int c = 0; c < 3; ++c)
                out.pixels[(y * out_width + x) * 3 + c] = bilinear_channel(src, src_x, src_y, c);
        }
    }
    return out;
}

RgbImage letterbox_square(const RgbImage& src, int output_size, LetterboxTransform& transform)
{
    RgbImage out;
    if (!src.valid() || output_size <= 0)
        return out;
    transform.output_size = output_size;
    transform.scale = std::min(static_cast<float>(output_size) / static_cast<float>(src.width),
                               static_cast<float>(output_size) / static_cast<float>(src.height));
    const int scaled_w = std::max(1, static_cast<int>(std::lround(src.width * transform.scale)));
    const int scaled_h = std::max(1, static_cast<int>(std::lround(src.height * transform.scale)));
    transform.pad_x = (output_size - scaled_w) / 2;
    transform.pad_y = (output_size - scaled_h) / 2;
    const RgbImage resized = resize_bilinear(src, scaled_w, scaled_h);
    out.width = output_size;
    out.height = output_size;
    out.pixels.assign(static_cast<std::size_t>(output_size * output_size * 3), 0);
    for (int y = 0; y < scaled_h; ++y) {
        for (int x = 0; x < scaled_w; ++x) {
            const std::size_t src_i = static_cast<std::size_t>((y * scaled_w + x) * 3);
            const std::size_t dst_i = static_cast<std::size_t>(((y + transform.pad_y) * output_size + x + transform.pad_x) * 3);
            out.pixels[dst_i + 0] = resized.pixels[src_i + 0];
            out.pixels[dst_i + 1] = resized.pixels[src_i + 1];
            out.pixels[dst_i + 2] = resized.pixels[src_i + 2];
        }
    }
    return out;
}

RgbImage crop_resize(const RgbImage& src, RectF crop, int out_width, int out_height)
{
    RgbImage out;
    if (!src.valid() || out_width <= 0 || out_height <= 0)
        return out;
    crop = clamp_rect(crop, src.width, src.height);
    if (crop.w < 2.0f || crop.h < 2.0f)
        return out;
    out.width = out_width;
    out.height = out_height;
    out.pixels.resize(static_cast<std::size_t>(out_width * out_height * 3));
    for (int y = 0; y < out_height; ++y) {
        const float sy = crop.y + (static_cast<float>(y) + 0.5f) * crop.h / static_cast<float>(out_height) - 0.5f;
        for (int x = 0; x < out_width; ++x) {
            const float sx = crop.x + (static_cast<float>(x) + 0.5f) * crop.w / static_cast<float>(out_width) - 0.5f;
            for (int c = 0; c < 3; ++c)
                out.pixels[(y * out_width + x) * 3 + c] = bilinear_channel(src, sx, sy, c);
        }
    }
    return out;
}

RectF undo_letterbox(RectF rect, const LetterboxTransform& transform, int original_width, int original_height) noexcept
{
    const float inv = transform.scale > 1e-6f ? 1.0f / transform.scale : 1.0f;
    rect.x = (rect.x - static_cast<float>(transform.pad_x)) * inv;
    rect.y = (rect.y - static_cast<float>(transform.pad_y)) * inv;
    rect.w *= inv;
    rect.h *= inv;
    return clamp_rect(rect, original_width, original_height);
}

void box_blur_channel(std::vector<std::uint8_t>& rgba, int width, int height, int channel, int radius)
{
    if (width <= 0 || height <= 0 || channel < 0 || channel > 3 || radius <= 0 ||
        rgba.size() != static_cast<std::size_t>(width * height * 4))
        return;
    std::vector<std::uint8_t> copy = rgba;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            int count = 0;
            for (int yy = std::max(0, y - radius); yy <= std::min(height - 1, y + radius); ++yy) {
                for (int xx = std::max(0, x - radius); xx <= std::min(width - 1, x + radius); ++xx) {
                    sum += copy[(yy * width + xx) * 4 + channel];
                    ++count;
                }
            }
            rgba[(y * width + x) * 4 + channel] = static_cast<std::uint8_t>(sum / std::max(1, count));
        }
    }
}

void dilate_channel(std::vector<std::uint8_t>& rgba, int width, int height, int channel, int radius)
{
    if (width <= 0 || height <= 0 || channel < 0 || channel > 3 || radius <= 0 ||
        rgba.size() != static_cast<std::size_t>(width * height * 4))
        return;
    std::vector<std::uint8_t> copy = rgba;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::uint8_t best = 0;
            for (int yy = std::max(0, y - radius); yy <= std::min(height - 1, y + radius); ++yy) {
                for (int xx = std::max(0, x - radius); xx <= std::min(width - 1, x + radius); ++xx)
                    best = std::max(best, copy[(yy * width + xx) * 4 + channel]);
            }
            rgba[(y * width + x) * 4 + channel] = best;
        }
    }
}

} // namespace baddiecam::ai
