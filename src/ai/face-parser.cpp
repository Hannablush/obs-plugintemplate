#include "face-parser.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace baddiecam::ai {

bool FaceParser::load(const std::wstring& path, InferenceBackend backend, std::string& error)
{
    return session_.load(path, backend, error);
}

bool FaceParser::parse(const RgbImage& frame, RectF face, int mask_width, int mask_height,
                       RgbaMask& mask, RgbaMask& glam_mask, std::string& error)
{
    mask = {};
    glam_mask = {};
    if (!frame.valid() || face.w < 8.0f || face.h < 8.0f || mask_width <= 0 || mask_height <= 0) {
        error = "Parser received invalid input.";
        return false;
    }
    const RectF crop = expand_to_square(face, 1.48f, 1.72f, frame.width, frame.height);
    const RgbImage input_image = crop_resize(frame, crop, kInputSize, kInputSize);
    if (!input_image.valid()) {
        error = "Unable to create parser crop.";
        return false;
    }

    constexpr std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
    constexpr std::array<float, 3> stdev{0.229f, 0.224f, 0.225f};
    const std::size_t plane = static_cast<std::size_t>(kInputSize * kInputSize);
    std::vector<float> input(plane * 3);
    for (int y = 0; y < kInputSize; ++y) {
        for (int x = 0; x < kInputSize; ++x) {
            const std::size_t src = static_cast<std::size_t>((y * kInputSize + x) * 3);
            const std::size_t dst = static_cast<std::size_t>(y * kInputSize + x);
            for (int c = 0; c < 3; ++c) {
                const float value = static_cast<float>(input_image.pixels[src + c]) / 255.0f;
                input[dst + plane * static_cast<std::size_t>(c)] = (value - mean[c]) / stdev[c];
            }
        }
    }

    std::vector<TensorOutput> outputs;
    if (!session_.run(input, {1, 3, kInputSize, kInputSize}, outputs, error) || outputs.empty())
        return false;
    const TensorOutput& logits = outputs.front();
    if (logits.shape.size() != 4 || logits.shape[1] < 19 || logits.data.size() < 19 * plane) {
        error = "Unexpected face parser output shape.";
        return false;
    }

    std::vector<std::uint8_t> classes(plane, 0);
    for (std::size_t i = 0; i < plane; ++i) {
        int best_class = 0;
        float best_value = logits.data[i];
        for (int c = 1; c < 19; ++c) {
            const float value = logits.data[static_cast<std::size_t>(c) * plane + i];
            if (value > best_value) {
                best_value = value;
                best_class = c;
            }
        }
        classes[i] = static_cast<std::uint8_t>(best_class);
    }

    mask.width = mask_width;
    mask.height = mask_height;
    mask.pixels.assign(static_cast<std::size_t>(mask_width * mask_height * 4), 0);
    glam_mask.width = mask_width;
    glam_mask.height = mask_height;
    glam_mask.pixels.assign(static_cast<std::size_t>(mask_width * mask_height * 4), 0);
    std::vector<std::uint8_t> eye_seed(static_cast<std::size_t>(mask_width * mask_height), 0);

    const float sx = static_cast<float>(mask_width) / static_cast<float>(frame.width);
    const float sy = static_cast<float>(mask_height) / static_cast<float>(frame.height);
    const int x0 = std::max(0, static_cast<int>(std::floor(crop.x * sx)));
    const int y0 = std::max(0, static_cast<int>(std::floor(crop.y * sy)));
    const int x1 = std::min(mask_width, static_cast<int>(std::ceil((crop.x + crop.w) * sx)));
    const int y1 = std::min(mask_height, static_cast<int>(std::ceil((crop.y + crop.h) * sy)));

    std::size_t skin_count = 0;
    std::size_t face_count = 0;
    for (int my = y0; my < y1; ++my) {
        const float fy = (static_cast<float>(my) + 0.5f) / sy;
        const int py = std::clamp(static_cast<int>(((fy - crop.y) / crop.h) * kInputSize), 0, kInputSize - 1);
        for (int mx = x0; mx < x1; ++mx) {
            const float fx = (static_cast<float>(mx) + 0.5f) / sx;
            const int px = std::clamp(static_cast<int>(((fx - crop.x) / crop.w) * kInputSize), 0, kInputSize - 1);
            const int cls = classes[static_cast<std::size_t>(py * kInputSize + px)];
            const std::size_t out = static_cast<std::size_t>((my * mask_width + mx) * 4);
            // Automatic facial zones are normalized to the detected face box, not to a
            // fixed screen position or the expanded parser crop. No user alignment is needed.
            const float nx = clampf((fx - face.x) / std::max(face.w, 1.0f), -0.35f, 1.35f);
            const float ny = clampf((fy - face.y) / std::max(face.h, 1.0f), -0.35f, 1.35f);

            const bool skin = cls == 1;
            const bool ear_skin = cls == 7 || cls == 8;
            const bool protect = cls == 2 || cls == 3 || cls == 4 || cls == 5 || cls == 6 ||
                                 cls == 10 || cls == 11 || cls == 12 || cls == 13 || cls == 17 || cls == 18;
            const bool eye = cls == 4 || cls == 5 || cls == 6;
            const bool trusted_face = (cls >= 1 && cls <= 14) || cls == 17 || cls == 18;

            if (skin || ear_skin) {
                mask.pixels[out + 0] = skin ? 255 : 175;
                ++skin_count;
            }
            if (protect)
                mask.pixels[out + 1] = 255;
            if (eye)
                eye_seed[static_cast<std::size_t>(my * mask_width + mx)] = 255;
            if (trusted_face) {
                mask.pixels[out + 3] = static_cast<std::uint8_t>(std::lround(clampf(face.score, 0.0f, 1.0f) * 255.0f));
                ++face_count;
            }

            // Second tracked mask: R cheeks, G lips, B eyes, A forehead/T-zone.
            if (skin) {
                const auto gaussian = [](float x, float y, float cx, float cy, float sx, float sy) {
                    const float dx = (x - cx) / sx;
                    const float dy = (y - cy) / sy;
                    return std::exp(-(dx * dx + dy * dy) * 1.65f);
                };
                const float cheeks = std::max(gaussian(nx, ny, 0.30f, 0.58f, 0.18f, 0.14f),
                                              gaussian(nx, ny, 0.70f, 0.58f, 0.18f, 0.14f));
                const float forehead = gaussian(nx, ny, 0.50f, 0.29f, 0.25f, 0.17f);
                const float nose_zone = gaussian(nx, ny, 0.50f, 0.50f, 0.11f, 0.28f);
                glam_mask.pixels[out + 0] = static_cast<std::uint8_t>(std::lround(clampf(cheeks, 0.0f, 1.0f) * 255.0f));
                glam_mask.pixels[out + 3] = static_cast<std::uint8_t>(std::lround(clampf(std::max(forehead, nose_zone), 0.0f, 1.0f) * 255.0f));
            }
            if (cls == 12 || cls == 13)
                glam_mask.pixels[out + 1] = 255;
            if (eye)
                glam_mask.pixels[out + 2] = 255;
        }
    }

    const std::size_t crop_area = static_cast<std::size_t>(std::max(1, (x1 - x0) * (y1 - y0)));
    const float skin_fraction = static_cast<float>(skin_count) / static_cast<float>(crop_area);
    const float face_fraction = static_cast<float>(face_count) / static_cast<float>(crop_area);
    if (skin_fraction < 0.035f || skin_fraction > 0.80f || face_fraction < 0.09f) {
        error = "Face parser sanity check rejected the mask.";
        mask = {};
        glam_mask = {};
        return false;
    }

    // Convert eye labels into soft regions immediately below each eye.
    const int down_min = std::max(1, static_cast<int>(std::lround(face.h * sy * 0.025f)));
    const int down_max = std::max(down_min + 1, static_cast<int>(std::lround(face.h * sy * 0.13f)));
    const int side = std::max(1, static_cast<int>(std::lround(face.w * sx * 0.035f)));
    for (int y = 0; y < mask_height; ++y) {
        for (int x = 0; x < mask_width; ++x) {
            if (eye_seed[static_cast<std::size_t>(y * mask_width + x)] == 0)
                continue;
            for (int dy = down_min; dy <= down_max; ++dy) {
                const int yy = y + dy;
                if (yy >= mask_height)
                    break;
                const float falloff = 1.0f - static_cast<float>(dy - down_min) / static_cast<float>(std::max(1, down_max - down_min + 1));
                const std::uint8_t value = static_cast<std::uint8_t>(std::lround(220.0f * falloff));
                for (int dx = -side; dx <= side; ++dx) {
                    const int xx = x + dx;
                    if (xx < 0 || xx >= mask_width)
                        continue;
                    auto& target = mask.pixels[static_cast<std::size_t>((yy * mask_width + xx) * 4 + 2)];
                    target = std::max(target, value);
                }
            }
        }
    }

    dilate_channel(mask.pixels, mask.width, mask.height, 1, 1);
    box_blur_channel(mask.pixels, mask.width, mask.height, 0, 1);
    box_blur_channel(mask.pixels, mask.width, mask.height, 2, 2);
    box_blur_channel(mask.pixels, mask.width, mask.height, 3, 1);
    box_blur_channel(glam_mask.pixels, glam_mask.width, glam_mask.height, 0, 2);
    dilate_channel(glam_mask.pixels, glam_mask.width, glam_mask.height, 1, 1);
    box_blur_channel(glam_mask.pixels, glam_mask.width, glam_mask.height, 1, 1);
    dilate_channel(glam_mask.pixels, glam_mask.width, glam_mask.height, 2, 1);
    box_blur_channel(glam_mask.pixels, glam_mask.width, glam_mask.height, 2, 1);
    box_blur_channel(glam_mask.pixels, glam_mask.width, glam_mask.height, 3, 2);
    glam_mask.sequence = mask.sequence;
    return true;
}

} // namespace baddiecam::ai
