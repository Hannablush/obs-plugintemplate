#include "mask-stabilizer.hpp"

#include <cmath>

namespace baddiecam::ai {

void MaskStabilizer::reset()
{
    previous_ = {};
}

RgbaMask MaskStabilizer::update(const RgbaMask& current, bool trusted_face, float motion, float stability, std::uint64_t sequence)
{
    if (!current.valid()) {
        reset();
        return {};
    }
    if (!previous_.valid() || previous_.width != current.width || previous_.height != current.height) {
        previous_ = current;
        previous_.sequence = sequence;
        return previous_;
    }

    RgbaMask out = current;
    out.sequence = sequence;
    const float s = clampf(stability, 0.0f, 1.0f);
    const float motion_release = clampf(motion * 2.2f, 0.0f, 1.0f);
    const float history = trusted_face ? clampf(0.18f + s * 0.68f - motion_release * 0.48f, 0.08f, 0.86f)
                                       : clampf(0.22f + s * 0.20f, 0.18f, 0.45f);
    for (std::size_t i = 0; i < out.pixels.size(); ++i) {
        const float now = trusted_face ? static_cast<float>(current.pixels[i]) : 0.0f;
        const float old = static_cast<float>(previous_.pixels[i]);
        out.pixels[i] = static_cast<std::uint8_t>(std::lround(old * history + now * (1.0f - history)));
    }
    previous_ = out;
    return out;
}

} // namespace baddiecam::ai
