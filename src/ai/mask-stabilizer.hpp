#pragma once

#include "image.hpp"

namespace baddiecam::ai {

class MaskStabilizer {
public:
    void reset();
    RgbaMask update(const RgbaMask& current, bool trusted_face, float motion, float stability, std::uint64_t sequence);

private:
    RgbaMask previous_;
};

} // namespace baddiecam::ai
