#pragma once

#include <obs-module.h>
#include <string_view>

namespace baddiecam {

struct BeautyPreset {
    std::string_view id;
    std::string_view label;
    double master;
    double smoothing;
    double pore;
    double blemish;
    double detail;
    double complexion;
    double under_eye;
    double shine;
    double glass;
    double crisp;
    double eye_brighten;
    double lip_polish;
    double doll_blush;
    double facial_light;
    double rosy;
    double warmth;
};

const BeautyPreset* find_preset(std::string_view id) noexcept;
void apply_preset(obs_data_t* settings, std::string_view id);
void add_preset_items(obs_property_t* list);

} // namespace baddiecam
