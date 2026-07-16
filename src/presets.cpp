#include "presets.hpp"

#include <array>

namespace baddiecam {

static constexpr std::array<BeautyPreset, 5> kPresets = {{
    {"meitu_max", "Meitu Max Pretty", 92, 88, 84, 82, 42, 72, 70, 78, 58, 52, 52, 44, 30, 42, 6, 0},
    {"glass_doll", "Glass Doll Camera", 90, 84, 80, 72, 38, 68, 62, 82, 76, 56, 58, 50, 36, 50, 9, -2},
    {"ulzzang", "Soft Ulzzang Angel", 84, 76, 70, 60, 50, 66, 60, 74, 48, 48, 48, 34, 24, 38, 4, -3},
    {"natural", "Natural Pretty HD", 72, 60, 54, 48, 64, 42, 46, 62, 25, 42, 28, 20, 12, 20, 2, 0},
    {"low_light", "Low-Light Rescue", 86, 82, 74, 78, 44, 62, 72, 88, 36, 38, 46, 28, 18, 48, 2, 3},
}};

const BeautyPreset* find_preset(std::string_view id) noexcept
{
    for (const auto& preset : kPresets)
        if (preset.id == id)
            return &preset;
    return nullptr;
}

void apply_preset(obs_data_t* settings, std::string_view id)
{
    const auto* p = find_preset(id);
    if (!p || !settings)
        return;
    obs_data_set_double(settings, "master_beauty", p->master);
    obs_data_set_double(settings, "skin_smoothing", p->smoothing);
    obs_data_set_double(settings, "pore_refinement", p->pore);
    obs_data_set_double(settings, "blemish_evening", p->blemish);
    obs_data_set_double(settings, "detail_restore", p->detail);
    obs_data_set_double(settings, "complexion_perfect", p->complexion);
    obs_data_set_double(settings, "under_eye_perfect", p->under_eye);
    obs_data_set_double(settings, "shine_control", p->shine);
    obs_data_set_double(settings, "glass_skin", p->glass);
    obs_data_set_double(settings, "eye_hair_crisp", p->crisp);
    obs_data_set_double(settings, "eye_brighten", p->eye_brighten);
    obs_data_set_double(settings, "lip_polish", p->lip_polish);
    obs_data_set_double(settings, "doll_blush", p->doll_blush);
    obs_data_set_double(settings, "facial_light", p->facial_light);
    obs_data_set_double(settings, "rosy_tone", p->rosy);
    obs_data_set_double(settings, "warmth", p->warmth);
}

void add_preset_items(obs_property_t* list)
{
    obs_property_list_add_string(list, "Custom / keep sliders", "custom");
    for (const auto& p : kPresets)
        obs_property_list_add_string(list, p.label.data(), p.id.data());
}

} // namespace baddiecam
