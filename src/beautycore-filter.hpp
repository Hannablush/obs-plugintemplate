#pragma once

#include <obs-module.h>

namespace baddiecam {

const char* filter_get_name(void* unused);
void* filter_create(obs_data_t* settings, obs_source_t* source);
void filter_destroy(void* data);
void filter_update(void* data, obs_data_t* settings);
void filter_defaults(obs_data_t* settings);
obs_properties_t* filter_properties(void* data);
obs_source_frame* filter_video(void* data, obs_source_frame* frame);
void filter_tick(void* data, float seconds);
void filter_render(void* data, gs_effect_t* effect);

} // namespace baddiecam
