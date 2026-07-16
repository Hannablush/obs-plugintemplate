#include <obs-module.h>

#include "beautycore-filter.hpp"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("baddiecam-beautycore-ai", "en-US")

MODULE_EXPORT const char* obs_module_description(void)
{
    return "Automatic face-aware AI beauty filter with no facial geometry warping.";
}

bool obs_module_load(void)
{
    obs_source_info info{};
    info.id = "baddiecam_beautycore_ai_filter";
    info.type = OBS_SOURCE_TYPE_FILTER;
    info.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_ASYNC | OBS_SOURCE_SRGB;
    info.get_name = baddiecam::filter_get_name;
    info.create = baddiecam::filter_create;
    info.destroy = baddiecam::filter_destroy;
    info.update = baddiecam::filter_update;
    info.get_defaults = baddiecam::filter_defaults;
    info.get_properties = baddiecam::filter_properties;
    info.filter_video = baddiecam::filter_video;
    info.video_tick = baddiecam::filter_tick;
    info.video_render = baddiecam::filter_render;
    obs_register_source(&info);
    blog(LOG_INFO, "[BaddieCam BeautyCore AI] module loaded — v2.0.2 async frame feed enabled");
    return true;
}

void obs_module_unload(void)
{
    blog(LOG_INFO, "[BaddieCam BeautyCore AI] module unloaded");
}
