#include "beautycore-filter.hpp"

#include "ai/ai-engine.hpp"
#include "presets.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

namespace baddiecam {
namespace {

using Clock = std::chrono::steady_clock;

struct RenderSettings {
    float master = 0.92f;
    float smoothing = 0.88f;
    float pore = 0.84f;
    float blemish = 0.82f;
    float detail = 0.42f;
    float complexion = 0.72f;
    float under_eye = 0.70f;
    float shine = 0.78f;
    float glass = 0.58f;
    float crisp = 0.52f;
    float eye_brighten = 0.52f;
    float lip_polish = 0.44f;
    float doll_blush = 0.30f;
    float facial_light = 0.42f;
    float rosy = 0.06f;
    float warmth = 0.0f;
    float mask_strength = 1.0f;
    int quality = 1;
    int preview = 0;
};

struct FilterData {
    obs_source_t* context = nullptr;
    gs_effect_t* effect = nullptr;
    gs_texture_t* mask_texture = nullptr;
    gs_texture_t* glam_texture = nullptr;
    gs_eparam_t* p_mask = nullptr;
    gs_eparam_t* p_glam_mask = nullptr;
    gs_eparam_t* p_texel = nullptr;
    gs_eparam_t* p_master = nullptr;
    gs_eparam_t* p_smoothing = nullptr;
    gs_eparam_t* p_pore = nullptr;
    gs_eparam_t* p_blemish = nullptr;
    gs_eparam_t* p_detail = nullptr;
    gs_eparam_t* p_complexion = nullptr;
    gs_eparam_t* p_under_eye = nullptr;
    gs_eparam_t* p_shine = nullptr;
    gs_eparam_t* p_glass = nullptr;
    gs_eparam_t* p_crisp = nullptr;
    gs_eparam_t* p_eye_brighten = nullptr;
    gs_eparam_t* p_lip_polish = nullptr;
    gs_eparam_t* p_doll_blush = nullptr;
    gs_eparam_t* p_facial_light = nullptr;
    gs_eparam_t* p_rosy = nullptr;
    gs_eparam_t* p_warmth = nullptr;
    gs_eparam_t* p_mask_strength = nullptr;
    gs_eparam_t* p_quality = nullptr;
    gs_eparam_t* p_preview = nullptr;

    std::mutex settings_mutex;
    RenderSettings settings;

    ai::AiEngine ai;
    ai::InferenceBackend backend = ai::InferenceBackend::Auto;
    std::string backend_key;
    std::atomic<std::int64_t> ai_interval_ns{83'333'333};
    std::atomic<std::int64_t> last_submit_ns{0};
    std::uint64_t uploaded_sequence = 0;
};

static std::wstring utf8_to_wide(const char* text)
{
    if (!text)
        return {};
#ifdef _WIN32
    const int count = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
    if (count <= 1)
        return {};
    std::wstring out(static_cast<std::size_t>(count), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text, -1, out.data(), count);
    if (!out.empty() && out.back() == L'\0')
        out.pop_back();
    return out;
#else
    std::wstring out;
    while (*text)
        out.push_back(static_cast<unsigned char>(*text++));
    return out;
#endif
}

static std::int64_t now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

static std::uint8_t clamp_byte(float value)
{
    return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0f, 255.0f)));
}

static void yuv_to_rgb(int y, int u, int v, std::uint8_t& r, std::uint8_t& g, std::uint8_t& b)
{
    const float yf = 1.164383f * static_cast<float>(y - 16);
    const float uf = static_cast<float>(u - 128);
    const float vf = static_cast<float>(v - 128);
    r = clamp_byte(yf + 1.792741f * vf);
    g = clamp_byte(yf - 0.213249f * uf - 0.532909f * vf);
    b = clamp_byte(yf + 2.112402f * uf);
}

static bool sample_rgb(const obs_source_frame* frame, int x, int y, std::uint8_t& r, std::uint8_t& g, std::uint8_t& b)
{
    if (!frame || frame->width == 0 || frame->height == 0 || !frame->data[0])
        return false;
    x = std::clamp(x, 0, static_cast<int>(frame->width) - 1);
    y = std::clamp(y, 0, static_cast<int>(frame->height) - 1);
    switch (frame->format) {
    case VIDEO_FORMAT_BGRA:
    case VIDEO_FORMAT_BGRX: {
        const std::uint8_t* p = frame->data[0] + static_cast<std::size_t>(y) * frame->linesize[0] + static_cast<std::size_t>(x) * 4;
        b = p[0]; g = p[1]; r = p[2];
        return true;
    }
    case VIDEO_FORMAT_RGBA: {
        const std::uint8_t* p = frame->data[0] + static_cast<std::size_t>(y) * frame->linesize[0] + static_cast<std::size_t>(x) * 4;
        r = p[0]; g = p[1]; b = p[2];
        return true;
    }
    case VIDEO_FORMAT_BGR3: {
        const std::uint8_t* p = frame->data[0] + static_cast<std::size_t>(y) * frame->linesize[0] + static_cast<std::size_t>(x) * 3;
        b = p[0]; g = p[1]; r = p[2];
        return true;
    }
    case VIDEO_FORMAT_NV12: {
        if (!frame->data[1])
            return false;
        const int yy = frame->data[0][static_cast<std::size_t>(y) * frame->linesize[0] + x];
        const std::uint8_t* uv = frame->data[1] + static_cast<std::size_t>(y / 2) * frame->linesize[1] + static_cast<std::size_t>(x / 2) * 2;
        yuv_to_rgb(yy, uv[0], uv[1], r, g, b);
        return true;
    }
    case VIDEO_FORMAT_I420: {
        if (!frame->data[1] || !frame->data[2])
            return false;
        const int yy = frame->data[0][static_cast<std::size_t>(y) * frame->linesize[0] + x];
        const int uu = frame->data[1][static_cast<std::size_t>(y / 2) * frame->linesize[1] + x / 2];
        const int vv = frame->data[2][static_cast<std::size_t>(y / 2) * frame->linesize[2] + x / 2];
        yuv_to_rgb(yy, uu, vv, r, g, b);
        return true;
    }
    case VIDEO_FORMAT_YUY2: {
        const std::uint8_t* p = frame->data[0] + static_cast<std::size_t>(y) * frame->linesize[0] + static_cast<std::size_t>(x / 2) * 4;
        const int yy = p[(x & 1) ? 2 : 0];
        yuv_to_rgb(yy, p[1], p[3], r, g, b);
        return true;
    }
    default:
        return false;
    }
}

static ai::RgbImage make_ai_frame(const obs_source_frame* frame)
{
    ai::RgbImage out;
    if (!frame || frame->width < 16 || frame->height < 16)
        return out;
    out.width = std::min(640, static_cast<int>(frame->width));
    out.height = std::max(1, static_cast<int>(std::lround(static_cast<double>(out.width) * frame->height / frame->width)));
    out.pixels.resize(static_cast<std::size_t>(out.width * out.height * 3));
    for (int y = 0; y < out.height; ++y) {
        const int sy = std::min(static_cast<int>(frame->height) - 1,
                                static_cast<int>((static_cast<std::int64_t>(y) * frame->height) / out.height));
        for (int x = 0; x < out.width; ++x) {
            const int sx = std::min(static_cast<int>(frame->width) - 1,
                                    static_cast<int>((static_cast<std::int64_t>(x) * frame->width) / out.width));
            std::uint8_t r = 0, g = 0, b = 0;
            if (!sample_rgb(frame, sx, sy, r, g, b))
                return {};
            const std::size_t i = static_cast<std::size_t>((y * out.width + x) * 3);
            out.pixels[i + 0] = r;
            out.pixels[i + 1] = g;
            out.pixels[i + 2] = b;
        }
    }
    return out;
}

static ai::InferenceBackend parse_backend(const char* value)
{
    if (value && std::strcmp(value, "directml") == 0)
        return ai::InferenceBackend::DirectML;
    if (value && std::strcmp(value, "cpu") == 0)
        return ai::InferenceBackend::CPU;
    return ai::InferenceBackend::Auto;
}

static void start_ai(FilterData* filter, const std::string& backend_key, ai::InferenceBackend backend, float stability)
{
    char* detector_path = obs_module_file("models/face_detection_yunet_2023mar.onnx");
    char* parser_path = obs_module_file("models/face_parsing_resnet18.onnx");
    ai::AiConfig config;
    config.detector_model = utf8_to_wide(detector_path);
    config.parser_model = utf8_to_wide(parser_path);
    config.backend = backend;
    config.stability = stability;
    bfree(detector_path);
    bfree(parser_path);
    filter->backend_key = backend_key;
    filter->backend = backend;
    filter->ai.start(std::move(config));
}

static void load_effect_params(FilterData* f)
{
    f->p_mask = gs_effect_get_param_by_name(f->effect, "mask_image");
    f->p_glam_mask = gs_effect_get_param_by_name(f->effect, "glam_image");
    f->p_texel = gs_effect_get_param_by_name(f->effect, "texel_step");
    f->p_master = gs_effect_get_param_by_name(f->effect, "master_beauty");
    f->p_smoothing = gs_effect_get_param_by_name(f->effect, "skin_smoothing");
    f->p_pore = gs_effect_get_param_by_name(f->effect, "pore_refinement");
    f->p_blemish = gs_effect_get_param_by_name(f->effect, "blemish_evening");
    f->p_detail = gs_effect_get_param_by_name(f->effect, "detail_restore");
    f->p_complexion = gs_effect_get_param_by_name(f->effect, "complexion_perfect");
    f->p_under_eye = gs_effect_get_param_by_name(f->effect, "under_eye_perfect");
    f->p_shine = gs_effect_get_param_by_name(f->effect, "shine_control");
    f->p_glass = gs_effect_get_param_by_name(f->effect, "glass_skin");
    f->p_crisp = gs_effect_get_param_by_name(f->effect, "eye_hair_crisp");
    f->p_eye_brighten = gs_effect_get_param_by_name(f->effect, "eye_brighten");
    f->p_lip_polish = gs_effect_get_param_by_name(f->effect, "lip_polish");
    f->p_doll_blush = gs_effect_get_param_by_name(f->effect, "doll_blush");
    f->p_facial_light = gs_effect_get_param_by_name(f->effect, "facial_light");
    f->p_rosy = gs_effect_get_param_by_name(f->effect, "rosy_tone");
    f->p_warmth = gs_effect_get_param_by_name(f->effect, "warmth");
    f->p_mask_strength = gs_effect_get_param_by_name(f->effect, "mask_strength");
    f->p_quality = gs_effect_get_param_by_name(f->effect, "quality_mode");
    f->p_preview = gs_effect_get_param_by_name(f->effect, "preview_mode");
}

static bool preset_changed(obs_properties_t*, obs_property_t*, obs_data_t* settings)
{
    const char* preset = obs_data_get_string(settings, "preset");
    if (preset && std::strcmp(preset, "custom") != 0)
        apply_preset(settings, preset);
    return true;
}

static obs_properties_t* make_beauty_group()
{
    obs_properties_t* group = obs_properties_create();
    obs_properties_add_float_slider(group, "master_beauty", "1. MASTER BEAUTY", 0, 100, 1);
    obs_properties_add_float_slider(group, "skin_smoothing", "2. SKIN SMOOTHING", 0, 100, 1);
    obs_properties_add_float_slider(group, "pore_refinement", "3. PORE REFINEMENT", 0, 100, 1);
    obs_properties_add_float_slider(group, "blemish_evening", "4. BLEMISH + REDNESS EVENING", 0, 100, 1);
    obs_properties_add_float_slider(group, "detail_restore", "5. REAL SKIN DETAIL", 0, 100, 1);
    obs_properties_add_float_slider(group, "complexion_perfect", "6. COMPLEXION PERFECT", 0, 100, 1);
    obs_properties_add_float_slider(group, "under_eye_perfect", "7. UNDER-EYE PERFECT", 0, 100, 1);
    obs_properties_add_float_slider(group, "shine_control", "8. FOREHEAD / NOSE SHINE CONTROL", 0, 100, 1);
    obs_properties_add_float_slider(group, "glass_skin", "9. GLASS SKIN FINISH", 0, 100, 1);
    obs_properties_add_float_slider(group, "eye_hair_crisp", "10. EYES / LASHES / HAIR CRISP", 0, 100, 1);
    obs_properties_add_float_slider(group, "eye_brighten", "11. AUTOMATIC EYE BRIGHTEN", 0, 100, 1);
    obs_properties_add_float_slider(group, "lip_polish", "12. AUTOMATIC LIP POLISH", 0, 100, 1);
    obs_properties_add_float_slider(group, "doll_blush", "13. AUTOMATIC DOLL BLUSH", 0, 100, 1);
    obs_properties_add_float_slider(group, "facial_light", "14. FOREHEAD / T-ZONE BEAUTY LIGHT", 0, 100, 1);
    obs_properties_add_float_slider(group, "rosy_tone", "15. ROSY TONE", -100, 100, 1);
    obs_properties_add_float_slider(group, "warmth", "16. SKIN WARMTH", -100, 100, 1);
    return group;
}

static void update_mask_texture(gs_texture_t*& texture, const ai::RgbaMask& mask)
{
    const std::uint8_t* planes[] = {mask.pixels.data()};
    if (!texture || gs_texture_get_width(texture) != static_cast<std::uint32_t>(mask.width) ||
        gs_texture_get_height(texture) != static_cast<std::uint32_t>(mask.height)) {
        gs_texture_destroy(texture);
        texture = gs_texture_create(static_cast<std::uint32_t>(mask.width), static_cast<std::uint32_t>(mask.height),
                                    GS_RGBA, 1, planes, GS_DYNAMIC);
    } else {
        gs_texture_set_image(texture, mask.pixels.data(), static_cast<std::uint32_t>(mask.width * 4), false);
    }
}

static void upload_latest_masks(FilterData* filter)
{
    ai::RgbaMask mask;
    ai::RgbaMask glam;
    if (!filter->ai.get_latest_masks(mask, glam, filter->uploaded_sequence))
        return;
    update_mask_texture(filter->mask_texture, mask);
    update_mask_texture(filter->glam_texture, glam);
}

} // namespace

const char* filter_get_name(void*)
{
    return "BaddieCam BeautyCore AI";
}

void* filter_create(obs_data_t* settings, obs_source_t* source)
{
    auto* filter = new FilterData;
    filter->context = source;
    char* effect_path = obs_module_file("effects/beautycore_ai.effect");
    char* error = nullptr;
    obs_enter_graphics();
    filter->effect = gs_effect_create_from_file(effect_path, &error);
    if (filter->effect) {
        const std::uint8_t zero[4] = {0, 0, 0, 0};
        const std::uint8_t* planes[] = {zero};
        filter->mask_texture = gs_texture_create(1, 1, GS_RGBA, 1, planes, GS_DYNAMIC);
        filter->glam_texture = gs_texture_create(1, 1, GS_RGBA, 1, planes, GS_DYNAMIC);
        load_effect_params(filter);
    }
    obs_leave_graphics();
    bfree(effect_path);
    if (error) {
        blog(LOG_ERROR, "[BaddieCam BeautyCore AI] effect error: %s", error);
        bfree(error);
    }
    if (!filter->effect) {
        delete filter;
        return nullptr;
    }
    filter_update(filter, settings);
    return filter;
}

void filter_destroy(void* data)
{
    auto* filter = static_cast<FilterData*>(data);
    if (!filter)
        return;
    filter->ai.stop();
    obs_enter_graphics();
    gs_texture_destroy(filter->mask_texture);
    gs_texture_destroy(filter->glam_texture);
    gs_effect_destroy(filter->effect);
    obs_leave_graphics();
    delete filter;
}

void filter_update(void* data, obs_data_t* settings)
{
    auto* filter = static_cast<FilterData*>(data);
    if (!filter || !settings)
        return;
    RenderSettings render;
    render.master = static_cast<float>(obs_data_get_double(settings, "master_beauty") / 100.0);
    render.smoothing = static_cast<float>(obs_data_get_double(settings, "skin_smoothing") / 100.0);
    render.pore = static_cast<float>(obs_data_get_double(settings, "pore_refinement") / 100.0);
    render.blemish = static_cast<float>(obs_data_get_double(settings, "blemish_evening") / 100.0);
    render.detail = static_cast<float>(obs_data_get_double(settings, "detail_restore") / 100.0);
    render.complexion = static_cast<float>(obs_data_get_double(settings, "complexion_perfect") / 100.0);
    render.under_eye = static_cast<float>(obs_data_get_double(settings, "under_eye_perfect") / 100.0);
    render.shine = static_cast<float>(obs_data_get_double(settings, "shine_control") / 100.0);
    render.glass = static_cast<float>(obs_data_get_double(settings, "glass_skin") / 100.0);
    render.crisp = static_cast<float>(obs_data_get_double(settings, "eye_hair_crisp") / 100.0);
    render.eye_brighten = static_cast<float>(obs_data_get_double(settings, "eye_brighten") / 100.0);
    render.lip_polish = static_cast<float>(obs_data_get_double(settings, "lip_polish") / 100.0);
    render.doll_blush = static_cast<float>(obs_data_get_double(settings, "doll_blush") / 100.0);
    render.facial_light = static_cast<float>(obs_data_get_double(settings, "facial_light") / 100.0);
    render.rosy = static_cast<float>(obs_data_get_double(settings, "rosy_tone") / 100.0);
    render.warmth = static_cast<float>(obs_data_get_double(settings, "warmth") / 100.0);
    render.mask_strength = static_cast<float>(obs_data_get_double(settings, "mask_strength") / 100.0);
    render.quality = static_cast<int>(obs_data_get_int(settings, "quality_mode"));
    render.preview = static_cast<int>(obs_data_get_int(settings, "preview_mode"));
    {
        std::scoped_lock lock(filter->settings_mutex);
        filter->settings = render;
    }

    const int ai_fps = std::clamp(static_cast<int>(obs_data_get_int(settings, "ai_fps")), 5, 20);
    filter->ai_interval_ns.store(1'000'000'000LL / ai_fps);
    const float stability = static_cast<float>(obs_data_get_double(settings, "mask_stability") / 100.0);
    filter->ai.set_stability(stability);

    const char* backend_text = obs_data_get_string(settings, "ai_backend");
    const std::string backend_key = backend_text ? backend_text : "auto";
    const auto backend = parse_backend(backend_text);
    if (filter->backend_key != backend_key)
        start_ai(filter, backend_key, backend, stability);
}

void filter_defaults(obs_data_t* settings)
{
    obs_data_set_default_string(settings, "preset", "meitu_max");
    obs_data_set_default_double(settings, "master_beauty", 92);
    obs_data_set_default_double(settings, "skin_smoothing", 88);
    obs_data_set_default_double(settings, "pore_refinement", 84);
    obs_data_set_default_double(settings, "blemish_evening", 82);
    obs_data_set_default_double(settings, "detail_restore", 42);
    obs_data_set_default_double(settings, "complexion_perfect", 72);
    obs_data_set_default_double(settings, "under_eye_perfect", 70);
    obs_data_set_default_double(settings, "shine_control", 78);
    obs_data_set_default_double(settings, "glass_skin", 58);
    obs_data_set_default_double(settings, "eye_hair_crisp", 52);
    obs_data_set_default_double(settings, "eye_brighten", 52);
    obs_data_set_default_double(settings, "lip_polish", 44);
    obs_data_set_default_double(settings, "doll_blush", 30);
    obs_data_set_default_double(settings, "facial_light", 42);
    obs_data_set_default_double(settings, "rosy_tone", 6);
    obs_data_set_default_double(settings, "warmth", 0);
    obs_data_set_default_double(settings, "mask_strength", 100);
    obs_data_set_default_int(settings, "quality_mode", 1);
    obs_data_set_default_int(settings, "preview_mode", 0);
    obs_data_set_default_string(settings, "ai_backend", "auto");
    obs_data_set_default_int(settings, "ai_fps", 12);
    obs_data_set_default_double(settings, "mask_stability", 68);
}

obs_properties_t* filter_properties(void* data)
{
    auto* filter = static_cast<FilterData*>(data);
    obs_properties_t* props = obs_properties_create();
    obs_properties_add_text(props, "about_ai",
        "AUTOMATIC AI BEAUTY — NO FACE WARPING, NO MANUAL FACE POSITION",
        OBS_TEXT_INFO);

    obs_property_t* preset = obs_properties_add_list(props, "preset", "QUICK BEAUTY LOOK",
        OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
    add_preset_items(preset);
    obs_property_set_modified_callback(preset, preset_changed);

    obs_properties_add_group(props, "beauty_controls", "BEAUTY CONTROLS",
                             OBS_GROUP_NORMAL, make_beauty_group());

    obs_properties_t* ai_group = obs_properties_create();
    obs_property_t* backend = obs_properties_add_list(ai_group, "ai_backend", "AI BACKEND",
        OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
    obs_property_list_add_string(backend, "Auto — DirectML then CPU", "auto");
    obs_property_list_add_string(backend, "DirectML GPU", "directml");
    obs_property_list_add_string(backend, "CPU", "cpu");
    obs_properties_add_int_slider(ai_group, "ai_fps", "AI MASK FPS", 5, 20, 1);
    obs_properties_add_float_slider(ai_group, "mask_stability", "MASK STABILITY", 0, 100, 1);
    obs_properties_add_float_slider(ai_group, "mask_strength", "AI MASK STRENGTH", 0, 100, 1);
    obs_property_t* quality = obs_properties_add_list(ai_group, "quality_mode", "GPU BEAUTY QUALITY",
        OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(quality, "Performance", 0);
    obs_property_list_add_int(quality, "Balanced", 1);
    obs_property_list_add_int(quality, "Ultra", 2);
    obs_property_t* preview = obs_properties_add_list(ai_group, "preview_mode", "CALIBRATION PREVIEW",
        OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(preview, "Final Beauty", 0);
    obs_property_list_add_int(preview, "Skin Mask", 1);
    obs_property_list_add_int(preview, "Protection Mask", 2);
    obs_property_list_add_int(preview, "Under-Eye Mask", 3);
    obs_property_list_add_int(preview, "AI Confidence", 4);
    obs_property_list_add_int(preview, "Automatic Glam Zones", 5);

    if (filter) {
        const auto status = filter->ai.status();
        const std::string label = "AI STATUS: " + status.message +
            " | inference " + std::to_string(static_cast<int>(std::lround(status.inference_ms))) + " ms";
        obs_properties_add_text(ai_group, "ai_runtime_status", label.c_str(), OBS_TEXT_INFO);
    }
    obs_properties_add_group(props, "ai_controls", "AUTOMATIC AI MASK ENGINE",
                             OBS_GROUP_NORMAL, ai_group);
    return props;
}

obs_source_frame* filter_video(void* data, obs_source_frame* frame)
{
    auto* filter = static_cast<FilterData*>(data);
    if (!filter || !frame)
        return frame;
    const std::int64_t now = now_ns();
    std::int64_t last = filter->last_submit_ns.load();
    if (now - last < filter->ai_interval_ns.load())
        return frame;
    if (!filter->last_submit_ns.compare_exchange_strong(last, now))
        return frame;
    ai::RgbImage ai_frame = make_ai_frame(frame);
    if (ai_frame.valid())
        filter->ai.submit(std::move(ai_frame), frame->timestamp);
    return frame;
}

void filter_render(void* data, gs_effect_t*)
{
    auto* filter = static_cast<FilterData*>(data);
    if (!filter || !filter->effect) {
        return;
    }
    obs_source_t* target = obs_filter_get_target(filter->context);
    const std::uint32_t width = target ? obs_source_get_base_width(target) : 0;
    const std::uint32_t height = target ? obs_source_get_base_height(target) : 0;
    if (!width || !height) {
        obs_source_skip_video_filter(filter->context);
        return;
    }

    if (!obs_source_process_filter_begin(filter->context, GS_RGBA, OBS_NO_DIRECT_RENDERING))
        return;

    upload_latest_masks(filter);
    RenderSettings s;
    {
        std::scoped_lock lock(filter->settings_mutex);
        s = filter->settings;
    }
    vec2 texel;
    vec2_set(&texel, 1.0f / static_cast<float>(width), 1.0f / static_cast<float>(height));
    gs_effect_set_texture(filter->p_mask, filter->mask_texture);
    gs_effect_set_texture(filter->p_glam_mask, filter->glam_texture);
    gs_effect_set_vec2(filter->p_texel, &texel);
    gs_effect_set_float(filter->p_master, s.master);
    gs_effect_set_float(filter->p_smoothing, s.smoothing);
    gs_effect_set_float(filter->p_pore, s.pore);
    gs_effect_set_float(filter->p_blemish, s.blemish);
    gs_effect_set_float(filter->p_detail, s.detail);
    gs_effect_set_float(filter->p_complexion, s.complexion);
    gs_effect_set_float(filter->p_under_eye, s.under_eye);
    gs_effect_set_float(filter->p_shine, s.shine);
    gs_effect_set_float(filter->p_glass, s.glass);
    gs_effect_set_float(filter->p_crisp, s.crisp);
    gs_effect_set_float(filter->p_eye_brighten, s.eye_brighten);
    gs_effect_set_float(filter->p_lip_polish, s.lip_polish);
    gs_effect_set_float(filter->p_doll_blush, s.doll_blush);
    gs_effect_set_float(filter->p_facial_light, s.facial_light);
    gs_effect_set_float(filter->p_rosy, s.rosy);
    gs_effect_set_float(filter->p_warmth, s.warmth);
    gs_effect_set_float(filter->p_mask_strength, s.mask_strength);
    gs_effect_set_int(filter->p_quality, s.quality);
    gs_effect_set_int(filter->p_preview, s.preview);
    obs_source_process_filter_end(filter->context, filter->effect, width, height);
}

} // namespace baddiecam
