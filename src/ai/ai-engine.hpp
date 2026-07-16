#pragma once

#include "face-parser.hpp"
#include "mask-stabilizer.hpp"
#include "yunet.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace baddiecam::ai {

struct AiConfig {
    std::wstring detector_model;
    std::wstring parser_model;
    InferenceBackend backend = InferenceBackend::Auto;
    int ai_fps = 12;
    float stability = 0.68f;
};

struct AiStatus {
    bool ready = false;
    bool using_directml = false;
    std::string message;
    float inference_ms = 0.0f;
    std::uint64_t processed_frames = 0;
    std::uint64_t dropped_frames = 0;
};

class AiEngine {
public:
    AiEngine();
    ~AiEngine();
    AiEngine(const AiEngine&) = delete;
    AiEngine& operator=(const AiEngine&) = delete;

    void start(AiConfig config);
    void stop();
    void submit(RgbImage frame, std::uint64_t timestamp_ns);
    bool get_latest_masks(RgbaMask& mask, RgbaMask& glam_mask, std::uint64_t& last_sequence) const;
    AiStatus status() const;
    void set_stability(float value) noexcept;

private:
    struct QueuedFrame {
        RgbImage frame;
        std::uint64_t timestamp_ns = 0;
    };

    enum class TrackState { Lost, Reacquire, Tracking, ShortHold };

    void worker_main();
    void process_frame(const QueuedFrame& item);
    std::optional<RectF> select_candidate(const std::vector<RectF>& faces, float& motion);
    void publish_masks(RgbaMask mask, RgbaMask glam_mask);
    void publish_release_mask(int width, int height, float motion);
    void set_status(std::string message, bool ready, bool directml, float inference_ms = 0.0f);

    AiConfig config_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::optional<QueuedFrame> pending_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<float> stability_{0.68f};
    std::thread worker_;

    mutable std::mutex result_mutex_;
    RgbaMask latest_mask_;
    RgbaMask latest_glam_mask_;
    AiStatus status_;

    YuNetDetector detector_;
    FaceParser parser_;
    MaskStabilizer stabilizer_;
    MaskStabilizer glam_stabilizer_;

    TrackState track_state_ = TrackState::Lost;
    RectF tracked_face_{};
    RectF reacquire_face_{};
    int reacquire_count_ = 0;
    int miss_count_ = 0;
    std::uint64_t sequence_ = 0;
};

} // namespace baddiecam::ai
