#include "ai-engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <utility>

namespace baddiecam::ai {

AiEngine::AiEngine() = default;
AiEngine::~AiEngine() { stop(); }

void AiEngine::start(AiConfig config)
{
    stop();
    config_ = std::move(config);
    stability_.store(clampf(config_.stability, 0.0f, 1.0f));
    stop_requested_.store(false);
    {
        std::scoped_lock lock(result_mutex_);
        latest_mask_ = {};
        latest_glam_mask_ = {};
        status_ = {};
        status_.message = "Loading local AI models...";
    }
    track_state_ = TrackState::Lost;
    tracked_face_ = {};
    reacquire_face_ = {};
    reacquire_count_ = 0;
    miss_count_ = 0;
    sequence_ = 0;
    stabilizer_.reset();
    glam_stabilizer_.reset();
    worker_ = std::thread(&AiEngine::worker_main, this);
}

void AiEngine::stop()
{
    stop_requested_.store(true);
    queue_cv_.notify_all();
    if (worker_.joinable())
        worker_.join();
    {
        std::scoped_lock lock(queue_mutex_);
        pending_.reset();
    }
}

void AiEngine::submit(RgbImage frame, std::uint64_t timestamp_ns)
{
    if (!frame.valid() || stop_requested_.load())
        return;
    std::scoped_lock lock(queue_mutex_);
    if (pending_) {
        std::scoped_lock status_lock(result_mutex_);
        ++status_.dropped_frames;
    }
    pending_ = QueuedFrame{std::move(frame), timestamp_ns};
    queue_cv_.notify_one();
}

bool AiEngine::get_latest_masks(RgbaMask& mask, RgbaMask& glam_mask, std::uint64_t& last_sequence) const
{
    std::scoped_lock lock(result_mutex_);
    if (!latest_mask_.valid() || !latest_glam_mask_.valid() || latest_mask_.sequence == last_sequence)
        return false;
    mask = latest_mask_;
    glam_mask = latest_glam_mask_;
    last_sequence = latest_mask_.sequence;
    return true;
}

void AiEngine::set_stability(float value) noexcept
{
    stability_.store(clampf(value, 0.0f, 1.0f));
}

AiStatus AiEngine::status() const
{
    std::scoped_lock lock(result_mutex_);
    return status_;
}

void AiEngine::set_status(std::string message, bool ready, bool directml, float inference_ms)
{
    std::scoped_lock lock(result_mutex_);
    status_.message = std::move(message);
    status_.ready = ready;
    status_.using_directml = directml;
    status_.inference_ms = inference_ms;
}

void AiEngine::worker_main()
{
    try {
        std::string error;
    if (!detector_.load(config_.detector_model, config_.backend, error)) {
        set_status("Face detector failed: " + error, false, false);
        return;
    }
    if (!parser_.load(config_.parser_model, config_.backend, error)) {
        set_status("Face parser failed: " + error, false, detector_.using_directml());
        return;
    }
    const bool dml = detector_.using_directml() && parser_.using_directml();
    set_status(dml ? "AI ready — DirectML" : "AI ready — CPU", true, dml);

    while (!stop_requested_.load()) {
        std::optional<QueuedFrame> item;
        {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, [&] { return stop_requested_.load() || pending_.has_value(); });
            if (stop_requested_.load())
                break;
            item = std::move(pending_);
            pending_.reset();
        }
        if (!item)
            continue;
        const auto begin = std::chrono::steady_clock::now();
        process_frame(*item);
        const auto end = std::chrono::steady_clock::now();
        const float ms = std::chrono::duration<float, std::milli>(end - begin).count();
        {
            std::scoped_lock lock(result_mutex_);
            status_.inference_ms = ms;
            ++status_.processed_frames;
        }
    }
    } catch (const std::exception& e) {
        set_status(std::string("AI worker stopped safely: ") + e.what(), false, false);
    } catch (...) {
        set_status("AI worker stopped safely after an unknown native error.", false, false);
    }
}

std::optional<RectF> AiEngine::select_candidate(const std::vector<RectF>& faces, float& motion)
{
    motion = 1.0f;
    if (faces.empty())
        return std::nullopt;

    auto largest_credible = [&]() -> RectF {
        return *std::max_element(faces.begin(), faces.end(), [](const RectF& a, const RectF& b) {
            return a.score * std::sqrt(std::max(1.0f, a.area())) < b.score * std::sqrt(std::max(1.0f, b.area()));
        });
    };

    if (track_state_ == TrackState::Lost || track_state_ == TrackState::Reacquire ||
        tracked_face_.area() <= 1.0f) {
        const RectF candidate = largest_credible();
        if (track_state_ != TrackState::Reacquire || iou(candidate, reacquire_face_) < 0.30f) {
            reacquire_face_ = candidate;
            reacquire_count_ = 1;
            track_state_ = TrackState::Reacquire;
            return std::nullopt;
        }
        reacquire_face_.x = reacquire_face_.x * 0.42f + candidate.x * 0.58f;
        reacquire_face_.y = reacquire_face_.y * 0.42f + candidate.y * 0.58f;
        reacquire_face_.w = reacquire_face_.w * 0.42f + candidate.w * 0.58f;
        reacquire_face_.h = reacquire_face_.h * 0.42f + candidate.h * 0.58f;
        reacquire_face_.score = candidate.score;
        if (++reacquire_count_ < 2)
            return std::nullopt;
        tracked_face_ = reacquire_face_;
        track_state_ = TrackState::Tracking;
        miss_count_ = 0;
        motion = 0.0f;
        return tracked_face_;
    }

    const float norm = std::max(12.0f, 0.5f * (tracked_face_.w + tracked_face_.h));
    const RectF* best = nullptr;
    float best_score = -1e9f;
    float best_motion = 1.0f;
    for (const auto& face : faces) {
        const float dx = face.cx() - tracked_face_.cx();
        const float dy = face.cy() - tracked_face_.cy();
        const float distance = std::sqrt(dx * dx + dy * dy) / norm;
        const float scale_ratio = std::max(face.w / std::max(1.0f, tracked_face_.w),
                                           tracked_face_.w / std::max(1.0f, face.w));
        const float overlap = iou(face, tracked_face_);
        const float score = face.score + overlap * 0.92f - distance * 0.65f - std::max(0.0f, scale_ratio - 1.0f) * 0.30f;
        if (score > best_score) {
            best_score = score;
            best = &face;
            best_motion = distance;
        }
    }

    if (!best || best_motion > 0.62f || best_score < 0.28f) {
        track_state_ = TrackState::Reacquire;
        reacquire_face_ = largest_credible();
        reacquire_count_ = 1;
        return std::nullopt;
    }

    motion = clampf(best_motion, 0.0f, 1.0f);
    const float follow = clampf(0.50f + motion * 0.34f, 0.50f, 0.84f);
    tracked_face_.x = tracked_face_.x * (1.0f - follow) + best->x * follow;
    tracked_face_.y = tracked_face_.y * (1.0f - follow) + best->y * follow;
    tracked_face_.w = tracked_face_.w * (1.0f - follow) + best->w * follow;
    tracked_face_.h = tracked_face_.h * (1.0f - follow) + best->h * follow;
    tracked_face_.score = best->score;
    track_state_ = TrackState::Tracking;
    miss_count_ = 0;
    return tracked_face_;
}

void AiEngine::publish_masks(RgbaMask mask, RgbaMask glam_mask)
{
    std::scoped_lock lock(result_mutex_);
    latest_mask_ = std::move(mask);
    latest_glam_mask_ = std::move(glam_mask);
}

void AiEngine::publish_release_mask(int width, int height, float motion)
{
    RgbaMask empty;
    empty.width = std::max(1, width);
    empty.height = std::max(1, height);
    empty.pixels.assign(static_cast<std::size_t>(empty.width * empty.height * 4), 0);
    RgbaMask empty_glam = empty;
    const auto seq = ++sequence_;
    empty = stabilizer_.update(empty, false, motion, stability_.load(), seq);
    empty_glam = glam_stabilizer_.update(empty_glam, false, motion, stability_.load(), seq);
    empty_glam.sequence = seq;
    publish_masks(std::move(empty), std::move(empty_glam));
}

void AiEngine::process_frame(const QueuedFrame& item)
{
    std::vector<RectF> faces;
    std::string error;
    const int mask_width = 320;
    const int mask_height = std::max(1, static_cast<int>(std::lround(320.0 * item.frame.height / std::max(1, item.frame.width))));

    if (!detector_.detect(item.frame, faces, error)) {
        publish_release_mask(mask_width, mask_height, 1.0f);
        set_status("Detector error: " + error, false, detector_.using_directml());
        return;
    }

    float motion = 1.0f;
    const auto face = select_candidate(faces, motion);
    if (!face) {
        ++miss_count_;
        if (miss_count_ <= 2 && track_state_ == TrackState::Tracking)
            track_state_ = TrackState::ShortHold;
        else if (miss_count_ > 3 && track_state_ != TrackState::Reacquire)
            track_state_ = TrackState::Lost;
        publish_release_mask(mask_width, mask_height, motion);
        return;
    }

    RgbaMask current;
    RgbaMask glam;
    if (!parser_.parse(item.frame, *face, mask_width, mask_height, current, glam, error)) {
        ++miss_count_;
        publish_release_mask(mask_width, mask_height, std::max(0.55f, motion));
        return;
    }

    {
        std::scoped_lock lock(result_mutex_);
        if (!status_.ready) {
            status_.ready = true;
            status_.using_directml = detector_.using_directml() && parser_.using_directml();
            status_.message = status_.using_directml ? "AI ready — DirectML" : "AI ready — CPU";
        }
    }

    const auto seq = ++sequence_;
    current = stabilizer_.update(current, true, motion, stability_.load(), seq);
    glam = glam_stabilizer_.update(glam, true, motion, stability_.load(), seq);
    glam.sequence = seq;
    publish_masks(std::move(current), std::move(glam));
}

} // namespace baddiecam::ai
