#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace baddiecam::ai {

enum class InferenceBackend {
    Auto,
    DirectML,
    CPU,
};

struct TensorOutput {
    std::string name;
    std::vector<std::int64_t> shape;
    std::vector<float> data;
};

class OrtSession {
public:
    OrtSession();
    ~OrtSession();
    OrtSession(const OrtSession&) = delete;
    OrtSession& operator=(const OrtSession&) = delete;

    bool load(const std::wstring& model_path, InferenceBackend backend, std::string& error);
    bool run(const std::vector<float>& input, const std::vector<std::int64_t>& shape,
             std::vector<TensorOutput>& outputs, std::string& error);
    [[nodiscard]] bool loaded() const noexcept;
    [[nodiscard]] bool using_directml() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace baddiecam::ai
