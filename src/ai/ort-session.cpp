#include "ort-session.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <functional>
#include <numeric>

#ifdef BADDIECAM_WITH_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#ifdef _WIN32
#include <dml_provider_factory.h>
#endif
#endif

namespace baddiecam::ai {

struct OrtSession::Impl {
#ifdef BADDIECAM_WITH_ONNXRUNTIME
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "BaddieCamBeautyCoreAI"};
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> input_names_owned;
    std::vector<const char*> input_names;
    std::vector<std::string> output_names_owned;
    std::vector<const char*> output_names;
#endif
    bool directml = false;
};

OrtSession::OrtSession() : impl_(std::make_unique<Impl>()) {}
OrtSession::~OrtSession() = default;

bool OrtSession::load(const std::wstring& model_path, InferenceBackend backend, std::string& error)
{
#ifndef BADDIECAM_WITH_ONNXRUNTIME
    (void)model_path;
    (void)backend;
    error = "ONNX Runtime was not compiled into this build.";
    return false;
#else
    impl_->session.reset();
    impl_->input_names_owned.clear();
    impl_->input_names.clear();
    impl_->output_names_owned.clear();
    impl_->output_names.clear();
    impl_->directml = false;

    auto create_session = [&](bool request_directml, std::string& attempt_error) -> bool {
        try {
            Ort::SessionOptions options;
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            options.DisableMemPattern();
            options.SetIntraOpNumThreads(1);
            options.SetLogSeverityLevel(3);

#ifdef _WIN32
            if (request_directml) {
                OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_DML(options, 0);
                if (status != nullptr) {
                    const char* message = Ort::GetApi().GetErrorMessage(status);
                    attempt_error = message ? message : "unknown DirectML initialization error";
                    Ort::GetApi().ReleaseStatus(status);
                    return false;
                }
            }
#else
            if (request_directml) {
                attempt_error = "DirectML is available only on Windows.";
                return false;
            }
#endif

            auto session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), options);
            Ort::AllocatorWithDefaultOptions allocator;
            std::vector<std::string> inputs;
            std::vector<std::string> outputs;
            const std::size_t input_count = session->GetInputCount();
            const std::size_t output_count = session->GetOutputCount();
            for (std::size_t i = 0; i < input_count; ++i) {
                auto name = session->GetInputNameAllocated(i, allocator);
                inputs.emplace_back(name.get() ? name.get() : "input");
            }
            for (std::size_t i = 0; i < output_count; ++i) {
                auto name = session->GetOutputNameAllocated(i, allocator);
                outputs.emplace_back(name.get() ? name.get() : "output");
            }

            impl_->session = std::move(session);
            impl_->input_names_owned = std::move(inputs);
            impl_->output_names_owned = std::move(outputs);
            for (auto& name : impl_->input_names_owned)
                impl_->input_names.push_back(name.c_str());
            for (auto& name : impl_->output_names_owned)
                impl_->output_names.push_back(name.c_str());
            impl_->directml = request_directml;
            return true;
        } catch (const Ort::Exception& e) {
            attempt_error = e.what();
        } catch (const std::exception& e) {
            attempt_error = e.what();
        }
        impl_->session.reset();
        impl_->input_names_owned.clear();
        impl_->input_names.clear();
        impl_->output_names_owned.clear();
        impl_->output_names.clear();
        impl_->directml = false;
        return false;
    };

    if (backend == InferenceBackend::CPU)
        return create_session(false, error);
    if (backend == InferenceBackend::DirectML)
        return create_session(true, error);

    std::string directml_error;
    if (create_session(true, directml_error))
        return true;
    std::string cpu_error;
    if (create_session(false, cpu_error))
        return true;
    error = "DirectML failed (" + directml_error + "); CPU fallback failed (" + cpu_error + ").";
    return false;
#endif
}

bool OrtSession::run(const std::vector<float>& input, const std::vector<std::int64_t>& shape,
                     std::vector<TensorOutput>& outputs, std::string& error)
{
#ifndef BADDIECAM_WITH_ONNXRUNTIME
    (void)input;
    (void)shape;
    (void)outputs;
    error = "ONNX Runtime was not compiled into this build.";
    return false;
#else
    if (!impl_->session || impl_->input_names.empty()) {
        error = "Session is not loaded.";
        return false;
    }
    try {
        const std::int64_t expected = std::accumulate(shape.begin(), shape.end(), std::int64_t{1}, std::multiplies<>());
        if (expected <= 0 || static_cast<std::size_t>(expected) != input.size()) {
            error = "Input tensor size does not match its shape.";
            return false;
        }
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value tensor = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(input.data()), input.size(),
                                                            shape.data(), shape.size());
        auto values = impl_->session->Run(Ort::RunOptions{nullptr}, impl_->input_names.data(), &tensor, 1,
                                          impl_->output_names.data(), impl_->output_names.size());
        outputs.clear();
        outputs.reserve(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (!values[i].IsTensor())
                continue;
            auto info = values[i].GetTensorTypeAndShapeInfo();
            const auto tensor_shape = info.GetShape();
            const std::size_t count = info.GetElementCount();
            const float* data = values[i].GetTensorData<float>();
            TensorOutput out;
            out.name = i < impl_->output_names_owned.size() ? impl_->output_names_owned[i] : "output";
            out.shape = tensor_shape;
            out.data.assign(data, data + count);
            outputs.emplace_back(std::move(out));
        }
        return !outputs.empty();
    } catch (const Ort::Exception& e) {
        error = e.what();
    } catch (const std::exception& e) {
        error = e.what();
    }
    return false;
#endif
}

bool OrtSession::loaded() const noexcept
{
#ifdef BADDIECAM_WITH_ONNXRUNTIME
    return impl_ && impl_->session != nullptr;
#else
    return false;
#endif
}

bool OrtSession::using_directml() const noexcept
{
    return impl_ && impl_->directml;
}

} // namespace baddiecam::ai
