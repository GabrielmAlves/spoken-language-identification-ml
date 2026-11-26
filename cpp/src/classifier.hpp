#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace lid {

class Classifier {
public:
	explicit Classifier(const std::filesystem::path& model_path);

	std::vector<float> logits(const float* input, int64_t batch);
	const std::string& input_name() const { return input_name_; }
	const std::string& output_name() const { return output_name_; }
	int64_t n_mels() const { return n_mels_; }
	int64_t n_frames() const { return n_frames_; }
	size_t n_classes() const { return n_classes_; }
	size_t clip_size() const { return static_cast<size_t>(n_mels_ * n_frames_); }

private:
	Ort::Env env_;
	Ort::Session session_;
	Ort::MemoryInfo memory_;
	std::string input_name_;
	std::string output_name_;
	int64_t n_mels_ = 0;
	int64_t n_frames_ = 0;
	size_t n_classes_ = 0;
};

std::vector<double> softmax(const float* logits, size_t n);

}
