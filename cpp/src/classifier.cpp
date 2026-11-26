#include "classifier.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace lid {

Classifier::Classifier(const std::filesystem::path& model_path)
	: env_(ORT_LOGGING_LEVEL_WARNING, "lid"),
	  session_(env_, model_path.c_str(), Ort::SessionOptions{}),
	  memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
	Ort::AllocatorWithDefaultOptions allocator;
	input_name_ = session_.GetInputNameAllocated(0, allocator).get();
	output_name_ = session_.GetOutputNameAllocated(0, allocator).get();
	const auto in_shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	const auto out_shape = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	if (in_shape.size() != 4 || in_shape[1] != 1 || in_shape[2] <= 0 || in_shape[3] <= 0) {
		throw std::runtime_error("entrada do modelo deveria ser [batch, 1, n_mels, n_frames]");
	}
	if (out_shape.size() != 2 || out_shape[1] <= 0) {
		throw std::runtime_error("saída do modelo deveria ser [batch, n_classes]");
	}
	n_mels_ = in_shape[2];
	n_frames_ = in_shape[3];
	n_classes_ = static_cast<size_t>(out_shape[1]);
}

std::vector<float> Classifier::logits(const float* input, int64_t batch) {
	const std::array<int64_t, 4> shape{batch, 1, n_mels_, n_frames_};
	const size_t count = static_cast<size_t>(batch) * clip_size();
	auto tensor = Ort::Value::CreateTensor<float>(memory_, const_cast<float*>(input), count, shape.data(), shape.size());
	const char* in_names[] = {input_name_.c_str()};
	const char* out_names[] = {output_name_.c_str()};
	auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
	const float* out = outputs[0].GetTensorData<float>();
	const size_t n = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
	return std::vector<float>(out, out + n);
}

std::vector<double> softmax(const float* logits, size_t n) {
	const float max_logit = *std::max_element(logits, logits + n);
	std::vector<double> probs(n);
	double total = 0.0;
	for (size_t i = 0; i < n; ++i) {
		probs[i] = std::exp(static_cast<double>(logits[i]) - max_logit);
		total += probs[i];
	}
	for (double& p : probs) {
		p /= total;
	}
	return probs;
}

}
