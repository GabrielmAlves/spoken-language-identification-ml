#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

constexpr int64_t kMels = 64;
constexpr int64_t kFrames = 498;
constexpr float kRtol = 1e-3f;
constexpr float kAtol = 1e-4f;

std::vector<float> read_floats(const fs::path& path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		throw std::runtime_error("não foi possível abrir " + path.string());
	}
	const auto bytes = static_cast<size_t>(file.tellg());
	if (bytes % sizeof(float) != 0) {
		throw std::runtime_error(path.string() + " não contém float32");
	}
	std::vector<float> data(bytes / sizeof(float));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
	return data;
}

std::vector<float> run(Ort::Session& session, const std::string& input_name, const std::string& output_name, float* data, size_t count, int64_t batch) {
	const std::array<int64_t, 4> shape{batch, 1, kMels, kFrames};
	auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	auto tensor = Ort::Value::CreateTensor<float>(memory, data, count, shape.data(), shape.size());
	const char* in_names[] = {input_name.c_str()};
	const char* out_names[] = {output_name.c_str()};
	auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);
	const float* out = outputs[0].GetTensorData<float>();
	const size_t n = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
	return std::vector<float>(out, out + n);
}

bool compare(const std::string& label, const std::vector<float>& got, const std::vector<float>& expected, size_t n_classes) {
	if (got.size() != expected.size()) {
		std::cerr << label << ": tamanho " << got.size() << " != " << expected.size() << "\n";
		return false;
	}
	float max_abs = 0.0f;
	bool close = true;
	for (size_t i = 0; i < got.size(); ++i) {
		const float diff = std::fabs(got[i] - expected[i]);
		max_abs = std::max(max_abs, diff);
		if (diff > kAtol + kRtol * std::fabs(expected[i])) {
			close = false;
		}
	}
	const size_t rows = got.size() / n_classes;
	size_t same = 0;
	for (size_t r = 0; r < rows; ++r) {
		const auto g = got.begin() + static_cast<std::ptrdiff_t>(r * n_classes);
		const auto e = expected.begin() + static_cast<std::ptrdiff_t>(r * n_classes);
		const auto end = static_cast<std::ptrdiff_t>(n_classes);
		if (std::max_element(g, g + end) - g == std::max_element(e, e + end) - e) {
			++same;
		}
	}
	std::cout << label << ": max_abs_diff=" << max_abs << " allclose=" << (close ? "sim" : "não") << " argmax igual=" << same << "/" << rows << "\n";
	return close && same == rows;
}

int main(int argc, char** argv) {
	if (argc != 3) {
		std::cerr << "uso: " << argv[0] << " <modelo.onnx> <pasta_fixtures>\n";
		return 2;
	}
	try {
		const fs::path model_path = argv[1];
		const fs::path fixtures = argv[2];

		Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "lid");
		Ort::SessionOptions options;
		Ort::Session session(env, model_path.c_str(), options);

		Ort::AllocatorWithDefaultOptions allocator;
		const std::string input_name = session.GetInputNameAllocated(0, allocator).get();
		const std::string output_name = session.GetOutputNameAllocated(0, allocator).get();
		const auto out_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
		const auto n_classes = static_cast<size_t>(out_shape.back());
		std::cout << "entrada: " << input_name << " | saída: " << output_name << " | classes: " << n_classes << "\n";

		std::vector<float> input = read_floats(fixtures / "input.bin");
		const std::vector<float> expected = read_floats(fixtures / "logits.bin");
		const auto clip_size = static_cast<size_t>(kMels * kFrames);
		if (input.size() % clip_size != 0) {
			throw std::runtime_error("input.bin não é múltiplo de 64x498");
		}
		const size_t batch = input.size() / clip_size;
		if (expected.size() != batch * n_classes) {
			throw std::runtime_error("logits.bin não bate com o número de clipes");
		}
		std::cout << "clipes: " << batch << "\n";

		const auto batched = run(session, input_name, output_name, input.data(), input.size(), static_cast<int64_t>(batch));
		std::vector<float> single;
		for (size_t i = 0; i < batch; ++i) {
			const auto out = run(session, input_name, output_name, input.data() + i * clip_size, clip_size, 1);
			single.insert(single.end(), out.begin(), out.end());
		}

		const bool ok_batched = compare("batch " + std::to_string(batch), batched, expected, n_classes);
		const bool ok_single = compare("batch 1", single, expected, n_classes);
		const bool ok = ok_batched && ok_single;
		std::cout << (ok ? "OK: o C++ reproduz os logits do torch" : "FALHA: o C++ diverge do torch") << "\n";
		return ok ? 0 : 1;
	} catch (const std::exception& e) {
		std::cerr << "erro: " << e.what() << "\n";
		return 1;
	}
}
