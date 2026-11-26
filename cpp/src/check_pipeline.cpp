#include "classifier.hpp"
#include "log_mel.hpp"
#include "util.hpp"
#include "wav.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

constexpr float kFeatureRtol = 1e-4f;
constexpr float kFeatureAtol = 1e-4f;
constexpr float kLogitRtol = 1e-3f;
constexpr float kLogitAtol = 1e-4f;

int main(int argc, char** argv) {
	if (argc != 3) {
		std::cerr << "uso: " << argv[0] << " <raiz_fleurs> <pasta_modelo>\n";
		return 2;
	}
	try {
		const fs::path root = argv[1];
		const fs::path model_dir = argv[2];
		const fs::path fixtures = model_dir / "fixtures";

		const std::vector<std::string> paths = lid::json_string_values(lid::read_text(fixtures / "fixtures.json"), "path");
		const std::vector<float> expected_input = lid::read_floats(fixtures / "input.bin");
		const std::vector<float> expected_logits = lid::read_floats(fixtures / "logits.bin");

		const lid::LogMel log_mel;
		const auto& config = log_mel.config();
		const auto clip_size = static_cast<size_t>(config.n_mels * config.n_frames());
		if (paths.empty() || expected_input.size() != paths.size() * clip_size) {
			throw std::runtime_error("fixtures.json e input.bin não têm o mesmo número de clipes");
		}

		std::vector<float> features;
		bool features_ok = true;
		float features_max = 0.0f;
		for (size_t i = 0; i < paths.size(); ++i) {
			const auto clip = log_mel(lid::read_wav(root / paths[i], config.sample_rate));
			const auto cmp = lid::compare(clip.data(), expected_input.data() + i * clip_size, clip_size, kFeatureRtol, kFeatureAtol);
			std::cout << paths[i] << ": max_abs_diff=" << cmp.max_abs_diff << " allclose=" << (cmp.allclose ? "sim" : "não") << "\n";
			features_ok = features_ok && cmp.allclose;
			features_max = std::max(features_max, cmp.max_abs_diff);
			features.insert(features.end(), clip.begin(), clip.end());
		}
		std::cout << "log-Mel C++ x Python: max_abs_diff=" << features_max << " allclose=" << (features_ok ? "sim" : "não") << " (" << paths.size() << " clipes)\n";

		lid::Classifier classifier(model_dir / "lid_cnn.onnx");
		if (classifier.n_mels() != config.n_mels || classifier.n_frames() != config.n_frames()) {
			throw std::runtime_error("o modelo espera outro tamanho de log-Mel");
		}
		const auto logits = classifier.logits(features.data(), static_cast<int64_t>(paths.size()));
		if (logits.size() != expected_logits.size()) {
			throw std::runtime_error("logits.bin não bate com o número de clipes");
		}
		const auto cmp = lid::compare(logits.data(), expected_logits.data(), logits.size(), kLogitRtol, kLogitAtol);
		const size_t same = lid::same_argmax(logits, expected_logits, classifier.n_classes());
		const bool logits_ok = cmp.allclose && same == paths.size();
		std::cout << "logits wav->C++ x torch: max_abs_diff=" << cmp.max_abs_diff << " allclose=" << (cmp.allclose ? "sim" : "não") << " argmax igual=" << same << "/" << paths.size() << "\n";

		const bool ok = features_ok && logits_ok;
		std::cout << (ok ? "OK: o pipeline C++ (wav -> log-Mel -> ONNX) reproduz o Python" : "FALHA: o pipeline C++ diverge do Python") << "\n";
		return ok ? 0 : 1;
	} catch (const std::exception& e) {
		std::cerr << "erro: " << e.what() << "\n";
		return 1;
	}
}
