#include "classifier.hpp"
#include "log_mel.hpp"
#include "util.hpp"
#include "wav.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
	return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv) {
	if (argc < 3) {
		std::cerr << "uso: " << argv[0] << " <pasta_modelo> <arquivo.wav> [mais arquivos.wav ...]\n";
		std::cerr << "a pasta do modelo deve conter lid_cnn.onnx e model.json\n";
		return 2;
	}
	try {
		const fs::path model_dir = argv[1];
		const auto load_start = Clock::now();
		const std::vector<std::string> labels = lid::json_string_array(lid::read_text(model_dir / "model.json"), "labels");
		lid::Classifier classifier(model_dir / "lid_cnn.onnx");
		const lid::LogMel log_mel;
		const auto& config = log_mel.config();
		if (labels.size() != classifier.n_classes()) {
			throw std::runtime_error("model.json tem " + std::to_string(labels.size()) + " rótulos, o modelo tem " + std::to_string(classifier.n_classes()) + " classes");
		}
		if (classifier.n_mels() != config.n_mels || classifier.n_frames() != config.n_frames()) {
			throw std::runtime_error("o modelo espera outro tamanho de log-Mel");
		}
		std::cout << "modelo carregado em " << std::fixed << std::setprecision(1) << elapsed_ms(load_start, Clock::now()) << " ms\n";

		int failures = 0;
		for (int a = 2; a < argc; ++a) {
			const fs::path wav = argv[a];
			try {
				const auto t0 = Clock::now();
				const auto features = log_mel(lid::read_wav(wav, config.sample_rate));
				const auto t1 = Clock::now();
				const auto logits = classifier.logits(features.data(), 1);
				const auto t2 = Clock::now();
				const auto probs = lid::softmax(logits.data(), logits.size());
				const auto best = static_cast<size_t>(std::max_element(probs.begin(), probs.end()) - probs.begin());

				std::cout << "\n" << wav.string() << "\n";
				std::cout << "  idioma: " << labels[best] << " (" << std::setprecision(1) << 100.0 * probs[best] << "%)\n";
				for (size_t i = 0; i < labels.size(); ++i) {
					std::cout << "    " << std::left << std::setw(8) << labels[i] << std::right << std::setw(6) << std::setprecision(1) << 100.0 * probs[i] << "%\n";
				}
				std::cout << "  tempo: pré-processamento " << std::setprecision(1) << elapsed_ms(t0, t1) << " ms, inferência " << elapsed_ms(t1, t2) << " ms\n";
			} catch (const std::exception& e) {
				std::cerr << "\nerro: " << e.what() << "\n";
				++failures;
			}
		}
		return failures == 0 ? 0 : 1;
	} catch (const std::exception& e) {
		std::cerr << "erro: " << e.what() << "\n";
		return 1;
	}
}
