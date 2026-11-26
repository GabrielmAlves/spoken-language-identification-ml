#include "classifier.hpp"
#include "util.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

constexpr float kRtol = 1e-3f;
constexpr float kAtol = 1e-4f;

bool report(const std::string& label, const std::vector<float>& got, const std::vector<float>& expected, size_t n_classes) {
	if (got.size() != expected.size()) {
		std::cerr << label << ": tamanho " << got.size() << " != " << expected.size() << "\n";
		return false;
	}
	const auto cmp = lid::compare(got.data(), expected.data(), got.size(), kRtol, kAtol);
	const size_t rows = got.size() / n_classes;
	const size_t same = lid::same_argmax(got, expected, n_classes);
	std::cout << label << ": max_abs_diff=" << cmp.max_abs_diff << " allclose=" << (cmp.allclose ? "sim" : "não") << " argmax igual=" << same << "/" << rows << "\n";
	return cmp.allclose && same == rows;
}

int main(int argc, char** argv) {
	if (argc != 3) {
		std::cerr << "uso: " << argv[0] << " <modelo.onnx> <pasta_fixtures>\n";
		return 2;
	}
	try {
		const fs::path fixtures = argv[2];
		lid::Classifier classifier(argv[1]);
		const size_t n_classes = classifier.n_classes();
		std::cout << "entrada: " << classifier.input_name() << " | saída: " << classifier.output_name() << " | classes: " << n_classes << "\n";

		const std::vector<float> input = lid::read_floats(fixtures / "input.bin");
		const std::vector<float> expected = lid::read_floats(fixtures / "logits.bin");
		const size_t clip_size = classifier.clip_size();
		if (input.size() % clip_size != 0) {
			throw std::runtime_error("input.bin não é múltiplo de n_mels x n_frames");
		}
		const size_t batch = input.size() / clip_size;
		if (expected.size() != batch * n_classes) {
			throw std::runtime_error("logits.bin não bate com o número de clipes");
		}
		std::cout << "clipes: " << batch << "\n";

		const auto batched = classifier.logits(input.data(), static_cast<int64_t>(batch));
		std::vector<float> single;
		for (size_t i = 0; i < batch; ++i) {
			const auto out = classifier.logits(input.data() + i * clip_size, 1);
			single.insert(single.end(), out.begin(), out.end());
		}

		const bool ok_batched = report("batch " + std::to_string(batch), batched, expected, n_classes);
		const bool ok_single = report("batch 1", single, expected, n_classes);
		const bool ok = ok_batched && ok_single;
		std::cout << (ok ? "OK: o C++ reproduz os logits do torch" : "FALHA: o C++ diverge do torch") << "\n";
		return ok ? 0 : 1;
	} catch (const std::exception& e) {
		std::cerr << "erro: " << e.what() << "\n";
		return 1;
	}
}
