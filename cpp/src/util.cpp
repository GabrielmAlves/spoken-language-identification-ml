#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <regex>
#include <stdexcept>

namespace lid {

std::vector<float> read_floats(const std::filesystem::path& path) {
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

std::string read_text(const std::filesystem::path& path) {
	std::ifstream file(path);
	if (!file) {
		throw std::runtime_error("não foi possível abrir " + path.string());
	}
	return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::vector<std::string> json_string_array(const std::string& text, const std::string& key) {
	const std::regex array_re("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
	std::smatch match;
	if (!std::regex_search(text, match, array_re)) {
		throw std::runtime_error("chave \"" + key + "\" não encontrada no JSON");
	}
	const std::string body = match[1].str();
	const std::regex item_re("\"([^\"]*)\"");
	std::vector<std::string> items;
	for (auto it = std::sregex_iterator(body.begin(), body.end(), item_re); it != std::sregex_iterator(); ++it) {
		items.push_back((*it)[1].str());
	}
	return items;
}

std::vector<std::string> json_string_values(const std::string& text, const std::string& key) {
	const std::regex value_re("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
	std::vector<std::string> values;
	for (auto it = std::sregex_iterator(text.begin(), text.end(), value_re); it != std::sregex_iterator(); ++it) {
		values.push_back((*it)[1].str());
	}
	return values;
}

Comparison compare(const float* got, const float* expected, size_t n, float rtol, float atol) {
	Comparison result;
	for (size_t i = 0; i < n; ++i) {
		const float diff = std::fabs(got[i] - expected[i]);
		result.max_abs_diff = std::max(result.max_abs_diff, diff);
		if (!(diff <= atol + rtol * std::fabs(expected[i]))) {
			result.allclose = false;
		}
	}
	return result;
}

size_t same_argmax(const std::vector<float>& got, const std::vector<float>& expected, size_t n_classes) {
	const size_t rows = std::min(got.size(), expected.size()) / n_classes;
	const auto width = static_cast<std::ptrdiff_t>(n_classes);
	size_t same = 0;
	for (size_t r = 0; r < rows; ++r) {
		const auto g = got.begin() + static_cast<std::ptrdiff_t>(r * n_classes);
		const auto e = expected.begin() + static_cast<std::ptrdiff_t>(r * n_classes);
		if (std::max_element(g, g + width) - g == std::max_element(e, e + width) - e) {
			++same;
		}
	}
	return same;
}

}
