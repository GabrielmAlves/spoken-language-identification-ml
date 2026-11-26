#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace lid {

struct Comparison {
	float max_abs_diff = 0.0f;
	bool allclose = true;
};

std::vector<float> read_floats(const std::filesystem::path& path);
std::string read_text(const std::filesystem::path& path);
std::vector<std::string> json_string_array(const std::string& text, const std::string& key);
std::vector<std::string> json_string_values(const std::string& text, const std::string& key);
Comparison compare(const float* got, const float* expected, size_t n, float rtol, float atol);
size_t same_argmax(const std::vector<float>& got, const std::vector<float>& expected, size_t n_classes);

}
