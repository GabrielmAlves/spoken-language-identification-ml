#pragma once

#include <filesystem>
#include <vector>

namespace lid {

std::vector<double> read_wav(const std::filesystem::path& path, int sample_rate);

}
