#include "wav.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace lid {
namespace {

constexpr uint16_t kFormatPcm = 1;
constexpr uint16_t kFormatFloat = 3;
constexpr uint16_t kFormatExtensible = 0xFFFE;

uint16_t read_u16(const unsigned char* p) {
	return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

uint32_t read_u32(const unsigned char* p) {
	return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

struct Format {
	uint16_t tag = 0;
	uint16_t channels = 0;
	uint32_t sample_rate = 0;
	uint16_t bits = 0;
};

}

std::vector<double> read_wav(const std::filesystem::path& path, int sample_rate) {
	const std::string name = path.string();
	std::ifstream file(path, std::ios::binary);
	if (!file) {
		throw std::runtime_error("não foi possível abrir " + name);
	}
	const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
		throw std::runtime_error(name + ": não é um arquivo WAV (RIFF/WAVE)");
	}

	Format format;
	bool has_format = false;
	const unsigned char* data = nullptr;
	size_t data_size = 0;
	size_t pos = 12;
	while (pos + 8 <= bytes.size() && data == nullptr) {
		const unsigned char* chunk = bytes.data() + pos;
		const uint32_t size = read_u32(chunk + 4);
		const unsigned char* body = chunk + 8;
		const size_t available = bytes.size() - pos - 8;
		if (std::memcmp(chunk, "fmt ", 4) == 0) {
			if (size < 16 || size > available) {
				throw std::runtime_error(name + ": bloco fmt inválido");
			}
			format.tag = read_u16(body);
			format.channels = read_u16(body + 2);
			format.sample_rate = read_u32(body + 4);
			format.bits = read_u16(body + 14);
			if (format.tag == kFormatExtensible) {
				if (size < 40) {
					throw std::runtime_error(name + ": bloco fmt extensível inválido");
				}
				format.tag = read_u16(body + 24);
			}
			has_format = true;
		} else if (std::memcmp(chunk, "data", 4) == 0) {
			if (!has_format) {
				throw std::runtime_error(name + ": bloco data antes do bloco fmt");
			}
			data = body;
			data_size = std::min<size_t>(size, available);
		}
		pos += 8 + static_cast<size_t>(size) + (size & 1u);
	}
	if (!has_format) {
		throw std::runtime_error(name + ": bloco fmt ausente");
	}
	if (data == nullptr) {
		throw std::runtime_error(name + ": bloco data ausente");
	}
	if (format.channels != 1) {
		throw std::runtime_error(name + ": esperado mono, encontrado " + std::to_string(format.channels) + " canais");
	}
	if (format.sample_rate != static_cast<uint32_t>(sample_rate)) {
		throw std::runtime_error(name + ": taxa de amostragem " + std::to_string(format.sample_rate) + " Hz, esperado " + std::to_string(sample_rate) + " Hz (resampling não suportado)");
	}

	std::vector<double> samples;
	if (format.tag == kFormatPcm && format.bits == 16) {
		samples.resize(data_size / 2);
		for (size_t i = 0; i < samples.size(); ++i) {
			const auto value = static_cast<int16_t>(read_u16(data + 2 * i));
			samples[i] = static_cast<double>(value) / 32768.0;
		}
	} else if (format.tag == kFormatFloat && format.bits == 32) {
		samples.resize(data_size / 4);
		for (size_t i = 0; i < samples.size(); ++i) {
			const uint32_t raw = read_u32(data + 4 * i);
			float value;
			std::memcpy(&value, &raw, sizeof(value));
			samples[i] = static_cast<double>(value);
		}
	} else {
		throw std::runtime_error(name + ": formato não suportado (código " + std::to_string(format.tag) + ", " + std::to_string(format.bits) + " bits); use PCM 16 bits ou float 32 bits");
	}
	if (samples.empty()) {
		throw std::runtime_error(name + ": arquivo sem amostras");
	}
	return samples;
}

}
