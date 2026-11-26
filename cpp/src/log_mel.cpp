#include "log_mel.hpp"

#include <algorithm>
#include <cmath>

namespace lid {
namespace {

constexpr double kPi = 3.14159265358979323846;

}

double hz_to_mel(double hz) {
	return 2595.0 * std::log10(1.0 + hz / 700.0);
}

double mel_to_hz(double mel) {
	return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

std::vector<double> hann_periodic(int n) {
	std::vector<double> window(static_cast<size_t>(n));
	for (int k = 0; k < n; ++k) {
		window[static_cast<size_t>(k)] = 0.5 - 0.5 * std::cos(2.0 * kPi * k / n);
	}
	return window;
}

std::vector<double> mel_filterbank(const PreprocessingConfig& config) {
	const int n_points = config.n_mels + 2;
	const double mel_min = hz_to_mel(config.f_min);
	const double mel_max = hz_to_mel(config.f_max);
	const double step = (mel_max - mel_min) / (n_points - 1);
	std::vector<double> points(static_cast<size_t>(n_points));
	for (int i = 0; i < n_points; ++i) {
		const double mel = i == n_points - 1 ? mel_max : mel_min + i * step;
		points[static_cast<size_t>(i)] = mel_to_hz(mel);
	}

	const int n_bins = config.n_bins();
	std::vector<double> filterbank(static_cast<size_t>(config.n_mels * n_bins));
	for (int m = 0; m < config.n_mels; ++m) {
		const double lower = points[static_cast<size_t>(m)];
		const double center = points[static_cast<size_t>(m + 1)];
		const double upper = points[static_cast<size_t>(m + 2)];
		for (int k = 0; k < n_bins; ++k) {
			const double freq = static_cast<double>(k * config.sample_rate) / config.n_fft;
			const double rising = (freq - lower) / (center - lower);
			const double falling = (upper - freq) / (upper - center);
			filterbank[static_cast<size_t>(m * n_bins + k)] = std::max(0.0, std::min(rising, falling));
		}
	}
	return filterbank;
}

std::vector<double> fix_length(const std::vector<double>& x, size_t length) {
	if (x.size() >= length) {
		const auto start = static_cast<std::ptrdiff_t>((x.size() - length) / 2);
		return std::vector<double>(x.begin() + start, x.begin() + start + static_cast<std::ptrdiff_t>(length));
	}
	std::vector<double> padded(x);
	padded.resize(length, 0.0);
	return padded;
}

LogMel::LogMel(const PreprocessingConfig& config)
	: config_(config), window_(hann_periodic(config.n_fft)), filterbank_(mel_filterbank(config)) {
	const int n = config_.n_fft;
	const int n_bins = config_.n_bins();
	std::vector<double> base_cos(static_cast<size_t>(n));
	std::vector<double> base_sin(static_cast<size_t>(n));
	for (int i = 0; i < n; ++i) {
		base_cos[static_cast<size_t>(i)] = std::cos(2.0 * kPi * i / n);
		base_sin[static_cast<size_t>(i)] = std::sin(2.0 * kPi * i / n);
	}
	cos_.resize(static_cast<size_t>(n_bins * n));
	sin_.resize(static_cast<size_t>(n_bins * n));
	for (int k = 0; k < n_bins; ++k) {
		for (int i = 0; i < n; ++i) {
			const auto idx = static_cast<size_t>((static_cast<long long>(k) * i) % n);
			cos_[static_cast<size_t>(k * n + i)] = base_cos[idx];
			sin_[static_cast<size_t>(k * n + i)] = base_sin[idx];
		}
	}
}

std::vector<float> LogMel::operator()(const std::vector<double>& samples) const {
	const std::vector<double> x = fix_length(samples, static_cast<size_t>(config_.clip_samples));
	const int n = config_.n_fft;
	const int n_bins = config_.n_bins();
	const int n_frames = config_.n_frames();
	const int n_mels = config_.n_mels;

	std::vector<double> frame(static_cast<size_t>(n));
	std::vector<double> power(static_cast<size_t>(n_bins));
	std::vector<float> out(static_cast<size_t>(n_mels * n_frames));
	for (int t = 0; t < n_frames; ++t) {
		const double* src = x.data() + static_cast<size_t>(t) * static_cast<size_t>(config_.hop_length);
		for (int i = 0; i < n; ++i) {
			frame[static_cast<size_t>(i)] = src[i] * window_[static_cast<size_t>(i)];
		}
		for (int k = 0; k < n_bins; ++k) {
			const double* c = cos_.data() + static_cast<size_t>(k * n);
			const double* s = sin_.data() + static_cast<size_t>(k * n);
			double re = 0.0;
			double im = 0.0;
			for (int i = 0; i < n; ++i) {
				re += frame[static_cast<size_t>(i)] * c[i];
				im -= frame[static_cast<size_t>(i)] * s[i];
			}
			power[static_cast<size_t>(k)] = re * re + im * im;
		}
		for (int m = 0; m < n_mels; ++m) {
			const double* weights = filterbank_.data() + static_cast<size_t>(m * n_bins);
			double energy = 0.0;
			for (int k = 0; k < n_bins; ++k) {
				energy += weights[k] * power[static_cast<size_t>(k)];
			}
			out[static_cast<size_t>(m * n_frames + t)] = static_cast<float>(std::log(std::max(energy, config_.log_floor)));
		}
	}
	return out;
}

}
