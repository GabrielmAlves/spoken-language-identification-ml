#pragma once

#include <cstddef>
#include <vector>

namespace lid {

struct PreprocessingConfig {
	int sample_rate = 16000;
	int clip_samples = 80000;
	int n_fft = 400;
	int hop_length = 160;
	int n_mels = 64;
	double f_min = 0.0;
	double f_max = 8000.0;
	double log_floor = 1e-6;

	int n_bins() const { return n_fft / 2 + 1; }
	int n_frames() const { return (clip_samples - n_fft) / hop_length + 1; }
};

double hz_to_mel(double hz);
double mel_to_hz(double mel);
std::vector<double> hann_periodic(int n);
std::vector<double> mel_filterbank(const PreprocessingConfig& config);
std::vector<double> fix_length(const std::vector<double>& x, size_t length);

class LogMel {
public:
	explicit LogMel(const PreprocessingConfig& config = PreprocessingConfig{});

	std::vector<float> operator()(const std::vector<double>& samples) const;
	const PreprocessingConfig& config() const { return config_; }

private:
	PreprocessingConfig config_;
	std::vector<double> window_;
	std::vector<double> cos_;
	std::vector<double> sin_;
	std::vector<double> filterbank_;
};

}
