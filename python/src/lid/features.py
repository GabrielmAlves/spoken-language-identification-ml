import numpy as np

from lid.config import MelConfig, PreprocessingConfig


def hann_periodic(n: int) -> np.ndarray:
	k = np.arange(n)
	return 0.5 - 0.5 * np.cos(2 * np.pi * k / n)


def frame(x: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
	return np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop_length]


def power_spectrogram(frames: np.ndarray, window: np.ndarray, n_fft: int) -> np.ndarray:
	spectrum = np.fft.rfft(frames * window, n=n_fft, axis=-1)
	return np.abs(spectrum) ** 2


def hz_to_mel(f):
	return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
	return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_points(mel: MelConfig) -> np.ndarray:
	mels = np.linspace(hz_to_mel(mel.f_min), hz_to_mel(mel.f_max), mel.n_mels + 2)
	return mel_to_hz(mels)


def mel_filterbank(mel: MelConfig, sample_rate: int, n_fft: int) -> np.ndarray:
	freqs = np.arange(n_fft // 2 + 1) * sample_rate / n_fft
	points = mel_points(mel)
	lower = points[:-2, None]
	center = points[1:-1, None]
	upper = points[2:, None]
	rising = (freqs - lower) / (center - lower)
	falling = (upper - freqs) / (upper - center)
	return np.maximum(0.0, np.minimum(rising, falling))


def fix_length(x: np.ndarray, length: int) -> np.ndarray:
	if len(x) >= length:
		start = (len(x) - length) // 2
		return x[start:start + length]
	return np.pad(x, (0, length - len(x)))


def log_mel(x: np.ndarray, config: PreprocessingConfig, filterbank: np.ndarray | None = None) -> np.ndarray:
	if filterbank is None:
		filterbank = mel_filterbank(config.mel, config.audio.sample_rate, config.stft.n_fft)
	x = fix_length(np.asarray(x, dtype=np.float64), config.audio.clip_samples)
	frames = frame(x, config.stft.n_fft, config.stft.hop_length)
	power = power_spectrogram(frames, hann_periodic(config.stft.n_fft), config.stft.n_fft)
	mel = filterbank @ power.T
	return np.log(np.maximum(mel, config.mel.log_floor)).astype(np.float32)
