import numpy as np
import pytest
from scipy.signal import get_window

from lid.config import PreprocessingConfig
from lid.features import fix_length, hann_periodic, hz_to_mel, log_mel, mel_filterbank, mel_points, mel_to_hz

CONFIG = PreprocessingConfig()
N_FREQS = CONFIG.stft.n_fft // 2 + 1
N_FRAMES = (CONFIG.audio.clip_samples - CONFIG.stft.n_fft) // CONFIG.stft.hop_length + 1


@pytest.fixture(scope="module")
def filterbank() -> np.ndarray:
	return mel_filterbank(CONFIG.mel, CONFIG.audio.sample_rate, CONFIG.stft.n_fft)


def test_hann_matches_scipy_periodic():
	assert np.allclose(hann_periodic(400), get_window("hann", 400, fftbins=True))


def test_mel_scale_htk_reference_and_inverse():
	assert hz_to_mel(1000.0) == pytest.approx(1000.0, abs=0.1)
	freqs = np.array([0.0, 440.0, 1000.0, 8000.0])
	assert np.allclose(mel_to_hz(hz_to_mel(freqs)), freqs)


def test_mel_points_span_band():
	points = mel_points(CONFIG.mel)
	assert len(points) == CONFIG.mel.n_mels + 2
	assert points[0] == pytest.approx(CONFIG.mel.f_min)
	assert points[-1] == pytest.approx(CONFIG.mel.f_max)
	assert np.all(np.diff(points) > 0)


def test_filterbank_shape_and_range(filterbank):
	assert filterbank.shape == (CONFIG.mel.n_mels, N_FREQS)
	assert filterbank.min() >= 0.0
	assert filterbank.max() <= 1.0
	assert np.all(filterbank.sum(axis=1) > 0)


def test_filterbank_covers_band(filterbank):
	covered = np.flatnonzero(filterbank.sum(axis=0) > 1e-9)
	assert covered[0] == 1
	assert covered[-1] == N_FREQS - 2


@pytest.mark.parametrize("length", [50000, 80000, 190000])
def test_log_mel_shape_dtype_finite(length):
	rng = np.random.default_rng(0)
	features = log_mel(0.1 * rng.standard_normal(length), CONFIG)
	assert features.shape == (CONFIG.mel.n_mels, N_FRAMES) == (64, 498)
	assert features.dtype == np.float32
	assert np.isfinite(features).all()


def test_silence_hits_log_floor():
	features = log_mel(np.zeros(CONFIG.audio.clip_samples), CONFIG)
	assert np.allclose(features, np.log(CONFIG.mel.log_floor))


def test_tone_peaks_at_matching_band(filterbank):
	t = np.arange(CONFIG.audio.clip_samples) / CONFIG.audio.sample_rate
	features = log_mel(np.sin(2 * np.pi * 1000.0 * t), CONFIG, filterbank)
	bin_1khz = round(1000.0 * CONFIG.stft.n_fft / CONFIG.audio.sample_rate)
	assert np.argmax(features.mean(axis=1)) == np.argmax(filterbank[:, bin_1khz])


def test_precomputed_filterbank_gives_same_result(filterbank):
	x = np.random.default_rng(1).standard_normal(CONFIG.audio.clip_samples)
	assert np.array_equal(log_mel(x, CONFIG), log_mel(x, CONFIG, filterbank))


def test_fix_length_center_crop_and_right_pad():
	x = np.arange(10.0)
	assert np.array_equal(fix_length(x, 4), [3.0, 4.0, 5.0, 6.0])
	assert np.array_equal(fix_length(x, 12), [*range(10), 0.0, 0.0])
	assert np.array_equal(fix_length(x, 10), x)
