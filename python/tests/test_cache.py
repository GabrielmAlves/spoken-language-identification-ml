import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from lid.cache import build_eval, build_train, load_eval, load_train, read_wav, write_meta
from lid.config import PreprocessingConfig
from lid.features import log_mel, log_mel_full
from lid.manifest import write_manifest

CONFIG = PreprocessingConfig()
LENGTHS = [50000, 80000, 123457]


@pytest.fixture
def dataset(tmp_path):
	root = tmp_path / "fleurs"
	rng = np.random.default_rng(0)
	rows = []
	for i, n in enumerate(LENGTHS):
		rel = f"pt_br/dev/{i}.wav"
		(root / rel).parent.mkdir(parents=True, exist_ok=True)
		sf.write(root / rel, 0.1 * rng.standard_normal(n), 16000, subtype="PCM_16")
		rows.append({"path": rel, "language": "pt_br", "split": "dev", "gender": "MALE", "num_samples": n, "sentence_id": i})
	manifest_path = tmp_path / "dev.csv"
	write_manifest(pd.DataFrame(rows), manifest_path)
	preprocessing_path = tmp_path / "preprocessing.json"
	CONFIG.to_json(preprocessing_path)
	return root, manifest_path, preprocessing_path, pd.DataFrame(rows)


def test_log_mel_full_equals_log_mel_at_clip_length():
	x = np.random.default_rng(1).standard_normal(CONFIG.audio.clip_samples)
	assert np.array_equal(log_mel_full(x, CONFIG), log_mel(x, CONFIG))


def test_log_mel_full_keeps_whole_clip():
	x = np.random.default_rng(2).standard_normal(160000)
	assert log_mel_full(x, CONFIG).shape == (64, (160000 - 400) // 160 + 1)


def test_read_wav_rejects_wrong_rate_and_stereo(tmp_path):
	sf.write(tmp_path / "a.wav", np.zeros(1000), 8000)
	sf.write(tmp_path / "b.wav", np.zeros((1000, 2)), 16000)
	with pytest.raises(ValueError):
		read_wav(tmp_path / "a.wav")
	with pytest.raises(ValueError):
		read_wav(tmp_path / "b.wav")


def test_eval_cache_round_trip(dataset, tmp_path):
	root, manifest_path, preprocessing_path, manifest = dataset
	out = tmp_path / "cache"
	out.mkdir()
	features = build_eval(manifest, root, CONFIG)
	np.save(out / "dev.npy", features)
	write_meta(out, "dev", features.shape, manifest_path, preprocessing_path)

	loaded = load_eval(out, "dev", manifest_path, preprocessing_path)
	assert loaded.shape == (len(LENGTHS), 64, 498)
	for i, path in enumerate(manifest["path"]):
		assert np.array_equal(loaded[i], log_mel(read_wav(root / path), CONFIG))


def test_train_cache_offsets_and_slices(dataset, tmp_path):
	root, manifest_path, preprocessing_path, manifest = dataset
	out = tmp_path / "cache"
	out.mkdir()
	feats, offsets = build_train(manifest, root, CONFIG)
	np.save(out / "dev_feats.npy", feats)
	np.save(out / "dev_offsets.npy", offsets)
	write_meta(out, "dev", feats.shape, manifest_path, preprocessing_path)

	feats, offsets = load_train(out, manifest_path, preprocessing_path, split="dev")
	assert offsets[0] == 0 and offsets[-1] == feats.shape[1]
	assert np.all(np.diff(offsets) >= 498)
	assert feats.dtype == np.float16
	for i, path in enumerate(manifest["path"]):
		expected = log_mel_full(read_wav(root / path), CONFIG)
		assert np.allclose(feats[:, offsets[i]:offsets[i + 1]], expected, atol=1e-2)


def test_load_rejects_changed_manifest(dataset, tmp_path):
	root, manifest_path, preprocessing_path, manifest = dataset
	out = tmp_path / "cache"
	out.mkdir()
	features = build_eval(manifest, root, CONFIG)
	np.save(out / "dev.npy", features)
	write_meta(out, "dev", features.shape, manifest_path, preprocessing_path)
	write_manifest(manifest.iloc[::-1], manifest_path)
	with pytest.raises(ValueError):
		load_eval(out, "dev", manifest_path, preprocessing_path)
