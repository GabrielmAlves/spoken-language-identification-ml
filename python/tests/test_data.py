import numpy as np
import pandas as pd
import pytest

from lid.data import center_crop, crop_starts, feature_stats, load_labels, targets

OFFSETS = np.array([0, 498, 1100, 2500])


def test_labels_are_sorted_languages():
	labels = load_labels()
	assert labels == sorted(labels) and len(labels) == 5


def test_targets_follow_labels_order():
	manifest = pd.DataFrame({"language": ["pt_br", "de_de", "en_us"]})
	assert targets(manifest, ["de_de", "en_us", "pt_br"]).tolist() == [2, 0, 1]


def test_targets_reject_unknown_language():
	with pytest.raises(ValueError):
		targets(pd.DataFrame({"language": ["it_it"]}), ["de_de"])


def test_center_starts_match_notebook_formula():
	expected = [OFFSETS[i] + (OFFSETS[i + 1] - OFFSETS[i] - 498) // 2 for i in range(len(OFFSETS) - 1)]
	assert crop_starts(OFFSETS).tolist() == expected


def test_random_starts_stay_inside_clip_and_are_reproducible():
	starts = [crop_starts(OFFSETS, rng=np.random.default_rng(s)) for s in range(200)]
	for row in starts:
		assert (row >= OFFSETS[:-1]).all() and (row + 498 <= OFFSETS[1:]).all()
	assert np.array_equal(crop_starts(OFFSETS, rng=np.random.default_rng(7)), crop_starts(OFFSETS, rng=np.random.default_rng(7)))
	assert len({tuple(row) for row in starts}) > 1


def test_short_clip_is_rejected():
	with pytest.raises(ValueError):
		crop_starts(np.array([0, 400]))


def test_center_crop_copies_the_right_frames():
	feats = np.tile(np.arange(OFFSETS[-1], dtype=np.float16), (3, 1))
	out = center_crop(feats, OFFSETS)
	assert out.shape == (3, 3, 498) and out.dtype == np.float32
	for i, start in enumerate(crop_starts(OFFSETS)):
		assert np.array_equal(out[i, 0], feats[0, start:start + 498].astype(np.float32))


def test_feature_stats_match_numpy_with_small_chunks():
	feats = np.random.default_rng(0).normal(-5, 3, (4, 1000)).astype(np.float16)
	mean, std = feature_stats(feats, chunk=128)
	ref = feats.astype(np.float64)
	assert np.allclose(mean, ref.mean(axis=1)) and np.allclose(std, ref.std(axis=1))
