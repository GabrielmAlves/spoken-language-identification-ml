import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from lid.data import Dataset
from lid.model import ARCHS, LanguageCNN, count_params
from lid.train import DeviceData, load_config, spec_augment, train_run

N_MELS = 16


@pytest.mark.parametrize("arch", list(ARCHS))
@pytest.mark.parametrize("norm", ["none", "global", "cmvn"])
def test_output_shape(arch, norm):
	model = LanguageCNN(arch, 5, 64, norm).eval()
	assert model(torch.randn(3, 1, 64, 498)).shape == (3, 5)


def test_cnn_small_is_tiny_and_vgg4_is_larger():
	assert count_params(LanguageCNN("cnn_small")) < 10_000
	assert count_params(LanguageCNN("vgg4")) > 300_000


def test_global_norm_stats_travel_in_state_dict():
	model = LanguageCNN("cnn_small", 5, 64, "global")
	model.norm.set_stats(np.full(64, 2.0), np.full(64, 4.0))
	state = model.state_dict()
	assert torch.allclose(state["norm.mean"], torch.full((64,), 2.0))
	x = torch.full((1, 1, 64, 10), 6.0)
	assert torch.allclose(model.norm(x), torch.ones_like(x))


def test_cmvn_removes_channel_offset():
	model = LanguageCNN("cnn_small", 5, 64, "cmvn").eval()
	x = torch.randn(2, 1, 64, 498)
	assert torch.allclose(model(x), model(x + 3.0), atol=1e-4)


def test_spec_augment_masks_with_sample_mean():
	torch.manual_seed(0)
	x = torch.randn(4, 1, 64, 498)
	out = spec_augment(x, freq_masks=2, freq_width=8, time_masks=2, time_width=40)
	assert out.shape == x.shape
	changed = out != x
	assert changed.any()
	mean = x.mean(dim=(2, 3), keepdim=True).expand_as(x)
	assert torch.equal(out[changed], mean[changed])


def make_dataset(rng):
	lengths = rng.integers(498, 700, 20)
	offsets = np.concatenate([[0], np.cumsum(lengths)])
	y = np.arange(20) % 5
	feats = rng.normal(0, 1, (N_MELS, offsets[-1])).astype(np.float16)
	for i in range(20):
		feats[y[i], offsets[i]:offsets[i + 1]] += 4
	y_dev = np.arange(10) % 5
	X_dev = rng.normal(0, 1, (10, N_MELS, 498)).astype(np.float32)
	X_dev[np.arange(10), y_dev] += 4
	return Dataset(["a", "b", "c", "d", "e"], feats, offsets, y, X_dev, y_dev, pd.DataFrame(), pd.DataFrame())


def test_device_data_batch_matches_numpy_crop():
	data = make_dataset(np.random.default_rng(0))
	device_data = DeviceData(data, "cpu")
	starts = data.train_offsets[:-1][:3] + 5
	batch = device_data.batch(starts)
	assert batch.shape == (3, 1, N_MELS, 498)
	assert torch.equal(batch[1, 0], torch.from_numpy(data.train_feats[:, starts[1]:starts[1] + 498].astype(np.float32)))


def test_train_run_writes_artifacts_and_skips_when_done(tmp_path):
	config = tmp_path / "exp.yaml"
	config.write_text("exp_id: test\nmodel:\n  arch: cnn_small\ntrain:\n  epochs: 2\n  batch_size: 8\naugment:\n  crop: random\n  spec_augment: {time_width: 20}\n")
	cfg = load_config(config)
	assert cfg["augment"]["spec_augment"]["freq_masks"] == 2 and cfg["train"]["patience"] == 8
	data = DeviceData(make_dataset(np.random.default_rng(1)), "cpu")
	csv = tmp_path / "experiments.csv"
	run_dir = tmp_path / "runs" / "test_s0"
	result = train_run(cfg, 0, data, run_dir, csv, commit="abc", log=lambda *_: None)
	for name in ["best.pt", "last.pt", "history.csv", "dev_logits.npy", "config.yaml", "done.json"]:
		assert (run_dir / name).exists()
	assert len(pd.read_csv(run_dir / "history.csv")) == 2
	assert np.load(run_dir / "dev_logits.npy").shape == (10, 5)
	assert json.loads((run_dir / "done.json").read_text())["epochs_run"] == 2
	again = train_run(cfg, 0, data, run_dir, csv, commit="abc", log=lambda *_: None)
	assert again["macro_f1"] == result["macro_f1"]
	assert len(pd.read_csv(csv)) == 1
