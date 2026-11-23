import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from lid.export import Ensemble, export_onnx, fixture_indices, load_runs, validate, write_fixtures
from lid.model import LanguageCNN

N_MELS, N_FRAMES = 64, 498
LABELS = ["de_de", "en_us", "es_419", "fr_fr", "pt_br"]


def save_run(run_dir, arch, norm, seed):
	torch.manual_seed(seed)
	model = LanguageCNN(arch, 5, N_MELS, norm)
	if norm == "global":
		model.norm.set_stats(np.linspace(-8, -2, N_MELS), np.linspace(1, 3, N_MELS))
	run_dir.mkdir(parents=True)
	cfg = {"model": {"arch": arch, "norm": norm}}
	torch.save({"model": model.state_dict(), "config": cfg, "labels": LABELS, "epoch": 3, "dev_macro_f1": 0.5}, run_dir / "best.pt")
	return run_dir


@pytest.mark.parametrize("norm", ["global", "cmvn"])
def test_single_model_export_matches_torch(tmp_path, norm):
	run = save_run(tmp_path / "run", "cnn_small", norm, 0)
	model, labels, sources = load_runs([run], N_MELS)
	assert labels == LABELS and sources[0]["epoch"] == 3
	path = tmp_path / "m.onnx"
	export_onnx(model, path, N_MELS, N_FRAMES)
	X = np.random.default_rng(0).normal(-5, 2, (5, 1, N_MELS, N_FRAMES)).astype(np.float32)
	report = validate(path, model, X, np.arange(5))
	assert report["ok"], report


def test_ensemble_averages_logits_and_exports(tmp_path):
	runs = [save_run(tmp_path / f"r{s}", "cnn_small", "cmvn", s) for s in range(2)]
	model, _, sources = load_runs(runs, N_MELS)
	assert isinstance(model, Ensemble) and len(sources) == 2
	x = torch.randn(3, 1, N_MELS, N_FRAMES)
	with torch.no_grad():
		expected = (model.models[0](x) + model.models[1](x)) / 2
		assert torch.allclose(model(x), expected, atol=1e-6)
	path = tmp_path / "e.onnx"
	export_onnx(model, path, N_MELS, N_FRAMES)
	assert validate(path, model, x.numpy())["ok"]


def test_fixtures_roundtrip(tmp_path):
	y = np.array([0, 0, 0, 1, 1, 2, 3, 4, 4])
	idx = fixture_indices(y, 2)
	assert y[idx].tolist() == [0, 0, 1, 1, 2, 3, 4, 4]
	X = np.random.default_rng(1).normal(size=(len(idx), 1, N_MELS, N_FRAMES)).astype(np.float32)
	logits = np.random.default_rng(2).normal(size=(len(idx), 5)).astype(np.float32)
	write_fixtures(tmp_path, X, logits, [f"a/{i}.wav" for i in idx], y[idx], LABELS)
	meta = json.loads((tmp_path / "fixtures.json").read_text())
	back = np.fromfile(tmp_path / meta["input"]["file"], "<f4").reshape(meta["input"]["shape"])
	assert np.array_equal(back, X) and len(meta["clips"]) == len(idx)
