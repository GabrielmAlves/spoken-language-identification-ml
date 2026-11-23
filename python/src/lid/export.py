import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from lid.config import REPO_ROOT, PreprocessingConfig
from lid.data import load_dataset
from lid.features import num_frames
from lid.manifest import sha256_file
from lid.model import LanguageCNN

INPUT_NAME = "log_mel"
OUTPUT_NAME = "logits"


class Ensemble(nn.Module):
	def __init__(self, models: list[nn.Module]):
		super().__init__()
		self.models = nn.ModuleList(models)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return torch.stack([model(x) for model in self.models]).mean(dim=0)


def load_run(run_dir: Path, n_mels: int) -> tuple[LanguageCNN, dict]:
	ckpt = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)
	cfg = ckpt["config"]
	model = LanguageCNN(cfg["model"]["arch"], len(ckpt["labels"]), n_mels, cfg["model"]["norm"])
	model.load_state_dict(ckpt["model"])
	return model.eval(), ckpt


def load_runs(run_dirs: list[Path], n_mels: int) -> tuple[nn.Module, list[str], list[dict]]:
	models, sources, labels = [], [], None
	for run_dir in run_dirs:
		model, ckpt = load_run(run_dir, n_mels)
		if labels is not None and ckpt["labels"] != labels:
			raise ValueError(f"{run_dir}: rótulos diferentes das outras execuções")
		labels = ckpt["labels"]
		models.append(model)
		sources.append({"run": Path(run_dir).name, "epoch": ckpt["epoch"], "dev_macro_f1": ckpt["dev_macro_f1"]})
	model = models[0] if len(models) == 1 else Ensemble(models).eval()
	return model, labels, sources


def export_onnx(model: nn.Module, path: Path, n_mels: int, n_frames: int, opset: int = 18) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	example = torch.randn(2, 1, n_mels, n_frames)
	batch = torch.export.Dim("batch", min=1, max=4096)
	torch.onnx.export(
		model.eval(),
		(example,),
		str(path),
		input_names=[INPUT_NAME],
		output_names=[OUTPUT_NAME],
		opset_version=opset,
		dynamo=True,
		dynamic_shapes=({0: batch},),
	)
	import onnx
	onnx.checker.check_model(str(path), full_check=True)


@torch.no_grad()
def torch_logits(model: nn.Module, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
	return np.concatenate([model(torch.from_numpy(X[i:i + batch_size])).numpy() for i in range(0, len(X), batch_size)])


def onnx_logits(session, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
	return np.concatenate([session.run([OUTPUT_NAME], {INPUT_NAME: X[i:i + batch_size]})[0] for i in range(0, len(X), batch_size)])


def validate(path: Path, model: nn.Module, X: np.ndarray, y: np.ndarray | None = None, atol: float = 1e-4, rtol: float = 1e-3) -> dict:
	import onnxruntime as ort
	session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
	ref = torch_logits(model, X)
	got = onnx_logits(session, X)
	single = session.run([OUTPUT_NAME], {INPUT_NAME: X[:1]})[0]
	report = {
		"clips": len(X),
		"max_abs_diff": float(np.abs(got - ref).max()),
		"same_argmax": bool((got.argmax(1) == ref.argmax(1)).all()),
		"batch1_matches": bool(np.allclose(single, ref[:1], atol=atol, rtol=rtol)),
		"allclose": bool(np.allclose(got, ref, atol=atol, rtol=rtol)),
	}
	if y is not None:
		report["torch_macro_f1"] = float(f1_score(y, ref.argmax(1), average="macro"))
		report["onnx_macro_f1"] = float(f1_score(y, got.argmax(1), average="macro"))
	report["ok"] = report["allclose"] and report["same_argmax"] and report["batch1_matches"]
	return report


def fixture_indices(y: np.ndarray, per_class: int) -> np.ndarray:
	return np.concatenate([np.flatnonzero(y == c)[:per_class] for c in np.unique(y)])


def write_fixtures(out_dir: Path, X: np.ndarray, logits: np.ndarray, paths: list[str], y: np.ndarray, labels: list[str]) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)
	np.ascontiguousarray(X, "<f4").tofile(out_dir / "input.bin")
	np.ascontiguousarray(logits, "<f4").tofile(out_dir / "logits.bin")
	meta = {
		"input": {"file": "input.bin", "dtype": "float32_le", "shape": list(X.shape)},
		"logits": {"file": "logits.bin", "dtype": "float32_le", "shape": list(logits.shape)},
		"labels": labels,
		"clips": [{"path": p, "language": labels[t], "predicted": labels[int(l.argmax())]} for p, t, l in zip(paths, y, logits)],
	}
	(out_dir / "fixtures.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Exporta o modelo treinado para ONNX e valida com onnxruntime")
	parser.add_argument("--runs", type=Path, nargs="+", required=True)
	parser.add_argument("--cache", type=Path, required=True)
	parser.add_argument("--out", type=Path, required=True)
	parser.add_argument("--name", default="lid_cnn")
	parser.add_argument("--opset", type=int, default=18)
	parser.add_argument("--per-class", type=int, default=2)
	parser.add_argument("--validate-every", type=int, default=7)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	preprocessing_path = REPO_ROOT / "models" / "preprocessing.json"
	config = PreprocessingConfig.from_json(preprocessing_path)
	n_mels, n_frames = config.mel.n_mels, num_frames(config.audio.clip_samples, config)
	model, labels, sources = load_runs(args.runs, n_mels)

	onnx_path = args.out / f"{args.name}.onnx"
	export_onnx(model, onnx_path, n_mels, n_frames, args.opset)
	print(f"ONNX gravado em {onnx_path} ({onnx_path.stat().st_size / 2**20:.1f} MiB)")

	data = load_dataset(args.cache)
	if data.labels != labels:
		raise ValueError("labels.json diverge dos rótulos do checkpoint")
	X_dev = np.asarray(data.X_dev, np.float32)[:, None]
	subset = np.arange(0, len(X_dev), args.validate_every)
	report = validate(onnx_path, model, X_dev[subset], data.y_dev[subset])
	print(json.dumps(report, indent=2))
	if not report["ok"]:
		raise RuntimeError("ONNX diverge do torch")

	idx = fixture_indices(data.y_dev, args.per_class)
	X_fix = np.ascontiguousarray(X_dev[idx])
	write_fixtures(args.out / "fixtures", X_fix, torch_logits(model, X_fix), data.dev_manifest["path"].iloc[idx].tolist(), data.y_dev[idx], labels)

	meta = {
		"onnx": onnx_path.name,
		"opset": args.opset,
		"input": {"name": INPUT_NAME, "shape": ["batch", 1, n_mels, n_frames], "dtype": "float32"},
		"output": {"name": OUTPUT_NAME, "shape": ["batch", len(labels)], "dtype": "float32"},
		"labels": labels,
		"preprocessing_sha256": sha256_file(preprocessing_path),
		"sources": sources,
		"validation": report,
	}
	(args.out / "model.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
	print(f"fixtures e model.json em {args.out}")


if __name__ == "__main__":
	main()
