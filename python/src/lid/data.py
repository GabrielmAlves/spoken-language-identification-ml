import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lid.cache import load_eval, load_train
from lid.config import REPO_ROOT
from lid.manifest import read_manifest

N_FRAMES = 498


def load_labels(path: Path = REPO_ROOT / "models" / "labels.json") -> list[str]:
	return json.loads(path.read_text(encoding="utf-8"))


def targets(manifest: pd.DataFrame, labels: list[str]) -> np.ndarray:
	unknown = set(manifest["language"]) - set(labels)
	if unknown:
		raise ValueError(f"idiomas fora do labels.json: {sorted(unknown)}")
	return manifest["language"].map(labels.index).to_numpy(np.int64)


def crop_starts(offsets: np.ndarray, n: int = N_FRAMES, rng: np.random.Generator | None = None) -> np.ndarray:
	offsets = np.asarray(offsets, np.int64)
	lengths = np.diff(offsets)
	if (lengths < n).any():
		raise ValueError(f"{(lengths < n).sum()} clipes com menos de {n} frames")
	if rng is None:
		shift = (lengths - n) // 2
	else:
		shift = rng.integers(0, lengths - n + 1)
	return offsets[:-1] + shift


def crop(feats: np.ndarray, starts: np.ndarray, n: int = N_FRAMES) -> np.ndarray:
	out = np.empty((len(starts), feats.shape[0], n), np.float32)
	for i, start in enumerate(starts):
		out[i] = feats[:, start:start + n]
	return out


def center_crop(feats: np.ndarray, offsets: np.ndarray, n: int = N_FRAMES) -> np.ndarray:
	return crop(feats, crop_starts(offsets, n), n)


def feature_stats(feats: np.ndarray, chunk: int = 1 << 20) -> tuple[np.ndarray, np.ndarray]:
	total = np.zeros(feats.shape[0], np.float64)
	total_sq = np.zeros(feats.shape[0], np.float64)
	for start in range(0, feats.shape[1], chunk):
		block = np.asarray(feats[:, start:start + chunk], np.float64)
		total += block.sum(axis=1)
		total_sq += (block ** 2).sum(axis=1)
	mean = total / feats.shape[1]
	std = np.sqrt(np.maximum(total_sq / feats.shape[1] - mean ** 2, 0.0))
	return mean, std


@dataclass
class Dataset:
	labels: list[str]
	train_feats: np.ndarray
	train_offsets: np.ndarray
	y_train: np.ndarray
	X_dev: np.ndarray
	y_dev: np.ndarray
	train_manifest: pd.DataFrame
	dev_manifest: pd.DataFrame


def load_dataset(cache_dir: Path, repo: Path = REPO_ROOT) -> Dataset:
	manifests = repo / "manifests"
	preprocessing = repo / "models" / "preprocessing.json"
	labels = load_labels(repo / "models" / "labels.json")
	train_manifest = read_manifest(manifests / "train.csv")
	dev_manifest = read_manifest(manifests / "dev.csv")
	train_feats, train_offsets = load_train(cache_dir, manifests / "train.csv", preprocessing)
	X_dev = load_eval(cache_dir, "dev", manifests / "dev.csv", preprocessing)
	data = Dataset(
		labels=labels,
		train_feats=train_feats,
		train_offsets=train_offsets,
		y_train=targets(train_manifest, labels),
		X_dev=X_dev,
		y_dev=targets(dev_manifest, labels),
		train_manifest=train_manifest,
		dev_manifest=dev_manifest,
	)
	if len(train_offsets) - 1 != len(data.y_train) or len(X_dev) != len(data.y_dev):
		raise ValueError("cache e manifestos com número de clipes diferente")
	return data
