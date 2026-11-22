import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from lid.config import REPO_ROOT, PreprocessingConfig
from lid.features import log_mel, log_mel_full, mel_filterbank, num_frames
from lid.fleurs import SPLITS, fleurs_root
from lid.manifest import read_manifest, sha256_file

PROGRESS_EVERY = 1000


def read_wav(path: Path, sample_rate: int = 16000) -> np.ndarray:
	x, sr = sf.read(path)
	if sr != sample_rate:
		raise ValueError(f"{path}: taxa de amostragem {sr}, esperado {sample_rate}")
	if x.ndim != 1:
		raise ValueError(f"{path}: esperado mono, shape {x.shape}")
	return x


def _progress(split: str, i: int, total: int, start: float) -> None:
	done = i + 1
	if done % PROGRESS_EVERY == 0 or done == total:
		print(f"{split}: {done}/{total} ({time.perf_counter() - start:.0f} s)", flush=True)


def _filterbank(config: PreprocessingConfig) -> np.ndarray:
	return mel_filterbank(config.mel, config.audio.sample_rate, config.stft.n_fft)


def build_eval(manifest: pd.DataFrame, root: Path, config: PreprocessingConfig) -> np.ndarray:
	filterbank = _filterbank(config)
	n_frames = num_frames(config.audio.clip_samples, config)
	out = np.empty((len(manifest), config.mel.n_mels, n_frames), np.float32)
	start = time.perf_counter()
	for i, path in enumerate(manifest["path"]):
		out[i] = log_mel(read_wav(root / path, config.audio.sample_rate), config, filterbank)
		_progress("eval", i, len(manifest), start)
	return out


def train_offsets(manifest: pd.DataFrame, root: Path, config: PreprocessingConfig) -> np.ndarray:
	lengths = [num_frames(sf.info(root / path).frames, config) for path in manifest["path"]]
	return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)


def build_train(manifest: pd.DataFrame, root: Path, config: PreprocessingConfig) -> tuple[np.ndarray, np.ndarray]:
	filterbank = _filterbank(config)
	offsets = train_offsets(manifest, root, config)
	feats = np.empty((config.mel.n_mels, offsets[-1]), np.float16)
	start = time.perf_counter()
	for i, path in enumerate(manifest["path"]):
		x = read_wav(root / path, config.audio.sample_rate)
		mel = log_mel_full(x, config, filterbank)
		expected = offsets[i + 1] - offsets[i]
		if mel.shape[1] != expected:
			raise ValueError(f"{path}: {mel.shape[1]} frames, cabeçalho do wav prevê {expected}")
		feats[:, offsets[i]:offsets[i + 1]] = mel
		_progress("train", i, len(manifest), start)
	return feats, offsets


def write_meta(out_dir: Path, split: str, shape: tuple, manifest_path: Path, preprocessing_path: Path) -> None:
	meta = {
		"split": split,
		"shape": list(shape),
		"manifest_sha256": sha256_file(manifest_path),
		"preprocessing_sha256": sha256_file(preprocessing_path),
	}
	(out_dir / f"{split}_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def check_meta(cache_dir: Path, split: str, manifest_path: Path, preprocessing_path: Path) -> dict:
	meta = json.loads((cache_dir / f"{split}_meta.json").read_text(encoding="utf-8"))
	if meta["manifest_sha256"] != sha256_file(manifest_path):
		raise ValueError(f"cache de {split} foi gerado com outro manifesto: regenere o cache")
	if meta["preprocessing_sha256"] != sha256_file(preprocessing_path):
		raise ValueError(f"cache de {split} foi gerado com outro preprocessing.json: regenere o cache")
	return meta


def load_eval(cache_dir: Path, split: str, manifest_path: Path, preprocessing_path: Path) -> np.ndarray:
	meta = check_meta(cache_dir, split, manifest_path, preprocessing_path)
	features = np.load(cache_dir / f"{split}.npy", mmap_mode="r")
	if list(features.shape) != meta["shape"]:
		raise ValueError(f"{split}.npy com shape {features.shape}, meta diz {meta['shape']}")
	return features


def load_train(cache_dir: Path, manifest_path: Path, preprocessing_path: Path, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
	meta = check_meta(cache_dir, split, manifest_path, preprocessing_path)
	feats = np.load(cache_dir / f"{split}_feats.npy", mmap_mode="r")
	offsets = np.load(cache_dir / f"{split}_offsets.npy")
	if list(feats.shape) != meta["shape"] or offsets[-1] != feats.shape[1]:
		raise ValueError(f"{split}_feats.npy e {split}_offsets.npy não batem com o meta")
	return feats, offsets


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Gera o cache de log-Mel de um split do FLEURS")
	parser.add_argument("--root", type=Path, default=fleurs_root())
	parser.add_argument("--split", required=True, choices=SPLITS)
	parser.add_argument("--out", type=Path, required=True)
	parser.add_argument("--manifest", type=Path)
	parser.add_argument("--preprocessing", type=Path, default=REPO_ROOT / "models" / "preprocessing.json")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	manifest_path = args.manifest or REPO_ROOT / "manifests" / f"{args.split}.csv"
	config = PreprocessingConfig.from_json(args.preprocessing)
	manifest = read_manifest(manifest_path)
	args.out.mkdir(parents=True, exist_ok=True)
	start = time.perf_counter()

	if args.split == "train":
		feats, offsets = build_train(manifest, args.root, config)
		np.save(args.out / f"{args.split}_feats.npy", feats)
		np.save(args.out / f"{args.split}_offsets.npy", offsets)
		shape = feats.shape
	else:
		features = build_eval(manifest, args.root, config)
		np.save(args.out / f"{args.split}.npy", features)
		shape = features.shape

	write_meta(args.out, args.split, shape, manifest_path, args.preprocessing)
	print(f"{args.split}: shape {shape} em {time.perf_counter() - start:.0f} s -> {args.out}")


if __name__ == "__main__":
	main()
