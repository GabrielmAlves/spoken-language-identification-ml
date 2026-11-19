import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from lid.fleurs import LANGUAGES, SPLITS, fleurs_root, read_tsv, tsv_path

REPO_ROOT = Path(__file__).resolve().parents[3]
COLUMNS = ["path", "language", "split", "gender", "num_samples", "sentence_id"]
LABELS = sorted(LANGUAGES)


def build_manifest(root: Path, split: str) -> pd.DataFrame:
	rows = []
	for language in LANGUAGES:
		path = tsv_path(root, language, split)
		if not path.exists():
			raise FileNotFoundError(f"TSV ausente: {path}")
		for record in read_tsv(path):
			rows.append({
				"path": f"{language}/{split}/{record['file']}",
				"language": language,
				"split": split,
				"gender": record["gender"],
				"num_samples": int(record["num_samples"]),
				"sentence_id": int(record["sentence_id"]),
			})

	manifest = pd.DataFrame(rows, columns=COLUMNS)
	duplicated = manifest.loc[manifest["path"].duplicated(), "path"]
	if not duplicated.empty:
		raise ValueError(f"{split}: {len(duplicated)} paths duplicados, ex.: {duplicated.iloc[0]}")
	return manifest.sort_values(["language", "path"]).reset_index(drop=True)


def check_no_sentence_overlap(manifests: dict[str, pd.DataFrame]) -> None:
	names = list(manifests)
	for i, first in enumerate(names):
		for second in names[i + 1:]:
			shared = set(manifests[first]["sentence_id"]) & set(manifests[second]["sentence_id"])
			if shared:
				raise ValueError(f"{len(shared)} frases compartilhadas entre {first} e {second}")


def write_manifest(manifest: pd.DataFrame, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	manifest.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def read_manifest(path: Path) -> pd.DataFrame:
	return pd.read_csv(path, dtype={"path": str, "language": str, "split": str, "gender": str})


def write_labels(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(LABELS, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as file:
		while chunk := file.read(chunk_size):
			digest.update(chunk)
	return digest.hexdigest()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Gera os manifestos do FLEURS a partir dos TSV")
	parser.add_argument("--root", type=Path, default=fleurs_root())
	parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "manifests")
	parser.add_argument("--labels", type=Path, default=REPO_ROOT / "models" / "labels.json")
	parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	manifests = {split: build_manifest(args.root, split) for split in args.splits}
	check_no_sentence_overlap(manifests)

	for split, manifest in manifests.items():
		write_manifest(manifest, args.out_dir / f"{split}.csv")
	write_labels(args.labels)

	counts = pd.concat(manifests.values()).pivot_table(
		index="split", columns="language", values="path", aggfunc="count", margins=True, margins_name="total"
	)
	print(counts.to_string())
	print()
	for split in args.splits:
		path = args.out_dir / f"{split}.csv"
		print(f"{path.name:10} {sha256_file(path)}")
	print(f"{args.labels.name:10} {sha256_file(args.labels)}")


if __name__ == "__main__":
	main()
