import csv
import os
from pathlib import Path

REPO_ID = "google/fleurs"
LANGUAGES = ["pt_br", "en_us", "es_419", "fr_fr", "de_de"]
SPLITS = ["dev", "test", "train"]
SAMPLE_RATE = 16000
TSV_FIELDS = ["sentence_id", "file", "raw", "normalized", "chars", "num_samples", "gender"]


def fleurs_root() -> Path:
	return Path(os.environ.get("FLEURS_ROOT", "~/data/fleurs")).expanduser()


def split_directory(root: Path, language: str, split: str) -> Path:
	return root / language / split


def tsv_path(root: Path, language: str, split: str) -> Path:
	return split_directory(root, language, split) / f"{split}.tsv"


def read_tsv(path: Path) -> list[dict]:
	records = []
	with path.open(encoding="utf-8", newline="") as file:
		reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
		for line_number, row in enumerate(reader, start=1):
			if not row:
				continue
			if len(row) != len(TSV_FIELDS):
				raise ValueError(
					f"{path}:{line_number}: esperava {len(TSV_FIELDS)} colunas, encontrou {len(row)}"
				)
			records.append(dict(zip(TSV_FIELDS, row)))
	return records


def tsv_wav_names(path: Path) -> set[str]:
	return {record["file"] for record in read_tsv(path)}
