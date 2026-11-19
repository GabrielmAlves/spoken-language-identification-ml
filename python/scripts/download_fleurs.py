import csv
import os
import shutil
import tarfile
import soundfile as sf

from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "google/fleurs"
LANGUAGES = ["pt_br", "en_us", "es_419", "fr_fr", "de_de"]
SPLITS = ["dev", "test", "train"]
TAR_HAS_SPLIT_PREFIX = True

def fleurs_root() -> Path:
	root = Path(os.environ.get("FLEURS_ROOT", "~/data/fleurs")).expanduser()
	root.mkdir(parents=True, exist_ok=True)
	return root

def tsv_wav_names(tsv: Path) -> set[str]:
	with tsv.open(encoding="utf-8", newline="") as file:
		reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
		return {row[1] for row in reader if row}

def is_complete(split_directory: Path) -> bool:
	tsv = split_directory / f"{split_directory.name}.tsv"
	if not tsv.exists():
		return False

	expected = tsv_wav_names(tsv)
	present = {p.name for p in split_directory.glob("*.wav")}
	return expected <= present

def download_split(language: str, split: str, root: Path) -> None:
	tag = f"[{language}/{split}]"
	split_directory = root / language / split

	if is_complete(split_directory):
		print(f"{tag} completo, pulando...")
		return

	download_directory = root / "_downloads"
	print(f"{tag} baixando tsv...", flush=True)
	tsv_path = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
					filename=f"data/{language}/{split}.tsv",
					local_dir=download_directory))
	print(f"{tag} baixando áudio...", flush=True)
	tar_path = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
					filename=f"data/{language}/audio/{split}.tar.gz",
					local_dir=download_directory))

	print(f"{tag} extraindo...", flush=True)

	destination = root / language if TAR_HAS_SPLIT_PREFIX else split_directory
	destination.mkdir(parents=True, exist_ok=True)

	with tarfile.open(tar_path, "r:gz") as tar_file:
		tar_file.extractall(destination, filter="data")

	split_directory.mkdir(parents=True, exist_ok=True)
	shutil.copy2(tsv_path, split_directory / f"{split}.tsv")
	tar_path.unlink()

	if not is_complete(split_directory):
		raise RuntimeError(f"{tag} incompleto após extração; "
				   f"confira TAR_HAS_SPLIT_PREFIX e o conteúdo de {split_directory}")
	print(f"{tag} ok", flush=True)

def summarize(root: Path) -> None:
	print(f"\n{'idioma':8}{'split':7}{'wavs':>6} amostras ok?")
	for language in LANGUAGES:
		for split in SPLITS:
			wavs = sorted((root / language / split).glob("*.wav"))
			if not wavs:
				print(f"{language:8}{split:7}{'AUSENTE':>6}")
				continue
			sample = wavs[:: max(1, len(wavs) // 5)][:5]
			ok = all(sf.info(w).samplerate == 16000 and sf.info(w).channels == 1
				for w in sample)
			print(f"{language:8}{split:7}{len(wavs):6}  {'sim' if ok else 'NÃO'}")

def main():
	root = fleurs_root()
	print(f"destino: {root}")

	for split in SPLITS:
		for language in LANGUAGES:
			download_split(language, split, root)
	summarize(root)

if __name__ == "__main__":
	main()

