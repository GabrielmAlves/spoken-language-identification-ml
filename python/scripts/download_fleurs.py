import argparse
import shutil
import tarfile
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download

from lid.fleurs import (
	LANGUAGES,
	REPO_ID,
	SAMPLE_RATE,
	SPLITS,
	fleurs_root,
	split_directory,
	tsv_path,
	tsv_wav_names,
)

TAR_HAS_SPLIT_PREFIX = True


def is_complete(root: Path, language: str, split: str) -> bool:
	tsv = tsv_path(root, language, split)
	if not tsv.exists():
		return False
	expected = tsv_wav_names(tsv)
	present = {p.name for p in split_directory(root, language, split).glob("*.wav")}
	return expected <= present


def fetch(filename: str, download_directory: Path) -> Path:
	return Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
				    filename=filename, local_dir=download_directory))


def download_tsv(language: str, split: str, root: Path) -> None:
	tag = f"[{language}/{split}]"
	destination = tsv_path(root, language, split)
	if destination.exists():
		print(f"{tag} tsv já existe, pulando...")
		return
	print(f"{tag} baixando tsv...", flush=True)
	source = fetch(f"data/{language}/{split}.tsv", root / "_downloads")
	destination.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(source, destination)


def download_split(language: str, split: str, root: Path) -> None:
	tag = f"[{language}/{split}]"
	directory = split_directory(root, language, split)

	if is_complete(root, language, split):
		print(f"{tag} completo, pulando...")
		return

	download_directory = root / "_downloads"
	print(f"{tag} baixando tsv...", flush=True)
	tsv_source = fetch(f"data/{language}/{split}.tsv", download_directory)
	print(f"{tag} baixando áudio...", flush=True)
	tar_source = fetch(f"data/{language}/audio/{split}.tar.gz", download_directory)

	print(f"{tag} extraindo...", flush=True)
	destination = root / language if TAR_HAS_SPLIT_PREFIX else directory
	destination.mkdir(parents=True, exist_ok=True)
	with tarfile.open(tar_source, "r:gz") as tar_file:
		tar_file.extractall(destination, filter="data")

	directory.mkdir(parents=True, exist_ok=True)
	shutil.copy2(tsv_source, tsv_path(root, language, split))
	tar_source.unlink()

	if not is_complete(root, language, split):
		raise RuntimeError(f"{tag} incompleto após extração; "
				   f"confira TAR_HAS_SPLIT_PREFIX e o conteúdo de {directory}")
	print(f"{tag} ok", flush=True)


def summarize(root: Path, languages: list[str], splits: list[str]) -> None:
	print(f"\n{'idioma':8}{'split':7}{'wavs':>6} amostras ok?")
	for language in languages:
		for split in splits:
			wavs = sorted(split_directory(root, language, split).glob("*.wav"))
			if not wavs:
				print(f"{language:8}{split:7}{'AUSENTE':>6}")
				continue
			sample = wavs[:: max(1, len(wavs) // 5)][:5]
			ok = all(sf.info(w).samplerate == SAMPLE_RATE and sf.info(w).channels == 1
				 for w in sample)
			print(f"{language:8}{split:7}{len(wavs):6}  {'sim' if ok else 'NÃO'}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Baixa o FLEURS do Hugging Face")
	parser.add_argument("--root", type=Path, default=fleurs_root())
	parser.add_argument("--languages", nargs="+", default=LANGUAGES, choices=LANGUAGES)
	parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
	parser.add_argument("--tsv-only", action="store_true")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	root = args.root
	root.mkdir(parents=True, exist_ok=True)
	print(f"destino: {root}")

	for split in args.splits:
		for language in args.languages:
			if args.tsv_only:
				download_tsv(language, split, root)
			else:
				download_split(language, split, root)

	shutil.rmtree(root / "_downloads", ignore_errors=True)

	if not args.tsv_only:
		summarize(root, args.languages, args.splits)


if __name__ == "__main__":
	main()
