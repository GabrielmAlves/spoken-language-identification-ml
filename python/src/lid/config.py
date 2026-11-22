import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

CONVENTIONS = {
	"crop": "center",
	"pad": "right_zeros",
	"window": "hann_periodic",
	"center": False,
	"power": 2.0,
	"mel_scale": "htk",
	"mel_norm": None,
	"log": "natural",
}


@dataclass(frozen=True)
class AudioConfig:
	sample_rate: int = 16000
	clip_samples: int = 80000


@dataclass(frozen=True)
class StftConfig:
	n_fft: int = 400
	hop_length: int = 160


@dataclass(frozen=True)
class MelConfig:
	n_mels: int = 64
	f_min: float = 0.0
	f_max: float = 8000.0
	log_floor: float = 1e-6


@dataclass(frozen=True)
class PreprocessingConfig:
	version: int = 1
	audio: AudioConfig = field(default_factory=AudioConfig)
	stft: StftConfig = field(default_factory=StftConfig)
	mel: MelConfig = field(default_factory=MelConfig)

	def to_dict(self) -> dict:
		return {**asdict(self), "conventions": CONVENTIONS}

	@classmethod
	def from_dict(cls, data: dict) -> "PreprocessingConfig":
		if data.get("conventions") != CONVENTIONS:
			raise ValueError("convenções do pré-processamento divergem da versão do código")
		return cls(
			version=data["version"],
			audio=AudioConfig(**data["audio"]),
			stft=StftConfig(**data["stft"]),
			mel=MelConfig(**data["mel"]),
		)

	def to_json(self, path: Path) -> None:
		path.parent.mkdir(parents=True, exist_ok=True)
		path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

	@classmethod
	def from_json(cls, path: Path) -> "PreprocessingConfig":
		return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Grava o contrato de pré-processamento em JSON")
	parser.add_argument("--out", type=Path, default=REPO_ROOT / "models" / "preprocessing.json")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	config = PreprocessingConfig()
	config.to_json(args.out)
	if PreprocessingConfig.from_json(args.out) != config:
		raise RuntimeError("o JSON gravado não reproduz a configuração")
	print(f"contrato v{config.version} gravado em {args.out}")


if __name__ == "__main__":
	main()
