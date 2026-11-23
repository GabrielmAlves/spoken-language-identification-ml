import argparse
import json
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, f1_score

from lid.config import REPO_ROOT
from lid.data import N_FRAMES, Dataset, crop_starts, feature_stats, load_dataset
from lid.model import GlobalNorm, LanguageCNN, count_params

CSV_COLUMNS = ["exp_id", "seed", "split", "macro_f1", "accuracy", "commit", "date"]
TRAIN_DEFAULTS = {
	"epochs": 30,
	"batch_size": 64,
	"lr": 1e-3,
	"weight_decay": 1e-2,
	"patience": 8,
	"label_smoothing": 0.0,
	"amp": True,
}
SPEC_AUGMENT_DEFAULTS = {"freq_masks": 2, "freq_width": 8, "time_masks": 2, "time_width": 40}


def load_config(path: Path) -> dict:
	cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
	if "exp_id" not in cfg or "arch" not in cfg.get("model", {}):
		raise ValueError(f"{path}: config precisa de exp_id e model.arch")
	cfg["model"].setdefault("norm", "global")
	cfg["train"] = {**TRAIN_DEFAULTS, **(cfg.get("train") or {})}
	augment = cfg.get("augment") or {}
	augment.setdefault("crop", "center")
	if augment["crop"] not in ("center", "random"):
		raise ValueError(f"{path}: crop deve ser center ou random")
	spec = augment.get("spec_augment")
	augment["spec_augment"] = {**SPEC_AUGMENT_DEFAULTS, **spec} if spec else None
	cfg["augment"] = augment
	return cfg


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
	result = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
	return result.stdout.strip() or "unknown"


def default_device() -> str:
	return "cuda" if torch.cuda.is_available() else "cpu"


class DeviceData:
	def __init__(self, data: Dataset, device: str | None = None, n_frames: int = N_FRAMES):
		self.device = torch.device(device or default_device())
		self.labels = list(data.labels)
		self.n_frames = n_frames
		self.n_mels = data.train_feats.shape[0]
		self.offsets = np.asarray(data.train_offsets, np.int64)
		self.mean, self.std = feature_stats(data.train_feats)
		self.feats = torch.from_numpy(np.array(data.train_feats, np.float16)).to(self.device)
		self.y_train = torch.from_numpy(np.array(data.y_train, np.int64)).to(self.device)
		self.X_dev = torch.from_numpy(np.array(data.X_dev, np.float32)).to(self.device)
		self.y_dev = np.array(data.y_dev, np.int64)
		self.y_dev_t = torch.from_numpy(self.y_dev).to(self.device)
		self.frames = torch.arange(n_frames, device=self.device)

	@property
	def n_train(self) -> int:
		return len(self.offsets) - 1

	def batch(self, starts: np.ndarray) -> torch.Tensor:
		idx = torch.from_numpy(np.asarray(starts, np.int64)).to(self.device)[:, None] + self.frames
		return self.feats[:, idx].permute(1, 0, 2).unsqueeze(1).float()


def spec_augment(x: torch.Tensor, freq_masks: int, freq_width: int, time_masks: int, time_width: int) -> torch.Tensor:
	b, _, n_freq, n_time = x.shape
	mask = torch.zeros((b, 1, n_freq, n_time), dtype=torch.bool, device=x.device)
	for count, width, size, shape in ((freq_masks, freq_width, n_freq, (1, 1, -1, 1)), (time_masks, time_width, n_time, (1, 1, 1, -1))):
		pos = torch.arange(size, device=x.device).view(shape)
		for _ in range(count):
			w = torch.randint(0, width + 1, (b, 1, 1, 1), device=x.device)
			start = (torch.rand((b, 1, 1, 1), device=x.device) * (size - w + 1)).floor()
			mask |= (pos >= start) & (pos < start + w)
	return torch.where(mask, x.mean(dim=(2, 3), keepdim=True), x)


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 256, amp: bool = False) -> torch.Tensor:
	model.eval()
	out = []
	for i in range(0, len(X), batch_size):
		with torch.autocast(X.device.type, enabled=amp):
			out.append(model(X[i:i + batch_size].unsqueeze(1)).float())
	return torch.cat(out)


def evaluate_logits(logits: torch.Tensor, y: np.ndarray) -> dict:
	pred = logits.argmax(dim=1).cpu().numpy()
	loss = F.cross_entropy(logits, torch.from_numpy(y).to(logits.device)).item()
	return {
		"loss": loss,
		"macro_f1": f1_score(y, pred, average="macro"),
		"accuracy": accuracy_score(y, pred),
	}


def save_atomic(obj, path: Path) -> None:
	tmp = path.with_name(path.name + ".tmp")
	torch.save(obj, tmp)
	os.replace(tmp, path)


def build_model(cfg: dict, data: DeviceData) -> LanguageCNN:
	model = LanguageCNN(cfg["model"]["arch"], len(data.labels), data.n_mels, cfg["model"]["norm"])
	if isinstance(model.norm, GlobalNorm):
		model.norm.set_stats(data.mean, data.std)
	return model.to(data.device)


def train_run(cfg: dict, seed: int, data: DeviceData, run_dir: Path, csv_path: Path | None = None, commit: str | None = None, log=print) -> dict:
	run_dir = Path(run_dir)
	run_dir.mkdir(parents=True, exist_ok=True)
	done = run_dir / "done.json"
	if done.exists():
		result = json.loads(done.read_text(encoding="utf-8"))
		log(f"{run_dir.name}: já concluído (dev macro-F1 {result['macro_f1']:.4f}), pulando")
		return result

	tc, augment = cfg["train"], cfg["augment"]
	device = data.device
	amp = tc["amp"] and device.type == "cuda"
	batch_size = tc["batch_size"]
	steps = data.n_train // batch_size
	torch.backends.cudnn.benchmark = True

	set_seed(seed)
	model = build_model(cfg, data)
	opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
	sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=tc["epochs"] * steps, pct_start=0.1)
	scaler = torch.amp.GradScaler(device.type, enabled=amp)
	rng = np.random.default_rng(seed)
	state = {"epoch": 0, "best_f1": -1.0, "best_epoch": 0, "bad_epochs": 0, "history": []}

	last = run_dir / "last.pt"
	if last.exists():
		ckpt = torch.load(last, map_location=device, weights_only=False)
		model.load_state_dict(ckpt["model"])
		opt.load_state_dict(ckpt["opt"])
		sched.load_state_dict(ckpt["sched"])
		scaler.load_state_dict(ckpt["scaler"])
		rng.bit_generator.state = ckpt["rng"]
		torch.set_rng_state(ckpt["torch_rng"].cpu())
		if device.type == "cuda" and ckpt["cuda_rng"] is not None:
			torch.cuda.set_rng_state(ckpt["cuda_rng"].cpu())
		state = ckpt["state"]
		log(f"{run_dir.name}: retomando depois da época {state['epoch']}")
	else:
		(run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "seed": seed}, sort_keys=False), encoding="utf-8")
	log(f"{run_dir.name}: {count_params(model):,} parâmetros, {steps} passos por época, device {device}")

	while state["epoch"] < tc["epochs"] and state["bad_epochs"] < tc["patience"]:
		epoch = state["epoch"] + 1
		start = time.perf_counter()
		model.train()
		order = rng.permutation(data.n_train)
		starts = crop_starts(data.offsets, data.n_frames, rng if augment["crop"] == "random" else None)
		total = torch.zeros((), device=device)
		for step in range(steps):
			ids = order[step * batch_size:(step + 1) * batch_size]
			x = data.batch(starts[ids])
			y = data.y_train[torch.from_numpy(ids).to(device)]
			if augment["spec_augment"]:
				x = spec_augment(x, **augment["spec_augment"])
			with torch.autocast(device.type, enabled=amp):
				loss = F.cross_entropy(model(x), y, label_smoothing=tc["label_smoothing"])
			opt.zero_grad(set_to_none=True)
			scaler.scale(loss).backward()
			scaler.step(opt)
			scaler.update()
			sched.step()
			total += loss.detach()

		dev = evaluate_logits(predict(model, data.X_dev, amp=amp), data.y_dev)
		improved = dev["macro_f1"] > state["best_f1"]
		if improved:
			state.update(best_f1=dev["macro_f1"], best_epoch=epoch, bad_epochs=0)
			save_atomic({"model": model.state_dict(), "config": cfg, "seed": seed, "labels": data.labels, "epoch": epoch, "dev_macro_f1": dev["macro_f1"]}, run_dir / "best.pt")
		else:
			state["bad_epochs"] += 1
		row = {
			"epoch": epoch,
			"train_loss": total.item() / steps,
			"dev_loss": dev["loss"],
			"dev_macro_f1": dev["macro_f1"],
			"dev_accuracy": dev["accuracy"],
			"lr": sched.get_last_lr()[0],
			"seconds": time.perf_counter() - start,
		}
		state["history"].append(row)
		state["epoch"] = epoch
		save_atomic({
			"model": model.state_dict(),
			"opt": opt.state_dict(),
			"sched": sched.state_dict(),
			"scaler": scaler.state_dict(),
			"rng": rng.bit_generator.state,
			"torch_rng": torch.get_rng_state(),
			"cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
			"state": state,
		}, last)
		pd.DataFrame(state["history"]).to_csv(run_dir / "history.csv", index=False)
		log(f"época {epoch:2d} | loss {row['train_loss']:.3f} | dev loss {row['dev_loss']:.3f} | "
			f"dev macro-F1 {row['dev_macro_f1']:.4f} | acc {row['dev_accuracy']:.4f} | {row['seconds']:.0f} s{' *' if improved else ''}")

	best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
	model.load_state_dict(best["model"])
	logits = predict(model, data.X_dev, amp=amp)
	metrics = evaluate_logits(logits, data.y_dev)
	np.save(run_dir / "dev_logits.npy", logits.cpu().numpy())
	result = {
		"exp_id": cfg["exp_id"],
		"seed": seed,
		"split": "dev",
		"macro_f1": metrics["macro_f1"],
		"accuracy": metrics["accuracy"],
		"commit": commit or git_commit(),
		"date": datetime.now().isoformat(timespec="seconds"),
	}
	if csv_path is not None:
		csv_path = Path(csv_path)
		pd.DataFrame([result], columns=CSV_COLUMNS).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
	summary = {**result, "best_epoch": state["best_epoch"], "epochs_run": state["epoch"], "params": count_params(model)}
	done.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
	log(f"{run_dir.name}: melhor época {state['best_epoch']}, dev macro-F1 {metrics['macro_f1']:.4f}, acc {metrics['accuracy']:.4f}")
	return summary


def run_experiment(config_path: Path, seeds: list[int], data: DeviceData, runs_dir: Path, csv_path: Path | None = None) -> pd.DataFrame:
	cfg = load_config(config_path)
	results = [train_run(cfg, seed, data, Path(runs_dir) / f"{cfg['exp_id']}_s{seed}", csv_path) for seed in seeds]
	table = pd.DataFrame(results)
	std = table["macro_f1"].std(ddof=1) if len(table) > 1 else 0.0
	print(f"{cfg['exp_id']}: dev macro-F1 {table['macro_f1'].mean():.4f} ± {std:.4f} ({len(table)} sementes)")
	return table


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Treina a CNN de identificação de idioma")
	parser.add_argument("--config", type=Path, nargs="+", required=True)
	parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
	parser.add_argument("--cache", type=Path, required=True)
	parser.add_argument("--runs", type=Path, required=True)
	parser.add_argument("--csv", type=Path)
	parser.add_argument("--device")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	data = DeviceData(load_dataset(args.cache), args.device)
	for config_path in args.config:
		run_experiment(config_path, args.seeds, data, args.runs, args.csv)


if __name__ == "__main__":
	main()
