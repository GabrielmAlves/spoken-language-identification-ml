import torch
from torch import nn

ARCHS = {
	"cnn_small": {"channels": [16, 32], "convs_per_block": 1, "pools": [2, 2], "dropout": 0.0},
	"vgg4": {"channels": [32, 64, 128, 128], "convs_per_block": 2, "pools": [2, 2, 2, 2], "dropout": 0.3},
	"vgg4_freqpool": {"channels": [32, 64, 128, 128], "convs_per_block": 2, "pools": [2, 2, (2, 1), (2, 1)], "dropout": 0.3},
}
NORMS = ("none", "global", "cmvn")


class GlobalNorm(nn.Module):
	def __init__(self, n_mels: int):
		super().__init__()
		self.register_buffer("mean", torch.zeros(n_mels))
		self.register_buffer("std", torch.ones(n_mels))

	def set_stats(self, mean, std, eps: float = 1e-5) -> None:
		self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
		self.std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(eps))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return (x - self.mean[:, None]) / self.std[:, None]


class Cmvn(nn.Module):
	def __init__(self, eps: float = 1e-5):
		super().__init__()
		self.eps = eps

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		centered = x - x.mean(dim=-1, keepdim=True)
		std = (centered * centered).mean(dim=-1, keepdim=True).sqrt()
		return centered / (std + self.eps)


def conv_block(c_in: int, c_out: int, convs: int, pool) -> nn.Sequential:
	layers = []
	for i in range(convs):
		layers += [
			nn.Conv2d(c_in if i == 0 else c_out, c_out, 3, padding=1, bias=False),
			nn.BatchNorm2d(c_out),
			nn.ReLU(inplace=True),
		]
	layers.append(nn.MaxPool2d(pool))
	return nn.Sequential(*layers)


class LanguageCNN(nn.Module):
	def __init__(self, arch: str = "cnn_small", n_classes: int = 5, n_mels: int = 64, norm: str = "global"):
		super().__init__()
		if arch not in ARCHS:
			raise ValueError(f"arquitetura desconhecida: {arch}")
		if norm not in NORMS:
			raise ValueError(f"normalização desconhecida: {norm}")
		spec = ARCHS[arch]
		self.norm = {"none": nn.Identity(), "global": GlobalNorm(n_mels), "cmvn": Cmvn()}[norm]
		blocks = []
		c_in = 1
		for c_out, pool in zip(spec["channels"], spec["pools"]):
			blocks.append(conv_block(c_in, c_out, spec["convs_per_block"], pool))
			c_in = c_out
		self.features = nn.Sequential(*blocks)
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.dropout = nn.Dropout(spec["dropout"])
		self.head = nn.Linear(c_in, n_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(self.norm(x))
		return self.head(self.dropout(self.pool(x).flatten(1)))


def count_params(model: nn.Module) -> int:
	return sum(p.numel() for p in model.parameters() if p.requires_grad)
