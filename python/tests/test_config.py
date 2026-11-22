import json

import pytest

from lid.config import REPO_ROOT, PreprocessingConfig


def test_json_round_trip(tmp_path):
	config = PreprocessingConfig()
	path = tmp_path / "preprocessing.json"
	config.to_json(path)
	assert PreprocessingConfig.from_json(path) == config


def test_divergent_conventions_are_rejected():
	data = PreprocessingConfig().to_dict()
	data["conventions"] = {**data["conventions"], "center": True}
	with pytest.raises(ValueError):
		PreprocessingConfig.from_dict(data)


def test_versioned_contract_matches_code():
	path = REPO_ROOT / "models" / "preprocessing.json"
	assert PreprocessingConfig.from_json(path) == PreprocessingConfig()
	assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1
