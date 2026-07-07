"""End-to-end smoke: 10 train steps on synthetic data, checkpoint, resume."""

import math
from pathlib import Path

import yaml

from prism_tse import train as train_mod
from prism_tse.config import Config

SMOKE_YAML = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


def _smoke_cfg(synth_corpus: Config, run_dir: Path) -> Config:
    with open(SMOKE_YAML) as f:
        cfg = Config.from_dict(yaml.safe_load(f))
    cfg.paths.data_root = synth_corpus.paths.data_root
    cfg.paths.run_dir = str(run_dir)
    cfg.train.steps = 10
    cfg.train.val_every = 5
    cfg.train.ckpt_every = 5
    return cfg


def test_smoke_train_and_resume(synth_corpus, tmp_path):
    run_dir = tmp_path / "run"
    cfg = _smoke_cfg(synth_corpus, run_dir)

    result = train_mod.main(cfg, device="cpu")
    assert result["step"] == 10
    assert math.isfinite(result["loss"])
    assert (run_dir / "ckpt" / "latest.pt").exists()
    assert (run_dir / "config.yaml").exists()

    # resume for 5 more steps
    cfg2 = _smoke_cfg(synth_corpus, run_dir)
    cfg2.train.steps = 15
    result2 = train_mod.main(cfg2, resume="auto", device="cpu")
    assert result2["step"] == 15
    assert math.isfinite(result2["loss"])
