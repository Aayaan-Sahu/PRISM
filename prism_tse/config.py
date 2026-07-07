"""Config system: nested dataclasses with YAML load/save.

Usage:
    cfg = Config.from_yaml("prism_tse/configs/default.yaml")
    cfg.save(run_dir / "config.yaml")
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class AudioCfg:
    sr: int = 16000
    n_fft: int = 512          # 32 ms window
    hop: int = 256            # 16 ms hop; must be n_fft // 2 (COLA of sqrt-Hann)
    power: float = 0.3        # magnitude compression exponent


@dataclass
class ModelCfg:
    hidden: int = 256         # feature dim of the separator body
    spk_dim: int = 128        # conditioning vector dim (output of speaker MLP)
    emb_dim: int = 192        # ECAPA embedding dim (frozen encoder output)
    n_lstm: int = 3
    freq_conv_blocks: int = 2
    freq_conv_ch: int = 8
    freq_conv_kernel: int = 5


@dataclass
class MixerCfg:
    segment_s: float = 4.0
    scenario_probs: dict = field(default_factory=lambda: {
        "target_alone": 0.10,
        "target_noise": 0.20,
        "target_1interf": 0.25,
        "target_1interf_noise": 0.20,
        "target_2interf_noise": 0.10,
        "target_absent": 0.15,
    })
    sir_db: list = field(default_factory=lambda: [-5.0, 20.0])
    snr_db: list = field(default_factory=lambda: [0.0, 25.0])
    lufs: list = field(default_factory=lambda: [-38.0, -22.0])
    clip_prob: float = 0.10
    reverb_prob: float = 0.9
    early_ms: float = 50.0    # ground truth = target * early RIR (first 50 ms)
    enroll_degrade_prob: float = 0.5
    enroll_snr_db: list = field(default_factory=lambda: [15.0, 30.0])
    min_utt_s: float = 3.0    # min duration for target / enrollment utterances
    min_interf_s: float = 1.0
    enroll_crop_s: float = 4.0
    active_thresh_db: float = -40.0  # frame active if within this of the max frame
    min_target_s: float = 1.0        # guaranteed target duration when present


@dataclass
class TrainCfg:
    batch: int = 32
    steps: int = 400_000
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup: int = 10_000
    clip: float = 5.0
    workers: int = 8
    val_every: int = 5_000
    ckpt_every: int = 5_000
    keep_last: int = 3
    seed: int = 1234
    sisdr_w: float = 1.0
    spec_w: float = 30.0
    spec_alpha: float = 8.0
    absent_w: float = 10.0
    n_val: int = 500
    log_every: int = 50
    fake_ecapa: bool = False  # offline tests: deterministic stand-in for ECAPA
    amp: bool = True          # bf16 autocast on CUDA (no GradScaler needed)


@dataclass
class PathsCfg:
    data_root: str = "data_root"
    run_dir: str = "runs/prism_tse"
    ecapa_dir: str = ""       # "" -> {data_root}/pretrained/spkrec-ecapa-voxceleb


@dataclass
class Config:
    audio: AudioCfg = field(default_factory=AudioCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    mixer: MixerCfg = field(default_factory=MixerCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        cfg = cls()
        for section_field in dataclasses.fields(cls):
            overrides = d.get(section_field.name) or {}
            section = getattr(cfg, section_field.name)
            for k, v in overrides.items():
                if not hasattr(section, k):
                    raise KeyError(f"Unknown config key: {section_field.name}.{k}")
                setattr(section, k, v)
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    # -- derived paths -----------------------------------------------------
    @property
    def ecapa_dir(self) -> str:
        if self.paths.ecapa_dir:
            return self.paths.ecapa_dir
        return str(Path(self.paths.data_root) / "pretrained" / "spkrec-ecapa-voxceleb")

    @property
    def manifests_dir(self) -> Path:
        return Path(self.paths.data_root) / "manifests"

    @property
    def segment_samples(self) -> int:
        n = int(round(self.mixer.segment_s * self.audio.sr))
        return n - (n % self.audio.hop)  # keep a whole number of hops
