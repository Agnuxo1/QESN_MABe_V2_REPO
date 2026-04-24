"""Config loading and validation for QESN."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class QESNConfig:
    lattice_size: int = 64
    steps: int = 100
    coupling: float = 0.20
    decay: float = 1.0
    seed: int = 42

    def validate(self) -> None:
        if self.lattice_size <= 0:
            raise ValueError("lattice_size must be positive")
        if self.steps < 0:
            raise ValueError("steps must be >= 0")
        if not (0.0 <= self.coupling <= 0.25):
            raise ValueError("coupling must be in [0, 0.25]")
        if not (0.0 < self.decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "QESNConfig":
        allowed = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        cfg = cls(**allowed)
        cfg.validate()
        return cfg


def load_config(path: str | Path) -> QESNConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return QESNConfig.from_dict(data)
