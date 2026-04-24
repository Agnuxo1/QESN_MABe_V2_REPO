"""Unit tests for QESNConfig."""
from __future__ import annotations

import json
import pytest

from qesn.config import QESNConfig, load_config


def test_defaults_validate() -> None:
    cfg = QESNConfig()
    cfg.validate()  # must not raise


def test_to_from_dict_roundtrip() -> None:
    cfg = QESNConfig(lattice_size=32, steps=10, coupling=0.1, decay=0.95, seed=7)
    d = cfg.to_dict()
    cfg2 = QESNConfig.from_dict(d)
    assert cfg == cfg2


def test_from_dict_ignores_unknown_keys() -> None:
    cfg = QESNConfig.from_dict({"lattice_size": 8, "unknown": 42})
    assert cfg.lattice_size == 8


def test_invalid_lattice_size() -> None:
    with pytest.raises(ValueError):
        QESNConfig(lattice_size=0).validate()


def test_invalid_steps() -> None:
    with pytest.raises(ValueError):
        QESNConfig(steps=-1).validate()


def test_invalid_coupling() -> None:
    with pytest.raises(ValueError):
        QESNConfig(coupling=0.5).validate()


def test_invalid_decay() -> None:
    with pytest.raises(ValueError):
        QESNConfig(decay=0.0).validate()


def test_load_config_file(tmp_path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text(
        json.dumps({"lattice_size": 16, "steps": 5, "coupling": 0.15, "decay": 0.99, "seed": 1}),
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.lattice_size == 16
    assert cfg.steps == 5
