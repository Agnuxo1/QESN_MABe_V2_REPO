"""CLI smoke tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from qesn.cli import build_parser, main


def test_parser_builds() -> None:
    p = build_parser()
    assert p.prog == "qesn-mabe"


def test_info_runs(capsys) -> None:
    rc = main(["info"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "qesn-mabe" in out
    assert "Quantum-INSPIRED" in out or "quantum-inspired" in out.lower()


def test_status_runs(capsys) -> None:
    rc = main(["status"])
    assert rc == 0
    assert "qesn-mabe" in capsys.readouterr().out


def test_simulate_writes_json(tmp_path) -> None:
    out = tmp_path / "sim.json"
    rc = main([
        "simulate",
        "--lattice-size", "8",
        "--steps", "5",
        "--seed", "1",
        "--output", str(out),
    ])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["config"]["lattice_size"] == 8
    assert data["config"]["steps"] == 5
    assert "energy_start" in data
    assert "energy_end" in data
    assert "disclaimer" in data


def test_simulate_rejects_bad_coupling() -> None:
    with pytest.raises(ValueError):
        main([
            "simulate",
            "--lattice-size", "4",
            "--steps", "1",
            "--coupling", "0.9",
        ])
