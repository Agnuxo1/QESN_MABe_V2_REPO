"""qesn-mabe command-line interface.

Runnable commands:
    qesn-mabe info
    qesn-mabe status
    qesn-mabe simulate --lattice-size 64 --steps 100 --output out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import QESNConfig
from .diffusion import Lattice, diffuse


DISCLAIMER = (
    "QESN is a Quantum-INSPIRED classical energy-diffusion reservoir. "
    "It is NOT a quantum computer simulation. "
    "See BENCHMARK_DISCLAIMER.md for benchmark caveats."
)


def _ensure_utf8_stdout() -> None:
    # Windows console fix.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def cmd_info(_args: argparse.Namespace) -> int:
    print(f"qesn-mabe v{__version__}")
    print("Author: Francisco Angulo de Lafuente")
    print("License: Apache-2.0")
    print(DISCLAIMER)
    root = Path(__file__).resolve().parents[2]
    disclaimer_path = root / "BENCHMARK_DISCLAIMER.md"
    print(f"Disclaimer file: {disclaimer_path}")
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    print(f"qesn-mabe v{__version__} — OK")
    print("Python reference diffusion: available")
    print("Arrow extra installed: ", end="")
    try:
        import pyarrow  # noqa: F401
        print("yes")
    except Exception:
        print("no (install with `pip install qesn-mabe[arrow]`)")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    cfg = QESNConfig(
        lattice_size=args.lattice_size,
        steps=args.steps,
        coupling=args.coupling,
        decay=args.decay,
        seed=args.seed,
    )
    cfg.validate()

    import numpy as np

    rng = np.random.default_rng(cfg.seed)
    lat = Lattice(cfg.lattice_size, cfg.lattice_size, cfg.coupling, cfg.decay)
    # Seed a few energy sources.
    for _ in range(5):
        x = int(rng.integers(0, cfg.lattice_size))
        y = int(rng.integers(0, cfg.lattice_size))
        lat.inject(x, y, 1.0)

    e_start = lat.energy()
    diffuse(lat, cfg.steps)
    e_end = lat.energy()

    result = {
        "version": __version__,
        "config": cfg.to_dict(),
        "energy_start": e_start,
        "energy_end": e_end,
        "energy_delta": e_end - e_start,
        "disclaimer": DISCLAIMER,
    }
    out = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qesn-mabe", description=DISCLAIMER)
    p.add_argument("--version", action="version", version=f"qesn-mabe {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="Show build info and disclaimer path.").set_defaults(func=cmd_info)
    sub.add_parser("status", help="Show runtime status.").set_defaults(func=cmd_status)

    sim = sub.add_parser("simulate", help="Run a synthetic diffusion simulation.")
    sim.add_argument("--lattice-size", type=int, default=64)
    sim.add_argument("--steps", type=int, default=100)
    sim.add_argument("--coupling", type=float, default=0.20)
    sim.add_argument("--decay", type=float, default=1.0)
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--output", type=str, default=None, help="Write metrics JSON to this path.")
    sim.set_defaults(func=cmd_simulate)
    return p


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
