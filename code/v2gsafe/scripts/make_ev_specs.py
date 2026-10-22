"""Derive the constant-efficiency EV specification file (DD4) from EV2Gym's ``ev_specs_v2g_enabled2024.json``.

DD4 (design_decisions.md): a scalar efficiency per EV model, realised through our own spec JSON whose
``3ph_ch_efficiency`` table is constant across current levels (safety_layer.md, Corollary 2 item 2).
EV2Gym then builds a constant ``{current: eta%}`` dictionary per EV, and copies it as the discharge table
(EV2Gym ``utils.py`` spawn_single_EV), so zeta_c = zeta_d = constant per model.

Choice of the constant: the 3-phase efficiency of the original table at the current level closest to the
EV's nominal AC current I_nom = P_ev * 1000 / (400 V * sqrt(3)) (i.e. the efficiency at which the EV charges at
full power in the original data). Zero (missing) entries are skipped, as EV2Gym does when it fills the table.

Usage::

    python -m v2gsafe.scripts.make_ev_specs [--src <ev2gym json>] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "specs" / "ev_specs_const_eta.json"


def ev2gym_spec_path() -> Path:
    """Path of EV2Gym's V2G-enabled spec file inside the installed ``ev2gym`` package."""
    import ev2gym  # noqa: WPS433 (import inside function: optional dependency at import time)

    return Path(ev2gym.__file__).resolve().parent / "data" / "ev_specs_v2g_enabled2024.json"


def constant_efficiency(spec: dict, voltage: float = 400.0, phases: int = 3) -> float:
    """3-phase efficiency (percent) at the current level closest to the EV's nominal AC current."""
    i_nom = spec["max_ac_charge_power"] * 1000.0 / (voltage * math.sqrt(phases))
    levels = [(c, e) for c, e in zip(spec["ch_current"], spec["3ph_ch_efficiency"]) if e > 0]
    if not levels:
        raise ValueError("spec without non-zero 3-phase efficiencies")
    c_best, e_best = min(levels, key=lambda ce: (abs(ce[0] - i_nom), -ce[0]))
    return float(e_best)


def make_specs(src: Path) -> tuple[dict, dict]:
    """Return (new spec dict, metadata dict)."""
    specs = json.loads(Path(src).read_text())
    out, meta = {}, {"source": str(src), "rule": "3ph efficiency at the current level closest to I_nom", "models": {}}
    for name, sp in specs.items():
        eta = constant_efficiency(sp)
        new = dict(sp)
        new["3ph_ch_efficiency"] = [eta] * len(sp["ch_current"])
        out[name] = new
        meta["models"][name] = {"eta_percent": eta, "original_3ph": sp["3ph_ch_efficiency"],
                                "battery_capacity": sp["battery_capacity"],
                                "max_ac_charge_power": sp["max_ac_charge_power"],
                                "max_ac_discharge_power": sp["max_ac_discharge_power"]}
    return out, meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=None, help="EV2Gym spec JSON (default: installed ev2gym package)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    src = args.src or ev2gym_spec_path()
    specs, meta = make_specs(src)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(specs, indent=2) + "\n")
    meta_path = args.out.with_suffix(".meta.json")
    meta["source"] = Path(src).name
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    for name, m in meta["models"].items():
        print(f"{name:18s} eta = {m['eta_percent']:.0f} %")
    print(f"wrote {args.out} and {meta_path}")


if __name__ == "__main__":
    main()
