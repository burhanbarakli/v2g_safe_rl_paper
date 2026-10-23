"""THE single mapping: method-name strings (column `method`, SCHEMA_ADDITIONS.md §4) -> display names
(style/terminology.md), plotting style and grouping. Edit only here.

Display-name rules (terminology.md): "safety layer" (never shield/projection layer), "SA-Attn" for the
proposed attention policy + safety layer, "perfect-information optimum" for the oracle.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass

# Okabe-Ito colour-blind-safe palette
OI = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
    "grey": "#7F7F7F",
}


@dataclass(frozen=True)
class MethodInfo:
    name: str
    label: str            # display name for tables and legends
    group: str            # learned | heuristic | mpc | oracle | conformal | ablation | sensitivity | other
    color: str
    marker: str
    linestyle: str        # "-" = with safety layer, "--" = raw / penalty, "-." / ":" = variants
    filled: bool          # filled marker = with safety layer
    order: int
    safety_layer: bool
    learned: bool


def _m(name, label, group, color, marker, ls, filled, order, safe, learned):
    return MethodInfo(name, label, group, color, marker, ls, filled, order, safe, learned)


V, B, G, O, S, P, K, Y = (OI["vermillion"], OI["blue"], OI["green"], OI["orange"], OI["skyblue"],
                          OI["purple"], OI["black"], OI["grey"])

METHODS: dict[str, MethodInfo] = {m.name: m for m in [
    # learned, main (DD33: + PPO-Lagrangian, Deep Sets, layer without departure bound)
    _m("attn_safe", "SA-Attn (ours)", "learned", V, "o", "-", True, 0, True, True),
    _m("attn_penalty", "Attention+penalty", "learned", V, "o", "--", False, 1, False, True),
    _m("attn_lagrangian", "Attention+PPO-Lagrangian", "learned", O, "H", "-.", False, 2, False, True),
    _m("attn_proj_nolb", "Attention+safety layer without departure bound", "learned", V, "o", ":", True, 3, True, True),
    _m("deepsets_safe", "Deep Sets+safety layer", "learned", S, "p", "-", True, 4, True, True),
    _m("mlp_safe", "MLP+safety layer", "learned", B, "s", "-", True, 5, True, True),
    _m("mlp_penalty", "MLP+penalty", "learned", B, "s", "--", False, 6, False, True),
    _m("attn_safe_trainN50", "SA-Attn trained at N = 50 (reference)", "learned", Y, "o", ":", False, 7, True, True),
    # EV-GNN (DD27, DD35): in-distribution rows are re-labelled by load.py (in_distribution = true)
    _m("evgnn", "EV-GNN (zero-shot)", "learned", P, "*", "--", False, 8, False, True),
    _m("evgnn_safe", "EV-GNN+safety layer (zero-shot)", "learned", P, "*", "-", True, 9, True, True),
    _m("evgnn_id", "EV-GNN (in-distribution)", "learned", P, "*", ":", False, 10, False, True),
    _m("evgnn_safe_id", "EV-GNN+safety layer (in-distribution)", "learned", P, "*", "-.", True, 11, True, True),
    # rule-based
    _m("llf_safe", "LLF+safety layer", "heuristic", G, "P", "-", True, 20, True, False),
    _m("llf", "LLF", "heuristic", G, "P", "--", False, 21, False, False),
    _m("sllf_safe", "sLLF+safety layer", "heuristic", G, "d", "-", True, 22, True, False),
    _m("sllf", "sLLF", "heuristic", G, "d", "--", False, 23, False, False),
    _m("price_llf_safe", "Price-threshold+safety layer", "heuristic", Y, "<", "-", True, 24, True, False),
    _m("price_llf", "Price-threshold", "heuristic", Y, "<", "--", False, 25, False, False),
    _m("afap_safe", "AFAP+safety layer", "heuristic", O, "^", "-", True, 26, True, False),
    _m("afap", "AFAP", "heuristic", O, "^", "--", False, 27, False, False),
    _m("alap_safe", "ALAP+safety layer", "heuristic", S, "v", "-", True, 28, True, False),
    _m("alap", "ALAP", "heuristic", S, "v", "--", False, 29, False, False),
    _m("rr_safe", "Round Robin+safety layer", "heuristic", Y, "D", "-", True, 30, True, False),
    _m("rr", "Round Robin", "heuristic", Y, "D", "--", False, 31, False, False),
    # MPC (DD34: HiGHS port in the grid; Gurobi eMPC only on the literature configuration)
    _m("empc_highs_safe", "eMPC-HiGHS+safety layer", "mpc", K, "X", "-", True, 40, True, False),
    _m("empc_highs", "eMPC-HiGHS", "mpc", K, "X", "--", False, 41, False, False),
    _m("empc_highs_24h_safe", "eMPC-HiGHS-24h+safety layer", "mpc", K, "8", "-", True, 42, True, False),
    _m("empc_highs_24h", "eMPC-HiGHS-24h", "mpc", K, "8", "--", False, 43, False, False),
    _m("empc_safe", "eMPC (Gurobi)+safety layer", "mpc", K, "d", "-", True, 44, True, False),
    _m("empc", "eMPC (Gurobi)", "mpc", K, "d", "--", False, 45, False, False),
    # reference
    _m("oracle", "Perfect-information optimum", "oracle", "#999999", "h", ":", True, 99, True, False),
    # evaluation variants (DD31, DD33) and ablations
    _m("attn_penalty_safedeploy", "Attention+penalty, safety layer at deployment", "ablation", V, "o", "-.", True, 60, True, True),
    _m("attn_safe_noadm", "SA-Attn, no admission control", "ablation", V, "o", "-.", True, 61, True, True),
    _m("attn_penalty_noadm", "Attention+penalty, no admission control", "ablation", V, "o", ":", False, 62, False, True),
    _m("attn_safe_beta0", "SA-Attn, \u03b2 = 0", "ablation", V, "o", "-.", True, 63, True, True),
    _m("attn_safe_curriculum", "SA-Attn, curriculum", "ablation", V, "o", "-.", True, 64, True, True),
    _m("attn_safe_small", "SA-Attn, small network", "ablation", V, "o", "-.", True, 65, True, True),
    _m("attn_penalty_dense", "Attention+penalty-dense", "ablation", V, "o", ":", False, 66, False, True),
    _m("attn_penalty_w1", "Attention+penalty, w = 1", "sensitivity", V, "o", ":", False, 67, False, True),
    _m("attn_penalty_w100", "Attention+penalty, w = 100", "sensitivity", V, "o", ":", False, 68, False, True),
    # conformal (alpha variants are generated by info())
    _m("attn_safe_conf_point", "SA-Attn-point (uncalibrated)", "conformal", V, "o", ":", True, 89, True, True),
    # sensitivity (DD28)
    _m("attn_safe_dt5", "SA-Attn, \u0394t = 5 min", "sensitivity", V, "o", ":", True, 70, True, True),
]}

CONF_RE = re.compile(r"^attn_safe_conf_a(?P<alpha>[0-9.]+(?:e-?[0-9]+)?)$")

# method sets used by tables/figures (only those present in the data are used)
MAIN_LEARNED = ["attn_safe", "attn_penalty", "attn_lagrangian", "deepsets_safe", "mlp_safe", "mlp_penalty"]
SCALING_METHODS = ["attn_safe", "attn_penalty", "attn_lagrangian", "attn_proj_nolb", "deepsets_safe", "mlp_safe",
                   "mlp_penalty", "attn_safe_trainN50", "evgnn", "evgnn_safe", "evgnn_id", "evgnn_safe_id",
                   "llf_safe", "sllf_safe", "price_llf_safe", "afap_safe", "alap_safe", "rr_safe",
                   "empc_highs_safe", "empc_highs", "empc_highs_24h_safe"]
FIG_SCALING = ["attn_safe", "attn_penalty", "attn_lagrangian", "deepsets_safe", "mlp_safe", "mlp_penalty",
               "attn_safe_trainN50", "evgnn", "llf_safe", "price_llf_safe", "empc_highs_safe"]
FIG_CONSTRAINTS = ["attn_safe", "attn_penalty", "attn_lagrangian", "attn_proj_nolb", "mlp_penalty", "evgnn",
                   "llf", "empc_highs", "empc_highs_safe"]
FIG_INFERENCE = ["attn_safe", "deepsets_safe", "mlp_safe", "evgnn", "llf_safe", "price_llf_safe",
                 "empc_highs_safe", "empc_highs_24h_safe"]
ABLATION_2X2 = {("Attention", "safety layer"): "attn_safe", ("Attention", "penalty"): "attn_penalty",
                ("MLP", "safety layer"): "mlp_safe", ("MLP", "penalty"): "mlp_penalty"}
# T4 / Fig. S2 (DD45: only variants that were run; extended-set ablations were not run)
T4_ROWS = ["attn_safe", "attn_penalty", "attn_lagrangian", "attn_proj_nolb", "attn_penalty_safedeploy",
           "deepsets_safe", "mlp_safe", "mlp_penalty"]
EXTRA_ABLATIONS = ["attn_lagrangian", "attn_proj_nolb", "attn_penalty_safedeploy", "deepsets_safe"]
SENS_WEIGHT_METHODS = ["attn_penalty", "attn_penalty_w1", "attn_penalty_w100"]
SENS_METHODS = ["attn_safe", "attn_penalty", "mlp_safe", "llf_safe", "price_llf_safe", "empc_highs_safe",
                "attn_safe_dt5"]
GUARANTEE_PAIRS = [("Attention", "attn_safe", "attn_penalty"), ("MLP", "mlp_safe", "mlp_penalty")]
TRANSFER_REF = {"attn_safe": "attn_safe_trainN50"}     # DD36: transfer gap to the in-distribution reference
REFERENCE = "attn_safe"
# fallback-case mapping (SCHEMA_ADDITIONS.md §5, consistency M7): paper case -> code label, column, step code
FALLBACK_CASES = [("(i)", "FB-1*", "fb1star_steps", 3), ("(ii)", "FB-2", "fb2_steps", 2),
                  ("(iii)", "FB-1", "fb1_steps", 1), ("(iv)", "FB-0", "fb0_steps", 0)]

_warned: set[str] = set()


def conf_alpha(method: str) -> float | None:
    m = CONF_RE.match(str(method))
    return float(m.group("alpha")) if m else None


def info(method: str) -> MethodInfo:
    if method in METHODS:
        return METHODS[method]
    a = conf_alpha(method)
    if a is not None:
        return MethodInfo(method, f"SA-Attn-CP (α = {a:g})", "conformal", V, "o", ":",
                          True, 50 + a, True, True)
    if method not in _warned:
        print(f"[names] WARNING: method '{method}' not in scripts/names.py; generic style", file=sys.stderr)
        _warned.add(method)
    return MethodInfo(method, method, "other", Y, "x", "-", True, 200, False, False)


def label(method: str) -> str:
    return info(method).label


def method_order(methods) -> list[str]:
    return sorted(set(methods), key=lambda m: (info(m).order, m))
