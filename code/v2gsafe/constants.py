"""Physical and numerical constants shared by all modules.

Values and symbols follow ``method/nomenclature.md``; every constant cites the place where it is defined.
"""
from __future__ import annotations

import math

#: Single numerical tolerance in code, kW or kWh (safety_layer.md Sec. 6.4).
EPS_NUM: float = 1e-9
#: Per-charger actuation margin of EV2Gym (rounding + discharge dead-band), kW (DD16, safety_layer.md Sec. 6.3).
EPS_Q: float = 0.008
#: Saturation threshold of the actuation map, kW (safety_layer.md Sec. 6.3).
EPS_SAT: float = 1e-3
#: Charger maximum power 32 A * 400 V * sqrt(3) / 1000 = 22.17 kW (formulation.md Sec. 1.2).
P_CS: float = 32.0 * 400.0 * math.sqrt(3) / 1000.0
#: Threshold-table bisection tolerance, kWh (algorithm.md Alg. 6).
THRESHOLD_TOL: float = 1e-10

#: Feature / action normalisation constants (formulation.md Sec. 4.2-4.3).
E_REF: float = 100.0   # kWh
P_REF: float = 11.0    # kW
C_REF: float = 0.25    # EUR/kWh
H_F: int = 8           # forecast window, steps

#: Feature dimensions (formulation.md Sec. 4.2).
F_C: int = 12
F_G: int = 38

#: Reporting threshold for caps, admission reductions and unserved EVs, kWh (EV2Gym rounding unit; formulation 1.3).
REPORT_TOL_KWH: float = 0.01

#: Penalty weights of the penalty-based baselines, EUR/kWh (DD22, algorithm.md Alg. 5).
W_E_DEFAULT: float = 10.0
W_P_DEFAULT: float = 10.0
