"""check_O0.py -- numerical check of Lemma O0 (formulation.md Sec. 3.2): EV2Gym's maximum charging power satisfies
Pbar(E) <= min{Pnom, rho_kappa (E_cap - E)}, rho_kappa = (1 - exp(-kappa))/dt, with equality outside the one-step
transition band, and sup_E Pbar/(E_cap - E) = rho_kappa. Uses the closed form (S7) (= EV._charge to 1e-14, check_A1.py)."""
import json, itertools, numpy as np, sys
sys.path.insert(0, "/home/claude/work/method")
from projection_ref import phi_ev2gym, P_CS
specs = json.load(open("/home/claude/work/ext/EV2Gym/ev2gym/data/ev_specs_v2g_enabled2024.json"))
worst_excess, worst_sup_rel, worst_eq_off_band, n = -np.inf, 0.0, 0.0, 0
for (name, sp), zeta, ts, dt_min in itertools.product(specs.items(), [0.85, 0.9, 0.95, 1.0],
                                                     [0.7, 0.75, 0.8, 0.85, 0.9], [5, 15]):
    C = sp["battery_capacity"]; dt = dt_min / 60; Pn = zeta * min(sp["max_ac_charge_power"], P_CS); m = 5
    E = np.linspace(5.0, C * (1 - 1e-9), 20001)
    Pbar = (phi_ev2gym(E, C, Pn, ts, m, dt) - E) / dt
    d = Pn * dt / C; kappa = m * d / (1 - ts); rho = (1 - np.exp(-kappa)) / dt
    rhs = np.minimum(Pn, rho * (C - E))
    worst_excess = max(worst_excess, float(np.max(Pbar - rhs)))
    worst_sup_rel = max(worst_sup_rel, abs(float(np.max(Pbar / (C - E))) / rho - 1))
    s = E / C; off = (s <= ts - d - 1e-9) | (s >= ts)
    worst_eq_off_band = max(worst_eq_off_band, float(np.max(np.abs(Pbar - rhs)[off])))
    n += 1
print(f"Lemma O0 on {n} configurations (8 models x 4 zeta x 5 s_tr x 2 dt, m=5):")
print(f"  max [Pbar - min(Pnom, rho_kappa (Ecap-E))] = {worst_excess:.2e} kW   (<= 0: outer bound holds)")
print(f"  max |off-band difference| = {worst_eq_off_band:.2e} kW   (equality outside the transition band)")
print(f"  max |sup Pbar/(Ecap-E) / rho_kappa - 1| = {worst_sup_rel:.2e}")
