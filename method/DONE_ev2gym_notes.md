DONE (A2, task 1, 2026-10-21): I wrote /home/claude/work/method/ev2gym_notes.md. It covers EV2Gym at commit 55ef368c (ev2gym 2.0.0, Python 3.13.16, venv /home/claude/work/ext/venv) and gives file:line references for time, actions, the EV/charger/transformer models, prices, rewards and states, baselines, metrics, scenario keys and the implications for the safety layer. All claims were checked with small scripts (scratchpad t1-t6.py); no tracked file in the repo was changed.

Key findings:
- Actions are per port in [-1,1]. P_cmd = a·32 A·400 V·√3 = a·22.17 kW, capped by the EV limit (11 kW).
- The env's reported, billed and transformer power equals the battery-side ΔE/dt. Energy accounting is therefore lossless, and η only derates power. With the shipped V2G spec tables, η depends on the commanded current.
- Two nonlinearities matter: the CC-CV taper (random transition_soc 0.7-0.9 with heterogeneous specs) and a +0.01 kWh ceil on capacity.
- Departure is deterministic. The EV is controlled for steps t_arr..t_dep inclusive (τ = t_dep − k + 1). The default target is a full battery.
- The transformer limit is never enforced, only logged.
- Departure energy is soft in the Gurobi oracle (with a units bug) and hard in the MPCs.
- Only a restricted (≤2000-variable) Gurobi licence is available, and the full-horizon oracle crashes under numpy 2. We should build our own HiGHS oracle.

Open questions are listed at the end of the notes.
