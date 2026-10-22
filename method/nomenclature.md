# Nomenclature

Agent A2 (Method and Theory). Version 5, 2026-10-22, revised after the internal proof check, the implementation feedback and internal review round 1 (renamed in v5: conformal scores $S_j\to e_j$, minibatch $B_{\mathrm{mb}}\to n_{\mathrm{mb}}$, admission taper slope $\rho_i\to\varrho_i^{\mathrm{in}}$).

This file defines one symbol per concept for `formulation.md`, `safety_layer.md` and `algorithm.md`.
- Section references: F = `formulation.md`, S = `safety_layer.md`, Alg. = `algorithm.md` sections Alg. 1-7.
- "A1"-"A5" always denote assumptions.
- Unit "–" means dimensionless.
- Terms follow `style/terminology.md`.

## Indices and sets

| Symbol | Meaning | Unit | Defined in |
|---|---|---|---|
| $i$ | Charger index. While an EV is connected, its quantities carry the charger's index ("EV $i$" = the session at charger $i$). In episode-level statements (oracle LP), $i$ indexes sessions in $\mathcal V$ | – | F 1.1 |
| $t$ | Decision step, $t\in\{0,\dots,T-1\}$; the action of step $t$ acts on $[t\Delta t,(t+1)\Delta t)$ | – | F 1.1 |
| $k$ | Number of future steps (argument of $R_i$, $B_i$, $\Psi_{i,k}$); also the lag of a forecast | – | S 2.2 |
| $j$ | Local index: sorted breakpoints (S 4.2), calibration samples (S 8) | – | S 4.2, S 8 |
| $\mathcal N=\{1,\dots,N\}$ | Chargers of the station | – | F 1.1 |
| $\mathcal N(t)$ | Occupied chargers at step $t$: $\{i: t_i^{\mathrm{arr}}\le t\le d_i\}$ | – | F 1.2 |
| $n(t)$ | $\lvert\mathcal N(t)\rvert$, number of connected EVs | – | F 1.2 |
| $\mathcal V$ | All EV sessions of an episode | – | F 3 |
| $\mathcal F$ | Free set of the projection, $\{i: l_i<x_i-\lambda^{*}<u_i\}$ | – | S 4.1 (Prop. 1) |
| $\varpi$ | Permutation of the chargers (Lemma N1) | – | F 5 |

## Time and station parameters

| Symbol | Meaning | Unit | Defined in |
|---|---|---|---|
| $N$ | Number of chargers (one port each, DD5) | – | F 1.1 |
| $N_{\max}$ | Padding size of a minibatch (largest $N$ in it) | – | Alg. 2 |
| $\Delta t$ | Step length (0.25 h main, 1/12 h sensitivity, DD13) | h | F 1.1 |
| $T$ | Horizon (episode length) | steps | F 1.1 |
| $N_{\mathrm{day}}$ | Steps per day, $24/\Delta t$ | steps | F 4.2 |
| $P_{\mathrm{tr}}(t)$ | Transformer rating (EV2Gym `tr.max_power[t]`); $P_{\mathrm{tr}}^{\mathrm{nom}}=\max_tP_{\mathrm{tr}}(t)$ | kW | F 1.1 |
| $\rho_{\mathrm{tr}}$ | Transformer rating per charger, $P_{\mathrm{tr}}/N\in\{4,7\}$ (scenario tightness level) | kW | Alg. 7 |
| $P^{\mathrm{inf}}(t)$ | Inflexible load (EV2Gym `tr.inflexible_load[t]`) | kW | F 1.1 |
| $P^{\mathrm{pv}}(t)\ge 0$ | PV generation (EV2Gym `-tr.solar_power[t]`) | kW | F 1.1 |
| $C(t)$ | Forward transformer limit for the EV aggregate, $P_{\mathrm{tr}}-P^{\mathrm{inf}}+P^{\mathrm{pv}}$ (DD7) | kW | F 1.4 |
| $C_{\mathrm{rev}}(t)$ | Reverse (export) limit for the EV aggregate, $P_{\mathrm{tr}}+P^{\mathrm{inf}}-P^{\mathrm{pv}}$ (DD7) | kW | F 1.4 |
| $\varepsilon_q$ | Per-charger actuation margin of EV2Gym (rounding and dead-band), 0.008 | kW | S 6.3 |
| $\varepsilon_{\mathrm{sat}}$ | Saturation threshold of the actuation map, 0.001 | kW | S 6.3 |
| $\epsilon_{\mathrm{num}}$ | Single numerical tolerance in code, $10^{-9}$ | kW or kWh | S 6.4 |
| $\varepsilon_C(t)$ | Margin on the forward limit, $n(t)\varepsilon_q$ (0 in the exact model; DD16) | kW | S 6.3 |
| $\tilde C(t)$ | Forward limit used by the safety layer, $C(t)-\varepsilon_C(t)$ | kW | S 6.3 |
| $c^{\mathrm{ch}}(t)$ | Price paid for charging energy (= `-env.charge_prices[0,t]`) | €/kWh | F 1.1 |
| $c^{\mathrm{dis}}(t)$ | Price received for discharged energy (= `env.discharge_prices[0,t]`) | €/kWh | F 1.1 |
| $f^{\mathrm{dis}}$ | EV2Gym `discharge_price_factor`, $c^{\mathrm{dis}}=f^{\mathrm{dis}}c^{\mathrm{ch}}$ (1 in our scenarios) | – | F 1.1 |

## EV model

| Symbol | Meaning | Unit | Defined in |
|---|---|---|---|
| $t_i^{\mathrm{arr}}$ | Arrival step = first controlled step (EV2Gym `time_of_arrival`) | – | F 1.2 |
| $d_i$ | Departure step = **last controlled step** (EV2Gym `time_of_departure`); the EV leaves after the action of step $d_i$ | – | F 1.2 |
| $\tau_i(t)$ | Remaining actions including the current one, $d_i-t+1$ | steps | F 1.2 |
| $E_i(t)$ | Battery energy at the start of step $t$ (before the action); departure energy is $E_i(d_i+1)$ | kWh | F 1.2 |
| $E_i^{\mathrm{cap}}$ | Usable capacity (EV2Gym `battery_capacity`) | kWh | F 1.2 |
| $E_i^{\min}$ | Minimum energy (EV2Gym `min_battery_capacity`, 5 kWh) | kWh | F 1.2 |
| $E_i^{\mathrm{arr}}$ | Energy at arrival, $E_i(t_i^{\mathrm{arr}})$ | kWh | F 1.2 |
| $E_i^{\mathrm{des}}$ | Desired departure energy (EV2Gym `desired_capacity`) | kWh | F 1.3 |
| $E_i^{\mathrm{req}}$ | Required departure energy after the arrival cap (DD2) and admission control (DD9, DD20) | kWh | F 1.3, S 3.4 |
| $p_i(t)$ | Commanded power of charger $i$ (output of the safety layer); $p>0$ charging | kW | F 1.2 |
| $\tilde p_i(t)$ | Realised (simulator-reported) power | kW | S 6.3 |
| $p^{+},p^{-}$ | $\max(p,0)$, $\max(-p,0)$ | kW | F 1.2 |
| $\eta_i^{\mathrm c},\eta_i^{\mathrm d}$ | Charge and discharge efficiency of the energy map (general model, DD1); 1 in the EV2Gym instantiation | – | F 1.2 |
| $g_i(p)$ | Energy map, $\eta_i^{\mathrm c}p^{+}-p^{-}/\eta_i^{\mathrm d}$ | kW | F 1.2, (S1) |
| $\bar P_i(E)$ | State-dependent maximum charging power (taper and capacity included) | kW | F 1.2, (S4) |
| $\bar P_i^{\mathrm{tap}}(E)$ | Power limit including the taper but not the capacity term | kW | (S4) |
| $\bar P_i^{\mathrm{nom}}$ | Nominal maximum charging power, $\sup_E\bar P_i(E)$ | kW | F 1.2 |
| $P_i^{\mathrm{dis}}$ | Maximum discharging power (magnitude) | kW | F 1.2 |
| $p_i^{\min}(E)$ | Physical lower power limit, $-\min\{P_i^{\mathrm{dis}},\eta_i^{\mathrm d}(E-E_i^{\min})^{+}/\Delta t\}$ | kW | (S3) |
| $p_i^{\max}(E)$ | Physical upper power limit, $=\bar P_i(E)=u_i$ | kW | (S4) |
| $L$ | Downward rate bound of $\bar P^{\mathrm{tap}}$ in the sufficient condition for A1 | kW/kWh | S 2.1 |
| $\varphi_i(E)$ | One-step maximum-charging map, $E+\Delta t\,\eta_i^{\mathrm c}\bar P_i(E)$ | kWh | (S2) |
| $\Psi_{i,k}(E)$ | $k$-fold composition $\varphi_i^{k}(E)$, $\Psi_{i,0}=\mathrm{id}$; maximum energy after $k$ steps | kWh | (S6) |
| $R_i(k,E)$ | Reachable-energy operator, $\Psi_{i,k}(E)-E$; $R_i(0,\cdot)=0$ | kWh | (S5) |
| $B_i(k)$ | Backward-reachable threshold: least energy from which $E_i^{\mathrm{req}}$ is reachable in $k$ steps; $\hat B_i(k)$ its certified numerical value (S10) | kWh | (S8), (S10) |
| $l_i(t),u_i(t)$ | Lower and upper power bound of the safety layer | kW | (S11) |
| $\tilde l_i(t)$ | $l_i$ if EV $i$ is individually feasible, else $u_i$ | kW | S 7.3 |
| $\chi_i(t)$ | Laxity, $(E_i(t)-\hat B_i(\tau_i(t)))/(\eta_i^{\mathrm c}\bar P_i^{\mathrm{nom}}\Delta t)$ | steps | S 3.3 |
| $\Delta E_i^{\mathrm{sh}}$ | Departure shortfall, $(E_i^{\mathrm{req}}-E_i(d_i+1))^{+}$ | kWh | S 7.3 |
| $\Delta E_i^{\mathrm{def}}(t)$ | Feasibility deficit, $(\hat B_i(\tau_i(t))-E_i(t))^{+}$ (Lemma 3) | kWh | S 7.3 |
| $\varrho_i^{\mathrm{in}}$ | Slope of the inner (minorant) linear taper used by admission control | 1/h | S 7.4 |
| $\varrho_{\kappa,i}$ | Exact taper slope $(1-e^{-\kappa_i})/\Delta t$; $\bar P_i(E)\le\min\{\bar P_i^{\mathrm{nom}},\varrho_{\kappa,i}(E^{\mathrm{cap}}-E)\}$ (Lemma O0) | 1/h | F 3.2 |
| $\varrho_i^{\mathrm{out}}$ | Outer taper slope in the oracle row (O4c), $1.01\,\varrho_{\kappa,i}$ in code | 1/h | F 3.2 |
| $\Psi_i^{\mathrm{LP}}(E)$ | Maximum departure energy reachable in the admission-LP model (exact first step, minorant afterwards) | kWh | S 7.4 |
| $E_i^{\mathrm{tgt}}$ | Departure target inside the admission LP (DD25): $E_j^{\mathrm{req}}$ for the new EV, $\min\{E_i^{\mathrm{req}},\Psi_i^{\mathrm{LP}}(E_i(t_0))-10^{-6}\}$ for connected EVs | kWh | S 7.4 |

## EV2Gym-specific constants

| Symbol | Meaning | Unit | Defined in |
|---|---|---|---|
| $\zeta_i^{\mathrm c},\zeta_i^{\mathrm d}$ | EV2Gym `charge_efficiency`/`discharge_efficiency`, constant over current levels per EV (DD4); act only as power derating under DD1 | – | F 1.2 |
| $P^{\mathrm{cs}}$ | Charger maximum power, $32\,\mathrm A\cdot 400\,\mathrm V\cdot\sqrt3/1000=22.17$ | kW | F 1.2 |
| $P_i^{\mathrm{ev}},P_i^{\mathrm{ev,dis}}$ | EV AC charge/discharge power limits (`max_ac_charge_power`, `max_ac_discharge_power`) | kW | F 1.2 |
| C0 | Condition $P_i^{\mathrm{ev}}\le P^{\mathrm{cs}}$ (true for all 8 models) | – | S 2.3 |
| $\varsigma$ | State of charge, $E/E^{\mathrm{cap}}$ | – | S 2.3 |
| $\varsigma_i^{\mathrm{tr}}$ | Transition SoC of the taper (`transition_soc`, $U[0.7,0.9]$) | – | S 2.3 |
| $m$ | Taper multiplier (`transition_soc_multiplier`, 5; DD17) | – | S 2.3 |
| $\delta_i$ | SoC gain of one full-power step, $\bar P_i^{\mathrm{nom}}\Delta t/E_i^{\mathrm{cap}}$ | – | S 2.3 |
| $\kappa_i$ | Taper exponent, $m\delta_i/(1-\varsigma_i^{\mathrm{tr}})$ | – | S 2.3 |
| $\hat\varphi$ | Uncapped taper branch function in (S7) | – | (S7) |
| $h$ | $\hat\varphi(\varsigma)-\varsigma-\delta$ (proof of Lemma E1) | – | S 2.3 |
| $\hat\delta$ | Commanded SoC gain of a sub-maximal pilot (proof of Lemma E2) | – | S 2.3 |
| $\varsigma_x$, $\Omega$ | Shifted transition SoC of a sub-maximal pilot and $1-\varsigma_x$ (proof of Lemma E2) | – | S 2.3 |
| $z$, $z_0(m)$ | Auxiliary variable in Lemma E2; positive root of $1-e^{-mz}=z$ ($m>1$), $z_0(5)=0.99302$ | – | S 2.3 |
| $\kappa^{*}(m)$ | Exact-delivery threshold $m\,z_0(m)$; $\kappa^{*}(5)=4.965$ | – | S 2.3 |

## Policy, safety layer and learning

| Symbol | Meaning | Unit | Defined in |
|---|---|---|---|
| $a_i(t)$ | Raw action of charger $i$ (Gaussian sample) | – | F 4.3 |
| $P^{\mathrm{ref}}$ | Action and power scale, 11 | kW | F 4.3 |
| $x_i(t)$ | Raw action in kW, $P^{\mathrm{ref}}a_i(t)$ | kW | F 4.3 |
| $\mathcal K(t)$ | Feasible set $\{p: l\le p\le u,\ -C_{\mathrm{rev}}\le\mathbf 1^\top p\le\tilde C\}$ | kW | S 4.1 |
| $\Pi_t$ | Euclidean projection onto $\mathcal K(t)$ (the safety layer) | kW | S 4.1 |
| $\lambda$, $\lambda^{*}$ | Common shift (multiplier of the aggregate constraint); its optimal value | kW | S 4.1 (Prop. 1) |
| $\mathcal S(\lambda)$ | $\sum_i\operatorname{clip}(x_i-\lambda,l_i,u_i)$ | kW | (S14) |
| $\vartheta$ | Active aggregate target, $\tilde C$ or $-C_{\mathrm{rev}}$ | kW | S 4.1, S 4.2 |
| $\lambda_{(j)}$, $\mathcal S_j$ | Sorted breakpoints $\{x_i-u_i\}\cup\{x_i-l_i\}$ and $\mathcal S(\lambda_{(j)})$ | kW | S 4.2 |
| $\upsilon$ | Interpolation parameter in the proof of Prop. 1(i) | – | S 4.1 |
| $\mu^{\pm},\nu^{l},\nu^{u}$ | KKT multipliers (aggregate, box) | kW | S 4.1 |
| $D$ | Diagonal matrix of free-set indicators | – | S 4.1 (Prop. 1(iii)) |
| $J$ | Jacobian $\partial p/\partial x$ | – | (S16) |
| FB-0, FB-1, FB-2, FB-1* | Fallback cases (maximal discharge; LLF filling; maximal charging; contradictory limits) | – | S 7.2 |
| $b$ | Remaining power budget in the fallback | kW | S 7.2 |
| $\xi_i(t)\in\mathbb R^{F_c}$ | Charger-token features ($F_c=12$) | – | F 4.2 |
| $\xi_0(t)\in\mathbb R^{F_g}$ | Global-token features ($F_g=38$) | – | F 4.2 |
| $o_i(t)$ | Occupancy flag | – | F 4.2 |
| $E^{\mathrm{ref}},c^{\mathrm{ref}}$ | Normalisation constants, 100 kWh and 0.25 €/kWh | kWh, €/kWh | F 4.2 |
| $H_f$ | Forecast window, 8 | steps | F 4.2 |
| $\hat P^{\mathrm{inf}}(t+k),\hat P^{\mathrm{pv}}(t+k)$ | Forecasts (true value plus Gaussian noise, DD6) | kW | F 4.2 |
| $d_m, n_L, n_H$ | Embedding width (128), encoder layers (3), attention heads (4) | – | F 5 |
| $h_i$ | Encoder output of token $i$ | – | F 5 |
| $\theta,\omega$ | Actor and critic parameters | – | F 5 |
| $\mu_{\theta,i},\sigma_{\theta,i}$ | Mean and standard deviation of the Gaussian of charger $i$ | – | F 5 |
| $\pi_\theta,V_\omega$ | Policy and value function | –, – | F 5 |
| $s$ | MDP state (observation) | – | F 4 |
| $r(t)$ | Reward = realised profit of step $t$ | € | F 4.4 |
| $\tilde r(t)$ | Learning reward, $r(t)/(N r^{\mathrm{ref}})$, $r^{\mathrm{ref}}=c^{\mathrm{ref}}P^{\mathrm{ref}}\Delta t$ (DD15) | – | F 4.4 |
| $r^{\mathrm{pen}}(t)$ | Reward of the penalty-based baselines | € | Alg. 5 |
| $w_E,w_P$ | Penalty weights: departure shortfall and transformer overload energy (10 €/kWh each, DD22) | €/kWh | Alg. 5 |
| $\beta$ | Weight of the projection-distance regulariser | – | S 9 |
| $n_{\mathrm{mb}}$ | Minibatch size (fixed normaliser of the masked actor loss) | transitions | Alg. 2, S 9 |
| $\phi_i,\bar\phi$ | Deep Sets per-token embedding and its masked mean (context) | – | F 5 (Lemma N2) |
| $\Lambda_E,\Lambda_P$ | Lagrange multipliers of PPO-Lagrangian (shortfall, overload); $\Lambda_{\max}=1000$ | – | Alg. 5 |
| $\eta_\Lambda$ | Dual step size of PPO-Lagrangian ($\{0.5,2,8\}$, tuned) | – | Alg. 5 |
| $c_E(t),c_P(t)$ | Per-step constraint costs (departure shortfall, overload energy) | kWh | Alg. 5 |
| $\gamma$ | Discount factor | – | F 4.5 |
| $\lambda_{\mathrm{GAE}}$ | GAE parameter (distinct from $\lambda$) | – | Alg. 2 |
| $\epsilon_{\mathrm{clip}}$ | PPO clip range | – | Alg. 2 |
| $c_{\mathrm{ent}},c_{\mathrm{vf}}$ | Entropy and value-loss coefficients | – | Alg. 2 |

## Oracle and conformal extension

| Symbol | Meaning | Unit | Defined in |
|---|---|---|---|
| $J^{\mathrm{LP}}$, $J^{\mathrm{LP,ceil}}$ | Hard lexicographic oracle profit (exact model; rounding-aware), with taper rows (DD32); used for the cost-of-the-guarantee analysis | € | F 3.3 |
| $J^{\mathrm{pen}}(\pi)$ | Penalised profit of a controller: profit $-w_E\cdot$shortfall $-w_P\cdot$overload energy, $w_E=w_P=10$ €/kWh (DD32) | € | F 3.4 |
| $J^{\mathrm{pen,LP}}$ | Soft-penalty oracle; upper bound on $J^{\mathrm{pen}}$ of every controller (Prop. O2); reference of the main gap $\mathrm{gap}^{\mathrm{pen}}$ | € | F 3.4 |
| $\mathrm{Pr}(p)$ | Profit of a schedule incl. throughput cost | € | F 3.1 |
| $o^{\pm}(t)$ | Overload slacks of the soft oracle | kW | F 3.1 |
| $c^{\mathrm{thr}}$ | Throughput (degradation) cost, 0 main, 0.05 sensitivity (DD39) | €/kWh | F 3.1, F 4.4 |
| $J^{\pi}$ | Undivided episode profit of controller $\pi$ in € | € | F 3 |
| $z_i$, $z^{*}$ | Shortfall slack of session $i$; minimal total shortfall (stage 1) | kWh | F 3 |
| $G_i^{\max}$ | Bound on EV2Gym's unreported ceil gain, $0.01(d_i-t_i^{\mathrm{arr}}+1)$ | kWh | F 3 |
| $\alpha$ | Miscoverage level of the conformal bound | – | S 8 |
| $n_{\mathrm{cal}}$ | Calibration set size (per configuration, DD23) | – | S 8 |
| $\hat q(\cdot)$ | Lower-quantile predictor of the departure step | steps | S 8 |
| $e_j$ | Conformity score $\hat q(\mathbf w_j)-d_j$ | steps | S 8 |
| $\mathbf w_i$ | Predictor features of session $i$ (arrival step, weekday, arrival energy, requested energy, EV model) | – | S 8 |
| $\hat Q_{1-\alpha}$ | Conformal quantile, the $\lceil(n_{\mathrm{cal}}+1)(1-\alpha)\rceil$-th smallest score | steps | S 8 |
| $\hat d_i^{\mathrm{lo}}$ | Conformal lower bound on $d_i$ | – | S 8 |
| $\tau_i^{\mathrm{lo}}(t)$ | $\max\{1,\hat d_i^{\mathrm{lo}}-t+1\}$ | steps | S 8 |

## Assumptions and conditions (full statements in S 10)

| Label | Short form |
|---|---|
| C0 | $P_i^{\mathrm{ev}}\le P^{\mathrm{cs}}$ (EV2Gym closed form (S7), Lemmas E1 and E2) |
| A1 | $\varphi_i$ continuous and non-decreasing |
| A1′ | $\varphi_i$ 1-Lipschitz ($\bar P_i$ non-increasing); used only for Lemma 3 and the numerical room in the DD2 cap |
| A2 | $d_i$ and $E_i^{\mathrm{req}}$ known at arrival |
| A3 | One-step model: realised energy $\ge E+\Delta t\,g_i(p)$, realised power in $[p,p+\varepsilon_q]$ |
| A4 | Individual feasibility at arrival, $E_i^{\mathrm{arr}}\ge\hat B_i(\tau_i(t_i^{\mathrm{arr}}))$ |
| A5 | Aggregate per-step feasibility, $\sum_i l_i\le\tilde C$, $\sum_i u_i\ge -C_{\mathrm{rev}}$ and $\tilde C\ge-C_{\mathrm{rev}}$ (closed loop) |
| A6 | Exogenous headroom, $C(t)\ge N\varepsilon_q$ and $C_{\mathrm{rev}}(t)\ge0$ (policy-independent; Theorem 1′(a),(b)) |
