# Problem formulation, MDP and attention policy

Agent A2 (Method and Theory). Version 2, 2026-10-22, revised after the internal proof check (change log in `DONE_method.md`). Symbols: `nomenclature.md`. The safety layer, its bounds and all proofs about constraints are in `safety_layer.md` (cited as S x.y). EV2Gym facts are from `ev2gym_notes.md` (pinned commit `55ef368`). Binding design decisions are DD1-DD24 in `design_decisions.md`.

---

## 1 System model

### 1.1 Station, time and prices

A charge point operator (CPO) runs a station with $N$ chargers $\mathcal N=\{1,\dots,N\}$ behind one transformer. Each charger hosts at most one EV (DD5). Time is discretised into steps $t=0,\dots,T-1$ of length $\Delta t$ [h]. Main experiments use $\Delta t=0.25$ h with $T=112$, i.e. 28 h from 05:00 as in EV2Gym's `V2GProfitMax.yaml`. A sensitivity study uses $\Delta t=1/12$ h with $T=336$ (DD13). The action of step $t$ is held constant on $[t\Delta t,(t+1)\Delta t)$.

Exogenous signals per step:
- transformer rating $P_{\mathrm{tr}}(t)$ [kW];
- inflexible load $P^{\mathrm{inf}}(t)\ge0$ [kW];
- PV generation $P^{\mathrm{pv}}(t)\ge0$ [kW], with $P^{\mathrm{pv}}(t)=-$`tr.solar_power[t]`;
- energy prices $c^{\mathrm{ch}}(t)$ (paid for charging) and $c^{\mathrm{dis}}(t)$ (received for discharging) [€/kWh].

In EV2Gym, $c^{\mathrm{ch}}(t)=-$`env.charge_prices[0,t]` and $c^{\mathrm{dis}}(t)=$`env.discharge_prices[0,t]` $=f^{\mathrm{dis}}c^{\mathrm{ch}}(t)$, with $f^{\mathrm{dis}}=1$ in our scenarios. Both may be negative.

### 1.2 EV energy dynamics and per-step physical limits

**Timing (EV2Gym convention, ev2gym_notes §3).**
- EV $i$ is controlled at the steps $t=t_i^{\mathrm{arr}},\dots,d_i$ inclusive, i.e. $d_i-t_i^{\mathrm{arr}}+1$ actions.
- $d_i$ (`time_of_departure`) is the **last controlled step**. The EV leaves after the action of step $d_i$ has been applied.
- $E_i(t)$ is the energy at the start of step $t$. The **departure energy** is therefore $E_i(d_i+1)$.
- At decision step $t$ the number of remaining actions, including the current one, is
$$\tau_i(t)=d_i-t+1\ \ (\ge 1\text{ while connected}).\tag{F1}$$
- $\mathcal N(t)=\{i:\ t_i^{\mathrm{arr}}\le t\le d_i\}$ is the set of occupied chargers and $n(t)=|\mathcal N(t)|$.

**General energy update (DD1).** Let $p_i(t)$ [kW] be the grid-side power of charger $i$ ($p>0$ charging, $p<0$ V2G discharging). Then
$$E_i(t+1)=E_i(t)+\Delta t\,g_i\big(p_i(t)\big),\qquad g_i(p)=\eta_i^{\mathrm c}p^{+}-\frac{p^{-}}{\eta_i^{\mathrm d}},\tag{F2}$$
with $\eta_i^{\mathrm c},\eta_i^{\mathrm d}\in(0,1]$. The map $g_i$ is continuous, strictly increasing and piecewise linear, with $g_i(0)=0$ and inverse
$$g_i^{-1}(y)=y/\eta_i^{\mathrm c}\text{ if }y\ge0,\qquad g_i^{-1}(y)=\eta_i^{\mathrm d}y\text{ if }y<0.$$

**Per-step physical limits.**
$$p_i^{\min}(E)\le p_i(t)\le \bar P_i(E),\quad E=E_i(t),\tag{F3}$$
$$p_i^{\min}(E)=-\min\Big\{P_i^{\mathrm{dis}},\ \eta_i^{\mathrm d}\frac{(E-E_i^{\min})^{+}}{\Delta t}\Big\}.\tag{F4}$$
- $\bar P_i(E)\ge0$ is the **state-dependent maximum charging power**. It includes the charging taper and the capacity limit, $\eta_i^{\mathrm c}\bar P_i(E)\Delta t\le E_i^{\mathrm{cap}}-E$.
- $\bar P_i^{\mathrm{nom}}:=\sup_E\bar P_i(E)$ is its nominal (low-SoC) value.
- $P_i^{\mathrm{dis}}\ge0$ is the maximum discharging power.
- (F4) keeps $E_i(t+1)\ge E_i^{\min}$ whenever $E_i(t)\ge E_i^{\min}$.
- Under (F3), $E_i(t+1)\in[E_i^{\min},E_i^{\mathrm{cap}}]$.

**EV2Gym instantiation (DD1, DD4).** EV2Gym reports, bills and sends to the transformer the battery-side energy change. Our decision variable $p_i$ is this reported power, so
$$\eta_i^{\mathrm c}=\eta_i^{\mathrm d}=1,\quad g_i(p)=p,\quad E_i(t+1)=E_i(t)+\Delta t\,p_i(t).\tag{F5}$$
EV2Gym's scalar efficiencies $\zeta_i^{\mathrm c},\zeta_i^{\mathrm d}$ (one per EV model through our own spec file, DD4) only derate the limits:
- $\bar P_i^{\mathrm{nom}}=\zeta_i^{\mathrm c}\min(P^{\mathrm{cs}},P_i^{\mathrm{ev}})$;
- $P_i^{\mathrm{dis}}=\zeta_i^{\mathrm d}\min(P^{\mathrm{cs}},P_i^{\mathrm{ev,dis}})$;
- $P^{\mathrm{cs}}=22.17$ kW.

The taper (`transition_soc` $\varsigma_i^{\mathrm{tr}}\sim U[0.7,0.9]$, multiplier $m=5$, kept at 5 by DD17) enters through $\bar P_i(E)=(\varphi_i(E)-E)/\Delta t$, where $\varphi_i$ is EV2Gym's full-power one-step map. Its closed form is (S7) in S 2.3; it requires $P_i^{\mathrm{ev}}\le P^{\mathrm{cs}}$ (condition C0, true for all 8 EV models).

The paper states the consequence of this convention as a limitation: V2G arbitrage has no round-trip loss in this instantiation. The theory is stated for the general $g_i$.

**Physical meaning of (F2).** Battery energy changes by $\eta^{\mathrm c}p$ when charging and by $p/\eta^{\mathrm d}$ when discharging, while the grid sees $p$. EV2Gym's `current_capacity` is additionally rounded up to 0.01 kWh after each active step. This rounding only adds energy and is handled in S 6.3.

### 1.3 Departure energy requirement

Each EV $i$ has a desired departure energy $E_i^{\mathrm{des}}$ (EV2Gym `desired_capacity`, the full battery by default).

At arrival, the requirement is capped to what is individually reachable during the stay (DD2):
$$E_i^{\mathrm{req}}=\min\big\{E_i^{\mathrm{des}},\ y_i^{*}\big\},\qquad E_i^{\mathrm{arr}}\ge\hat B_i\big(\tau_i(t_i^{\mathrm{arr}});y_i^{*}\big).\tag{F6}$$
Here $\hat B_i(k;y)$ is the certified backward-reachable threshold for target $y$ (S 3.1). $y_i^{*}$ is the certified target returned by `cap_request` (S 3.4): it starts at $E_i^{\mathrm{arr}}+R_i(\tau_i(t_i^{\mathrm{arr}}),E_i^{\mathrm{arr}})-10^{-6}$ kWh and is lowered only by the amount needed to absorb bisection error. In exact arithmetic (F6) is DD2's $\min\{E_i^{\mathrm{des}},\,E_i^{\mathrm{arr}}+R_i(\tau_i(t_i^{\mathrm{arr}}),E_i^{\mathrm{arr}})\}$ up to $10^{-6}$ kWh. Admission control (DD9, S 7.4) may lower $E_i^{\mathrm{req}}$ further.

The **departure energy requirement** is
$$E_i(d_i+1)\ \ge\ E_i^{\mathrm{req}}.\tag{F7}$$

Remark (taper and full-battery targets): EV2Gym's taper approaches full SoC only asymptotically, $1-(1-s)e^{-\kappa}$ per step. With $E_i^{\mathrm{des}}=E_i^{\mathrm{cap}}$, the cap (F6) is therefore active for almost every EV, usually by a negligible amount (median 0.001 kWh in `verify_projection.py`, T4). Cap statistics must be reported with a threshold, e.g. "capped by more than 0.01 kWh". This threshold is EV2Gym's own rounding unit.

### 1.4 Transformer limit

The net transformer power is $P^{\mathrm{inf}}(t)-P^{\mathrm{pv}}(t)+\sum_{i\in\mathcal N}p_i(t)$. It must stay in $[-P_{\mathrm{tr}}(t),P_{\mathrm{tr}}(t)]$ (symmetric thermal limit, DD7). Equivalently, the EV aggregate must satisfy the **transformer limit**
$$-C_{\mathrm{rev}}(t)\le\sum_{i\in\mathcal N}p_i(t)\le C(t),\tag{F8}$$
$$C(t)=P_{\mathrm{tr}}(t)-P^{\mathrm{inf}}(t)+P^{\mathrm{pv}}(t),\qquad C_{\mathrm{rev}}(t)=P_{\mathrm{tr}}(t)+P^{\mathrm{inf}}(t)-P^{\mathrm{pv}}(t).$$

Properties:
- $C(t)+C_{\mathrm{rev}}(t)=2P_{\mathrm{tr}}(t)\ge0$ always.
- $C(t)<0$ only if the inflexible load exceeds $P_{\mathrm{tr}}+P^{\mathrm{pv}}$.
- $C_{\mathrm{rev}}(t)<0$ only if PV export exceeds $P_{\mathrm{tr}}+P^{\mathrm{inf}}$.

Empty chargers have $p_i(t)=0$. The safety layer uses the realised $P^{\mathrm{inf}}(t)$ and $P^{\mathrm{pv}}(t)$ of the current step (DD6).

Violation metric (DD7), in kW per step and reported as kWh, i.e. multiplied by $\Delta t$:
$$\max(0,P^{\mathrm{net}}-P_{\mathrm{tr}})+\max(0,-P^{\mathrm{net}}-P_{\mathrm{tr}}).$$

---

## 2 Online control problem

At each step $t$ the CPO chooses $p(t)\in\mathbb R^N$ using only information available at $t$:
- arrival data of connected EVs ($t_i^{\mathrm{arr}}$, $d_i$ or its conformal bound, $E_i^{\mathrm{arr}}$, $E_i^{\mathrm{req}}$, model limits);
- the current energies $E_i(t)$;
- current and past prices, plus day-ahead prices for the next $H_f$ steps;
- measured $P^{\mathrm{inf}}(t)$ and $P^{\mathrm{pv}}(t)$, and forecasts of them.

Future arrivals are unknown. The objective is the expected episode profit
$$\max\ \mathbb E\Big[\sum_{t=0}^{T-1}\Delta t\sum_{i\in\mathcal N(t)}\big(c^{\mathrm{dis}}(t)p_i^{-}(t)-c^{\mathrm{ch}}(t)p_i^{+}(t)\big)\Big]$$
subject to (F2)-(F3), (F7) for every EV and (F8) at every step.

Our controller is a learned attention policy (Sec. 5) whose raw output passes through the safety layer (S 4). The safety layer makes (F3) and (F8) hold at every step. It makes (F7) hold for every EV under the assumptions of Theorem 1 (S 5).

---

## 3 Perfect-information optimum (oracle LP, DD10)

The oracle knows the whole episode: all sessions $\mathcal V$ with $(t_i^{\mathrm{arr}},d_i,E_i^{\mathrm{arr}},E_i^{\mathrm{req}})$, all prices and the realised $C(t)$ and $C_{\mathrm{rev}}(t)$. The $E_i^{\mathrm{req}}$ is the DD2-capped value (F6) and does not include admission-control reductions. For $i\in\mathcal V$ and $t\in\mathcal T_i=\{t_i^{\mathrm{arr}},\dots,d_i\}$ it uses the variables:
- $p_i^{+}(t),p_i^{-}(t)\ge0$;
- $E_i(t)$ for $t\in\{t_i^{\mathrm{arr}},\dots,d_i+1\}$;
- $z_i\ge0$.

The constraints are:
$$
\begin{aligned}
&\text{(O1)}\ \ E_i(t_i^{\mathrm{arr}})=E_i^{\mathrm{arr}},\\
&\text{(O2)}\ \ E_i(t+1)=E_i(t)+\Delta t\,(\eta_i^{\mathrm c}p_i^{+}(t)-p_i^{-}(t)/\eta_i^{\mathrm d}),\quad t\in\mathcal T_i,\\
&\text{(O3)}\ \ E_i^{\min}\le E_i(t)\le E_i^{\mathrm{cap}},\quad t=t_i^{\mathrm{arr}}+1,\dots,d_i+1,\\
&\text{(O4)}\ \ p_i^{+}(t)\le\bar P_i^{\mathrm{nom}},\quad p_i^{-}(t)\le P_i^{\mathrm{dis}},\\
&\text{(O5)}\ \ E_i(d_i+1)+z_i\ge E_i^{\mathrm{req}},\\
&\text{(O6)}\ \ -C_{\mathrm{rev}}(t)\le\sum_{i:\,t\in\mathcal T_i}\big(p_i^{+}(t)-p_i^{-}(t)\big)\le C(t),\quad t=0,\dots,T-1.
\end{aligned}
$$

It is solved lexicographically (DD19) with HiGHS [huangfu2018parallelizing]:
- **Stage 1:** $z^{*}=\min\sum_i z_i$ subject to (O1)-(O6). This gives $z^{*}=0$ whenever all requirements are jointly feasible.
- **Stage 2:**
$$J^{\mathrm{LP}}=\max\ \sum_t\Delta t\sum_{i:\,t\in\mathcal T_i}\big(c^{\mathrm{dis}}(t)p_i^{-}(t)-c^{\mathrm{ch}}(t)p_i^{+}(t)\big)\quad\text{s.t. (O1)-(O6)},\ \sum_i z_i\le z^{*}+10^{-6}.$$

**EV2Gym instantiation.** With $\eta^{\mathrm c}=\eta^{\mathrm d}=1$ and $f^{\mathrm{dis}}=1$, the objective depends only on the net power $p_i=p_i^{+}-p_i^{-}$. One variable $p_i(t)\in[-P_i^{\mathrm{dis}},\bar P_i^{\mathrm{nom}}]$ per session and step is then enough, with objective $-\sum_t\Delta t\,c^{\mathrm{ch}}(t)\sum_i p_i(t)$. Size at $N=200$: about $3\times10^4$ variables, well within HiGHS.

**Relaxations, both of which can only increase the optimum.**
1. **Taper relaxation.** The state-dependent constraint $p_i^{+}\le\bar P_i(E_i)$ is replaced by $p_i^{+}\le\bar P_i^{\mathrm{nom}}$. Since $\bar P_i(E)\le\bar P_i^{\mathrm{nom}}$ for all $E$, every physically admissible schedule satisfies (O4). The capacity part of $\bar P_i(E)$ is kept exactly through (O3).
2. **Complementarity relaxation.** Simultaneous $p^{+}>0$ and $p^{-}>0$ is allowed. Increasing both by $\epsilon$ changes the objective by $\epsilon\Delta t(c^{\mathrm{dis}}-c^{\mathrm{ch}})$ and the energy by $\epsilon\Delta t(\eta^{\mathrm c}-1/\eta^{\mathrm d})\le0$. So it is never strictly profitable when $c^{\mathrm{dis}}\le c^{\mathrm{ch}}$, and in our scenarios it is exact. In general it is a relaxation of the MILP with complementarity binaries.

A tighter optional variant keeps the LP structure with a linear taper majorant $p_i^{+}\le\bar\rho_i(E_i^{\mathrm{cap}}-E_i)$, where $\bar\rho_i=\sup_E\bar P_i(E)/(E_i^{\mathrm{cap}}-E)$. It is still an upper bound. It is not needed for DD10.

**Proposition O1 (upper bound).**
(a) Let $\pi$ be any controller, including non-anticipative ones, whose closed-loop trajectory in the model (F2)-(F3) satisfies (F8) at every step and has total shortfall $\sum_i(E_i^{\mathrm{req}}-E_i(d_i+1))^{+}\le z^{*}$. Then $J^{\pi}\le J^{\mathrm{LP}}$.
(b) In EV2Gym, let $J^{\mathrm{LP,ceil}}$ be the optimum of the same LP with $E_i^{\mathrm{req}}$ and $E_i^{\min}$ replaced by $E_i^{\mathrm{req}}-G_i^{\max}$ and $E_i^{\min}-G_i^{\max}$, where $G_i^{\max}=0.01\,(d_i-t_i^{\mathrm{arr}}+1)$ kWh. Then $J^{\pi}\le J^{\mathrm{LP,ceil}}$ for every controller whose realised (reported) powers satisfy (F8) and whose realised total shortfall is $\le z^{*}$.

*Proof.*
(a) Set $p^{\pm}=p_i(t)^{\pm}$, take $E_i(t)$ from the trajectory and $z_i=(E_i^{\mathrm{req}}-E_i(d_i+1))^{+}$. Then:
- (O1) and (O2) hold by (F2);
- (O3) holds by (F3)-(F4);
- (O4) holds because $p_i\le\bar P_i(E_i)\le\bar P_i^{\mathrm{nom}}$ and $p_i\ge p_i^{\min}\ge-P_i^{\mathrm{dis}}$;
- (O5) holds by the definition of $z_i$, and (O6) is (F8);
- $\sum_i z_i\le z^{*}$.

So the trajectory is feasible for stage 2, and its objective equals $J^{\pi}$.

(b) Let $\tilde p_i(t)$ be the reported powers and $E_i^{\mathrm{nom}}(t)=E_i^{\mathrm{arr}}+\Delta t\sum_{s<t}\tilde p_i(s)$. EV2Gym bills exactly $\tilde p$, so $J^{\pi}$ is the LP objective at $\tilde p$. The ceil adds less than 0.01 kWh per active step, so $E_i^{\mathrm{real}}(t)-G_i^{\max}\le E_i^{\mathrm{nom}}(t)\le E_i^{\mathrm{real}}(t)\le E_i^{\mathrm{cap}}$. Hence:
- (O3) holds with $E^{\min}-G^{\max}$;
- (O5) holds with $E^{\mathrm{req}}-G^{\max}$ and $z_i=(E_i^{\mathrm{req}}-E_i^{\mathrm{real}}(d_i+1))^{+}$;
- (O4) holds because $\tilde p\le\bar P_i(E^{\mathrm{real}})\le\bar P^{\mathrm{nom}}$.

The rest is as in (a). $\square$

**Optimality gap.** The main metric is $\mathrm{gap}(\pi)=(J^{\mathrm{LP}}-J^{\pi})/|J^{\mathrm{LP}}|$ per episode. Profits are usually negative, because charging to the target costs money, so the absolute value is needed. Episodes with $|J^{\mathrm{LP}}|<1$ € are excluded from the relative metric, and the absolute gap in € per EV is also reported. $J^{\mathrm{LP,ceil}}-J^{\mathrm{LP}}$ is reported once per scenario to show that the ceil effect is negligible. Because of the two relaxations, the reported gap overestimates the true gap.

The bound applies only to controllers that respect (F8). Raw heuristics and penalty-based RL can exceed $J^{\mathrm{LP}}$ by violating the transformer limit; this is flagged in the tables.

---

## 4 Markov decision process

### 4.1 Decision epochs and information

There is one decision per step $t$. The full system state contains:
- the set of connected EVs with their data;
- their energies $E_i(t)$;
- the time $t$ and calendar day;
- the exogenous process state (price series, load/PV series, generator state of the arrival process).

Future arrivals, prices beyond $H_f$ steps and realised future loads are not observed. Strictly, the problem is therefore a POMDP; we follow standard practice and condition the policy on the observation below. The observation is a deterministic function of the information listed in Sec. 2, plus the safety-layer quantities $l_i,u_i$, which are themselves functions of that information.

### 4.2 Observation: charger tokens and a global token

The observation is a set of $N$ charger tokens $\xi_i(t)\in\mathbb R^{F_c}$ plus one global token $\xi_0(t)\in\mathbb R^{F_g}$. No feature encodes the charger's index or position.

Normalisation constants:
- $E^{\mathrm{ref}}=100$ kWh;
- $P^{\mathrm{ref}}=11$ kW (the most common EV AC limit);
- $c^{\mathrm{ref}}=0.25$ €/kWh (EV2Gym 2022-23 mean price 0.18 €/kWh);
- $N_{\mathrm{day}}=24/\Delta t$.

**Charger token ($F_c=12$).** All features are 0 for an empty charger. Empty chargers are also masked as attention keys (Sec. 5).

| # | Feature | Normalisation |
|---|---|---|
| 1 | occupancy $o_i\in\{0,1\}$ | – |
| 2 | current energy $E_i(t)$ | $/E^{\mathrm{ref}}$ |
| 3 | capacity $E_i^{\mathrm{cap}}$ | $/E^{\mathrm{ref}}$ |
| 4 | SoC $E_i(t)/E_i^{\mathrm{cap}}$ | – |
| 5 | remaining required energy $(E_i^{\mathrm{req}}-E_i(t))^{+}$ | $/E^{\mathrm{ref}}$ |
| 6 | remaining steps $\tau_i(t)$; conformal variant: $\tau_i^{\mathrm{lo}}(t)$ (S 8) | $/N_{\mathrm{day}}$ |
| 7 | laxity $\chi_i(t)$ (S 3.3), clipped to $[-N_{\mathrm{day}},2N_{\mathrm{day}}]$ | $/N_{\mathrm{day}}$ |
| 8 | nominal max charging power $\bar P_i^{\mathrm{nom}}$ | $/P^{\mathrm{ref}}$ |
| 9 | max discharging power $P_i^{\mathrm{dis}}$ | $/P^{\mathrm{ref}}$ |
| 10 | physical lower limit $p_i^{\min}(E_i(t))$ (F4) | $/P^{\mathrm{ref}}$ |
| 11 | safety-layer lower bound $l_i(t)$ (S 3.2) | $/P^{\mathrm{ref}}$ |
| 12 | safety-layer upper bound $u_i(t)=\bar P_i(E_i(t))$ (taper and capacity included) | $/P^{\mathrm{ref}}$ |
| (13, 14) | $\eta_i^{\mathrm c},\eta_i^{\mathrm d}$, only if not 1 (not used in the EV2Gym instantiation) | – |

**Global token ($F_g=38$).**

| # | Feature | Normalisation |
|---|---|---|
| 1-2 | $c^{\mathrm{ch}}(t)$, $c^{\mathrm{dis}}(t)$ | $/c^{\mathrm{ref}}$ |
| 3-10 | $c^{\mathrm{ch}}(t+k)$, $k=1..H_f$, $H_f=8$ (day-ahead prices are published before delivery, so the next 2 h are known; EV2Gym's own state uses 20 future prices) | $/c^{\mathrm{ref}}$ |
| 11-12 | $\sin(2\pi\,\mathrm{hour}/24)$, $\cos(2\pi\,\mathrm{hour}/24)$ | – |
| 13 | weekend flag | – |
| 14 | remaining episode fraction $(T-t)/T$ | – |
| 15-16 | $C(t)$, $C_{\mathrm{rev}}(t)$ (realised, DD6) | $/P_{\mathrm{tr}}^{\mathrm{nom}}$ |
| 17-18 | $P^{\mathrm{inf}}(t)$, $P^{\mathrm{pv}}(t)$ (measured) | $/P_{\mathrm{tr}}^{\mathrm{nom}}$ |
| 19-26 | $\hat P^{\mathrm{inf}}(t+k)$, $k=1..H_f$ (true value + Gaussian noise, DD6) | $/P_{\mathrm{tr}}^{\mathrm{nom}}$ |
| 27-34 | $\hat P^{\mathrm{pv}}(t+k)$, $k=1..H_f$ | $/P_{\mathrm{tr}}^{\mathrm{nom}}$ |
| 35 | occupancy ratio $n(t)/N$ | – |
| 36 | rating per charger $P_{\mathrm{tr}}^{\mathrm{nom}}/(NP^{\mathrm{ref}})$ | – |
| 37-38 | aggregate tightness $\sum_{i\in\mathcal N(t)}l_i(t)$, $\sum_{i\in\mathcal N(t)}u_i(t)$ | $/P_{\mathrm{tr}}^{\mathrm{nom}}$ |

$P_{\mathrm{tr}}^{\mathrm{nom}}=\max_tP_{\mathrm{tr}}(t)$. The arrival profile (workplace, public, private) is deliberately **not** a feature; the policy infers it from the tokens and the time features (DD24, `algorithm.md` Alg. 7.1). All global features are intensive, i.e. ratios that do not grow with $N$. This keeps the inputs in distribution when a policy trained at $N\in\{10,20\}$ is applied at $N=200$ with $P_{\mathrm{tr}}$ scaled in proportion to $N$. When $c^{\mathrm{dis}}=f^{\mathrm{dis}}c^{\mathrm{ch}}$, no separate discharge-price forecast is needed.

### 4.3 Action

The policy outputs one **raw action** $a_i(t)\in\mathbb R$ per charger (Gaussian sample, Sec. 5). It is scaled to kW as $x_i(t)=P^{\mathrm{ref}}a_i(t)$.

The executed power is
$$p(t)=\Pi_t(x(t)),$$
the projection onto the feasible set $\mathcal K(t)$ (S 4), or the fallback rule when $\mathcal K(t)=\emptyset$ (S 7.2). Actions of empty chargers are ignored, with $l_i=u_i=0$.

$p_i(t)$ is mapped to EV2Gym's normalised action with upward rounding (S 6.3). No tanh squashing is used: the raw action is unbounded, and all bounding is done by $\Pi_t$.

### 4.4 Reward

The reward is the realised profit of the step in €:
$$r(t)=\Delta t\sum_{i\in\mathcal N(t)}\big(c^{\mathrm{dis}}(t)\tilde p_i^{-}(t)-c^{\mathrm{ch}}(t)\tilde p_i^{+}(t)\big),\tag{F9}$$
where $\tilde p_i$ is the reported power (S 6.3). This equals EV2Gym's per-step `total_costs` (reward function `V2G_costs_simple`).

The reward has **no penalty for the departure energy requirement or the transformer limit**. Both are enforced by the safety layer, so a penalty would be identically zero on every step where A5 holds. On infeasible steps (S 7) the controller has no influence, because the fallback rule is deterministic, so a penalty would add variance without a learnable signal.

Optional shaping (off in the main experiments): a battery-throughput proxy $-c^{\mathrm{thr}}\Delta t\sum_i|\tilde p_i|$ with $c^{\mathrm{thr}}=0.01$ €/kWh, for a sensitivity run only.

**Learning reward.** To make returns intensive, so that one critic can serve episodes of different $N$, PPO is trained on
$$\tilde r(t)=r(t)/(N\,r^{\mathrm{ref}}),\qquad r^{\mathrm{ref}}=c^{\mathrm{ref}}P^{\mathrm{ref}}\Delta t.$$
This is design decision DD15. All reported results use $r(t)$ in €. The same normalisation is applied to $r^{\mathrm{pen}}$ for the penalty-based baselines.

**Why $\beta\|x-\Pi(x)\|^2$ is a loss term and not a reward term.**
1. It is a function of the raw action alone and has no effect on the physical transition. In the reward, it would be credited through noisy advantage estimates. As an actor-loss term evaluated at the policy mean, its gradient is exact and cheap: by the Moreau identity, $\nabla_x\tfrac12\operatorname{dist}^2(x,\mathcal K)=x-\Pi(x)$ (S 9).
2. Keeping $r$ equal to the physical profit makes returns of all methods directly comparable and keeps the "cost of the guarantee" well defined.
3. Distance terms between pre- and post-projection actions are an established remedy against drift of raw actions [chen2021enforcing; markgraf2025safe].

### 4.5 Transition, horizon, discount

**Transition.**
- The exogenous process (arrivals with their attributes, prices, loads, PV) evolves independently of the actions.
- Energies evolve by EV2Gym's battery model under the commanded $p(t)$; in the exact model (A3, S 6) this is (F5).
- An EV arriving at $t+1$ adds a token. Its $E^{\mathrm{req}}$ is fixed by (F6) and admission control.
- An EV with $d_i=t$ leaves after step $t$.

**Horizon and discount.**
- The horizon is $T$ steps: 112 at 15 min, 336 at 5 min.
- With `empty_ports_at_end_of_simulation=True`, every EV departs at $d_i\le T-2$ (ev2gym_notes §11.7). Step $T$ is therefore a true terminal state with value 0, not a truncation.
- The discount is $\gamma=0.99$, an effective horizon of about 100 steps (25 h at 15 min). The sensitivity run at 5 min uses $\gamma=0.99^{1/3}\approx0.9967$, which keeps the same horizon in hours.

---

## 5 Attention policy and critic

**Encoder (one per network; the actor and critic do not share weights).** A transformer encoder [vaswani2017attention] in the pre-LayerNorm arrangement [ba2016layer; xiong2020layer], without positional encoding, so that it acts on the set of charger tokens.
- **Embeddings:**
  - $e_i=W_c\xi_i+b_c+v_c$ for chargers;
  - $e_0=W_g\xi_0+b_g+v_g$ for the global token.
  - $v_c,v_g\in\mathbb R^{d_m}$ are two learned type embeddings. They distinguish token types, not positions.
- **Sequence:** $H^{(0)}=[e_0;e_1;\dots;e_N]\in\mathbb R^{(N+1)\times d_m}$.
- **Key-padding mask:** charger key $i\ge1$ is masked iff $o_i=0$. The global key is never masked, so every query has at least one valid key and the softmax is well defined.
- **$n_L$ pre-LayerNorm layers:**
$$H\leftarrow H+\mathrm{MHA}(\mathrm{LN}(H);\text{mask}),\qquad H\leftarrow H+\mathrm{FFN}(\mathrm{LN}(H)),$$
  with $\mathrm{FFN}(h)=W_2\,\mathrm{GELU}(W_1h+b_1)+b_2$, hidden width $4d_m$. A final LN follows.
- **Defaults:** $d_m=128$, $n_L=3$, $n_H=4$, dropout 0. Masked logits are set to $-\infty$, or to $-10^9$ in float32.
- **No positional encoding.**

**Actor head (shared across chargers).**
- $(\mu_{\theta,i},\ell_i)=\mathrm{MLP}_a(h_i)$, with sizes $d_m\to64\to2$ and GELU.
- $\log\sigma_{\theta,i}=\operatorname{clamp}(\ell_i,-5,0.5)$.
- Policy:
$$\pi_\theta(a\mid s)=\prod_{i\in\mathcal N(t)}\mathcal N\big(a_i;\mu_{\theta,i},\sigma_{\theta,i}^2\big).$$
  Empty chargers do not enter the product (`algorithm.md`, Alg. 2).
- **Initialisation:** last-layer weights $\times0.01$, mean bias 0, log-std bias $-0.5$ (initial std about 6.7 kW).

**Critic.**
- Attention pooling (pooling by multi-head attention [lee2019set]) with one learned query $q\in\mathbb R^{d_m}$ ($n_H$ heads, same key mask) over the encoder outputs of all unmasked tokens, global token included.
- Then $V_\omega(s)=\mathrm{MLP}_v(\mathrm{pool})$, with sizes $d_m\to64\to1$.
- The returns are intensive (Sec. 4.4), so an averaging readout is adequate.
- A global-token readout ($V=\mathrm{MLP}_v(h_0)$) is an acceptable alternative.

**Lemma N1 (equivariance, invariance, size independence).** Let $\varpi$ be a permutation of $\mathcal N$ acting on the charger tokens and masks, with the global token fixed. Then:
(i) $\mu_\theta,\sigma_\theta$ are permutation-equivariant: $\mu_{\theta,\varpi(i)}(\varpi s)=\mu_{\theta,i}(s)$, hence $\pi_\theta(\varpi a\mid\varpi s)=\pi_\theta(a\mid s)$;
(ii) $V_\omega(\varpi s)=V_\omega(s)$;
(iii) the safety layer is equivariant: $\Pi(\varpi x;\varpi l,\varpi u,\tilde C,C_{\mathrm{rev}})=\varpi\,\Pi(x;l,u,\tilde C,C_{\mathrm{rev}})$, and so is the fallback rule, except for the final charger-index tie-break of its ordering key;
(iv) outputs of unmasked tokens and the value do not depend on the features or the number of masked tokens (padding invariance);
(v) the number of parameters does not depend on $N$.

*Proof.* Let $P$ be the $(N+1)\times(N+1)$ permutation matrix that fixes index 0 and permutes $1..N$ by $\varpi$.
1. Embeddings, LN, FFN, residual additions and the actor head act row-wise, so they commute with $P$: $f(PH)=Pf(H)$.
2. For one attention head, $A(H)=\mathrm{softmax}\big(QK^\top/\sqrt{d_k}+M\big)V$, with $Q=HW_Q$, $K=HW_K$, $V=HW_V$ and $M_{jk}=-\infty$ if key $k$ is masked (0 otherwise). $M$ depends only on the column (key) index, so permuting the tokens and their mask gives $M\mapsto PMP^\top$. Row-wise softmax satisfies $\mathrm{softmax}(PXP^\top)=P\,\mathrm{softmax}(X)P^\top$. Therefore
$$A(PH)=P\,\mathrm{softmax}(QK^\top/\sqrt{d_k}+M)P^\top PV=P\,A(H).$$
  Concatenation of heads and the output projection are row-wise.
3. By composition, the encoder is equivariant, and so are $\mu_\theta,\sigma_\theta$ (i). The Gaussian product is invariant under joint permutation of $a$ and $s$.
4. For (ii), pooling computes $\sum_k\mathrm{softmax}_k(q^\top K_k/\sqrt{d_k}+M_k)V_k$. This is a sum over keys and is invariant under any permutation of them.
5. For (iii), the problem $\min\|p-x\|^2$ s.t. $l\le p\le u$, $-C_{\mathrm{rev}}\le\mathbf 1^\top p\le\tilde C$ is invariant under jointly permuting $p,x,l,u$. Its minimiser is unique (Proposition 1), so it is equivariant. The fallback depends on chargers only through $(l_i,u_i,p_i^{\min},\chi_i)$ and a sort, which is equivariant up to ties.
6. For (iv), a masked key receives weight $e^{-\infty}=0$ in every softmax, so its value never enters the output of any other token. Its own row is computed but discarded. Row-wise operations do not mix tokens.
7. For (v), every weight matrix has a shape determined by $(F_c,F_g,d_m,n_H,n_L)$, never by $N$. $\square$

**Parameter counts** ($d_m=128$, $n_L=3$, FFN 512, $F_c=12$, $F_g=38$):

| Network | Parameters |
|---|---|
| Actor | 610,370 |
| Critic | 676,481 |
| Total | 1,286,851, independent of $N$ |
| Small variant ($d_m=64$, $n_L=2$), total | 228,067 |
| MLP baseline actor ($N_{\max}=200$, two hidden layers of 256) | 792,976; grows linearly with $N_{\max}$ |

**Consequence for zero-shot transfer.** By (iv), a station with $N$ chargers, $n$ of them occupied, gives the same per-EV outputs as a station with only those $n$ chargers, except through the global features 35-38. These features are ratios. Together with (i) and (iii), this is the structural basis for applying weights trained at $N\in\{10,20\}$ to $N\in[5,200]$. It is not by itself a performance guarantee: attention over 200 tokens averages over more keys than seen in training. This is exactly what C3 measures.
