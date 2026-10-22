# Safety layer: bounds, projection and feasibility invariance

Agent A2 (Method and Theory). Version 5, 2026-10-22, revised after the internal proof check, the implementation feedback (DD25-DD29) and internal review round 1 (DD30-DD41) (`review/proof_check.md`; change log in `DONE_method.md`). Symbols: `nomenclature.md`. Model equations (F1)-(F9): `formulation.md`.
Executable specification: `projection_ref.py`. Numerical checks: `check_A1.py` (results in `check_A1_results.txt`) and `verify_projection.py` (results in `verify_projection_results.txt`).

The **safety layer, a closed-form projection onto the feasible set**, maps the raw per-charger action $x(t)=P^{\mathrm{ref}}a(t)$ [kW] to the commanded power
$$p(t)=\Pi_t(x(t)),\qquad p_i=\operatorname{clip}(x_i-\lambda,\,l_i,\,u_i).$$
The per-EV bounds $l_i,u_i$ encode the physical limits and the departure energy requirement. The common shift $\lambda$ enforces the transformer limit. The layer is used in both training and deployment.

**Overview of results.**

| Result | Statement |
|---|---|
| **Theorem 1′** | **Main result (DD30): transformer limit for every raw-action sequence; departure requirement per EV; Theorem 1 as special case** |
| Lemma 0 | Reach operator: $\varphi_i^k$ is the maximum energy after $k$ steps |
| Lemmas E1-E3 | EV2Gym satisfies A1, A1′ and A3 under the stated configuration |
| Proposition 1 | Projection: feasibility condition, closed form, uniqueness; (iii) Jacobian (supplementary, DD40) |
| Lemma 1 | One-step individual invariance |
| Theorem 1 | Feasibility invariance; Corollaries 1 (localisation) and 2 (EV2Gym) |
| Lemma 2 | Fallback rule |
| Lemma 3 | Shortfall accounting |
| Proposition 2 | Conformal departure bound |

Equations are numbered (S1)-(S16) in order of appearance.

---

## 1 Scope

- Each step $t$ has occupied chargers $\mathcal N(t)$.
- For $i\in\mathcal N(t)$ we write $E=E_i(t)$ and $\tau=\tau_i(t)=d_i-t+1\ge1$, the remaining actions including the current one (F1).
- Empty chargers get $l_i=u_i=0$ and are ignored below. EV2Gym also forces their action to 0.
- All statements hold for every EV separately, with its own parameters. The index $i$ is dropped when no confusion arises.
- The raw action is any **finite** vector $x\in\mathbb R^N$. Non-finite entries (NaN or Inf from a diverging network) are replaced by 0 before the layer, and their number is logged (`sanitize()`).
- Numerical tolerance: all inequalities in this document are exact in real arithmetic. The code checks them with one tolerance, $\epsilon_{\mathrm{num}}=10^{-9}$ (kW for powers, kWh for energies; Sec. 6.4).

---

## 2 Per-EV model and reachable energy

### 2.1 Energy map and per-step physical bounds

$$g(p)=\eta^{\mathrm c}p^{+}-p^{-}/\eta^{\mathrm d}\quad(\text{continuous, strictly increasing, }g(0)=0;\ g\equiv\mathrm{id}\text{ in EV2Gym}),\tag{S1}$$
$$\varphi(E)=E+\Delta t\,\eta^{\mathrm c}\,\bar P(E)\qquad\text{(energy after one step of maximum charging)},\tag{S2}$$
$$p^{\min}(E)=-\min\Big\{P^{\mathrm{dis}},\ \eta^{\mathrm d}(E-E^{\min})^{+}/\Delta t\Big\},\tag{S3}$$
$$u(E)=p^{\max}(E)=\bar P(E),\qquad \bar P(E)=\min\Big\{\bar P^{\mathrm{tap}}(E),\ \frac{E^{\mathrm{cap}}-E}{\eta^{\mathrm c}\Delta t}\Big\}.\tag{S4}$$

- $\bar P^{\mathrm{tap}}(E)\le\bar P^{\mathrm{nom}}$ is the power limit including the CC-CV taper.
- By (S3)-(S4), every $p\in[p^{\min}(E),\bar P(E)]$ keeps $E+\Delta t\,g(p)\in[E^{\min},E^{\mathrm{cap}}]$ when $E\in[E^{\min},E^{\mathrm{cap}}]$.
- Since $p^{\min}\le0\le\bar P$, the interval is never empty.
- $\varphi(E)=\min\{E+\Delta t\eta^{\mathrm c}\bar P^{\mathrm{tap}}(E),E^{\mathrm{cap}}\}$.

**Assumption A1 (monotone maximum-charging map).** $\varphi$ is continuous and non-decreasing on $[E^{\min},E^{\mathrm{cap}}]$.

**Assumption A1′ (non-expansive maximum-charging map).** $\varphi$ is 1-Lipschitz: $\varphi(E')-\varphi(E)\le E'-E$ for $E<E'$. By (S2) this is equivalent to $\bar P$ being non-increasing in $E$. A1′ is used only in Lemma 3 and to give numerical room in the DD2 cap (Sec. 3.4). Theorem 1 does not need it.

**A sufficient condition for A1.** Suppose $\bar P^{\mathrm{tap}}$ decreases at rate at most $L$ [kW/kWh], i.e. $\bar P^{\mathrm{tap}}(E)-\bar P^{\mathrm{tap}}(E')\le L(E'-E)$ for $E<E'$, with
$$\eta^{\mathrm c}\Delta t\,L\le1.$$
Then $\varphi^{\mathrm{tap}}(E')-\varphi^{\mathrm{tap}}(E)\ge(E'-E)(1-\eta^{\mathrm c}\Delta tL)\ge0$, and the minimum with the constant $E^{\mathrm{cap}}$ preserves monotonicity. So A1 can fail only for a taper whose power drops faster than $1/(\eta^{\mathrm c}\Delta t)$ per kWh. In that case, charging more now could leave less reachable energy one step later.

### 2.2 Reachable-energy operator

For $k\ge0$ define $\Psi_k=\varphi^{k}$ ($k$-fold composition, $\Psi_0=\mathrm{id}$) and the **reachable-energy operator** $R(k,E)$:
$$R(0,\cdot)=0,\qquad R(k,E)=\Delta t\,\eta^{\mathrm c}\bar P(E)+R\big(k-1,\varphi(E)\big),\tag{S5}$$
$$\Psi_k(E)=E+R(k,E).\tag{S6}$$

**Lemma 0 (reach).** Under A1:
(i) $\Psi_k$ is continuous and non-decreasing;
(ii) for any admissible sequence $p(t')\in[p^{\min}(E(t')),\bar P(E(t'))]$, $t'=t,\dots,t+k-1$, with exact dynamics (F2): $E(t+k)\le\Psi_k(E(t))$, with equality for full-power charging.

So $R(k,E)$ is the maximum energy that can be added in $k$ steps from $E$.

*Proof.* (i) A composition of continuous non-decreasing maps is continuous and non-decreasing.
(ii) Induction on $k$. The case $k=0$ is trivial. For $k\ge1$, $E(t+1)=E+\Delta t g(p(t))\le E+\Delta t g(\bar P(E))=\varphi(E)$ because $g$ is increasing. By the induction hypothesis from $E(t+1)$ and by (i):
$$E(t+k)\le\Psi_{k-1}(E(t+1))\le\Psi_{k-1}(\varphi(E))=\Psi_k(E).$$
Full-power charging gives equality at every step. $\square$

Remark on monotonicity: $R(k,\cdot)$ itself is not monotone, and under A1′ it is non-increasing. The monotone object is $\Psi_k=E+R(k,E)$. The passage from "$\varphi$ monotone" to "$\Psi_k$ monotone" is exactly step (i). Without A1, part (ii) fails: charging less now could leave more reachable energy later.

### 2.3 EV2Gym's taper: closed form, A1 and exact delivery

**Condition C0.** $P_i^{\mathrm{ev}}\le P^{\mathrm{cs}}$. This holds for all 8 EV models (11 or 7.4 kW versus 22.17 kW). Under C0, a full-current command is capped exactly at the EV limit and does not shift the taper. For $P^{\mathrm{ev}}>P^{\mathrm{cs}}$, EV2Gym shifts the full-current transition, and the closed form below is wrong (by 0.72 kWh for a hypothetical 40 kW EV; `review/proof_check_numeric_results.txt`, Part C).

In SoC units $\varsigma=E/E^{\mathrm{cap}}$, let:
- $\delta=\bar P^{\mathrm{nom}}\Delta t/E^{\mathrm{cap}}$, the SoC gain of one full-power step, with $\bar P^{\mathrm{nom}}=\zeta^{\mathrm c}P^{\mathrm{ev}}$ under C0;
- $\varsigma^{\mathrm{tr}}$ = `transition_soc`;
- $m$ = `transition_soc_multiplier`;
- $\kappa=m\delta/(1-\varsigma^{\mathrm{tr}})$.

Reading `EV._charge` (`ev.py:240-355`) at full current gives
$$
\frac{\varphi(E)}{E^{\mathrm{cap}}}=\min\{\hat\varphi(\varsigma),\,\varsigma+\delta\},\quad
\hat\varphi(\varsigma)=\begin{cases}
\varsigma+\delta, & \varsigma+\delta\le\varsigma^{\mathrm{tr}},\\
1-(1-\varsigma^{\mathrm{tr}})\exp\!\big(-m\frac{\varsigma+\delta-\varsigma^{\mathrm{tr}}}{1-\varsigma^{\mathrm{tr}}}\big), & \varsigma<\varsigma^{\mathrm{tr}}<\varsigma+\delta,\\
1-(1-\varsigma)\,e^{-\kappa}, & \varsigma\ge\varsigma^{\mathrm{tr}},
\end{cases}\tag{S7}
$$
with $\varphi(E)=E^{\mathrm{cap}}\min\{1,\varsigma+\delta\}$ when $\varsigma^{\mathrm{tr}}=1$ (no taper).

(S7) matches `EV._charge` to $1.4\times10^{-14}$ kWh on 768 configurations: 8 EV models × 4 values of $\zeta^{\mathrm c}$ × 6 values of $\varsigma^{\mathrm{tr}}$ × $m\in\{1,5\}$ × $\Delta t\in\{5,15\}$ min (`check_A1.py`). The reviewer's independent check over 300 random configurations ($m\in[0.5,10]$, $\Delta t\le60$ min) gives the same accuracy.

**Lemma E1 (A1 and A1′ hold for EV2Gym, for every $\Delta t$).** Under C0, for all $\delta>0$, $\varsigma^{\mathrm{tr}}\in(0,1]$ and $m>0$, the map $\varphi$ in (S7) is continuous, non-decreasing and 1-Lipschitz.

*Proof.*
1. **Continuity.** At $\varsigma=\varsigma^{\mathrm{tr}}-\delta$, the first two branches of $\hat\varphi$ both equal $\varsigma^{\mathrm{tr}}$. At $\varsigma=\varsigma^{\mathrm{tr}}$, the last two both equal $1-(1-\varsigma^{\mathrm{tr}})e^{-\kappa}$. A minimum of continuous functions is continuous.
2. **Slopes.** The branch slopes are 1, $m\exp(\cdot)>0$ and $e^{-\kappa}\in(0,1)$, and $\varsigma+\delta$ has slope 1. A minimum of non-decreasing functions is non-decreasing (A1).
3. **1-Lipschitz on the middle interval.** Only the middle branch can have slope above 1 (when $m>1$). On $I=(\varsigma^{\mathrm{tr}}-\delta,\varsigma^{\mathrm{tr}})$, $h(\varsigma)=\hat\varphi(\varsigma)-\varsigma-\delta$ is strictly concave ($h''<0$) with $h(\varsigma^{\mathrm{tr}}-\delta)=0$. If $h(\varsigma_1)<0$ at some $\varsigma_1\in I$, concavity gives $h'(\varsigma_1)\le h(\varsigma_1)/(\varsigma_1-\varsigma^{\mathrm{tr}}+\delta)<0$, i.e. $\hat\varphi'(\varsigma_1)<1$. So wherever the minimum in (S7) selects $\hat\varphi$ strictly, its slope is below 1; elsewhere it selects $\varsigma+\delta$, with slope 1.
4. **Conclusion.** $\varphi$ is piecewise $C^1$ with slopes in $[0,1]$, hence 1-Lipschitz (A1′). $\square$

In terms of the sufficient condition, $\eta^{\mathrm c}\Delta t\,L$ (with $\eta^{\mathrm c}=1$ in EV2Gym, so $\Delta t\cdot L$) equals:
- $1-e^{-\kappa}<1$ in the upper taper region;
- $1-\hat\varphi'<1$ in the middle region;
- exactly 1 at the full-battery kink of untapered EVs, where $\varphi$ is flat but still monotone.

So EV2Gym's exponential taper satisfies the condition for **every** step length. Numerically, the maximum of $\Delta t L$ over the taper configurations is 0.549 at 5 min and 0.909 at 15 min ($m=5$), and 0.147 and 0.380 for $m=1$. A1 holds on all 768 configurations, and A1′ holds with excess $1.4\times10^{-14}$ kWh (`check_A1_results.txt`).

**Lemma E2 (exact delivery of sub-maximal charging commands in EV2Gym).** Assume C0. Let $m>1$ and let $z_0(m)\in(0,1)$ be the positive root of $1-e^{-mz}=z$. If
$$\kappa\le\kappa^{*}(m):=m\,z_0(m),$$
then every command $p\in(0,\bar P(E)]$, sent as pilot $p$ (i.e. normalised action $p/(\zeta^{\mathrm c}P^{\mathrm{cs}})$), is delivered exactly: $E^{+}=E+\Delta t\,p$ before the 0.01 kWh rounding. Discharge commands $p\in[p^{\min}(E),0)$ are always delivered exactly.

*Proof.* Let $\hat\delta=p\Delta t/E^{\mathrm{cap}}$ be the commanded SoC gain, with $\hat\delta\le\hat\delta^{*}:=(\varphi(E)-E)/E^{\mathrm{cap}}\le\delta$. For a reduced pilot, EV2Gym shifts the transition to $\varsigma_x=1-(\hat\delta/\delta)(1-\varsigma^{\mathrm{tr}})$; write $\Omega:=1-\varsigma_x=\hat\delta(1-\varsigma^{\mathrm{tr}})/\delta$. The delivered SoC gain is $\min\{\hat\delta,\,\hat\varphi_x(\varsigma)-\varsigma\}$, where $\hat\varphi_x$ is (S7) with $(\varsigma^{\mathrm{tr}},\delta)$ replaced by $(\varsigma_x,\hat\delta)$. There are three cases.

1. **Case $\varsigma+\hat\delta\le\varsigma_x$:** the gain is $\hat\delta$.
2. **Case $\varsigma\ge\varsigma_x$:** $\hat\varphi_x(\varsigma)-\varsigma=(1-\varsigma)(1-e^{-m\hat\delta/\Omega})=(1-\varsigma)(1-e^{-\kappa})$, independent of $\hat\delta$. Since $\varsigma\ge\varsigma_x\ge\varsigma^{\mathrm{tr}}$, we have $\hat\delta\le\hat\delta^{*}\le(1-\varsigma)(1-e^{-\kappa})$, so the gain is $\hat\delta$.
3. **Case $\varsigma<\varsigma_x<\varsigma+\hat\delta$:** let $y=\varsigma+\hat\delta-\varsigma_x>0$ and $z=y/\Omega$. Then $\hat\varphi_x(\varsigma)-\varsigma-\hat\delta=\Omega(1-e^{-mz}-z)$. The function $z\mapsto1-e^{-mz}-z$ is concave, zero at 0 and at $z_0$, so it is $\ge0$ iff $z\le z_0$. Now $z=1-(1-\varsigma-\hat\delta)/\Omega$, so $z\le z_0$ iff $1-\varsigma-\hat\delta\ge(1-z_0)\,\hat\delta(1-\varsigma^{\mathrm{tr}})/\delta$.
   - (a) If $\varsigma<\varsigma^{\mathrm{tr}}$ (then $\varsigma+\delta>\varsigma^{\mathrm{tr}}$, otherwise case 3 is impossible): $1-\varsigma-\hat\delta\ge1-\varphi(E)/E^{\mathrm{cap}}\ge(1-\varsigma^{\mathrm{tr}})e^{-m(\varsigma+\delta-\varsigma^{\mathrm{tr}})/(1-\varsigma^{\mathrm{tr}})}\ge(1-\varsigma^{\mathrm{tr}})e^{-\kappa}$. With $\hat\delta\le\delta$, it suffices that $e^{-\kappa}\ge1-z_0=e^{-mz_0}$, i.e. $\kappa\le mz_0$.
   - (b) If $\varsigma\ge\varsigma^{\mathrm{tr}}$: $\hat\delta\le(1-\varsigma)(1-e^{-\kappa})$ gives $1-\varsigma-\hat\delta\ge(1-\varsigma)e^{-\kappa}$, and the right-hand side of the condition is at most $(1-z_0)(1-\varsigma)(1-e^{-\kappa})m/\kappa$. It suffices that $\kappa/(e^{\kappa}-1)\ge m(1-z_0)$. At $\kappa=mz_0$, $mz_0/(e^{mz_0}-1)=mz_0(1-z_0)/z_0=m(1-z_0)$. The left side is decreasing in $\kappa$, so the condition is again $\kappa\le mz_0$.

Discharging has no taper. `_discharge` delivers $\zeta^{\mathrm d}\max(P_{\mathrm{cmd}},-P^{\mathrm{ev,dis}})\Delta t$, floored at $E^{\min}$, and (S3) keeps $p$ inside both limits. $\square$

Numbers:
- $z_0(5)=0.99302$ and $\kappa^{*}(5)=4.965$.
- In our configuration ($m=5$, $\varsigma^{\mathrm{tr}}\le0.9$, all 8 models, $\zeta^{\mathrm c}\le1$) $\kappa\le2.391$ at 15 min and $\le0.797$ at 5 min. The numerical check confirms exact delivery to $1.4\times10^{-14}$ kWh.
- The condition is sharp in the right direction. The reviewer found exact delivery at $\kappa\le4.78$ and under-delivery of 0.026 kWh at $\kappa=9.57$ ($\Delta t=60$ min).
- **For $m\le1$ Lemma E2 fails.** EV2Gym then under-delivers sub-maximal commands near the transition by up to 0.106 kWh per 15-min step. EV2Gym's taper multiplier is kept at $m=5$ (DD17).

---

## 3 Per-EV bounds

### 3.1 Backward-reachable thresholds

$$B(0)=E^{\mathrm{req}},\qquad B(k)=\min\{E\in[E^{\min},E^{\mathrm{cap}}]:\ \varphi(E)\ge B(k-1)\},\quad k\ge1.\tag{S8}$$

**Well-definedness.**
- $E^{\mathrm{req}}\le E^{\mathrm{des}}\le E^{\mathrm{cap}}$, and $\varphi(E^{\mathrm{cap}})=E^{\mathrm{cap}}\ge B(k-1)$, so the set is non-empty. By continuity (A1) it is closed, so the minimum exists.
- $\varphi(E)\ge E$ gives $B(k)\le B(k-1)$.
- If $\varphi(E^{\min})\ge B(k-1)$, then $B(k)=E^{\min}$.

**Lemma (threshold characterisation).** Under A1, $\{E\in[E^{\min},E^{\mathrm{cap}}]:\Psi_k(E)\ge E^{\mathrm{req}}\}=[B(k),E^{\mathrm{cap}}]$ for every $k\ge0$. Hence EV $i$ is **individually feasible** at $t$, meaning there exists an admissible control that meets (F7) by Lemma 0, if and only if
$$E_i(t)\ \ge\ B_i(\tau_i(t))\quad\Longleftrightarrow\quad \Psi_{i,\tau_i(t)}(E_i(t))\ \ge\ E_i^{\mathrm{req}}.\tag{S9}$$

*Proof.* Induction on $k$. The case $k=0$ is the definition. For $k\ge1$, $\Psi_k(E)=\Psi_{k-1}(\varphi(E))\ge E^{\mathrm{req}}$ iff $\varphi(E)\ge B(k-1)$ by the induction hypothesis. Since $\varphi$ is continuous and non-decreasing, $\{E:\varphi(E)\ge B(k-1)\}$ is an up-set, closed, with minimum $B(k)$. $\square$

**Certified tables (robust to numerical error).** The proofs in Sec. 5 and 8 use only the following properties of the table $\hat B$ that is actually stored:
$$\hat B(0)\ge E^{\mathrm{req}},\qquad \varphi(\hat B(k))\ge\hat B(k-1),\qquad \hat B(k)\le\max\{\hat B(k-1),E^{\min}\},\qquad \hat B(k)\in[E^{\min},E^{\mathrm{cap}}]\quad(k\ge1).\tag{S10}$$
- Any table satisfying the first two conditions has $\hat B\ge B$ componentwise, by induction: $\varphi(\hat B(k))\ge\hat B(k-1)\ge B(k-1)$, so $\hat B(k)\ge B(k)$. The exact table is the least conservative one.
- The third condition (monotonicity) is used only in Proposition 2(ii).
- `thresholds()` in `projection_ref.py` computes each level by bisection on $[E^{\min},\hat B(k-1)]$ and returns the upper (feasible) end, so all four conditions hold by construction. The tolerance is $10^{-10}$ kWh.
- **Closed form (no taper,** $\bar P^{\mathrm{tap}}\equiv\bar P^{\mathrm{nom}}$**):** $\varphi(E)=\min\{E+\eta^{\mathrm c}\bar P^{\mathrm{nom}}\Delta t,E^{\mathrm{cap}}\}$ and $B(k)=\max\{E^{\min},\,E^{\mathrm{req}}-k\,\eta^{\mathrm c}\bar P^{\mathrm{nom}}\Delta t\}$ (exact and certified).

**Certification does not replace A1.** The reviewer's closed-loop test with a taper that drops faster than $1/\Delta t$ ("STEEP", A1 violated) still produces certified tables. Yet Lemma 1(a) fails ($l>u$ for 92-96 EV-steps), and EVs whose whole stay satisfied A5 missed $E^{\mathrm{req}}$ by up to 1.78 kWh. A1 is therefore a necessary ingredient, not a technical convenience.

### 3.2 Lower and upper bound

For a connected EV at step $t$ with $E=E_i(t)$ and $\tau=\tau_i(t)$:
$$
u_i(t)=\bar P(E),\qquad
l_i(t)=\begin{cases}
\max\Big\{p^{\min}(E),\ g^{-1}\Big(\dfrac{\hat B(\tau-1)-E}{\Delta t}\Big)\Big\}, & E\ge\hat B(\tau)\ \ \text{(individually feasible)},\\[2mm]
\bar P(E), & E<\hat B(\tau)\ \ \text{(individually infeasible: charge at maximum power)}.
\end{cases}\tag{S11}
$$

**Equivalence with the defining property.** With the exact table, $l_i(t)$ is the smallest $p\in[p^{\min}(E),p^{\max}(E)]$ with
$$E+\Delta t\,g(p)+R\big(\tau-1,\ E+\Delta t\,g(p)\big)\ \ge\ E^{\mathrm{req}},$$
and equals $p^{\max}(E)$ if no such $p$ exists.

*Proof.*
1. By the threshold characterisation, the condition reads $E+\Delta t g(p)\ge B(\tau-1)$. Since $g$ is strictly increasing, this is $p\ge g^{-1}((B(\tau-1)-E)/\Delta t)$.
2. Intersected with $[p^{\min},p^{\max}]$, the minimum is the stated maximum, provided it does not exceed $p^{\max}=\bar P(E)$.
3. That proviso is equivalent to $\varphi(E)\ge B(\tau-1)$, i.e. to $E\ge B(\tau)$. $\square$

**Closed form for constant power** ($\bar P^{\mathrm{tap}}\equiv\bar P^{\mathrm{nom}}$, general $\eta$). Substituting the closed-form table, the $E^{\min}$ branch is dominated by $p^{\min}$, because $g^{-1}((E^{\min}-E)/\Delta t)=-\eta^{\mathrm d}(E-E^{\min})/\Delta t\le p^{\min}(E)$. With $y=E^{\mathrm{req}}-E-(\tau-1)\eta^{\mathrm c}\bar P^{\mathrm{nom}}\Delta t$ [kWh]:
$$
l_i=\max\Big\{p^{\min}(E),\ \frac{y}{\eta^{\mathrm c}\Delta t}\ \text{if }y\ge0,\ \ \frac{\eta^{\mathrm d}y}{\Delta t}\ \text{if }y<0\Big\},\qquad\text{feasible iff } E\ge E^{\mathrm{req}}-\tau\,\eta^{\mathrm c}\bar P^{\mathrm{nom}}\Delta t.\tag{S12}
$$
In the EV2Gym instantiation ($g=\mathrm{id}$) this is DD3's formula with the correct physical lower limit:
$$l_i=\max\Big\{-\min\big(P^{\mathrm{dis}},\tfrac{E-E^{\min}}{\Delta t}\big),\ \frac{E^{\mathrm{req}}-E}{\Delta t}-(\tau-1)\bar P^{\mathrm{nom}}\Big\}.$$
- The capacity term needs no separate treatment, because $E^{\mathrm{req}}\le E^{\mathrm{cap}}$.
- A negative $y$ means the EV may V2G-discharge down to the energy from which $\tau-1$ full-power steps still reach $E^{\mathrm{req}}$.

**Computation with taper.**
1. **Threshold table (default, DD18).** At arrival compute $\hat B(k)$, $k=0..\tau^{\mathrm{arr}}$, by bisection on $\varphi$. Then (S11) costs $O(1)$ per EV per step. Correctness of the bisection: $\{E:\varphi(E)\ge B(k-1)\}$ is an up-set (A1).
2. **Direct bisection on $p$** (cross-check only). The predicate $p\mapsto[\Psi_{\tau-1}(E+\Delta tg(p))\ge E^{\mathrm{req}}]$ is monotone (A1 and $g$ increasing). Bisection on $[p^{\min},p^{\max}]$ converges to $l$, and returning the upper end is conservative. The cost is $O(\tau\log(1/\epsilon))$.

In code, the feasibility test in (S11) is $E\ge\hat B(\tau)-\epsilon_{\mathrm{num}}$ (Sec. 6.4), and $l_i:=\min\{l_i,u_i\}$ is enforced. Under A1 this guard is inactive beyond rounding (Lemma 1(a)). Without A1 it keeps $l\le u$, so the projection stays well defined. This is why parts (a) and (b) of Theorem 1′ do not need A1.

### 3.3 Laxity

$$\chi_i(t)=\frac{E_i(t)-\hat B_i(\tau_i(t))}{\eta^{\mathrm c}\bar P_i^{\mathrm{nom}}\Delta t}\quad[\text{steps}].$$
- $\chi_i\ge0$ iff the EV is individually feasible.
- Without taper and with $\hat B>E^{\min}$, $\chi_i=\tau_i-(E^{\mathrm{req}}-E_i)/(\eta^{\mathrm c}\bar P^{\mathrm{nom}}\Delta t)$, the classical laxity.
- It is used as token feature 7 and as the priority order of the fallback rule. In EV2Gym $\eta^{\mathrm c}=1$.

### 3.4 Arrival cap (DD2) and A4

**Assumption A4 (individual feasibility at arrival).** $E_i^{\mathrm{arr}}\ge\hat B_i(\tau_i(t_i^{\mathrm{arr}}))$ for the stored table.

DD2 enforces A4 by construction (F6), with the same routine as the layer (`cap_request()`):
1. If the table for $E^{\mathrm{des}}$ certifies $E^{\mathrm{arr}}$, keep $E^{\mathrm{req}}=E^{\mathrm{des}}$.
2. Otherwise start from $y=\Psi_{\tau}(E^{\mathrm{arr}})-10^{-6}$ and lower $y$ geometrically until the table for $y$ certifies $E^{\mathrm{arr}}$.

Why this terminates quickly:
- In exact arithmetic with the exact table, A1 alone gives $E^{\mathrm{arr}}\ge B(\tau;y)$ for $y=\Psi_\tau(E^{\mathrm{arr}})-10^{-6}$, so the first candidate is admissible.
- A1′ makes $\Psi_\tau$ 1-Lipschitz, which gives room $B(\tau;y)\le E^{\mathrm{arr}}-10^{-6}$. This absorbs the bisection error of $\hat B$, so the first candidate is usually also certified numerically.
- The value $y=\min(E^{\mathrm{arr}},E^{\mathrm{des}})$ is always certified, so the loop terminates.

**Admission control re-certifies A4.** When admission control (Sec. 7.4) lowers $E_j^{\mathrm{req}}$, the new table is produced by the same certifying loop (`cap_request` with the reduced target as $E^{\mathrm{des}}$), not by a bare `thresholds()` call. A4 therefore holds for every stored table.

---

## 4 The projection

### 4.1 Problem, feasibility and closed form

With $x=P^{\mathrm{ref}}a\in\mathbb R^N$ [kW], bounds $l\le u$ from (S11), $\tilde C=C-\varepsilon_C$ ($\varepsilon_C=0$ in the exact model; Sec. 6.3) and $C_{\mathrm{rev}}$:
$$p^{*}=\Pi_t(x)=\arg\min_{p\in\mathbb R^N}\tfrac12\|p-x\|_2^2\quad\text{s.t.}\quad l\le p\le u,\ \ -C_{\mathrm{rev}}\le\mathbf 1^\top p\le\tilde C.\tag{S13}$$
Its feasible set is $\mathcal K(t)$.

**Proposition 1 (projection).**
(i) $\mathcal K(t)\neq\emptyset$ iff $l\le u$, $\sum_il_i\le\tilde C$, $\sum_iu_i\ge-C_{\mathrm{rev}}$ and $-C_{\mathrm{rev}}\le\tilde C$.
(ii) Let
$$\mathcal S(\lambda)=\sum_i\operatorname{clip}(x_i-\lambda,l_i,u_i).\tag{S14}$$
If $\mathcal K(t)\ne\emptyset$, the unique minimiser of (S13) is
$$p_i^{*}=\operatorname{clip}(x_i-\lambda^{*},l_i,u_i),\qquad \lambda^{*}=\begin{cases}0,& \mathcal S(0)\in[-C_{\mathrm{rev}},\tilde C],\\ \text{any root of }\mathcal S(\lambda)=\tilde C\ \ (\text{then }\lambda^{*}>0),& \mathcal S(0)>\tilde C,\\ \text{any root of }\mathcal S(\lambda)=-C_{\mathrm{rev}}\ \ (\text{then }\lambda^{*}<0),& \mathcal S(0)<-C_{\mathrm{rev}}.\end{cases}\tag{S15}$$
All roots give the same $p^{*}$. $\lambda^{*}$ is unique whenever the free set $\mathcal F=\{i:l_i<x_i-\lambda^{*}<u_i\}$ is non-empty.
(iii) *(Supplementary material, DD40; not used by the main algorithm, which treats $\Pi_t$ as part of the environment.)* On the open set where $\mathcal F$ and the active aggregate constraint are locally constant, with $D=\mathrm{diag}(\mathbb 1[i\in\mathcal F])$:
$$\frac{\partial p^{*}}{\partial x}=J=\begin{cases}D-\dfrac{(D\mathbf 1)(\mathbf 1^\top D)}{\mathbf 1^\top D\mathbf 1}, & \text{aggregate constraint active and }\mathcal F\ne\emptyset,\\ 0, & \text{aggregate constraint active and }\mathcal F=\emptyset,\\ D, & \text{aggregate constraint inactive}.\end{cases}\tag{S16}$$

*Proof.*
**(i) Feasibility.**
- Necessity: if $p\in\mathcal K$, then $l\le p\le u$ gives $l\le u$ and $\sum l\le\mathbf 1^\top p\le\sum u$. Combined with $-C_{\mathrm{rev}}\le\mathbf 1^\top p\le\tilde C$, this gives the inequalities.
- Sufficiency: as $\upsilon$ runs over $[0,1]$, $p(\upsilon)=l+\upsilon(u-l)$ stays in the box and $\mathbf 1^\top p(\upsilon)$ takes every value in $[\sum l,\sum u]$. This interval meets the non-empty interval $[-C_{\mathrm{rev}},\tilde C]$ iff $\sum l\le\tilde C$ and $\sum u\ge-C_{\mathrm{rev}}$.

**(ii) Solution.**
1. **Existence and uniqueness.** (S13) minimises a strictly convex quadratic over a non-empty polyhedron, so the minimiser exists and is unique. With linear constraints, the KKT conditions are necessary and sufficient.
2. **KKT system.** With multipliers $\mu^{+},\mu^{-}\ge0$ (aggregate) and $\nu^{l},\nu^{u}\ge0$ (box):
$$p-x+(\mu^{+}-\mu^{-})\mathbf 1-\nu^{l}+\nu^{u}=0,\quad \mu^{+}(\mathbf 1^\top p-\tilde C)=0,\quad \mu^{-}(\mathbf 1^\top p+C_{\mathrm{rev}})=0,\quad \nu^{l}_i(p_i-l_i)=0,\quad \nu^{u}_i(u_i-p_i)=0.$$
3. **Choice of $\lambda^{*}$.** $\mathcal S$ is continuous, non-increasing and piecewise linear, with $\mathcal S(\lambda)=\sum u$ for $\lambda\le\min_i(x_i-u_i)$ and $\mathcal S(\lambda)=\sum l$ for $\lambda\ge\max_i(x_i-l_i)$.
   - If $\mathcal S(0)>\tilde C\ge\sum l$, the intermediate value theorem gives a root of $\mathcal S=\tilde C$. Every such root is $>0$.
   - The case $\mathcal S(0)<-C_{\mathrm{rev}}\le\sum u$ is symmetric, with $\lambda^{*}<0$.
4. **Verification of the KKT conditions.** Set $p=\operatorname{clip}(x-\lambda^{*},l,u)$, $\mu^{+}=\max(\lambda^{*},0)$, $\mu^{-}=\max(-\lambda^{*},0)$, and per coordinate:
   - if $l_i<x_i-\lambda^{*}<u_i$: $\nu_i^{l}=\nu_i^{u}=0$;
   - if $x_i-\lambda^{*}\le l_i$: $\nu_i^{l}=l_i-x_i+\lambda^{*}\ge0$, $\nu_i^{u}=0$;
   - if $x_i-\lambda^{*}\ge u_i$: $\nu_i^{u}=x_i-\lambda^{*}-u_i\ge0$, $\nu_i^{l}=0$.

   Stationarity holds coordinate-wise. Complementarity holds because $\lambda^{*}>0$ only if $\mathbf 1^\top p=\tilde C$, $\lambda^{*}<0$ only if $\mathbf 1^\top p=-C_{\mathrm{rev}}$, and in the first case $\mathbf 1^\top p=\mathcal S(0)$ is feasible. Hence $p=p^{*}$.
5. **All roots give the same $p^{*}$.** If $\lambda_1<\lambda_2$ are both roots, each coordinate $\operatorname{clip}(x_i-\lambda,l_i,u_i)$ is non-increasing in $\lambda$. Their sum is equal at $\lambda_1$ and $\lambda_2$, so every coordinate is equal.
6. **Uniqueness of $\lambda^{*}$.** If $\mathcal F\ne\emptyset$, both one-sided slopes of $\mathcal S$ at $\lambda^{*}$ are $\le-|\mathcal F|<0$.

**(iii) Jacobian.**
- If the aggregate constraint is inactive, then locally $p_i=x_i$ for $i\in\mathcal F$ and $p_i\in\{l_i,u_i\}$ is constant otherwise, so $J=D$.
- If it is active with target $\vartheta\in\{\tilde C,-C_{\mathrm{rev}}\}$, then $\sum_{i\in\mathcal F}(x_i-\lambda)+\sum_{i\notin\mathcal F}p_i=\vartheta$. This gives $\lambda=(\sum_{i\in\mathcal F}x_i+\sum_{i\notin\mathcal F}p_i-\vartheta)/|\mathcal F|$ and $\partial\lambda/\partial x_j=\mathbb 1[j\in\mathcal F]/|\mathcal F|$. Hence $\partial p_i/\partial x_j=\mathbb 1[i\in\mathcal F]\big(\mathbb 1[i=j]-\mathbb 1[j\in\mathcal F]/|\mathcal F|\big)$, which is (S16).
- If $\mathcal F=\emptyset$, $p$ is locally constant.
- $p^{*}$ fails to be differentiable only on the null set where some $x_i-\lambda^{*}\in\{l_i,u_i\}$ or $\mathcal S(0)\in\{\tilde C,-C_{\mathrm{rev}}\}$. There, the matrices (S16) of the adjacent regions are elements of Clarke's generalised Jacobian. $\square$

Remarks:
- $J$ is symmetric and idempotent. It is the orthogonal projector onto $\{v: v_i=0\ \forall i\notin\mathcal F,\ \mathbf 1^\top v=0\}$ (active case) or onto $\{v:v_i=0\ \forall i\notin\mathcal F\}$ (inactive case).
- As a Euclidean projection onto a convex set, $\Pi_t$ is non-expansive: $\|\Pi_t(x)-\Pi_t(y)\|\le\|x-y\|$.

### 4.2 Algorithms (floating-point robust)

**Feasibility test (A5 at step $t$).** $\mathcal K(t)\ne\emptyset$ iff $\sum_il_i\le\tilde C$, $\sum_iu_i\ge-C_{\mathrm{rev}}$ and $\tilde C\ge-C_{\mathrm{rev}}$; $l\le u$ holds by construction. All three conditions are checked (`is_feasible()`). The projection is called only on feasible steps; otherwise the fallback rule (Sec. 7.2) applies.

**Sorted breakpoints, $O(N\log N)$ per row** (`project()`).
1. Replace non-finite $x_i$ by 0. Compute $p^{0}=\operatorname{clip}(x,l,u)$. If $\mathbf 1^\top p^{0}\in[-C_{\mathrm{rev}},\tilde C]$, return it with $\lambda=0$.
2. Otherwise set the target $\vartheta=\tilde C$ (if $\mathbf 1^\top p^{0}>\tilde C$) or $\vartheta=-C_{\mathrm{rev}}$.
3. Form the $2N$ breakpoints $\{x_i-u_i\}\cup\{x_i-l_i\}$ with events $+1$ ($i$ becomes free) and $-1$ ($i$ becomes fixed at $l_i$). Sort them into $\lambda_{(1)}\le\dots\le\lambda_{(2N)}$.
4. Compute $\mathcal S_1=\sum u$ and $\mathcal S_{j+1}=\mathcal S_j-n^{\mathrm{free}}_j(\lambda_{(j+1)}-\lambda_{(j)})$, where $n^{\mathrm{free}}_j$ is the cumulative sum of events up to $j$. **Overwrite $\mathcal S_{2N}:=\sum_il_i$**, computed directly. In exact arithmetic the recursion gives this value; in floating point it differs by rounding.
5. Let $j^{*}$ be the first index with $\mathcal S_{j^{*}}\le\vartheta$.
   - **If $j^{*}$ exists:** interpolate linearly on $[\lambda_{(j^{*}-1)},\lambda_{(j^{*})}]$: $\lambda=\lambda_{(j^{*}-1)}+(\mathcal S_{j^{*}-1}-\vartheta)(\lambda_{(j^{*})}-\lambda_{(j^{*}-1)})/(\mathcal S_{j^{*}-1}-\mathcal S_{j^{*}})$, or $\lambda=\lambda_{(1)}$ if $j^{*}=1$.
   - **If no index satisfies $\mathcal S_j\le\vartheta$:** this happens only when A5 is tight to rounding. Take $\lambda=\lambda_{(2N)}+1+10^{-6}\max_i|x_i|$, which is strictly beyond the last breakpoint, so that $p=l$ exactly in floating point.
6. Return $p=\operatorname{clip}(x-\lambda,l,u)$.
7. **Post-condition.** If the aggregate constraint was active and $\mathbf 1^\top p\notin[-C_{\mathrm{rev}}-\epsilon_{\mathrm{num}},\tilde C+\epsilon_{\mathrm{num}}]$, recompute that row with the safe bisection below.

**Why this is needed (review issue M1).** In exact arithmetic $\mathcal S_{2N}=\sum l\le\vartheta$ guarantees that $j^{*}$ exists. In floating point, the accumulated $\mathcal S_{2N}$ can exceed $\vartheta$ when A5 holds with a slack below rounding level, about $10^{-12}\max|x|$. The original specification then took index 0 by default, i.e. $\lambda=\lambda_{(1)}$ and $p=u$. In the reviewer's targeted test, 8742 of 20000 exactly tight rows did this, overshooting $\tilde C$ by up to 418 kW. With steps 4, 5 and 7 the failure cannot occur.

**Safe bisection, $O(N\log(1/\epsilon))$** (`_bisect_safe()`, also used by `project_bisect()`).
- **Upper case:** bisect $\mathcal S(\lambda)=\vartheta$ on $[0,\ \max_i(x_i-l_i)+1+10^{-6}\max|x|]$, and always keep the end where $\mathcal S\le\tilde C$ as evaluated.
- **Lower case:** use $[\min_i(x_i-u_i)-1-10^{-6}\max|x|,\ 0]$, keeping the end where $\mathcal S\ge-C_{\mathrm{rev}}$.
- Return $\operatorname{clip}(x-\lambda,l,u)$ at the kept end.
- The padded bracket ends make $\mathcal S$ at the bracket end equal to $\sum l$ (resp. $\sum u$) exactly in floating point. The aggregate constraint therefore holds as evaluated, whatever the magnitude of $x$.
- `project_bisect()` additionally tries the exact polish $\lambda=(\sum_{\mathcal F}x_i+\sum_{\notin\mathcal F}p_i-\vartheta)/|\mathcal F|$ and accepts it only if it passes the post-condition.

**Batched and masked (PyTorch).** Rows are padded to $N_{\max}$. Padded or empty entries have $x=l=u=0$. Their breakpoints are $\{0,0\}$ with events $+1,-1$, which change no segment slope. All operations used — sort, gather, cumsum, `any`, `argmax`, `where` — exist in PyTorch. Pseudocode is in `algorithm.md`, Alg. 1. Use float64 for the layer.

**Numerical confirmation** (`verify_projection.py`, `verify_projection_results.txt`):
- **T1:** on 3000 random instances with $N$ from 1 to 200, the closed form agrees with SciPy SLSQP to $3.4\times10^{-10}$ kW and with bisection to $7.6\times10^{-13}$, with KKT residual $1.4\times10^{-12}$.
- **T1b:** 20000 rows each in five adversarial families (exactly tight upper, exactly tight lower, tight upper with $|x|\in[10^4,10^8]$, general with $|x|\in[10^4,10^8]$, hyperplane $\tilde C=-C_{\mathrm{rev}}$). Maximum aggregate violation $\le1.0\times10^{-9}$ kW ($=\epsilon_{\mathrm{num}}$), zero box violations, and agreement with bisection to $\le1.4\times10^{-15}$ relative. Non-finite inputs are replaced by 0.
- **T2:** the Jacobian (S16) agrees with central finite differences to $2.1\times10^{-8}$ (500 points, 162 with $J=0$).

### 4.3 Relation to known projections (stated openly)

(S13)-(S15) is the Euclidean projection onto a box intersected with a two-sided hyperplane slab, i.e. the continuous quadratic knapsack problem:
- Breakpoint algorithms and their linear-time variants are classical [kiwiel2007breakpoint].
- A semismooth-Newton method with an explicit generalised Jacobian exists for this projection [wang2022efficient].
- The form $\operatorname{clip}(x_i-\lambda,l_i,u_i)$ with a common level found by one-dimensional search is also the rate rule of smoothed least-laxity-first scheduling [chen2022smoothed].

None of this algebra is new. What differs in our use is:
1. the input is a learned per-charger action of a size-agnostic policy;
2. $l_i$ comes from backward reachability including the taper and the V2G sign range;
3. the aggregate constraint is two-sided (forward and reverse);
4. the layer is applied during training as part of the environment;
5. it is combined with the multi-step invariance argument of Sec. 5.

---

## 5 Feasibility preservation and invariance

### 5.1 Main result (DD30)

*Intuition.* The safety layer never lets an EV fall below the least energy from which its departure requirement is still reachable at full power. The common shift $\lambda$ keeps the station inside the transformer band. The first property can fail only when the transformer cannot serve all EVs' minimum needs at once, an infeasible step. The second never fails, because then the fallback rule serves the transformer first.

Theorem 1′ states the result in two layers:
- the transformer limit holds for every raw-action sequence, under conditions on the simulator and the exogenous grid only;
- the departure requirement is a per-EV statement that is conditional on the EV's own stay.

The global Theorem 1 (Sec. 5.3) is the special case in which no step is infeasible. The induction is the classical laxity argument of [chen2022smoothed] (feasibility iff all laxities stay non-negative), extended to:
- the taper, via certified thresholds that require A1;
- V2G, via a signed lower bound;
- a two-sided aggregate coupling;
- arbitrary raw actions of a learned policy.

**Assumption A6 (exogenous headroom).** For every step $t$: $C(t)\ge N\varepsilon_q$ and $C_{\mathrm{rev}}(t)\ge0$. A6 depends only on $P_{\mathrm{tr}},P^{\mathrm{inf}},P^{\mathrm{pv}}$, not on the policy, and is checked online at every step.

**Theorem 1′ (feasibility preservation).** Consider the closed loop in which, at every step $t$:
- the commanded power is $p(t)=\Pi_t(x(t))$ if the feasibility test of Sec. 4.2 passes, and the fallback rule of Sec. 7.2 otherwise;
- $(x(t))_t$ is an arbitrary sequence of raw actions (non-finite entries replaced by 0) — in particular the actions of any policy, during training and deployment.

Assume A3 and A6. Then:
- **(a) Transformer limit, for every raw-action sequence:** $-C_{\mathrm{rev}}(t)\le\sum_i\tilde p_i(t)\le C(t)$ at every step $t$, on feasible and fallback steps alike.
- **(b) Physical limits:** $p_i^{\min}(E_i(t))\le p_i(t)\le\bar P_i(E_i(t))$ at every step, for every connected EV.

If, in addition, A1, A2 and A4 hold:
- **(c) Departure requirement, per EV:** every EV $i$ with $p_i(t)\ge l_i(t)$ for all $t\in[t_i^{\mathrm{arr}},d_i]$ meets $E_i(d_i+1)\ge E_i^{\mathrm{req}}$. In particular, this holds for every EV whose stay contains no infeasible step, whatever happens at other steps and whatever the raw actions are.
- **(d) Global special case (Theorem 1):** if A5 holds at every step, every EV meets its departure requirement.

**Instantiation in EV2Gym.**
- A3 holds under the configuration of Corollary 2 (Lemmas E2, E3).
- A1 holds by Lemma E1.
- A2 holds by construction (known departure; Proposition 2 relaxes it).
- A4 holds by the request pipeline (Sec. 3.4 and 7.4).
- A6 holds under the DD24 scaling (Corollary 2, item 8) except on events of probability below $2\times10^{-9}$ per episode.

The proof is given after Corollary 2 (Sec. 5.4).

### 5.2 One-step invariance

**Lemma 1 (one-step individual invariance).** Assume A1 and A3, and let $\hat B$ satisfy (S10). Let EV $i$ be connected at $t$ with $\tau=\tau_i(t)$ and individually feasible, $E=E_i(t)\ge\hat B(\tau)$. Then:
(a) $p^{\min}(E)\le l_i(t)\le u_i(t)$;
(b) for every $p\in[l_i(t),u_i(t)]$, the next energy satisfies $E_i(t+1)\ge\hat B(\tau-1)$.

In particular:
- if $\tau\ge2$, EV $i$ is individually feasible at $t+1$, because $\tau_i(t+1)=\tau-1$;
- if $\tau=1$ ($t=d_i$), then $E_i(d_i+1)\ge\hat B(0)\ge E_i^{\mathrm{req}}$.

*Proof.*
(a) $p^{\min}\le l$ by (S11). For $l\le u$: $p^{\min}\le0\le\bar P(E)$. Moreover,
$$g^{-1}((\hat B(\tau-1)-E)/\Delta t)\le\bar P(E)\iff\hat B(\tau-1)\le\varphi(E).$$
By A1 and (S10), $\varphi(E)\ge\varphi(\hat B(\tau))\ge\hat B(\tau-1)$.
(b) Let $y=(\hat B(\tau-1)-E)/\Delta t$. Since $l\ge g^{-1}(y)$ and $g$ is increasing, $p\ge l$ gives $g(p)\ge y$. By A3, $E_i(t+1)\ge E+\Delta t\,g(p)\ge\hat B(\tau-1)$. $\square$

A1 enters exactly once: $E\ge\hat B(\tau)\Rightarrow\varphi(E)\ge\varphi(\hat B(\tau))$. As shown at the end of Sec. 3.1, certification of the table does not compensate for a violation of A1.

**Assumptions of the theorem (full list in Sec. 10).**
- **A2 (known data).** $d_i$ and $E_i^{\mathrm{req}}$ are known at arrival.
- **A3 (one-step model).** For every connected $i$ and every commanded $p_i\in[p^{\min}(E_i),\bar P(E_i)]$:
  - the realised energy satisfies $E_i(t)+\Delta t\,g(p_i)\le E_i(t+1)\le E_i^{\mathrm{cap}}$;
  - the realised power satisfies $p_i\le\tilde p_i\le p_i+\varepsilon_q$;
  - empty chargers draw $\tilde p_i=0$.

  In the exact model, $\varepsilon_q=0$ and the first inequality is an equality.
- **A5 (aggregate per-step feasibility, closed loop).** At every step $t$ of the closed loop:
$$\sum_{i\in\mathcal N(t)}l_i(t)\le\tilde C(t),\qquad\sum_{i\in\mathcal N(t)}u_i(t)\ge-C_{\mathrm{rev}}(t),\qquad\tilde C(t)\ge-C_{\mathrm{rev}}(t).$$
  The last condition means $2P_{\mathrm{tr}}(t)\ge n(t)\varepsilon_q$ and is checked like the others.

### 5.3 Global invariance (special case)

**Theorem 1 (feasibility invariance).** Consider the closed loop in which, at every step $t$, the commanded powers are $p(t)=\Pi_t(x(t))$ for an arbitrary finite raw action $x(t)\in\mathbb R^N$, in particular for actions sampled from any policy, during training and deployment. Suppose:
- A1: $\varphi_i$ is continuous and non-decreasing for every EV;
- A2: $d_i$ and $E_i^{\mathrm{req}}$ are known at arrival;
- A3: the one-step model holds with margin $\varepsilon_q$, and $\tilde C=C-n(t)\varepsilon_q$;
- A4: $E_i^{\mathrm{arr}}\ge\hat B_i(\tau_i(t_i^{\mathrm{arr}}))$ for every EV;
- A5: aggregate per-step feasibility holds at every step.

Then for every step $t$ and every EV $i$:
(a) the transformer limit holds for the realised powers, $-C_{\mathrm{rev}}(t)\le\sum_i\tilde p_i(t)\le C(t)$;
(b) the per-step physical limits hold, $p_i^{\min}(E_i(t))\le p_i(t)\le\bar P_i(E_i(t))$;
(c) every EV meets its departure energy requirement, $E_i(d_i+1)\ge E_i^{\mathrm{req}}$.

*Proof.* Induction over $t$ on the invariant
$$\mathrm{I}(t):\quad E_i(t)\ge\hat B_i(\tau_i(t))\quad\text{for all }i\in\mathcal N(t).$$

*Base and arrivals.* An EV belongs to $\mathcal N(t)$ either because it arrived at $t$, in which case its threshold inequality is A4, or because it was in $\mathcal N(t-1)$ with $d_i\ge t$, in which case the induction step applies. $\mathcal N(0)$ consists of arrivals only, so I(0) holds.

*Step $t\to t+1$.* Assume I($t$).
1. **Bounds.** Every $i\in\mathcal N(t)$ is individually feasible, so $l_i(t)$ is given by the first line of (S11), and $p_i^{\min}\le l_i\le u_i$ by Lemma 1(a). Empty chargers have $l_i=u_i=0$.
2. **Projection is well defined.** By A5 and Proposition 1(i), $\mathcal K(t)\ne\emptyset$. Hence $p(t)=\Pi_t(x(t))\in\mathcal K(t)$, whatever $x(t)$ is.
3. **Claim (b).** $p_i(t)\in[l_i,u_i]\subseteq[p_i^{\min}(E_i(t)),\bar P_i(E_i(t))]$.
4. **Claim (a).** $\mathbf 1^\top p(t)\in[-C_{\mathrm{rev}}(t),\tilde C(t)]$. By A3,
$$\sum_i\tilde p_i\in\Big[\sum_ip_i,\ \sum_ip_i+n(t)\varepsilon_q\Big]\subseteq[-C_{\mathrm{rev}}(t),\ \tilde C(t)+\varepsilon_C(t)]=[-C_{\mathrm{rev}}(t),C(t)].$$
5. **Continuation.** For each $i\in\mathcal N(t)$, Lemma 1(b) gives $E_i(t+1)\ge\hat B_i(\tau_i(t)-1)$.
   - If $d_i=t$, the EV departs with $E_i(d_i+1)\ge\hat B_i(0)\ge E_i^{\mathrm{req}}$. This proves (c) for this EV.
   - If $d_i>t$, then $i\in\mathcal N(t+1)$ and $\tau_i(t+1)=\tau_i(t)-1$, so its part of I($t+1$) holds.
6. **New arrivals.** EVs arriving at $t+1$ satisfy their part of I($t+1$) by A4.
7. **Coverage.** All departures occur at $d_i\le T-2$ in EV2Gym (formulation 4.5), so (c) is established for every EV of the episode. $\square$

**Corollary 1 (localisation; no global A5 needed).** Under A1-A4:
- every EV $i$ with $p_i(t)\ge l_i(t)$ at every step of its stay meets its departure energy requirement;
- in particular, this holds for every EV whose stay contains no infeasible step, whatever happens at other steps;
- claim (b) holds at every step, fallback steps included;
- the transformer limit for realised powers holds at every step except FB-0, FB-2 and FB-1* steps (Sec. 7.2), where no admissible command satisfies it. These steps cannot occur under A6 (Theorem 1′(a)).

*Proof.* The per-EV part of the induction (Lemma 1 and A4) uses only $p_i(t)\in[l_i(t),u_i(t)]$ during the stay of EV $i$. Fallback outputs satisfy $p^{\min}\le p\le u$, so $p_i\ge l_i$ implies $p_i\in[l_i,u_i]$. On feasible steps the transformer part of the induction step applies unchanged. On FB-1 steps, Lemma 2(a) gives $\mathbf 1^\top p=\tilde C\ge-C_{\mathrm{rev}}$, and A3 transfers this to the realised powers. $\square$

**Corollary 2 (EV2Gym instantiation).** In EV2Gym at the pinned commit, A1 and A3 hold if all of the following configuration items hold:
1. C0: $P_i^{\mathrm{ev}}\le P^{\mathrm{cs}}$ for every EV model (true for the 8 models);
2. **efficiency constant over current levels, per EV.** DD4 is realised by our own spec JSON whose `3ph_ch_efficiency` table is constant across current levels: the table value at the current level closest to the model's nominal AC current, i.e. 0.90 for all models and 0.93 for the Hyundai Kona (IMPLEMENTATION_NOTES I-17). Without the table key, EV2Gym draws a random scalar per EV instead. The layer reads $\zeta_i^{\mathrm c},\zeta_i^{\mathrm d}$ from each EV object;
3. one port per charger (DD5), 3-phase chargers and EVs (`phases: 3`, `ev_phases = 3`);
4. zero dead-bands: EV `min_ac_charge_power = min_discharge_power = 0` (the heterogeneous-spec default), and charger `min_charge_current = min_discharge_current = 0` (`V2GProfitMax.yaml`);
5. `v2g_enabled: True`, so that negative actions and $P^{\mathrm{dis}}>0$ are available;
6. `transition_soc_multiplier` $m=5$, with $\kappa_i\le\kappa^{*}(5)$ for all EVs (Lemmas E1 and E2; verified for all 8 models at 5 and 15 min);
7. the actuation map of Sec. 6.3 with $\varepsilon_q=0.008$ kW, and `empty_ports_at_end_of_simulation=True` (coverage in Theorem 1);
8. **A6 under the DD24 scaling.** EV2Gym scales the inflexible-load peak to $m_LP_{\mathrm{tr}}$ with $m_L\sim\mathcal N(0.4,0.1)$ and clips the load at $P_{\mathrm{tr}}$. PV is $\mathrm{pv}(t)\,m_PP_{\mathrm{tr}}$ with $\mathrm{pv}\le0.853\cdot1.1=0.94$ and $m_P\sim\mathcal N(0.4,0.1)$.
   - $C(t)\ge P_{\mathrm{tr}}(1-m_L)\ge N\varepsilon_q$ unless $m_L>1-\varepsilon_q/\rho_{\mathrm{tr}}\ge0.997$ ($\rho_{\mathrm{tr}}\ge3$ kW), which is a 5.97σ event.
   - $C_{\mathrm{rev}}(t)\ge P_{\mathrm{tr}}(1-0.94m_P)\ge0$ unless $m_P>1.06$, which is a 6.6σ event.
   - Both together have probability below $2\times10^{-9}$ per episode. Any violation is logged as an FB-0 or FB-2 step.

With $E_i^{\mathrm{req}}$ from (F6), A4 also holds, so Theorem 1 and Corollary 1 hold with A5 as the only remaining condition.

### 5.4 Proof of Theorem 1′

*(a), (b) on feasible steps.* By the guard $l\le u$ (Sec. 3.2) and the feasibility test, $\mathcal K(t)\ne\emptyset$ (Proposition 1(i)), so $p(t)\in\mathcal K(t)$. Then $p_i\in[l_i,u_i]\subseteq[p_i^{\min},\bar P_i]$, and step 4 of the proof of Theorem 1 (A3) gives the realised band.

*(a), (b) on fallback steps.* Under A6, $\tilde C(t)=C(t)-n(t)\varepsilon_q\ge0$ and $C_{\mathrm{rev}}(t)\ge0$, so:
- FB-1* ($\tilde C<-C_{\mathrm{rev}}$) is impossible;
- FB-2 ($\sum u<-C_{\mathrm{rev}}$) is impossible, because $u\ge0$;
- FB-0 (budget $\tilde C-\sum p^{\min}<0$) is impossible, because $p^{\min}\le0$.

Hence every infeasible step is an FB-1 step. Lemma 2(a) gives $p^{\min}\le p\le u$ and $\mathbf 1^\top p=\tilde C\in[-C_{\mathrm{rev}},\tilde C]$, and A3 transfers this to the realised powers. Neither part uses A1, A2, A4 or A5.

*(c)* is Corollary 1. *(d)* is Theorem 1. $\square$

**Closed-loop evidence.**
- `verify_projection.py` T4 drives EV2Gym `EV` objects with random raw actions (standard deviation 16.5 kW) in three settings (5138 EVs). No EV whose stay contained no infeasible step missed $E^{\mathrm{req}}$, and the transformer limit was never violated on feasible steps.
- The reviewer's independent closed loop covers about 30000 EVs over 7 configurations that satisfy A1, including lossy $\eta$, 5-min steps, conformal $\tau^{\mathrm{lo}}$, and $\pm10^4$ and Cauchy actions. It found zero violations of Theorem 1(a)-(c), Corollary 1, Lemma 2(a) and Lemma 3.

---

## 6 Assumptions A1 and A3 in EV2Gym

### 6.1 A1, A1′
Lemma E1 (under C0) shows that A1 and A1′ hold analytically for every $\Delta t$. The numerical check is in Sec. 2.3.

### 6.2 Exact delivery
Lemma E2 covers charging for $m=5$ and our parameter range. Discharge delivery is always exact.

### 6.3 Actuation, rounding and margin (A3 with $\varepsilon_q$)

EV2Gym's charger has two effects that the actuation map must handle (`ev_charger.py:157-186`):
- it rounds the normalised action to 5 decimals, a power step of $\zeta P^{\mathrm{cs}}10^{-5}\le2.22\times10^{-4}$ kW;
- it zeroes discharge currents below 0.01 A, a dead-band of $|p|<0.0069$ kW.

It also rounds `current_capacity` up to 0.01 kWh after each active step (`ev.py:182-189`). The reported energy is computed before this rounding.

**Actuation map** (`actuation()` in `projection_ref.py`), with $\varepsilon_{\mathrm{sat}}=10^{-3}$ kW:
1. If $p_i\ge u_i-\varepsilon_{\mathrm{sat}}$ and $u_i>0$, send $a^{\mathrm{env}}_i=1$ (full current).
2. Else if $p_i>0$, send $a^{\mathrm{env}}_i=\lceil10^5p_i/(\zeta_i^{\mathrm c}P^{\mathrm{cs}})\rceil/10^5$.
3. Else if $p_i<0$, send $a^{\mathrm{env}}_i=\lceil10^5p_i/(\zeta_i^{\mathrm d}P^{\mathrm{cs}})\rceil/10^5$ (rounded towards zero).
4. Else send 0.

Rounding is upwards up to floating-point resolution. The code computes $\lceil\cdot-10^{-9}\rceil$ to avoid spurious round-ups of exact grid values, so the command can be below $p_i$ by at most about $10^{-14}$ kW, which is covered by $\epsilon_{\mathrm{num}}$ (Sec. 6.4). Non-finite $p$ is rejected by an assertion.

**Lemma E3 (A3 in EV2Gym).** Under Corollary 2's configuration, the realised reported power satisfies $p_i\le\tilde p_i\le p_i+\varepsilon_q$ with $\varepsilon_q=0.008$ kW, and the realised energy satisfies $E_i+\Delta t\,p_i\le E_i(t+1)\le E_i^{\mathrm{cap}}$, both up to $\epsilon_{\mathrm{num}}$.

*Proof.*
1. **Saturated case.** Full current delivers $\varphi(E)-E=\Delta t\,u_i$, and $u_i-p_i\le\varepsilon_{\mathrm{sat}}$.
2. **Case $0<p_i<u_i-\varepsilon_{\mathrm{sat}}$.** The commanded power $\zeta^{\mathrm c}a^{\mathrm{env}}P^{\mathrm{cs}}\in[p_i,p_i+2.22\times10^{-4}]\subset[p_i,u_i)$ is delivered exactly by Lemma E2. EV2Gym's `round(·,5)` leaves a value already on the grid unchanged.
3. **Case $p_i<0$.**
   - The rounded command lies in $[p_i,p_i+2.22\times10^{-4}]$.
   - If the dead-band applies, $\tilde p_i=0$ and $\tilde p_i-p_i=|p_i|<0.0069+2.22\times10^{-4}$.
   - Otherwise $\tilde p_i\ge p_i\ge p^{\min}(E)$, so neither the EV discharge limit nor the $E^{\min}$ floor clips it.
4. **Margin.** In all cases $0\le\tilde p_i-p_i\le\max\{10^{-3},\,0.0072\}<0.008$ kW.
5. **Energy.** $E(t+1)=\lceil E+\Delta t\tilde p_i\rceil_{0.01}\ge E+\Delta t\tilde p_i\ge E+\Delta tp_i$. Also $E+\Delta t\tilde p_i\le\varphi(E)\le E^{\mathrm{cap}}$. All 8 capacities are multiples of 0.01 kWh, and are exactly representable after the ceil in IEEE double (checked by the reviewer), so the ceil cannot exceed $E^{\mathrm{cap}}$. $\square$

Thus the safety layer uses $\tilde C(t)=C(t)-n(t)\varepsilon_q$ (DD16). The cost is at most 1.6 kW at $N=200$. The reverse limit needs no margin, because $\tilde p\ge p$. The reviewer's run **without** the tightening produced realised violations of up to 0.0073 kW on A5 steps and on every FB-1 step, so the margin is necessary. The ceil can only add energy. Its effect on the oracle bound is treated by Proposition O1(b) in `formulation.md`.

### 6.4 Numerical tolerance

The code uses a single tolerance, $\epsilon_{\mathrm{num}}=10^{-9}$ (kW or kWh):
- in the individual-feasibility test, $E\ge\hat B(\tau)-\epsilon_{\mathrm{num}}$;
- in the projection post-condition (Sec. 4.2).

It is absorbed as follows:
- **Power side.** The actual actuation error is at most 0.00715 kW per charger, against $\varepsilon_q=0.008$ kW. The unused margin of $n(t)\cdot8.5\times10^{-4}$ kW exceeds $\epsilon_{\mathrm{num}}$, so the realised transformer limit holds exactly.
- **Energy side.** Floating-point deficits are about $10^{-15}$ kWh per step. With the feasibility tolerance they accumulate to at most $T\epsilon_{\mathrm{num}}\le3.4\times10^{-7}$ kWh per session. This is far below EV2Gym's satisfaction tolerance of 0.001 kWh and the 0.01 kWh ceil.

Statements "exact" or "$\ge$" in this document are to be read up to this tolerance in code.

---

## 7 When A5 fails: remark, fallback rule, accounting, admission control

### 7.1 Remark: local bounds cannot guarantee A5

A5 is a **closed-loop** condition, because $l_i(t)$ depends on the energies the policy has produced. Per-EV bounds look only at each EV against its own deadline. They do not account for EVs competing for the same future capacity. This holds even when the capacity is constant.

*Counterexample.*
- Two identical EVs, $\Delta t=1$ h, $\bar P\equiv10$ kW, $P^{\mathrm{dis}}=0$, $E^{\min}=0$, $E_i(0)=0$, $E^{\mathrm{req}}=10$ kWh.
- $t^{\mathrm{arr}}=0$ and $d=1$, i.e. two actions each. Constant $C(0)=C(1)=10$ kW, with $C_{\mathrm{rev}}$ large.
- A joint plan exists: $p(0)=p(1)=(5,5)$.
- At $t=0$: $\tau=2$, $\hat B(1)=0$, so $l_i(0)=\max\{0,10-10\}=0$. A5 holds, and the raw action $x=(0,0)$ is projected to $p(0)=(0,0)$.
- At $t=1$: $l_i(1)=10$ each, so $\sum l=20>C(1)=10$, and A5 fails.

*Policy-independent sufficient condition.* $l_i(E)$ is non-increasing in $E$: $p^{\min}$ is non-increasing and $g^{-1}((\hat B-E)/\Delta t)$ is decreasing. Suppose that, at every step $t$,
$$C_{\mathrm{rev}}(t)\ge0,\qquad\tilde C(t)\ge-C_{\mathrm{rev}}(t),\qquad\sum_{i\in\mathcal N(t)}l_i^{\max}(t)\le\tilde C(t),\qquad l_i^{\max}(t):=l_i\ \text{evaluated at }E=\hat B_i(\tau_i(t)).$$
Then A5 holds at every step for **every** policy. The proof is by induction on I($t$), as in Theorem 1. If I($t$) holds, then $E_i(t)\ge\hat B_i(\tau_i(t))$ gives $l_i(t)\le l_i^{\max}(t)$, and $u_i\ge0$ gives $\sum u\ge0\ge-C_{\mathrm{rev}}$. So A5 holds at $t$, and the induction step of Theorem 1 yields I($t+1$).

But $l_i^{\max}(t)=\bar P_i(\hat B_i(\tau_i))$ whenever $\hat B_i(\tau_i)>E^{\min}$. The condition therefore asks the transformer to serve all non-trivially constrained EVs at full power simultaneously, which is very conservative.

The guarantee is therefore stated conditionally on A5 (Theorem 1) and localised per EV (Corollary 1), and its failure frequency is measured (risk R8). A layer that certifies future joint feasibility would need time-coupled constraints, e.g. nested deadline constraints. The projection would then no longer have the single-sum form (S15). This is left as future work. Admission control (Sec. 7.4) attacks the problem at arrival only. The transformer part of the result (Theorem 1′(a)) does not depend on A5 at all.

### 7.2 Fallback rule (DD8)

The fallback is used when the feasibility test of Sec. 4.2 fails. $C$ below always denotes the tightened $\tilde C$. Since $l\le u$ by construction, Proposition 1(i) leaves the following exclusive cases:

- **FB-1\*** ($\tilde C<-C_{\mathrm{rev}}$; the two limits contradict each other after tightening, which requires $2P_{\mathrm{tr}}<n\varepsilon_q$, e.g. a demand-response capacity drop to almost zero): apply the FB-1 procedure below with forward priority. If $\sum l\le\tilde C$, every EV reaches $l_i$ and the forward limit holds. The reverse limit is violated unavoidably.
- **FB-2** ($\tilde C\ge-C_{\mathrm{rev}}$ and $\sum_iu_i<-C_{\mathrm{rev}}$; PV export exceeds what all EVs can absorb): $p_i=u_i$ for all occupied $i$.
- **FB-1** ($\tilde C\ge-C_{\mathrm{rev}}$ and $\sum_il_i>\tilde C$; the departure requirements need more power than the transformer allows, and the transformer limit has priority):
  1. Set $p_i=p_i^{\min}(E_i)$ for all occupied $i$, and the budget $b=\tilde C-\sum_ip_i^{\min}$.
  2. **FB-0 sub-case:** if $b<0$, keep $p=p^{\min}$, the maximal discharge.
  3. Otherwise, visit the occupied EVs in the order given by the key ($\chi_i$ ascending, $l_i-p_i^{\min}$ descending, charger index ascending). Individually infeasible EVs have $\chi_i<0$ and come first. For each EV set $\Delta=\min\{l_i-p_i^{\min},\,b\}$, $p_i\leftarrow p_i+\Delta$, $b\leftarrow b-\Delta$, and stop when $b=0$.

FB-1 and FB-2 cannot occur together, because $\sum u<-C_{\mathrm{rev}}\le\tilde C<\sum l$ would contradict $\sum l\le\sum u$. Code and text use the same ordering key (`fallback()`, `np.lexsort`).

**Lemma 2 (fallback).**
(a) In case FB-1 with $b\ge0$, the output satisfies $p^{\min}\le p\le l\le u$ and $\mathbf 1^\top p=\tilde C(t)\in[-C_{\mathrm{rev}}(t),\tilde C(t)]$. So the transformer limit for realised powers holds (A3 margin), and the per-step physical limits hold.
(b) Every EV that is individually feasible at $t$ and receives $p_i=l_i$ remains individually feasible (Lemma 1). These are a prefix of the ordering, plus every feasible EV with $l_i=p_i^{\min}$. EVs that are infeasible at $t$ have $l_i=u_i$ and remain infeasible unless they reach $u_i$.
(c) FB-0 requires $\tilde C(t)<\sum_ip_i^{\min}\le0$, i.e. an inflexible load above $P_{\mathrm{tr}}+P^{\mathrm{pv}}$ up to the margin. FB-2 requires $C_{\mathrm{rev}}(t)<0$, i.e. PV export above $P_{\mathrm{tr}}+P^{\mathrm{inf}}$. In both cases the commanded output minimises the commanded violation, $(\mathbf 1^\top p-\tilde C)^{+}$ or $(-C_{\mathrm{rev}}-\mathbf 1^\top p)^{+}$, over the box $[p^{\min},u]$. No admissible command avoids it. The realised violation can differ from the commanded one by up to $n\varepsilon_q$. Under A6 neither case (nor FB-1*) can occur (Theorem 1′).

*Proof.*
(a) Since $\sum_i(l_i-p_i^{\min})>\tilde C-\sum p^{\min}=b$, the budget is exhausted before every EV reaches $l_i$. So $\mathbf 1^\top p=\tilde C$ and $p\le l$. Moreover $\tilde C\ge-C_{\mathrm{rev}}$.
(b) Lemma 1(b) with $p=l$.
(c) $\mathbf 1^\top p$ over the box is minimised at $p^{\min}$ and maximised at $u$. $\square$

In EV2Gym, the inflexible load is clipped to at most $P_{\mathrm{tr}}$, so $C(t)\ge P^{\mathrm{pv}}(t)\ge0$. FB-0 can then only arise when $C(t)<n(t)\varepsilon_q$ and no EV can discharge. FB-2 is excluded by the scenario design in `algorithm.md`, Alg. 7 (PV peak below $P_{\mathrm{tr}}$).

**Why the rule starts from $p^{\min}$.**
- EVs with $l_i=p_i^{\min}$ stay exactly at their bound, so they remain feasible, while the urgent EVs receive the largest possible budget.
- With $g=\mathrm{id}$, the aggregate deficit $\sum_i(l_i-p_i)=\sum l-\tilde C$ is the same for every rule with $\mathbf 1^\top p=\tilde C$. LLF only decides who carries it.

### 7.3 Shortfall accounting

**Lemma 3 (deficit bound).** Assume A1, A1′, A3 and A4, and let $\hat B$ satisfy (S10). Define $\tilde l_i(t)=l_i(t)$ if $E_i(t)\ge\hat B_i(\tau_i(t))$ and $\tilde l_i(t)=u_i(t)$ otherwise, and the feasibility deficit $\Delta E_i^{\mathrm{def}}(t)=(\hat B_i(\tau_i(t))-E_i(t))^{+}$ with $\tau_i(d_i+1)=0$. Then, for any sequence of commanded powers with $p_i(t)\in[p^{\min},u_i]$:
$$\Delta E_i^{\mathrm{sh}}=(E_i^{\mathrm{req}}-E_i(d_i+1))^{+}\ \le\ \Delta E_i^{\mathrm{def}}(d_i+1)\ \le\ \sum_{t=t_i^{\mathrm{arr}}}^{d_i}\Delta t\,\big(g(\tilde l_i(t))-g(p_i(t))\big)^{+}.$$
In the EV2Gym instantiation, the fleet's total shortfall is therefore at most
$$\Delta t\sum_{t\in\text{FB-1}}\Big(\sum_il_i(t)-\tilde C(t)\Big)+\Delta t\sum_{t\in\text{FB-0}}\sum_i\big(l_i(t)-p_i^{\min}(t)\big).$$

*Proof.* $\Delta E^{\mathrm{sh}}\le\Delta E^{\mathrm{def}}(d_i+1)$ because $\hat B(0)\ge E^{\mathrm{req}}$, and $\Delta E^{\mathrm{def}}(t_i^{\mathrm{arr}})=0$ by A4. We show $\Delta E^{\mathrm{def}}(t+1)\le\Delta E^{\mathrm{def}}(t)+\Delta t\,(g(\tilde l)-g(p))^{+}$ and telescope.
- **Case $E\ge\hat B(\tau)$.** By (S11), $E+\Delta tg(l)\ge\hat B(\tau-1)$, so by A3
$$E(t+1)\ge E+\Delta tg(p)\ge\hat B(\tau-1)-\Delta t(g(l)-g(p))^{+}.$$
- **Case $E<\hat B(\tau)$.** Here $\tilde l=u$ and $E+\Delta tg(u)=\varphi(E)$. By (S10) and A1′,
$$\hat B(\tau-1)-\varphi(E)\le\varphi(\hat B(\tau))-\varphi(E)\le\hat B(\tau)-E=\Delta E^{\mathrm{def}}(t).$$
  Hence $\hat B(\tau-1)-E(t+1)\le\Delta E^{\mathrm{def}}(t)+\Delta t(g(u)-g(p))$.

For the fleet bound: on feasible steps $p_i\ge\tilde l_i$. On FB-2 steps $p=u$. On FB-1 steps $p\le l$ and $\sum_i(l_i-p_i)=\sum l-\tilde C$. $\square$

T4 confirms the bound on 5138 EVs, and the reviewer's check on about 30000 EVs, both with zero violations wherever A1′ holds. The logged per-step quantity $\Delta t\sum_i(g(\tilde l_i)-g(p_i))^{+}$ therefore bounds the departure shortfall a priori, and is reported alongside it.

### 7.4 Station-level request pipeline: arrival cap and admission control (DD2, DD9, DD20, DD25, DD31)

The arrival cap (DD2) and admission control are part of the **station operator's request setting, not of the controller** (DD31). For every test replay they are computed **once, before any controller runs**, and the resulting admitted requests $E_i^{\mathrm{req}}$ are shared by every method (SA-Attn, penalty and Lagrangian baselines, heuristics, MPC, EV-GNN) and by the oracles. Conformal variants use their own table, because their departure input is $\hat d^{\mathrm{lo}}$ instead of $d$.

**Pre-pass (per replay; `admission_prepass.py`).** Arrivals are processed in time order (ties by charger index). For the arriving EV $j$ at step $t_0$:
1. **Arrival state.** The EV enters with its nominal arrival state: $E_j^{\mathrm{arr}}$, $E_j^{\mathrm{des}}$, $d_j$ (or $\hat d_j^{\mathrm{lo}}$) and its EV parameters. All of these are exogenous, i.e. identical in every method.
2. **DD2 cap.** `cap_request` yields $E_j^{\mathrm{req}}$ and its certified table (Sec. 3.4).
3. **Admission LP.** The LP below is set up over the EVs admitted earlier and still connected at $t_0$, plus $j$.
   - The energies $E_i(t_0)$ of the earlier EVs are **read from a nominal plan**, not from any controller's trajectory.
   - After each admission decision, an elastic version of the same LP is solved and its trajectories $E_i(t)$ are stored as the nominal plan. Its slacks on targets and on the aggregate band are weighted $10^{5}$, and its objective $\max\sum_i\sum_k\min\{E_i(k),E_i^{\mathrm{tgt}}\}$ means "reach targets as early as possible".
   - The capacity profile uses the realised values at $t_0$, the forecasts made at $t_0$ (a function of (replay, $t$) only) and the last forecast held afterwards.
4. **Decision.** Targets follow DD25. The largest feasible request $y_{\max}$ is found by one LP (plus the exact 12-step bisection predicate). Earlier admitted requests are never reduced. If the LP is infeasible even at the lower end, $E_j^{\mathrm{req}}:=\min(E_j^{\mathrm{arr}},E_j^{\mathrm{req}})$ ("not restored").
5. **Re-certification.** The reduced request passes through `cap_request` again.

The admission LP at $t_0$ is:
$$
\begin{aligned}
&E_i(t+1)=E_i(t)+\Delta t\,p_i(t),\quad E_i^{\min}\le E_i(t+1)\le E_i^{\mathrm{cap}},\quad E_i(d_i+1)\ge E_i^{\mathrm{tgt}},\\
&-P_i^{\mathrm{dis}}\le p_i(t)\le\bar P_i^{\mathrm{nom}},\quad p_i(t)\le\varrho_i^{\mathrm{in}}\,(E_i^{\mathrm{cap}}-E_i(t))\ (t>t_0),\quad p_i(t_0)\in[p_i^{\min}(E_i(t_0)),\bar P_i(E_i(t_0))],\\
&-\hat C_{\mathrm{rev}}(t)\le\sum_ip_i(t)\le\hat C(t)-n(t_0)\varepsilon_q.
\end{aligned}
$$
- $\varrho_i^{\mathrm{in}}$ is the largest slope with $\bar P_i(E)\ge\min\{\bar P_i^{\mathrm{nom}},\varrho_i^{\mathrm{in}}(E_i^{\mathrm{cap}}-E)\}$ for all $E$. It is an **inner** taper approximation (grid of 2000 points, 1 % safety factor), so LP-feasible schedules are physically realisable.
- **DD25 targets:** $E_j^{\mathrm{tgt}}=E_j^{\mathrm{req}}$ for the new EV. For earlier EVs, $E_i^{\mathrm{tgt}}=\min\{E_i^{\mathrm{req}},\Psi_i^{\mathrm{LP}}(E_i(t_0))-10^{-6}\}$, where $\Psi_i^{\mathrm{LP}}$ is the maximum departure energy reachable in the LP model (exact first step, inner minorant afterwards).
- **Why DD25 is needed:** with $E^{\mathrm{req}}=E^{\mathrm{cap}}$, the linear minorant shrinks the gap $E^{\mathrm{cap}}-E$ by at most the factor $(1-\varrho^{\mathrm{in}}\Delta t)$ per step and never closes it. Literal targets would therefore make the LP infeasible whenever a full-target EV is connected, and every new request would collapse to $E^{\mathrm{arr}}$ (IMPLEMENTATION_NOTES I-3).

**Causality.** Decision $j$ uses only information available at $t_0$: arrivals so far with their nominal states, measured and forecast capacity. The pipeline is therefore implementable online by a station operator. It does not depend on any controller.

**A4 is preserved for every method.** The certificate $E_i^{\mathrm{arr}}\ge\hat B_i(\tau_i(t_i^{\mathrm{arr}});E_i^{\mathrm{req}})$ involves only the EV's own exogenous arrival state, its admitted request and its own threshold table. `cap_request` re-certifies it after any reduction, exactly as before. The nominal plan influences only *which* value is offered to `cap_request`, not the certificate. So A4, and with it Theorem 1′(c), holds for every controller that uses the safety layer, with the same requirement for all methods.

**What admission control guarantees:**
1. A4 for every EV, for every method.
2. Identical admitted requests for all methods and the oracles on a replay, so shortfall and profit are compared against the same requirement. Admission reductions are reported separately and are identical across methods.
3. Requests are never reduced after admission.
4. If the LP is feasible: there exists an open-loop schedule that meets the new EV's admitted request and the LP-reachable targets of the earlier EVs, **starting from the nominal-plan energies**. The schedule is physically realisable (inner taper) and respects the transformer limit under the **forecast** capacity, provided no further EV arrives.

**What admission control does not guarantee:**
1. A5 for any particular controller. A controller's realised energies may lie below the nominal plan, so it may need more capacity later than planned.
2. Feasibility under the realised capacity: forecasts are noisy.
3. Feasibility after future arrivals.
4. That earlier EVs reach their full admitted request when it exceeds $\Psi^{\mathrm{LP}}$ (only up to the LP-model gap).

Admission control therefore does not enter Theorem 1′. An evaluation pass without admission control (`_noadm`) is run for SA-Attn and Attention+penalty.

### 7.5 What is counted in the experiments

Per episode and per method:
1. **Headline guarantee metrics (DD30):**
   - fraction of EVs whose stay contains no infeasible step ("A5-complete EVs");
   - fraction of steps in fallback, by case FB-0/1/2/1*;
   - both versus $N$, profile and $\rho_{\mathrm{tr}}$.
2. **A5 slack:** the minimum over steps of $\tilde C-\sum l$; binding rate of the forward limit.
3. **EVs touched by an infeasible step:** those with $p_i<l_i$ at least once.
4. **Departure shortfall** $\Delta E^{\mathrm{sh}}$, against the **post-admission request** for all methods (main), and against the DD2-capped request and $E^{\mathrm{des}}$ (secondary). Also the number of EVs with $\Delta E^{\mathrm{sh}}>0.001$ kWh.
5. **The Lemma 3 deficit bound.**
6. **Admission:** number and kWh of reduced requests, identical across methods; "not restored" count.
7. **DD2 caps:** number capped by more than 0.01 kWh, and kWh.
8. **Transformer violation** in kWh (DD7 metric) and violating steps. These are 0 for every safety-layer method unless A6 fails.
9. **Guard counters:** non-finite raw actions, post-condition hand-overs to bisection.
10. **Economics:** profit, penalised profit $J^{\mathrm{pen}}$ (DD32), optimality gaps, cost of the guarantee.

## 8 Conformal lower bound on the departure step (DD12, DD23)

We use split (inductive) conformal prediction [papadopoulos2002inductive; vovk2022algorithmic; lei2018distributionfree; angelopoulos2023conformal], in its one-sided, quantile-regression-based form [romano2019conformalized]. Conformal bounds have been used as inputs to safe planning and control before [lindemann2023safe]. The policy and the safety layer no longer see $d_i$, and A2 is weakened to "$E_i^{\mathrm{req}}$ known at arrival".

The procedure is:
1. **Predictor.** Train $\hat q$, a lower-quantile regressor of $d_i$ (gradient boosting with quantile loss at level $\alpha$), on features $\mathbf w_i$: arrival step, weekday, $E_i^{\mathrm{arr}}$, $E_i^{\mathrm{des}}-E_i^{\mathrm{arr}}$, EV model. Use a separate training split of simulated sessions.
2. **Calibration (DD23).** Draw $n_{\mathrm{cal}}$ calibration sessions from independently seeded episodes of the **same scenario configuration as the test episodes**: same $N$, $\Delta t$, spawn multiplier, scenario type, transformer-tightness level and date range. Take one session per episode, chosen uniformly at random among that episode's EVs. Compute the scores $e_j=\hat q(\mathbf w_j)-d_j$.
3. **Quantile.** $\hat Q_{1-\alpha}$ is the $k$-th smallest score with $k=\lceil(n_{\mathrm{cal}}+1)(1-\alpha)\rceil$, or $+\infty$ if $k>n_{\mathrm{cal}}$.
4. **Bound.** For a test EV,
$$\hat d_i^{\mathrm{lo}}=\max\big\{t_i^{\mathrm{arr}},\ \lceil\hat q(\mathbf w_i)-\hat Q_{1-\alpha}\rceil\big\},\qquad \tau_i^{\mathrm{lo}}(t)=\max\{1,\ \hat d_i^{\mathrm{lo}}-t+1\}.$$
   The ceiling is valid because $d_i$ is an integer: $d\ge y\iff d\ge\lceil y\rceil$. The maximum with $t_i^{\mathrm{arr}}$ is valid because $d_i\ge t_i^{\mathrm{arr}}$ surely.
5. **Use.** The layer uses $\tau_i^{\mathrm{lo}}$ in place of $\tau_i$ in (S11), in the DD2 cap and in the features. Departures are observed when they happen: the EV is removed from $\mathcal N$.

**SA-Attn-CP is not zero-shot.** It needs calibration data from the target configuration (5000 simulated sessions). The policy itself is not retrained, but the bound is calibrated per configuration. Conformal variants also have their own admission table (Sec. 7.4).

**Why calibration must match the scenario.** The law of "a uniformly chosen EV of an episode" depends on the scenario, including $N$: choosing uniformly among all sessions is size-biased towards chargers with many sessions. Calibration is therefore done separately for every test configuration, in particular for every $N$ of the zero-shot study. One calibration set of 5000 sessions per configuration is cheap, because sessions come from the simulator's arrival model only.

**Proposition 2 (conformal departure guarantee).** Let the calibration sessions and the test session be exchangeable. This holds for the sampling scheme above when the test EV is a uniformly chosen EV of an independent test episode of the same configuration and $\hat q$ is fitted on separate data. Then:
(i) $\mathbb P(d_i\ge\hat d_i^{\mathrm{lo}})\ge1-\alpha$;
(ii) under A1, A3 and A4 (with $\tau^{\mathrm{lo}}$), on the event $\{d_i\ge\hat d_i^{\mathrm{lo}}\}\cap\{p_i(t)\ge l_i(t)\ \forall t\in[t_i^{\mathrm{arr}},d_i]\}$, the departure requirement is met. Hence
$$\mathbb P\big(E_i(d_i+1)\ge E_i^{\mathrm{req}}\big)\ \ge\ 1-\alpha-\mathbb P\big(\exists t\le d_i:\ p_i(t)<l_i(t)\big),$$
and $\ge1-\alpha$ if A5 holds **almost surely** at every step of the stay.

*Proof.*
(i) By exchangeability, the rank of $e_{n+1}=\hat q(\mathbf w_i)-d_i$ among $e_1,\dots,e_{n+1}$ is uniform up to ties, so $\mathbb P(e_{n+1}\le\hat Q_{1-\alpha})\ge k/(n_{\mathrm{cal}}+1)\ge1-\alpha$ [vovk2022algorithmic; lei2018distributionfree]. Then $e_{n+1}\le\hat Q_{1-\alpha}\iff d_i\ge\hat q(\mathbf w_i)-\hat Q_{1-\alpha}\iff d_i\ge\lceil\hat q-\hat Q_{1-\alpha}\rceil$, and the maximum with $t_i^{\mathrm{arr}}$ changes nothing.
(ii) On the event, the layer's horizon $\hat d^{\mathrm{lo}}$ does not exceed the true one.
- Lemma 1 applied with $\tau^{\mathrm{lo}}$ up to $t=\hat d^{\mathrm{lo}}$ gives $E_i(\hat d^{\mathrm{lo}}+1)\ge\hat B(0)$.
- For $\hat d^{\mathrm{lo}}<t\le d_i$, we have $\tau^{\mathrm{lo}}=1$. By the third condition in (S10), $\hat B(1)\le\max\{\hat B(0),E^{\min}\}$; since $E_i(t)\ge\hat B(0)$ and $E_i(t)\ge E^{\min}$, it follows that $E_i(t)\ge\hat B(1)$. Lemma 1 with $\tau=1$ then gives $E_i(t+1)\ge\hat B(0)\ge E^{\mathrm{req}}$ again.
- Hence $E_i(d_i+1)\ge E^{\mathrm{req}}$. The probability bound is the union bound.

The last statement needs A5 almost surely. The bound does **not** hold conditionally on the event that A5 held, because that event may be correlated with $d_i$. $\square$

**What exactly is guaranteed.**
- The probability in Proposition 2 is over calibration data and a uniformly chosen EV of a test episode. It equals the **expected within-episode fraction** of covered EVs (and of EVs meeting $E^{\mathrm{req}}$).
- It is neither a per-EV guarantee nor conditional on $\mathbf w_i$. The reviewer found sub-group coverage of 0.70-0.89 for nominal 0.8-0.95.
- It is also not the pooled fraction over all test EVs. We therefore report the episode-averaged fraction, with the pooled fraction as secondary.

**Remarks.**
- If the EV leaves before $\hat d_i^{\mathrm{lo}}$, the requirement may be missed. That is the $\alpha$-event, and its shortfall is reported.
- After $\hat d_i^{\mathrm{lo}}$, the EV is still controlled until it actually leaves. It may V2G-discharge but never below $E^{\mathrm{req}}$.
- The price of conservatism appears as larger $l_i$, more DD2 caps and lower profit. It is reported against $\alpha\in\{0.01,0.05,0.1,0.2\}$.

---

## 9 Gradient handling (DD11, DD21)

**Main algorithm: the projection is part of the environment.**
- The policy is a Gaussian over raw actions $a$.
- PPO [schulman2017proximal] uses the likelihood ratio of the **raw sample** $a$.
- The map $a\mapsto p=\Pi_t(P^{\mathrm{ref}}a)$, together with the fallback and the actuation, is part of the transition kernel: $P(s'\mid s,a)=P_{\mathrm{env}}(s'\mid s,\Pi_t(P^{\mathrm{ref}}a))$.

The return of this MDP is the return of the projected actions. The **score-function (vanilla policy-gradient) estimator**, and equivalently the gradient of the PPO surrogate at $\theta=\theta_{\mathrm{old}}$, is unbiased for it. This requires only a density smooth in $\theta$ and bounded rewards; the rewards are bounded because $\Pi$ bounds $p$ even for unbounded $a$. The clipped PPO surrogate and GAE [schulman2016highdimensional] are biased estimators anyway. "Unbiased" below always refers to the score-function gradient.

For stochastic policies, applying the projection in the environment or in the policy is equivalent in this sense [markgraf2025safe]. This avoids two documented pitfalls:
1. the bias of evaluating the score of the projected action under the unprojected density [gros2020safe];
2. the zero-gradient problem of back-propagating through an active projection [kasaura2023benchmarking].

**Variance reduction (unbiased for the score-function gradient).**
- **Tokens with no influence.** If $u_i(t)-l_i(t)\le\epsilon_{\mathrm{num}}$, then $p_i$ does not depend on $a$ beyond $\epsilon_{\mathrm{num}}$, and neither does any other $p_j$: the term $\operatorname{clip}(x_i-\lambda,l_i,u_i)$ is constant in $\lambda$. The set of such tokens depends only on the state. Because the Gaussian factorises, dropping them from the log-probability and the entropy removes a zero-mean term that is uncorrelated with the kept terms. Using the threshold $\epsilon_{\mathrm{num}}$ instead of exact equality introduces a bias proportional to an influence of at most $10^{-9}$ kW, which is negligible.
- **Fallback steps.** On fallback steps $p$ is a function of the state only, so these transitions are masked out of the actor loss and kept for the critic.
- **Normalisation.** To keep the masked estimator literally unbiased, the actor loss is normalised by the **fixed** minibatch size $n_{\mathrm{mb}}$, i.e. $\sum_{\text{kept}}(\cdot)/n_{\mathrm{mb}}$, not by the random number of kept transitions.
- **Effect on clipping.** Masking also changes which samples PPO clips, which is beneficial.
- **Evidence.** The reviewer measured variance ratios of 0.19 (token mask) and 0.83 (fallback mask) at unchanged expectation.

**Drift regulariser.**
- Many raw actions alias to the same $p$. The policy mean can therefore drift far outside $\mathcal K$ without any change in the return, which keeps the Gaussian in its saturated tails. We add to the actor loss
$$\mathcal L_{\mathrm{reg}}(\theta)=\beta\ \mathbb E_{s}\Big[\frac1{n(t)}\sum_{i\in\mathcal N(t)}\Big(\frac{\bar x_i-\Pi_t(\bar x)_i}{P^{\mathrm{ref}}}\Big)^{2}\Big],\qquad\bar x=P^{\mathrm{ref}}\mu_\theta(s),$$
  with $\beta=0.01$ by default and $\beta=0$ as an ablation. It is evaluated on feasible transitions with the stored $(l,u,\tilde C,C_{\mathrm{rev}})$.
- By the Moreau identity, $\nabla_x\tfrac12\|x-\Pi(x)\|^2=x-\Pi(x)$ everywhere, and the function is $C^1$. Implementing the loss with $\Pi_t(\bar x)$ under stop-gradient therefore gives exactly the true gradient, up to the constant factor $2\beta/(nP_{\mathrm{ref}}^2)$. No Jacobian of $\Pi$ is required (T3: $3\times10^{-7}$).
- Pre/post-projection distance terms are an established remedy [chen2021enforcing; markgraf2025safe].

**Alternatives (not used in the main algorithm).**
- **True (sub)gradient.** For a deterministic actor, $\nabla_\theta Q(s,\Pi(x_\theta(s)))=J^\top\nabla_pQ\cdot\partial x_\theta/\partial\theta$ with $J$ from (S16). This suffers from the zero-gradient problem [kasaura2023benchmarking].
- **Straight-through ($J\approx I$).** This avoids zero gradients but is biased [gros2020safe].

---

## 10 Assumptions and modelling choices

| Label | Statement | Where used | How it is checked or reported |
|---|---|---|---|
| C0 | $P_i^{\mathrm{ev}}\le P^{\mathrm{cs}}$ | (S7), Lemmas E1 and E2 | True for all 8 models; config check |
| A1 | $\varphi_i$ continuous and non-decreasing (sufficient: $\eta^{\mathrm c}\Delta tL\le1$) | Lemma 0, (S8), Lemma 1, Theorem 1 | Proven for EV2Gym's taper (Lemma E1); numerically on 768 + 300 configurations; necessary (STEEP test, Sec. 3.1) |
| A1′ | $\varphi_i$ 1-Lipschitz ($\bar P_i$ non-increasing) | Lemma 3; numerical room in the DD2 cap | Proven (Lemma E1); excess $1.4\times10^{-14}$ kWh |
| A2 | $d_i$ and $E_i^{\mathrm{req}}$ known at arrival | Theorem 1 | Exact in EV2Gym; relaxed for $d_i$ by Proposition 2 |
| A3 | Realised energy $\ge E+\Delta tg(p)$ and $\le E^{\mathrm{cap}}$; realised power in $[p,p+\varepsilon_q]$ | Lemma 1, Theorem 1(a) | Lemmas E2 and E3 under Corollary 2's configuration; logged as $\max_t(\sum\tilde p-\sum p)$ and $\min_t(E(t+1)-E(t)-\Delta tp)$ |
| A4 | $E_i^{\mathrm{arr}}\ge\hat B_i(\tau_i(t_i^{\mathrm{arr}}))$ | Theorem 1 base, Theorem 1′(c) | By construction (`cap_request` in the station-level pipeline, also after admission, DD31); caps reported with threshold 0.01 kWh |
| A6 | $C(t)\ge N\varepsilon_q$ and $C_{\mathrm{rev}}(t)\ge0$ (exogenous headroom, policy-independent) | Theorem 1′(a),(b) | Holds under DD24 scaling except with probability $<2\times10^{-9}$ per episode (Corollary 2, item 8); checked online, failures logged as FB-0/FB-2 |
| A5 | $\sum l_i\le\tilde C$, $\sum u_i\ge-C_{\mathrm{rev}}$, $\tilde C\ge-C_{\mathrm{rev}}$ at every closed-loop step | Theorem 1, Theorem 1′(d) | Checked online at every step; infeasible steps, FB-0/1/2/1*, A5 slack and the Lemma 3 deficit reported per $N$ and scenario |
| $\epsilon_{\mathrm{num}}$ | Single numerical tolerance $10^{-9}$ | Sec. 6.4 | Absorbed by the unused part of $\varepsilon_q$ and far below 0.001 kWh |
| DD1 | EV2Gym lossless accounting ($g=\mathrm{id}$), efficiencies as derating | All | Stated as a limitation |
| DD4 | Efficiency constant over current levels per EV model (own spec JSON) | Lemma E2, actuation | Spec file archived with the code |
| DD5 | One port per charger | Token = charger | Config check |
| DD6 | Safety layer uses realised $P^{\mathrm{inf}}(t)$ and $P^{\mathrm{pv}}(t)$ | $C(t),C_{\mathrm{rev}}(t)$ | Stated |
| DD7 | Symmetric transformer limit, own violation metric | (F8) | Sec. 7.5 |
| DD17 | `transition_soc_multiplier` $m=5$ | Lemma E2 | Config check; $\max_i\kappa_i$ logged |
| — | `empty_ports_at_end_of_simulation=True`; capacities are multiples of 0.01 kWh | Theorem 1(c) coverage; Lemma E3 | Config check; true for all 8 models |
