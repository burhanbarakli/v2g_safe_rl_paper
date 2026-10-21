# Proof check of the safety layer (Internal Review, A6)

Reviewer: A6 (independent; did not write the method). Date: 2026-10-22.
Files checked: `method/safety_layer.md` (the main object), with the definitions it uses in `formulation.md`, `nomenclature.md`, `algorithm.md`, `design_decisions.md` and `ev2gym_notes.md`. I also read `projection_ref.py` and the EV2Gym source at commit `55ef368` (`ev.py`, `ev_charger.py`, `ev2gym_env.py`, `utils.py`, `transformer.py`, `V2GProfitMax.yaml`).
Independent numerics: `review/proof_check_numeric.py` gives `review/proof_check_numeric_results.txt`. This is my own code; `projection_ref` is imported only for comparison.

## Verdict overview

| # | Item | Verdict |
|---|---|---|
| 1 | Projection: feasibility, KKT closed form, λ signs, uniqueness, algorithms, Jacobian | **Proofs CONFIRMED. Algorithm spec and reference code: ERROR (M1, numerical), plus minor fixes** |
| 2 | Reach operator R, A1, thresholds, bisection for $l_i$, EV2Gym taper (Lemmas E1 and E2) | CONFIRMED WITH MINOR FIX |
| 3 | Lemma 1, Theorem 1, time indexing, A3/A4, the $n\varepsilon_q$ tightening, dead-band, $E^{\min}$ floor | CONFIRMED WITH MINOR FIX |
| 4 | Fallback F0/F1/F2, Lemma 2, Lemma 3 | CONFIRMED WITH MINOR FIX |
| 5 | Conformal Proposition 2 | CONFIRMED WITH MINOR FIX |
| 6 | Gradient handling (raw-sample likelihood, masking $l=u$ tokens and fallback steps, Moreau regulariser) | CONFIRMED WITH MINOR FIX (wording) |
| 7 | Permutation equivariance (Lemma F1) | CONFIRMED |
| 8 | Notation, undefined symbols, units | CONFIRMED WITH MINOR FIX (several clashes and wrong cross-references) |

There is **one major issue (M1)**. It is not a flaw in any proof. It is a floating-point failure of the specified sorted-breakpoint algorithm, which `projection_ref.py` is said to implement and which a PyTorch port must reproduce. When it triggers, the commanded power is $p=u$, which can exceed the transformer limit by hundreds of kW.

I found no error in the mathematics of Lemma 0, Proposition 1, Lemmas 1-3, E1-E3, Theorem 1 or Proposition 2. The numerical evidence (summarised at the end) agrees with every claim, given the stated assumptions.

---

## Item 1: Projection (S 4)

**Checked and correct.**
- **Feasibility.** Prop. 1(i) states non-emptiness iff $l\le u$, $\sum l\le\tilde C$, $\sum u\ge-C_{\rm rev}$ and $-C_{\rm rev}\le\tilde C$. Both necessity and the sufficiency argument via $l+\theta(u-l)$ are correct.
- **KKT conditions.** The stationarity $p-x+(\mu^+-\mu^-)\mathbf 1-\nu^l+\nu^u=0$ is correct. So are the multiplier choices $\nu^l_i=l_i-x_i+\lambda^*$ and $\nu^u_i=x_i-\lambda^*-u_i$, and the signs: upper aggregate active gives $\lambda^*=\mu^+>0$; lower active gives $\lambda^*=-\mu^-<0$. Complementarity holds in all three cases.
- **Uniqueness.** $p^*$ is unique by strict convexity over a polyhedron, and all roots give the same $p^*$ by coordinate-wise monotonicity. $\lambda^*$ is unique whenever $\mathcal F\neq\emptyset$, because both one-sided slopes are $\le-|\mathcal F|<0$.
- **Jacobian (S16).** The formula is correct, including these degenerate cases:
  - $|\mathcal F|=1$ with the aggregate active gives $J=0$;
  - tokens with $l_i=u_i$ are never free;
  - an empty free set gives $J=0$;
  - at kinks, the neighbouring-region matrices belong to the Clarke Jacobian. The projector remark is correct.
- **Breakpoint algorithm in exact arithmetic.** Ties produce zero-length segments, and the crossing segment can never be a zero-length one, because $S_{j^*-1}>\theta\ge S_{j^*}$. So the "flat segment" branch is vacuous. In the upper case $j^*=1$ is impossible; in the lower case it means $\sum u=-C_{\rm rev}$ exactly, and $p=u$ is then correct. Padding with $x=l=u=0$ is harmless. Non-stable `torch.sort` is also harmless, since the order inside a tie group only affects zero-length segments.
- **Bisection.** The brackets are valid: $\max(x-l)>0$ in the upper case and $\min(x-u)<0$ in the lower case. With 200 fixed iterations, the method terminates.

Numerics (Part A): 6,800 instances of 12 kinds, including $l=u$ for some or all tokens, padding, ties, $N=1$, a hyperplane slab ($\tilde C=-C_{\rm rev}$), exactly tight A5, $|x|\sim10^6$, and infeasible rows.
- **Own solver against references.** The own solver (direct evaluation of $s$ at every breakpoint, no cumulative sums) agrees with:
  - `project_bisect` to $3\times10^{-15}$ (relative);
  - an LP optimality certificate (HiGHS: $\max_{q\in\mathcal K}(x-p)^\top(q-p)$) to $\le1.4\times10^{-14}$;
  - SLSQP to $3.8\times10^{-11}$ kW.
- **Jacobian.** (S16) matches central finite differences to $5.8\times10^{-9}$ on 398 points, 42 of them with $|\mathcal F|=1$.
- **`projection_ref.project`.** Wherever its output is feasible, it agrees with the own solver to $10^{-14}$.

### M1 (MAJOR, implementation and specification): the breakpoint search fails when A5 is tight to rounding

**Location.**
- `safety_layer.md` S 4.2 "Sorted breakpoints", step 5 ("Let $j^*$ be the first index with $S_{j^*}\le\theta$");
- `projection_ref.py:141-149` (`j = np.argmax(S <= target, 1)`);
- `algorithm.md` A1.2 (`j = torch.argmax((S <= tgt).to(int8))`).

**Failure.** $S_j$ is accumulated as $\sum u-\sum_k n^{\rm free}_k\Delta\beta_k$. In floating point, the last value $S_{2N}$ differs from $\sum l$ by rounding. Suppose A5 holds with a slack below that rounding (exactly tight, or tighter than about $10^{-12}\max|x|$). Then no index satisfies $S_j\le\theta$, `argmax` of an all-False mask returns 0, $\lambda=\beta_1=\min_i(x_i-u_i)$ is used, and the output is $p=u$.
- In the targeted test, 8,742 of 20,000 exactly tight rows return $p=u$, with $\sum p$ exceeding $\tilde C$ by up to 418 kW.
- In the random families, every infeasible output of `project` was $p=u$; in the targeted sweep, only rows with slack ≲ $10^{-12}\max|x|$ failed. The `general`, `eq_lu_some`, `padded` and `huge` families had 65, 13, 15 and 19 such rows. The generator clamps $\tilde C$ to $\sum l$ in a few percent of draws, so those are most likely tight rows; I did not check this row by row.
- The failure band grows with the raw-action magnitude. At slack $10^{-10}$ kW it fails 0/2996 times for $|x|\sim16$ but 1521/2992 for $|x|\sim10^7$. At slack $10^{-8}$ it fails 203/2993 for $|x|\sim10^7$. The raw action is deliberately unbounded, and S 9 itself discusses mean drift.

**Assessment.**
- In normal operation the event is rare, because the A5 slack is a continuous quantity.
- When it occurs, the layer silently commands maximum power at every charger, and Theorem 1(a) fails for the executed algorithm.
- `verify_projection.py` T1 never generated tight-A5 rows, so this was not caught.

**Correction.**
1. **Text.** In S 4.2 step 5, add: "$S_{2N}=\sum l\le\theta$ guarantees that $j^*$ exists. In floating point, set $S_{2N}:=\sum_i l_i$ (computed exactly as in the A5 check), and if no index satisfies $S_j\le\theta+\epsilon$, take $\lambda=\beta_{2N}$ (so $p=l$)."
2. **Code** (`project` and A1.2): `has = (S <= tgt+eps).any(-1); j = where(has, argmax(...), 2N-1)`, and overwrite `S[:, -1] = l.sum(-1)`.
3. **Post-condition.** After `clip(x-λ,l,u)`, assert $-C_{\rm rev}-10^{-9}\le\mathbf 1^\top p\le\tilde C+10^{-9}$. On failure, fall back to `project_bisect` with polish.
4. **Tests.** Add tight-A5 rows and $|x|\in[10^4,10^8]$ rows to T1.

### Minor issues (item 1)

- **m1.1** `projection_ref.project` has no feasibility or NaN guard. Infeasible rows are left to the caller, which is documented, but nothing asserts it. A NaN raw action (from a diverging network) propagates to $p$ and then into `math.ceil` in `actuation`. Theorem 1's "arbitrary $x\in\mathbb R^N$" excludes NaN/Inf.
  → Add `assert np.isfinite(x).all()` (or replace with 0) and an infeasibility assertion.
- **m1.2** `fallback()` in `projection_ref.py:212` sorts by laxity only (a stable argsort, so ties fall back to index order). S 7.2, A1.3 and DONE item 5 specify a secondary key, "larger $l_i-p_i^{\min}$ first".
  → Make code and text agree. This only matters at laxity ties.
- **m1.3** `algorithm.md` A1.1 computes `feasible = (sum l <= Ctil) and (sum u >= -Crev)` without the third A5 condition $\tilde C\ge-C_{\rm rev}$. That condition fails only if $2P_{\rm tr}<n\varepsilon_q$, i.e. $P_{\rm tr}<0.8$ kW at $N=200$, for example during a transformer capacity-drop event.
  → Add the condition. If it fails, both F-cases are ill-defined; prioritise the forward limit as in F1.

---

## Item 2: R, A1, thresholds, $l_i$, EV2Gym taper (S 2, S 3)

**Checked and correct.**
- **Composition of monotone maps.** If $\varphi$ is continuous and non-decreasing, so is $\Psi_k=\varphi^k=E+R(k,E)$ (Lemma 0(i)). $R(k,\cdot)$ itself is not monotone, and under A1′ it is non-increasing; the remark is correct. The recursion (S6) is correct.
- **Composed map for $l_i$.** $p\mapsto\Psi_{\tau-1}(E+\Delta t\,g(p))$ is non-decreasing because $g$ is strictly increasing. So the predicate of the direct bisection is monotone on $[p^{\min},p^{\max}]$, and it is true at $p^{\max}$ iff $E\ge B(\tau)$. The bracket and the "upper end is conservative" claim are correct.
- **Capacity capping.** $\varphi=\min\{\varphi^{\rm tap},E^{\rm cap}\}$ preserves continuity and monotonicity.
- **Thresholds.** The minimum in (S9) exists by continuity, and the threshold characterisation $\{\Psi_k\ge E^{\rm req}\}=[B(k),E^{\rm cap}]$ is correct by induction.
- **Bisection for $B(k)$.** The bracket $[E^{\min},B(k-1)]$ has the predicate false at the lower end (checked) and true at the upper end ($\varphi(E)\ge E$). Under A1 the feasible set is an up-set, and the returned upper end is certified, giving (S10).
- **Equivalence (S11).** The equivalence with the defining property is correct. So is the closed form (S12) with general $\eta$, including the domination of the $E^{\min}$ branch by $p^{\min}$.
- **A1 sufficient condition.** $\eta^{\rm c}\Delta tL\le1$ is a correct sufficient condition.

**EV2Gym taper, re-derived from `ev.py:240-355`.** At full current the pilot is $\min(\zeta P_{cs},\zeta P_{ev})\Delta t/C$ and is capped at `max_dsoc` $=\zeta P_{ev}\Delta t/C$. If $P_{ev}\le P_{cs}$, the pilot equals `max_dsoc`, the transition is not shifted, and the three branches of (S5) follow, with the final `min` against `dsoc_limit`. I re-derived (S5) and checked it against `EV._charge` on 300 random configurations ($m\in\{0.5,1,2,5,10\}$, $\Delta t\in\{5,15,30,60\}$ min, $\zeta\in[0.85,1]$, $s^{\rm tr}\in[0.7,0.9]\cup\{1\}$). Results:
- (S5) matches `EV._charge` to $1.4\times10^{-14}$ kWh;
- A1 holds: the minimum grid increment is 0;
- A1′ holds: the excess is $2\times10^{-14}$.

Lemma E1's proof (concavity of $h$, slopes in $[0,1]$) is correct, and it holds for $m<1$ as well.

**Lemma E2.** I verified the case analysis algebraically:
- $w=x(1-s^{\rm tr})/\delta$;
- the middle-case identity $\hat\varphi_x-s-x=w(1-e^{-mz}-z)$;
- (b) reduces to $\kappa/(e^\kappa-1)\ge m(1-z_0)$, with equality at $\kappa=mz_0$, because $e^{mz_0}=1/(1-z_0)$.

The condition is also numerically sharp in the right direction (Part C):
- exact delivery at $\kappa=2.39$, $2.66$, $3.83$ and $4.78$ (all $<\kappa^*=4.965$), with error $7\times10^{-15}$;
- under-delivery of 0.026 kWh at $\kappa=9.57$ ($\Delta t=60$ min).

**A1 is necessary, not just convenient.** I ran a closed loop with a taper that drops faster than $1/\Delta t$ (Part B, "STEEP"). There, Lemma 1(a) fails: $l>u$ for EVs with $E\ge\hat B(\tau)$ (92 and 96 EV-steps). Among EVs whose whole stay had A5 at every step, 65 and 71 missed $E^{\rm req}$, by up to 1.78 kWh. The bisection table still satisfies (S10) in that case. **Certification (S10) alone does not protect against an A1 violation.** The paper should say so next to the statement "A1 enters exactly once".

### Minor issues (item 2)

- **m2.1** (S 2.3, (S5), Lemmas E1 and E2): the derivation silently assumes $P_{ev}\le P_{cs}$. For $P_{ev}>P_{cs}$, EV2Gym shifts the full-current transition to $s_x=1-\rho(1-s^{\rm tr})$ with $\rho=P_{cs}/P_{ev}$, and the taper exponent uses $\zeta P_{ev}$. (S5) is then wrong by 0.72 kWh in my test ($P_{ev}=40$ kW).
  → State "$P^{\rm ev}_i\le P^{\rm cs}$ (true for all 8 models)" as a condition of (S5), E1 and E2, and add it to Corollary 2.
- **m2.2** (S 2.1, last sentence; S 2.3 bullets): "drops faster than $1/\Delta t$" should read $1/(\eta^{\rm c}\Delta t)$. "$\Delta t\cdot L$" in S 2.3 is $\eta^{\rm c}\Delta tL$ with $\eta=1$.
- **m2.3** (S 3.4, "Under A1′ … the first candidate is certified in exact arithmetic"): in exact arithmetic with the exact table, A1 alone gives $E^{\rm arr}\ge B(\tau;\Psi_\tau(E^{\rm arr})-10^{-6})$. A1′ is what gives the room $B(\tau;y)\le E^{\rm arr}-10^{-6}$ that absorbs bisection error.
  → Reword along these lines.
- **m2.4** (S 7.4, `algorithm.md` A3 step 1c): after admission control lowers $E^{\rm req}_j$, the table is recomputed with `thresholds()` but A4 ($E^{\rm arr}\ge\hat B(K)$) is not re-checked. Because $B(K;y)$ is monotone in $y$, A4 holds in exact arithmetic. Numerically it can fail by the bisection tolerance when $E^{\rm arr}$ sits on the threshold.
  → Re-certify with the `cap_request` loop, i.e. lower further until $E^{\rm arr}\ge\hat B(K)$.

---

## Item 3: Lemma 1, Theorem 1, A3/A4, actuation margin (S 5, S 6)

**Lemma 1.**
- (a) $l\le u$ is $\hat B(\tau-1)\le\varphi(E)$, which follows from A1 and (S10).
- (b) $g(p)\ge y$ and A3 give $E(t+1)\ge\hat B(\tau-1)$.

Both are correct.

**Time indexing (verified in EV2Gym).**
- `ev2gym_env.step` calls `cs.step` (charging) first. `is_departing(cs.current_step)` then removes EVs with `current_step >= time_of_departure` after their charging in that step. So an EV acts at $t=d_i$ and the checked quantity is $E(d_i+1)$, and the EV2Gym satisfaction metric is computed on the same post-action capacity.
- Spawning happens at the end of step $t_{\rm arr}-1$ (`time_of_arrival == current_step+1`), so the first action is at $t_{\rm arr}$ and $E(t_{\rm arr})=E^{\rm arr}$.
- Hence $\tau=d-t+1$, the base case through A4, and the departure case $\tau=1\Rightarrow E(d+1)\ge\hat B(0)\ge E^{\rm req}$ are correct.
- Coverage: `utils.py:254` refuses to spawn if $\text{stay}+\text{step}+4\ge T$, so $d=\lfloor\text{stay}+\text{step}+3\rfloor\le T-2$.

**Theorem 1 induction.** The base, arrivals (A4), the step and departures are correct. Parts (b) and (c) hold for every raw action in $\mathbb R^N$, provided A5 holds along the closed loop. The statement is correctly conditional; A5 is itself trajectory-dependent, as S 7.1 says.

**A3 and the margin.** I redid Lemma E3 from `ev_charger.py:154-197` and `ev.py:146-189`:
- **Positive case.** The rounded command lies in $[p,p+\zeta P_{cs}10^{-5}]\subset[p,u)$ and is exactly delivered (E2).
- **Saturated case.** $\tilde p=u\le p+10^{-3}$.
- **Discharge.** The dead-band applies iff $|a|\cdot32<0.01$ A. Then $|p|<\zeta^{\rm d}(0.006928+2.2\times10^{-4})\le0.00715$ kW and $\tilde p=0$. Otherwise the clip to $P^{\rm ev,dis}$ and the $E^{\min}$ floor are inactive because $p\ge p^{\min}$.

So $0\le\tilde p-p<0.008$ kW per occupied charger. The tightening is applied to the **commanded** sum, and the theorem's claim is about the **realised** sum: $\sum\tilde p\le\sum p+n\varepsilon_q\le\tilde C+n\varepsilon_q=C$, while the reverse side is safe because $\tilde p\ge p$. This placement is correct and sufficient.
- **Necessity.** Without the tightening (Part B, "NO tightening", raw actions $+10^4$), 3,460 A5 steps had a realised violation of up to 0.0073 kW, and every F1 step violated the limit.
- **Dead-band and floor.** The dead-band only raises energy and power, so it is covered on the forward side and harmless on the reverse side. The $E^{\min}$ floor never binds because $p\ge p^{\min}$; at float level it reproduces the same energy.
- **The 0.01 kWh ceil** only adds energy. For all six capacities, `ceil(E_cap*100)/100 == E_cap` in IEEE double (checked), so $E\le E^{\rm cap}$ survives the ceil.
- **Other EV2Gym dead-bands.** These are inactive in the configuration: `min_ac_charge_power`/`min_discharge_power` default to 0 for heterogeneous specs, and `min_charge_current = min_discharge_current = 0` in `V2GProfitMax.yaml`.

Numerics (Part B): 9 configurations × 120 episodes, over 100,000 steps and about 30,000 EVs. The configurations cover the EV2Gym taper, lossy $\eta\ne1$ (with and without taper), exact and A3-noisy actuation (rounding, dead-band, ceil), 5-min steps and conformal $\tau^{\rm lo}$. Raw actions were Gaussian (σ = 16.5 kW), $\pm10^4$, mixed and Cauchy.

In every configuration that satisfies A1:
- 0 realised transformer violations on A5 steps;
- 0 commanded powers outside $[p^{\min},u]$;
- 0 departure misses among EVs whose stay had only A5 steps (1,052-1,743 per configuration);
- 0 misses among EVs with $p\ge l$ throughout (Corollary 1);
- 0 broken invariants.

### Minor issues (item 3)

- **m3.1** (Corollary 1 proof, "On fallback steps, $p_i\le l_i$"): this is false on F2 steps, where $p=u\ge l$. The conclusion survives.
  → Replace with "fallback outputs satisfy $p^{\min}\le p\le u$, so $p_i\ge l_i$ implies $p_i\in[l_i,u_i]$".
- **m3.2** Theorem 1 could be stated more strongly, at no cost:
  - (b) holds at every step, fallback steps included;
  - (a) also holds on F1 steps (Lemma 2(a)), so the transformer limit can fail only on F0/F2 steps.
  → Strengthen Corollary 1's last bullet.
- **m3.3** (Corollary 2): the configuration list is incomplete for A3. Add:
  - $P^{\rm ev}\le P^{\rm cs}$ (m2.1);
  - EV `min_ac_charge_power = min_discharge_power = 0`;
  - charger `min_charge_current = min_discharge_current = 0`;
  - 3-phase EVs and chargers;
  - `v2g_enabled: True`;
  - an efficiency that is constant over current levels.

  On the last point, EV2Gym's heterogeneous loader (`utils.py:270-296`) either builds a current-dependent dict from `3ph_ch_efficiency` or, if that key is absent, draws a **random scalar** $\zeta\in[0.95,1]$ per EV. So DD4's "scalar per model" must be realised as a spec JSON whose efficiency table is constant across current levels. The layer must read $\zeta$ per EV.
- **m3.4** (Lemma E3, "rounding is always upwards"; `actuation()` uses `ceil(·-1e-9)`): rounding is upward only to within about $10^{-14}$ kW. `check_A1_results.txt` also shows a realised energy $8.9\times10^{-16}$ kWh below $E+\Delta tp$. `bounds()` absorbs this with a $10^{-12}$ tolerance, while `algorithm.md` A1.1 uses no tolerance, which is inconsistent.
  → State once that all energy statements hold up to $\epsilon_{\rm num}\approx10^{-9}$ kWh. Alternatively, certify with $\hat B(0)=E^{\rm req}+10^{-9}$ and keep a single tolerance.

---

## Item 4: Fallback and shortfall (S 7.2, S 7.3)

**Exclusivity.** F1 and F2 are exclusive, because $\sum u<-C_{\rm rev}\le\tilde C<\sum l$ would contradict $\sum l\le\sum u$.

**Lemma 2(a).** If $b\ge0$, then $\sum(l-p^{\min})>b$, so the budget is used up and $\mathbf 1^\top p=\tilde C\ge-C_{\rm rev}$, with $p^{\min}\le p\le l\le u$. The realised sum is therefore $\le C$. **F1 never violates the transformer limit**, with the A3 margin. Part B confirms this: 0 violations on more than 25,000 F1 steps (excluding the deliberate no-tightening run).

**Lemma 3.**
- The per-EV telescoping is correct in both cases. In case 2 it uses (S10), A1 and A1′: $\hat B(\tau-1)-\varphi(E)\le\varphi(\hat B(\tau))-\varphi(E)\le\hat B(\tau)-E$. It also uses $\varphi(E)=E+\Delta t\,g(u)$, which holds because $u\ge0$.
- The fleet bound is correct for $g=\mathrm{id}$. On F1 steps every occupied EV has $p\le l$, and $\tilde l=l$ also for infeasible EVs (where $l=u$), so $\sum_i(\tilde l_i-p_i)^+=\sum l-\tilde C$. On F2 steps $p=u$; on A5 steps $p\ge\tilde l$.
- Part B confirms it with lossy $\eta$ and the A3 noise: 0 per-EV violations and 0 fleet-bound violations in the configurations where A1′ holds. Under the A1-violating taper, it fails for 75-87 EVs, as expected.

### Minor issues (item 4)

- **m4.1** (Lemma 2(b)): "Every EV with $p_i=l_i$ remains individually feasible" holds only for EVs that were feasible at $t$. Infeasible EVs have $l=u$ and stay infeasible.
  → Add "that is individually feasible at $t$".
- **m4.2** (Lemma 2(c), "No EV action can avoid it"): this is true for commanded powers. Realised powers may exceed the commanded $p^{\min}$ by up to $\varepsilon_q$ each.
  → Add "up to $n\varepsilon_q$".
- **m4.3** (S 7.1, policy-independent sufficient condition): add $\tilde C(t)\ge-C_{\rm rev}(t)$. Also say that the condition gives A5 at every step by induction on I(t).

---

## Item 5: Conformal bound (S 8)

**Quantile index.** The score is $S=\hat q-d$, and a large score means "left earlier than predicted". The lower bound $d\ge\hat q-\hat Q$ therefore needs the upper $(1-\alpha)$ score quantile: the $\lceil(n+1)(1-\alpha)\rceil$-th smallest score, with $+\infty$ when $k>n$. This is the correct index. $\mathbb P(S_{n+1}\le S_{(k)})\ge k/(n+1)\ge1-\alpha$ holds under exchangeability, and ties only help. The ceiling step is valid for integer $d$.

**Proposition 2(ii).** The argument with $\tau^{\rm lo}$ is correct (Lemma 1 up to $\hat d^{\rm lo}$, then $\tau^{\rm lo}=1$), and so is the union bound.

Numerics (Part E, 2,000 calibration repetitions with $n_{\rm cal}=300$ and episode-coupled sessions): marginal coverage is 0.952, 0.891 and 0.806 at $\alpha=0.05$, 0.1 and 0.2, within Monte Carlo error. Coverage on a hard sub-group is only 0.887, 0.810 and 0.701, so the bound is marginal, not conditional. This agrees with the paper's remark.

### Minor issues (item 5)

- **m5.1** ("and $\ge1-\alpha$ when A5 holds at every step of the stay"): this holds if A5 holds almost surely. It does not hold conditionally on the event that A5 held, which may be correlated with $d_i$.
  → Reword.
- **m5.2** (exchangeability; `algorithm.md` A4): the law of "a uniformly chosen EV of an episode" depends on the scenario, **including $N$**. Choosing uniformly among all sessions size-biases towards ports with many sessions, through the random normaliser $\sum_pK_p$. The calibration set must therefore use the same $N$, $\Delta t$, spawn multiplier, scenario and date range as the test episodes. A4 does not fix $N$, while C3 evaluates zero-shot at $N\in\{5,\dots,200\}$.
  → Either calibrate per $N$, or sample "a uniformly random port with at least one session, then a uniformly random session of that port", which is $N$-free because ports are i.i.d. In both cases, state the scheme.
- **m5.3** (what is guaranteed): the guarantee is $\mathbb P(d\ge\hat d^{\rm lo})$ for a uniformly chosen EV. This equals the **expected within-episode fraction** of covered EVs. It is not a per-EV guarantee, not conditional on $\mathbf w$, and not the pooled fraction over all test EVs, which A4 reports.
  → Report the episode-averaged fraction, or note the difference.
- **m5.4** (proof of (ii), "$E_i(t)\ge\hat B(0)\ge\hat B(1)$"): this uses $\hat B(1)\le\max\{\hat B(0),E^{\min}\}$, which is not part of (S10). `thresholds()` satisfies it.
  → Add $\hat B(k)\le\max\{\hat B(k-1),E^{\min}\}$ to (S10).

---

## Item 6: Gradient handling (S 9, `algorithm.md` A2)

**Raw-sample likelihood, with Π in the environment.** Let $J(\theta)=\mathbb E_{\pi_\theta}[\sum_t\gamma^tr_t]$ in the MDP whose kernel is $P_{\rm env}(\cdot\mid s,\Pi_t(P^{\rm ref}a))$. This is exactly the return of the projected actions. The score-function gradient $\mathbb E[\sum_t\nabla\log\pi_\theta(a_t\mid o_t)G_t]$ is unbiased for it. The conditions are a Gaussian density smooth in θ and bounded rewards; the rewards are bounded **because** $\Pi$ bounds $p$ even for unbounded $a$. A measurable kernel suffices, and $\Pi$ is continuous. This also holds for observation-conditioned policies (POMDP) with Monte Carlo returns or an action-independent baseline.

**Masking tokens with $l_i=u_i$: unbiased.** Rigorous argument:
1. $p_i\equiv l_i$, and $p_j$ ($j\ne i$) does not depend on $a_i$: the term $\operatorname{clip}(x_i-\lambda,l_i,u_i)$ is constant in $\lambda$, and the fallback and actuation do not read $x$.
2. The next observation contains no raw action, so every future reward and state is a function of $(s_t,a_{t,-i},\text{future noise})$ only.
3. Because the Gaussian factorises, $a_{t,i}\perp(a_{t,-i},\text{future})\mid s_t$, and $\mathbb E[\nabla_\theta\log\pi_i(a_{t,i}\mid s_t)\mid s_t]=0$.

Hence the dropped term $\nabla\log\pi_i\cdot\hat A_t$ has zero mean. Its covariance with every kept term is also zero, so masking **strictly reduces** variance.

**Masking fallback steps: unbiased** by the same argument at the step level. On a fallback step $p$ is a function of the state only, and $\hat A_t$ (GAE) does not depend on $a_t$.

Numerics (Part D):
- **Bandit with an active aggregate.** Exact gradient (quadrature) 0.4531; full estimator 0.4589 ± 0.0152; masked estimator 0.4562 ± 0.0066; variance ratio 0.19.
- **Two-step problem with a fallback step.** Full 0.4605 ± 0.0167; masked 0.4589 ± 0.0152; variance ratio 0.83.

**Moreau regulariser.** $\nabla_x\tfrac12{\rm dist}^2(x,\mathcal K)=x-\Pi(x)$ holds for the fixed, state-dependent set. With stop-gradient on $\Pi(\bar x)$, the implemented loss has exactly this gradient, up to the constant factor $2\beta/(nP_{\rm ref}^2)$.

### Minor issues (item 6)

- **m6.1** "Unbiased" should be qualified. It refers to the score-function (vanilla PG) gradient, equivalently the PPO surrogate gradient at $\theta=\theta_{\rm old}$. The clipped PPO surrogate and GAE are biased anyway. Dropping zero-mean factors also changes which samples PPO clips, which is beneficial.
- **m6.2** `infl = occ & (u - l > 1e-9)` also masks tokens whose influence on $p$ is at most $10^{-9}$ kW. The resulting bias is negligible, but it should be stated.
- **m6.3** `mean_w[...]` divides by the random number of feasible transitions in the minibatch. That is a ratio estimator, and exact unbiasedness needs a fixed normaliser ($\sum_w(\cdot)/B$). The effect is negligible; use the fixed normaliser if "unbiased" is to be claimed literally.

---

## Item 7: Permutation equivariance (Lemma F1, formulation S 5)

**Proof steps.**
- The key-only mask satisfies $M\mapsto PMP^\top$.
- Row-wise softmax satisfies $\operatorname{softmax}(PXP^\top)=P\operatorname{softmax}(X)P^\top$, which gives $A(PH)=PA(H)$.
- Row-wise layers commute with $P$.
- Pooling is a sum over keys and therefore invariant.
- $\Pi$ is equivariant because the minimiser is unique.
- The fallback is equivariant up to the index tie-break, which the lemma states.
- Padding invariance holds because masked keys get exactly zero weight. A $-10^9$ logit in float32 also underflows to exactly 0.

**Verdict: CONFIRMED.**

One remark: (iv) is about the network for fixed token contents. The global features 35-36 ($n/N$ and $P_{\rm tr}^{\rm nom}/(NP^{\rm ref})$) do depend on $N$, and the "Consequence" paragraph says so correctly.

---

## Item 8: Notation, symbols, units

**Units** are consistent throughout: kW, kWh, h, steps; $\kappa,\delta,\zeta$ are dimensionless; $\rho$ is in 1/h; $L$ is in kW/kWh. Minor issues:

- **m8.1** Wrong cross-references in `nomenclature.md`:
  - $\mathcal F$, $D$ and $J$ are listed as defined in "S 4.4", which does not exist; they are in S 4.1 (Prop. 1);
  - $\lambda$ and $s(\lambda)$ are defined in S 4.1, not S 4.2;
  - $j$ (breakpoints) is in S 4.2, not S 4.3.
- **m8.2** `nomenclature.md` defines $\Delta E^{\rm def}_i(t)$, but Lemma 3 calls the same quantity $\Delta(t)$.
  → Use one symbol.
- **m8.3** Symbol clashes:
  - $\theta$: policy parameters, the aggregate target in Prop. 1(iii) and S 4.2, and the interpolation parameter in the proof of 1(i);
  - $s$: SoC, the function $s(\lambda)$, and the state in S 9;
  - $\delta$: the SoC step and the Kronecker $\delta_{ij}$;
  - $w$: $1-s_x$ in Lemma E2 and the conformal features $\mathbf w$;
  - **"F1"**: equation (F1), fallback case F1, Lemma F1 and Proposition F1;
  - **"A1"**: Assumption A1, algorithm section A1, and agent A1 in S 8.

  → Rename the fallback cases (for example FB-0/FB-1/FB-2), the algorithm sections (for example Alg. 1-6) and the target ($\vartheta$).
- **m8.4** Symbols used but missing from the nomenclature: $\varepsilon_{\rm sat}$, $L$ (A1 rate), $\hat\varphi$, $h$, $z$, $s_x$, $p^{\max}$, $\Psi$ (listed, but without $\Psi_0$), $\varepsilon_{\rm num}$ (if adopted, see m3.4), $\mathcal K$ (listed).
- **m8.5** The equation numbers are out of order ((S6)-(S7) before (S5); (S9) before (S8)). This is cosmetic.
- **m8.6** `algorithm.md` A1.1 computes the laxity as `(E-Bhat)/(Pnom*dt)`, without $\eta^{\rm c}$. This is correct for EV2Gym, but say so, or include $\eta^{\rm c}$ to match S 3.3.

---

## Summary of numerical evidence (`proof_check_numeric_results.txt`)

| Check | Result |
|---|---|
| Own projection vs LP certificate / SLSQP / `project_bisect` | ≤1.4e-14 (normalised) / 3.8e-11 kW / 3e-15 |
| `projection_ref.project` on feasible rows | agrees to 1e-14 where its output is in $\mathcal K$; **returns $p=u$ when A5 is tight to rounding (M1)** |
| Jacobian (S16) vs finite differences | 5.8e-9 (398 points, 42 with $|\mathcal F|=1$) |
| EV2Gym (S5), A1, A1′ (300 random configurations) | 1.4e-14 kWh; A1 and A1′ hold for $m\in\{0.5,\dots,10\}$, $\Delta t\le60$ min |
| Lemma E2 | exact for $\kappa\le4.78$; fails at $\kappa=9.57>\kappa^*$ |
| Closed loop with A1 (7 configurations) | 0 violations of Thm 1(a)-(c), Cor. 1, Lemma 2(a), Lemma 3 |
| Closed loop without the $n\varepsilon_q$ tightening | realised violations up to 0.0073 kW on A5 steps; all F1 steps violate |
| Closed loop with A1 violated | Lemma 1(a) fails; up to 1.78 kWh shortfall for EVs with A5 at every step |
| Gradient masking | unbiased; variance ×0.19 (token) and ×0.83 (fallback step) |
| Conformal | marginal coverage at nominal level; sub-group coverage below it |
