# Algorithms, hyperparameters, baselines and scenarios

Agent A2 (Method and Theory). Version 4, 2026-10-22, revised after the internal proof check and the implementation feedback (`code_repo/code/IMPLEMENTATION_NOTES.md`; change log in `DONE_method.md`). Symbols: `nomenclature.md`. Equations (F·): `formulation.md`; (S·): `safety_layer.md`. Design decisions DD1-DD29: `design_decisions.md`.

The NumPy reference `projection_ref.py` is the executable specification of Alg. 1 and Alg. 3. Any PyTorch [paszke2019pytorch] port must match it to $10^{-9}$ in float64 on the tests T1 and T1b of `verify_projection.py`, including the tight-A5 and large-magnitude families.

Conventions:
- One EV2Gym environment = one station with $N$ chargers, one port each (DD5).
- A **wrapper** around the env owns all safety-layer state (threshold tables, $E^{\mathrm{req}}$, $\rho_i$), computes observations, applies the safety layer, the fallback rule and the actuation map, and logs metrics. EV2Gym itself is not patched.
- The wrapper never reads `env.EVs_profiles`, which contains future arrivals.
- **Section names.** Algorithms are numbered Alg. 1-7. "A1"-"A5" always denote assumptions, never algorithm sections.
- `EPS_NUM` $=\epsilon_{\mathrm{num}}=10^{-9}$ is the single numerical tolerance (S 6.4).

---

## Alg. 1 Safety-layer forward pass (batched)

### Alg. 1.1 Per-step bounds (wrapper, NumPy, one env)

```
input : env at decision step t (= env.current_step), per-EV stored data {Bhat_i[0..K_i], E_req_i, d_i (or dlo_i), Pnom_i, Pdis_i, phi_i}
output: occ[N] (bool), pmin[N], l[N], u[N], chi[N], Ctil, Crev, feasible   (kW; empty chargers: 0)

for each charger i:
    ev = cs_i.evs_connected[0]
    if ev is None: occ[i]=False; pmin[i]=l[i]=u[i]=0; continue
    occ[i] = True
    E   = ev.current_capacity                              # kWh
    tau = d_i - t + 1          (conformal variant: max(1, dlo_i - t + 1))
    pmin[i] = -min(Pdis_i, max(0, E - E_min_i)/dt)          # (S3), g = id
    u[i]    = (phi_i(E) - E)/dt                             # (S4) with (S7)
    if E >= Bhat_i[tau] - EPS_NUM:                          # individually feasible (S9)
        l[i] = min(max(pmin[i], (Bhat_i[tau-1] - E)/dt), u[i])   # (S11); min(.,u) is a float guard only
    else:
        l[i] = u[i]                                         # infeasible: charge at max (S11)
    chi[i] = (E - Bhat_i[tau])/(eta_c_i*Pnom_i*dt)          # laxity, steps (eta_c_i = 1 in EV2Gym)
n    = sum(occ)
C    = P_tr[t] - P_inf[t] + P_pv[t]                         # (F8), realised values (DD6)
Crev = P_tr[t] + P_inf[t] - P_pv[t]
Ctil = C - n*eps_q                                          # eps_q = 0.008 kW (DD16, S 6.3)
feasible = (sum(l[occ]) <= Ctil) and (sum(u[occ]) >= -Crev) and (Ctil >= -Crev)   # A5, Prop. 1(i)
```

For the general model with $\eta\ne1$, replace $(\hat B-E)/\Delta t$ by $g^{-1}((\hat B-E)/\Delta t)$ and $p^{\min}$ by (S3). Each threshold table is computed **once at arrival** by `cap_request` (Alg. 3, step 1; DD18), so the per-step cost is $O(N)$.

### Alg. 1.2 Projection (PyTorch, batched over B rows, padded to $N_{\max}$; floating-point robust)

```
EPS_NUM = 1e-9

def project_batched(x, l, u, C, Crev, occ):
    # x,l,u: (B,N) float64 [kW]; C (= Ctil), Crev: (B,); occ: (B,N) bool. Precondition: feasible rows only.
    z    = torch.zeros_like(x)
    x    = torch.where(torch.isfinite(x), x, z)                         # NaN/Inf guard (count and log)
    x    = torch.where(occ, x, z); l = torch.where(occ, l, z); u = torch.where(occ, u, z)
    p0   = torch.maximum(torch.minimum(x, u), l)                        # clip(x, l, u)
    s0   = p0.sum(-1)
    up   = s0 > C
    lo   = s0 < -Crev
    tgt  = torch.where(up, C, -Crev)                                    # target vartheta
    lam_all = torch.cat([x - u, x - l], -1)                             # (B,2N) breakpoints
    ev   = torch.cat([torch.ones_like(x), -torch.ones_like(x)], -1)     # +1 enters free set, -1 leaves
    lam_all, idx = torch.sort(lam_all, -1)
    ev   = torch.gather(ev, -1, idx)
    nfree = torch.cumsum(ev, -1)
    su   = u.sum(-1, keepdim=True)
    S    = torch.cat([su, su - torch.cumsum(nfree[:, :-1] * (lam_all[:, 1:] - lam_all[:, :-1]), -1)], -1)
    S[:, -1] = l.sum(-1)                                                # exact end value (review M1)
    hit  = S <= tgt[:, None]
    has  = hit.any(-1)
    j    = torch.where(has, torch.argmax(hit.to(torch.int8), -1), torch.full_like(s0, 2*x.shape[-1]-1, dtype=torch.long))
    jm   = torch.clamp(j - 1, min=0)
    b1, b2 = lam_all.gather(-1, jm[:, None])[:, 0], lam_all.gather(-1, j[:, None])[:, 0]
    S1, S2 = S.gather(-1, jm[:, None])[:, 0],    S.gather(-1, j[:, None])[:, 0]
    den  = S1 - S2
    lam  = torch.where(den > 0, b1 + (S1 - tgt) * (b2 - b1) / den.clamp_min(1e-300), b2)
    lam  = torch.where(j == 0, b2, lam)
    pad  = 1.0 + 1e-6 * x.abs().amax(-1)
    lam  = torch.where(has, lam, lam_all[:, -1] + pad)                  # no hit (A5 tight): p = l exactly
    lam  = torch.where(up | lo, lam, torch.zeros_like(lam))
    p    = torch.maximum(torch.minimum(x - lam[:, None], u), l)
    sp   = p.sum(-1)
    bad  = (up | lo) & ((sp > C + EPS_NUM) | (sp < -Crev - EPS_NUM))    # post-condition
    if bad.any():
        p[bad] = bisect_safe(x[bad], l[bad], u[bad], C[bad], Crev[bad], up[bad])
    return torch.where(occ, p, z), lam

def bisect_safe(x, l, u, C, Crev, up, iters=200):
    # keeps the feasible end: upper rows s(hi) <= C, lower rows s(lo) >= -Crev (as evaluated)
    pad = 1.0 + 1e-6 * x.abs().amax(-1)
    lo  = torch.where(up, torch.zeros_like(C), (x - u).amin(-1) - pad)
    hi  = torch.where(up, (x - l).amax(-1) + pad, torch.zeros_like(C))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s   = torch.maximum(torch.minimum(x - mid[:, None], u), l).sum(-1)
        ok_hi = up & (s <= C);  ok_lo = (~up) & (s >= -Crev)
        hi = torch.where(ok_hi | ((~up) & ~ok_lo), mid, hi)
        lo = torch.where(ok_lo | (up & ~ok_hi), mid, lo)
    lam = torch.where(up, hi, lo)
    return torch.maximum(torch.minimum(x - lam[:, None], u), l)
```

Notes:
- `torch.argmax` returns the first maximal index.
- Padded entries have $x=l=u=0$ and do not change any segment slope.
- Run under `torch.no_grad()`: the main algorithm does not differentiate through $\Pi$ (DD11).
- The optional Jacobian (S16) is `D - outer(d,d)/d.sum()` with `d = occ & (x-lam > l) & (x-lam < u)`, used only if the aggregate constraint is active.
- Log the number of NaN/Inf replacements and of post-condition hand-overs to `bisect_safe`. Both should be 0 in normal operation.

### Alg. 1.3 Fallback (wrapper; only if `feasible` is False; S 7.2)

```
if Ctil < -Crev:                          case = FB-1*   (contradictory limits; forward priority, procedure below)
elif sum(u[occ]) < -Crev:                 p = where(occ, u, 0);  case = FB-2;  done
else:                                     case = FB-1    (sum(l) > Ctil)
p = where(occ, pmin, 0); b = Ctil - sum(p)
if b < 0:                                 case = FB-0;  done          # keep maximal discharge
order = occupied chargers sorted by key (chi ascending, (l - pmin) descending, index ascending)
for i in order:
    add = min(l[i] - pmin[i], b); p[i] += add; b -= add
    if b <= 0: break
log case, sum(l)-Ctil, per-EV (l_tilde - p)^+ * dt
```

### Alg. 1.4 Actuation (wrapper; S 6.3, Lemma E3)

```
assert all(isfinite(p))
for occupied i:
    if u[i] > 0 and p[i] >= u[i] - 1e-3:  a_env[i] = 1.0
    elif p[i] > 0:                         a_env[i] = min(1, ceil(1e5*p[i]/(zeta_c[i]*P_cs) - 1e-9)/1e5)
    elif p[i] < 0:                         a_env[i] = max(-1, ceil(1e5*p[i]/(zeta_d[i]*P_cs) - 1e-9)/1e5)
    else:                                  a_env[i] = 0
empty chargers: a_env[i] = 0
```

Requirements:
- The env configuration must satisfy Corollary 2 in S 5: `v2g_enabled: True`, efficiencies constant over current levels via our spec JSON (DD4), zero dead-bands, 3 phases, multiplier 5.
- The layer reads $\zeta_i^{\mathrm c},\zeta_i^{\mathrm d}$ from each EV object.
- After `env.step`, read the reported powers $\tilde p_i$ from `cs.current_power_output`. Log $\max(\sum\tilde p-\sum p)$, which must be $\le n\varepsilon_q$, and the realised transformer violation.

---

## Alg. 2 PPO training with a variable number of chargers (DD14)

PPO [schulman2017proximal] with GAE [schulman2016highdimensional] and Adam [kingma2015adam], in a single-file implementation in the style of CleanRL [huang2022cleanrl].

```
Setup
  E_env = 16 parallel envs: 8 with N=10, 8 with N=20 (mixed batches; default).
          Curriculum variant (ablation): N=10 only for the first 20 % of env steps, then mixed.
  Each env: own seed stream; per episode draw profile ~ U{workplace, public, private} and rho_tr ~ U{4, 7} kW (Alg. 7).
  Networks: actor (attention, 610k params), critic (attention + pooling, 676k params); float32.
  Adam(lr=3e-4, eps=1e-5) over actor+critic; lr linearly decayed to 0 over the run.

Loop until 3e6 env steps (transitions):
  # ---------- rollout ----------
  for step in 1..128 (per env, vectorised over the 16 envs):
      obs = wrapper.observe()   -> tokens (N_env,12), global (38,), occ, l, u, Ctil, Crev, feasible flag
      with no_grad: mu, logstd = actor(obs); v = critic(obs)
      a  ~ Normal(mu, exp(logstd))                 # one raw action per charger (empty ones too, ignored)
      infl = occ & (u - l > EPS_NUM)               # tokens whose action can influence p (S 9)
      logp = sum_{i in infl} log N(a_i; mu_i, sigma_i)
      p = project(P_ref*a) if feasible else fallback()   # Alg. 1.2 / Alg. 1.3
      env.step(actuation(p)); r = profit (F9); r_tilde = r/(N*r_ref)        # DD15
      store (obs, a, logp, v, r_tilde, done, feasible, infl)   # padded to the env's own N
      on episode end (t = T): done = True (true terminal; no bootstrap), reset
  # ---------- advantages ----------
  GAE(gamma=0.99, lambda_GAE=0.95) per env on r_tilde with V(s_T)=0 at terminals; returns = adv + v
  # ---------- update ----------
  flatten 16x128 = 2048 transitions; pad token tensors to N_max = 20 (max N in buffer) with occ=False
  for epoch in 1..10:
      for minibatch of B_mb = 512 transitions (shuffled):
          mu, logstd = actor(obs_mb); v = critic(obs_mb)
          logp_new = sum_{i in infl} log N(a_i; mu_i, sigma_i)
          ratio = exp(logp_new - logp_old)
          A = (adv - mean)/std   (minibatch normalisation)
          w = feasible_mb                                     # actor loss masked on fallback steps (DD21)
          L_clip = sum_w[ min(ratio*A, clip(ratio, 1-0.2, 1+0.2)*A) ] / B_mb       # fixed normaliser
          H      = sum_w[ mean_{i in infl} entropy_i ] / B_mb
          xbar   = P_ref*mu
          with no_grad: pbar = project_batched(xbar, l, u, Ctil, Crev, occ)  (feasible rows; else pbar = xbar)
          L_reg  = sum_w[ mean_{i in occ} ((xbar_i - pbar_i)/P_ref)^2 ] / B_mb    # Moreau gradient (S 9)
          L_vf   = mean[(v - returns)^2]
          loss   = -L_clip + c_vf*L_vf - c_ent*H + beta*L_reg
          step Adam with grad-norm clip 0.5
      early stop the epoch loop if approx_kl > 0.03
  log: episode profit (EUR), infeasible steps, FB-0/1/2/1*, caps, admission reductions, A5 slack, guard counters, entropy, KL
```

**Batching episodes with different $N$.**
- Each env writes into its own buffer of width $N_{\mathrm{env}}$.
- At update time, all token tensors are right-padded to $N_{\max}$ = the largest $N$ in the batch (20 in training) with `occ=False`, $x=l=u=0$.
- By Lemma N1(iv), padding changes neither the actor outputs of real tokens nor the critic value. The log-probability, entropy and regulariser sums run over masks, so padding contributes nothing.

**On the masks (DD21).**
- The token mask uses the threshold $u_i-l_i>\epsilon_{\mathrm{num}}$; the resulting bias is of the order of an influence of $10^{-9}$ kW and is negligible.
- With the fixed normaliser $B_{\mathrm{mb}}$, both masks leave the score-function gradient unbiased. The clipped surrogate and GAE are biased estimators in any case (S 9).

---

## Alg. 3 Deployment loop with fallback and admission control

```
input: trained actor; replay of a test episode; mode in {known-d, conformal(alpha)}; admission in {on, off}
for t = 0..T-1:
  1. Arrivals at t (EVs present at t that were not present at t-1), for each new EV j:
     a. read (E_arr, E_des, C_cap, E_min, Pnom, Pdis, s_tr, m, zeta) from the EV object; d_j = ev.time_of_departure
        (conformal mode: d_j hidden from the controller; dlo_j = conformal_bound(w_j) (Alg. 4); use dlo_j below)
     b. K = (d_j or dlo_j) - t + 1
        E_req_j, Bhat_j = cap_request(E_des, E_arr, K, phi_j, E_min, C_cap)     # DD2 cap, certifies A4
     c. if admission == on:                                                     # DD20
           rho_j = taper_minorant(phi_j)                                        # S 7.4
           # targets: new EV E_req_j; connected EVs min(E_req_i, Psi_LP_i(E_i) - 1e-6)   (DD25)
           if not admission_LP_feasible(N(t), targets):                         # forecasts for t' > t
               y = bisect_largest_feasible(E_req_j, lower=min(E_arr, E_req_j), iters=12)  # via one LP for y_max (S 7.4)
               E_req_j, Bhat_j = cap_request(y, E_arr, K, phi_j, E_min, C_cap)  # re-certifies A4
               log reduction
     d. log capped/reduced amounts
  2. bounds: Alg. 1.1 -> occ, pmin, l, u, chi, Ctil, Crev, feasible
  3. obs = features(...)   (F 4.2); a = mu_theta(obs); x = P_ref*a
  4. if feasible: p = project(x, l, u, Ctil, Crev, occ)          # Proposition 1, Alg. 1.2
     else:        p, case = fallback(l, u, pmin, Ctil, Crev, chi, occ)   # S 7.2, Alg. 1.3, logged
  5. a_env = actuation(p, u, zeta)  (Alg. 1.4);  env.step(a_env)
  6. read p_tilde (reported), energies; log transformer violation (DD7 metric), sum(p_tilde)-sum(p),
     per-EV deficit dt*(l_tilde - p)^+  (Lemma 3)
  7. Departures at t (EVs with d_i = t; conformal mode: EVs that disappear): shortfall vs. DD2 E_req and vs. admitted E_req
end: profit J^pi; oracle J^LP (and J^LP,ceil) for the same replay (F 3, DD19); gap; all counters of S 7.5
```

`admission_LP_feasible` is the LP of S 7.4, solved with HiGHS [huangfu2018parallelizing] (`highspy`, to be installed in the venv). The oracle LP of F 3 uses the same solver.

---

## Alg. 4 Conformal calibration (DD12, DD23)

Split conformal prediction [papadopoulos2002inductive; vovk2022algorithmic; lei2018distributionfree; angelopoulos2023conformal] with a one-sided conformalised quantile regressor [romano2019conformalized].

```
For every test configuration c = (profile, N, rho_tr, dt, start hour, spawn multiplier, date range)   # DD23, DD24
  Data generation (seed ranges disjoint from RL training, validation and test):
    S_train(c): 2,000 episodes of c; take ALL sessions -> fit predictor q_hat
                (sklearn GradientBoostingRegressor(loss='quantile', alpha), 200 trees, depth 3)
                features w = [episode arrival step t_arr mod N_day (start hour fixed per c), weekday, E_arr/E_cap,
                              (E_des-E_arr)/E_ref, one-hot of the 6 distinct (capacity, AC power) spec classes]
    Sessions are drawn with EV2Gym's own EV_spawner after replicating reset's seeding/date logic (same law,
    cheaper than full resets); generated once per c for all alpha and cached.
    S_cal(c):   n_cal = 5,000 independent episodes of c; from each, pick ONE session uniformly at random
                (calibration points are i.i.d., hence exchangeable with a uniformly chosen EV of a test episode of c)
  Calibration:
    scores S_j = q_hat(w_j) - d_j,  j = 1..n_cal
    k = ceil((n_cal + 1)*(1 - alpha));  Q = k-th smallest score  (Q = +inf if k > n_cal)
  Online bound for a test EV of configuration c:
    dlo = max(t_arr, ceil(q_hat(w) - Q - 1e-9))   # integer; the -1e-9 only lowers the bound (still valid); S 8
Report per configuration and alpha in {0.01, 0.05, 0.1, 0.2}:
  coverage as the EPISODE-AVERAGED fraction of EVs with d >= dlo (the quantity Proposition 2 guarantees),
  the pooled fraction as secondary, mean (d - dlo) in steps, shortfall of uncovered EVs, profit,
  cost of the guarantee vs. known-d SA-Attn.
```

One predictor is fitted per $(c,\alpha)$. The configurations are the 36 test configurations plus the literature configuration of Alg. 7, so there are $37\times4=148$ calibrations. Generating the sessions needs only EV2Gym's arrival model, not RL rollouts.

**Cost (measured, IMPLEMENTATION_NOTES I-7/I-22).** EV2Gym's spawner costs about 4 ms per EV, because it fills a 101-entry efficiency table per EV. This amounts to about 1 h per $N=200$ configuration and **10-15 CPU-hours in total**. The earlier estimate of under 1 CPU-hour was wrong. Three measures keep it manageable:
- sessions are generated once per configuration for all $\alpha$;
- sessions are cached, so re-calibration takes seconds;
- session generation runs in parallel workers and parallel configurations.

Calibration stays per configuration, including $\rho_{\mathrm{tr}}$, as DD23 requires, although the session law does not depend on $\rho_{\mathrm{tr}}$. SA-Attn-CP is trained with $\tau^{\mathrm{lo}}$ calibrated at $\alpha=0.1$ for each training configuration; these coincide with test configurations and reuse their calibrations.

---

## Alg. 5 Baselines and penalty reward (DD22)

**Penalty reward** (penalty-based learned baselines only):
$$r^{\mathrm{pen}}(t)=r(t)-w_E\sum_{i:\,d_i=t}\big(E_i^{\mathrm{req}}-E_i(d_i+1)\big)^{+}-w_P\,\Delta t\Big[\Big(\sum_i\tilde p_i-C(t)\Big)^{+}+\Big(-C_{\mathrm{rev}}(t)-\sum_i\tilde p_i\Big)^{+}\Big].$$
- **Departure-shortfall weight** $w_E=10$ €/kWh per kWh of shortfall at departure.
- **Transformer-overload weight** $w_P=10$ €/kWh per kWh of overload energy (overload power in kW times $\Delta t$), in either direction.

Fairness rules:
1. 10 €/kWh is more than 11 times the highest hourly price in the 2022-23 data (0.871 €/kWh), so no violation can be profitable.
2. The weights are fixed a priori and not tuned, so every learned method gets the same tuning budget (Alg. 6). A sensitivity with $w_E=w_P\in\{1,100\}$ is run for the selected configuration only.
3. Same learning-reward normalisation $r^{\mathrm{pen}}/(N r^{\mathrm{ref}})$, the same observation (including the $l_i$ and $u_i$ features), the same PPO code and hyperparameter grid.
4. $E^{\mathrm{req}}$ is the same DD2-capped value.
5. The departure penalty is sparse, as in EV2Gym's `V2G_profitmax`. An optional dense variant, "penalty-dense", adds $w_E\Delta t\sum_i(l_i-p_i)^{+}$ each step. It is reported as an ablation, because it gives the penalty method the information of our lower bound.

Without the safety layer, a raw action is executed after the **per-EV physical clip only**: $p_i=\operatorname{clip}(x_i,p_i^{\min},u_i)$. EV2Gym does this internally anyway; we make it explicit so that actuation is identical. The clip contains neither the departure bound $l_i$ nor the transformer coupling.

Further rules for the penalty-based methods:
- Admission control is off; $E^{\mathrm{req}}$ is the DD2-capped value.
- $r^{\mathrm{pen}}$ uses the untightened $C(t)$ and $C_{\mathrm{rev}}(t)$ and the realised powers.
- The dense variant uses the commanded $p$.
- In their actor loss, the token mask is $o_i\wedge(u_i-p_i^{\min}>\epsilon_{\mathrm{num}})$ (the action can influence $p_i$), and there is no fallback mask.

| # | Method | Policy / rule | Safety layer | Reward | Notes |
|---|---|---|---|---|---|
| 1 | MLP+penalty | MLP actor/critic (2×256, tanh) on the concatenation of charger tokens in charger order, zero-padded to $N_{\max}=200$ slots, plus the global token; outputs 200 means and log-stds | No (physical clip only) | $r^{\mathrm{pen}}$ | Runs at any $N\le200$ with the same weights (zero padding). Not permutation-equivariant; slots 21-200 are never filled in training |
| 2 | MLP+safety layer | as 1 | Yes, $\Pi_t$ + fallback | $r$ + $\beta$-regulariser | Isolates the effect of the architecture |
| 3 | Attention+penalty | attention policy (F 5) | No (physical clip only) | $r^{\mathrm{pen}}$ | Isolates the effect of the safety layer |
| 4 | **Attention+safety layer (SA-Attn, ours)** | attention policy | Yes | $r$ + $\beta$-regulariser | Main method |
| 5a | AFAP | $x_i=u_i$ (charge at maximum) | raw and +$\Pi$ | – | Raw violates the transformer limit; +$\Pi$ is safe |
| 5b | ALAP | $x_i=\max(0,l_i^{\mathrm{G2V}})$, $l^{\mathrm{G2V}}$ = (S11) with $p^{\min}:=0$ | raw and +$\Pi$ | – | Our implementation (EV2Gym's ignores $\eta$ and the taper) |
| 5c | Round Robin | G2V: rotating pointer over occupied chargers, each gets $u_i$ until $\tilde C(t)$ is used (last partially); pointer advances by one per step | raw and +$\Pi$ | – | Forward-transformer-safe by construction, not departure-aware |
| 5d | LLF | G2V: allocate $\tilde C(t)$ in increasing laxity $\chi_i$ at $u_i$ each | raw and +$\Pi$ | – | Classical baseline, cf. [chen2022smoothed] |
| 6 | EV2Gym MPC | `eMPC_V2G` (`baselines/mpc/eMPC.py`): economic MPC with V2G, transformer constraint and hard departure constraint; horizon 10 steps. Run through an adapter subclass with the identical formulation that fixes two numpy-2/efficiency issues (1-D solution vector; percent efficiency divided by 100) without modifying EV2Gym | raw (as published) and +$\Pi$ | – | Uses true prices and future arrivals within the horizon (information advantage, flagged); one efficiency (first EV) for all EVs; zero actions when infeasible (published behaviour). Restricted Gurobi licence $\Rightarrow$ only $N\le25$. For larger $N$: **eMPC-HiGHS**, a related but **not identical** port: net-power LP (exact for $c^{\mathrm{dis}}=c^{\mathrm{ch}}$), per-EV limits, exact first-step bounds including the taper, the wrapper's $E^{\mathrm{req}}$, an elastic departure constraint (1000 €/kWh), and EV2Gym's internal load/PV forecasts made unbiased by the scenario config (`forecast_mean: 100`, `forecast_std: 10`). It is reported under its own name |
| 7 | Perfect-information optimum | LP of F 3 (HiGHS), lexicographic shortfall-then-profit (DD19) | – | – | Upper bound $J^{\mathrm{LP}}$ (Proposition O1); not a controller |
| 8 | EV-GNN [orfanoudakis2025scalable] (optional, DD27) | public code at a pinned commit, its own training script, default reward (penalty-based), algorithm and hyperparameters (not tuned). The script accepts one scenario per run, so **one model per arrival profile** is trained at $N=20$, $\rho_{\mathrm{tr}}=4$ kW, 5 seeds | raw and +$\Pi$ | its own | Evaluated zero-shot on all $(N,\rho_{\mathrm{tr}})$ of its profile and reported as "trained per profile"; this protocol difference (per-profile vs. mixed training, single training size) is stated. Run unmodified via a launcher restoring numpy aliases removed in numpy 2. **GNN-DT [orfanoudakis2026graph] is not included**: no public code was found, and the paper says so |
| 9 | SA-Attn-CP($\alpha$) | as 4, with $\tau^{\mathrm{lo}}$ in features and layer | Yes | $r$ | $\alpha\in\{0.01,0.05,0.1,0.2\}$; plus SA-Attn-point ($\hat Q=0$, uncalibrated) as ablation |
| 10 | Ablations of 4 | no admission control; $\beta=0$; curriculum vs mixed; small network ($d_m=64$, $n_L=2$) | Yes | $r$ | – |

"+$\Pi$" means the method's output is used as the raw action $x$ of the safety layer, with the same fallback, admission control and actuation. All "+$\Pi$" variants carry the guarantee of Theorem 1, so their profits can be compared fairly with SA-Attn. The **cost of the guarantee** is profit(Attention+penalty) − profit(SA-Attn), together with the violations the penalty method incurs.

---

## Alg. 6 Hyperparameters, tuning and statistics

| Group | Parameter | Default |
|---|---|---|
| Network | embedding $d_m$ | 128 |
|  | encoder layers $n_L$ | 3 (pre-LN) |
|  | heads $n_H$ | 4 |
|  | FFN width | 512, GELU |
|  | dropout | 0 |
|  | actor head | $d_m\to d_m/2\to2$ ($128\to64\to2$), shared over chargers; critic head $d_m\to d_m/2\to1$ |
|  | $\log\sigma$ clamp / init | $[-5,0.5]$ / $-0.5$ |
|  | critic readout | attention pooling (1 learned query, 4 heads), $128\to64\to1$ |
|  | actor / critic weights | separate encoders |
|  | precision | float32 networks, float64 safety layer |
| PPO | optimiser | Adam, $\epsilon=10^{-5}$ |
|  | learning rate | $3\times10^{-4}$, linear decay to 0 |
|  | discount $\gamma$ | 0.99 (15 min); $0.99^{1/3}$ (5 min) |
|  | GAE $\lambda_{\mathrm{GAE}}$ | 0.95 |
|  | clip $\epsilon_{\mathrm{clip}}$ | 0.2 |
|  | entropy coefficient $c_{\mathrm{ent}}$ | 0.0 (tuned) |
|  | value coefficient $c_{\mathrm{vf}}$ | 0.5 |
|  | value clipping | off |
|  | max grad norm | 0.5 |
|  | target-KL early stop | 0.03 |
|  | parallel envs | 16 (8 × $N=10$, 8 × $N=20$) |
|  | rollout length | 128 steps per env (2048 transitions per update) |
|  | minibatch $B_{\mathrm{mb}}$ | 512 transitions |
|  | epochs per update | 10 |
|  | total env steps per run | $3\times10^6$ (about 26,800 episodes of 112 steps) |
|  | seeds | 5 (0-4), each a full run |
| Safety layer | $\beta$ (regulariser) | 0.01 |
|  | $P^{\mathrm{ref}}$ | 11 kW |
|  | $\varepsilon_q$ / $\varepsilon_{\mathrm{sat}}$ / $\epsilon_{\mathrm{num}}$ | 0.008 kW / 0.001 kW / $10^{-9}$ |
|  | threshold bisection tolerance | $10^{-10}$ kWh |
| Features | $E^{\mathrm{ref}}$, $c^{\mathrm{ref}}$, $H_f$ | 100 kWh, 0.25 €/kWh, 8 steps |
| Penalty baselines | $w_E$, $w_P$ | 10 €/kWh each (fixed, DD22) |
| Evaluation | test episodes per configuration | 100 for $N\le50$, 50 for $N\ge100$ (fixed replays; Alg. 7.3) |
|  | action at test time | deterministic, $a=\mu_\theta$ |
| Conformal | $n_{\mathrm{cal}}$ / training episodes | 5,000 / 2,000 per configuration |

**Tuning protocol (equal budget for all learned methods).**
- The tuning is done on the 10-charger station only (tight level $\rho_{\mathrm{tr}}=4$ kW), with validation seeds disjoint from training and test seeds. The 20 validation episodes cycle through the profiles (7/7/6).
- Full grid of 18 configurations per learned method, for methods 1-4. Method 9 and the SA-Attn ablations inherit the selection of method 4, and the penalty ablations that of method 3. EV-GNN (method 8) uses its published hyperparameters (DD27):
  - learning rate $\in\{1\times10^{-4},3\times10^{-4},1\times10^{-3}\}$;
  - minibatch $\in\{256,1024\}$;
  - entropy coefficient $\in\{0,10^{-3},10^{-2}\}$.
- Each configuration is trained for $0.5\times10^{6}$ env steps with seed 0 and evaluated on 20 validation episodes.
- Selection uses each method's own training objective: mean validation return in $r$ for safety-layer methods and in $r^{\mathrm{pen}}$ for penalty methods.
- The selected configuration is retrained with 5 seeds for $3\times10^6$ steps.
- The total tuning budget is the same for every method: $18\times0.5\times10^6=9\times10^6$ env steps.
- $\beta$, the architecture and all other PPO constants are fixed.

**Statistics.**
- Five independent training seeds per method, following the reproducibility guidance of [henderson2018deep].
- Aggregate scores over seeds and test episodes are reported with stratified-bootstrap 95 % confidence intervals, interquartile mean alongside the mean [agarwal2021deep].
- Paired comparisons between two methods on the same replays (common random numbers) use the Wilcoxon signed-rank test [wilcoxon1945individual] on per-episode profit and per-episode shortfall, with Holm correction across $N$.

---

## Alg. 7 Experiment scenarios: scaling across $N$ and arrival profiles (DD24)

Goal: per-charger congestion must be comparable across station sizes and arrival profiles, so that zero-shot transfer (C3) tests the policy rather than a change in difficulty. Every exogenous quantity therefore scales linearly with $N$, and the spawn rate of each profile is set so that mean charger occupancy is the same.

### Alg. 7.1 Arrival profiles

EV2Gym provides three arrival profiles (`scenario` key; `utils.py:477-553` spawner, `utils.py:200-252` session sampler):
- arrival rates per 15-min slot come from `data/distribution-of-arrival.csv` (weekdays) and `-weekend.csv`;
- mean stay per arrival half-hour comes from `data/mean-session-length-per.csv` (stay $\sim\mathcal N(\mu,0.2\mu)$);
- mean requested energy comes from `data/mean-demand-per-arrival.csv` (energy $\sim\mathcal N(\mu,0.5\mu)$);
- the CSV columns `home`/`work` are renamed to `private`/`workplace` in `loaders.py:82-89`.

| Profile | Arrivals | Stays (table means) | EV2Gym special rule |
|---|---|---|---|
| **workplace** | weekday mornings (median arrival 09:15) | about 8 h for morning arrivals, 2-4 h after noon | arrivals only 06-18 h on weekdays; weekend dates are skipped |
| **public** | spread over the day, peak in the afternoon and evening | 3-5 h by day, 10-12 h for evening arrivals (overnight) | weekend table on weekends |
| **private** (residential) | late afternoon and evening (median 19:00) | 13-15 h for arrivals 15-19 h (overnight, departure next morning) | weekend table on weekends |

**Start hour and the episode horizon.** The horizon is $T=112$ steps of 15 min, i.e. 28 h, for all profiles. With `empty_ports_at_end_of_simulation=True`, EV2Gym **does not spawn** an EV whose departure would fall within 4 steps of the episode end (`utils.py:254-256`). It neither truncates the session nor caps the request. So the horizon never cuts a session short, but sessions that do not fit are silently missing.

With EV2Gym's default start at 05:00, the overnight private sessions do not fit: **about 60 % of private spawn attempts are dropped** (and 24 % of public ones), and the remaining private sample is biased towards short stays (mean 9 h instead of 12.4 h). We therefore use a per-profile start hour:
- workplace starts at 05:00, as in `V2GProfitMax.yaml`, and ends at 09:00 the next day. No spawn attempt is dropped.
- public and private start at 12:00 and end at 16:00 the next day. The whole evening-arrival / morning-departure cycle then fits. Drops fall to 6.8 % (public) and 3.7 % (private); these are the latest evening arrivals with the longest stays.

The episode starts empty. With the 12:00 start this matches the near-empty midday state of residential and public chargers. The remaining drops are reported as a limitation, and the episode-start time enters the policy through the time-of-day features.

**Equalised occupancy.** `spawn_multiplier` is set per profile so that mean charger occupancy (fraction of charger-steps with an EV) equals that of EV2Gym's default workplace setting (multiplier 5). This was measured with `profile_check.py` (results in `profile_check_results.txt`; 20 chargers, 40 seeds per row, $T=112$, 15 min):

| Profile | start | `spawn_multiplier` | days | EVs per charger per day | mean stay [h] | median arrival | occupancy | spawn attempts dropped | mean request [kWh] | energy per charger per day [kWh] |
|---|---|---|---|---|---|---|---|---|---|---|
| workplace | 05:00 | 5 | weekdays | 0.65 | 7.7 | 09:15 | 0.208 | 0.0 % | 14.6 | 9.5 |
| public | 12:00 | 4 | both | 0.66 | 8.1 | 16:00 | 0.224 | 6.8 % | 15.9 | 10.5 |
| private | 12:00 | 2.5 | both | 0.42 | 12.2 | 18:45 | 0.214 | 3.7 % | 23.5 | 9.9 |

Equalising occupancy also equalises the energy demand per charger to within ±5 %. Private stations get fewer but longer and larger sessions.

What is **not** equalised:
- **Timing.** The concurrency profile over the day differs: private demand coincides with the evening price peak and the night, workplace demand with the morning. This is the intended difference between profiles.
- **Weekdays.** Workplace is weekdays only (EV2Gym rule). Public and private use `simulation_days: both`, so the weekend flag in the global token varies.

**No profile flag in the observation.** The global token does **not** contain the profile; $F_g=38$ is unchanged. Reasons:
1. Everything the decision needs from the profile is visible in the tokens (remaining stay $\tau_i$, requested energy, laxity) and in the time-of-day and weekend features.
2. A policy without the flag can be deployed at stations whose profile is mixed or unknown.
3. Leaving the flag out makes the profile a further zero-shot dimension: training mixes the profiles, and testing reports results per profile.

The cost is that the critic cannot anticipate profile-specific future arrivals beyond what time of day reveals. The critic is used only in training, and partial observability of future arrivals exists anyway (F 4.1). The conformal predictor (Alg. 4) is calibrated per configuration, profile included. This uses station metadata that a CPO has; it is not an observation of the policy.

### Alg. 7.2 Scaling with $N$ and grid parameters

| Item | Setting | EV2Gym key | Scaling with $N$ |
|---|---|---|---|
| Station sizes | train $N\in\{10,20\}$; test $N\in\{5,10,20,50,100,200\}$ | `number_of_charging_stations`, `number_of_ports_per_cs: 1` | – |
| Transformer | one transformer, $P_{\mathrm{tr}}=\rho_{\mathrm{tr}}N$ with two tightness levels: **tight** $\rho_{\mathrm{tr}}=4$ kW per charger (EV2Gym's V2GProfitMax ratio, 100 kW / 25 chargers) and **moderate** $\rho_{\mathrm{tr}}=7$ kW per charger | `number_of_transformers: 1`, `transformer.max_power` $=\rho_{\mathrm{tr}}N$ | linear |
| Inflexible load | on; peak over the episode $\approx0.4\,P_{\mathrm{tr}}$ | `inflexible_loads.include: True`, `inflexible_loads_capacity_multiplier_mean: 0.4` | EV2Gym rescales the load to multiplier × $P_{\mathrm{tr}}$, hence linear |
| PV | on; peak $\approx0.853\cdot U[0.9,1.1]\cdot\mathcal N(0.4,0.1)\,P_{\mathrm{tr}}$, i.e. about $0.38P_{\mathrm{tr}}$ on average and below $0.7P_{\mathrm{tr}}$ even at $+3$ standard deviations | `solar_power.include: True`, `solar_power_capacity_multiplier_mean: 0.4` | multiplier × $P_{\mathrm{tr}}$, hence linear; keeps $C_{\mathrm{rev}}>0$ always, so FB-2 cannot occur |
| Demand response | off | `demand_response.include: False` | – |
| Arrivals | profile-specific (Alg. 7.1) | `scenario`, `spawn_multiplier`, `hour`, `simulation_days` | EV2Gym spawns per charger independently, so occupancy per charger is $N$-independent |
| EV fleet | heterogeneous specs from our own JSON (DD4: efficiency constant over current levels), `transition_soc_multiplier: 5`, `desired_capacity: 1.0` (main) and 0.8 (sensitivity, DD2), `min_time_of_stay: 180` | `heterogeneous_ev_specs: True`, `ev_specs_file`, `ev.*` | – |
| Time | $\Delta t=15$ min, $T=112$, `random_day: True`, `random_hour: False`; 5-min sensitivity with $T=336$ (same start hours) | `timescale`, `simulation_length`, `hour` | – |
| Prices | NL day-ahead (EV2Gym), `discharge_price_factor: 1` | – | identical |
| Forecasts in the observation | $\max\{0,\text{true}\cdot(1+0.1\epsilon)\}$, $\epsilon\sim\mathcal N(0,1)$ i.i.d. per $(t,k)$; last true value (and price) held beyond $T$ (DD6) | our wrapper | – |
| EV2Gym's internal load/PV forecasts (used only by eMPC-HiGHS) | unbiased: `forecast_mean: 100`, `forecast_std: 10` (EV2Gym defaults 30/5 are biased) | `inflexible_loads.*`, `solar_power.*` | – |
| Load/PV randomness | redrawn per episode: the wrapper sets `env.tr_seed` to the episode seed before each reset (EV2Gym's `tr_seed: -1` would reuse the constructor seed for all resets); every replay is a (configuration, seed) pair with a fingerprint of sessions, prices, loads, PV and rating | attribute, no patch | – |
| Horizon handling | `empty_ports_at_end_of_simulation: True` (every EV departs by $T-2$; no session truncated; dropped spawns logged) | constructor argument | – |

**What scaling achieves.** The expected aggregate demand and the transformer headroom per charger, $(P_{\mathrm{tr}}-P^{\mathrm{inf}}+P^{\mathrm{pv}})/N$, have the same mean for every $N$. Relative fluctuations of the aggregate shrink roughly as $1/\sqrt N$, so infeasible steps are expected to become rarer at large $N$. This is reported per $N$ as part of the C3 analysis, not treated as a confound.

### Alg. 7.3 Training and test grid

**Training (one run covers all training configurations).**
- $N\in\{10,20\}$: 8 envs each (Alg. 2).
- Per episode, draw the profile uniformly from {workplace, public, private} and $\rho_{\mathrm{tr}}$ uniformly from $\{4,7\}$ kW.
- This gives 12 training configurations, mixed within every PPO batch.

**Test grid.**
- 6 sizes × 2 tightness levels × 3 profiles = **36 configurations**.
- 100 replays each for $N\le50$, and **50 replays for $N\in\{100,200\}$** to bound compute. Paired comparisons on common replays keep the confidence intervals usable at 50.
- The replays are generated once and shared by all methods.
- In addition, the **literature configuration**: EV2Gym's original `V2GProfitMax.yaml` (25 chargers, 100 kW, workplace, multiplier 5, start 05:00, loads/PV off), 100 replays, for comparability with EV2Gym-based work. It uses our constant-efficiency spec file instead of EV2Gym's current-dependent tables, because Corollary 2 requires it. This is stated as a deviation.

**Sensitivities (DD28).** These are run at $N=20$, $\rho_{\mathrm{tr}}=4$ kW, for all three profiles (6 extra configurations):
- desired SoC 0.8 (DD2), re-using the main policies;
- $\Delta t=5$ min with $T=336$, which needs a separately trained SA-Attn policy (5 seeds, $\gamma=0.99^{1/3}$; baselines are evaluated too).

Configs: `test_grid_sensitivity.yaml`, `configs/ablations/train_attn_safe_dt5.yaml`. The main grid is unchanged.

Total: $4\cdot2\cdot3\cdot100+2\cdot2\cdot3\cdot50+100=2400+600+100=$ **3100 test episodes per evaluated policy**.

### Alg. 7.4 Compute estimate

**Measured.** EV2Gym step time (pinned commit, one CPU core, random actions) is 1.9 ms at $N=20$ and 6.6 ms at $N=200$; reset takes 0.18-0.34 s. Indicative wrapper and PPO timings come from the implementation (`code_repo/code/README.md` Sec. 4, `make bench`; 2 CPU cores, no GPU):

| $N$ | wrapper observe | wrapper step (layer + actuation + EV2Gym) | observe at arrival with admission LP (mean) | actor forward (1 thread) | PPO update, 2048 transitions |
|---|---|---|---|---|---|
| 10 | 0.36 ms | 0.59 ms | 0.70 ms | 1.3 ms | 16.6 s (10 epochs, minibatch 512) |
| 50 | 1.08 ms | 1.37 ms | 5.5 ms | 2.1 ms | 9.4 s per epoch |
| 200 | 3.34 ms | 3.73 ms | 66 ms | 5.3 ms | 38.6 s per epoch (minibatch 64; 512 exceeds 7 GB RAM) |

On 2 cores, one training update (128 rollout steps of 16 envs + 10 epochs) takes about 18 s, i.e. **about 7 h per $3\times10^6$-step run**. With a GPU and at least 16 cores, about 0.5 h per run is expected. These timings are indicative, not results. At $N=200$, the admission LP dominates controller time at arrival steps.

**Training runs** (each $3\times10^6$ env steps, own PPO code):

| Item | Runs |
|---|---|
| Main: MLP+penalty, MLP+safety layer, Attention+penalty, SA-Attn, SA-Attn-CP ($\alpha=0.1$) — 5 × 5 seeds | 25 |
| Ablations: $\beta=0$, curriculum, small network, penalty-dense — 4 × 5 seeds | 20 |
| Penalty-weight sensitivity $w\in\{1,100\}$ — 2 × 3 seeds | 6 |
| **Total (own PPO)** | **51 runs ≈ $1.5\times10^8$ env steps** |
| Tuning: methods 1-4 × 18 configurations × $0.5\times10^6$ | 72 short runs, $3.6\times10^7$ env steps |
| Optional EV-GNN (DD27): 3 profiles × 5 seeds, with its own TD3 script | 15 runs, about $4.5\times10^7$ env steps, GPU recommended |
| Optional sensitivity (DD28): $\Delta t=5$ min SA-Attn | 5 runs, $1.5\times10^7$ env steps |

The "no admission control" ablation is evaluation-only. SA-Attn-CP is evaluated at all $\alpha\in\{0.01,0.05,0.1,0.2\}$ without retraining; Proposition 2 holds for any policy. Training needs about 30 GPU-hours plus about 6 GPU-hours of tuning, or about 120-180 CPU-hours on 16 cores; on 2 cores, about 7 h per run.

**Evaluation.**
- 3100 episodes × 112 steps per policy, about 0.5 CPU-hours per policy plus admission LPs.
- About 93 evaluated policies (81 learned evaluations including ablation variants, and 12 baseline policies): **about 50-80 CPU-hours**, parallel per configuration.
- eMPC-HiGHS at $N\ge50$ takes several seconds per step and is the most expensive baseline.
- Oracle: 3100 lexicographic LPs at 0.01-10 s each, at most 3 CPU-hours.
- **Conformal calibration: 10-15 CPU-hours** (measured, Alg. 4), parallelisable.
- Sensitivities (DD28): a few CPU-hours.
