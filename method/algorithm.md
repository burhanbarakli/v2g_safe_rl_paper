# Algorithms, hyperparameters, baselines and scenarios

Agent A2 (Method and Theory). Version 5, 2026-10-22, revised after the internal proof check, the implementation feedback and internal review round 1 (`code_repo/code/IMPLEMENTATION_NOTES.md`; change log in `DONE_method.md`). Symbols: `nomenclature.md`. Equations (F·): `formulation.md`; (S·): `safety_layer.md`. Design decisions DD1-DD41: `design_decisions.md`.

The NumPy reference `projection_ref.py` is the executable specification of Alg. 1 and Alg. 3. Any PyTorch [paszke2019pytorch] port must match it to $10^{-9}$ in float64 on the tests T1 and T1b of `verify_projection.py`, including the tight-A5 and large-magnitude families.

Conventions:
- One EV2Gym environment = one station with $N$ chargers, one port each (DD5).
- A **wrapper** around the env owns all safety-layer state (threshold tables, $E^{\mathrm{req}}$, $\varrho_i^{\mathrm{in}}$), computes observations, applies the safety layer, the fallback rule and the actuation map, and logs metrics. EV2Gym itself is not patched.
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
      for minibatch of n_mb = 512 transitions (shuffled):
          mu, logstd = actor(obs_mb); v = critic(obs_mb)
          logp_new = sum_{i in infl} log N(a_i; mu_i, sigma_i)
          ratio = exp(logp_new - logp_old)
          A = (adv - mean)/std   (minibatch normalisation)
          w = feasible_mb                                     # actor loss masked on fallback steps (DD21)
          L_clip = sum_w[ min(ratio*A, clip(ratio, 1-0.2, 1+0.2)*A) ] / n_mb       # fixed normaliser
          H      = sum_w[ mean_{i in infl} entropy_i ] / n_mb
          xbar   = P_ref*mu
          with no_grad: pbar = project_batched(xbar, l, u, Ctil, Crev, occ)  (feasible rows; else pbar = xbar)
          L_reg  = sum_w[ mean_{i in occ} ((xbar_i - pbar_i)/P_ref)^2 ] / n_mb    # Moreau gradient (S 9)
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
- With the fixed normaliser $n_{\mathrm{mb}}$, both masks leave the score-function gradient unbiased. The clipped surrogate and GAE are biased estimators in any case (S 9).

---

## Alg. 3 Request pipeline and deployment loop (DD31)

### Alg. 3.0 Station-level request pipeline (once per replay, before any controller; S 7.4)

```
input: replay (sessions with nominal arrival states, loads/PV/forecasts as functions of (replay, t)); departure mode
       in {known-d, conformal(alpha)}; admission in {on, off}
plan = {}                                         # nominal energy trajectories of admitted EVs
for arrivals j in time order (ties: charger index):
    t0 = t_arr_j; K = (d_j or dlo_j) - t0 + 1
    E_req_j, Bhat_j = cap_request(E_des_j, E_arr_j, K, phi_j, E_min, C_cap)      # DD2, certifies A4
    if admission == on:
        E_i(t0) for earlier connected EVs := plan[i](t0)                         # nominal, controller-independent
        targets: j -> E_req_j; i -> min(E_req_i, Psi_LP_i(E_i(t0)) - 1e-6)       # DD25
        capacity: realised at t0, forecasts made at t0, last held; minus n(t0)*eps_q
        if elastic plan with E_req_j has zero slack: keep E_req_j                 # fast path
        else: y_max = LP maximising E_j(d_j+1); y = 12-step bisection with predicate y <= y_max + 1e-6
              (not restored -> min(E_arr_j, E_req_j)); E_req_j, Bhat_j = cap_request(y, ...)   # re-certifies A4
        plan = elastic LP (slack weight 1e5, objective sum_i sum_k min(E_i(k), target_i)) -> store trajectories
    table[(charger, t0)] = (E_req_j, Bhat_j, E_arr_j, d_ctrl_j)
cache table per (test configuration, replay seed, eps_q, targets mode, forecast noise, departure mode)
```

The table is identical for every method and for the oracles on a replay. Conformal variants have their own table because they use $\hat d^{\mathrm{lo}}$. The pipeline is causal: decision $j$ uses information up to $t_0$ only.

### Alg. 3.1 Deployment loop

```
input: trained actor (or any controller); replay; request table from Alg. 3.0
for t = 0..T-1:
  1. arrivals at t: look up (E_req_j, Bhat_j) in the table by (charger, t); check E_arr and d_ctrl against the table
  2. bounds: Alg. 1.1 -> occ, pmin, l, u, chi, Ctil, Crev, feasible (incl. the A6 check, logged)
  3. obs = features(...)   (F 4.2); a = mu_theta(obs); x = P_ref*a (non-finite entries -> 0, counted)
  4. if feasible: p = project(x, l, u, Ctil, Crev, occ)                    # Proposition 1, Alg. 1.2
     else:        p, case = fallback(l, u, pmin, Ctil, Crev, chi, occ)     # S 7.2, Alg. 1.3, logged
     (raw methods: p = clip(x, pmin, u) instead of steps 4-; their requests still come from the table)
  5. a_env = actuation(p, u, zeta) (Alg. 1.4); env.step(a_env)
  6. read p_tilde, energies; log transformer violation (DD7), sum(p_tilde)-sum(p), per-EV deficit (Lemma 3),
     A5-complete flags per EV
  7. departures: shortfall vs. admitted E_req (main), vs. DD2-capped E_req and E_des (secondary)
end: profit, J^pen (F 3.4); oracles J^pen,LP, J^LP,ceil (and J^LP) for the same replay and table; gaps; S 7.5 counters
```

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

One predictor is fitted per $(c,\alpha)$. The configurations are the 43 test configurations of Alg. 7.3, so there are $43\times4=172$ calibrations. Generating the sessions needs only EV2Gym's arrival model, not RL rollouts.

**Cost (measured, IMPLEMENTATION_NOTES I-7/I-22).** EV2Gym's spawner costs about 4 ms per EV, because it fills a 101-entry efficiency table per EV. This amounts to about 1 h per $N=200$ configuration and **10-15 CPU-hours in total**. The earlier estimate of under 1 CPU-hour was wrong. Three measures keep it manageable:
- sessions are generated once per configuration for all $\alpha$;
- sessions are cached, so re-calibration takes seconds;
- session generation runs in parallel workers and parallel configurations.

Calibration stays per configuration, including $\rho_{\mathrm{tr}}$, as DD23 requires, although the session law does not depend on $\rho_{\mathrm{tr}}$. SA-Attn-CP is trained with $\tau^{\mathrm{lo}}$ calibrated at $\alpha=0.1$ for each training configuration; these coincide with test configurations and reuse their calibrations.

---

## Alg. 5 Methods, baselines and ablations (DD22, DD27, DD31, DD33-DD35)

**Common rules for all methods.**
- **Requests.** All methods use the same admitted requests from Alg. 3.0 (DD31). Conformal variants use their own table.
- **Raw methods.** Methods without the safety layer execute $p_i=\operatorname{clip}(x_i,p_i^{\min},u_i)$, the per-EV physical clip that EV2Gym applies anyway. This clip contains neither $l_i$ nor the transformer coupling.
- **"+$\Pi$" variants** pass the method's output as raw action $x$ through the safety layer, with the fallback rule and actuation. They inherit Theorem 1′.
- **Penalised profit.** Every method is evaluated with $J^{\mathrm{pen}}$ at fixed $w_E=w_P=10$ €/kWh (F 3.4).

**Penalty reward** (training reward of penalty-based methods; weight $w=w_E=w_P\in\{1,10,100\}$ €/kWh tuned, DD37):
$$r^{\mathrm{pen}}(t)=r(t)-w\sum_{i:\,d_i=t}\big(E_i^{\mathrm{req}}-E_i(d_i+1)\big)^{+}-w\,\Delta t\Big[\Big(\sum_i\tilde p_i-C(t)\Big)^{+}+\Big(-C_{\mathrm{rev}}(t)-\sum_i\tilde p_i\Big)^{+}\Big].$$
- Shortfall is measured against the admitted request, and the untightened $C$, $C_{\mathrm{rev}}$ are used.
- The dense ablation adds $-w\Delta t\sum_i(l_i-p_i)^{+}$ with the commanded $p$.
- At the reported weight of 10 €/kWh, more than 11 times the highest hourly price, no violation pays at the optimum of the penalised objective. Learned policies can still violate.

**PPO-Lagrangian** (`attn_lagrangian`, DD33(3); constrained MDP [altman2021constrained], Lagrangian relaxation with dual ascent [tessler2018reward]).
- **Problem.** $\max_\theta J_r(\theta)$ s.t. $J_E(\theta)\le d_E$, $J_P(\theta)\le d_P$, with $d_E=d_P=0$.
- **Costs.** $J_E$ and $J_P$ are the expected episode costs. The per-step costs are $c_E(t)=\sum_{i:d_i=t}(E_i^{\mathrm{req}}-E_i(d_i+1))^{+}$ and $c_P(t)=\Delta t\,[(\sum\tilde p-C)^{+}+(-C_{\mathrm{rev}}-\sum\tilde p)^{+}]$ [kWh], normalised as $c/(NP^{\mathrm{ref}}\Delta t)$.
- **Policy.** Attention policy, physical clip only (no safety layer), profit reward.
- **Critic.** Three heads (value and two cost values), each with GAE at $(\gamma,\lambda_{\mathrm{GAE}})$.
- **Policy update.** PPO on the combined advantage $(A_r-\Lambda_EA_E-\Lambda_PA_P)/(1+\Lambda_E+\Lambda_P)$, then per-minibatch normalisation.
- **Dual update.** Once per rollout, $\Lambda_k\leftarrow\operatorname{clip}(\Lambda_k+\eta_\Lambda(\hat J_k-d_k),0,\Lambda_{\max})$, with $\Lambda_{\max}=1000$, $\Lambda_k(0)=0$ and $\eta_\Lambda\in\{0.5,2,8\}$ tuned.
- **Scale.** $\Lambda=40$ corresponds to $w=10$ €/kWh at $c^{\mathrm{ref}}=0.25$ €/kWh.

**sLLF** (`sllf`, `sllf_safe`, DD33(5); smoothed least-laxity-first [chen2022smoothed], price-unaware, G2V only).
- Next-slot laxity against the certified thresholds: $\chi^0_i=(E_i-\hat B_i(\tau_i-1))/(\bar P_i^{\mathrm{nom}}\Delta t)$.
- Rate $p_i=\operatorname{clip}(\bar P_i^{\mathrm{nom}}(L-\chi_i^0),0,u_i)$, with a common level $L$ found by bisection (feasible end) such that $\sum_ip_i=\max(0,\tilde C)$; $p=u$ if $\sum u\le\max(0,\tilde C)$.
- This is the paper's rule $r_i=[\bar r_i(L-l_i+1)]$ clipped to $[0,\min(\bar r_i,e_i)]$, with $\bar r_i=\bar P_i^{\mathrm{nom}}$, the "+1" absorbed into $L$, and the cap $u_i$ including the taper.

**Price-threshold heuristic** (`price_llf`, `price_llf_safe`, DD33(6)). Let $\tilde c$ be the median of all prices seen so far in the episode together with the known $H_f$-step price window.
- If $c^{\mathrm{ch}}(t)<\tilde c$: LLF allocation of $\tilde C(t)$ (charge).
- If $c^{\mathrm{ch}}(t)>\tilde c$: $x_i=p_i^{\min}$ (maximum V2G discharge).
- Otherwise: idle.

The "+$\Pi$" variant restores the departure and transformer constraints.

**Method table.**

| # | Method | Policy / rule | Safety layer | Training reward | Notes |
|---|---|---|---|---|---|
| 1 | **SA-Attn (ours)** `attn_safe` | attention policy (F 5) | yes | $r$ + $\beta$-regulariser | main method |
| 2 | Attention+penalty `attn_penalty` | attention policy | no | $r^{\mathrm{pen}}$ ($w$ tuned) | the "cost of the guarantee" twin |
| 3 | Attention+penalty, deploy-only projection `attn_penalty_safedeploy` (DD33(2)) | checkpoints of 2 | at deployment only | – | isolates training-time use of the layer |
| 4 | Attention+transformer-only projection `attn_proj_nolb` (DD33(1)) | attention policy | box $[p^{\min},u]$ + aggregate, $l_i:=p_i^{\min}$ | $r^{\mathrm{pen}}$ ($w$ tuned), $\beta=0.01$ | per-step transformer layer without the departure bound (the closest provable prior); observation still contains the departure-aware $l_i$; transformer part of Theorem 1′ holds, (c) does not |
| 5 | PPO-Lagrangian `attn_lagrangian` (DD33(3)) | attention policy | no | $r$, costs $c_E$, $c_P$ | constrained-RL baseline |
| 6 | Deep Sets + safety layer `deepsets_safe` (DD33(4)) | per-token MLP with mean-pooled context (F 5, Lemma N2) | yes | $r$ + $\beta$ | non-attention size-agnostic policy; evidence for or against C1 |
| 7 | MLP + safety layer `mlp_safe`; MLP+penalty `mlp_penalty` | MLP (2×256) on charger tokens zero-padded to 200 slots + global token | yes / no | $r$ / $r^{\mathrm{pen}}$ | **fixed-size reference only.** Slots beyond 20 receive no gradient in training, so results are reported at $N\le20$ and on the 25-charger literature configuration |
| 8 | Heuristics AFAP, ALAP, Round Robin, LLF | as in version 4 (G2V) | raw and +$\Pi$ | – | price-unaware |
| 9 | sLLF; price-threshold LLF | above | raw and +$\Pi$ | – | canonical laxity scheduler; simple price-aware rule |
| 10 | eMPC-HiGHS `empc_highs` (DD34) | net-power receding-horizon LP, horizon 10 steps (2.5 h), true prices and future arrivals within the horizon, wrapper's $E^{\mathrm{req}}$, exact first-step bounds, elastic departure (1000 €/kWh) | raw and +$\Pi$ | – | the single MPC for **all $N$** of the grid. Information advantage flagged |
| 11 | eMPC-HiGHS 24 h `empc_highs_24h` (DD33(7)) | as 10 with a 96-step (24 h) horizon | raw and +$\Pi$ | – | long-horizon receding LP: the "why RL?" baseline |
| 12 | EV2Gym eMPC (Gurobi) `empc` (DD34) | EV2Gym `eMPC_V2G` via the adapter subclass | raw and +$\Pi$ | – | **only on the 25-charger literature configuration**, as a cross-check of eMPC-HiGHS |
| 13 | EV-GNN [orfanoudakis2025scalable] (DD27, DD35) | public code, own TD3 script, reward and hyperparameters; one model per profile at $N=20$, $\rho_{\mathrm{tr}}=4$ kW, 5 seeds | raw and +$\Pi$ | its own | reported **in-distribution** (its training configuration) **and zero-shot** (all other $(N,\rho_{\mathrm{tr}})$ of its profile); training steps and wall-clock reported. +$\Pi$: its bounded per-charger action is scaled to kW as $x_i=a_i^{\mathrm{GNN}}\cdot P^{\mathrm{cs}}\zeta_i$ and passed through the layer |
| 14 | GNN-DT [orfanoudakis2026graph] (DD35) | public code (github.com/StavrosOrf/DT4EVs, MIT, pinned commit) | – | – | optional; needs Gurobi-generated offline trajectories and its own modified EV2Gym copy (porting estimate 2-4 person-days). Run per profile at $N=20$ like EV-GNN if compute allows; otherwise the paper states this status accurately |
| 15 | Perfect-information oracles | soft-penalty LP $J^{\mathrm{pen,LP}}$ (main gap); hard lexicographic LP $J^{\mathrm{LP,ceil}}$ (cost-of-guarantee analysis) | – | – | Propositions O1, O2; not controllers |
| 16 | SA-Attn-CP($\alpha$) | as 1 with $\tau^{\mathrm{lo}}$ (Alg. 4) | yes | $r$ | trained at $\alpha=0.1$, evaluated at $\alpha\in\{0.01,0.05,0.1,0.2\}$; SA-Attn-point ($\hat Q=0$) as ablation; needs per-configuration calibration, so not zero-shot |
| 17 | In-distribution reference `attn_safe_trainN50` (DD36) | as 1, trained at $N=50$ (all profiles and ratings), 3 seeds | yes | $r$ | evaluated at $N\in\{50,100,200\}$; the transfer gap of SA-Attn is measured against it |
| 18 | Ablations of 1 | $\beta=0$; curriculum; small network; no admission (`_noadm`, evaluation pass also for 2) | yes | $r$ | – |
| 19 | Penalty ablations | penalty-dense; $w\in\{1,100\}$ (3 seeds) | no | $r^{\mathrm{pen}}$ | – |

## Alg. 6 Hyperparameters, tuning and statistics

| Group | Parameter | Default |
|---|---|---|
| Network | embedding $d_m$ / layers $n_L$ / heads $n_H$ / FFN | 128 / 3 (pre-LN) / 4 / 512, GELU |
|  | dropout | 0 |
|  | actor head / critic head | $d_m\to d_m/2\to2$ / $d_m\to d_m/2\to1$, after attention pooling |
|  | $\log\sigma$ clamp / init | $[-5,0.5]$ / $-0.5$ |
|  | actor / critic weights | separate encoders |
|  | precision | float32 networks, float64 safety layer |
| PPO | optimiser | Adam [kingma2015adam], $\epsilon=10^{-5}$ |
|  | learning rate | $3\times10^{-4}$, linear decay to 0 (tuned) |
|  | discount $\gamma$ | 0.99 (15 min); $0.99^{1/3}$ (5 min) |
|  | GAE $\lambda_{\mathrm{GAE}}$ / clip $\epsilon_{\mathrm{clip}}$ | 0.95 / 0.2 |
|  | entropy coefficient $c_{\mathrm{ent}}$ | 0.0 (tuned) |
|  | value coefficient / value clipping / max grad norm | 0.5 / off / 0.5 |
|  | target-KL early stop | 0.03 |
|  | parallel envs / rollout length | 16 (8 × $N=10$, 8 × $N=20$) / 128 steps per env |
|  | minibatch $n_{\mathrm{mb}}$ / epochs | 512 (tuned; in the grid) / 10 |
|  | env steps per run / seeds | $3\times10^6$ / 5 (3 for `trainN50` and the $w$ ablations) |
| Safety layer | $\beta$ / $P^{\mathrm{ref}}$ | 0.01 / 11 kW |
|  | $\varepsilon_q$ / $\varepsilon_{\mathrm{sat}}$ / $\epsilon_{\mathrm{num}}$ | 0.008 kW / 0.001 kW / $10^{-9}$ |
| Features | $E^{\mathrm{ref}}$, $c^{\mathrm{ref}}$, $H_f$ | 100 kWh, 0.25 €/kWh, 8 steps |
| Evaluation | $J^{\mathrm{pen}}$ weights | $w_E=w_P=10$ €/kWh, fixed for all methods |
|  | test episodes per configuration | 100 for $N\le50$, 50 for $N\ge100$ |
|  | action at test time | deterministic, $a=\mu_\theta$ |
| Conformal | $n_{\mathrm{cal}}$ / training episodes | 5,000 / 2,000 per configuration |

**Tuning protocol (DD37; equal budget for every learned method).**
- **Tuned methods:** `attn_safe`, `attn_penalty`, `attn_lagrangian`, `mlp_safe`, `mlp_penalty`, `deepsets_safe`, `attn_proj_nolb`.
- **Budget:** $K=12$ configurations each (the default point plus 11 sampled without replacement from the method's grid, with a fixed sampling seed per method), $0.5\times10^6$ env steps, seed 0.
- **Training data:** each method's own **training mixture** ($N\in\{10,20\}$, three profiles, $\rho_{\mathrm{tr}}\in\{4,7\}$), exactly as in final training.
- **Grids:**
  - common: lr $\in\{1,3,10\}\times10^{-4}$ × minibatch $\in\{256,512,1024\}$ (the default 512 is in the grid) × $c_{\mathrm{ent}}\in\{0,10^{-3},10^{-2}\}$;
  - penalty-reward methods additionally $w\in\{1,10,100\}$ €/kWh;
  - `attn_lagrangian` additionally $\eta_\Lambda\in\{0.5,2,8\}$.
- **Validation:** 24 episodes cycled over the training-mixture configurations, from a seed range disjoint from training and test replays.
- **Selection criterion:** mean validation $J^{\mathrm{pen}}$ at the fixed $w=10$ €/kWh for **every** method. A tuned penalty weight therefore cannot win by lowering its own penalty.
- **Inheritance:** conformal, ablation, dt5 and `trainN50` variants inherit from `attn_safe`; penalty-dense and the $w$ ablations inherit the PPO keys of `attn_penalty`.
- **Not tuned:** EV-GNN uses its published hyperparameters (DD27). $\beta$ and the architecture are fixed for all safety-layer methods.
- **Enforcement:** `reproduce.sh` refuses final training without a tuning result or an explicit `--hparams` file. Runs with the explicit defaults are labelled "untuned".

**Statistics (DD38).**
- **Unit of analysis = training seed.** Replay-level values are averaged within each seed first. 5 seeds; 3 for `trainN50` and the $w$ ablations.
- **Pre-declared primary endpoints:**
  - (E1) penalised-profit gap $\mathrm{gap}^{\mathrm{pen}}$ at $N=200$;
  - (E2) cost of the guarantee per $N$, in profit and in $J^{\mathrm{pen}}$;
  - (E3) transformer-violation energy;
  - (E4) A5-complete EV fraction.
- **Main tests:** paired $t$-tests on seed means (same replays for all methods), with stratified-bootstrap 95 % confidence intervals over seeds and replays.
- **Why not Wilcoxon on seeds:** a Wilcoxon signed-rank test with 5 pairs cannot reach $p<0.05$ two-sided (it needs $n\ge6$) [wilcoxon1945individual].
- **Secondary test:** Wilcoxon on replay-level differences of the seed-averaged policies, stated as episode-level only.
- **Multiplicity:** Holm correction across all comparisons of one results table (family = table).
- **Aggregate reporting:** IQM and probability of improvement with stratified-bootstrap intervals [agarwal2021deep]. Five independent seeds is the minimum recommended in [henderson2018deep]; if the intervals of a primary endpoint overlap, SA-Attn and Attention+penalty get 10 seeds.

## Alg. 7 Experiment scenarios: scaling across $N$ and arrival profiles (DD24, DD28, DD36, DD39)

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
3. Profiles are mixed in training, so the profile is **in-distribution**; no profile transfer is claimed (DD36). Results are reported per profile.

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

**What is zero-shot (DD36).**
- Zero-shot refers to the **station size $N$**: test $N\in\{5,50,100,200\}$ against training $N\in\{10,20\}$.
- It also refers to the **out-of-training tightness level** $\rho_{\mathrm{tr}}=3$ kW, a stress level.
- Profiles are not zero-shot.
- SA-Attn-CP is not zero-shot: it needs per-configuration calibration data.

**Main test grid (43 configurations, 3550 episodes per evaluated policy).**
- 6 sizes $N\in\{5,10,20,50,100,200\}$ × 2 tightness levels $\rho_{\mathrm{tr}}\in\{4,7\}$ × 3 profiles = 36 configurations.
- Stress level $\rho_{\mathrm{tr}}=3$ kW at $N\in\{20,200\}$ × 3 profiles = 6 configurations.
- The **literature configuration**: EV2Gym's `V2GProfitMax.yaml` (25 chargers, 100 kW, workplace, multiplier 5, start 05:00, loads/PV off), with our constant-efficiency spec file (Corollary 2; stated as a deviation).
- **Replays:** 100 for $N\le50$ and for the literature configuration, 50 for $N\in\{100,200\}$. The replays are generated once and shared by all methods.

**In-distribution reference (DD36).** `attn_safe_trainN50` (16 envs at $N=50$, all profiles and ratings, 3 seeds) is evaluated at $N\in\{50,100,200\}$, including $\rho_{\mathrm{tr}}=3$ at $N=200$. The transfer gap of SA-Attn (trained at 10/20) to this reference is reported separately from the gap to the oracle.

**Per-$N$ difficulty diagnostics** (review M4). Because per-charger quantities scale linearly with $N$, aggregate fluctuations shrink as roughly $1/\sqrt N$. For each $N$ we therefore report the fraction of steps with a binding forward limit, the A5-failure rate and the A5-complete EV fraction. This shows whether large-$N$ instances are easier or harder.

**Sensitivities (DD28, DD39).** These are run at $N=20$, $\rho_{\mathrm{tr}}=4$ kW, all three profiles (4 × 3 = 12 configurations, 100 replays each):
- desired SoC 0.8 (DD2);
- $\Delta t=5$ min ($T=336$, separately trained SA-Attn, 5 seeds, $\gamma=0.99^{1/3}$);
- throughput cost $c^{\mathrm{thr}}=0.05$ €/kWh, charged on $|\tilde p|$ in reward, profit, eMPC and oracle objectives;
- discharge price factor $f^{\mathrm{dis}}=0.9$.

The main learned policies are evaluated without retraining (except the 5-min policy), together with all baselines and the oracles. Main results remain EV2Gym-native (DD1), which is stated as a limitation, together with the share of profit from V2G discharging.

### Alg. 7.4 Compute estimate

**Measured.** EV2Gym step 1.9 ms ($N=20$) and 6.6 ms ($N=200$). For the wrapper and PPO timings, see version 4 of this section and `code_repo/code/README.md` Sec. 4 (2 cores: about 18 s per training update, about 7 h per $3\times10^6$-step run). At $N=200$: admission pre-pass 7-25 s per replay; oracle (hard, ceil and soft LPs) 1.5-8.5 s per replay.

**Core set (required for the main claims).**

| Item | Count | Estimate |
|---|---|---|
| Tuning: 7 methods × 12 configurations × $0.5\times10^6$ steps | 84 runs = $4.2\times10^7$ steps | about 7 GPU-h (30-40 CPU-node-h) |
| Training: 8 methods (`attn_safe`, `attn_penalty`, `attn_lagrangian`, `attn_proj_nolb`, `deepsets_safe`, `mlp_safe`, `mlp_penalty`, `attn_safe_conformal`) × 5 seeds × $3\times10^6$ | 40 runs = $1.2\times10^8$ steps | about 20 GPU-h (80-120 node-h) |
| Admission tables | 3550 replays | about 3 CPU-h |
| Conformal calibrations: 43 configurations × 4 $\alpha$ | 172 | about 12-18 CPU-h |
| Evaluation: about 89 policy passes × 3550 episodes (incl. `_noadm`, `safedeploy`, conformal) | about $3.2\times10^5$ episodes | about 60-100 CPU-h (eMPC-HiGHS dominates at $N=200$) |
| Oracles (hard, ceil, soft) | 3550 replays | about 3-5 CPU-h |
| **Core total** | | **about 27 GPU-h + 80-130 CPU-h** |

**Extended set** (ablations, $w$ ablations, `trainN50` reference, sensitivities, 24-h eMPC, Gurobi eMPC on the literature configuration, EV-GNN; GNN-DT only as a stub): about 36-46 GPU-h + 70-110 CPU-h.

**Note.** The code's `reproduce.sh` places the `trainN50` reference (DD36) and `empc_highs_24h` (DD33(7)) in the extended set, which is optional by default. Since the review asks for both, they should be run for the paper.
