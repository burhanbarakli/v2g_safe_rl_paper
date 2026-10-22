# Internal review, round 1 (A6, referee simulation for SAUCIS)

Reviewer: A6 (internal; did not write the paper). Date: 2026-10-22.
Object: `paper_repo/manuscript` (main.tex + sections/*.tex, Markdown mirror `manuscript.md`), `paper_repo/supplementary`, read against `lit/merged/gap_analysis.md` (R1-R12), `lit/merged/paper_matrix.csv`, `method/*.md`, `review/proof_check.md` and the code in `code_repo/code`.
Locations are given as `file:line` in `manuscript/sections/` (e.g. `05:36` = `05_experimental_setup.tex`, line 36) plus the section number.
Results are placeholders by design. I judge the planned design, not missing numbers. I did not re-check the proofs line by line: `review/proof_check.md` already did that and found no mathematical error. My comments on the theory are about what it means and how it is sold.

One fact check made for this review: the GNN-DT repository `github.com/StavrosOrf/DT4EVs` exists, is public under the MIT licence, and its README names the GNN-DT paper (checked 2026-10-22). This contradicts `05:40` (see M6).

---

## 1. Summary and recommendation

**Summary.**
The paper proposes SA-Attn, a V2G charging controller for a single-transformer charge-point-operator station. It combines a charger-token transformer encoder with a shared per-token Gaussian head and a closed-form Euclidean projection onto a box intersected with a two-sided sum constraint.
The per-EV lower bound $l_i$ of the box is a taper-aware latest-start bound derived from a reachable-energy operator. Theorem 1 shows by induction that, under A1-A5, the transformer limit and every departure requirement hold for any raw action. Steps on which aggregate feasibility (A5) fails are handled by a least-laxity-first fallback, admission control and an a-priori deficit bound.
A split-conformal lower bound on the departure step turns the departure guarantee into a marginal coverage statement.
The planned evaluation trains PPO on $N\in\{10,20\}$ chargers in EV2Gym and tests zero-shot up to $N=200$. Performance is reported as the gap to a hard-constrained, rounding-aware perfect-information LP, and the "cost of the guarantee" is the profit difference to a penalty-trained twin.

**Recommendation if the experiments turn out as planned: major revision.** It would be minor revision if M1, M3, M4 and M6 are closed before submission.

Reasons:
- **For acceptance.**
  - The paper is technically careful and honest about prior art. It says outright that the projection is the sLLF and quadratic-knapsack form (`04:136-141`).
  - The assumptions are explicit. The A1 necessity remark and the closed-loop nature of A5 are stated rather than hidden.
  - The oracle is hard-constrained and its relaxations are disclosed.
  - The evaluation protocol is well specified: shared replays, a reported tuning budget, seeds, rliable-style intervals and pinned commits.
  - A careful, self-critical safe-RL paper of this kind is above the usual SAUCIS level.
- **Against acceptance in the present form.**
  1. The novelty is a combination of known parts, and the closest work (R1, `orfanoudakis2026can`) is still unread. Table 1 carries a row of "unknown" cells.
  2. The "guarantee" is sold more strongly than Theorem 1 supports. The title, abstract and contribution list say "for any raw action", while A5 depends on the past raw actions.
  3. Several design choices make the comparisons asymmetric in favour of SA-Attn:
     - admission control lowers SA-Attn's own requirements;
     - the oracle bound does not cover controllers with shortfall;
     - the MLP is a strawman beyond $N=20$;
     - EV-GNN is trained out of distribution.
  4. Baselines that safe-RL and EV-scheduling reviewers will ask for are missing: PPO-Lagrangian, a transformer-only projection without $l_i$, a non-attention size-agnostic policy, deploy-only projection, a long-horizon receding LP and an in-distribution model trained at large $N$.
  5. The manuscript is far above the SAUCIS length guidance for experimental papers.

---

## 2. Major issues

### M1. Novelty is not yet establishable; positioning must be closed and recalibrated
**What.**
- The novelty sentence "To our knowledge, no published controller combines such bounds with a learned policy that runs at any station size and comes with a proof of multi-step feasibility" (`01:24`, repeated at `02:71`) is followed by a TODO (`01:25`).
- Table 1 has a row for `orfanoudakis2026can` marked "abstract only" with "unknown" in three columns (`02:93`).
- The text says "the abstract describes neither the mechanism nor a formal guarantee" (§2.2, `02:~38`). That is a GNN-TD3 with a safety layer meant to preserve charging requirements and power limits, from the EV2Gym group.
- R2 (`eland2026safe`), R3 (`song2025longterm`) and R4 (the published EV-GNN, "not in version read" in Table 1) are also open.
- Hedges such as "In all size-agnostic controllers that we could read in full" (`02:53`) and "in the version we read" (Yan et al., §2.2) tell the referee that the literature review is incomplete.

**Why it matters.**
- The paper itself says each ingredient exists (`01:40`):
  - sLLF gives the clip-around-a-common-level rule and the laxity feasibility characterisation;
  - HPC-RL (`yan2026hierarchical`) gives the same latest-start lower bound $P_{\min}=\max\{0,\ \text{remaining energy}-\text{remaining intervals}\times P\}$;
  - Wan et al. give a provably per-step-feasible projection for the transformer limit;
  - Tabas and Zhang give an invariance proof for a closed-form layer;
  - Sadeghianpourhamami et al. give train-small/test-large with an oracle gap;
  - GNN-DT, EV-GNN and HetGPS give zero-shot size transfer.
- The contribution therefore depends entirely on no one having done the combination. If R1 shows a projection with requirement-preserving bounds on a GNN policy, C1+C2 collapse, and the paper must follow the decision rule in gap analysis §(e).
- A likely reviewer is from the EV2Gym group and will know R1 in detail.
- The transfer range (10/20 to 200) is not larger than prior work: GNN-DT goes 25 to 100, HetGPS goes to 3,218 EVs, and EV-GNN reaches 1,000 charge points (scale, possibly with retraining). Scale should not be presented as a selling point.

**Fix.**
1. Obtain R1, R2 and R3 through the library and the published EV-GNN (R4). Rewrite their Table 1 rows from full text, remove "abstract only", "unknown" and "not in version read", and delete `01:25` and `07:63-64`.
2. Apply the gap-analysis decision rule explicitly.
3. Re-check `yan2026hierarchical` (arXiv v1, Aug 2026) and `arif2026constraint` for new versions (R5).
4. Recast the contributions:
   - **C1** "a charger-token policy as the vehicle for C2". This is not a contribution in itself; see M5 for the evidence it needs.
   - **C2** "the first proof that a learned, size-agnostic V2G policy keeps the transformer limit unconditionally and the departure requirement for every EV whose stay sees no aggregate-infeasible step", together with the measured frequency of such steps versus $N$.
   - **C3** "the measured cost of the guarantee against a hard-constrained optimum".
   - Contribution 3 as worded ("We define a zero-shot evaluation", `01:37`) is weak, because the protocol has a 2020 precedent. Make it an empirical finding instead.
5. Say in one sentence that Theorem 1's induction is the classical laxity argument of Chen et al. (their Prop. 1) extended to taper, V2G and a two-sided coupling. Reviewers will say this anyway; saying it first removes the attack.

### M2. The guarantee is oversold relative to Theorem 1
**What.**
- The abstract says "the paper proves that the transformer limit and every departure energy requirement hold at every step for any raw action" (`00:14`). The same wording appears in contribution 2 (`01:36`), the Theorem 1 prose "whatever the raw action" and "The theorem holds for any raw action" (`04:202`, `04:216`), the conclusion (`08:8`) and the title ("Constraint-Guaranteed").
- But A5 is a closed-loop condition. The paper's own two-EV example (`04:225-227`) shows that some raw-action sequences cause A5 to fail.
- So "for any raw action" is true only per step and conditional on A5 holding along the trajectory that these very actions produce. It is false as a statement about raw-action sequences.
- In addition, a departure requirement does not "hold at every step"; it holds at departure.

**Why it matters.**
- A safe-RL referee will read "for any raw action" as "for any policy" and then find A5. That reads as a bait-and-switch, and the guarantee looks weaker than it is.
- The paper actually has a stronger, cleaner result that it under-sells. By Corollary 1 and Lemma S3, the **transformer limit holds at every step for every raw action sequence**, fallback steps of case (iii) included. The only exceptions are cases (i), (ii) and (iv), which the paper argues cannot occur in its scenarios (`04:~245`).
- Only the departure part is conditional.
- The fallback-free policy-independent condition is so conservative (full power for all constrained EVs, S1.5) that the departure guarantee is in practice an empirical quantity.
- In addition, the departure "guarantee" is relative to a requirement that the controller itself may lower through admission control (see M3). A referee can call that a guarantee by redefinition.

**Fix.**
1. Restructure the result into two statements:
   - **Theorem 1(a)**, unconditional: the transformer limit holds for every raw action sequence under A1 and A3 and the scenario conditions that exclude cases (i), (ii) and (iv), which should be stated as an assumption.
   - **Theorem 1(b)**, per-EV and conditional: the departure requirement of EV $i$ holds whenever $p_i\ge l_i$ throughout its stay. In particular it holds when A5 holds on every step of that stay.
2. Rewrite the abstract, contribution 2 and the conclusion accordingly. Replace "every departure energy requirement hold at every step for any raw action" with something like "the transformer limit holds at every step for any raw action, and the departure requirement of every EV whose stay contains no aggregate-infeasible step is met".
3. Change the title, e.g. "... Scale-Agnostic V2G Charging Control with Provable Transformer and Departure Constraints". Alternatively keep "Safe" and drop "Constraint-Guaranteed".
4. Make "fraction of EVs whose stay is A5-complete" and "fraction of steps in fallback, by case, versus $N$" primary metrics in Table 2 and Figure 4. They are the empirical half of the guarantee.
5. Add one sentence on what A3 means outside the simulator. On real hardware an EV's battery management system may draw less than commanded, and A3 then fails. Suggest a margin on $\hat B$ or a measured-energy feedback as the deployment remedy (see Q7).

### M3. Admission control and the oracle make service and profit comparisons asymmetric
**What.**
1. **Admission control.**
   - SA-Attn runs with admission control, which "lowers the request of the new EV" (`04:250-256`). Penalty baselines run without it (`supp S2.5`; `configs/train_attn_penalty.yaml`: `admission: false`).
   - A lowered request means SA-Attn buys less energy, which directly raises its profit and lowers its measured shortfall when that shortfall is computed against the admitted requirement. The code records both, `dep_short` (admitted) and `dep_short_dd2` (capped), in `v2gsafe/eval/metrics.py:65-66`, but the manuscript does not say which one is reported.
   - The cost of the guarantee, "the profit of Attention+penalty minus the profit of SA-Attn" (`05:51`), is therefore biased in favour of SA-Attn.
2. **Oracle coverage.**
   - Proposition S1 bounds only controllers with total shortfall $\le z^*$ against the *capped* $E^{\rm req}$ (`oracle.py:3`: "DD2-capped value ... without admission").
   - Any method with shortfall falls outside the bound, and its gap can be negative or meaningless. That includes SA-Attn on fallback steps or with admission reductions, all penalty methods, EV-GNN and the heuristics.
   - The paper flags only transformer violators (`03:103`) and says "the reported gap overestimates the true gap". That is true only for covered controllers.

**Why it matters.** The two headline numbers, the gap and the cost of the guarantee, are not comparable across methods that deliver different energy. A referee who notices that SA-Attn may lower its own targets will discount both.

**Fix.**
- **(a) Report shortfall against one common reference for all methods.** Use the capped $E^{\rm req}$ and also $E^{\rm des}$, and count admission reductions as undelivered energy in the main service metric. Report the admitted-requirement shortfall only as a secondary diagnostic.
- **(b) Run the same admission control for every method,** including penalty, heuristics, MPC and EV-GNN, so that the admitted targets are the same function of the state. Alternatively, run SA-Attn without admission control as the main variant and keep admission control as an add-on. The ablation "remove admission control" (`05:43`) should then be promoted to the main table.
- **(c) Make the gap service-consistent.** For each method and episode, re-solve the oracle with $E^{\rm req}_i$ replaced by $\min\{E^{\rm req}_i, E_i^{\pi}(d_i+1)\}$, i.e. the "oracle at equal service". Report this as the scheduling gap, next to the gap to the full-service oracle. Alternatively, report a service-adjusted profit $J^\pi - w\cdot\text{shortfall}$ for several values of $w$ (e.g. the penalty weights 1, 10 and 100 €/kWh), so the cost of the guarantee becomes a curve rather than one number.
- **(d) Report the number of excluded episodes** ($|J^{\rm LP,ceil}|<1$ €). Use the ratio of sums alongside the mean of ratios, because a relative gap with a small denominator is heavy-tailed.

### M4. Zero-shot claim validity: the test distribution shift is mild by construction, and one claim is false
**What.**
- "Every exogenous quantity scales linearly with $N$" (`05:13`): the rating $\rho_{\rm tr}N$, load and PV at $0.4P_{\rm tr}$, and the spawn multiplier per charger. All global features are ratios (`04:~27`).
- At larger $N$ the per-charger statistics are therefore identical, and the aggregate fluctuates relatively less, as $1/\sqrt N$. The transformer constraint may then bind *less* often at $N=200$ than at $N=10$.
- An improving or flat gap with $N$ would then reflect statistical multiplexing, not transfer ability.
- "The policy does not observe the profile, so the profile forms a further zero-shot dimension" (`05:19`) is incorrect. Training mixes all three profiles (`05:21`), so profile is in-distribution.
- Conformal calibration is repeated per test configuration with 5,000 sessions drawn from that configuration (`04:264`, `04:281`). SA-Attn-CP is therefore not zero-shot.

**Why it matters.** The zero-shot result is one of three contributions and supports the "scale-agnostic" title. A referee will ask what actually shifts between train and test.

**Fix.**
1. Delete the profile zero-shot claim, or make it true with a leave-one-profile-out run: train on two profiles, test on the third.
2. Add at least one test axis where per-charger quantities change with $N$:
   - (i) held-out ratings, e.g. $\rho_{\rm tr}\in\{3,5.5\}$ kW;
   - (ii) a sub-linear rating $P_{\rm tr}\propto N^{0.8}$ or a fixed transformer with growing $N$ (diversity factor), which is the realistic case for station expansion;
   - (iii) a held-out occupancy, e.g. a spawn multiplier ×1.5.
3. Report per $N$ the fraction of steps on which the forward limit binds and the A5-failure rate. This shows whether large-$N$ instances are harder or easier.
4. Add an in-distribution reference: SA-Attn trained at $N=100$ or $200$ (fewer seeds are acceptable). The transfer loss can then be separated from method loss. This is the "larger-$N$ training" baseline every reviewer of a zero-shot paper asks for.
5. State that SA-Attn-CP requires per-configuration calibration data. Alternatively, adopt the $N$-free calibration scheme of `proof_check.md` m5.2: a random port, then a random session of that port. That scheme makes CP transferable across $N$ and also removes 5,000 calibration episodes per configuration.

### M5. Missing baselines and ablations that reviewers will request
The ablation list (`05:43`: no admission control, $\beta=0$, curriculum, smaller network) tests none of the paper's claims. Requested additions, in priority order:

1. **Transformer-only projection (no $l_i$, i.e. $l_i=p^{\min}_i$).**
   - This is the per-step layer of Wan et al., the closest provable prior.
   - It isolates the value of the multi-step lower bound, which is the core of C2.
   - Without it, the reader cannot tell whether $l_i$ costs or gains profit.
2. **Deploy-only projection.**
   - Train with the penalty, or unconstrained, and project at test time only.
   - The paper stresses that the layer acts "during training and deployment" (`04:139-141`, `00:14`). This is the ablation that shows whether training-time use matters.
   - The checkpoints already exist: Attention+penalty plus the layer at test time.
3. **PPO-Lagrangian** (or RCPO `tessler2018reward`) with cost limit $\approx0$ on shortfall and overload.
   - A fixed penalty weight is the weakest constrained-RL baseline.
   - The penalty weights are not tuned (`supp S3.2`: "Penalty weights ... are fixed a priori"), and the sensitivity runs at 1 and 100 €/kWh use only 3 seeds.
   - A CPO or CVPO variant would match `eland2026safe`. At least one Lagrangian method is expected by any safe-RL referee.
4. **A non-attention size-agnostic policy.**
   - Deep Sets / mean-pooled context, or a per-token MLP that sees its own token plus the global token, with no inter-charger interaction.
   - Because the projection already performs the coupling, it is plausible that attention adds little. This ablation is the only evidence for C1.
5. **Replace or retire the zero-padded MLP beyond $N=20$.**
   - "zero-padded to 200 slots" (`05:30`): weights attached to slots 21-200 receive zero gradient during training, because those inputs are always zero, so at $N\ge50$ they are at initialisation.
   - MLP+safety layer at $N=200$ is "safe but random", a strawman that reviewers will dismiss.
   - Keep the MLP only at $N\le20$ and on the 25-charger literature configuration, and use item 4 as the size-agnostic baseline.
6. **A price-aware heuristic plus the layer.**
   - AFAP, ALAP, round robin and LLF ignore prices, so they show only that RL learns price timing.
   - Add a threshold rule (charge or discharge by price quantile within the known day-ahead window) passed through the same projection.
   - Add sLLF itself (`chen2022smoothed`), which is the canonical laxity scheduler and the source of the projection form.
7. **A receding-horizon LP with known day-ahead prices, connected EVs only and horizon to the last departure,** plus the layer.
   - At $N=200$ HiGHS solves this in well under a second.
   - It is the natural "why RL?" baseline (ACN `lee2021adaptive` runs exactly this kind of controller in practice).
   - The present MPC has a 10-step (2.5 h) horizon and true arrivals (`05:36`). That is both an information advantage and an arbitrary handicap.
   - Make the HiGHS formulation the single MPC at all $N$. Validate it against the EV2Gym eMPC at $N\le25$, rather than switching implementations at $N=25$ because of a "commercial solver licence" (`05:37`). Gurobi academic licences are free, and the licence argument will not convince.
8. **Optional ablations.**
   - The dense penalty variant, which is described in `supp S2.5` as an ablation but missing from `05:43`. Align the text.
   - A log-$N$ attention temperature scaling as a cheap fix if attention dilution at $N=200$ shows up (`04:43`).
   - $H_f$ larger than 8. Day-ahead prices are known 12-36 h ahead, so restricting the policy to 2 h of prices (`04:24`) is unexplained and handicaps both RL and the comparison with MPC.

### M6. Baseline protocols: EV-GNN, GNN-DT and the MPC
**What.**
- **EV-GNN** (`05:38-39`) is trained with TD3 on a single profile at $N=20$, $\rho_{\rm tr}=4$ kW and evaluated on both ratings. SA-Attn trains on both ratings, both sizes and all profiles.
  - Every $\rho_{\rm tr}=7$ kW and $N=10$ result is therefore out of distribution for EV-GNN only.
  - The EV-GNN training budget (steps) is not stated.
  - Table 2 lists "EV-GNN / +safety layer" (`06:35`), but §5.2 never describes EV-GNN with the layer, including how its bounded TD3 action maps to $x$.
- **GNN-DT** is "not included, because we found no public code for it" (`05:40`). This is factually wrong: `github.com/StavrosOrf/DT4EVs` is public (MIT) and its README is the GNN-DT paper. Our own gap analysis even lists it.
- **The MPC** has the issues described in M5.7.

**Why it matters.**
- An EV2Gym-group reviewer will spot the GNN-DT statement immediately, and it undermines trust.
- The EV-GNN asymmetry makes any large-$N$ win over EV-GNN uninterpretable.

**Fix.**
1. **EV-GNN.**
   - Write a thin scenario sampler so EV-GNN trains on the same mixture as SA-Attn.
   - Or, symmetrically, also train SA-Attn under the EV-GNN protocol (single profile, $N=20$, $\rho=4$) and compare like with like.
   - Report EV-GNN steps and wall-clock time, and describe "EV-GNN + layer".
2. **GNN-DT.**
   - Either run GNN-DT from DT4EVs on the 25-charger literature configuration and at least one zero-shot size, or state a valid reason for exclusion, e.g. offline data generation cost or its own oracle-trajectory requirement.
   - Correct `05:40` in any case.

### M7. Tuning budget and statistical plan
**What.**
- Tuning uses 18 configurations, each at $0.5\times10^6$ steps, with **seed 0 only**, on the **tight 10-charger station** (`05:56`, `supp S3.2`).
- The final training, however, uses a mixture of $N$, profiles and ratings at $3\times10^6$ steps.
- The penalty weights, the $\beta$ value and the architecture are not tuned. This favours the method whose design was iterated by the authors.
- Statistics: "Wilcoxon signed-rank test on per-episode profit and shortfall, with Holm correction across $N$" (`05:62`). It is not stated how the 5 seeds enter.
  - Pooling seeds × episodes is pseudo-replication, because training seeds are the unit of algorithmic variability.
  - Averaging over seeds tests only episode variability.
  - Holm "across $N$" ignores the further multiplicity over profiles, ratings, methods and metrics.
- Five seeds is the minimum the cited guidance allows.

**Why it matters.** Without these details, small profit differences, which is what the cost of the guarantee is likely to be, cannot be interpreted.

**Fix.**
1. Tune on the training mixture, or at least on validation episodes from the mixture.
2. Use 2 seeds per configuration, or a successive-halving scheme with the same total steps.
3. Include the penalty weight $w\in\{1,10,100\}$ in the penalty methods' grid, and $\beta$ in SA-Attn's grid, so the budgets are symmetric.
4. Pre-declare the primary endpoints:
   - (E1) gap at $N=200$;
   - (E2) cost of the guarantee per $N$;
   - (E3) violation energy;
   - (E4) A5-complete EV fraction.
5. Test the pre-declared endpoints with a hierarchical scheme: seed-level aggregates with a stratified bootstrap over seeds and configurations (rliable), plus a paired episode-level test on seed-averaged values, stated as such.
6. Apply Holm over the full family of primary comparisons.
7. Use 10 seeds for SA-Attn and Attention+penalty if compute allows. This costs about 10 extra runs.

### M8. The oracle relaxation can dominate the gap
**What.**
- The LP relaxes the taper to the nominal power (`03:92`), while the desired energy is the full battery (`05:10`). With full-battery targets almost every EV ends in the constant-voltage region, so the relaxation is large and $N$-independent.
- The LP can postpone the last kWh into cheap slots that are physically unreachable.
- Yet the EV2Gym taper is exactly piecewise linear in $E$ above the transition SoC: $\bar P(E)=E^{\rm cap}(1-\varsigma)(1-e^{-\kappa})/\Delta t$ from Eq. S1. That is a concave piecewise-linear constraint that an LP handles exactly. The admission LP (`supp S2.4`) already uses an inner linear taper.
- In addition, $J^{\rm LP,ceil}$ corrects for a simulator artefact (unbilled 0.01 kWh rounding). A referee may find this odd and want to see its size.

**Fix.**
1. Add an outer (relaxing) linear taper $p_i(t)\le\min\{\bar P^{\rm nom}_i,\ a_i+b_i(E^{\rm cap}_i-E_i(t))\}$. Choose it so it upper-bounds the true $\bar P_i$; it keeps the LP a valid upper bound and tightens it substantially.
2. Report the gap to both the relaxed and the tightened oracle, and the size of the rounding correction $J^{\rm LP,ceil}-J^{\rm LP}$ in the main text (one number per $N$).
3. Decompose the gap into an information part (oracle minus long-horizon receding LP, M5.7) and a policy part (receding LP minus SA-Attn). Only the policy part reflects the method.

### M9. Economic realism of V2G inflates the profit the guarantee is "costing"
**What.**
- Discharge price equals charge price (`05:10`), and the experiments use lossless accounting ($\eta=1$, `03:~40`).
- There is no degradation cost.
- The result is that any price spread pays for battery cycling.

**Why it matters.**
- The cost of the guarantee is measured against arbitrage profit that would not exist with round-trip losses (about 10-15%), grid fees or degradation (about 0.03-0.10 €/kWh throughput).
- V2G referees will flag this as the first thing.

**Fix.**
- Add a sensitivity run with $c^{\rm dis}=c^{\rm ch}-\delta$, e.g. $\delta=0.05$ €/kWh as a degradation or network fee. It can be added in the wrapper's reward and in the LP objective without patching EV2Gym.
- Report the share of profit from discharging. This also tests whether the theory's lossy $g_i$ (`03:~38`) matters, which currently "the experiments do not exercise" (`07:~45`).

### M10. Length and fit to SAUCIS
**What.**
- The PDF has 23 pages in the 10-pt article fallback and about 14,000 words including back matter.
- SAUCIS guidance for experimental papers is ≤25 *double-spaced* pages (roughly 7,000-8,000 words) and ≤7 figures and tables (`saucis_journal_requirements.md`).
- The main text has Tables 1-2, Figures 1-5 and Algorithm 1, which is at or above the limit.
- The journal's scope list names no energy topics. RL and attention fit, but the power-systems and OR density, with 5 assumptions, 2 propositions, 1 lemma, 1 theorem, 1 corollary and 22 equations, is heavy for a CS-generalist readership.

**Fix.** Cut the main text by about 35%:
- Move Proposition 1(iii) to the supplement. The Jacobian is not used by the method: PPO treats the projection as part of the environment, and the regulariser uses Moreau with stop-gradient (`04:~285-290`).
- Move the rounding-aware oracle details (`03:94-102`) and the $\varepsilon_q$ discussion to the supplement.
- Shorten §2 to one table plus about 1 page.
- Merge Figure 2(a) and 2(b) into the architecture figure, or move them to the supplement.
- Compress §4.6 into half a page.
- Add a two-sentence "intuition" paragraph before Theorem 1 for non-specialist readers.
- Confirm the build with the real `saucis.cls` (currently `\IfFileExists{saucis.cls}` falls back to `article`), and count pages there.

### M11. Compliance with the SAUCIS AI policy (submission risk, not a science issue)
**What.**
- SAUCIS states that AI-generated "substantial sections" (abstract, introduction, literature review, discussion) are "inappropriate" and must be preliminary drafts thoroughly rewritten by the authors.
- The current text was drafted by agents, and the AI statement (`10:~35-43`) asserts that it was "thoroughly reviewed, rewritten and verified by the authors".

**Fix.**
- The human authors must actually rewrite the abstract, introduction, related work and discussion in their own words before the statement is true.
- They must write all result interpretation (the TODOs in §6-7 already require this).
- Repeat the disclosure in Methods or Acknowledgements, as the policy requires.
- Otherwise the statement itself becomes a misrepresentation risk.

---

## 3. Minor issues

- **m1** `00:14`, `08:8`: "every departure energy requirement hold at every step". A requirement is met *at departure*; see M2.
- **m2** `05:32`: "more than eleven times the highest hourly price ..., so no violation can pay". Write "so that no violation pays at the optimum of the penalised objective". Learned policies can still violate, which is the point of the paper.
- **m3** The "cost of the guarantee" sign convention ("A positive cost means ...", `06:~72`) should be defined once in §5.3 and in the caption of Figure 5.
- **m4** Notation clashes:
  - $\mathcal S(\lambda)$ (projection) and $S_j$ (conformal scores);
  - $\rho_{\rm tr}$ (rating per charger) and $\rho_i$ (taper slope, S2.4);
  - $B(k)$ (thresholds) and $B_{\rm mb}$ (minibatch);
  - $\chi$ (laxity) is fine, but $\varpi$ for permutations is unusual.
  - Add a nomenclature table to the supplement (`method/nomenclature.md` exists).
- **m5** §4.1: list the $F_c=12$ and $F_g=38$ features in a supplementary table with their normalisations. The text gives them only in prose, and reproducibility needs the exact list.
- **m6** §3.7: say plainly that the problem is a POMDP and that PPO with a feed-forward critic on the current observation is used. "Following standard practice" is vague.
- **m7** Eq. 9: define $J^\pi$ as the undivided episode profit in € (it is defined only implicitly at `03:~122`).
- **m8** Table 1:
  - The GNN-DT entry "reward within 5% of oracle" refers to an oracle with soft penalties. Say so.
  - The "This work" guarantee cell should read "transformer: unconditional (A1, A3); departure: conditional on A5" (M2).
  - "Gap to optimum" for EV-GNN, "oracle or MPC baseline", is ambiguous.
- **m9** Table 2: the columns are truncated in the mirror ("Gap [", `06`). Check the LaTeX header units and add a column "A5-complete EVs [%]" (M2).
- **m10** Figure paths `../../analysis/figures/...` point outside the manuscript tree. Copy the figures into `manuscript/figures` for the LaTeX-source upload.
- **m11** Figure 2 caption: about 120 words. Shorten it and move the explanation to the text.
- **m12** §5.1: "with one constant efficiency per model". Give the values, or point to the spec file in the supplement.
- **m13** §5.2: the heuristics "with its output passed through the safety layer". Say whether admission control is applied to them (M3).
- **m14** §5.4: "plus the tuning runs and the optional EV-GNN and 5-min runs". Either a baseline is in the paper or it is not, so remove "optional". Add hardware and wall-clock time (TODO `05:~65`).
- **m15** §4.7: "the score-function gradient is unbiased". Qualify this as the vanilla PG and PPO surrogate gradient at $\theta_{\rm old}$ (proof check m6.1). Clipping and GAE are biased anyway.
- **m16** §4.3: "the implementation checks the aggregate limit after each projection ... safe bisection". Mention in one clause that this guards the floating-point failure found in the proof check (M1 there). Reviewers like seeing that the guarantee survives finite precision.
- **m17** Abstract: "Performance is reported as the optimality gap ... and the cost of the guarantee as the profit difference". This is procedural. Replace it with the result sentence once available and keep the abstract under 250 words.
- **m18** Keywords: alphabetical and 5 items, compliant. Consider "Safety layer" instead of "Attention mechanism".
- **m19** References:
  - Several published works are cited as arXiv preprints: Set Transformer (ICML 2019), GAE (ICLR 2016), Adam (ICLR 2015), CleanRL (JMLR 2022).
  - Some years follow the online-first date rather than the issue: Kiwiel (Math. Program. 112, 2008, bib says 2006), Wang et al. (Optimization 71, 2022, bib 2021), Huangfu and Hall (MPC 10, 2018, bib 2017).
  - Cite the archival versions (IEEE style).
  - `markgraf2025safe` and `yan2026hierarchical` are arXiv only. Mark them as preprints in the text where they carry an argument.
- **m20** §2.3: "aggregation gave the first size-independent states". "First" claims are risky. Use "early".
- **m21** §3.6: "which never pays when $c^{\rm dis}\le c^{\rm ch}$ and is therefore exact in our scenarios". With $\eta=1$ and $c^{\rm dis}=c^{\rm ch}$, simultaneous charging and discharging is cost- and energy-neutral, so it is harmless. With $\eta<1$ (M9 sensitivity) and negative prices, however, it lets the LP "burn" energy for money, which no real charger can do. Then the relaxation is no longer exact. Add a complementarity argument or a check.
- **m22** §6.4: "with penalty weights of 1 and 100 €/kWh" uses 3 seeds (`supp S3.3`), while the main runs use 5. Harmonise, or state it in the text.
- **m23** Conclusion: "using the known form of a common shift". Fine, but the conclusion repeats the method. Spend the space on limitations that matter to a CPO.
- **m24** Supplementary S2.1 TODO (agreement of Eq. S1 with EV2Gym) and Lemma S6 "proof sketch". A referee may ask for the full proof of A3 because the guarantee depends on it. Provide it in the supplement.

---

## 4. Likely referee questions and suggested answers

1. **"Isn't the safety layer just sLLF applied to an RL action, and Theorem 1 just Chen et al.'s laxity argument?"**
   Yes for the algebra, and we say so (§4.3). New are:
   - the taper-aware certified thresholds, for which A1 is necessary (our STEEP counterexample);
   - the two-sided V2G coupling;
   - the explicit separation of an unconditional transformer guarantee from a conditional, per-EV departure guarantee;
   - the use of the layer in training of a size-agnostic learned policy;
   - the measurement of how often the conditional part fails versus $N$, and what the guarantee costs.
   Prepare a one-paragraph "what is new" box.
2. **"How is your work different from [orfanoudakis2026can]?"**
   This needs the full text (M1). Prepare a row-by-row comparison: layer equations, proof, size transfer, oracle.
3. **"Why RL at all, when day-ahead prices are known and an LP solves N=200 in milliseconds?"**
   Use the receding-horizon LP baseline (M5.7). If RL is close to or better than it without arrival forecasts, argue from inference cost, model-freeness and robustness to forecast error. If the LP wins, reframe: the paper shows how to make RL safe and scalable, and the layer can wrap any controller, MPC included. Do not hide this; reviewers will ask.
4. **"A5 can fail. How often, and is the guarantee then vacuous?"**
   Report the A5-failure step rate and the A5-complete EV fraction per $N$, profile and rating, with and without admission control. Report measured shortfall against the a-priori bound of Eq. 20. Stress that the transformer limit never depends on A5.
5. **"Admission control lowers requirements. Is that not cheating the departure guarantee?"**
   Report shortfall against the capped and the desired requirement for all methods, and give admission control to all methods (M3).
6. **"Why does training-time safety matter if training is in simulation?"**
   - The train and deploy action distributions coincide.
   - No penalty weights need tuning.
   - Safe online fine-tuning on a real station becomes possible.
   - Show the deploy-only ablation (M5.2) to quantify it.
7. **"A3 assumes the EV draws at least what you command. Real BMSs do not."**
   Discuss robustness: a delivery shortfall $\delta$ per step translates into at most $\tau\delta$ of departure deficit by the same telescoping as Lemma S4. A safety margin on $\hat B$, or re-computing $l_i$ from measured energy each step (already the case), absorbs a bounded under-delivery. Optionally add a test with random under-delivery.
8. **"Why PPO and not SAC or TD3 like EV-GNN?"**
   - With projection in the environment, the stochastic Gaussian likelihood-ratio gradient is unbiased (Markgraf et al.).
   - On-policy updates handle mixed-$N$ padded batches simply.
   - Deterministic actor-critics need a gradient through the projection, which is zero on active coordinates (Kasaura et al.).
9. **"Does attention matter, given the projection does the coupling?"**
   Answer with the Deep Sets / per-token ablation (M5.4).
10. **"Why only 5 seeds and 50 episodes at N≥100?"**
    The seed-level bootstrap intervals and an a-priori power estimate from pilot runs justify the numbers. Add seeds if the intervals overlap at the primary endpoints.
11. **"How does attention behave at N=200 with only 10-20 keys in training?"**
    Show attention entropy versus $N$. If dilution appears, apply the log-$N$ temperature fix as an ablation.
12. **"Is the conformal guarantee useful if it holds only marginally and needs per-station calibration?"**
    - It is the standard distribution-free guarantee.
    - Report the sub-group coverage (e.g. by arrival hour) to show where it is weak.
    - Use the $N$-free port-level calibration so one calibration set serves all station sizes with the same per-port law.
13. **"Results only in a simulator with synthetic arrivals."**
    EV2Gym's arrival model is fitted to ElaadNL data. Optionally replay ACN-Data sessions through the same wrapper for one configuration.
14. **"What is the computational overhead at N=200 and at N=1000?"**
    Report Table S8. Attention is $O(N^2)$ per layer and the layer $O(N\log N)$. Say what happens at 1,000 chargers, at least by a timing extrapolation.

---

## 5. The five changes that would most increase acceptance probability

1. **Close R1-R4 and recalibrate the claims** (M1). Read `orfanoudakis2026can`, `eland2026safe`, `song2025longterm` and the published EV-GNN in full. Remove every "abstract only", "unknown" and "could read in full". Correct the GNN-DT code statement.
2. **Restate the guarantee honestly and more strongly** (M2). Present the transformer limit as unconditional and the departure requirement as per-EV conditional. Retitle and rewrite the abstract and contributions accordingly. Make the A5-complete EV fraction a headline metric.
3. **Make the comparisons symmetric** (M3, M6, M7):
   - common shortfall reference and admission control for all methods (or none);
   - a service-consistent oracle gap;
   - EV-GNN trained on the same mixture;
   - symmetric tuning, including the penalty weights;
   - a hierarchical statistical plan with pre-declared endpoints.
4. **Add the four decisive baselines** (M5):
   - transformer-only projection (no $l_i$);
   - deploy-only projection;
   - PPO-Lagrangian;
   - a Deep Sets / per-token size-agnostic policy;
   plus a long-horizon receding LP and an SA-Attn model trained at large $N$. Retire the zero-padded MLP beyond $N=20$.
5. **Strengthen the zero-shot test and cut the length** (M4, M10). Add held-out or sub-linear ratings and a leave-one-profile-out run, and drop the false profile claim. Bring the main text to about 8,000 words and ≤7 figures and tables in the official `saucis.cls`, moving Proposition 1(iii), the rounding-aware oracle details and most of §4.6 to the supplement.
