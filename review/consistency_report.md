# Consistency report, round 1 (A6: style, consistency and internal review)

Date: 2026-10-22. Reviewer: A6. I did not write the manuscript and I did not edit it. This file only reports.

Inputs checked:
- Manuscript: `paper_repo/manuscript/` (main.tex, preamble.tex, sections/*.tex, refs.bib, main.pdf, main_anonymous.pdf, .log, .blg, .bbl).
- Supplementary material: `paper_repo/supplementary/` (supplementary.tex, s1-s4, PDFs, .log, .blg).
- Markdown mirrors: `paper_repo/manuscript.md`, `paper_repo/supplementary.md`. Both are in sync with the .tex files (spot-checked on 9 recent phrases).
- Method docs: `method/*.md`.
- Code: `code_repo/code/`, covering configs, `v2gsafe/*`, the README and the notes.
- Analysis: `analysis_repo/` (README, scripts).
- Literature: `lit/merged/*`.
- Style rules: `style/*`.
- Journal rules: `journal/journal_requirements.md`.

Line numbers refer to the individual .tex files.

## 0. Summary

| Severity | Count |
|---|---|
| Blocking | 3 |
| Major | 14 |
| Minor | 48 |

Severities:
- **Blocking** means the issue must be resolved before the experiments are launched or before submission, because otherwise a false statement or a policy breach ends up in the paper.
- **Major** means a reviewer or reader would be misled, or a placeholder would be filled from an output that does not exist or does not match the text.
- **Minor** covers wording, notation and style.

What is clean:
- **References and cross-references:**
  - Every `\ref`, `\eqref` and `\cite` resolves. There are no undefined references and no BibTeX warnings in `main.log`, `main.blg`, `supplementary.log` or `supplementary.blg`, and there is no "??" in either PDF.
  - All 76 cited keys are in refs.bib, `lit/merged/references.bib` and `verified_refs.json` (status `verified`), and every refs.bib entry is cited.
- **Results:** no result number and no result interpretation appears outside `\RES{}` placeholders. Section 2 lists the few measured scenario statistics.
- **Numbers:** every hyperparameter, scenario parameter and parameter count checked in Section 1 matches the method docs and the code. The exceptions are listed in Section 1.
- **Journal format:**
  - The abstract has 339 words including the placeholder, below the limit of 400.
  - There are 5 keywords, in alphabetical order.
  - Table captions are above the tables and figure captions below the figures.
  - Cross-references use the "Figure n" / "Equation n" / "Table n" forms through macros.
  - All Article Information Form items are present.
  - The anonymised PDFs show no names, affiliation or link, and the PDF Author field is empty.

---

## 1. Numbers: manuscript and supplementary vs method docs vs code

### 1.1 Values that match (verified in all three places)

| Quantity | Paper location | Method doc | Code | Status |
|---|---|---|---|---|
| EV2Gym commit `55ef368c` | 05:7, s2:5, Tab. S1, AIF:42 | DD header | `utils/__init__.py:19`, README:51 | OK |
| EV-GNN commit `01aa1b7a` | 05:38, s2:140, AIF:42 | DD27 | `third_party/EVGNN_COMMIT`, `evgnn_adapter.md:5` | OK |
| $\rho_{tr}\in\{4,7\}$ kW; 4 kW = 100 kW/25 | 05:14, Tab. S1 | DD24 | `base.yaml:41`, `test_grid.yaml:11,19` | OK |
| Load/PV multiplier 0.4 | 05:15, Tab. S1 | DD24, Alg. 7 | `base.yaml:47-48`, `scenarios.py:90-92` | OK (but see m15) |
| Start hours 05:00/12:00/12:00; spawn multipliers 5/4/2.5; weekdays/both/both | 05:18, Tab. S1, s2:80 | DD24 | `scenarios.py:42-44` | OK |
| Occupancy about 0.21; energy within about 5% | 05:18, s2:81 | DD24 (0.208/0.224/0.214; 9.5/10.5/9.9 kWh) | profile_check_results | OK |
| Dropped spawns 6.8% / 3.7% | s2:80 | DONE_method open issue 10 | profile_check | OK (measured, see Section 2) |
| $\Delta t$=15 min, $T$=112 (28 h); 5 min, $T$=336 | 05:18,25, Tab. S1 | DD13, DD24, DD28 | `base.yaml:42-43`, `scenarios.py:6`, `train_attn_safe_dt5.yaml` | OK |
| $\gamma$=0.99; $0.99^{1/3}$=0.996655 | 03:126, 05:25, Tab. S2 | Alg. 6 | `base.yaml:20`, `train_attn_safe_dt5.yaml:5` | OK |
| Test grid of 36 configurations; 100/50 replays; +100 literature; 3,100 episodes | 05:22-23, Tab. S1 | Alg. 7.3 | `test_grid.yaml` | OK |
| Training $N\in\{10,20\}$, 16 envs (8+8), rollout 128 | 04:301, Alg. S1, Tab. S2 | Alg. 2/6 | `base.yaml:14,30-32` | OK |
| $3\times10^6$ steps, 5 seeds | 05:57, Tab. S2 | Alg. 6 | `base.yaml:13`, `pipeline.py:332` | OK |
| Tuning: 18 configurations, $0.5\times10^6$ steps, seed 0, 20 validation episodes | 05:56, s3:35-36 | Alg. 6 | `tuning.py:182-184` | Grid OK; **training setting wrong (B3)** |
| 51 runs (25+20+6), $1.5\times10^8$ steps; 72 tuning runs, $3.6\times10^7$; EV-GNN 15; dt5 5 | 05:59, s3:73-74 | Alg. 7.4 | `pipeline.py:31-33,164-167` | Arithmetic OK; **ablation set wrong (M1)** |
| Penalty-weight sensitivities: 3 seeds, $w\in\{1,100\}$ | s3:74, 06:77 | Alg. 7.4 | `pipeline.py:33,67`, `ablations/train_attn_penalty_w*.yaml` | OK |
| $d_m$=128, $n_L$=3, $n_H$=4, FFN 512 GELU, dropout 0, head 128→64→2, critic 128→64→1 | 04:31-33, Tab. S2 | Alg. 6 | `base.yaml:6-9`, `attention_policy.py` | OK |
| Parameters: actor 610,370, critic 676,481, total 1,286,851 | 04:41, s1:29, Tab. S2 | formulation Sec. 5 | recomputed analytically from `attention_policy.py`: exact | OK |
| Small network 228,067; MLP actor 792,976 | s1:30-31 | I-21 | recomputed: 228,067 = actor 105,698 + critic 122,369; MLP actor 792,976 | OK (wording m20) |
| $F_c$=12, $F_g$=38 | 04:19 | nomenclature | `constants.py:150-151`, `wrapper.py:311-353` (feature list matches the text item by item) | OK |
| $P^{ref}$=11 kW, $E^{ref}$=100 kWh, $c^{ref}$=0.25 €/kWh, $H_f$=8 | 03:110,121, 04:22-24, Tab. S2 | nomenclature | `constants.py:144-147` | OK |
| $\varepsilon_q$=0.008 kW; 1.6 kW at $N$=200; 2.22e-4 kW; 0.0069 kW; saturation 1e-3 kW | 04:189, s2:52-53,67-70 | DD16 | `constants.py:135-137`, `actuation.py` | OK (checked: discharge 2.2e-4+6.93e-3 < 8e-3) |
| Tolerance $10^{-9}$ | 04:292, s1:7, Tab. S2 | S 6.4 | `constants.py:133` | OK |
| $w_E=w_P$=10 €/kWh; above 11× max price (0.871 €/kWh) | 05:32, s2:135, Tab. S2 | DD22, algorithm.md:280 | `base.yaml:63-64`, `wrapper.py:427` | OK |
| $\beta$=0.01 (0 for penalty methods) | 04:295, Tab. S2 | DD11 | `base.yaml:27`, `train_*_penalty.yaml` | OK (penalty β=0 not stated, m19) |
| $\alpha\in\{0.01,0.05,0.1,0.2\}$, $n_{cal}$=5,000, CP-trained at α=0.1 | 04:282, Tab. S2, s3:71 | DD12, DD23 | `base.yaml:66-73`, `train_attn_safe_conformal.yaml` | OK |
| Clip 0.2, value coefficient 0.5, 10 epochs, KL 0.03, grad clip 0.5, λ_GAE 0.95 | Tab. S2, Alg. S1 | Alg. 6 | `base.yaml:16-26` | OK |
| lr grid {1e-4,3e-4,1e-3}, minibatch {256,1024}, entropy {0,1e-3,1e-2} | Tab. S2 | Alg. 6 | `tuning.py:182` | OK (but `base.yaml:15` default 512 is outside the grid, B3) |
| $z_0(5)$=0.99302, 4.965, $\kappa\le$2.391 / 0.797 | s2:57-58 | nomenclature, DONE | check_A1 | OK |
| 8 V2G EV models | 04:71, 05:9 | C0 | `specs/ev_specs_const_eta.json` (8 entries) | OK |
| MPC horizon 10 steps, $N\le25$ | 05:36-37 | Alg. 5 row 6 | `baselines/mpc.py` | OK |
| EV-GNN: per profile, $N$=20, ρ=4, 5 seeds | 05:39 | DD27 | `evgnn_adapter.md` | OK (training budget m19) |
| 0.01 kWh reporting threshold | 05:49, 06:63 | formulation 1.3 | `constants.py:154` | OK |

### 1.2 Numbers that do not match

| ID | Location | Paper says | Docs / code say | Severity | Fix |
|---|---|---|---|---|---|
| B3 | 05_experimental_setup.tex:56; s3_training.tex:34; algorithm.md:355 | Each tuning configuration is "trained for $0.5\times10^6$ steps on the tight 10-charger station" | `tuning.py:196-207` overrides only lr, minibatch, entropy and steps. Training therefore uses the full mixture ($N\in\{10,20\}$, 3 profiles, both ratings). Only *validation* is on the tight 10-charger station (`tuning.py:210-217`) | **Blocking** (decide before runs) | Either change the text to "trained on the training mixture and validated on 20 episodes of the tight 10-charger station", or add `envs.groups`/`scenario.rho_tr` overrides to `tuning.jobs`. Align algorithm.md:355 too. |
| B3 (cont.) | code: `pipeline.py:117-122`, `reproduce.sh` default stage `all` (no `tune`), `base.yaml:15` | "The selected configuration is retrained with five seeds" (05:57) | Without `--tune`, `_selected_overlay` silently returns `[]`. Final runs then use the base defaults (minibatch 512, which is outside the tuning grid, lr 3e-4, entropy 0) with no warning | **Blocking** (pre-run) | Make `train` fail, or at least warn loudly, when `selected.yaml` is missing for a tuned method. Alternatively, set the base default to a grid point and state in the paper what is used if tuning is skipped. |
| M2 | 03_problem_formulation.tex:102; formulation.md:161 | "Episodes with $\lvert J^{LP,ceil}\rvert<1$ € are excluded" | `analysis_repo/scripts/load.py:69,307` uses `GAP_EPS = 1e-9`; `stats.py:388` documents 1e-9 € | Major | Set `GAP_EPS = 1.0` in load.py and the STAT_METHODS text, or change the paper to the implemented rule. Also report how many episodes are dropped. |
| M1 | 05:43 (ablations "remove admission control, set β=0, curriculum, smaller network"); s3:74 ("Four ablations with five seeds") | The 4 trained ablations appear to be no-admission, β=0, curriculum and small | `pipeline.py:31-32` and algorithm.md:478 train β=0, curriculum, small and **penalty-dense**. No-admission is evaluation-only (algorithm.md:485, `names.py` `attn_safe_noadm`). algorithm.md:308 still lists the old set | Major | 05:43: "Ablations of SA-Attn set β=0, train with a curriculum or use a smaller network (retrained), and remove admission control at evaluation time; a dense-penalty variant of Attention+penalty is a further ablation (Section S2.5)". s3:74: name the four trained ablations. Fix algorithm.md:308. |
| m16 | 03:53 | Cap "lowers the value of Eq. 4 by at most $10^{-6}$ kWh" | `bounds.py:149-166` starts at reach−1e-6 and then lowers it geometrically (1e-6, 1e-5, ...) until it is certified; formulation.md:64 says "lowered only by the amount needed" | Minor | "...by about $10^{-6}$ kWh, and further only where needed to absorb bisection error". |
| m19 | Tab. S1/S2 | Missing values | `min_time_of_stay` 180 min (`base.yaml:49`); $E^{min}$=5 kWh; Adam ε 1e-5; log σ clamp [−5,0.5], init −0.5; advantage normalisation; deterministic test action $a=\mu_\theta$ (algorithm.md Alg. 6); conformal `n_train`=2000 and GBR (200 trees, depth 3); EV-GNN trained 3×10^6 steps (the adapter's command line) instead of its default 10^7; β=0 for penalty methods; small network head 64→32; tolerance unit (kW or kWh) | Minor | Add the rows. Main text 05:38 "published hyperparameters" should add "with the same budget of $3\times10^6$ steps". |
| m20 | s1:30 | "the small ablation network ... has 228,067 parameters" | 228,067 = actor + critic | Minor | "...has 228,067 parameters in total (actor and critic; head width 32)". |

---

## 2. Result claims and measured numbers (no experiments have run)

- No experimental result number or interpretation appears outside `\RES{}`: 1 in the abstract, 25 in Results, 3 in Discussion, 3 in the Conclusion, 1 in s3 and 14 in s4. The TODO markers are in place.
- No sentence reports or interprets results. Borderline sentences, all acceptable:
  - 04:43 ("...does not guarantee performance there, because attention over 200 tokens averages over more keys") is an a-priori caveat, not a finding.
  - 06:60 ("Theorem 1 predicts zero") is a prediction from the theorem.
- Measured numbers that come from AI-run preliminary scripts rather than the experiments:
  - s2:80-81 (dropped spawns 6.8% / 3.7%, energy within about 5%, occupancy about 0.21);
  - s2:57-58 (κ bounds);
  - s2:70 ("would exceed the forward limit by up to about $n(t)\cdot0.007$ kW").
  - These are scenario-design statistics, not results. The authors must re-measure or confirm them (AI policy). Label them as scenario-generator statistics in s2.
- The cover letter (`cover_letter.md`) has no result claims, only a `[RESULT]` placeholder.

---

## 3. Notation (vs `method/nomenclature.md`) and units

| ID | Location | Issue | Severity | Fix |
|---|---|---|---|---|
| m11a | 04_method.tex:36 | $\mathcal N$ is both the charger set ($\mathcal N(t)$) and the Gaussian $\mathcal N(a_i;\mu,\sigma^2)$ in the same equation | Minor | Write the Gaussian as $\mathrm{Normal}(\cdot)$ or $\phi_{\mathcal G}$. |
| m11b | 04:265 | $k$ is the conformal quantile index, while nomenclature reserves $k$ for step counts and forecast lags (and uses no symbol here) | Minor | "the $\lceil(n_{cal}+1)(1-\alpha)\rceil$-th smallest score" without $k$, or use $k_\alpha$ (also s1:169). |
| m11c | s1_proofs.tex:169-170 | $S_{n+1}$ and "$S_1,\dots,S_{n+1}$" use $n$, which is $n(t)$ elsewhere | Minor | Use $S_{n_{cal}+1}$. |
| m11d | 03:23 vs 03:111 vs nomenclature | $p_i(t)$ is called "grid-side power" (03:23), "executed counterpart" (03:111) and "commanded power" (Section 4, A3, nomenclature) | Minor | Define once in 3.2: "$p_i(t)$ the commanded power ... (the simulator-reported power is $\tilde p_i$, Section 4.4)". Replace "executed" with "commanded". |
| m11e | 04:30-33 | $e_0,e_i$ (embeddings) and $\theta,\omega$ (actor/critic parameters) are used without definition | Minor | Add "$e_i=W_c\xi_i+b_c+v_c$" and "actor parameters $\theta$, critic parameters $\omega$". |
| m11f | s2:46-50 vs 03:88 | $z$ (Lemma S6 auxiliary) vs $z_i$ (shortfall slack); $h$ (s2 proof) vs $h_i$ (encoder output) | Minor | Rename the auxiliaries (e.g. $\zeta$, $\hat h$) or note the local scope. |
| m12 | 04:83-86 | Eq. 16 is introduced as "in the EV2Gym instantiation the lower bound becomes ...", but EV2Gym has a taper (Lemma S5). Eq. 16 holds only with $\eta=1$ **and** no taper (safety_layer.md:184) | Minor | "...and with $\eta^c=\eta^d=1$ and no taper the lower bound becomes". |
| m36 | Alg. S1 (s3:12-30) vs Tab. S2 | $B_{mb}$, $c_{vf}$, $c_{ent}$, $\epsilon_{clip}$ appear in Alg. S1 but not in Tab. S2 (only λ_GAE is) | Minor | Add the symbols to the Tab. S2 parameter names. |
| — | Fallback labels | Paper: cases (i)-(iv). Nomenclature, code and logs: FB-1*, FB-2, FB-1, FB-0. See M7 | Major (M7) | See Section 4. |
| m28 | 04:244 | Eq. 20 is stated as an a-priori bound without its extra hypotheses | Minor | Add "(under A1, A3, A4 and a non-increasing $\bar P_i$, Lemma S4; EV2Gym instantiation)". |

Further notation checks:
- **Units:**
  - Δt in [h] with values in min, €/kWh, kW and kWh are consistent.
  - β is dimensionless and $\chi$ is in steps.
  - The gap is dimensionless in Eq. 9 and in % in the tables (consistent with `load.py:306`).
- **Symbols defined before use:** all symbols are defined before use or with an explicit forward reference ($R_i$ in Eq. 4 → Section 4.2; $\varepsilon_q$ in 4.3 → 4.4).
- **Other symbols:** $\Ctil$, $\Crev$, $\Bh$, $\chi$, $\mathcal S$, $\vartheta$, $J^{LP,ceil}$, $G_i^{max}$, $\hat d^{lo}$ and $\tau^{lo}$ match the nomenclature.

---

## 4. Cross-references, numbering and result placeholders vs analysis outputs

### 4.1 Resolution and numbering

- All references resolve, and the logs are clean. The `xr` S-/M- prefixes work: main text "Table S1", "Lemma S1-S6", "Proposition S1", "Algorithm S1"; supplementary "Equation 3", "Lemma 1", "Figure 3".
- The main-text float order matches the first-citation order. Every float is on the same page as, or after, its first citation:
  - Figures 1-5;
  - Table 1 on p. 4, Table 2 on p. 15;
  - Algorithm 1 on p. 11 (cited on p. 10).
- The main text has 7 figures + tables, which meets the limit of 7 (DD29).

| ID | Location | Issue | Severity | Fix |
|---|---|---|---|---|
| m21a | 06:82 (`fig6_cost_of_guarantee.pdf` = Figure 5); analysis `README.md:95-96` | The analysis figure plan still says "Fig. 5 = fig5_example_day (main text), Fig. 6 = cost". The manuscript moved the example day to Figure S6, and the paper README:28 documents the new mapping. Two READMEs disagree | Minor | Update the analysis README plan, or rename the files (`fig5_cost_of_guarantee`, `figS6_example_day`) in `figures.py:354,446` and the .tex files. |
| m21b | s4 | Supplementary tables print as S3=T2, S4=T3, S5=T5, S6=T4, S7=T7, S8=T6, S9=TS1, S10=sensitivity. Figure S5 (pipeline) is cited first, at s2:84, before S1-S4 | Minor | Reorder the `\resulttable` calls (T2, T3, T4, T5, T6, T7), or map the names in a table note. Move the pipeline figure into S2, or cite it after S1-S4. |

### 4.2 Placeholder file names vs `make_all.py` outputs

- Figure file names match exactly what `figures.py` writes:
  - `fig3_zero_shot_scaling.pdf` (figures.py:227);
  - `fig4_constraint_satisfaction.pdf` (275);
  - `fig6_cost_of_guarantee.pdf` (446);
  - `fig5_example_day.pdf` (354);
  - `figS1_training_curves`, `figS2_ablation`, `figS3_conformal_alpha`, `figS4_inference_time`;
  - `fig2_concepts` and `figS5_pipeline`, which are copied via `make_all.py:144`.
- Table files `T1`-`T7.tex` and `TS1.tex` are written with `\label{tab:t1}` ... `tab:ts1` (tables.py:161), and they match `\Tab{tab:t1}` and `S-tab:t2`... `tab:ts1`.
- The relative path `../../analysis/` resolves only in the planned `paper/` + `analysis/` layout, not in the current workspace (`analysis_repo`), as paper README:28 notes. Keep this in mind when building in the workspace: the placeholders will show.

Content mismatches between the text and the analysis outputs:

| ID | Location | Issue | Severity | Fix |
|---|---|---|---|---|
| M3 | 06:59 ("per method and station size"), 06:67 (placeholder "versus $N$, per method"), 06:69 caption "across station sizes"; s4:12 | `figures.py:230-275` (`f4_constraints`) draws horizontal bars **per method × tightness level, pooled over $N$**. There is no $N$ axis | Major | Either rewrite f4 to plot both metrics against $N$ (like fig3), or change the text and caption to "per method and transformer rating, pooled over station sizes" and point to Table S4 for the per-$N$ values. |
| M4 | 06:62-63 | Asks for (a) infeasible steps "split into fallback cases", (b) shortfall "as a fraction of the deficit bound", (c) capped and admission-reduced requests "as listed in Table S4" | T3 (`tables.py:304-322`) has unserved %, energy not served, episodes with unserved EV, violation energy and max, episodes with violation, and total fallback activations. It has no per-case counts, deficit bound, capped or reduced columns. The raw columns exist (`fb0_steps`... in `load.py:313`; `n_evs_admission_reduced`, deficit in `metrics.py`) | Major | Add the columns to T3 (FB per case, deficit-bound ratio, n capped >0.01 kWh, n reduced, kWh reduced), or drop these requests from 06:62-63. |
| M5 | 06:76; 07:26 ("euros per EV and per year") | Cost of the guarantee "in € per EV and in percent of $\lvert J^{LP,ceil}\rvert$" | T5 (`tables.py:350-375`) and fig6 report **€ per charger and episode** and Δgap in pp | Major | Use one unit everywhere: "€ per charger and episode" (and Δgap in pp), or add € per EV and % of $\lvert J\rvert$ columns to T5. |
| M6 | 05:23, 07:19 (literature configuration); s4:78-93 (`tab:sensitivity`) | The literature configuration (100 episodes) and the sensitivity table (SoC 0.8, Δt 5 min, penalty weights 1/100) are promised | No analysis output exists. `tables.py`/`figures.py` contain no `sensitivity`, `soc0.8`, `dt5` or `literature` handling; T1-T7 use the grid only (README). The literature configuration appears only in `T2_full_per_configuration.csv`. The penalty-weight runs already appear in T4 (EXTRA_ABLATIONS), so the S10 row duplicates T4 | Major | Add a `TS2` sensitivity table and a literature-configuration table/row to `tables.py`, or state that these cells are filled by hand from `summary_by_config.csv`. Remove the penalty-weight row from S10 or from T4. |
| m22 | s4:29-30, s4:59-60 | Asks for infeasible-step change at "N=200", projection distance, profit change, safety-layer share of step time, admission LP time and oracle solve time | T4 has gap/unserved/violation at {10,20} and N≥50 only; T6 has training h and total inference ms per N | Minor | Add the columns (proj_dist_mean, infeasible_steps, oracle `solve_time_s` are in the raw schema), or rephrase the placeholders to match T4/T6. |
| m25 | 07:19 | "compare with Table 1 only on the literature configuration" | Table 1 has no numbers, and no literature-configuration output exists (M6) | Minor | "compare with published EV2Gym numbers only on the literature configuration (Section 5.1)". |
| M7 | 04:234-243, 06:62, nomenclature:125, `fallback.py:17`, `load.py:67` | Paper cases (i)-(iv) vs code/log/nomenclature FB-1*, FB-2, FB-1, FB-0, with numeric codes 3, 2, 1, 0. The mapping is (i)=FB-1*, (ii)=FB-2, (iii)=FB-1, (iv)=FB-0 and is written nowhere. Filling 06:62 from `fb0_steps` etc. invites mislabelling ((i)≠FB-0). Also `fallback.py:31-32` returns FB-0 for a contradictory step with negative budget, so (iv) can also arise under (i) | Major | Add the mapping to nomenclature.md, to the column labels of the analysis tables and as a footnote in Section 4.5. Optionally rename the code labels to FB-i…FB-iv. |
| M8 | 05:29-41 vs 06:35-37 (T1 placeholder rows) and `names.py` | Section 5.2 introduces no EV-GNN+safety layer, eMPC+safety layer, "eMPC"/"eMPC-HiGHS" names or SA-Attn-point (uncalibrated). These appear in T1/T2/T3 rows and legends (`names.py:52-71`) and in s4:45 | Major | 05:35-41: add "EV-GNN and both MPC variants are also run with their output passed through the safety layer"; name the MPCs "eMPC" and "eMPC-HiGHS" as the tables will; add SA-Attn-point to 05:44. |
| M9 | 05:49, 05:51, 03:52 | "share of EVs that miss $E_i^{req}$" | `metrics.py:146-166`: `energy_not_served_kwh` / `frac_evs_unserved` use the requirement **after admission control**, while `*_dd2` use the arrival-capped request. Penalty methods run without admission (s2:136), so SA-Attn is scored against possibly reduced targets and the baselines against unreduced ones | Major | State in 5.3 which reference the main metric uses. Report the DD2-referenced shortfall as well (columns exist), at least in T3 and in the cost-of-guarantee comparison. |

---

## 5. Citations

### 5.1 Key consistency

- Cited keys: 76. In refs.bib: 76. In `references.bib`: 76/76. In `verified_refs.json`: 76/76, all `verified`. Uncited bib entries: 0.
- Read levels: 32 full_text (all WebFetch digests, per gap_analysis.md:7), 41 abstract_only, 3 metadata_only.

### 5.2 All Positioning-table citations (Table 1, 02:75-98) checked against paper_matrix.csv and gap_analysis.md

| Row | Cell(s) checked | Matrix / gap analysis | Verdict |
|---|---|---|---|
| EV2Gym [orfanoudakis2025ev2gym] | flat; penalty/penalty; none; no; oracle as baseline | "overload NOT blocked + penalty"; "Oracle reported as a baseline, no explicit gap metric" | Supported |
| EV-GNN [orfanoudakis2025scalable] | graph; penalty/penalty; none; "not in version read"; oracle or MPC baseline | Preprint digest: separate model per scale; ORCH: heavily weighted penalties. R4: the published version was not checked for size transfer, although style_profile.md shows the published nature.com HTML was reachable | Supported for the preprint. Close R4 by reading the published article |
| GNN-DT [orfanoudakis2026graph] | penalty/penalty; 25 to 100 chargers; reward within 5% of oracle | Digest: 25→5/50/75/100; journal abstract: within 5% | Supported. "25 to 100" comes from a WebFetch digest (see M12) |
| HetGPS [wang2026hetgps] | reward and heuristic; not constrained; empirical; 8 to 32 transformers; no gap | Matrix and gap row agree. "8 → 16 and 32 transformers ... yes (n)"; PI-LP "is a linearised reference, not an optimum" | Supported, but rests on an (n) number (M12) |
| Yan et al. [yan2026hierarchical] | hard (per-EV rule); not modelled; first-order power balance; no; gap to MPC | Gap row: "Prop. 4.1 proves only first-order power-balance feasibility"; "No transformer or station cap"; "Gap to MPC" | Supported |
| Wan et al. [wan2025logicinformed] | not modelled; hard QP projection; per-step feasibility; no; no | Matrix: Lemma 1 feasibility by construction; departure energy not a constraint | Supported |
| sLLF [chen2022smoothed] | rule, any size; laxity rule; hard common level; feasibility theorems; not learned; no | Gap row agrees | Supported (but see m1 on "same form") |
| Sadeghianpourhamami [sadeghianpourhamami2020definition] | aggregate state; penalty; not modelled; none; scaled arrivals; yes | Gap row agrees | Supported |
| Orfanoudakis et al. 2026 [orfanoudakis2026can] (abstract only) | graph policy; safety layer; safety layer; unknown ×3 | Abstract only; "HIGHEST-PRIORITY NOVELTY CHECK" (R1) | Consistent, but see B2 |

### 5.3 Claims attributed to papers (all such sentences, plus a random sample of 15 other citations)

The random sample: qiu2023, michailidis2025, ye2022, korkas2024, tabas2022, fan2024, gu2024, su2025, zhang2025modelfree, eland2026, ceusters2023, li2023learning, markgraf2025, jin2026, vanhuffelen2025. All other attribution sentences were also checked.

| ID | Location | Claim | Evidence | Severity | Fix |
|---|---|---|---|---|---|
| B2 | 01:24-25, 02:71-72, 02:93, 07:63 | "To our knowledge, no published controller combines ... with a proof of multi-step feasibility" | The closest work, orfanoudakis2026can (GNN-TD3 with a safety layer meant to preserve charging requirements and power limits), has been read as an abstract only. song2025longterm and eland2026safe are also abstract only. Gap analysis R1 is marked "Critical, before G1 is finalised" and is still open | **Blocking** (before submission) | Obtain CCIS 2841 pp. 11-25 (SpringerLink via Sakarya library) and fill row 93 and 02:28-29. Then keep or weaken the novelty sentences (TODOs 01:25, 02:29, 07:64). |
| M10 | 01:11 | "Most RL controllers ... encode the departure energy requirement **and the transformer limit** as weighted penalties [sadeghian2020, ye2022, korkas2024, ev2gym]" | sadeghian2020: "no transformer limit". ye2022: "station peak only via demand charge". korkas2024: "no aggregate transformer cap in model". Only EV2Gym penalises transformer overload | Major | Split the sentence: departure-energy penalties [sadeghian, ye, korkas, ev2gym]; transformer-overload penalties [ev2gym, orfanoudakis2025scalable, orfanoudakis2026graph]. |
| M11 | 02:27 | "The EV2Gym authors have also added a safety layer to a graph-enhanced ... TD3 agent [fujimoto2018addressing]" | The claim belongs to orfanoudakis2026can. fujimoto2018 only defines TD3 | Major | "...(TD3) [fujimoto2018addressing] agent [orfanoudakis2026can]". |
| M12 | 02:52; 02:88; 02:51/02:87 | "from 1,236 to 3,218 vehicles"; "8 to 32 transformers"; GNN-DT "25 to 100 chargers" | gap_analysis.md:28 marks 1,236→2,239/3,218 and 8→16/32 as **(n)** (WebFetch digest, must be checked against the PDF, R11). GNN-DT size range comes from a digest of the preprint (gap_analysis.md:7: every FT-digest number needs checking) | Major | Verify against the arXiv PDFs (2608.00679, 2502.01778) and the published GNN-DT article before submission. Record this in paper_matrix (read_level `full_text_pdf`). |
| m6/m7 | 02:20, 02:22 | 02:20: "Weighted violation terms ... do not prevent violations at test time [tabas2022, fan2024]". 02:22: "Reviews classify both approaches [Lagrangian and trust-region CMDP] as soft or probabilistic [gu2024, su2025]" | fan2024 is a SAC-Lagrangian CMDP (it belongs in 02:22, where it is already cited). su2025 classifies Lagrangian/barrier/GP as soft and groups "projection/trust region" under structure adjustment. Its classification of CPO as soft is not in the matrix; gu2024 supports only "expected cumulative vs instantaneous" | Minor | 02:20: cite tabas2022 (and ev2gym) only. 02:22: "Reviews note that both bound an expected cumulative cost rather than each step [gu2024], and classify Lagrangian methods as soft [su2025]". |
| m1 | 01:30; 04:137 | "The projection has the same form as the rate rule of sLLF"; "is also the rate rule of sLLF" | Matrix: sLLF $r_i=\clip(\bar r_i(L-\text{laxity}_i+1),0,\min(\bar r_i,e_i))$. The common level is scaled per EV by $\bar r_i$, so it is a weighted variant, identical only for equal $\bar r_i$ | Minor | "a closely related form (a common level, scaled per EV by its peak rate)". |
| m2 | 01:21 vs 02:25 | zhang2025dumes is called a "rule-based shield" in the Introduction but an "action-replacement layer" in Related Work | Matrix: "decision-level safety layer detects and replaces unsafe actions" | Minor | Use "action-replacement layers" in 01:21. |
| m3 | 02:52 | jin2026telemetry "has been deployed on a larger feeder" | Matrix: zero-shot simulation on a 322-bus feeder | Minor | "applied in simulation to a larger feeder". |
| m4 | 02:45 | "aggregation gave the first size-independent states" | The sentence has no citation and the matrix has no evidence for "first" | Minor | "Early size-independent states in EV charging came from aggregation [sadeghian2020]". |
| m5 | 02:68 | "provable projection layers exist in safe RL [tabas2022, wan2025]" | tabas2022 is a gauge map, not a projection (02:33 says so) | Minor | "provable action-correction layers". |
| m8 | 02:28 | "the layer preserves charging requirements and enforces power limits" | Matrix/gap: the abstract says the layer is "intended to" or "meant to" do so | Minor | "is intended to preserve ... and enforce ...". |
| m-ok | — | Other attributions checked and supported: 01:12-15; 02:7-16, 02:24-25, 02:32-34, 02:38-39, 02:43-47, 02:50, 02:57-62; 04:136, 04:262, 04:287-290, 04:300; 05:61-63 | Matrix agrees | — | — |

Further notes on WebFetch-digest dependence:
- 02:13 ("left many users short") rests on the TD3 38% satisfaction figure, which is marked (n) (gap_analysis.md:24). The sentence is qualitative, but check it in the PDF.
- No other (n) number is used in the text.

### 5.4 Reference list format (IEEE)

| ID | Issue | Severity | Fix |
|---|---|---|---|
| M14 | main.tex:49 falls back to `ieeetr` because `IEEEtran.bst` is missing. Two problems follow. (1) `main.bbl` has 24 "arXiv preprint" entries, many of them for peer-reviewed papers: fujimoto2018 (ICML), vaswani2017 and zaheer2017 (NeurIPS), lee2019set (ICML 2019; bib year 2018), kingma2015adam (ICLR 2015; bib year 2014), schulman2016highdimensional (ICLR 2016; bib year 2015), achiam2017 and amos2017 (ICML), agrawal2019, agarwal2021, paszke2019 and romano2019 (NeurIPS), xiong2020 (ICML), hu2021updet (ICLR), tessler2018 (ICLR 2019), huang2022cleanrl (JMLR 2022). (2) Acronyms are lower-cased in titles: "Ev2gym", "v2g", "ev", "Hetgps" (main.bbl:32,69,71,100,117,129,144,156,168,174,288,294,320) | Major (journal requires IEEE) | Install IEEEtran.bst (or the template's copy). Convert these entries to `@inproceedings` with venue and year, keeping the arXiv ID only for true preprints (dalal2018, ba2016, markgraf2025, wan2025, yan2026, wang2026hetgps, zhang2022transfer). Brace the acronyms in refs.bib titles ({EV2Gym}, {V2G}, {EV}, {HetGPS}, {GNN}). |

---

## 6. Terminology (`style/terminology.md`)

| ID | Location | Found | Required | Severity |
|---|---|---|---|---|
| m9a | 01:40 | "measured price of the guarantee" | "cost of the guarantee" | Minor |
| m9b | 02:12, 05:9, s2 Tab. S1 "Ports", s2:40 | "charger port", "one port each", "Ports" | "charger". Quoting EV2Gym's port model is allowed only when it is marked as such | Minor |
| m9c | 06:37, 07:60, s2:139 | "HiGHS port" (software port), which collides with the charger meaning | "HiGHS re-implementation (eMPC-HiGHS)" | Minor |
| m9d | s1:193 | "the transformer constraint is Eq. 6" | "transformer limit" | Minor |
| m9e | 03:79, s2:84, s4:60, Fig. S5 caption | "oracle" for our LP | "perfect-information LP/optimum" (use "oracle" only for other papers) | Minor |
| m9f | 06:42 heading, fig3 file name | "Zero-shot scaling" | "Zero-shot transfer across station sizes" | Minor |
| m9g | 05:51, 06:74-77, 07:13, s4 | "penalty method", "penalty-trained policy", "penalty-based policy" for the same baseline | "Attention+penalty" (or define "penalty-based methods" once for both penalty baselines) | Minor |
| m9h | 01:17 vs 02:53 | "heuristic filters" vs "heuristic screens" | One term | Minor |
| m10 | 00_abstract.tex:8; supplementary.tex:47 | Literal "SA-Attn" instead of `\method{}`, which breaks the G2 rename. "safe attention policy (SA-Attn)" also conflates the network ("attention policy") with the full method | `\method{}`; expand as "safe attention-based controller (\method{})" or similar | Minor |
| M13 (title) | main.tex:24 | "Scale-Agnostic and Constraint-Guaranteed" | The text says "size-agnostic". The guarantee is conditional on A1-A5, A5 is a closed-loop condition that may fail (04:225-229), and admission control is heuristic. "Constraint-Guaranteed" overclaims and will draw reviewer objections | **Major** | E.g. "Attention-Based Safe Reinforcement Learning with Provable Per-Step Feasibility for Size-Agnostic V2G Charging Control". |

Consistent throughout: safety layer / projection, charge point operator, transformer limit, departure energy requirement, infeasible step / fallback rule, admission control, V2G discharging, optimality gap.

---

## 7. Journal requirements (SAUCIS)

| Item | Status | Note |
|---|---|---|
| Abstract ≤ 400 words, no citations or references, third person | OK | 339 words including the 22-word placeholder, so about 80 words remain for the result sentences |
| 3-5 keywords, alphabetical, bold "Keywords" | OK | Attention mechanism, Electric vehicle charging, Safe reinforcement learning, Vehicle-to-grid, Zero-shot transfer |
| IEEE references | **Not met** | M14 (ieeetr fallback, arXiv entries, lower-cased acronyms) |
| Captions: figures below, tables above | OK | Also in analysis-generated tables (`tables.py:164`) |
| "Figure n"/"Equation n"/"Table n" referencing | OK | Through `\Fig`, `\Eq`, `\Tab`, `\Sec` macros; no "Fig." or "Eq." anywhere |
| Number only equations referenced later | Not met (m13) | Unreferenced numbered equations: eq:tau (03:18), eq:policy (04:35), eq:phi (04:50), eq:reach (04:60), eq:lclosed (04:84), eq:conformal (04:266), eq:admission (s2:119). Use `equation*` or reference them |
| No paragraph indentation | OK | preamble.tex |
| ≤ 7 figures + tables (experimental paper) | OK | 5 figures + 2 tables |
| Length about 25 double-spaced pages | At risk (m23) | Body is about 9,100 words before results (the linter counts 9,838 including the abstract and the AIF), while DD29 targets about 8,500. Results and Discussion will add more |
| Article Information Form items | OK | Contributions, Acknowledgments, Conflict of Interest, Support, Ethical Approval, Availability, AI Statement and Plagiarism Statement are present, in template order, before the References |
| Anonymised build | OK | main_anonymous.pdf and supplementary_anonymous.pdf show "Anonymous author(s)" and "Removed for double-blind review" in Contributions and Acknowledgments, and withhold the link. The PDF Author/Title fields are empty. Optional: the CreationDate timezone (+03) is a weak location hint (m27) |
| AI disclosure in Methods or Acknowledgements plus the AI Statement | Present | The Acknowledgments sentence (AIF:20) is kept in the anonymous build |
| AI statement accurate and consistent with policy | **Not met (B1)** | See below |
| Figures as EPS | Pending | The analysis writes EPS; the manuscript uses PDF. Convert at submission |

**B1 (Blocking): the AI statement is not accurate.** The statement (10_article_information_form.tex:44-51) lists three purposes: literature search, code assistance and "preliminary wording for parts of the manuscript". The workspace shows AI agents also:
- derived the method and theory (safety-layer bounds, proofs, assumptions; `method/*.md`);
- designed the experiments (DD1-DD29: scenarios, baselines, tuning);
- wrote the analysis pipeline that generates every table and figure;
- drafted **all** sections, including the abstract, the introduction, the related work and the result-reporting sentence frames in Results, Discussion and Conclusion;
- produced the conceptual figures.

The policy calls AI-generated abstract, introduction and literature review "inappropriate" unless the authors thoroughly revise them. The policy also requires a detailed explanation of how and for what purposes AI was used, and forbids AI from producing or reporting findings. Two problems follow:
- "suggested preliminary wording for parts of the manuscript" understates the use;
- "No AI tool was used to generate, report or interpret the research results" needs qualification, because the AI-written scripts compute every reported statistic and the AI wrote the result-sentence frames.

Fix: before submission, the authors must:
1. Rewrite the sections themselves.
2. Write every result sentence themselves, replacing the AI frames instead of filling them.
3. Extend the statement to name, as accurate for the final version, each of these uses:
   - support in developing the method and proofs, verified by the authors;
   - experiment design and simulation and analysis code;
   - preliminary drafting of all sections;
   - conceptual figures.
4. Add: "The statistics are computed by scripts the authors reviewed. All reported values were checked, and all interpretations were written, by the authors."

---

## 8. Style linter and style issues the linter cannot detect

### 8.1 Linter runs

Command: `python3 style/lint.py <file> --profile style_thresholds.json`

| File | Words | Restricted phrases | Sentences > 35 words | Paragraphs > 180 words | Demonstrative-opener paragraphs | "This"+verb | Consecutive same opener | Blocking flags |
|---|---|---|---|---|---|---|---|---|
| manuscript.md | 9,838 | 0 | 0 | 0 | 0 | 0 | 0 | **14** |
| supplementary.md | 4,740 | 0 | 0 | 0 | 0 | 0 | 0 | **15** |

All blocking flags come from the acronym check, and all are false positives:
- Manuscript: "used but never defined": A1-A5, S1-S8, SA. "Used before first definition": V2G, because of the title.
- Supplementary: A1, A3-A5, S1-S10, SA, and V2G.

A1-A5 are assumption labels, S1-S10 are supplementary item numbers and "SA" is the stem of SA-Attn. No action is needed on the text. Teach `lint.py` to ignore `A\d`, `S\d+` and hyphenated method names. Section mean sentence lengths are 16.0-26.0 words, within MEAN_LO/HI.

### 8.2 Issues the linter cannot detect

| ID | Location | Issue | Fix |
|---|---|---|---|
| m30 | 01:24, 02:71, 02:72-73 (and the abstract and conclusion) | The same novelty claim ("no prior work joins ... proof of multi-step feasibility") appears as a paragraph-final restatement three times | Keep it once, in Section 2.5. In the Introduction, point to Section 2.5. |
| m31 | 00:14, 01:36, 08:7, cover letter | "the transformer limit and every departure energy requirement hold at every step". A departure requirement holds *at departure*; what holds at every step is its continued feasibility | "...the transformer limit holds at every step and every departure energy requirement is met at departure, for any raw action". |
| m26 | 00:8, 04:12 (Fig. 1 caption), 08:7 | "closed-form" projection, although λ* is found by a sort and one pass, with a bisection fallback (Alg. 1) | "exact projection computed in $O(N\log N)$". |
| m24 | 05:13, s2:74 | "Every exogenous quantity scales linearly with $N$". Prices do not | "Every grid quantity and the arrival rate scale linearly with $N$". |
| m15 | 04:243, 05:15 | "Inflexible load and PV are scaled to a mean peak of $0.4P_{tr}$, so the PV peak stays below the rating ... so case (ii) cannot occur". A mean peak does not imply that every peak is below the rating (the multiplier is drawn from N(0.4, 0.1)). The real reason is EV2Gym's clipping of PV at the rating, or a margin of more than 6σ | State the actual argument (clipping), or say "case (ii) did not occur" once measured. |
| m14 | 04:234 | "four exclusive cases" while (iv) is defined "within case (iii)" (and the code also returns (iv) under (i)) | "three cases, the third with a sub-case (iv)". |
| m29 | 04:292 | "Two masks ...: tokens with ..." The colon introduces only the first mask | "Two masks reduce variance ... First, tokens with ... Second, transitions of fallback steps ...". |
| m32 | 01:30 vs 04:139 | "box intersected with one sum constraint" vs "the aggregate constraint is two-sided" | "a box intersected with a two-sided sum constraint (a slab)". |
| m33 | 05:25, 05:38 vs s3:73 | Main text presents EV-GNN and the 5-min study as part of the evaluation; s3 calls them "optional" | Use one status. |
| m34 | 05:44 | It is not stated that SA-Attn-CP is trained once at α=0.1 and recalibrated for the other α | Add half a sentence (as in s3:71-76). |
| m23 | whole main text | About 9,100 body words before results vs the DD29 target of 8,500 | Candidates to move to the supplement: 04:132-133 (floating-point details), 04:250-257 (admission control details), 03:189-201 (the J^LP vs J^LP,ceil discussion). |

Additional notes from reading:
- No rhetorical questions, three-adjective lists, banned words or em dashes were found.
- "First"-type claims are hedged ("To our knowledge", "Our screening ... found no"); see B2.

---

## 9. Code ↔ paper: algorithm boxes

| Step | Paper (Alg. 1 / Section 4 / Alg. S1) | Code | Verdict |
|---|---|---|---|
| Bounds | $u=\bar P(E)$; $l=\max\{p^{min},g^{-1}((\hat B(\tau-1)-E)/\Delta t)\}$ if $E\ge\hat B(\tau)$, else $u$ | `bounds.py:169-183`: same, plus $l\gets\min(l,u)$ and feasibility tested with a −1e-9 tolerance | Match (m17: mention the clip to $u$ and the tolerance) |
| Laxity | $(E-\hat B(\tau))/(\eta^c\bar P^{nom}\Delta t)$ | `bounds.py:182` (η=1) | Match |
| Feasibility test (A5) | $\sum l\le\tilde C$, $\sum u\ge-C_{rev}$, $\tilde C\ge-C_{rev}$ | `projection.py:46-56` | Match |
| Projection | clip; if the sum is in range, keep it; else sort the 2N breakpoints, one pass, post-check, safe bisection | `projection.py:76-135` | Match |
| Fallback order | (i) contradictory → (iii) procedure with forward priority; (ii) $\sum u<-C_{rev}$ → $u$; (iii) start at $p^{min}$, fill by increasing laxity up to $l-p^{min}$; (iv) budget<0 → $p^{min}$ | `fallback.py:20-43`: same order. Tie-breaks (l−p^min descending, index ascending) are not in the paper. Under (i) with negative budget the label is FB-0 | Match (m17 tie-breaks; M7 labels) |
| Actuation round-up | "rounded upwards on its action grid" | `actuation.py:57-74`: ceil to 1e-5, **plus full current when $p\ge u-10^{-3}$ kW**; NaN sanitising in `wrapper.py:362-365` | Alg. 1 omits the saturation rule and the sanitising (m17). s2:66-67 covers the saturation |
| Masking | tokens with $u-l\le10^{-9}$ dropped from log-prob and entropy; fallback steps masked from the actor loss, kept for the critic; fixed $B_{mb}$ normaliser | `wrapper.py:309` (`infl`), `buffers.py:103` (`w=feasible`), `ppo.py:141-155` (`/Bm`) | Match. Entropy is averaged per influential token while log-prob is summed (m18) |
| β regulariser | Eq. 21 with $\bar x=P^{ref}\mu$, $\Pi_t(\bar x)$ under stop-gradient, mean over $n(t)$, expectation over all states | `ppo.py:156-170`: same, but **only on feasible, non-fallback transitions** (`wreg = w·feas`), normalised by $B_{mb}$ | Minor deviation (m18): add "on feasible steps" to Eq. 21 or the text |
| Reward normalisation | $r/(N r^{ref})$, $r^{ref}=c^{ref}P^{ref}\Delta t$ | `ppo.py:299,330` (N = number of chargers) | Match |
| PPO loop | 16 envs, 128 steps, GAE(0.99, 0.95), 10 epochs, clip 0.2, grad clip 0.5, KL 0.03 early stop | `ppo.py`, `base.yaml` | Match. KL is checked after each full epoch, which is consistent. Advantage normalisation is not stated (m19) |
| Penalty reward | $r-10\cdot$shortfall$-10\cdot$overload energy, untightened limits, reported powers; dense adds $w_E\Delta t\sum(l-p)^+$ | `wrapper.py:425-429` (shortfall vs the DD2-capped request) | Match |
| Admission LP | linear inner taper, exact first step, $\hat C-n(t_0)\varepsilon_q$, connected-EV targets $\min\{E^{req},\Psi^{LP}-10^{-6}\}$, only the new request reduced, re-certified | `wrapper.py:240-252`, `admission.py` | Match |
| Conformal bound | GBR lower quantile; features (arrival step, weekday, arrival energy, requested energy, EV model); one session per episode; $\lceil(n+1)(1-\alpha)\rceil$-th score; $\max\{t^{arr},\lceil\hat q-\hat Q\rceil\}$ | `conformal.py:1-99` | Match ("requested energy" = $E^{des}-E^{arr}$, the desired, not the capped, energy; fine) |

---

## 10. Complete issue list by severity

**Blocking (3):** B1 (AI statement accuracy), B2 (novelty claim rests on abstract-only closest work), B3 (tuning protocol text vs code, and the silent untuned default in `pipeline.py`).

**Major (14):**
- M1: ablation set.
- M2: gap exclusion threshold of 1 € vs 1e-9.
- M3: Fig. 4 content.
- M4: constraint metrics requested from Table S4 are not produced.
- M5: cost of the guarantee per EV vs per charger.
- M6: no output for the sensitivity table or the literature configuration.
- M7: fallback-case label mapping.
- M8: compared methods incomplete in Section 5.2.
- M9: shortfall reference (after admission vs DD2 cap).
- M10: transformer-penalty over-attribution in 01:11.
- M11: misplaced citation in 02:27.
- M12: (n) digest numbers in the text and table.
- M13: title overclaim and "scale" vs "size".
- M14: IEEE reference list.

**Minor (48):**
- Citations and claims: m1-m8.
- Terminology: m9a-h (counted as 1 item each: 8), m10.
- Notation: m11a-f (6), m12, m28, m36.
- Numbering and placeholders: m13, m21a, m21b, m22, m25.
- Wording and style: m14, m15, m16, m23, m24, m26, m29-m34.
- Tables and code details: m17-m20.
- Anonymisation: m27.

Counted as distinct entries in the tables above: 48 (m6 and m7 count separately; m35 is unused).
