# Consistency report, round 2 (A6)

Date: 2026-10-22. Reviewer: A6. This file only reports; the manuscript, code and analysis were not edited.

Inputs re-checked:
- Revised manuscript (`paper_repo/manuscript/`: main.tex, preamble.tex, sections 00-10, refs.bib, logs, PDFs incl. anonymous builds) and supplementary (`supplementary.tex`, s0-s5).
- Markdown mirrors `manuscript.md` and `supplementary.md`. Both were regenerated with `tools/tex2md.py` into a scratch copy, and both are byte-identical to the shipped mirrors, so they are in sync.
- Authors' response `paper_repo/response_to_internal_review_round1.md`.
- Method docs `method/*.md` (DD1-DD42, nomenclature v5, algorithm.md, safety_layer.md, formulation.md, check results).
- Code `code_repo/code/` (configs, `v2gsafe/*`, README, SCHEMA_ADDITIONS, IMPLEMENTATION_NOTES, tests).
- Analysis `analysis_repo/` (README, scripts `names.py`, `load.py`, `tables.py`, `figures.py`, `make_all.py`, `stats.py`).
- Bibliography (`lit/merged/references.bib`, `verified_refs.json`, `paper_matrix.csv`, `citation_fixes.md`, `gap_analysis.md`).
- Style (`style/lint.py`, `style_thresholds.json`, `terminology.md`) and journal rules (`journal/journal_requirements.md`).

Location codes:
- `NN:L` is `manuscript/sections/NN_*.tex`, line L.
- `sK:L` is `supplementary/sections/sK_*.tex`, line L.
- Other paths are relative to `/home/claude/work/`.

## 0. Summary

| Severity | Round 1 | Round 1 items now fixed | Open after round 2 (residual + new) |
|---|---|---|---|
| Blocking | 3 | 2 (B1 in text, B3) | **2** (B2 residual, B4 new) |
| Major | 14 | 12 (1 partly fixed, 1 not fixed) | **9** (M14 residual + 8 new) |
| Minor | 48 | 42 (4 partly fixed, 2 not fixed) | **31** |

What is clean in round 2:
- **Builds:**
  - Both documents and both anonymous builds have 0 warnings, 0 undefined references and no "??".
  - The `xr` cross-references resolve (Lemma S1-S7, Proposition S1-S3, Table S1-S14, Figure S1-S7).
  - All 17 numbered equations are referenced (round-1 m13 is fixed).
- **Floats:** the main text has 5 figures and 2 tables, which is within the limit of 7.
- **Mirrors:** both Markdown mirrors are in sync with the .tex files.
- **Results:** no result number or result interpretation appears outside `\RES{}` or `\TODO{}`. The scenario-generator statistics in s2:110-111 are labelled as design checks and carry a `\TODO`.
- **Citations:**
  - 79 keys are cited, and refs.bib has 79 entries with no uncited entry.
  - All 79 are in `references.bib` and in `verified_refs.json` with status `verified`.
- **Abstract and keywords:**
  - The abstract has 284 words by the linter count (273 by a plain word count).
  - There are 5 keywords in alphabetical order.
- **Article Information Form:** all items are present in template order. The AI statement now lists all five uses of the AI assistant.
- **Anonymised PDFs:** they show "Anonymous author(s)" and withhold the link, and the PDF Author and Title fields are empty.
- **Numbers:** every hyperparameter and scenario number checked matches docs and configs, apart from the items in Section 2.
- **Algorithm boxes:** Algorithm 1 and Algorithm S1 match the code (Section 9).

---

## 1. Round-1 items: status

### 1.1 Blocking and Major

| ID | Round-1 issue | Status | Evidence / remaining action |
|---|---|---|---|
| B1 | AI statement inaccurate | **Fixed in text**; author action pending | `10:44-55` and `declarations.md` list literature, method and proofs, experiment design, code and analysis pipeline, drafting and figures; the model is named. The Acknowledgments sentence `10:21` is kept in the anonymous build. Still open (author tasks, not agent-fixable): rewrite the AI-drafted sections; replace the AI-written result-sentence frames in 06/07/08 (e.g. `06:17` "At these sizes, the gap of \method{} is \RES{}") with the authors' own sentences, because the statement says no AI "reported" results. See also m25. |
| B2 | Novelty rests on abstract-only closest works | **Not fixed** (mitigated) | `02:68` describes orfanoudakis2026can and song2025longterm "only at that level", and Table 1 row `02:91` says "not stated in abstract". Gap-analysis R1-R3 are still open (`citation_fixes.md` §3: no open-access copy). **Still Blocking before submission.** |
| B3 | Tuning text vs code; silent defaults | **Fixed** | `05:71-75` and `s4:81-85` say the runs train on the mixture with 24 validation episodes and select on J^pen. `pipeline.py:155-172` refuses to train without `selected.yaml` or `--hparams`, and `tests/test_round3.py:241` tests this. `base.yaml:15` minibatch 512 is now in the grid `{256,512,1024}` (`tuning.py:38`). |
| M1 | Ablation set | Fixed | `05:54-56`, `s4:85` and `pipeline.py:49-52` agree. |
| M2 | Gap exclusion threshold | Fixed | `load.py:77` `GAP_EPS = 1.0`; count reported (`load.py:315-320`); `stats.py:387`. |
| M3 | Fig. 4 content | Fixed | `figures.py:241-245` plots against N (log x), with one panel per rating. The pooled version is `figS6_constraints_pooled` (Fig. S6). |
| M4 | Constraint metrics not produced | **Partly fixed** | T3b is new (`tables.py:360-391`). The share of EVs without infeasible steps, the capped energy and a per-N split are still missing. See B4 and M-N2. |
| M5 | Cost-of-the-guarantee unit | Fixed | T5 and fig6 report € per EV and % of \|J_LP,ceil\| (`tables.py:419-444`, `figures.py:424-464`). The new test-file mismatch is M-N1. |
| M6 | Sensitivity / literature outputs | Fixed | TS2 and TS3 (`tables.py:526-584`). w1/w100 appear only in TS2, because EXTRA_ABLATIONS (`names.py:111-113`) no longer lists them. |
| M7 | Fallback label mapping | Fixed | `04:217`, `s1:158-161`, `names.py:121-122`, the T3b note and SCHEMA §5 all agree. |
| M8 | Compared methods incomplete | Fixed | Every `names.py` method string is introduced in `05:30-56` or `s4:5-48`. |
| M9 | Shortfall reference | Fixed | DD31: admitted request for all methods (`wrapper.py:404-424`). |
| M10 | Transformer-penalty over-attribution | Fixed | `01:11-12` |
| M11 | Misplaced TD3 citation | Fixed | `02:26-27` |
| M12 | Digest numbers | Fixed | `02:46-47` and Table 1 use the verbatim checks of `citation_fixes.md` (GNN-DT 25 → 5-100; HetGPS 1,236 → 2,239/3,218). EV-GNN "1000 CPs / 70 groups" is still a preprint digest (m9). |
| M13 | Title overclaim | Fixed in paper | The working title follows DD30. The code README title is still the old one (m28). |
| M14 | IEEE reference list | **Not fixed** (partly) | Bib entries were upgraded. `main.blg:4` still uses `ieeetr.bst`. See M-R1. |

### 1.2 Minor (48)

| Items | Status | Note |
|---|---|---|
| m1-m8 (citation wording) | Fixed (8) | Checked at `01:21,29`, `02:17-19,27,41,48,65`. |
| m9a-h, m10 (terminology) | Fixed (9) | A grep of the sources finds no "port", "HiGHS port", "transformer constraint", "price of the guarantee", "heuristic screens" or "oracle" in our own prose. The remaining occurrences are labels and other papers' usage. |
| m11a Gaussian $\mathcal N$ | **Partly fixed** | The main text uses Normal(·), but `s2:96-97` writes $m_L\sim\mathcal N(0.4,0.1^2)$ again (m3). |
| m11b-e | Fixed (4) | |
| m11f auxiliary symbols | **Partly fixed** | `s2:34` $h(\varsigma)$ collides with the encoder output $h_i$ (m4). |
| m12, m28, m36, m13, m21b, m22, m25 | Fixed (7) | |
| m21a figure plan in analysis README | **Not fixed** | `analysis_repo/README.md:116-117` still lists "Fig. 5 = fig5_example_day" and "Fig. 6 = cost". In the paper, `fig6_cost_of_guarantee` is Fig. 5 and the example day is Fig. S7 (m1). |
| m14-m16, m24, m29-m34, m17-m20 | Fixed (16) | `attn_proj_nolb` introduces a new β inconsistency (M-N4). |
| m23 length | **Partly fixed** | 8,535 words of running text before Results are written; 23 pages single-spaced in the article class (m24). |
| m26 "closed-form" | **Partly fixed** | Fixed in the text, but Figure 1's artwork still says "closed form" (M-N6). |
| m27 PDF CreationDate timezone (+03) | **Not fixed** | `pdfinfo`: "Thu Oct 22 08:38:30 2026 +03" (m2). |

---

## 2. Numbers: text vs docs vs code

### 2.1 Values re-verified (match)

| Quantity | Paper | Doc | Code / config |
|---|---|---|---|
| Test grid: 36 + 6 stress + literature = 43 configurations; 100/50 replays; 3,550 episodes per policy | `05:21-24`, Tab. S2 `s2:139-140` | DD24, DD36 | `test_grid.yaml:1-27` (arithmetic: 2,400 + 600 + 450 + 100 = 3,550) |
| Stress level 3 kW at N ∈ {20, 200} | `05:21`, `s2:127` | DD36 | `test_grid.yaml:19-24` |
| Sensitivities: N=20, ρ=4, 3 profiles, SoC 0.8, Δt 5 min, c_thr 0.05, factor 0.9; only dt5 retrained | `05:26-28`, Tab. S2 | DD28, DD39, DD42(7) | `test_grid_sensitivity.yaml`, `ablations/train_attn_safe_dt5.yaml` (γ = 0.996655493) |
| Efficiencies 0.90 (×7) and 0.93 (×1); 8 models | `05:10`, `s2:47` | DD4 | `specs/ev_specs_const_eta.json` (Hyundai Kona 0.93) |
| P^cs = 22.17 kW; 2.22e-4 kW; 0.0069 kW; ≤ 0.0072 < ε_q | `s2:48,80-83` | DD16 | `constants.py:16` (22.170) |
| ε_q 0.008, ε_sat 0.001, tolerance 1e-9 | `04:167`, Alg. 1, Tab. S4 | nomenclature:41-43 | `constants.py:10-14` |
| P_ref 11, E_ref 100, c_ref 0.25, H_f 8, F_c 12, F_g 38 | `03:105,111`, `04:19-21`, Tab. S3, Tab. S4 | nomenclature | `constants.py:21-28`, `wrapper.py:300-337` (features item by item) |
| Tuning: 7 methods × 12 configs, 0.5e6 steps, seed 0, 24 validation episodes, J^pen at w=10; lr {1,3,10}e-4, n_mb {256,512,1024}, c_ent {0,1e-3,1e-2}, w {1,10,100}, η_Λ {0.5,2,8} | `05:71-75`, `s4:81-84`, Tab. S4 | DD37, DD42(1) | `tuning.py:36-46`, `pipeline.py:46-48` |
| Inheritance of selections | `s4:85` | algorithm.md:379 | `pipeline.py:55-62` (dense: see m22) |
| Lagrangian: combined advantage, Λ_max 1000, Λ(0)=0, cost limit 0 | `s4:19-22`, Tab. S4 | algorithm.md:300-303 | `ppo.py:57-60,381-384`, `base.yaml:67-72` |
| Deep Sets within 10 % of parameter count | `05:39`, `s1:32` | DD42(3) | `train_deepsets_safe.yaml:3` (+2.6 %/−4.8 %) |
| Small network d_m=64, n_L=2, FFN 256, head 32: 228,067 | `s1:30`, Tab. S4 | I-21 | `ablations/train_attn_safe_small.yaml`; FFN = 4·d_m (`attention_policy.py:55-60`) |
| Conformal: GBR 200 trees, depth 3; n_train 2,000; n_cal 5,000; α set; features | `04:229-231`, Tab. S4 | DD12, DD23 | `conformal.py:5-9,49-50,68`, `base.yaml:73-80` |
| eMPC-HiGHS 10 / 96 steps; elastic 1000 €/kWh | `05:46`, `s4:38-40` | DD33(7), DD34 | `mpc.py:10-19,140` |
| EV-GNN 3e6 steps, per profile at N=20, ρ=4, 5 seeds; action map a·ζ·P^cs | `05:49-50`, `s4:44-45` | DD27, DD35 | `evgnn_adapter.md:37,45`, `evgnn.py:6-7,51-53` |
| Compute: core about 32 GPU-h and 82-132 CPU-h; about 89 passes; extended about 31-41 GPU-h and 68-108 CPU-h | `05:83`, `s4:139-142` | (algorithm.md:510 is stale, m28) | `README.md:174-193` |
| A6 failure probability below 2e-9; pv ≤ 0.853·1.1 < 0.94; 1 − ε_q/ρ ≥ 0.997 (about 6σ); m_P > 1.06 (6.6σ) | `04:169`, `s2:96-100` | safety_layer.md:436-439 | own arithmetic: P(Z>5.97)+P(Z>6.6) ≈ 1.2e-9 |
| 768 configurations to 1.4e-14 kWh; 320 configurations to 1.7e-13 kW; κ ≤ 2.391/0.797; z0(5)=0.99302; κ*=4.965 | `s2:20-21,89-90`, `s3:69` | safety_layer.md:101,134; formulation.md:154 | `method/check_A1_results.txt`, `check_O0_results.txt` (see m19 on location) |
| EVs leave by step T−2 / not spawned within 4 steps of end | `s1:141`, `s2:108` | ev2gym_notes.md:107,350 | consistent |
| Fallback order, tie-breaks, FB-0 under (i) with negative budget | `04:209-217`, `s1:158-161` | nomenclature:127 | `fallback.py:20-43` |

### 2.2 Numbers or settings that do not match

| ID | Location | Paper says | Docs / code say | Severity | Fix |
|---|---|---|---|---|---|
| M-N4 | `s4:11,16`; Tab. S4 `s4:114` | `attn_proj_nolb` is grouped with the "penalty-based methods". These "use β=0, and their actor loss masks only tokens whose action cannot influence the commanded power". | `configs/train_attn_proj_nolb.yaml:2-8`: β = 0.01 "as SA-Attn" and the safety layer is on, so `buffers.py:121-124` sets `w = feasible`, which masks fallback steps. It is also tuned with w (`tuning.py:3-6`), as the paper says. | Major (decide before runs) | Either set β=0 for nolb in the config, or change `s4:16` to "Attention+penalty and MLP+penalty use β=0 ...; the variant without departure bound keeps β=0.01 and masks fallback steps like \method{}", and add a row to Tab. S4. |
| M-N3 | `05:42`, `s4:27-28` | The MLP is "a fixed-size reference ... at N ≤ 20 and on the literature configuration", and its weights beyond the training slots "never receive a gradient, so they serve only as a fixed-size reference at N≤20". | `pipeline.py:254-283` evaluates every checkpoint, MLP included, on all N groups. `names.py:99-104` puts `mlp_safe`/`mlp_penalty` in SCALING_METHODS and FIG_SCALING, so they enter T2, TS1 and Fig. 3 at N ≥ 50. `tables.py:394-416` T4 puts the MLP 2×2 cells in the "N ≥ 50" columns. | Major | Restrict the MLP evaluation to N ≤ 20 and the literature configuration (pipeline), or drop the MLP from SCALING_METHODS, FIG_SCALING and the T4 N≥50 columns. Otherwise the text must say that the MLP zero-shot numbers are shown and how to read them. |
| m5 | `05:16` vs `s2:129` | "Inflexible load and PV are scaled to a mean peak of 0.4 P_tr". Table S2 says PV "mean peak about 0.38 P_tr". | algorithm.md:450: 0.853 · U[0.9,1.1] · 0.4 ≈ 0.38. | Minor | `05:16`: "scaled with multipliers of mean 0.4 P_tr (mean PV peak about 0.38 P_tr)". |
| m13 | `05:83` | "82-132 CPU-hours of evaluation" | `README.md:177-181`: the CPU-h also cover the admission tables, conformal calibration and the optima (`s4:140-141` says this correctly). | Minor | "82-132 CPU-hours for evaluation, admission tables, calibration and optima". |
| m22 | `s4:85`; `05:55` | "the penalty ablations inherit that of Attention+penalty". The w used by the dense ablation is not stated. | `pipeline.py:61`: `attn_penalty_dense` takes the full `selected.yaml`, including the tuned w. `algorithm.md:379` says the dense ablation inherits only the PPO keys. The curriculum ablation (N=10 only for the first 20 % of steps, `base.yaml:34-38`) is described nowhere in the paper. | Minor | State that the dense variant uses the tuned w, and align algorithm.md. Add one clause on the curriculum to `05:54` or Tab. S4. |

---

## 3. Results and interpretation outside placeholders

- No result number or interpretation appears outside `\RES{}`/`\TODO{}`. All sections were read: 00, 06, 07 and 08, and s2-s5.
- Borderline statements are a-priori or design facts, not results:
  - `04:34` (attention over more keys);
  - `07:44` (A6 fails with negligible probability, which is derived);
  - `s4:132` (5-pair Wilcoxon).
- AI-run checks quoted as facts:
  - `s2:20-21` (1.4e-14, 768 configurations);
  - `s3:69` (1.7e-13, 320 configurations);
  - `s2:92` (n·0.007 kW);
  - `s2:110-111` (6.8 %, 3.7 %, about 5 %, about 0.21).
  - These are labelled as design checks, and the `\TODO` lines ask the authors to re-run them. They are acceptable.
- The cover letter holds only a `[RESULT]` placeholder. Its claim that SAUCIS has published RL and EV-forecasting work is unverified (m26).

---

## 4. Placeholders vs analysis outputs

### 4.1 File names (all match `make_all.py` outputs)

| Paper `\IfFileExists` | Producer | Status |
|---|---|---|
| `\restable{T1..T7,T3b,TS1..TS3}` → `\analysisdir/tables/<name>.tex` | `tables.py:190-197` writes `<name>.tex` with `\label{tab:<name lower>}` (`:168`) | OK; labels tab:t1 … tab:ts3 match. T7/TS2/TS3 fall back to placeholders when absent (`:600-603`). |
| `fig3_zero_shot_scaling.pdf`, `fig4_constraint_satisfaction.pdf`, `fig6_cost_of_guarantee.pdf` | `figures.py:237,243,464` | OK |
| `figS1_training_curves`, `figS2_ablation`, `figS3_conformal_alpha`, `figS4_inference_time`, `figS6_constraints_pooled`, `fig5_example_day` | `figures.py:191,421,523,542,293,372` | OK (fig5 needs `--log-steps`) |
| `fig2_concepts.pdf` (figchain) | `make_all.py:156-166` copies `figures_conceptual/fig_concepts.*` | OK. The local copy is identical (md5 4a4a01f4). |
| `figS5_pipeline.pdf` (figchain, `s5:102`) | `make_all.py:156` copies `figures_conceptual/figS5_pipeline.*` | **Stale local copy is used in the current build (M-N5).** |
| `figures/fig_architecture.pdf` (`04:11`) | hand-drawn, local only | The artwork contradicts the revised framing (M-N6). |

`\analysisdir` defaults to `../../analysis`, which does not exist in this workspace (`analysis_repo`). The current PDFs therefore show placeholders or local copies, as in round 1.

### 4.2 Content: placeholder vs columns, rows and tests

| ID | Location | Paper asks for | Pipeline produces | Severity | Fix |
|---|---|---|---|---|---|
| **B4** | `05:62,78`; `s4:130`; `00:18`; `06:61`; `07:17-18`; `08:12` | Primary endpoint: "share of EVs whose stay contains **no infeasible step**", "per N, Table T3b". | No table, figure or test computes it. `tables.py:360-391` (T3b) has no such column and is pooled over all N. `make_all.py:36` TEST_METRICS omits it. The raw log has only `n_evs_touched_infeasible` = EVs with **p_i < l_i at least once** (`SCHEMA_ADDITIONS.md:21`, `wrapper.py:383-384`). That is the Theorem 1(c) condition, not "stay contains no infeasible step". The latter is stricter: an EV can sit through an infeasible step and still receive l_i. It cannot be recomputed post hoc without per-charger step logs. | **Blocking (pre-run decision)** | Either (a) redefine the endpoint everywhere as "share of EVs with p_i ≥ l_i throughout their stay (Theorem 1(c))", which the code logs, or (b) add a per-EV "stay contained an infeasible step" counter before the runs. Then add `100·(1 − n_evs_touched_infeasible/n_evs)` per N to T3b (or T2) and to TEST_METRICS, with the test in `tests_T2.csv`. |
| M-N1 | `06:75`; `07:14` | "In penalised profit, the difference between the twins is \RES{difference in J^pen, stats/tests_T5.csv}". | `make_all.py:198-204`: `tests_T5` compares only `profit_per_ev` (raw profit). There is no J^pen test for the pair. | Major | Add `penalised_profit_eur` (and `pen_per_charger`) to the T5 comparisons, or point to `tests_T2.csv`, which holds SA-Attn vs Attention+penalty on J^pen per N. |
| M-N2 | several | Promised quantities with no output: (1) gap "as a ratio of sums" (`03:94`); (2) per N, "how often the forward limit binds" (`05:25`); (3) "energy removed by the cap" / "capped ... energy" (`03:54`, `05:63`, `06:62`); (4) "the number of such [simultaneous charge/discharge] steps is reported" (`s3:73`, also `03:98`); (5) "measured wall-clock training **and evaluation** time ... Table T6" (`s4:144`). | (1) No ratio-of-sums computation (grep). (2) No binding metric in `metrics.py` or `wrapper.py`. (3) `energy_capped_kwh` is logged (`SCHEMA_ADDITIONS.md:16`; `load.py:63` optional) but no table uses it; T3b has only counts. (4) Neither the oracle (`oracle.py`) nor SCHEMA logs simultaneous steps. (5) T6 has training time and inference ms only. | Major | For each item, either add the output (a ratio-of-sums gap row per table; a binding-steps counter in metrics.py before the runs; `energy_capped_kwh` in T3b; a count of simultaneous p⁺p⁻ steps in the oracle file; evaluation time in T6) or delete the promise. Items (2) and (4) need code changes **before the runs**. |
| m6 | `s5:18` | T2 holds "the per-N share of fallback steps". | The T2 column is "Fallback activations" (mean count per episode, `tables.py:301`). | Minor | "per-N number of fallback steps per episode". |
| m7 | `s5:59` | Fig. S3 shows "Coverage, energy not served and profit versus α". | `figures.py:484-513` panels: Δ profit, **unserved EVs (%)**, coverage. | Minor | "Coverage, share of unserved EVs and profit change versus α". |
| m8 | `s5:82-83` | TS1 reports "every method", and "the two gaps differ by \RES{difference ... Table TS1}". | `tables.py:509-520` has rows from SCALING_METHODS only (21 methods) and no difference column. | Minor | "for the main methods", and compute the difference from the two columns or add a column. |
| m10 | `06:58` | Fig. 4 shows the metrics "for each method". | FIG_CONSTRAINTS has 9 methods (`names.py:105-106`). | Minor | "for the main methods". |
| m11 | TS2 (`s5:84-86`) | Sensitivities (the paper names no method set). | The pipeline evaluates 7 learned methods (`pipeline.py:302-303`), but TS2 shows SENS_METHODS (`names.py:115-116`), which omits Attention+penalty's constrained peers (Lagrangian, Deep Sets, nolb, MLP+penalty). | Minor | Align SENS_METHODS with SENS_MAIN_METHODS or evaluate fewer methods. |
| m12 | T3b ratio column | The ratio of shortfall to the deficit bound of Eq. 13 for "the methods with the safety layer" (`s5:20`). | `tables.py:385-387` includes every `safety_layer=True` row, including `attn_proj_nolb`. Its layer uses l = p^min, so Lemma S4 does not apply. | Minor | Exclude nolb from the ratio column or footnote it. |
| m14 | `06:80` caption placeholder | Fig. 5 shows "Cost of the guarantee (€ per EV) versus N". | `figures.py:440-463` panel (b) also plots the penalty policy's overload and energy not served. | Minor | Mention panel (b) in the caption. |

The other placeholders checked match the generated outputs:
- `06:15-19` matches T1 and tests_T1.
- `06:43-46` matches T2 (transfer gap `tables.py:317-319`; EV-GNN in-distribution relabelling) and TS2 (stress).
- `06:59-62` matches T3 and T3b fallback cases.
- `06:73-74` matches T5.
- `06:88-90` matches T4, T7, T6 and TS2.
- `s5:38-39` matches T4 (dense row present).
- `s5:54-55` matches T7.
- `s5:69` matches T6.
- `s5:85-87` matches TS2 and TS3.
- `s5:97` matches fig5 (workplace_N20_rho4, `figures.py:296-372`).

---

## 5. Notation, terminology and method names

### 5.1 Method display names

`names.py` labels, the paper and the code method strings (SCHEMA §4) agree one to one:
- SA-Attn (ours), Attention+penalty, Attention+PPO-Lagrangian, Attention+safety layer without departure bound, Deep Sets+safety layer, MLP±;
- the N=50 reference, EV-GNN zero-shot/in-distribution (±layer), the six rules ±layer;
- eMPC-HiGHS, eMPC-HiGHS-24h, eMPC (Gurobi);
- the noadm/safedeploy variants, β=0, curriculum, small, penalty-dense, w=1/100;
- SA-Attn-CP(α), SA-Attn-point, Δt=5 min.

No unknown string remains. Note: `\method` (preamble) and the literal "SA-Attn" in `names.py:44-92` and `tables.py:305,488,504` must change together at G2.

### 5.2 Remaining inconsistencies

| ID | Location | Issue | Severity | Fix |
|---|---|---|---|---|
| m3 | `s2:96-97` | $m_L\sim\mathcal N(0.4,0.1^2)$ reuses $\mathcal N$ (charger set); round-1 m11a was fixed only in the main text | Minor | Write Normal(0.4, 0.1²). |
| m4 | `s2:34` | $h(\varsigma)$ vs encoder output $h_i$ (`04:27`) | Minor | Rename (e.g. $\hat h$). |
| m15 | `tables.py:142,373,482`; `figures.py:283,460`; `stats.py:392`; analysis README:53,95 | Generated tables print $J_{\mathrm{LP,ceil}}$ (subscript) while the paper uses $J^{\mathrm{LP,ceil}}$. The T3b note says "Lemma 3", but the paper's lemma is Lemma S4. The T7 header says "d − d^lo", but the paper writes $\hat d^{\mathrm{lo}}$. Figure axes say "transformer violation" where the text says "overload energy" | Minor | Use $J^{\mathrm{LP,ceil}}$, "Lemma S4" (or "Eq. 13 deficit bound"), $\hat d^{\mathrm{lo}}$ and "overload" in the generators. |
| m16 | `04:169` | A6 argument cited as "Section S2"; it is Section S2.3 (`app:a6`) | Minor | `\ref{S-app:a6}`. |
| m17 | `s4:131` | "the main tests are paired t-tests on seed means" | `stats.py:418-419`: one-sample t-test against deterministic baselines and Welch's test when seed labels differ (EV-GNN) | Minor | Add these two cases in one clause. |
| m18 | `07:64` | eMPC-HiGHS has "perfect information inside its horizon" | `mpc.py:11-12`: prices and arrivals are exact, but load and PV are EV2Gym's noisy forecasts (`05:47` is correct) | Minor | "exact prices and arrivals inside its horizon". |
| m21 | `05:41`; `03:54` | "shortfall is measured against this admitted requirement" for every method | For `*_noadm` rows the requirement is the arrival-capped one (`wrapper.py:422-423`; `SCHEMA_ADDITIONS.md:17`). The bound of Prop. S3 still holds, but the gap is measured against a larger requirement | Minor | Add "(without admission control: the capped request)" to `05:41`. |
| m27 | `03:94` | Exclusion rule names only \|J\| < 1 € | `load.py:269-271` also drops episodes whose oracle status is not optimal | Minor | Add "or a non-optimal solver status". |

Terminology (`terminology.md`) is otherwise followed. Unit consistency is unchanged from round 1 (OK).

---

## 6. Citations

### 6.1 Keys

- Cited 79; refs.bib 79; merged bib 114; `verified_refs.json` 132 entries.
- All cited keys are verified. There are no missing and no uncited keys.
- The keys added since round 1 include kiwiel2007breakpoint, wang2022efficient and reddy2026deadline. All three are verified.

### 6.2 Positioning table (Table 1, `02:83-93`): every row

| Row | Verdict | Evidence |
|---|---|---|
| EV2Gym | Supported | matrix; `citation_fixes.md:128` (38 % verbatim) |
| EV-GNN | Supported for the preprint ("(preprint)" in the cells) | matrix; `citation_fixes.md:50,120` |
| GNN-DT | **Mostly supported (m31)** | "25 to 5-100" is verbatim. "vs. **soft-penalty** optimum (abstract)": the journal abstract says only "oracle solver" (`citation_fixes.md:64,113`). "Soft-penalty" is our inference from EV2Gym's code oracle. |
| HetGPS | Supported (verbatim) | `citation_fixes.md:65,94,101` |
| Yan et al. | Supported | matrix; gap_analysis |
| Wan et al. | Supported | matrix |
| sLLF | Supported ("closely related" form) | matrix |
| Sadeghianpourhamami | Supported | matrix |
| orfanoudakis2026can | Supported at abstract level | matrix (abstract only); B2 |

### 6.3 Spot-check of 24 further citations against `paper_matrix.csv` / `citation_fixes.md`

Supported:
- 01:10-14 (qiu2023, michailidis2025, sadeghian, ye2022, korkas2024, tabas2022, altman, gu2024);
- 01:18 (EV-GNN, GNN-DT, HetGPS);
- 01:21 (wan2025, guan2024, zhang2025dumes);
- 01:29 (kiwiel2007, wang2022);
- 02:13 (diazlondono2026);
- 02:19 (gu2024, su2025);
- 02:20-21 (fan2024, zhang2025modelfree, zhang2026, eland2026, huang2026);
- 02:24 (demirel2025, zaman2026);
- 02:31 (li2023, tabas2022);
- 02:35 (markgraf2025);
- 02:40 (joshi2022);
- 02:42 (zhang2022transfer, salesortiz2024);
- 02:48 (jin2026);
- 02:55 (reddy2026);
- 02:56 (vanhuffelen2025: ELST ordering, equivalent to laxity order; christensen2025);
- 02:58-60 (dyo2022, ferragut2022, frendo2021, gabriele2026, lindemann2023);
- 04:258 (chen2021, markgraf2025);
- 05:79 (agarwal2021).

Issues:

| ID | Location | Claim | Evidence | Severity | Fix |
|---|---|---|---|---|---|
| M-N7 | `02:12` | "Optimisation-based controllers treat both requirements as hard constraints, and the Adaptive Charging Network (ACN) solves a convex program in receding horizon [lee2021adaptive]" | Matrix: ACN models energy as a **≤ constraint with a non-completion penalty**, and its limitations say "energy delivery best-effort (≤), not guaranteed". The sentence presents ACN as an example of hard departure constraints | Major | "...and the Adaptive Charging Network (ACN) solves a convex program with hard infrastructure limits and best-effort energy delivery in receding horizon [lee2021]". Keep "hard departure and transformer constraints" for diazlondono2026 only. |
| M-N8 | `s4:135` | "Five seeds is the minimum recommended by Henderson et al. [henderson2018deep]" | henderson2018 was read as an **abstract only**. The matrix records only "metrics vary strongly with seeds; calls for significance metrics". The paper does not recommend five seeds as a minimum; it shows that five-seed comparisons can mislead | Major | "Henderson et al. showed that results over few seeds can be misleading [henderson2018deep]; we therefore pre-declare ... and use ten seeds if ..." |
| m30 | `02:68` | "Two closely related **2026** works ... [orfanoudakis2026can] and [song2025longterm]" | song2025longterm is Applied Energy 2025 (refs.bib, matrix) | Minor | "Two closely related recent works". |
| m31 | `02:85` | GNN-DT gap cell "near-optimal reward vs. soft-penalty optimum (abstract)" | See 6.2 | Minor | "near-oracle reward (journal abstract)", as `citation_fixes.md:64` proposes. |
| m32 | `02:34` | "many infeasible actions map to the same projected point [kasaura2023benchmarking]" | The matrix credits kasaura2023 with the zero-gradient problem. Action aliasing is markgraf2025 Lemma 2 | Minor | Cite markgraf2025 for aliasing, or rephrase as "projection creates zero gradients [kasaura]". |
| m9 | `02:25,47`, Table 1 `02:86`; `02:55`; `02:42`; `02:45` | HetGPS (arXiv 2608.00679), reddy2026 and zhang2022transfer are preprints but are not flagged as such (Wan, Yan, Markgraf and Gabriele are). EV-GNN's "1000 charge points under 70 transformer groups" comes from a digest of the preprint (`citation_fixes.md:50`) | Minor | Add "a preprint" for HetGPS (it is central to Table 1), and add "(preprint)" to the EV-GNN numbers or check them against the published article. |

### 6.4 IEEE reference list (M-R1, residual M14)

| Problem | Evidence | Fix |
|---|---|---|
| `IEEEtran.bst` is still not used | `main.blg:4` "The style file: ieeetr.bst"; proxy blocked the download | Place the template's IEEEtran.bst next to main.tex before submission; `main.tex:52` switches automatically. |
| Lower-cased acronyms in 3 titles | `main.bbl:199` "operation of ev virtual power plant" (huang2026safe), `:223` "coordinated ev fast charging" (zaman2026graph), `:354` "Decision-focused rl for controlled ev charging" (gabriele2026forecasting) | Brace {EV}, {RL} in refs.bib and references.bib. |
| HTML entity in booktitle | `main.bbl:224` "Conference \&amp; Expo (ITEC) \&amp; ..." (zaman2026graph) | Replace `\&amp;` with `\&`. |
| Lower-case months from braced month fields | `main.bbl:49,55,182,213,350` "dec 2021", "oct 2021" | Use month macros (`month = dec`) or drop the month field. |

---

## 7. Journal requirements (SAUCIS)

| Item | Status | Note |
|---|---|---|
| Abstract ≤ 400 words, citation-free, no figure/table/section references, third person | OK | 284 words (linter), 1 `\RES` |
| 3-5 keywords, alphabetical, bold "Keywords" | OK | "Safe reinforcement learning" < "Safety layer" (space sorts first) |
| IEEE references | **Not met** | M-R1 |
| Captions: tables above, figures below | OK | Also in generated tables (`tables.py:171`) |
| "Figure n / Equation n / Table n" referencing | OK | Macros; no "Fig."/"Eq." in the sources; supplementary items print as "Table S4", "Lemma S4" |
| Number only the equations referred to later | OK | 17/17 referenced |
| ≤ 7 figures + tables in the main text | OK | 5 + 2 (+ Algorithm 1) |
| About 25 double-spaced pages | **At risk** (m24) | 8,535 words of running text before Results and Discussion are written; 23 single-spaced pages in the fallback class; saucis.cls unavailable |
| Article Information Form | OK | All 8 items, template order, before References |
| AI disclosure (tool, version, purpose; Methods/Acknowledgements + statement) | OK in text | `10:21`, `10:44-55`. Verify "accessed via Claude Cowork": the agent runs identify as Claude Code (Agent SDK) (m25). Author rewrite pending (B1 row). |
| Anonymised build | OK | No names, affiliations or link; Author/Title metadata empty; timezone hint remains (m2) |
| EPS figures | Pending (m23) | The analysis writes EPS. `fig_architecture`, `fig2_concepts` and `figS5_pipeline` exist only as PDF/SVG/PNG. |

---

## 8. Style linter

Command: `python3 style/lint.py <mirror> --profile style/style_thresholds.json`. My counts reproduce the authors' `lint_latest_*.md` exactly.

| File | Words | Restricted phrases | Sentences > 35 words | Paragraphs > 180 words | Demonstrative-opener paragraphs | "This"+verb | Consecutive same opener | Section mean sentence length | Blocking flags |
|---|---|---|---|---|---|---|---|---|---|
| manuscript.md | 9,562 | 0 | 0 | 0 | 0 | 0 | 0 | 16.6-25.4 (within 16-26) | 28 |
| supplementary.md | 7,416 | 0 | 0 | 0 | 0 | 0 | 0 | 16.2-23.1 | 34 |

All blocking flags come from the acronym check and are false positives:
- assumption labels A1-A6;
- supplementary item numbers S1-S14;
- analysis file names T1-TS3;
- FB-0/1/2;
- "SA" from SA-Attn;
- "DT4EV" from DT4EVs;
- "SAUCIS" inside a `\TODO`;
- V2G in the title.

No text action is needed. `lint.py` should learn to whitelist these patterns.

Issues the linter cannot detect are listed in Sections 4-6 and 10.

---

## 9. Algorithm boxes vs code

| Step | Paper | Code | Verdict |
|---|---|---|---|
| Non-finite raw actions → 0 | Alg. 1 `04:117` | `projection.py:39-43` (`sanitize`), `wrapper.py:350-353` | Match |
| Bounds with tolerance and l ≤ u guard | `04:57-65`, Alg. 1 `04:120` | `bounds.py:174-194` (`E ≥ B − EPS_NUM`, `min(l,u)`) | Match |
| Feasibility test (A5) | Alg. 1 `04:124` | `projection.py:46-56` | Match |
| Projection + post-check + safe bisection | Alg. 1 `04:125-129` | `projection.py:76-135` | Match |
| Fallback cases, order, tie-breaks, FB-0 under (i) | `04:209-217`, `s1:158-161` | `fallback.py:20-43` | Match |
| Actuation: full current if p ≥ u − 1e-3, else ceil on grid | Alg. 1 `04:134` | `actuation.py:57-74` | Match |
| Masks: u−l ≤ 1e-9 tokens; fallback transitions out of actor loss, kept for critic; fixed n_mb normaliser | `04:250-252`, Alg. S1 `s4:66-70` | `wrapper.py:298`, `buffers.py:121-124`, `ppo.py:161-175` | Match (entropy averaged per influential token; log-prob summed) |
| Regulariser on feasible steps, mean over n(t), stop-gradient projection | Eq. 14 `04:254-258`, Alg. S1 `s4:71` | `ppo.py:176-189` | Match |
| Reward r/(N r_ref) | `03:111`, Alg. S1 `s4:66` | `ppo.py:325,361-364` | Match |
| PPO loop: 16 envs × 128, GAE(0.99, 0.95), 10 epochs, clip 0.2, grad clip 0.5, epoch-level KL stop 0.03, linear lr decay, Adam ε 1e-5 | Alg. S1, Tab. S4 | `base.yaml:12-28`, `ppo.py:213-214` | Match |
| PPO-Lagrangian | `s4:18-22` | `ppo.py:57-60,381-384` | Match |
| Admission LP and conformal bound | `s3:11-31`, `04:228-231` | `admission.py`, `conformal.py:5-9,94-99` | Match (round 1) |

---

## 10. Further findings

| ID | Location | Issue | Severity | Fix |
|---|---|---|---|---|
| M-N5 | `s5:102` (figchain) → `supplementary/figures/figS5_pipeline.pdf` | The current supplementary PDF embeds the **stale round-1 pipeline figure**: md5 183b9bfb, 04:50, vs the updated `analysis_repo/figures_conceptual/figS5_pipeline.pdf`, md5 c76f7ab7, 07:49. The stale figure shows "optimality gap (%) relative to the LP optimum", "Wilcoxon signed-rank + Holm" and no J^pen, Lagrangian, Deep Sets or stress level. This contradicts its own caption (`s5:103`) and DD32/DD38. The figchain prefers the local copy whenever `\analysisdir` is missing | Major | Copy the updated figS5_pipeline.{pdf,png,svg} into `supplementary/figures/`. Fig. 2 is already identical. |
| M-N6 | `manuscript/figures/fig_architecture.pdf` (Figure 1, `04:11-12`) | The hand-drawn artwork (dated 02:01, never revised) says: "safety layer (closed form, provable feasibility under A1–A5)", "no penalty for the guaranteed constraints", "executed power p(t)", "β‖x − Π_t(x)‖² regulariser". This contradicts the caption ("under Assumptions A1–A6"), DD30 (no unconditional "guaranteed"), m26 ("closed-form" removed) and Eq. 14 (normalised mean over n(t) of the mean action, feasible steps only) | Major | Redraw the labels: "safety layer (exact projection; per-step feasibility under A1-A6)", "no penalty for the transformer limit or departure energy", "commanded power p(t)", "β-regulariser (Eq. 14)". |
| m1 | `analysis_repo/README.md:116-117` | The figure plan still has main Fig. 5 = fig5_example_day (round-1 m21a; the authors asked the orchestrator to align it) | Minor | Fig. 5 = fig6_cost_of_guarantee; Fig. S7 = fig5_example_day (or rename the files). |
| m2 | PDF metadata | CreationDate "+03" timezone in the anonymous PDFs (round-1 m27) | Minor | Set `\pdfinfoomitdate=1` or `SOURCE_DATE_EPOCH` for the anonymous build. |
| m19 | `s2:20`, `s3:69`, `s2:92` | "a check script of the code repository" / "closed-loop check of the code repository" | The scripts are `method/check_A1.py`, `method/check_O0.py` and the reviewer's `review/proof_check_numeric.py`, which are not in `code_repo/code` (the code tests cover related but different checks, e.g. `tests/test_round3.py:83`) | Minor | Move the scripts into `code/scripts/checks/` before release, or say "a check script released with the code". |
| m20 | `07:64` vs `05:46-47` | One baseline is described in two ways: see m18 | Minor | covered by m18 |
| m23 | Figures | EPS for the conceptual figures | Minor | Export EPS at submission. |
| m24 | Length | Section 7 table | Minor | Plan the cuts before Results are written. |
| m25 | `10:45` | Access route "via Claude Cowork": the agent sessions identify as Claude Code / Agent SDK | Minor | The authors confirm the product name in the final statement. |
| m26 | `cover_letter.md` | "SAUCIS has published work on reinforcement learning and on electric-vehicle energy forecasting" is not backed by any reference in the workspace | Minor | Cite 1-2 SAUCIS papers or drop the clause. |
| m28 | Doc and code staleness | `code/README.md:1` still has the old title "Scale-Agnostic and Constraint-Guaranteed" and `:9` says "DD1-DD29". `SCHEMA_ADDITIONS.md:37,56,130` say "main gap = oracle_profit_ceil_eur", contradicting DD32 and `load.py`. DD26 (`design_decisions.md:33`) is not marked superseded by DD32. DD37 (`:44`) lists 6 tuned methods, while the code tunes 7, incl. `attn_proj_nolb`. `algorithm.md:510` keeps trainN50 in the extended set (36-46 GPU-h), contradicting DD42(5) and the README. `nomenclature.md:48` says f^dis is "1 in our scenarios", without the 0.9 sensitivity. `analysis_repo/README.md:79` says "37 configurations" (now 43) | Minor | Update the documents (no paper change). |
| m29 | `s0` Table S1 | f^dis is listed, but the text never uses the symbol (only "discharge price factor"); δ, ψ and Ω (s2) are not listed | Minor | Drop f^dis or use it in `05:27`; local symbols are acceptable. |

---

## 11. Complete open-issue list

**Blocking (2):**
- **B2** (residual): the novelty claim rests on abstract-only closest works (R1-R3). Author library access is needed.
- **B4** (new, pre-run): the primary endpoint "share of EVs whose stay contains no infeasible step" has no logged quantity, no table and no test. The code logs only p_i < l_i (the Theorem 1(c) condition). Redefine the endpoint or add the counter before the runs, then add the column and test.

**Major (9):**
- **M-R1:** IEEE reference style (residual M14).
- **M-N1:** `tests_T5.csv` has no J^pen comparison.
- **M-N2:** five promised outputs are not produced: ratio-of-sums gap, binding rate, capped energy, simultaneous-step count, evaluation time. Two of them need code changes before the runs.
- **M-N3:** the MLP is evaluated and plotted at N ≥ 50, contrary to the paper.
- **M-N4:** `attn_proj_nolb` uses β=0.01 and fallback masking, whereas the paper says β=0.
- **M-N5:** the supplementary build embeds a stale Fig. S5.
- **M-N6:** Figure 1's artwork says "closed form", "A1–A5" and "guaranteed constraints".
- **M-N7:** ACN is wrongly attributed hard departure constraints (`02:12`).
- **M-N8:** the Henderson "five-seed minimum" attribution (`s4:135`) is unsupported.

**Minor (31):** m1-m32 as listed above, with m20 merged into m18, so 31 distinct entries (m30-m32 are citation items).

**Author tasks (not counted):**
- full-text reading for R1-R3;
- the final title;
- rewriting the AI-drafted sections and the result-sentence frames;
- writing all interpretation;
- hardware details and the repository link;
- re-running the scenario-generator statistics and the closed-form checks;
- installing IEEEtran.bst and saucis.cls.
