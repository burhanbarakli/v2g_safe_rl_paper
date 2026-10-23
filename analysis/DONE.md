# DONE — A4 Analysis and Visualization Agent

Date: 2026-10-23, revision 5 (G4 report, DD45–DD49): first run on the real core results in `code/results/raw`
(153,100 episodes, 30 methods, 43 configurations, 3,550 oracle episodes, 2 seeds per learned method; no extended set,
no EV-GNN, no step logs). `figures/`, `tables/`, `stats/`, `findings_numbers.md` and `load_report.txt` are produced
by `make_all.py --raw code/results/raw --out analysis`. Revisions 1–4 were built on synthetic data only.

## Changes in revision 5 (DD45–DD49)

| Item | Implementation |
|---|---|
| DD46 J^pen | `load._ensure_pen` recomputes J^pen = profit − 10·(`energy_not_served_shared_kwh` + `tr_violation_kwh`) for every method and pass; the file column `penalised_profit_eur` is not read. Energy not served and unserved EVs are the shared-requirement values (`*_own` kept). |
| DD46 bound check | after the oracle join the loader counts episodes with J^pen > J^pen_oracle + 1e-6 €; `make_all.py` stops unless the count is 0 (`--allow-bound-violations` to override). Real data: **0 of 153,100**. |
| DD47 gap | primary `gap_abs` = (J^pen_oracle − J^pen)/N, € per charger and episode; `gap_hard_abs` vs J^LP,ceil. Per-episode % gap, IQM-% and the 1 € exclusion are removed; TS1 holds relative gaps as ratios of sums (`stats.rel_gap_ratio`). |
| DD48 statistics | `summarize`: mean of the seed-averaged policy, min–max over seeds, per-seed values, replay bootstrap stratified by configuration (no t intervals). `seed_comparisons`: Wilcoxon signed-rank on replay-level differences of seed-averaged policies, Holm per test file, replay-bootstrap CI, per-seed values. STAT_METHODS.md rewritten. |
| DD45 scope | no rows/columns for runs that do not exist. T4/Fig. S2 rows = `names.T4_ROWS` (8 variants). TS2 = stress level only. Fig. S7 (`fig5_example_day`) not produced. |
| T3c/T3d | new count tables (all episodes, both seeds; per method and per N): infeasible steps, fallback activations, EVs with an infeasible step, unserved EVs (shared), overload steps, check column "unserved EVs in episodes without infeasible steps". |
| T7/Fig. S3 | per α: coverage vs target, shortfall vs shared requirement (per episode and total), gap, energy charged, known-d reference. |
| T2/Fig. 3 | absolute gap per N for SA-Attn, Deep Sets + layer, N = 50 reference, layer without departure bound, penalty/Lagrangian/MLP and the 3 safety-layer baselines with the lowest grid gap; transfer gap (SA-Attn and Deep Sets − reference) in € per charger. Main grid ρ_tr ∈ {4, 7} only (G4 pooled ρ_tr = 3 at N = 20, 200). |
| Tests | `test_smoke_never_written_into_analysis` now scans `analysis/` for the SMOKE watermark. Tests updated for J^pen/gap_abs, DD48 statistics (Wilcoxon against SciPy, Holm family), stress-only TS2; new tests for the file J^pen being ignored, the bound check, T3c check column, T4 rows, no % gap in main tables. `synth.py` keeps synthetic profits below the synthetic optimum. |

Publication polish (revision 5b):
- Captions ≤ 60 words; captions, headers, notes and row/group labels contain no design-decision ids, file names,
  internal table ids or review labels (enforced by `test_publication_text_has_no_internal_labels`).
- Terminology as in the manuscript: "Theorem 1(b)" for the per-EV departure statement (the manuscript's numbering;
  no "Corollary 1"), "safety layer", "Optimality gap (€ per charger and episode)", display names from `names.py`;
  fallback cases (i)–(iv) named as in the supplementary proof section.
- LaTeX layout for the manuscript geometry (A4, 2.5 cm margins, 10 pt article, text width 455 pt): wrapped stub column,
  narrower headers, the min–max range over seeds shown only for the primary column(s) (all ranges remain in the
  `.md`/`.csv`/`_values.csv` files), long tables continue in a second float (`\ContinuedFloat`, caption package).
  Natural widths are 0.71–0.99 of the text width, so `adjustbox` no longer scales; no overfull boxes, no float too large.
- Figures: axis labels use "Optimality gap (€ per charger and episode)"; Fig. S6 axis labels no longer overlap;
  "training sizes" label only in the top row of Figs. 3 and 4.

Results of revision 5 (real core runs, B = 10,000):
- Soft-oracle bound check: **0 of 153,100** episodes exceed J^pen_oracle + 1e-6 €.
- Corollary 1 episode check (T3c): 0 unserved EVs in episodes without infeasible steps for every non-conformal method
  with the safety layer and the departure bound; nonzero for the layer without departure bound (1,551), the conformal
  variants (α = 0.01: 81,742) and methods without the layer.
- `python -m pytest analysis/scripts/tests -q`: **41 passed** (≈ 16 min).
- Tables T1, T2, T3, T3b, T3c, T3d, T4, T5, T6, T7, TS1, TS2, TS3 (+ `_values.csv`, provenance.json, STAT_METHODS.md)
  compile with pdflatex (booktabs, adjustbox, eurosym) without errors or overfull boxes.
- Fig. S7 (`fig5_example_day`) is not produced (no step logs).

## Changes in revision 4 (DD43)

| Item | Implementation |
|---|---|
| E4 primary endpoint (N-M2) | `share_evs_stay_feasible` → `e4_pct`. It appears in T3 (pooled), in T3b **per N** and as the third row of Fig. 4 (vs N), and is tested in `stats/tests_T3.csv` together with E3 (overload) and energy not served. |
| Binding rates (M4, N-M2) | T3b per N: upper/lower aggregate limit binding (% of steps, from `agg_*_active_steps / n_steps`), p_i = l_i (% of token-steps, `lb_active_frac`), fallback cases (i)–(iv) per N |
| Cost of the guarantee (DD43(7), M3) | T5 reports Δ J^pen (primary) and Δ raw profit (secondary), penalty minus safety layer, as **ratios of sums** over matched episodes (`stats.ratio_diff`): € per EV served (denominator: admitted EVs, which are shared) and % of \|J^LP,ceil\|, each with a stratified bootstrap CI. Fig. 5 (`fig6_cost_of_guarantee`) shows both. `tests_T5.csv` covers J^pen and raw profit (consistency M-N1). |
| Statistics (DD43(5), N-M1) | primary: paired t-test on seed means; secondary: Wilcoxon on replay-level differences of seed-averaged policies; Holm per table family (one test file = one family); fixed 5 seeds. The per-configuration Wilcoxon file is removed, so no other Holm family exists. A Welch robustness column is added (not Holm-corrected). STAT_METHODS.md was rewritten to match. |
| Conformal (DD43(1)) | the main gap is now computed for conformal variants; T7 adds the gap column |
| MLP (DD43(3)) | the loader drops `mlp_safe`/`mlp_penalty` rows with N > 20 (logged) |
| Notation (consistency m15) | J^{LP,ceil}, J^{LP} superscripts, d̂^lo, "Lemma S4", "overload" in axes and headers |
| Figure mapping (m1, m21a) | file names match the manuscript `\resultfig`/`\figchain` paths exactly: main Fig. 5 = `fig6_cost_of_guarantee.pdf`, Fig. S7 = `fig5_example_day.pdf` (README table) |
| Fig. S5 (M-N5) | updated (E4, binding rates, DD43 statistics, MLP N ≤ 20). Copied to `paper_repo/supplementary/figures/figS5_pipeline.{pdf,png,svg}` (md5 identical) and `fig2_concepts.pdf` to `paper_repo/manuscript/figures/`. EPS of both conceptual figures added (consistency m23). |

Tests: **37 passed** (≈ 8 min), including new tests for MLP N ≤ 20, E4 and binding rates in T3/T3b/Fig. 4/tests_T3,
the ratio-of-sums example, tests_T5 metrics, notation in T5.tex, the conformal gap and the STAT_METHODS phrases.
Smoke analysis on A3's current smoke output (with DD43 columns): `/tmp/claude-0/smoke_analysis_r4`, completes in 42 s
with all tables and figures except Fig. S7 (no step logs) and TS3 (no literature configuration in the smoke run).

Open after revision 4: per-EV certification counts (`n_evs_certified`, `shortfall_among_certified_kwh`) and the LP
simultaneity count (N-M2 items 1, 3, 5) are not in A3's schema, so they are not tabulated. The reviewer argues that
pairing seeds by index has no basis. DD43(5) keeps the paired test, and the Welch column lets the authors check
that this choice does not drive any conclusion.

## Changes in revision 3 (kept for reference; revision 4 supersedes where they differ)

| Item | Implementation |
|---|---|
| DD32 main metric | `penalised_profit_eur` (J^pen) is the main metric. Main gap `gap_pct` = 100·(J^pen_oracle − J^pen)/\|J^pen_oracle\| uses `oracle_pen_profit_eur`. Raw profit, energy not served and overload are reported next to it (T1–T4, fig3/fig4). `gap_hard_pct` (vs J_LP,ceil) is used in T5 and TS1. `gap_lp_pct` (vs J_LP) appears only in TS1. |
| DD42-2 | conformal variants: `gap_pct` = NaN; own table T7 (profit per EV, J^pen, shortfall, coverage, EVs departed before the bound, d − d^lo, Δ profit vs known d) |
| M2 | `GAP_EPS = 1 €`; excluded episodes counted per gap type (`load_report.txt`, `findings_numbers.md`, `ds.gap_excluded`) |
| M3 | Fig 4 = energy not served and overload **vs N** (lines per method, log x, symlog y, panels per tightness); pooled bars → `figS6_constraints_pooled` |
| M4 | new T3b: infeasible steps; fallback cases (i) FB-1*, (ii) FB-2, (iii) FB-1, (iv) FB-0 (mapping in the headers and the note); shortfall / Lemma 3 deficit bound (`energy_not_served_kwh / a5_deficit_kwh`, episodes with a positive deficit, safety-layer methods); capped and admission-reduced requests and reduced energy |
| M5 | cost of the guarantee in **€ per EV** (profit / admitted EVs) and in **% of \|J_LP,ceil\|** (T5); fig6 and figS3 use € per EV |
| M6 | TS2 sensitivities (reference, SoC 0.8, Δt 5 min, throughput cost 0.05, discharge factor 0.9, penalty weights 1/100, stress ρ_tr = 3 kW); TS3 literature configuration. The penalty-weight runs moved from T4 to TS2. |
| M7 | fallback mapping in `names.FALLBACK_CASES`, table headers and STAT_METHODS.md |
| M8 / DD33–DD36 | display names for all new methods in `scripts/names.py`: Attention+PPO-Lagrangian, Attention+safety layer without departure bound, Deep Sets+safety layer, Attention+penalty with safety layer at deployment, no-admission variants, sLLF, Price-threshold, eMPC-HiGHS-24h, eMPC (Gurobi), SA-Attn trained at N = 50 (reference), EV-GNN in-distribution / zero-shot (relabelled from `in_distribution`). T2 has a transfer-gap column (SA-Attn − N = 50 reference, pp). |
| DD38 | `stats.seed_comparisons`: seed = unit; paired t-test over seeds (one-sample t-test against deterministic baselines, Welch if the seed labels differ); stratified bootstrap CI over seeds and common replays; Wilcoxon on replay-level differences as secondary; probability of improvement with a bootstrap CI; Holm per table family → `stats/tests_T1.csv`, `tests_T2.csv`, `tests_T5.csv`. Rewrote STAT_METHODS.md. |
| Families | grid (ρ_tr ∈ {4, 7}), stress (ρ_tr = 3), sensitivity (`_soc/_dt/_thr/_dpf`, also after a smoke `_T16`), literature |
| Fig S5 | pipeline updated: J^pen, soft and hard optimum, EV-GNN, DD33 baselines, 24 h MPC, stress level, DD38 statistics. Layout checked. fig2_concepts is unchanged. |
| LaTeX | tables wrapped in `adjustbox{max width=\linewidth}`; all 11 tables compile with no overfull boxes |

## Tests and checks

- `python -m pytest analysis/scripts/tests -q` → **33 passed** (≈ 6 min). New tests cover the 1 € exclusion and its
  count, no main gap for conformal variants, the EV-GNN in-distribution split, the paired t-test and PoI against
  SciPy and a hand count, the one-sample test against a deterministic baseline, the T3b fallback mapping and
  request columns, T5 units, the TS2 groups, TS3, the T2 transfer gap, all round-3 method strings, and the DD32 joins.
- **A3 smoke output** (408 episodes, 35 methods, 8 configurations incl. `_thr0.05`, `_dpf0.9`, `_dt5`, `_soc0.8`):
  `make_all.py --include-smoke --out /tmp/claude-0/smoke_analysis` completes in 34 s and produces all tables
  (TS3 is absent because the smoke run has no literature configuration). Fig 5 is skipped because the smoke run has no step logs. No method name is unknown.
  On the tiny smoke episodes, 172 of 408 episodes have |soft oracle| < 1 € and are excluded from the main gap.
- Nothing was written into `analysis/figures` or `analysis/tables`. `synthetic_test/` has been regenerated.

## Open issues

1. Lemma 3 ratio: uses the existing column `a5_deficit_kwh` (episode sum of the per-EV deficit). If the paper's
   bound is per EV, A3 needs a per-EV bound column; the ratio is then mean shortfall / bound per EV.
2. The "€ per EV" denominator is `n_evs` (EVs that arrived and were admitted). If "per EV served" should
   exclude unserved EVs, `n_evs − n_evs_unserved` is a one-line change (`load.add_derived`).
3. On short horizons (smoke, T = 16) many episodes have |J*| < 1 €. At T = 112 this should be rare. Check the count.
4. fig5 needs `--log-steps` runs for SA-Attn (and Attention+penalty) on `workplace_N20_rho4`.
5. Holm families: T1, T2, T5 tables (all methods × N × metrics in one family). DD38 asks for "family = table";
   if the main text reports only J^pen, restrict `TEST_METRICS` in `make_all.py` to that metric (less conservative).
6. `findings_numbers.md` lists values and source cells only; the authors write all interpretation (SAUCIS AI policy).
