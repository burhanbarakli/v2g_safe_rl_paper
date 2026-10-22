# DONE — A4 Analysis and Visualization Agent

Date: 2026-10-22, revision 4 (DD43, internal review round 2 M3/M4/N-M1/N-M2, consistency round 2). Revision 3
aligned the pipeline with DD30–DD42 and consistency items M2–M8. The folder becomes `analysis/`. **No real results exist; everything in `synthetic_test/` is fake.**
`figures/` and `tables/` contain only `.gitkeep` and the data-independent `tables/STAT_METHODS.md`.

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
