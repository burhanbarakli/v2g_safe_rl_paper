# analysis/ — statistics, tables and figures

Turns the raw result files of the experiments (`results/raw/`, schema in `code_spec.md` §2–§3) into
the statistics, tables (T1–T7, T3b–T3d, TS1–TS3), publication figures and a numbers-only summary, and holds
the conceptual diagrams. `figures/`, `tables/`, `stats/` and `findings_numbers.md` are produced by
`make_all.py` from the core runs in `code/results/raw` (2 training seeds per learned method; extended set,
other sensitivities, EV-GNN and the example-day figure not run, DD45). Everything under `synthetic_test/` is
fake test data.

## Requirements

Python 3.10+ with `pandas`, `numpy`, `scipy`, `matplotlib` (and `pytest` for the tests).
Optional: `pyarrow` to read `steps.parquet` (otherwise `steps.csv` is read).
The conceptual diagrams are rendered with `playwright` + Chromium (only needed to re-render them).

## One command

```bash
python analysis/scripts/make_all.py --raw code/results/raw --out analysis
```

Options: `--B 10000` (bootstrap resamples, default), `--B-iqm 2000`, `--no-oracle` (run without oracle
results; the optimality gap is then empty), `--include-smoke` (keep smoke rows; pipeline check only, see below),
`--allow-bound-violations` (continue although episodes exceed the soft-penalty optimum; default: stop, DD46).
Run time on the core runs (153,100 episodes): about 9 min with `--B 200`, longer with the default B = 10,000.

Pipeline test on synthetic data (writes only into `analysis/synthetic_test/`):

```bash
python analysis/scripts/make_all.py --synthetic          # generates synthetic_test/raw, then runs everything
python -m pytest analysis/scripts/tests -q                # 42 tests, ~16 min
```

Pipeline check on A3's smoke output (never into `analysis/`; refused there):

```bash
python analysis/scripts/make_all.py --raw code/results/smoke/raw --out /tmp/smoke_analysis --include-smoke --B 500
```
Smoke outputs are watermarked "SMOKE TEST OUTPUT – NOT RESULTS". Partial grids (missing methods, sizes,
tightness levels, step logs, train logs, conformal runs) are handled: empty cells show "–", figures that
lack data are skipped with a message.

## Inputs (read exactly as in code_spec.md §3)

| File | Use |
|---|---|
| `<raw>/<exp_id>/episodes.csv` | all per-episode metrics (§3.1); every listed column is required, `train_N`, `oracle_profit_eur`, `proj_dist_mean` may be empty |
| `<raw>/oracle__<test_cfg_id>/episodes.csv` | perfect-information optimum, plus `solve_time_s`, `solver_status`, `shortfall_stage_kwh` (§3.4) |
| `<raw>/<exp_id>/train_log.csv` | training curves (F2), training time (T6) (§3.3) |
| `<raw>/<exp_id>/meta.json` | optional; runs with `"smoke": true` are skipped |
| `<raw>/<exp_id>/steps.parquet` or `steps.csv` | optional, example day (F5) (§3.2); a `test_cfg_id` column is used if present |
| conformal columns `alpha`, `conformal_coverage` (alias `coverage_frac`), `n_evs_departed_before_bound`, `mean_d_minus_dlo` | T7, figS3 (SCHEMA_ADDITIONS.md §1) |
| `energy_not_served_shared_kwh`, `n_evs_unserved_shared` | shortfall against the shared admitted requirement, used for every method and pass (DD46) |
| `penalised_profit_eur` | **not used**: J^pen is recomputed as profit − 10·(shared ENS + overload energy) (DD46) |
| oracle columns `oracle_pen_profit_eur`, `oracle_profit_ceil_eur`, `profit_eur` | soft-penalty optimum (main gap), hard optimum J_LP,ceil (cost of the guarantee), J_LP (supplement) |
| `in_distribution` | EV-GNN rows with `true` are relabelled `evgnn_id` / `evgnn_safe_id` (DD35) |
| `fb1star_steps`, `fb2_steps`, `fb1_steps`, `fb0_steps`, `a5_deficit_kwh`, `n_evs_capped`, `n_evs_admission_reduced`, `energy_admission_reduced_kwh` | T3b (fallback cases (i)–(iv), Lemma 3 ratio, request pipeline) |
| `steps.parquet`/`steps.csv` columns `method`, `test_cfg_id`, `fallback_code` (−1 = none) | fig5 selects SA-Attn on `workplace_N20_rho4` (else the first grid configuration), replay 0, first seed |

The loader fails loudly (`SchemaError`) on a missing column, a non-numeric value in a numeric column,
empty required values, duplicate `(exp_id, test_cfg_id, replay_id)` rows, the same
`(method, seed, test_cfg_id, replay_id)` in two run directories, or missing oracle files. Smoke rows
(`smoke = true`) are dropped unless `--include-smoke`. All three oracle values are joined per
`(test_cfg_id, replay_id)`. After the join the loader counts the episodes with J^pen > J^pen of the soft-penalty
optimum + 1e-6 € (`load_report.txt`, `findings_numbers.md`); `make_all.py` stops if the count is not 0 (DD46).
Gaps per episode (DD47):
- `gap_abs` (**primary**): (J^pen_oracle − J^pen) / N, € per charger and episode, every method and pass;
- `gap_hard_abs`: (J^LP,ceil − profit) / N;
- relative gaps only as ratios of sums 100·Σ(J* − J)/Σ|J*| in the supplementary table TS1 (soft optimum,
  J^LP,ceil, J^LP). No per-episode percentage gap and no IQM of a percentage are reported.

Configuration families: `grid` (`<profile>_N<N>_rho<4|7>`, optional smoke suffix `_T16`), `stress`
(ρ_tr = 3 kW, DD36), `sensitivity` (`_soc`, `_dt`, `_thr`, `_dpf` tags), `literature`; T1–T7 use the grid.

Method names (column `method`, exactly the strings of A3's SCHEMA_ADDITIONS.md §4) are mapped to display
names (terminology.md), colours and markers in **one place: `scripts/names.py`**. Unknown names are still
analysed (generic style, warning).

## Outputs (under `--out`)

| Path | Content |
|---|---|
| `tables/T*.{tex,md,csv}` | LaTeX (booktabs, array, adjustbox safety net, caption package for tables continued in a second float; caption above, as SAUCIS requires; sized for a 455 pt text width; LaTeX shows the seed range only for the primary column), Markdown and display CSV with all ranges (UTF-8 with BOM) |
| `tables/T*_values.csv` | every cell as numbers: value, 95 % bootstrap CI (replays), min/max over seeds, per-seed values (`| seed k` columns), n episodes |
| `tables/provenance.json` | `table|[group / ]row|column` → list of exp_ids whose episodes entered the cell (oracle runs included for gap cells) |
| `tables/STAT_METHODS.md` | exact description of every statistic and test |
| `figures/fig*.{pdf,eps,svg,png}` | vector PDF/EPS/SVG and 300 dpi PNG (names below); `figures/data/*.csv` = plotted numbers + exp_ids |
| `stats/summary_by_config.csv`, `summary_by_N.csv`, `summary_by_N_rho.csv` | all summaries (mean, min–max over seeds, per-seed values, replay-bootstrap CI, IQM for the gap) |
| `stats/tests_T1.csv`, `tests_T2.csv`, `tests_T3.csv`, `tests_T5.csv` | comparisons per table family (DD48): Wilcoxon signed-rank on replay-level differences of the seed-averaged policies, Holm per file, stratified replay-bootstrap CI, per-seed values |
| `findings_numbers.md` | numbers-only listing of the key values with their table cell ids — no interpretation (SAUCIS AI policy) |
| `load_report.txt` | what the loader dropped or warned about |

Tables (main text unless marked S):

| Table | Content |
|---|---|
| T1 | main comparison at N ∈ {10, 20}: gap (€/charger, mean and IQM), J^pen, profit, energy not served, overload |
| T2 | zero-shot transfer vs N (main grid ρ_tr ∈ {4, 7}): absolute gap per N for SA-Attn, Deep Sets + layer, the N = 50 reference, the layer without departure bound, the penalty/Lagrangian/MLP variants and the 3 safety-layer baselines with the lowest gap; transfer gap to the N = 50 reference in € per charger (SA-Attn and Deep Sets) |
| T3 | constraint satisfaction per method, pooled over the grid: **E4**, unserved %, energy not served (shared requirement), overload, episodes with overload, shortfall / deficit bound, capped and admission-reduced requests |
| T3b | per N: E4, infeasible steps, fallback cases (i) FB-1*, (ii) FB-2, (iii) FB-1, (iv) FB-0, binding rates |
| T3c | counts per method over all episodes and both seeds: infeasible steps, fallback activations, EVs whose stay had an infeasible step, unserved EVs (shared), overload steps, check column "unserved EVs in episodes without infeasible steps" (Corollary 1, episode level) |
| T3d | the T3c counts per method and N |
| T4 | rows that were run: SA-Attn, Attention + penalty, Attention + PPO-Lagrangian, Attention + layer without departure bound, Attention + penalty with the layer at deployment, Deep Sets + layer, MLP + layer and MLP + penalty (N ≤ 20) |
| T5 | cost of the guarantee: Δ J^pen (primary) and Δ raw profit, penalty − safety layer, ratio of sums, € per EV served and % of \|J^LP,ceil\| |
| T6 | computation: training time (mean, min–max over seeds), inference ms/step vs N |
| T7 | conformal α: coverage vs target, shortfall vs the shared requirement (per episode and total), gap (€/charger), energy charged, with the known-departure reference |
| TS1 | (S) relative gaps as ratios of sums (soft optimum, J^LP,ceil, J^LP) |
| TS2 | (S) stress level ρ_tr = 3 kW per N (the only sensitivity that was run) |
| TS3 | (S) literature configuration |

Statistics (DD48, details in `tables/STAT_METHODS.md`): table cells = mean of the seed-averaged policy with
(min–max over the 2 seeds); 95 % CI = stratified bootstrap resampling test replays within configurations;
tests = Wilcoxon signed-rank on replay-level differences of seed-averaged policies, Holm per table family
(one `stats/tests_*.csv` file); no seed-level t-tests. The zero-padded MLP is reported only for N ≤ 20.

Figure plan (DD29; file names are exactly those referenced by the manuscript and supplementary `\resultfig` /
`\figchain` paths; the paper number differs from the file stem for two files):

| Paper | File (in `figures/`) | Content | Source |
|---|---|---|---|
| Fig. 1 | `fig_architecture.pdf` (paper `manuscript/figures/`) | architecture | hand-drawn, not produced here |
| Fig. 2 | `fig2_concepts.{pdf,eps,svg,png}` | (a) per-EV bounds, (b) projection geometry | `figures_conceptual/fig_concepts.*`, copied by make_all |
| Fig. 3 | `fig3_zero_shot_scaling.*` | gap (€/charger) and J^pen per charger vs N, same methods as T2 | figures.py |
| Fig. 4 | `fig4_constraint_satisfaction.*` | energy not served, overload and E4 vs N | figures.py |
| **Fig. 5** | **`fig6_cost_of_guarantee.*`** | cost of the guarantee (Δ J^pen primary, Δ profit secondary; € per EV served) vs N | figures.py |
| Fig. S1 | `figS1_training_curves.*` | training curves | figures.py |
| Fig. S2 | `figS2_ablation.*` | gap bars for the T4 rows | figures.py |
| Fig. S3 | `figS3_conformal_alpha.*` | per α: coverage vs target, shortfall vs shared requirement, gap, energy charged | figures.py |
| Fig. S4 | `figS4_inference_time.*` | inference time vs N | figures.py |
| Fig. S5 | `figS5_pipeline.{pdf,eps,svg,png}` | experimental pipeline | `figures_conceptual/figS5_pipeline.*`, copied by make_all and into `paper_repo/supplementary/figures/` |
| Fig. S6 | `figS6_constraints_pooled.*` | constraint metrics per method, pooled over N | figures.py |
| **Fig. S7** | **`fig5_example_day.*`** | example day: **not produced** (needs `--log-steps` runs, not run, DD45) | figures.py |

Figure style: Arial/Helvetica (Liberation Sans fallback), 8 pt, full width 6.5 in, TrueType embedded;
Okabe–Ito colours plus markers and line styles (solid = safety layer,
dashed = penalty/raw) so all figures read in greyscale; CI bands are opaque (EPS-safe). Fig. S1 bands show the
min–max over seeds.

## Integrity safeguards

- `synth.py` writes a `SYNTHETIC_DATA` sentinel and `"synthetic": true` into every `meta.json`.
  When the loader sees it, every figure gets the title "SYNTHETIC TEST DATA – NOT RESULTS" plus a
  watermark, every table caption/CSV/Markdown and `findings_numbers.md` carry the same mark, and
  `make_all.py` refuses to write outside a `synthetic_test/` directory.
- `findings_numbers.md` contains only measured values and their source cells.
- Smoke outputs ("SMOKE TEST OUTPUT – NOT RESULTS") are refused inside `analysis/`; the test
  `test_smoke_never_written_into_analysis` scans `analysis/` for that watermark.
- The soft-penalty bound check (DD46) stops the pipeline if any episode exceeds the optimum.

## Conceptual figures (`figures_conceptual/`)

`fig_concepts` (Fig. 2, generated by `make_concepts.py`: compact two-panel version, 12 px text on a 940 px
canvas ≈ 6 pt at 6.5 in) and `figS5_pipeline` (hand-written), as SVG (source), PDF and PNG. The detailed
single-panel versions `fig_lower_bound` and `fig_projection_geometry` (scripts `make_lower_bound.py`,
`make_projection_geometry.py`) are kept for slides or the supplement. All numbers in the diagrams come from
a concrete example checked in the scripts (projection against `method/projection_ref.py` and brute force).
Re-render: `python3 figures_conceptual/render.py`.
