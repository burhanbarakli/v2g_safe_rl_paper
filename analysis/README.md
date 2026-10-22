# analysis/ — statistics, tables and figures

Turns the raw result files of the experiments (`results/raw/`, schema in `code_spec.md` §2–§3) into
the statistics, tables (T1–T7), publication figures (F2–F9) and a numbers-only summary, and holds the
conceptual diagrams. No result in this folder is produced here: `figures/` and `tables/` stay empty
until `make_all.py` is run on real raw results. Everything under `synthetic_test/` is fake test data.

## Requirements

Python 3.10+ with `pandas`, `numpy`, `scipy`, `matplotlib` (and `pytest` for the tests).
Optional: `pyarrow` to read `steps.parquet` (otherwise `steps.csv` is read).
The conceptual diagrams are rendered with `playwright` + Chromium (only needed to re-render them).

## One command

```bash
python analysis/scripts/make_all.py --raw results/raw --out analysis
```

Options: `--B 10000` (bootstrap resamples, default), `--B-iqm 2000`, `--no-oracle` (run without oracle
results; the optimality gap is then empty), `--include-smoke` (keep smoke rows; pipeline check only, see below). Expected run time: about 8 min for the synthetic grid at 30 % of the replays (B = 10,000); the full real grid takes longer (lower `--B` for quick checks)
(about 2 min for a full synthetic grid with 30 % of the replays).

Pipeline test on synthetic data (writes only into `analysis/synthetic_test/`):

```bash
python analysis/scripts/make_all.py --synthetic          # generates synthetic_test/raw, then runs everything
python -m pytest analysis/scripts/tests -q                # 33 tests, ~6 min
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
| `penalised_profit_eur` (J^pen, DD32) | main comparison metric (computed as profit − 10·ENS − 10·overload if absent) |
| oracle columns `oracle_pen_profit_eur`, `oracle_profit_ceil_eur`, `profit_eur` | soft-penalty optimum (main gap), hard optimum J_LP,ceil (cost of the guarantee), J_LP (supplement) |
| `in_distribution` | EV-GNN rows with `true` are relabelled `evgnn_id` / `evgnn_safe_id` (DD35) |
| `fb1star_steps`, `fb2_steps`, `fb1_steps`, `fb0_steps`, `a5_deficit_kwh`, `n_evs_capped`, `n_evs_admission_reduced`, `energy_admission_reduced_kwh` | T3b (fallback cases (i)–(iv), Lemma 3 ratio, request pipeline) |
| `steps.parquet`/`steps.csv` columns `method`, `test_cfg_id`, `fallback_code` (−1 = none) | fig5 selects SA-Attn on `workplace_N20_rho4` (else the first grid configuration), replay 0, first seed |

The loader fails loudly (`SchemaError`) on a missing column, a non-numeric value in a numeric column,
empty required values, duplicate `(exp_id, test_cfg_id, replay_id)` rows, the same
`(method, seed, test_cfg_id, replay_id)` in two run directories, or missing oracle files. Smoke rows
(`smoke = true`) are dropped unless `--include-smoke`. All three oracle values are joined per
`(test_cfg_id, replay_id)`. Optimality gap per episode = `100·(J* − J^π)/|J*|`, **NaN if |J*| < 1 €**
(number of excluded episodes in `load_report.txt` and `findings_numbers.md`, consistency M2):
- `gap_pct` (**main**, DD32): penalised profit J^pen vs the soft-penalty optimum, for every method incl. the conformal variants (DD43(1));
- `gap_hard_pct`: raw profit vs J_LP,ceil (cost of the guarantee, T5, TS1);
- `gap_lp_pct`: raw profit vs J_LP (supplementary TS1 only, DD42-4).

Configuration families: `grid` (`<profile>_N<N>_rho<4|7>`, optional smoke suffix `_T16`), `stress`
(ρ_tr = 3 kW, DD36), `sensitivity` (`_soc`, `_dt`, `_thr`, `_dpf` tags), `literature`; T1–T7 use the grid.

Method names (column `method`, exactly the strings of A3's SCHEMA_ADDITIONS.md §4) are mapped to display
names (terminology.md), colours and markers in **one place: `scripts/names.py`**. Unknown names are still
analysed (generic style, warning).

## Outputs (under `--out`)

| Path | Content |
|---|---|
| `tables/T1..T7.{tex,md,csv}` | LaTeX (booktabs + adjustbox `max width`; caption above, as SAUCIS requires), Markdown and display CSV (UTF-8 with BOM, opens in Word/Excel) |
| `tables/T*_values.csv` | every cell as numbers: value, std, 95 % CI bounds, CI type, unit (seed/replay), n episodes |
| `tables/T2_full_per_configuration.csv` | all methods × all configurations incl. stress, sensitivity and literature (supplement) |
| `tables/provenance.json` | `table|[group / ]row|column` → list of exp_ids whose episodes entered the cell (oracle runs included for gap cells) |
| `tables/STAT_METHODS.md` | exact description of every statistic and test |
| `figures/fig*.{pdf,eps,svg,png}` | vector PDF/EPS/SVG and 300 dpi PNG (names below); `figures/data/*.csv` = plotted numbers + exp_ids |
| `stats/summary_by_config.csv`, `summary_by_N.csv`, `summary_by_N_rho.csv` | all summaries (mean, std, t and bootstrap CIs, IQM for the gap) |
| `stats/tests_T1.csv`, `tests_T2.csv`, `tests_T3.csv`, `tests_T5.csv` | seed-level comparisons per table family (DD43(5)): paired t on seed means (primary), Wilcoxon on replays (secondary), Holm per file, bootstrap CI, PoI |
| `findings_numbers.md` | numbers-only listing of the key values with their table cell ids — no interpretation (SAUCIS AI policy) |
| `load_report.txt` | what the loader dropped or warned about |

Tables (main text unless marked S):

| Table | Content |
|---|---|
| T1 | main comparison at N ∈ {10, 20}: J^pen, optimality gap (+ IQM), profit, energy not served, overload |
| T2 | zero-shot transfer vs N: J^pen and profit per charger, gap, unserved %, overload, fallback, transfer gap to the N = 50 reference (DD36) |
| T3 | constraint satisfaction per method, pooled over the grid: **E4** (EVs without infeasible step, primary endpoint), unserved %, energy not served, overload, episodes with overload, shortfall / deficit bound (Lemma S4), capped and admission-reduced requests |
| T3b | **per N**: E4, infeasible steps, fallback cases (i) FB-1*, (ii) FB-2, (iii) FB-1, (iv) FB-0, binding rates of the upper/lower aggregate limit (% of steps) and of the departure lower bound (% of token-steps) |
| T4 | ablation: 2 × 2 (attention/MLP × safety layer/penalty), DD33 baselines, evaluation variants, ablations |
| T5 | cost of the guarantee (DD43(7)): Δ J^pen (primary) and Δ raw profit (secondary), penalty − safety layer, ratio of sums, € per EV served and % of \|J^LP,ceil\| |
| T6 | computation: training time, inference ms/step vs N |
| T7 | conformal α trade-off (gap, profit per EV, J^pen, shortfall, coverage, EVs departed before the bound, d − d̂^lo, Δ profit vs known d) |
| TS1 | (S) raw-profit gaps vs J_LP,ceil and J_LP |
| TS2 | (S) sensitivities: reference, SoC 0.8, Δt 5 min, throughput cost 0.05, discharge factor 0.9, penalty weights 1/100, ρ_tr = 3 kW (M6) |
| TS3 | (S) literature configuration (M6) |

Statistics files (DD38, DD43(5)): `stats/tests_T1.csv` (J^pen, ENS, overload; N ∈ {10, 20}), `tests_T2.csv` (same,
all N), `tests_T3.csv` (E4, E3 overload, ENS), `tests_T5.csv` (safety layer vs penalty twin: J^pen primary, raw profit
secondary). Primary: paired t-test on seed means; secondary: Wilcoxon signed-rank on replay-level differences of
seed-averaged policies; Holm per table family (one file = one family); stratified bootstrap CIs, probability of
improvement; Welch on seed means as a robustness column. `stats/summary_*.csv` hold all summaries.
The zero-padded MLP is reported only for N ≤ 20 (DD43(3)).

Figure plan (DD29; file names are exactly those referenced by the manuscript and supplementary `\resultfig` /
`\figchain` paths; the paper number differs from the file stem for two files):

| Paper | File (in `figures/`) | Content | Source |
|---|---|---|---|
| Fig. 1 | `fig_architecture.pdf` (paper `manuscript/figures/`) | architecture | hand-drawn, not produced here |
| Fig. 2 | `fig2_concepts.{pdf,eps,svg,png}` | (a) per-EV bounds, (b) projection geometry | `figures_conceptual/fig_concepts.*`, copied by make_all |
| Fig. 3 | `fig3_zero_shot_scaling.*` | J^pen per charger and optimality gap vs N | figures.py |
| Fig. 4 | `fig4_constraint_satisfaction.*` | energy not served, overload and E4 vs N | figures.py |
| **Fig. 5** | **`fig6_cost_of_guarantee.*`** | cost of the guarantee (Δ J^pen primary, Δ profit secondary; € per EV served) vs N | figures.py |
| Fig. S1 | `figS1_training_curves.*` | training curves | figures.py |
| Fig. S2 | `figS2_ablation.*` | ablation bars | figures.py |
| Fig. S3 | `figS3_conformal_alpha.*` | conformal α trade-off | figures.py |
| Fig. S4 | `figS4_inference_time.*` | inference time vs N | figures.py |
| Fig. S5 | `figS5_pipeline.{pdf,eps,svg,png}` | experimental pipeline | `figures_conceptual/figS5_pipeline.*`, copied by make_all and into `paper_repo/supplementary/figures/` |
| Fig. S6 | `figS6_constraints_pooled.*` | constraint metrics per method, pooled over N | figures.py |
| **Fig. S7** | **`fig5_example_day.*`** | example day (needs `--log-steps`) | figures.py |

Figure style: Arial/Helvetica (Liberation Sans fallback), 8 pt, full width 6.5 in, TrueType embedded;
Okabe–Ito colours plus markers and line styles (solid = safety layer, Okabe–Ito colours plus markers and line styles (solid = safety layer,
dashed = penalty/raw) so all figures read in greyscale; CI bands are opaque (EPS-safe).

## Integrity safeguards

- `synth.py` writes a `SYNTHETIC_DATA` sentinel and `"synthetic": true` into every `meta.json`.
  When the loader sees it, every figure gets the title "SYNTHETIC TEST DATA – NOT RESULTS" plus a
  watermark, every table caption/CSV/Markdown and `findings_numbers.md` carry the same mark, and
  `make_all.py` refuses to write outside a `synthetic_test/` directory.
- `findings_numbers.md` contains only measured values and their source cells.

## Conceptual figures (`figures_conceptual/`)

`fig_concepts` (Fig. 2, generated by `make_concepts.py`: compact two-panel version, 12 px text on a 940 px
canvas ≈ 6 pt at 6.5 in) and `figS5_pipeline` (hand-written), as SVG (source), PDF and PNG. The detailed
single-panel versions `fig_lower_bound` and `fig_projection_geometry` (scripts `make_lower_bound.py`,
`make_projection_geometry.py`) are kept for slides or the supplement. All numbers in the diagrams come from
a concrete example checked in the scripts (projection against `method/projection_ref.py` and brute force).
Re-render: `python3 figures_conceptual/render.py`.
