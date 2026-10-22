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
results; the optimality gap is then empty), `--include-smoke` (keep smoke rows; pipeline check only, see below). Expected run time on a laptop: a few minutes
(about 2 min for a full synthetic grid with 30 % of the replays).

Pipeline test on synthetic data (writes only into `analysis/synthetic_test/`):

```bash
python analysis/scripts/make_all.py --synthetic          # generates synthetic_test/raw, then runs everything
python -m pytest analysis/scripts/tests -q                # 25 tests, ~3 min
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
| oracle column `oracle_profit_ceil_eur` | J_LP,ceil, reference of the main optimality gap (DD26) |
| `steps.parquet`/`steps.csv` columns `method`, `test_cfg_id`, `fallback_code` (−1 = none) | fig5 selects SA-Attn on `workplace_N20_rho4` (else the first grid configuration), replay 0, first seed |

The loader fails loudly (`SchemaError`) on a missing column, a non-numeric value in a numeric column,
empty required values, duplicate `(exp_id, test_cfg_id, replay_id)` rows, the same
`(method, seed, test_cfg_id, replay_id)` in two run directories, or missing oracle files. Smoke rows
(`smoke = true`) are dropped unless `--include-smoke`. Both oracle profits are joined per
`(test_cfg_id, replay_id)`. Optimality gap per episode = `100·(J* − J^π)/|J*|`: **main** (`gap_pct`,
all main tables and figures) with J* = J_LP,ceil (`oracle_profit_ceil_eur`, DD26); **supplementary**
(`gap_lp_pct`, table TS1) with J* = J_LP (oracle `profit_eur`). NaN if |J*| < 1e-9 or the oracle status is
not optimal. Configurations are classified as `grid` (`<profile>_N<N>_rho<rho>`, optional smoke suffix
`_T16`), `sensitivity` (`_soc0.8`, `_dt5`) or `other` (literature); T1–T7 use the grid only.

Method names (column `method`, exactly the strings of A3's SCHEMA_ADDITIONS.md §4) are mapped to display
names (terminology.md), colours and markers in **one place: `scripts/names.py`**. Unknown names are still
analysed (generic style, warning).

## Outputs (under `--out`)

| Path | Content |
|---|---|
| `tables/T1..T7.{tex,md,csv}` | LaTeX (booktabs; caption above, as SAUCIS requires), Markdown and display CSV (UTF-8 with BOM, opens in Word/Excel) |
| `tables/T*_values.csv` | every cell as numbers: value, std, 95 % CI bounds, CI type, unit (seed/replay), n episodes |
| `tables/T2_full_per_configuration.csv` | all methods × all 37 configurations (supplement) |
| `tables/provenance.json` | `table|[group / ]row|column` → list of exp_ids whose episodes entered the cell (oracle runs included for gap cells) |
| `tables/STAT_METHODS.md` | exact description of every statistic and test |
| `figures/fig*.{pdf,eps,svg,png}` | vector PDF/EPS/SVG and 300 dpi PNG (names below); `figures/data/*.csv` = plotted numbers + exp_ids |
| `tables/TS1.*` | supplementary: optimality gap w.r.t. J_LP,ceil and J_LP per N |
| `stats/summary_by_config.csv`, `summary_by_N.csv`, `summary_by_N_rho.csv` | all summaries (mean, std, t and bootstrap CIs, IQM for the gap) |
| `stats/paired_tests.csv` | Wilcoxon signed-rank (Holm across N and across all configurations), Welch t-test, effect sizes |
| `findings_numbers.md` | numbers-only listing of the key values with their table cell ids — no interpretation (SAUCIS AI policy) |
| `load_report.txt` | what the loader dropped or warned about |

Tables: T1 main comparison at N ∈ {10, 20}; T2 zero-shot scaling vs N; T3 constraint satisfaction;
T4 ablation (attention/MLP × safety layer/penalty + further ablations); T5 cost of the guarantee;
T6 computation; T7 conformal α trade-off (only if conformal runs exist); TS1 gap w.r.t. J_LP (supplement).

Figure plan (DD29: at most 7 figures in the main text):

| Paper | File | Source |
|---|---|---|
| Fig. 1 | `figures/fig_architecture.*` (project root `figures/`) | existing, hand-drawn |
| Fig. 2 | `fig2_concepts.*` (a) per-EV bounds, (b) projection geometry | `figures_conceptual/fig_concepts.*`, copied by make_all |
| Fig. 3 | `fig3_zero_shot_scaling.*` profit per charger and optimality gap vs N | figures.py |
| Fig. 4 | `fig4_constraint_satisfaction.*` energy not served, transformer violations | figures.py |
| Fig. 5 | `fig5_example_day.*` aggregate power, C(t), −C_rev(t), Σl/Σu band, price | figures.py (needs `--log-steps`) |
| Fig. 6 | `fig6_cost_of_guarantee.*` cost of the guarantee vs N | figures.py |
| Fig. S1 | `figS1_training_curves.*` | figures.py |
| Fig. S2 | `figS2_ablation.*` | figures.py |
| Fig. S3 | `figS3_conformal_alpha.*` | figures.py |
| Fig. S4 | `figS4_inference_time.*` | figures.py |
| Fig. S5 | `figS5_pipeline.*` experimental pipeline | `figures_conceptual/figS5_pipeline.*`, copied by make_all | Fonts: Arial/Helvetica (Liberation Sans fallback), 8 pt, full width 6.5 in,
TrueType embedded; Okabe–Ito colours plus markers and line styles (solid = safety layer,
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
