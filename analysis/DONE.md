# DONE — A4 Analysis and Visualization Agent

Date: 2026-10-22, revision 2 (aligned with A3's code: SCHEMA_ADDITIONS.md, DD25–DD29). Folder becomes
`analysis/`. **No real results exist; everything in `synthetic_test/` is fake.** `figures/` and `tables/`
hold only `.gitkeep` and the data-independent `tables/STAT_METHODS.md`.

## Files

| Path | Purpose |
|---|---|
| `scripts/names.py` | **single mapping** of A3's method strings → display names (terminology.md), style, groups, method sets |
| `scripts/load.py` | collect results; schema validation (`SchemaError`); smoke handling (`include_smoke`); joins both oracle profits; `gap_pct` (vs J_LP,ceil, DD26) and `gap_lp_pct` (vs J_LP); configuration family (grid / sensitivity / other, smoke suffix `_T16` accepted); conformal columns; step logs by `method` + `test_cfg_id` (parquet or CSV) |
| `scripts/stats.py` | seed-level aggregation, t and stratified-bootstrap CIs, IQM, Holm, Wilcoxon + Welch, effect sizes; empty groups return empty tables; writes `STAT_METHODS.md` (gap definitions per DD26) |
| `scripts/tables.py` | T1–T7 + TS1 (gap vs J_LP) → `.tex/.md/.csv/_values.csv`, `provenance.json` |
| `scripts/figures.py` | fig3–fig6 (main), figS1–figS4 (supplement); skips figures without data; training-size shading only when the N range covers it |
| `scripts/make_all.py` | one command, `--synthetic`, `--include-smoke` (watermark "SMOKE TEST OUTPUT – NOT RESULTS", refused inside `analysis/`), copies `fig2_concepts`, `figS5_pipeline`; `findings_numbers.md` |
| `scripts/synth.py`, `scripts/common.py` | synthetic generator (new schema: J_LP,ceil, conformal columns, eMPC-HiGHS, sensitivity configuration, fallback code −1); watermarks and helpers |
| `scripts/tests/test_pipeline.py` | 25 pytest tests |
| `figures_conceptual/fig_concepts.{svg,pdf,png}` + `make_concepts.py` | **Fig. 2**: (a) per-EV bounds, (b) projection geometry; 940 px canvas, 12 px text (≈ 6 pt at 6.5 in); numbers asserted in the script, (b) checked against `method/projection_ref.project` |
| `figures_conceptual/figS5_pipeline.{svg,pdf,png}` | Fig. S5 pipeline (renamed from fig_pipeline) |
| `figures_conceptual/fig_lower_bound.*`, `fig_projection_geometry.*` + scripts | detailed single-panel versions (slides/supplement) |
| `figures_conceptual/render.py` | SVG → PDF/PNG with Playwright/Chromium |
| `README.md` | how to run, inputs, outputs, figure plan (Fig. 1–6, S1–S5) |

## Tests and checks

- `python -m pytest analysis/scripts/tests -q` → **25 passed** (≈ 3 min). New tests cover: all A3 method strings mapped,
  with unique display names and no terms that terminology.md forbids; DD26 join (main gap vs J_LP,ceil, supplementary vs J_LP);
  fallback when `oracle_profit_ceil_eur` is missing; a **smoke-like fixture** (partial grid N ∈ {3, 5}, one seed, a
  method missing a configuration, a sensitivity configuration, `_T16` ids, all rows smoke) running end to end
  with watermarks; smoke outputs refused for `analysis/`, `analysis/figures` and any `…/analysis`; conceptual
  figures copied.
- **A3 smoke output** (`code/results/smoke/raw`, 250 episodes, 25 methods, 6 configurations) → `make_all.py
  --include-smoke --out /tmp/claude-0/smoke_analysis`: completes in 22 s. All tables and figures are produced, except
  fig5, because the smoke run has no step logs. A3's reported failure (empty `summary_table` → KeyError) was caused by smoke ids
  `…_T16` not counting as grid configurations; both causes are fixed. Nothing was written into `analysis/figures` or `analysis/tables`.
- The full synthetic run in `synthetic_test/` was regenerated with the new names. pdflatex compiles all tables (see revision 1).

## Open issues

1. fig5 needs step logs (`--log-steps`) for SA-Attn (and Attention+penalty for the overlay) on
   `workplace_N20_rho4`. Otherwise the first grid configuration with logs is used.
2. Sensitivity configurations (DD28: `_soc0.8`, `_dt5`, policy `attn_safe_dt5`) are loaded and summarised in
   `stats/summary_by_config.csv` and `tables/T2_full_per_configuration.csv`; there is no dedicated sensitivity table yet.
3. EV-GNN is labelled "(per profile)" (DD27); its seeds enter like the other seeds.
4. At 10 pt base two tables are up to 25 pt wider than 6.5 in; at SAUCIS's 9 pt they should fit (`Table.size` otherwise).
5. `findings_numbers.md` lists only values and source cells; the authors write all interpretation (SAUCIS AI policy).
