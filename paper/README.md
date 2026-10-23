# Manuscript repository (becomes `paper/` in the project)

Manuscript (working title, DD45; the authors choose the final title): *Size-Agnostic Safe Reinforcement Learning for V2G Charging with a Feasibility-Preserving Safety Layer*, prepared for SAUCIS (Sakarya University Journal of Computer and Information Sciences).

Status (2026-10-23, Option A of the G4 decision): the paper reports the core experiments as run (2 seeds per learned method; no extended set, no EV-GNN, no example-day figure, sensitivity only the stress rating ρ_tr = 3 kW). It is framed around the feasibility-preserving safety layer and size-agnostic safe control, with attention and Deep Sets as the two set architectures.
- **Results:** filled with measured values from `../analysis/` (tables, stats, findings_numbers.md). Every number is listed with its source cell in `results_number_sources.md`.
- **Discussion:** a DRAFT that starts with a visible TODO; the authors must rewrite the interpretation (SAUCIS AI policy).
- **Author TODOs:** the AI statement and the AI sentence in the Acknowledgments.
- **Turkish version:** `manuscript_tr/` and `supplementary_tr/` are **not** updated to this state.

## Layout

| Path | Content |
|---|---|
| `manuscript/main.tex`, `manuscript/preamble.tex` | Main text; class switch, shared preamble (packages, macros, theorem environments) |
| `manuscript/sections/00-08, 10` | Abstract ... Conclusion, Article Information Form |
| `supplementary/supplementary.tex`, `supplementary/sections/s0-s5` | Supplementary material: Table S1 nomenclature (s0), S1 proofs, S2 EV2Gym modelling notes and A6, S3 request pipeline and perfect-information optima, S4 compared methods, training, tuning, statistics and compute, S5 additional results |
| `manuscript/refs.bib` | Cited entries of both documents, generated from `lit/merged/references.bib` by `tools/extract_bib.py` |
| `manuscript/figures/` | Fig. 1 (`fig_architecture.pdf`), local copy of Fig. 2 (`fig2_concepts.pdf`) |
| `supplementary/figures/` | Local copy of the pipeline figure (`figS5_pipeline.pdf`; stale, no longer included) |
| `*.pdf`, `*_anonymous.pdf` | Current builds (with author block / double-blind) |
| `manuscript/Makefile` | `make` (both documents, two passes), `make anon`, `make bib`, `make md`, `make lint`, `make clean` |
| `manuscript.md`, `supplementary.md` | Markdown review mirrors (generated; LaTeX is authoritative) |
| `tools/extract_bib.py`, `tools/tex2md.py` | Bibliography extraction; LaTeX to Markdown conversion (resolves cross-document references) |
| `lint_latest_main.md`, `lint_latest_supplementary.md` (and older `lint_*`) | Style-lint reports; latest = revision round 1 |
| `response_to_internal_review_round1.md`, `response_to_internal_review_round2.md` | Item-by-item responses to the internal reviews and consistency reports, rounds 1 and 2 |
| `cover_letter.md`, `declarations.md`, `DONE.md` | Submission drafts and the writing report |

Cross-references between the two documents use the `xr` package (`S-` prefix in the main text, `M-` prefix in the supplement), so both documents must be compiled twice (`make` does this).

## Results that appear automatically

Generated results are read from `\analysisdir` (default `../../analysis`, i.e. `analysis/` next to `paper/`; override with `pdflatex "\def\analysisdir{<path>}\input{main}"`), and otherwise from local copies in `manuscript/tables`, `manuscript/figures`, `supplementary/tables` or `supplementary/figures`. Until a file exists, a red placeholder box or placeholder table is shown. The build was tested against `analysis_repo/synthetic_test` (no errors, no undefined references).

Figure and table mapping (results as run):

| Paper | File |
|---|---|
| Fig. 1 | `manuscript/figures/fig_architecture.pdf` |
| Fig. 2 | `fig2_concepts.pdf` |
| Fig. 3 | `fig3_zero_shot_scaling.pdf` |
| Fig. 4 | `fig4_constraint_satisfaction.pdf` (vs N, incl. E4) |
| Fig. 5 | `fig6_cost_of_guarantee.pdf` |
| Table 1 | positioning (hand-written) |
| Table 2 | `T1.tex` (`tab:t1`) |
| Figs. S1-S4 | `figS1_training_curves`, `figS2_ablation` (T4 variants), `figS3_conformal_alpha`, `figS4_inference_time` |
| Fig. S5 | `figS6_constraints_pooled.pdf` (the pipeline figure is no longer included because its text shows the old plan: 5 seeds, EV-GNN, 24-h MPC, t-tests) |
| Tables S5-S16 | `T2, T3, T3b, T3c, T3d, T5, T4, T7, T6, TS1, TS2, TS3` in this order (S1 nomenclature, S2 scenarios, S3 features, S4 hyperparameters) |
| Algorithms S1, S2 | safety-layer step, PPO training loop |
| not produced | `fig5_example_day` (not run, DD45) |

Generated tables need `adjustbox` (loaded in `preamble.tex` when installed).

## Official SAUCIS template

The template files could not be downloaded in the drafting workspace. To switch to the journal layout:

1. Download the SAUCIS LaTeX template (Overleaf "Sakarya University Journal of Computer and Information Sciences LaTeX Template", or `saucisLLatex_v_6.zip` linked from the DergiPark author guidelines).
2. Copy `saucis.cls`, `IEEEtran.bst` and any other class files into `manuscript/` next to `main.tex`.
3. Rebuild. `main.tex` detects `saucis.cls` automatically (`\documentclass[final,9pt,onecolumn]{saucis}`), and the bibliography switches from `ieeetr` (fallback, because `IEEEtran.bst` is not installed here) to `IEEEtran`.
4. The template probably defines its own front-matter macros (author/affiliation/ORCID, article history, keywords). Move the title, author block, abstract and keywords into those macros by hand; the section files need no change.
5. The template asks for figures as separate EPS files. Convert the PDF figures if the editorial office requires it.

Without `saucis.cls`, the standard `article` class (10 pt, A4, no paragraph indentation as SAUCIS requires) is used.

## Double-blind version

`\anonymousfalse` in `main.tex` gives the version with author information; `\anonymoustrue` (or `make anon`) hides authors, affiliations, contributions, acknowledgement names and the repository link, and keeps all other statements (including the AI statement).

## AI use

This draft was prepared with substantial help of an AI assistant (Claude, Anthropic, model claude-opus-5-5, via Claude Cowork): literature work, method derivation and proofs, experiment design, code and analysis pipeline, drafting of all sections and conceptual figures. The AI statement (DD41) now lists all of these. Under the SAUCIS AI policy, AI-drafted text may serve only as a preliminary draft that the authors thoroughly revise, and AI must not report or interpret results. The authors must therefore revise all sections, write the interpretation in Results and Discussion themselves, and confirm the AI statement in the Article Information Form.
