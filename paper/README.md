# Manuscript repository (becomes `paper/` in the project)

Manuscript: *Attention-Based Safe Reinforcement Learning for Scale-Agnostic and Constraint-Guaranteed V2G Charging Control*, prepared for SAUCIS (Sakarya University Journal of Computer and Information Sciences).

Status: Abstract, Introduction, Related Work, Problem Formulation, Proposed Method, Experimental Setup, Conclusion (except result sentences), the Article Information Form and the supplementary proofs and modelling notes are written. **Results and Discussion are skeletons**: every experimental number or claim is a red `[RESULT: ...]` placeholder (`\RES{}`), every author task a blue `[TODO: ...]` (`\TODO{}`). No experiment has been run; nothing in the manuscript reports a result.

## Layout

| Path | Content |
|---|---|
| `manuscript/main.tex`, `manuscript/preamble.tex` | Main text; class switch, shared preamble (packages, macros, theorem environments) |
| `manuscript/sections/00-08, 10` | Abstract ... Conclusion, Article Information Form |
| `supplementary/supplementary.tex`, `supplementary/sections/s1-s4` | Supplementary material: S1 proofs, S2 EV2Gym modelling notes, S3 training details, S4 additional results |
| `manuscript/refs.bib` | Cited entries of both documents, generated from `lit/merged/references.bib` by `tools/extract_bib.py` |
| `manuscript/figures/` | Fig. 1 (`fig_architecture.pdf`), local copy of Fig. 2 (`fig2_concepts.pdf`) |
| `supplementary/figures/` | Local copy of Fig. S5 (`figS5_pipeline.pdf`) |
| `*.pdf`, `*_anonymous.pdf` | Current builds (with author block / double-blind) |
| `manuscript/Makefile` | `make` (both documents, two passes), `make anon`, `make bib`, `make md`, `make lint`, `make clean` |
| `manuscript.md`, `supplementary.md` | Markdown review mirrors (generated; LaTeX is authoritative) |
| `tools/extract_bib.py`, `tools/tex2md.py` | Bibliography extraction; LaTeX to Markdown conversion (resolves cross-document references) |
| `lint_round0-4.md`, `lint_final_manuscript.md`, `lint_final_supplementary.md` | Style-lint history and final reports |
| `cover_letter.md`, `declarations.md`, `DONE.md` | Submission drafts and the writing report |

Cross-references between the two documents use the `xr` package (`S-` prefix in the main text, `M-` prefix in the supplement), so both documents must be compiled twice (`make` does this).

## Results that appear automatically

Result figures and tables are read from the analysis folder `analysis/` (`analysis_repo` in the workspace; see its README) relative to `paper/`: `../../analysis/figures/fig3_zero_shot_scaling.pdf`, `fig4_constraint_satisfaction.pdf`, `fig6_cost_of_guarantee.pdf` (main Figs. 3-5), `figS1_training_curves.pdf` ... `figS4_inference_time.pdf`, `fig5_example_day.pdf` (Figs. S1-S4, S6), and `../../analysis/tables/T1.tex` (main Table 2), `T2-T7.tex`, `TS1.tex` (supplementary). Until a file exists, a red placeholder box or placeholder table is shown. Fig. 2 and Fig. S5 use the analysis output if present, else the conceptual source, else the local copy.

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

This draft was prepared with substantial help of an AI assistant (Claude, Anthropic, model claude-opus-5-5). Under the SAUCIS AI policy, AI-drafted text may serve only as a preliminary draft that the authors thoroughly revise, and AI must not report or interpret results. The authors must therefore revise all sections, write the interpretation in Results and Discussion themselves, and confirm the AI statement in the Article Information Form.
