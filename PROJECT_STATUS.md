# PROJECT STATUS

**Project:** Attention-Based Safe RL for Scale-Agnostic and Constraint-Guaranteed V2G Charging Control
**Target journal:** SAUCIS (Sakarya University Journal of Computer and Information Sciences)
**Human author / decision maker:** Gakko (kucuker@sakarya.edu.tr)
**Last update:** 2026-10-21

## Current phase
Phase 1 complete -> **waiting at gate G1** (see `review/G1_report.md`, decisions D1-D5)

## Deviations from orchestrator prompt (with justification)
- Project root is `~/Documents/v2g Rl/v2g_safe_rl_paper/` (the folder the author connected) instead of `~/v2g_safe_rl_paper/`.
- Agent roles are executed by the orchestrator session and parallel sub-agents; each agent folder has a DONE.md.
- `paper_matrix.csv` has three extra columns (`read_level`, `evidence_source`, `source_part`) for traceability.
- Style profile is preliminary (5 of 13-15 corpus papers) because publisher hosts are blocked; to be completed before Phase 5.

## Work plan
| Phase | Lead | Output | Gate | Status |
|---|---|---|---|---|
| 0 Setup | Orchestrator | structure, git, plan, journal requirements | - | done |
| 1 Literature & novelty | A1, A6 | paper matrix, gap analysis, novelty, refs, style profile | G1 | done, awaiting G1 |
| 2 Method | A2, A6 | formulation, safety layer, theorem+proof, algorithm | G2 | not started |
| 3 Implementation & pilot | A3 | code, unit tests, pilot | G3 | not started |
| 4 Full experiments & analysis | A3, A4 | raw results, stats, figures, numbers-only summary | G4 | not started |
| 5 Writing | A5, A6 (+ human for Results interpretation/Discussion, per AI policy) | draft, lint, 2 review rounds | G5 | not started |
| 6 Final package | all | manuscript, cover letter, declarations, repro package | G6 | not started |

## Completed tasks
- [x] Directory structure, git repository
- [x] A5-1 journal requirements (`paper/journal_requirements.md`)
- [x] A1 search/verification/matrix: 85 verified refs (59 from 2022+), all 9 core papers
- [x] A1 gap analysis, novelty statement, related work draft
- [x] A6-1 lint.py, terminology, preliminary style profile
- [x] G1 report

## Open tasks
- [ ] Human decisions D1-D5 (G1)
- [ ] Read paywalled papers R1-R3 once PDFs are provided; re-confirm novelty
- [ ] Complete style corpus (8-10 more papers)
- [ ] Verify (n)-flagged numbers against PDFs

## Blockers
- Paywalled key papers (R1: orfanoudakis2026can).
- Network allow-list blocks arXiv/GitHub/publishers for scripts.
- No GPU (cloud 2 CPU/7 GB; local VM 4 CPU/3 GB).

## Next gate
G1: approve contribution set (D1) and workflow decisions (D2-D5).
