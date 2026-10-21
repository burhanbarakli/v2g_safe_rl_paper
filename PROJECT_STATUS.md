# PROJECT STATUS

**Project:** Attention-Based Safe RL for Scale-Agnostic and Constraint-Guaranteed V2G Charging Control
**Target journal:** SAUCIS (Sakarya University Journal of Computer and Information Sciences)
**Human author / decision maker:** Gakko (kucuker@sakarya.edu.tr)
**Last update:** 2026-10-21

## Current phase
Phase 1 (Literature and novelty) -> next gate **G1**

## Deviations from orchestrator prompt (with justification)
- Project root is `~/Documents/v2g Rl/v2g_safe_rl_paper/` (the folder the author connected) instead of `~/v2g_safe_rl_paper/`, so that all outputs stay in the author's chosen folder.
- Agent roles A1-A6 are executed by the orchestrator session and by sub-agents; every agent still writes to its own folder and leaves a DONE.md.

## Work plan
| Phase | Lead | Output | Gate | Status |
|---|---|---|---|---|
| 0 Setup | Orchestrator | structure, git, plan, journal requirements | - | in progress |
| 1 Literature & novelty | A1, A6 | paper matrix, gap analysis, novelty, refs, style profile | G1 | in progress |
| 2 Method | A2, A6 | formulation, safety layer, theorem+proof, algorithm | G2 | not started |
| 3 Implementation & pilot | A3 | code, unit tests, pilot | G3 | not started |
| 4 Full experiments & analysis | A3, A4 | raw results, stats, figures, findings | G4 | not started |
| 5 Writing | A5, A6 | draft, lint, 2 review rounds | G5 | not started |
| 6 Final package | all | manuscript, cover letter, declarations, repro package | G6 | not started |

## Completed tasks
- [x] Directory structure and git repository created.

## Open tasks
- [ ] A5-1 journal requirements (`paper/journal_requirements.md`)
- [ ] A1 literature search, verification, paper matrix
- [ ] A1 gap analysis, novelty statement, related work draft, style corpus
- [ ] A6-1 style profile, lint script, terminology
- [ ] G1 report

## Blockers
- Compute: no GPU on either the cloud workspace (2 CPU, 7 GB) or the local VM (4 CPU, 3 GB, aarch64). To be discussed at G2/G3 (experiment grid size).

## Next gate
G1: literature review + novelty statement.
