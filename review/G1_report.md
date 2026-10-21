# G1 Report — Literature review and novelty (Phase 1)

Date: 2026-10-21 · Orchestrator → human author (Gakko) · Status: **waiting for your decision**

## 1. What was done
- Project skeleton and git repository created in your folder `v2g Rl/v2g_safe_rl_paper/` (Phase 0).
- Four literature sub-agents screened about 1,440 records (OpenAlex, Crossref, Semantic Scholar, citation neighbourhoods of EV2Gym) and included **85 papers**. Every reference passed an automatic DOI/arXiv check with a title match (`literature/verified_refs.json`). 59 of the 85 are from 2022 or later.
- All nine papers of Section 2.3 are in the matrix. The DOIs are correct. Corrections: EV2Gym is 2025 (T-ITS 26(2), online Dec 2024) with five authors. DuMES, EV-GNN and Wu 2023 have additional authors. The laxity paper is Christensen et al. The SIGMETRICS paper is Ferragut et al. (PER 49(3)).
- Gap analysis, novelty statement, risk register and a Related Work draft are written (`literature/gap_analysis.md`, `literature/related_work_draft.md`).
- SAUCIS rules extracted with source links (`paper/journal_requirements.md`).
- Style linter, terminology list and a preliminary style profile (5 of 13-15 corpus papers measured) are ready (`style/`).

## 2. Main findings
1. **The combination appears open.** No paper we could read combines all three of: (i) one set of policy weights for any station size, (ii) a per-step guarantee for both the departure energy and the transformer limit, proven as an invariant, and (iii) zero-shot optimality gaps against a hard-constrained optimum.
2. **No single ingredient is new.** The work must be presented as a combination:
   - The projection form clip(a − λ, l, u) is the same as the smoothed least-laxity-first rule (Chen et al., TSG 2022), and the same as the textbook box-plus-sum projection.
   - HPC-RL (arXiv 2608.03409, Aug 2026) already uses our per-EV lower bound l_i, but gives no proof and no station limit.
   - GNN-DT and HetGPS already transfer one policy across sizes. Both handle the constraints with penalties or filters, and HetGPS explicitly disclaims pointwise feasibility.
   - Sadeghianpourhamami et al. (TSG 2020) already did train-small/test-large against an oracle.
3. **Biggest open risk (R1).** Orfanoudakis et al. 2026, *"Can A.I. Revolutionize EV Dispatch?"* (ECML PKDD 2025 workshops, CCIS 2841, DOI 10.1007/978-3-032-19102-1_1) is from the EV2Gym group. Its abstract describes a GNN + TD3 policy with a safety layer that should "preserve EV charging requirements and enforce power limits". Only the abstract is accessible here (Springer paywall).
4. **Benchmark caveat.** EV2Gym's built-in optimal solution (v2.0.0 code) treats departure energy as a soft penalty. A hard-constrained optimum must be added by us and documented.
5. **SAUCIS rules that affect the plan:**
   - English only; abstract ≤ 400 words; 3-5 keywords; IEEE references; LaTeX template (Overleaf).
   - Double-blind review, so an anonymised PDF is needed.
   - Similarity must be below 20%. No fees. CC BY-NC 4.0.
   - An Article Information Form is required, and it includes an **AI statement**.
   - **AI policy (11 Apr 2025):** AI "must not be employed to produce, report, or interpret research findings". AI-drafted abstract, introduction, literature review or discussion text is acceptable only as a preliminary draft that the authors thoroughly revise.

## 3. Open problems
| # | Problem | Effect | Proposed handling |
|---|---|---|---|
| P1 | R1-R3: full texts of orfanoudakis2026can, eland2026safe, song2025longterm (also DuMES, Sales-Ortiz e-Energy'24) not accessible | Novelty can only be confirmed conditionally | You download them through Sakarya University library access and put the PDFs in `literature/pdfs/`; I re-check novelty before G2 |
| P2 | Many "full-text" readings were model digests (WebFetch), not verbatim text | Numbers marked (n) in the gap analysis may be imprecise | Verify every number against the PDF before it enters the manuscript (A6 consistency check) |
| P3 | The sandbox network blocks arXiv, GitHub, DergiPark, Nature, MDPI, Elsevier, Springer for scripts | Style corpus incomplete; EV-GNN/GNN-DT code cannot be cloned from GitHub; PDF downloads fail | Option A: you add these domains to the allow-list (claude.ai › Settings › Capabilities); Option B: I keep using the in-app browser on your Mac (slower, works only while it is open) |
| P4 | Compute: no GPU. Cloud workspace 2 CPU / 7 GB; local VM 4 CPU / 3 GB | The full grid (6 sizes × 3 profiles × 2 capacities × ~10 methods × 5 seeds) is large | Decide at G2/G3; access to a GPU machine or TRUBA/university HPC would help |
| P5 | SAUCIS AI policy versus the orchestrator plan (A4 "findings", A5 Results/Discussion drafting) | Risk of desk rejection or later retraction if AI interprets results | See decision D3 |

## 4. Decisions needed from you
- **D1 — Contribution set.** Recommendation: proceed with C1-C3, framed as a combination. Add one small secondary contribution: a conformal lower bound on departure time feeding l_i, which addresses the known-departure assumption. Do not add the degradation Pareto study. Run the public EV-GNN or GNN-DT code as an extra baseline. The decision becomes final after P1. If R1 already proves the guarantee and tests size transfer, we switch to fallback (a) as the main contribution.
- **D2 — Paywalled papers.** Can you obtain the PDFs for P1 (at minimum orfanoudakis2026can)?
- **D3 — AI-policy workflow.**
  - Proposal: agents run experiments and generate tables and figures with scripts. For results, A4 produces only a factual, numbers-only summary without interpretation.
  - You write the Results interpretation and the Discussion yourself.
  - You thoroughly revise all AI-drafted text in the Introduction, Related Work and Abstract.
  - The AI statement follows Option A in `paper/journal_requirements.md`.
  - Do you accept this?
- **D4 — Network.** Do you choose allow-list (Option A) or in-app browser (Option B)?
- **D5 — Compute.** Is any GPU machine or HPC account available for Phase 3-4?

## 5. Files to review first
1. `literature/gap_analysis.md` (sections c-e)
2. `paper/journal_requirements.md` (sections 1-2)
3. `literature/paper_matrix.csv`
4. `literature/related_work_draft.md` (draft quality only; it will be rewritten later)

---
## G1 decision (human author, 2026-10-21)
- **D1:** Approved. Proceed with C1-C3 as a combination; add the small conformal departure-time extension; run public EV-GNN or GNN-DT code as an extra baseline.
- **D2:** Paywalled papers could not be obtained. Novelty statement stays conditional; risk R1-R3 remain open and are stated in the paper's positioning carefully.
- **D3:** The agents draft Introduction, Related Work, Method and Abstract in full; Results interpretation and Discussion are provided as drafts/skeletons (optional). The author revises as needed and remains responsible (SAUCIS AI policy).
- **D4:** Allow-list updated by the author (arxiv.org, github, dergipark confirmed reachable; www.nature.com and mdpi.com still blocked at check time).
- **D5:** Experiments will be run by the author on another computer. Scope change: this project provides the full code infrastructure (implementation, tests, configs, reproduce script) but does not execute the full experiment grid.
