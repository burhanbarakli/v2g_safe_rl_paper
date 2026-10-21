# DONE_L0: core competitors

## Produced (all in /home/claude/work/lit/parts/)
- `L0_verified.jsonl` and `L0_verified.bib`: produced by the script. There are 24 jsonl records for 23 unique keys. EV2Gym was verified twice: first as `orfanoudakis2024ev2gym`, then as `orfanoudakis2025ev2gym`, because Crossref gives 2025 (vol. 26, no. 2). **Use `orfanoudakis2025ev2gym`.** The superseded 2024 BibTeX entry was removed from the .bib; the jsonl was not edited.
- `L0_matrix.csv`: 23 rows.
- `L0_candidates.csv`: 160 unique screened records (23 include, 137 exclude).
- `L0_notes.md`: per-paper notes plus the theme synthesis.
- `L0_downloads.tsv`: one download, the EV2Gym 2.0.0 source tarball from PyPI, stored at /home/claude/work/pdfs/L0_ev2gym-2.0.0.tar.gz. Publisher and arXiv PDFs could not be fetched with curl because the proxy returned 403, so full texts were read through WebFetch.

## Counts
| Measure | Count |
|---|---|
| Screened (unique records) | 160 |
| Included | 23 |
| Included, full text | 10 |
| Included, abstract only | 13 |
| Verified with the script | 23 / 23 |
| Included from 2022 or later | 23 of 23 |

Where the 160 screened records came from:
- 9 core papers and 2 preprint duplicates.
- About 36 EV2Gym-citing works. OpenAlex lists 24 for W4405429570 and Semantic Scholar 33; the union was deduplicated.
- 30 EV-GNN citers and 2 PI-TD3 citers (Semantic Scholar).
- About 92 records from two OpenAlex keyword queries, run to test novelty.

Included papers by source:
- 9 core papers.
- 8 additional EV2Gym citers, meeting the 6-12 target: rossi2025smart, yilmaz2024reinforcement, guo2025hierarchical, demirel2025multiobjective, manzoor2026physics, rahali2025comparison, diazlondono2026open, jing2026hybrid. EV-GNN, GNN-DT and PI-TD3 also cite EV2Gym; they are counted as core.
- 6 extras found while testing novelty: yan2026hierarchical, wan2025logicinformed, zhang2025modelfree, song2025longterm, wang2026twostage, baweja2026nexusmarl.

## Metadata corrections relative to the task list
- **EV2Gym.** Five authors: Orfanoudakis, Diaz-Londono, Yılmaz, Palensky, Vergara. Published in IEEE T-ITS 26(2):2410-2421, 2025; online-first in December 2024, and OpenAlex says 2024.
- **EV-GNN.** Five authors: Pedro P. Vergara is also an author. Communications Engineering vol. 4, 2025.
- **GNN-DT.** Metadata correct (Energy and AI vol. 23, 100679, 2026). arXiv preprint 2502.01778 has a different title: "GNN-DT: Graph Neural Network Enhanced Decision Transformer for Efficient Optimization in Dynamic Environments".
- **PI-TD3.** Metadata correct (Applied Energy vol. 421, 128224, 2026). arXiv preprint 2510.12335.
- **HetGPS.** Five authors, with Saman Kumara Halgamuge confirmed. arXiv 2608.00679, 2026.
- **DuMES.** Six authors: Ward, T.E. and Cheng, L. are also authors. IET Smart Energy Systems 1(3):232-246, 2025.
- **Wu et al.** Five authors: Sean Peisert is also an author. SmartGridComm 2023, pp. 1-6.
- **Laxity paper.** "A Multi-Agent, Laxity-Based Aggregation Strategy for Cost-Effective Electric Vehicle Charging and Local Transformer Overload Prevention" by Christensen, Jørgensen and Ma, Sustainability 17(9):3847, 2025. **It is rule-based, not RL.**
- **SIGMETRICS paper.** "Scheduling EV charging with uncertain departure times" by Ferragut, Narbondo and Paganini, ACM SIGMETRICS PER 49(3):10-15, 2022. **It is a queueing/mean-field analysis, not RL.**

## EV2Gym facts for the methods section (from the paper and the v2.0.0 code)
- **Step length.** Configurable; 15 minutes in every shipped config and in both case studies.
- **Action.** One value per EVSE port, in [0,1] for G2V or [-1,1] for V2G. It maps to current as amps = a × I_max.
- **Transformer.** The limit has a time-varying max/min, inflexible load, PV and demand-response capacity cuts. **Overload is not prevented by the simulator, only measured.**
- **Baselines.**
  - Heuristics: AFAP, ALAP, Round Robin (and variants), CAFAP with power limit, DoNothing/Random.
  - MPC: eMPC and OCMF, each in V2G and G2V versions.
  - Oracle: Gurobi optimum with perfect information.
  - RL: Stable-Baselines3 algorithms.
- **Licence.** MIT, according to the LICENSE file in the PyPI sdist and the PyPI classifier. The paper itself states no licence.
- **Caveat.** In the code, the Gurobi profit-max oracle makes the transformer limit hard, but departure energy only enters as a quadratic penalty; the hard departure constraint is commented out. For an optimality gap against a "hard-constrained optimum", we must add the departure constraint ourselves.

## Novelty threat assessment
**Hypothesis tested.** Does any paper already combine (i) a single size-agnostic attention or graph policy that transfers zero-shot across station sizes with (ii) a per-step guarantee, provable by construction, of both EV departure energy and the transformer limit?

**Verdict: not found in the screened set.** No paper I read combines (i) and (ii). However, each half is well populated, and several papers come close on one axis.

Closest on (i), a size-agnostic policy with zero-shot transfer:
- **wang2026hetgps (HetGPS).**
  - Its policy is a heterogeneous-graph and attention SAC with the same 383,702 parameters at every scale, and it transfers zero-shot from 8 to 16 and 32 transformers (1,236 to 3,218 EVs).
  - Its safety mechanism is a physics-anchored voltage filter, but the authors explicitly state that they make no pointwise feasibility claim.
  - Departure energy is handled by reward terms plus a heuristic service map, with 99.06-100% success (not guaranteed).
  - The transformer overload coefficient is zero, and thermal limits are not constraints.
  - **This is the strongest threat on scalability and must be cited and contrasted:** network voltage versus station transformer, and empirical versus provable.
- **orfanoudakis2026graph (GNN-DT).** Uses GCN with a Decision Transformer and transfers zero-shot from 25 to 100 chargers. Constraints are penalty-only, and its power violations are higher than round robin (21.7 versus 10.5). The policy is learned offline.
- **orfanoudakis2025scalable (EV-GNN).** Its GCN architecture can handle variable N, but in the version read a separate model was trained per scale, and there is no constraint handling.
- **song2025longterm (MAHEM) and baweja2026nexusmarl.** Both use attention or transformer MARL, the latter with zero-shot transfer (in the cloud domain). Neither claims constraint guarantees (abstracts only).

Closest on (ii), per-step feasibility:
- **yan2026hierarchical (HPC-RL).**
  - Computes per-EV dynamic power bounds that force enough charging to meet departure SoC, using known departure time and target, together with GRG-based hard grid constraints. It reports 100% demand satisfaction.
  - It gives no proof for the departure bound, only a first-order proposition for power balance.
  - It has no transformer or station capacity constraint, does not handle aggregate infeasibility, has a fixed-size and unspecified architecture, and runs no zero-shot test.
  - **Its lower bound is conceptually the same as our l_i, so our paper must cite it and claim novelty only for the joint closed-form projection, its proof under aggregate feasibility, and its differentiable use in training.**
- **wan2025logicinformed (LIRL).**
  - Proves feasibility by construction (Lemma 1, a per-step quadratic-program projection) and reports zero transformer-capacity and current violations in an EV-station case.
  - Departure energy is not a constraint, the projection is solver-based rather than closed-form, and the policy is fixed-size.
- **zhang2025dumes (DuMES, abstract only).** A safety layer replaces unsafe actions, with claimed compliance, in a single fixed station. I could not verify a proof or the constraint set.
- **wu2023network.** Uses a Lagrangian plus an execution projection and reports 100% empirical feasibility, at the aggregate bus level.
- **zhang2025modelfree (PCPO).** Constrains expected cost; no per-step guarantee.
- **Non-RL schedulers.** jing2026hybrid and christensen2025multiagent enforce transformer capacity by construction or by rules, but do not guarantee departure energy (99.5% completion).

**Residual risk.** I could not read two items:
- **Eland, Orfanoudakis and Vergara (2026), "Safe Reinforcement Learning for V2G-Enabled Electric Vehicle Aggregators"** (CCIS, ECML-PKDD 2025 workshop, DOI 10.1007/978-3-032-19102-1_5). It comes from the EV2Gym group, is titled safe RL, and has no abstract available anywhere I could reach.
- **Orfanoudakis et al. (2026), "Can A.I. Revolutionize EV Dispatch?"** (DOI 10.1007/978-3-032-19102-1_1).

The first is the single most likely unseen threat; it may combine EV2Gym, safe RL and possibly the group's GNN policies. **Before submission, someone with Springer access should check it.**

The screen was also limited to the citation neighbourhood of EV2Gym, EV-GNN and PI-TD3, plus two keyword queries. Broader safe-RL and projection-layer literature (for example OptLayer or Dalal-style safety layers, action projection for EV charging, and ACN-Sim work) is assumed to be covered by other parts.

**Implication for positioning.** Claim the combination and be precise about it:
- closed-form, differentiable projection;
- proofs for both constraints under stated assumptions;
- the same layer used in training and deployment;
- size-agnostic attention;
- zero-shot transfer to about 200 chargers;
- an optimality gap measured against a hard-constrained oracle.

Do not claim that size-agnostic EV-charging policies or departure-feasibility bounds are new in themselves.

## Open issues and papers I could not access
- **Not readable.** The two CCIS chapters above (Eland et al.; Orfanoudakis et al.) were blocked at Springer and had no abstract in Crossref, OpenAlex or Semantic Scholar. They are kept as excluded with that reason.
- **DuMES** full text was blocked (Wiley), so it is abstract only.
- **EV-GNN** was read from the Research Square preprint v1. The published version may differ, for example by adding a code link or zero-shot experiments. nature.com, PMC and Europe PMC were blocked or rate-limited.
- **GNN-DT** was read from the arXiv preprint. Journal-specific claims ("within 5% of oracle", "unseen network topologies") come from the journal abstract.
- **Christensen et al.**: the results section was not parsed; outcome claims come from the abstract.
- **SMPD (wang2026twostage)**: the Stage II constraint mechanism was not visible in what I read.
- **Several EV2Gym citers** were read as abstract only because IEEE, ACM, Springer and KIT pages were not renderable or were blocked.
- **Service limits.** OpenAlex's daily budget was exhausted mid-session and Semantic Scholar rate-limited heavily, so the keyword searches for novelty were limited to two OpenAlex queries.
- **Rate-limited WebFetch pages**, which may hold full texts:
  - www.ebi.ac.uk (Europe PMC full-text XML for PMC12216737)
  - arxiv.org/a/orfanoudakis_s_1 (author listing)
  - doi.org/10.1049/ses2.70017

## Prompt-injection observations
- None found in the pages, PDFs or source code read. The `[agent-proxy]` and WebFetch notices were tool or infrastructure messages, not document content.
