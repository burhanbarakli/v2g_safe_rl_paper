# Style lint report: `manuscript.md`

Total words analysed: 13429

## Restricted phrases (whole manuscript)

| Phrase | Count | Rule | Status |
|---|---|---|---|
| (none) | | | OK |

## Acronyms

- Used but never defined: A1, A2, A3, A4, A5, SA
- Used before first definition: AI, V2G
- Defined more than twice (abstract + main text allowed): none

## Per-section checks

| Section | Words | Sentences | Mean len | >35 words | Paras >180 words | Demo-opener paras | 'This'+verb | Consecutive same opener | Status |
|---|---|---|---|---|---|---|---|---|---|
| Attention-Based Safe Reinforcement Learn | 30 | 3 | 10.0 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| Abstract | 339 | 15 | 22.6 | 0 | 0 | 0 | 0 | 0 | OK |
| 1 Introduction | 801 | 37 | 21.6 | 0 | 0 | 0 | 0 | 0 | OK |
| 2.1 Reinforcement learning for EV chargi | 320 | 18 | 17.8 | 0 | 0 | 0 | 0 | 0 | OK |
| 2.2 Constraint handling and safe reinfor | 589 | 27 | 21.8 | 0 | 0 | 0 | 0 | 2 | FLAG: repeat opener |
| 2.3 Size-agnostic architectures | 435 | 23 | 18.9 | 1 | 0 | 0 | 0 | 0 | FLAG: long sent |
| 2.4 Laxity-based scheduling and departur | 246 | 13 | 18.9 | 0 | 0 | 0 | 0 | 0 | OK |
| 2.5 Positioning | 187 | 10 | 18.7 | 0 | 0 | 0 | 0 | 0 | OK |
| 3.1 Station, time and prices | 98 | 5 | 19.6 | 0 | 0 | 0 | 0 | 0 | OK |
| 3.2 EV energy dynamics | 240 | 14 | 17.1 | 0 | 0 | 0 | 0 | 1 | FLAG: repeat opener |
| 3.3 Departure energy requirement | 120 | 6 | 20.0 | 0 | 0 | 0 | 0 | 0 | OK |
| 3.4 Transformer limit | 76 | 4 | 19.0 | 0 | 0 | 0 | 0 | 0 | OK |
| 3.5 Online control problem | 72 | 4 | 18.0 | 0 | 0 | 0 | 0 | 0 | OK |
| 3.6 Perfect-information optimum | 247 | 12 | 20.6 | 0 | 0 | 0 | 0 | 0 | OK |
| 3.7 Markov decision process | 286 | 15 | 19.1 | 0 | 0 | 0 | 0 | 0 | OK |
| 4 Proposed Method | 123 | 6 | 20.5 | 0 | 0 | 0 | 0 | 0 | OK |
| 4.1 Token representation and attention p | 618 | 29 | 21.3 | 0 | 0 | 0 | 0 | 0 | OK |
| 4.2 Per-EV bounds from reachable energy | 582 | 34 | 17.1 | 0 | 0 | 0 | 0 | 1 | FLAG: repeat opener |
| 4.3 Projection | 473 | 27 | 17.5 | 0 | 0 | 0 | 0 | 1 | FLAG: repeat opener |
| 4.4 Feasibility invariance | 591 | 40 | 14.8 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| 4.5 Fallback rule and admission control | 455 | 27 | 16.9 | 0 | 0 | 0 | 0 | 0 | OK |
| 4.6 Conformal lower bound on the departu | 336 | 16 | 21.0 | 0 | 0 | 0 | 0 | 1 | FLAG: repeat opener |
| 4.7 Training algorithm | 336 | 16 | 21.0 | 0 | 0 | 0 | 0 | 0 | OK |
| 5.1 Simulator and scenarios | 487 | 22 | 22.1 | 1 | 0 | 0 | 0 | 1 | FLAG: long sent, repeat opener |
| 5.2 Compared methods | 401 | 18 | 22.3 | 0 | 0 | 0 | 0 | 1 | FLAG: repeat opener |
| 5.3 Metrics | 155 | 7 | 22.1 | 0 | 0 | 0 | 0 | 0 | OK |
| 5.4 Hyperparameters, tuning and statisti | 178 | 9 | 19.8 | 0 | 0 | 0 | 0 | 1 | FLAG: repeat opener |
| 6 Results | 34 | 1 | 34.0 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| 6.1 Main comparison | 185 | 8 | 23.1 | 0 | 0 | 0 | 0 | 0 | OK |
| 6.2 Zero-shot scaling | 152 | 7 | 21.7 | 0 | 0 | 0 | 0 | 0 | OK |
| 6.3 Constraint satisfaction | 251 | 13 | 19.3 | 0 | 0 | 0 | 0 | 0 | OK |
| 6.4 Ablation study | 90 | 5 | 18.0 | 0 | 0 | 0 | 0 | 0 | OK |
| 6.5 Cost of the guarantee | 130 | 8 | 16.2 | 0 | 0 | 0 | 0 | 0 | OK |
| 6.6 Departure uncertainty | 94 | 5 | 18.8 | 0 | 0 | 0 | 0 | 0 | OK |
| 6.7 Computation time | 85 | 4 | 21.2 | 0 | 0 | 0 | 0 | 0 | OK |
| 7.1 Interpretation | 178 | 7 | 25.4 | 3 | 0 | 0 | 0 | 0 | FLAG: long sent |
| 7.2 Practical implications for charge po | 92 | 4 | 23.0 | 0 | 0 | 0 | 0 | 3 | FLAG: repeat opener |
| 7.3 Limitations | 430 | 24 | 17.9 | 0 | 0 | 0 | 0 | 0 | OK |
| 8 Conclusion | 270 | 12 | 22.5 | 0 | 0 | 0 | 0 | 0 | OK |
| A Proofs | 46 | 2 | 23.0 | 0 | 0 | 0 | 0 | 0 | OK |
| A.1 Reachable energy and thresholds (Lem | 160 | 8 | 20.0 | 0 | 0 | 0 | 0 | 0 | OK |
| A.2 The taper of EV2Gym (Lemma 3) and As | 513 | 28 | 18.3 | 0 | 0 | 0 | 0 | 2 | FLAG: repeat opener |
| A.3 Projection (Proposition 1) | 297 | 16 | 18.6 | 0 | 0 | 0 | 0 | 0 | OK |
| A.4 One-step invariance (Lemma 4) | 65 | 4 | 16.2 | 0 | 0 | 0 | 0 | 0 | OK |
| A.5 Feasibility invariance (Theorem 1 an | 336 | 21 | 16.0 | 1 | 0 | 0 | 0 | 1 | FLAG: long sent, repeat opener |
| A.6 Fallback and deficit bound (Lemmas 5 | 176 | 13 | 13.5 | 0 | 0 | 0 | 0 | 1 | FLAG: mean, repeat opener |
| A.7 Conformal departure guarantee (Propo | 135 | 13 | 10.4 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| A.8 Upper bound of the perfect-informati | 152 | 11 | 13.8 | 0 | 0 | 0 | 0 | 3 | FLAG: mean, repeat opener |
| Article Information Form | 319 | 18 | 17.7 | 0 | 0 | 0 | 0 | 0 | OK |

## Details

- [2.2 Constraint handling and sa] repeated opener: The CMDP framework replaces the fixed weight with a bound on expected cumulative cost [0], and Lagrangian schemes [0] an
- [2.2 Constraint handling and sa] repeated opener: A graph controller for active voltage control screens actions with an alternating-current power flow and states that the
- [2.3 Size-agnostic architecture] 36 words: In EV charging, size independence first came from aggregation: Sadeghianpourhamami et al. binned connected vehicles by time to departure and remaining charging ...
- [3.2 EV energy dynamics] repeated opener: The number of remaining actions at step X , including the current one, is
- [4.2 Per-EV bounds from reachab] repeated opener: The bound X keeps the energy above this threshold and permits V2G discharging while slack remains.
- [4.3 Projection] repeated opener: The form X with a common level found by one-dimensional search is also the rate rule of sLLF [0].
- [4.6 Conformal lower bound on t] repeated opener: The statement is neither a per-EV guarantee nor a guarantee conditional on X .
- [5.1 Simulator and scenarios] 36 words: All experiments use EV2Gym [0] at commit 55ef368 (code version 2.0.0) without changes to its source, and a wrapper computes the observations and applies the saf...
- [5.1 Simulator and scenarios] repeated opener: The original 25-charger configuration of EV2Gym (100 kW, workplace profile, no load or PV) adds 100 episodes for compara
- [5.2 Compared methods] repeated opener: [TODO: Confirm whether EV-GNN or GNN-DT runs on our commit as a further baseline].
- [5.4 Hyperparameters, tuning an] repeated opener: [TODO: Code and data repository link, anonymised for review].
- [7.1 Interpretation] 38 words: [TODO: Paragraph 1: state whether one set of weights trained at 10 and 20 chargers transfers to 200 chargers, citing Figure 5 and Table 4, and compare the gap a...
- [7.1 Interpretation] 36 words: [TODO: Paragraph 2: relate the measured cost of the guarantee in Figure 8 to the violations of the penalty-trained policy, and discuss whether the safety layer ...
- [7.1 Interpretation] 39 words: [TODO: Paragraph 3: discuss how often Assumption A5 failed and how much shortfall the fallback rule left per X , profile and rating (Figure 6), and relate the r...
- [7.2 Practical implications for] repeated opener: [TODO: Translate the measured cost of the guarantee into euros per EV and per year for a station of a given size].
- [7.2 Practical implications for] repeated opener: [TODO: Describe how the conformal variant could serve stations whose drivers do not declare departure times, and which c
- [7.2 Practical implications for] repeated opener: [TODO: Relate the inference time of Section 6.7 to the 15-min control step].
- [A.2 The taper of EV2Gym (Lemma] repeated opener: Suppose further that V2G is enabled and that all EVs leave before the episode ends, and let X be the positive root of X 
- [A.2 The taper of EV2Gym (Lemma] repeated opener: If the step crosses X , let X and X , so that the delivered gain minus X equals X .
- [A.5 Feasibility invariance (Th] 36 words: An EV belongs to X either because it arrives at X , in which case its part of the invariant is A4, or because it was in X with X , in which case the induction s...
- [A.5 Feasibility invariance (Th] repeated opener: An EV with X stays in X with X , so its part of X holds, and EVs that arrive at X satisfy their part by A4.
- [A.6 Fallback and deficit bound] repeated opener: If X , then X and X , and Equation 17 with the 1-Lipschitz property of X gives X .
- [A.8 Upper bound of the perfect] repeated opener: The power bounds hold because X and X .
- [A.8 Upper bound of the perfect] repeated opener: The departure constraint holds by the definition of X , and the transformer constraint is Equation 6.
- [A.8 Upper bound of the perfect] repeated opener: The experiments report the difference between the two optima once per scenario.


**Blocking flags: 29**
