# Style lint report: `/home/claude/work/paper_repo/manuscript.md`

Total words analysed: 12936

## Restricted phrases (whole manuscript)

| Phrase | Count | Rule | Status |
|---|---|---|---|
| (none) | | | OK |

## Acronyms

- Used but never defined: A1, A2, A3, A4, A5, AC, AI, CP, R1, R3, RESULT, S1, SA, SAUCIS
- Used before first definition: MLP, V2G
- Defined more than twice (abstract + main text allowed): none

## Per-section checks

| Section | Words | Sentences | Mean len | >35 words | Paras >180 words | Demo-opener paras | 'This'+verb | Consecutive same opener | Status |
|---|---|---|---|---|---|---|---|---|---|
| Attention-Based Safe Reinforcement Learn | 30 | 3 | 10.0 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| Abstract | 339 | 17 | 19.9 | 0 | 1 | 1 | 0 | 0 | FLAG: long para, demo openers |
| 1 Introduction | 758 | 48 | 15.8 | 0 | 0 | 1 | 0 | 0 | FLAG: mean, demo openers |
| 2.1 Reinforcement learning for EV chargi | 320 | 18 | 17.8 | 0 | 0 | 0 | 0 | 0 | OK |
| 2.2 Constraint handling and safe reinfor | 575 | 36 | 16.0 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| 2.3 Size-agnostic architectures | 419 | 26 | 16.1 | 0 | 0 | 0 | 0 | 0 | OK |
| 2.4 Laxity-based scheduling and departur | 242 | 17 | 14.2 | 0 | 0 | 0 | 0 | 1 | FLAG: mean, repeat opener |
| 2.5 Positioning | 181 | 12 | 15.1 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| 3.1 Station, time and prices | 97 | 6 | 16.2 | 0 | 0 | 0 | 0 | 0 | OK |
| 3.2 EV energy dynamics | 236 | 19 | 12.4 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| 3.3 Departure energy requirement | 114 | 9 | 12.7 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| 3.4 Transformer limit | 70 | 6 | 11.7 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| 3.5 Online control problem | 72 | 5 | 14.4 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| 3.6 Perfect-information optimum | 237 | 17 | 13.9 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| 3.7 Markov decision process | 274 | 19 | 14.4 | 0 | 0 | 0 | 0 | 1 | FLAG: mean, repeat opener |
| 4 Proposed Method | 113 | 8 | 14.1 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| 4.1 Token representation and attention p | 598 | 38 | 15.7 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| 4.2 Per-EV bounds from reachable energy | 558 | 42 | 13.3 | 1 | 0 | 0 | 0 | 1 | FLAG: mean, long sent, repeat opener |
| 4.3 Projection | 475 | 33 | 14.4 | 0 | 0 | 0 | 0 | 2 | FLAG: mean, repeat opener |
| 4.4 Feasibility invariance | 576 | 52 | 11.1 | 0 | 0 | 0 | 0 | 3 | FLAG: mean, repeat opener |
| 4.5 Fallback rule and admission control | 439 | 34 | 12.9 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| 4.6 Conformal lower bound on the departu | 338 | 25 | 13.5 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| 4.7 Training algorithm | 320 | 24 | 13.3 | 0 | 0 | 0 | 0 | 1 | FLAG: mean, repeat opener |
| 5.1 Simulator and scenarios | 474 | 33 | 14.4 | 0 | 0 | 0 | 0 | 2 | FLAG: mean, repeat opener |
| 5.2 Compared methods | 375 | 23 | 16.3 | 1 | 0 | 0 | 0 | 2 | FLAG: long sent, repeat opener |
| 5.3 Metrics | 152 | 9 | 16.9 | 0 | 0 | 0 | 0 | 1 | FLAG: repeat opener |
| 5.4 Hyperparameters, tuning and statisti | 182 | 13 | 14.0 | 0 | 0 | 0 | 0 | 1 | FLAG: mean, repeat opener |
| 6 Results | 24 | 2 | 12.0 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| 6.1 Main comparison | 140 | 9 | 15.6 | 0 | 0 | 0 | 0 | 3 | FLAG: mean, repeat opener |
| 6.2 Zero-shot scaling | 137 | 8 | 17.1 | 0 | 0 | 0 | 0 | 2 | FLAG: repeat opener |
| 6.3 Constraint satisfaction | 232 | 15 | 15.5 | 0 | 0 | 0 | 0 | 5 | FLAG: mean, repeat opener |
| 6.4 Ablation study | 64 | 6 | 10.7 | 0 | 0 | 0 | 0 | 3 | FLAG: mean, repeat opener |
| 6.5 Cost of the guarantee | 119 | 8 | 14.9 | 0 | 0 | 0 | 0 | 2 | FLAG: mean, repeat opener |
| 6.6 Departure uncertainty | 69 | 5 | 13.8 | 0 | 0 | 0 | 0 | 2 | FLAG: mean, repeat opener |
| 6.7 Computation time | 77 | 5 | 15.4 | 0 | 0 | 0 | 0 | 2 | FLAG: mean, repeat opener |
| 7.1 Interpretation | 160 | 8 | 20.0 | 0 | 0 | 0 | 0 | 0 | OK |
| 7.2 Practical implications for charge po | 87 | 1 | 87.0 | 1 | 0 | 0 | 0 | 0 | FLAG: mean, long sent |
| 7.3 Limitations | 426 | 29 | 14.7 | 0 | 0 | 0 | 0 | 5 | FLAG: mean, repeat opener |
| 8 Conclusion | 257 | 16 | 16.1 | 0 | 0 | 0 | 0 | 3 | FLAG: repeat opener |
| A Proofs | 45 | 3 | 15.0 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| A.1 Reachable energy and thresholds (Lem | 158 | 12 | 13.2 | 0 | 0 | 0 | 0 | 2 | FLAG: mean, repeat opener |
| A.2 The taper of EV2Gym (Lemma 3) and As | 502 | 38 | 13.2 | 0 | 1 | 0 | 0 | 3 | FLAG: mean, long para, repeat opener |
| A.3 Projection (Proposition 1) | 273 | 26 | 10.5 | 0 | 0 | 0 | 0 | 2 | FLAG: mean, repeat opener |
| A.4 One-step invariance (Lemma 4) | 59 | 9 | 6.6 | 0 | 0 | 0 | 0 | 1 | FLAG: mean, repeat opener |
| A.5 Feasibility invariance (Theorem 1 an | 323 | 28 | 11.5 | 0 | 0 | 0 | 0 | 1 | FLAG: mean, repeat opener |
| A.6 Fallback and deficit bound (Lemmas 5 | 155 | 18 | 8.6 | 0 | 0 | 0 | 0 | 3 | FLAG: mean, repeat opener |
| A.7 Conformal departure guarantee (Propo | 135 | 13 | 10.4 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| A.8 Upper bound of the perfect-informati | 152 | 11 | 13.8 | 0 | 0 | 0 | 0 | 3 | FLAG: mean, repeat opener |
| Article Information Form | 350 | 14 | 25.0 | 3 | 0 | 0 | 0 | 0 | FLAG: long sent |

## Details

- [Abstract] >1 This/These/It/There opener in paragraph starting: Charge point operators that offer vehicle-to-grid (V2G) serv...
- [1 Introduction] >1 This/These/It/There opener in paragraph starting: This paper proposes a safe attention policy (SA-Attn) for V2...
- [2.4 Laxity-based scheduling an] repeated opener: Conformal bounds have also tightened constraints in safe planning [0].
- [3.7 Markov decision process] repeated opener: The executed power is X , the output of the safety layer of Section 4.3, or the fallback rule of Section 4.5 when the fe
- [4.2 Per-EV bounds from reachab] 37 words: The threshold X is the least energy from which full-power charging still reaches X in the remaining steps; the lower bound X keeps the energy at or above this t...
- [4.2 Per-EV bounds from reachab] repeated opener: The stored table is certified, in the sense that
- [4.3 Projection] repeated opener: The problem in Equation 19 is the continuous quadratic knapsack problem, solved by breakpoint algorithms [0] and by a se
- [4.3 Projection] repeated opener: The form X with a common level found by one-dimensional search is also the rate rule of sLLF [0].
- [4.4 Feasibility invariance] repeated opener: A certified table does not compensate for a violation of A1.
- [4.4 Feasibility invariance] repeated opener: The transformer limit holds at every step except in cases (i), (ii) and (iv) of the fallback rule in Section 4.5.
- [4.4 Feasibility invariance] repeated opener: At X both lower bounds equal 10 kW, and X kW exceeds X .
- [4.7 Training algorithm] repeated opener: The safety layer, the fallback rule and the actuation map are treated as part of the environment.
- [5.1 Simulator and scenarios] repeated opener: The desired energy is the full battery, and a sensitivity run uses 80% state of charge.
- [5.1 Simulator and scenarios] repeated opener: The tight level equals the ratio of the 25-charger V2G profit-maximisation configuration of EV2Gym.
- [5.2 Compared methods] 36 words: [TODO: Confirm the MPC solver used for X (restricted Gurobi licence; our HiGHS port) and whether EV-GNN or GNN-DT runs on our commit as a further baseline.] The...
- [5.2 Compared methods] repeated opener: MLP+penalty combines both changes.
- [5.2 Compared methods] repeated opener: As late as possible (ALAP) charges at the lower bound of Equation 15 with discharging disabled.
- [5.3 Metrics] repeated opener: The cost of the guarantee is the profit of Attention+penalty minus the profit of SA-Attn on the same episodes.
- [5.4 Hyperparameters, tuning an] repeated opener: The safety layer runs in double precision.
- [6.1 Main comparison] repeated opener: [Result: comparison with Attention+penalty in profit and in violations, with the Wilcoxon test result].
- [6.1 Main comparison] repeated opener: [Result: comparison with the heuristics with and without the safety layer, and with the MPC].
- [6.1 Main comparison] repeated opener: [Result: methods whose profit exceeds X through transformer violations, if any].
- [6.2 Zero-shot scaling] repeated opener: [Result: change of the gap from the training sizes to X , and comparison with MLP+safety layer].
- [6.2 Zero-shot scaling] repeated opener: [Result: differences between the three arrival profiles and the two transformer ratings].
- [6.3 Constraint satisfaction] repeated opener: [Result: shortfall and violation energy of the penalty-based methods per X ].
- [6.3 Constraint satisfaction] repeated opener: [Result: fraction of infeasible steps per X , profile and rating, split into fallback cases].
- [6.3 Constraint satisfaction] repeated opener: [Result: measured shortfall against the deficit bound of Equation 20].
- [6.3 Constraint satisfaction] repeated opener: [Result: number of requests capped by more than 0.01 kWh and reduced by admission control].
- [6.3 Constraint satisfaction] repeated opener: [Result: whether and when the aggregate lower bound approaches the forward limit in this episode].
- [6.4 Ablation study] repeated opener: [Result: effect of X on the projection distance and on profit].
- [6.4 Ablation study] repeated opener: [Result: effect of the curriculum and of the small network].
- [6.4 Ablation study] repeated opener: [Result: effect of the dense penalty on Attention+penalty].
- [6.5 Cost of the guarantee] repeated opener: [Result: violations that the penalty-trained policy incurred for that profit].
- [6.5 Cost of the guarantee] repeated opener: [Result: sensitivity of the cost to the penalty weights 1 and 100 €/kWh].
- [6.6 Departure uncertainty] repeated opener: [Result: fraction of EVs that meet X , shortfall of uncovered EVs and mean slack X ].
- [6.6 Departure uncertainty] repeated opener: [Result: profit loss relative to SA-Attn with known departure, and comparison with the uncalibrated point prediction].
- [6.7 Computation time] repeated opener: [Result: share of the safety layer in that time, and comparison with the solve time of the MPC and of the LP].
- [6.7 Computation time] repeated opener: [Result: training wall-clock time per run].
- [7.2 Practical implications for] 87 words: [TODO: What a CPO gains from a controller whose transformer and departure guarantees hold for any raw action, e.g. for certification or for operation next to a ...
- [7.3 Limitations] repeated opener: The conformal extension relaxes only the departure step, and it gives a marginal coverage statement rather than a per-EV
- [7.3 Limitations] repeated opener: The requested energy is still assumed to be declared at arrival.
- [7.3 Limitations] repeated opener: The deficit bound of Lemma 6 limits the remaining shortfall but does not remove it.
- [7.3 Limitations] repeated opener: The reported optimality gap therefore overestimates the true gap.
- [7.3 Limitations] repeated opener: The LP also ignores the information limits of any online controller, so part of the gap is the price of not knowing the 
- [8 Conclusion] repeated opener: A split-conformal bound on the departure step relaxes the known-departure assumption at a stated coverage level.
- [8 Conclusion] repeated opener: [Result: one sentence on constraint satisfaction and on the frequency of infeasible steps].
- [8 Conclusion] repeated opener: [Result: one sentence on the cost of the guarantee].
- [A.1 Reachable energy and thres] repeated opener: For X , an admissible power X gives X , because X is increasing.
- [A.1 Reachable energy and thres] repeated opener: The set in Equation 14 is non-empty because X .
- [A.2 The taper of EV2Gym (Lemma] repeated opener: At X the last two branches both equal X .
- [A.2 The taper of EV2Gym (Lemma] repeated opener: The branch slopes are X , X and X , and X has slope 1.
- [A.2 The taper of EV2Gym (Lemma] repeated opener: If the step crosses X , let X and X .
- [A.3 Projection (Proposition 1)] repeated opener: For X , set X and X .
- [A.3 Projection (Proposition 1)] repeated opener: For X , set X and X .
- [A.4 One-step invariance (Lemma] repeated opener: If X , then X .
- [A.5 Feasibility invariance (Th] repeated opener: If X , then X with X , and its part of X holds.
- [A.6 Fallback and deficit bound] repeated opener: Define the feasibility deficit X with X .
- [A.6 Fallback and deficit bound] repeated opener: If X , then X and X .
- [A.6 Fallback and deficit bound] repeated opener: In case (iv), X . ∎
- [A.8 Upper bound of the perfect] repeated opener: The power bounds hold because X and X .
- [A.8 Upper bound of the perfect] repeated opener: The departure constraint holds by the definition of X , and the transformer constraint is Equation 6.
- [A.8 Upper bound of the perfect] repeated opener: The experiments report the difference between the two optima once per scenario.
- [Article Information Form] 41 words: \ifanonymous Removed for double-blind review. \else [TODO: Acknowledge contributors who do not meet the authorship criteria, or write "Not applicable".] \fi The...
- [Article Information Form] 52 words: \ifanonymous The code, the scenario configurations and the analysis scripts will be made public on acceptance; the link is withheld for double-blind review. \el...
- [Article Information Form] 63 words: The authors used Claude (Anthropic), model claude-opus-5-5, during the preparation of this work to (i) support literature searching and the screening of candida...


**Blocking flags: 87**
