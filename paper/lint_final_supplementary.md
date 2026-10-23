# Style lint report: `supplementary.md`

Total words analysed: 8954

## Restricted phrases (whole manuscript)

| Phrase | Count | Rule | Status |
|---|---|---|---|
| (none) | | | OK |

## Acronyms

- Used but never defined: A1, A2, A3, A4, A5, A6, AMD, CI, CUDA, DT4EV, E4, FB-0, FB-1, FB-2, GB, NVIDIA, RTX, S1, S10, S11, S12, S13, S14, S15, S16, S2, S3, S4, S5, S6, S7, S8, S9, SA, T2, T3, T4, T5, T6, T7, TS1, TS2, TS3, WSL2
- Used before first definition: V2G
- Defined more than twice (abstract + main text allowed): none

## Per-section checks

| Section | Words | Sentences | Mean len | >35 words | Paras >180 words | Demo-opener paras | 'This'+verb | Consecutive same opener | Status |
|---|---|---|---|---|---|---|---|---|---|
| Supplementary Material: Size-Agnostic Sa | 203 | 11 | 18.5 | 0 | 0 | 0 | 0 | 0 | OK |
| S1 Proofs | 66 | 3 | 22.0 | 0 | 0 | 0 | 0 | 0 | OK |
| S1.1 Size independence of the attention  | 311 | 15 | 20.7 | 0 | 0 | 0 | 0 | 0 | OK |
| S1.2 Reachable energy and thresholds | 208 | 12 | 17.3 | 0 | 0 | 0 | 0 | 0 | OK |
| S1.3 Projection | 411 | 22 | 18.7 | 0 | 0 | 0 | 0 | 0 | OK |
| S1.4 One-step invariance | 150 | 9 | 16.7 | 0 | 0 | 0 | 0 | 0 | OK |
| S1.5 Feasibility preservation | 485 | 30 | 16.2 | 0 | 0 | 0 | 0 | 0 | OK |
| S1.6 Fallback rule and deficit bound | 493 | 29 | 17.0 | 0 | 0 | 0 | 0 | 0 | OK |
| S1.7 Conformal departure guarantee | 159 | 9 | 17.7 | 0 | 0 | 0 | 0 | 0 | OK |
| S2 EV2Gym modelling notes | 46 | 2 | 23.0 | 0 | 0 | 0 | 0 | 0 | OK |
| S2.1 Charging taper and Assumption A1 | 313 | 18 | 17.4 | 0 | 0 | 0 | 0 | 0 | OK |
| S2.2 Actuation and Assumption A3 | 736 | 42 | 17.5 | 0 | 0 | 0 | 0 | 0 | OK |
| S2.3 Exogenous headroom (Assumption A6) | 117 | 6 | 19.5 | 0 | 0 | 0 | 0 | 0 | OK |
| S2.4 Scenarios, arrival profiles and obs | 306 | 16 | 19.1 | 0 | 0 | 0 | 0 | 0 | OK |
| S3.1 Station-level request pipeline | 518 | 23 | 22.5 | 0 | 0 | 0 | 0 | 0 | OK |
| S3.2 Linear programs of the optima | 319 | 19 | 16.8 | 0 | 0 | 0 | 0 | 0 | OK |
| S3.3 Upper bounds | 435 | 25 | 17.4 | 0 | 0 | 0 | 0 | 0 | OK |
| S4.1 Implementation of the compared meth | 658 | 31 | 21.2 | 0 | 0 | 0 | 0 | 0 | OK |
| S4.2 Training loop | 112 | 5 | 22.4 | 0 | 0 | 0 | 0 | 0 | OK |
| S4.3 Hyperparameters and tuning | 192 | 9 | 21.3 | 0 | 0 | 0 | 0 | 0 | OK |
| S4.4 Statistical analysis | 279 | 12 | 23.2 | 0 | 0 | 0 | 0 | 0 | OK |
| S4.5 Compute | 148 | 8 | 18.5 | 0 | 0 | 0 | 0 | 0 | OK |
| S5 Additional results | 96 | 5 | 19.2 | 0 | 0 | 0 | 0 | 0 | OK |
| S5.1 Learning curves and zero-shot trans | 220 | 10 | 22.0 | 1 | 0 | 0 | 0 | 0 | FLAG: long sent |
| S5.2 Constraint satisfaction | 547 | 26 | 21.0 | 1 | 1 | 0 | 0 | 0 | FLAG: long sent, long para |
| S5.3 Cost of the guarantee and constrain | 368 | 15 | 24.5 | 3 | 0 | 0 | 0 | 0 | FLAG: long sent |
| S5.4 Conformal departure bound | 235 | 11 | 21.4 | 0 | 0 | 0 | 0 | 0 | OK |
| S5.5 Computation time | 126 | 8 | 15.8 | 0 | 0 | 0 | 0 | 0 | FLAG: mean |
| S5.6 Relative gaps, stress level and lit | 352 | 17 | 20.7 | 1 | 0 | 0 | 0 | 0 | FLAG: long sent |
| S5.7 Pooled constraint metrics | 74 | 4 | 18.5 | 0 | 0 | 0 | 0 | 0 | OK |

## Details

- [S5.1 Learning curves and zero-] 45 words: Zero-shot transfer across station sizes N for policies trained at N ∈ {10, 20} (profiles and ρtr ∈ {4, 7} kW pooled; per episode): learned methods, SA-Attn trai...
- [S5.2 Constraint satisfaction] 41 words: E4 (share of EVs whose stay contains no infeasible step), infeasible steps, fallback activations per case and binding rates of the aggregate limits and of the d...
- [S5.3 Cost of the guarantee and] 53 words: Cost of the guarantee per station size N (profiles and tightness levels pooled): penalty-based policy minus the same architecture with the safety layer, in J^pe...
- [S5.3 Cost of the guarantee and] 43 words: Ranges for all columns, bootstrap confidence intervals and per-seed values are provided with the result data. Δ = Σ (penalty − safety layer) / Σ denominator ove...
- [S5.3 Cost of the guarantee and] 52 words: Architecture and constraint handling at the training sizes N ∈ {10, 20} and in zero-shot transfer (N ≥ 50): attention, Deep Sets and zero-padded MLP policies wi...
- [S5.6 Relative gaps, stress lev] 47 words: Relative optimality gaps as ratios of sums, per station size N (profiles and tightness levels pooled): 100 · Σ(J − J) / Σ|J| over all test episodes of the seed-...


**Blocking flags: 50**
