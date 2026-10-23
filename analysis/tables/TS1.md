**TS1.** Raw-profit optimality gap with respect to the hard perfect-information optimum J^LP,ceil and to J^LP (without EV2Gym's ceil gain), per station size N (profiles and tightness levels pooled).

|  | Method | Gap vs J^LP,ceil (%) (hard optimum) | Gap vs J^LP (%) (no ceil gain) |
|---|---|---:|---:|
| N = 5 | SA-Attn (ours) | 89.79 ± 0.65 | 35.96 ± 0.70 |
| N = 5 | Attention+penalty | 82.75 ± 7.24 | 32.73 ± 4.26 |
| N = 5 | Attention+PPO-Lagrangian | 116.87 ± 1.54 | 64.23 ± 1.26 |
| N = 5 | Attention+safety layer without departure bound | 77.08 ± 5.78 | 25.19 ± 4.80 |
| N = 5 | Deep Sets+safety layer | 70.98 ± 3.72 | 21.40 ± 3.67 |
| N = 5 | MLP+safety layer | 87.41 ± 0.37 | 33.92 ± 0.82 |
| N = 5 | MLP+penalty | 118.38 ± 0.21 | 65.55 ± 0.26 |
| N = 5 | LLF+safety layer | 118.12† | 65.39† |
| N = 5 | sLLF+safety layer | 118.12† | 65.39† |
| N = 5 | Price-threshold+safety layer | 83.59† | 33.74† |
| N = 5 | AFAP+safety layer | 118.07† | 65.34† |
| N = 5 | ALAP+safety layer | 108.50† | 55.11† |
| N = 5 | Round Robin+safety layer | 118.07† | 65.34† |
| N = 5 | eMPC-HiGHS+safety layer | 78.77† | 26.26† |
| N = 5 | eMPC-HiGHS | -118.30† | -124.12† |
| N = 10 | SA-Attn (ours) | 121.51 ± 0.47 | 45.45 ± 0.66 |
| N = 10 | Attention+penalty | 119.99 ± 5.87 | 46.60 ± 2.74 |
| N = 10 | Attention+PPO-Lagrangian | 168.79 ± 3.20 | 82.18 ± 2.26 |
| N = 10 | Attention+safety layer without departure bound | 101.68 ± 6.47 | 31.35 ± 4.31 |
| N = 10 | Deep Sets+safety layer | 96.14 ± 5.01 | 26.68 ± 3.29 |
| N = 10 | MLP+safety layer | 121.31 ± 2.65 | 45.07 ± 1.72 |
| N = 10 | MLP+penalty | 166.34 ± 4.33 | 78.80 ± 4.51 |
| N = 10 | LLF+safety layer | 171.76† | 84.39† |
| N = 10 | sLLF+safety layer | 171.76† | 84.39† |
| N = 10 | Price-threshold+safety layer | 124.83† | 48.56† |
| N = 10 | AFAP+safety layer | 171.72† | 84.37† |
| N = 10 | ALAP+safety layer | 151.97† | 69.15† |
| N = 10 | Round Robin+safety layer | 171.74† | 84.37† |
| N = 10 | eMPC-HiGHS+safety layer | 107.12† | 34.07† |
| N = 10 | eMPC-HiGHS | -130.93† | -127.38† |
| N = 20 | SA-Attn (ours) | 118.97 ± 1.23 | 48.38 ± 1.59 |
| N = 20 | Attention+penalty | 118.64 ± 1.58 | 50.99 ± 0.08 |
| N = 20 | Attention+PPO-Lagrangian | 154.27 ± 3.90 | 79.03 ± 2.82 |
| N = 20 | Attention+safety layer without departure bound | 98.94 ± 7.57 | 32.84 ± 6.39 |
| N = 20 | Deep Sets+safety layer | 89.37 ± 7.78 | 25.85 ± 3.59 |
| N = 20 | MLP+safety layer | 125.10 ± 4.63 | 55.07 ± 3.51 |
| N = 20 | MLP+penalty | 155.98 ± 0.96 | 80.45 ± 1.17 |
| N = 20 | LLF+safety layer | 157.85† | 81.86† |
| N = 20 | sLLF+safety layer | 157.85† | 81.86† |
| N = 20 | Price-threshold+safety layer | 111.84† | 50.85† |
| N = 20 | AFAP+safety layer | 157.84† | 81.85† |
| N = 20 | ALAP+safety layer | 143.38† | 70.10† |
| N = 20 | Round Robin+safety layer | 157.84† | 81.85† |
| N = 20 | eMPC-HiGHS+safety layer | 102.78† | 35.97† |
| N = 20 | eMPC-HiGHS | -108.07† | -99.96† |
| N = 50 | SA-Attn (ours) | 129.20 ± 5.27 | 51.35 ± 5.68 |
| N = 50 | Attention+penalty | 139.35 ± 6.51 | 60.31 ± 1.10 |
| N = 50 | Attention+PPO-Lagrangian | 180.65 ± 5.26 | 91.93 ± 4.44 |
| N = 50 | Attention+safety layer without departure bound | 112.68 ± 5.84 | 37.64 ± 5.50 |
| N = 50 | Deep Sets+safety layer | 98.99 ± 5.85 | 24.96 ± 3.86 |
| N = 50 | SA-Attn trained at N = 50 (reference) | 124.28 ± 2.69 | 47.85 ± 2.49 |
| N = 50 | LLF+safety layer | 185.35† | 96.10† |
| N = 50 | sLLF+safety layer | 185.35† | 96.10† |
| N = 50 | Price-threshold+safety layer | 131.81† | 60.76† |
| N = 50 | AFAP+safety layer | 185.35† | 96.10† |
| N = 50 | ALAP+safety layer | 162.33† | 79.31† |
| N = 50 | Round Robin+safety layer | 185.35† | 96.10† |
| N = 50 | eMPC-HiGHS+safety layer | 112.47† | 36.31† |
| N = 50 | eMPC-HiGHS | -119.42† | -116.35† |
| N = 100 | SA-Attn (ours) | 152.71 ± 8.41 | 55.38 ± 5.48 |
| N = 100 | Attention+penalty | 168.05 ± 9.46 | 63.69 ± 9.20 |
| N = 100 | Attention+PPO-Lagrangian | 205.82 ± 6.42 | 92.77 ± 4.75 |
| N = 100 | Attention+safety layer without departure bound | 133.84 ± 11.20 | 37.61 ± 8.39 |
| N = 100 | Deep Sets+safety layer | 113.46 ± 11.24 | 25.46 ± 6.37 |
| N = 100 | SA-Attn trained at N = 50 (reference) | 148.08 ± 2.84 | 53.28 ± 0.14 |
| N = 100 | LLF+safety layer | 211.67† | 96.72† |
| N = 100 | sLLF+safety layer | 211.67† | 96.72† |
| N = 100 | Price-threshold+safety layer | 147.25† | 48.41† |
| N = 100 | AFAP+safety layer | 211.67† | 96.72† |
| N = 100 | ALAP+safety layer | 188.20† | 81.18† |
| N = 100 | Round Robin+safety layer | 211.67† | 96.72† |
| N = 100 | eMPC-HiGHS+safety layer | 132.89† | 42.14† |
| N = 100 | eMPC-HiGHS | -142.61† | -124.74† |
| N = 200 | SA-Attn (ours) | 195.83 ± 10.82 | 85.34 ± 6.27 |
| N = 200 | Attention+penalty | 225.32 ± 9.76 | 110.22 ± 9.27 |
| N = 200 | Attention+PPO-Lagrangian | 257.98 ± 7.88 | 141.31 ± 6.30 |
| N = 200 | Attention+safety layer without departure bound | 163.33 ± 27.50 | 70.31 ± 7.93 |
| N = 200 | Deep Sets+safety layer | 140.02 ± 8.92 | 52.92 ± 1.80 |
| N = 200 | SA-Attn trained at N = 50 (reference) | 190.77 ± 1.66 | 81.38 ± 2.35 |
| N = 200 | LLF+safety layer | 264.35† | 146.32† |
| N = 200 | sLLF+safety layer | 264.35† | 146.32† |
| N = 200 | Price-threshold+safety layer | 184.58† | 74.59† |
| N = 200 | AFAP+safety layer | 264.35† | 146.32† |
| N = 200 | ALAP+safety layer | 240.77† | 119.63† |
| N = 200 | Round Robin+safety layer | 264.35† | 146.32† |
| N = 200 | eMPC-HiGHS+safety layer | 167.82† | 59.11† |
| N = 200 | eMPC-HiGHS | -132.38† | -178.26† |

*Values: mean ± standard deviation over training seeds of the seed-level means (configurations pooled as strata; n per cell in the _values.csv files). † deterministic method, single run (no seed variability). 95 % confidence intervals (t-based and bootstrap) are in the _values.csv files; definitions in STAT_METHODS.md.*
