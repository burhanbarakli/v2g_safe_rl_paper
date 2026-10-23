**TS3.** Literature configuration (literature_N25: EV2Gym V2GProfitMax, 25 chargers, 100 kW, workplace profile), for comparison with published EV2Gym results.

| Method | J^pen (€/episode) | Optimality gap (%) | Profit (€/episode) | Energy not served (kWh) | Overload (kWh) | Inference (ms/step) |
|---|---:|---:|---:|---:|---:|---:|
| SA-Attn (ours) | -49.01 ± 1.17 | 54.6 ± 1.4 | -47.12 ± 0.26 | 0.19 ± 0.14 | 0 ± 0 | 2.11 ± 0.01 |
| Attention+penalty | -53.99 ± 6.77 | 66.1 ± 11.9 | -47.83 ± 1.30 | 0.01 ± 0.02 | 0.60 ± 0.82 | 1.85 ± 0.10 |
| Attention+PPO-Lagrangian | -56.08 ± 9.09 | 78.2 ± 22.3 | -50.32 ± 1.00 | 1.8e-03 ± 2.6e-03 | 0.57 ± 0.81 | 1.82 ± 0.02 |
| Attention+safety layer without departure bound | -45.42 ± 0.18 | 38.4 ± 6.0 | -45.42 ± 0.18 | 7.7e-04 ± 1.4e-04 | 0 ± 0 | 2.13 ± 0.03 |
| Deep Sets+safety layer | -49.23 ± 1.60 | 49.2 ± 10.7 | -46.29 ± 1.61 | 0.29 ± 6.4e-04 | 0 ± 0 | 1.59 ± 0.03 |
| LLF+safety layer | -51.01† | 68.8† | -51.01† | 0† | 0† | 0.60† |
| LLF | -51.01† | 68.8† | -51.01† | 0† | 0† | 0.46† |
| sLLF+safety layer | -51.01† | 68.8† | -51.01† | 0† | 0† | 0.59† |
| sLLF | -51.01† | 68.8† | -51.01† | 0† | 0† | 0.44† |
| Price-threshold+safety layer | -49.02† | 63.1† | -49.02† | 0† | 0† | 0.64† |
| Price-threshold | -4936.17† | 20670.1† | 58.80† | 418.76† | 80.74† | 0.47† |
| AFAP+safety layer | -51.00† | 68.8† | -51.00† | 0† | 0† | 0.59† |
| AFAP | -62.68† | 95.0† | -51.03† | 0† | 1.16† | 0.45† |
| ALAP+safety layer | -50.12† | 64.1† | -50.12† | 0† | 0† | 0.59† |
| ALAP | -51.07† | 65.8† | -50.12† | 0† | 0.09† | 0.43† |
| Round Robin+safety layer | -51.00† | 68.8† | -51.00† | 0† | 0† | 0.60† |
| Round Robin | -51.00† | 68.8† | -51.00† | 0† | 0† | 0.44† |
| eMPC-HiGHS+safety layer | -47.73† | 48.9† | -46.61† | 0.11† | 0† | 2.08† |
| eMPC-HiGHS | -2642.91† | 10174.1† | -12.25† | 263.07† | 4.0e-04† | 2.00† |

*Values: mean ± standard deviation over training seeds of the seed-level means (configurations pooled as strata; n per cell in the _values.csv files). † deterministic method, single run (no seed variability). 95 % confidence intervals (t-based and bootstrap) are in the _values.csv files; definitions in STAT_METHODS.md.*
